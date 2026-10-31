"""First-party linear models for ``dscraft.automl``.

This internal module implements the linear-model family from scratch on ``numpy`` (plus the shared
solvers in :mod:`dscraft.automl._primitives`) -- no ``scikit-learn`` estimator algorithm is used.
The estimators are nonetheless **scikit-learn API-compatible**: each subclasses the appropriate
``sklearn.base`` mixin and implements ``fit``/``predict``(/``predict_proba``)/``get_params``/
``set_params`` and sets the standard fitted attributes (``n_features_in_``, ``coef_``,
``intercept_``, and ``classes_`` for the classifier). That compatibility is deliberate and is not
a contradiction of the "no sklearn algorithms" rule: only sklearn's *estimator protocol / base
classes* (plumbing) are used, while all the fit/predict *math* -- the normal-equation / closed-form
ridge solve, the gradient-descent elastic-net path, and the IRLS logistic / one-vs-rest multinomial
fit -- is this module's own. The payoff is that these estimators drop straight into an
``sklearn.pipeline.Pipeline`` and therefore into ``dscraft.automl.compile()`` unchanged.

Models provided:

- :class:`DSLinearRegressor` -- least-squares regression with optional L2 (ridge), L1 (lasso), or
  mixed (elastic-net) regularization. Pure L2 uses the closed-form penalized normal equations
  (exact, fast); any L1 component switches to a proximal-gradient (ISTA-style) path, since L1 has
  no closed form. The intercept is never penalized.
- :class:`DSLogisticClassifier` -- logistic regression. Binary problems use the IRLS solver
  (:func:`dscraft.automl._primitives.newton_irls`); multiclass problems use a one-vs-rest ensemble
  of binary IRLS fits, with probabilities normalized across classes. Supports an L2 penalty.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from dscraft.automl import _primitives as _p

__all__ = [
    "DSLinearRegressor",
    "DSLogisticClassifier",
]


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend an all-ones intercept column to ``X``."""
    return np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])


class DSLinearRegressor(RegressorMixin, BaseEstimator):
    """First-party linear regression with optional ridge/lasso/elastic-net regularization.

    Minimizes ``mean_squared_error + alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_2^2)``
    over the (non-intercept) weights ``w``. With ``alpha == 0`` this is ordinary least squares;
    with ``l1_ratio == 0`` it is ridge (solved in closed form); any ``l1_ratio > 0`` engages a
    proximal-gradient path for the L1 term.

    Args:
        alpha: overall regularization strength (``>= 0``). ``0`` disables regularization.
        l1_ratio: elastic-net mixing in ``[0, 1]`` -- ``0`` is pure L2 (ridge), ``1`` is pure L1
            (lasso), in between is elastic-net.
        fit_intercept: whether to fit an (unpenalized) intercept term.
        max_iter: iteration cap for the proximal-gradient path (ignored for the closed-form ridge
            path).
        learning_rate: step size for the proximal-gradient path.
        tol: convergence tolerance for the proximal-gradient path.

    Attributes (after :meth:`fit`):
        coef_: ``(n_features,)`` fitted weights.
        intercept_: fitted intercept (``0.0`` if ``fit_intercept=False``).
        n_features_in_: number of input features.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y) -> "DSLinearRegressor":
        """Fit the regression weights. Returns ``self``."""
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha!r}.")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {self.l1_ratio!r}.")
        X, y = _p.check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        design = _add_intercept(X) if self.fit_intercept else X

        uses_l1 = self.alpha > 0 and self.l1_ratio > 0
        if not uses_l1:
            weights = self._fit_closed_form_ridge(design, y)
        else:
            weights = self._fit_proximal_gradient(design, y)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        return self

    def _fit_closed_form_ridge(self, design: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Closed-form penalized normal equations: (X'X + l2*P) w = X'y, P skipping the intercept."""
        n_params = design.shape[1]
        l2 = self.alpha * (1.0 - self.l1_ratio)  # == alpha when l1_ratio == 0
        penalty = l2 * np.eye(n_params)
        if self.fit_intercept and n_params > 0:
            penalty[0, 0] = 0.0  # never penalize the intercept
        gram = design.T @ design + penalty
        rhs = design.T @ y
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            # Rank-deficient (e.g. collinear features, alpha==0): minimum-norm least squares.
            weights, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            return weights

    def _fit_proximal_gradient(self, design: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ISTA-style proximal gradient descent for the elastic-net (L1-containing) objective."""
        n_samples, n_params = design.shape
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        weights = np.zeros(n_params, dtype=np.float64)
        intercept_offset = 1 if self.fit_intercept else 0
        step = self.learning_rate
        for _ in range(self.max_iter):
            residual = design @ weights - y
            grad = (design.T @ residual) / n_samples
            # L2 part of elastic net (smooth) -- skip intercept.
            grad_l2 = 2.0 * l2 * weights
            if self.fit_intercept:
                grad_l2[0] = 0.0
            weights_new = weights - step * (grad + grad_l2)
            # Soft-threshold (proximal operator for L1) -- skip intercept.
            threshold = step * l1
            shrunk = np.sign(weights_new) * np.maximum(np.abs(weights_new) - threshold, 0.0)
            if self.fit_intercept:
                shrunk[0] = weights_new[0]
            if np.linalg.norm(shrunk - weights) < self.tol:
                weights = shrunk
                break
            weights = shrunk
        return weights

    def predict(self, X) -> np.ndarray:
        """Predict target values for ``X``."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return X @ self.coef_ + self.intercept_


class DSLogisticClassifier(ClassifierMixin, BaseEstimator):
    """First-party logistic-regression classifier (binary IRLS, multiclass one-vs-rest).

    Binary problems are fit with the exact Newton/IRLS solver; multiclass problems fit one binary
    IRLS model per class (one-vs-rest) and normalize the per-class sigmoid scores into a proper
    probability distribution. An optional L2 penalty (``C`` is its inverse strength, matching the
    usual logistic-regression convention) shrinks the non-intercept weights.

    Args:
        C: inverse L2 regularization strength (``> 0``); larger ``C`` means weaker regularization.
            The L2 penalty passed to the solver is ``1 / C``.
        fit_intercept: whether to fit an (unpenalized) intercept per class.
        max_iter: maximum IRLS iterations per binary fit.
        tol: IRLS convergence tolerance.

    Attributes (after :meth:`fit`):
        classes_: sorted unique class labels.
        coef_: ``(n_classes_binary, n_features)`` weights (one row for binary, one per class for
            multiclass).
        intercept_: per-model intercept(s).
        n_features_in_: number of input features.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y) -> "DSLogisticClassifier":
        """Fit the classifier. Returns ``self``."""
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C!r}.")
        X, y = _p.check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("Classifier needs at least 2 classes in y.")
        self.n_features_in_ = X.shape[1]
        design = _add_intercept(X) if self.fit_intercept else X
        l2 = 1.0 / self.C

        if self.classes_.shape[0] == 2:
            binary_y = (y == self.classes_[1]).astype(np.float64)
            weights, _ = _p.newton_irls(design, binary_y, l2=l2, max_iter=self.max_iter, tol=self.tol)
            self._weights = weights[np.newaxis, :]  # shape (1, n_params)
        else:
            rows = []
            for cls in self.classes_:
                binary_y = (y == cls).astype(np.float64)
                weights, _ = _p.newton_irls(
                    design, binary_y, l2=l2, max_iter=self.max_iter, tol=self.tol
                )
                rows.append(weights)
            self._weights = np.vstack(rows)  # shape (n_classes, n_params)

        if self.fit_intercept:
            self.intercept_ = self._weights[:, 0].copy()
            self.coef_ = self._weights[:, 1:].copy()
        else:
            self.intercept_ = np.zeros(self._weights.shape[0])
            self.coef_ = self._weights.copy()
        return self

    def _scores(self, X: np.ndarray) -> np.ndarray:
        design = _add_intercept(X) if self.fit_intercept else X
        return design @ self._weights.T  # (n_samples, n_models)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class-probability vectors for ``X``."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        scores = self._scores(X)
        if self.classes_.shape[0] == 2:
            p1 = _p.sigmoid(scores[:, 0])
            return np.column_stack([1.0 - p1, p1])
        # One-vs-rest: sigmoid each model, then normalize across classes so rows sum to 1.
        sig = _p.sigmoid(scores)
        row_sums = sig.sum(axis=1, keepdims=True)
        # Guard against an all-zero row (every OvR model saying "not me") -> uniform.
        safe = np.where(row_sums > 0, row_sums, 1.0)
        proba = sig / safe
        uniform_mask = (row_sums.ravel() == 0)
        if np.any(uniform_mask):
            proba[uniform_mask] = 1.0 / self.classes_.shape[0]
        return proba

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X``."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
