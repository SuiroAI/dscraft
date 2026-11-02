"""First-party gradient-boosted decision trees for ``dscraft.automl``.

This internal module implements gradient boosting from scratch -- the first-party replacement for
the XGBoost / LightGBM / CatBoost backends ``models.py`` currently wraps. It is built entirely on
``numpy``, the shared primitives in :mod:`dscraft.automl._primitives`, and the first-party CART in
:mod:`dscraft.automl._tree` (whose ``"gradient"`` split criterion was written precisely so this
module could reuse it rather than growing a parallel tree implementation).

The algorithm is second-order (Newton) gradient boosting, the same family XGBoost popularized, but
written here directly from the public objective -- never adapted from any library's source:

1. Start from a constant initial prediction ``F_0`` (the loss-minimizing constant: the mean for
   squared error, the log-odds for logistic, per-class log-probabilities for softmax).
2. At each round ``m``, compute the per-sample gradient ``g`` and Hessian ``h`` of the loss with
   respect to the current raw prediction ``F_{m-1}``.
3. Fit one regression tree (the ``"gradient"``-criterion :class:`~dscraft.automl._tree.DecisionTree`)
   to ``(g, h)``. That tree scores candidate splits by the regularized structure-score gain
   ``G^2 / (H + lambda)`` and sets each leaf to the Newton step ``-sum(g) / (sum(h) + lambda)``.
4. Update ``F_m = F_{m-1} + learning_rate * tree.predict(X)``.
5. Repeat for ``n_estimators`` rounds (or until early stopping on a held-out validation slice).

Multiclass classification trains ``n_classes`` parallel additive ensembles (one raw score per
class), tied together each round by a shared softmax that couples the per-class gradients -- the
standard multiclass-boosting construction.

Both estimators are **scikit-learn API-compatible** (``fit``/``predict``/``predict_proba``/
``get_params``/``set_params``, standard fitted attributes), for the same reason the linear models
are: so they drop into an ``sklearn.pipeline.Pipeline`` and ``dscraft.automl.compile()`` unchanged.
Only sklearn's estimator-protocol plumbing is used; all boosting math is this module's own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from dscraft.automl import _primitives as _p
from dscraft.automl._tree import DecisionTree

__all__ = [
    "DSBoostRegressor",
    "DSBoostClassifier",
    "DSQuantileBoostRegressor",
]


class _BaseGradientBooster(BaseEstimator):
    """Shared machinery for the regression and classification boosters.

    Holds the common hyperparameters and the shared per-round tree-fitting / early-stopping loop;
    subclasses supply the loss-specific pieces (initial prediction, gradient/Hessian, validation
    metric, and how a raw score becomes a prediction).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        subsample: float = 1.0,
        max_features=None,
        l2_regularization: float = 1.0,
        validation_fraction: Optional[float] = None,
        n_iter_no_change: int = 5,
        tol: float = 1e-4,
        random_state=None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        self.max_features = max_features
        self.l2_regularization = l2_regularization
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    def _validate_common(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample}.")
        if self.validation_fraction is not None and not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(f"validation_fraction must be in (0, 1) or None, got {self.validation_fraction}.")

    def _new_tree(self) -> DecisionTree:
        return DecisionTree(
            criterion="gradient",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            l2_regularization=self.l2_regularization,
        )


class DSBoostRegressor(RegressorMixin, _BaseGradientBooster):
    """First-party gradient-boosted-tree regressor.

    Minimizes squared-error loss by additively fitting ``"gradient"``-criterion regression trees
    to the pseudo-residuals, with shrinkage (``learning_rate``), optional row subsampling
    (``subsample``, stochastic gradient boosting), and optional early stopping on a held-out
    validation slice (set ``validation_fraction``).

    Args mirror the boosting hyperparameters in :class:`_BaseGradientBooster`.

    Attributes (after :meth:`fit`):
        init_: the constant initial prediction (the training mean).
        estimators_: list of fitted per-round trees.
        n_features_in_: number of input features.
        n_estimators_: number of rounds actually run (``<= n_estimators`` if early-stopped).
    """

    def fit(self, X, y) -> "DSBoostRegressor":
        """Fit the booster. Returns ``self``."""
        self._validate_common()
        X, y = _p.check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        rng = _p.as_generator(self.random_state)

        use_val = self.validation_fraction is not None
        if use_val:
            train_idx, val_idx = _p.train_validation_split(X.shape[0], self.validation_fraction, rng)
            X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        else:
            X_train, y_train, X_val, y_val = X, y, None, None

        self.init_ = float(np.mean(y_train))
        raw = np.full(X_train.shape[0], self.init_, dtype=np.float64)
        self.estimators_ = []
        best_val = np.inf
        rounds_no_change = 0

        for _ in range(self.n_estimators):
            gradients = _p.squared_error_gradient(y_train, raw)  # (pred - y)
            hessians = np.ones_like(gradients)  # second derivative of 0.5*(y-pred)^2 is 1
            sample_idx = self._subsample_indices(X_train.shape[0], rng)
            tree = self._new_tree()
            tree.fit(X_train[sample_idx], y_train[sample_idx],
                     gradients=gradients[sample_idx], hessians=hessians[sample_idx])
            raw += self.learning_rate * tree.predict(X_train)
            self.estimators_.append(tree)

            if use_val:
                val_pred = self._raw_predict(X_val)
                val_loss = _p.mean_squared_error(y_val, val_pred)
                if val_loss < best_val - self.tol:
                    best_val, rounds_no_change = val_loss, 0
                else:
                    rounds_no_change += 1
                    if rounds_no_change >= self.n_iter_no_change:
                        break

        self.n_estimators_ = len(self.estimators_)
        return self

    def _subsample_indices(self, n: int, rng) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        size = max(1, int(round(self.subsample * n)))
        return rng.choice(n, size=size, replace=False)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict(self, X) -> np.ndarray:
        """Predict target values for ``X``."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return self._raw_predict(X)


class DSQuantileBoostRegressor(RegressorMixin, _BaseGradientBooster):
    """First-party gradient-boosted-tree *quantile* regressor (pinball loss).

    Minimizes the pinball loss at a target quantile level ``alpha`` in (0, 1) by additively
    fitting ``"gradient"``-criterion trees to the pinball (sub)gradient. This is the first-party
    replacement for the previous third-party ``objective="quantile"`` backend: no boosting library
    is used, the fit is seeded (``random_state``) and reproducible, and the model is a plain tree
    ensemble whose prediction is a sum of leaf values -- it serializes to ordinary JSON with no
    pickled estimator or ONNX blob.

    The initial prediction is the empirical ``alpha``-quantile of the training targets (the
    loss-minimizing constant), and each round's gradient is the constant-magnitude pinball
    (sub)gradient (``-alpha`` where the current prediction is below the target, ``1-alpha`` where
    at/above). Row subsampling (``subsample``) makes it a genuinely stochastic, seeded learner.

    Args: ``alpha`` (the target quantile tau in (0, 1)); all others mirror
    :class:`_BaseGradientBooster`.

    Attributes (after :meth:`fit`): ``alpha_``, ``init_``, ``estimators_``, ``n_features_in_``,
    ``n_estimators_``.
    """

    def __init__(self, *, alpha: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y) -> "DSQuantileBoostRegressor":
        """Fit the quantile booster. Returns ``self``."""
        self._validate_common()
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}.")
        X, y = _p.check_X_y(X, y)
        y = y.astype(np.float64)
        self.alpha_ = float(self.alpha)
        self.n_features_in_ = X.shape[1]
        rng = _p.as_generator(self.random_state)

        use_val = self.validation_fraction is not None
        if use_val:
            train_idx, val_idx = _p.train_validation_split(X.shape[0], self.validation_fraction, rng)
            X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        else:
            X_train, y_train, X_val, y_val = X, y, None, None

        # Loss-minimizing constant for pinball loss is the empirical alpha-quantile.
        self.init_ = float(np.quantile(y_train, self.alpha_, method="linear"))
        raw = np.full(X_train.shape[0], self.init_, dtype=np.float64)
        self.estimators_ = []
        best_val = np.inf
        rounds_no_change = 0

        for _ in range(self.n_estimators):
            gradients = _p.quantile_gradient(y_train, raw, alpha=self.alpha_)
            hessians = np.ones_like(gradients)  # constant-curvature surrogate (standard GBM quantile)
            sample_idx = self._subsample_indices(X_train.shape[0], rng)
            tree = self._new_tree()
            tree.fit(X_train[sample_idx], y_train[sample_idx],
                     gradients=gradients[sample_idx], hessians=hessians[sample_idx])
            raw += self.learning_rate * tree.predict(X_train)
            self.estimators_.append(tree)

            if use_val:
                val_pred = self._raw_predict(X_val)
                val_loss = _p.pinball_loss(y_val, val_pred, alpha=self.alpha_)
                if val_loss < best_val - self.tol:
                    best_val, rounds_no_change = val_loss, 0
                else:
                    rounds_no_change += 1
                    if rounds_no_change >= self.n_iter_no_change:
                        break

        self.n_estimators_ = len(self.estimators_)
        return self

    def _subsample_indices(self, n: int, rng) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        size = max(1, int(round(self.subsample * n)))
        return rng.choice(n, size=size, replace=False)

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    def predict(self, X) -> np.ndarray:
        """Predict the ``alpha``-quantile of the target for ``X``."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return self._raw_predict(X)


class DSBoostClassifier(ClassifierMixin, _BaseGradientBooster):
    """First-party gradient-boosted-tree classifier (binary logistic + multiclass softmax).

    Binary problems boost a single additive ensemble against the logistic-loss gradient/Hessian,
    initialized at the training log-odds. Multiclass problems boost ``n_classes`` parallel
    ensembles (one raw score per class), coupled each round by a shared softmax so the per-class
    gradients ``p_k - y_k`` and Hessians ``p_k (1 - p_k)`` reflect the full class competition --
    the standard multiclass gradient-boosting construction.

    Args mirror :class:`_BaseGradientBooster`.

    Attributes (after :meth:`fit`):
        classes_: sorted unique class labels.
        init_: initial raw score(s) (log-odds for binary, per-class log-probabilities for
            multiclass).
        estimators_: for binary, a list of per-round trees; for multiclass, a list (per round) of
            per-class tree lists.
        n_features_in_, n_estimators_: as in :class:`DSBoostRegressor`.
    """

    def fit(self, X, y) -> "DSBoostClassifier":
        """Fit the booster. Returns ``self``."""
        self._validate_common()
        X, y = _p.check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("Classifier needs at least 2 classes in y.")
        self.n_features_in_ = X.shape[1]
        self._multiclass = self.classes_.shape[0] > 2
        rng = _p.as_generator(self.random_state)

        use_val = self.validation_fraction is not None
        if use_val:
            train_idx, val_idx = _p.train_validation_split(X.shape[0], self.validation_fraction, rng)
            X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
        else:
            X_train, y_train, X_val, y_val = X, y, None, None

        if self._multiclass:
            self._fit_multiclass(X_train, y_train, X_val, y_val, use_val, rng)
        else:
            self._fit_binary(X_train, y_train, X_val, y_val, use_val, rng)
        self.n_estimators_ = len(self.estimators_)
        return self

    # -- binary -------------------------------------------------------------

    def _fit_binary(self, X, y, X_val, y_val, use_val, rng) -> None:
        y01 = (y == self.classes_[1]).astype(np.float64)
        p = float(np.clip(np.mean(y01), 1e-6, 1 - 1e-6))
        self.init_ = float(np.log(p / (1.0 - p)))  # log-odds
        raw = np.full(X.shape[0], self.init_, dtype=np.float64)
        self.estimators_ = []
        best_val, rounds_no_change = np.inf, 0

        for _ in range(self.n_estimators):
            proba = _p.sigmoid(raw)
            gradients = _p.logistic_gradient(y01, proba)  # p - y
            hessians = _p.logistic_hessian_diag(proba)  # p(1-p)
            sample_idx = self._subsample_indices(X.shape[0], rng)
            tree = self._new_tree()
            tree.fit(X[sample_idx], y01[sample_idx],
                     gradients=gradients[sample_idx], hessians=hessians[sample_idx])
            raw += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            if use_val:
                val_loss = _p.log_loss(
                    (y_val == self.classes_[1]).astype(np.float64),
                    _p.sigmoid(self._binary_raw(X_val)),
                )
                if val_loss < best_val - self.tol:
                    best_val, rounds_no_change = val_loss, 0
                else:
                    rounds_no_change += 1
                    if rounds_no_change >= self.n_iter_no_change:
                        break

    def _binary_raw(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        return raw

    # -- multiclass ---------------------------------------------------------

    def _fit_multiclass(self, X, y, X_val, y_val, use_val, rng) -> None:
        n_classes = self.classes_.shape[0]
        class_index = {cls: i for i, cls in enumerate(self.classes_)}
        y_idx = np.array([class_index[v] for v in y])
        onehot = np.eye(n_classes)[y_idx]

        class_fractions = np.clip(onehot.mean(axis=0), 1e-6, None)
        self.init_ = np.log(class_fractions)  # per-class log-priors
        raw = np.tile(self.init_, (X.shape[0], 1))  # (n_samples, n_classes)
        self.estimators_ = []  # list (per round) of list-of-trees (per class)
        best_val, rounds_no_change = np.inf, 0

        for _ in range(self.n_estimators):
            proba = _p.softmax(raw)
            grad_all = _p.softmax_cross_entropy_gradient(onehot, proba)  # p - y, per class
            sample_idx = self._subsample_indices(X.shape[0], rng)
            round_trees = []
            for k in range(n_classes):
                gradients = grad_all[:, k]
                hessians = np.maximum(proba[:, k] * (1.0 - proba[:, k]), 1e-15)
                tree = self._new_tree()
                tree.fit(X[sample_idx], y_idx[sample_idx].astype(np.float64),
                         gradients=gradients[sample_idx], hessians=hessians[sample_idx])
                raw[:, k] += self.learning_rate * tree.predict(X)
                round_trees.append(tree)
            self.estimators_.append(round_trees)
            if use_val:
                val_onehot = np.eye(n_classes)[np.array([class_index[v] for v in y_val])]
                val_loss = _p.softmax_cross_entropy(val_onehot, _p.softmax(self._multiclass_raw(X_val)))
                if val_loss < best_val - self.tol:
                    best_val, rounds_no_change = val_loss, 0
                else:
                    rounds_no_change += 1
                    if rounds_no_change >= self.n_iter_no_change:
                        break

    def _multiclass_raw(self, X: np.ndarray) -> np.ndarray:
        raw = np.tile(self.init_, (X.shape[0], 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(X)
        return raw

    # -- shared -------------------------------------------------------------

    def _subsample_indices(self, n: int, rng) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        size = max(1, int(round(self.subsample * n)))
        return rng.choice(n, size=size, replace=False)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class-probability vectors for ``X``."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        if self._multiclass:
            return _p.softmax(self._multiclass_raw(X))
        p1 = _p.sigmoid(self._binary_raw(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X``."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
