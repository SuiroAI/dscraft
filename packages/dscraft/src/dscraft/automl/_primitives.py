"""Shared first-party numeric/optimization primitives for ``dscraft.automl``.

This internal module is the shared mathematical substrate every first-party
``dscraft.automl`` estimator is built on (the linear models in ``models.py``, the
gradient-boosting and forest ensembles, etc.). It exists so those estimators never reach
for a third-party *algorithm* library for the pieces that are genuinely "the technique" --
loss functions, their gradients/Hessians, link/activation functions, the gradient-descent /
Newton / IRLS optimization loop, standard evaluation metrics, and train/validation splitting.
Only ``numpy`` (array substrate) is used here -- no ``scipy`` solver, no ``scikit-learn``
estimator, no boosting library.

It is deliberately an *internal* module (leading-underscore name, not re-exported from
``dscraft.automl.__init__``): it is implementation detail shared across the subpackage's own
estimators, not a public API surface in its own right. The public, caller-facing surface stays
the ``build_model``/``build_clusterer``/``build_resampler`` factories and ``compile``.

Design notes that apply throughout:

- **Numerical stability is handled explicitly, not left to chance.** The logistic/softmax code
  paths clip inputs and use the log-sum-exp trick so that a large-magnitude score never produces
  ``inf``/``nan`` (see :func:`sigmoid`, :func:`softmax`, :func:`log_loss`). This matters because
  these primitives feed iterative solvers where a single ``nan`` silently poisons every subsequent
  iteration.
- **Every function validates its inputs and raises a clear ``ValueError``/``TypeError``** rather
  than letting a cryptic downstream NumPy broadcast error surface three calls deeper. This mirrors
  the error-loudly stance the rest of this package takes (see e.g. ``dscraft.eda.sketches``).
- **No global random state.** Any function that draws randomness takes an explicit
  ``numpy.random.Generator`` (or an integer seed coerced into one via :func:`as_generator`), so a
  caller passing ``random_state`` gets reproducible behavior without this module ever touching the
  process-global ``numpy.random`` singleton.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    # Generators / validation helpers
    "as_generator",
    "check_X_y",
    "check_array_2d",
    "train_validation_split",
    # Link / activation functions
    "sigmoid",
    "softmax",
    # Loss functions (value + gradient/hessian)
    "squared_error_loss",
    "squared_error_gradient",
    "absolute_error_gradient",
    "huber_loss",
    "huber_gradient",
    "log_loss",
    "logistic_gradient",
    "logistic_hessian_diag",
    "softmax_cross_entropy",
    "softmax_cross_entropy_gradient",
    # Metrics
    "accuracy_score",
    "log_loss_score",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    # Solver
    "gradient_descent",
    "newton_irls",
]


# ---------------------------------------------------------------------------
# Random state / input validation
# ---------------------------------------------------------------------------


def as_generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Coerce ``random_state`` into a ``numpy.random.Generator``.

    Accepts an existing ``numpy.random.Generator`` (returned as-is), an integer seed (used to
    build a fresh default-BitGenerator ``Generator``), or ``None`` (an unseeded ``Generator``).
    This is the single place this subpackage converts a caller-facing ``random_state`` into an
    actual RNG, so no first-party estimator ever touches the process-global ``numpy.random``
    singleton.

    Raises:
        TypeError: if ``random_state`` is not ``None``, an ``int``, or a
            ``numpy.random.Generator``.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    raise TypeError(
        "random_state must be None, an int seed, or a numpy.random.Generator, "
        f"got {type(random_state).__name__}."
    )


def check_array_2d(X, *, name: str = "X") -> np.ndarray:
    """Validate ``X`` as a finite 2-D float64 array.

    Args:
        X: array-like to validate.
        name: name used in error messages.

    Returns:
        ``X`` as a contiguous ``float64`` ``numpy.ndarray`` of shape ``(n_samples, n_features)``.

    Raises:
        ValueError: if ``X`` is not 2-D, is empty, or contains non-finite (NaN/inf) values.
        TypeError: if ``X`` cannot be coerced to a float64 array.
    """
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be coercible to a float64 array; got {type(X).__name__}.") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape!r}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape!r}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")
    return arr


def check_X_y(X, y, *, name_y: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix / target pair for a supervised fit.

    Returns ``(X, y)`` with ``X`` a finite 2-D float64 array (see :func:`check_array_2d`) and
    ``y`` a 1-D array with one entry per row of ``X``. ``y``'s dtype is left untouched (so a
    classifier can keep string/int class labels and a regressor can coerce to float itself).

    Raises:
        ValueError: if ``X``/``y`` shapes are inconsistent, or ``y`` is not 1-D.
        TypeError: via :func:`check_array_2d` if ``X`` is not numeric.
    """
    X_arr = check_array_2d(X, name="X")
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"{name_y} must be 1-D (one label per sample), got shape {y_arr.shape!r}.")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(
            f"X and {name_y} must have the same number of samples, got "
            f"{X_arr.shape[0]} and {y_arr.shape[0]}."
        )
    return X_arr, y_arr


def train_validation_split(
    n_samples: int,
    validation_fraction: float,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``range(n_samples)`` into shuffled train/validation index arrays.

    Used by estimators that support early stopping against a held-out slice. Returns a
    ``(train_indices, validation_indices)`` pair of int arrays; the validation set is the last
    ``round(validation_fraction * n_samples)`` entries of a single shuffled permutation, so the two
    arrays are disjoint and together cover every index exactly once.

    Args:
        n_samples: total number of samples (must be >= 2 so both splits can be non-empty).
        validation_fraction: fraction in ``(0.0, 1.0)`` held out for validation.
        random_state: seed / ``Generator`` for the shuffle (see :func:`as_generator`).

    Raises:
        ValueError: if ``n_samples < 2`` or ``validation_fraction`` is not in ``(0, 1)``, or the
            resulting validation split would be empty or leave an empty training split.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 to form a train/validation split, got {n_samples}.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction!r}.")
    rng = as_generator(random_state)
    permutation = rng.permutation(n_samples)
    n_validation = round(validation_fraction * n_samples)
    if n_validation < 1 or n_validation >= n_samples:
        raise ValueError(
            f"validation_fraction={validation_fraction!r} with n_samples={n_samples} yields a "
            f"degenerate split (validation size {n_validation}); choose a fraction that leaves "
            "both splits non-empty."
        )
    validation_indices = permutation[:n_validation]
    train_indices = permutation[n_validation:]
    return train_indices, validation_indices


# ---------------------------------------------------------------------------
# Link / activation functions
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid ``1 / (1 + exp(-z))``, elementwise.

    Implemented piecewise on the sign of ``z`` so that neither ``exp(-z)`` (for very negative
    ``z``) nor ``exp(z)`` (for very positive ``z``) ever overflows -- a plain
    ``1 / (1 + np.exp(-z))`` would emit a RuntimeWarning and return ``0``/``nan`` at the extremes.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a ``(n_samples, n_classes)`` score matrix.

    Subtracts each row's max before exponentiating (the standard log-sum-exp shift) so large
    scores can never overflow; the result of each row sums to ``1.0``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"softmax expects a 2-D (n_samples, n_classes) array, got shape {scores.shape!r}.")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Loss functions (regression)
# ---------------------------------------------------------------------------


def squared_error_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error ``mean((y_pred - y_true)**2)``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def squared_error_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample gradient of squared error w.r.t. the prediction: ``(y_pred - y_true)``.

    (The constant factor of 2 is folded into the caller's learning rate, as is conventional for
    boosting/GD, so this returns the residual directly.)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return y_pred - y_true


def absolute_error_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample (sub)gradient of absolute error: ``sign(y_pred - y_true)``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.sign(y_pred - y_true)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float) -> float:
    """Mean pinball (quantile) loss at quantile level ``alpha`` in (0, 1).

    ``L_alpha(y, f) = mean( max(alpha*(y-f), (alpha-1)*(y-f)) )``. Minimized (in the
    constant case) by the empirical ``alpha``-quantile of ``y``. This is the loss the
    first-party quantile booster targets; absolute error is the ``alpha=0.5`` special
    case up to a factor of 2.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def quantile_gradient(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float) -> np.ndarray:
    """Per-sample gradient of pinball loss w.r.t. the prediction, in ``(pred - y)`` sign.

    d/df of ``max(alpha*(y-f), (alpha-1)*(y-f))`` is ``-alpha`` when ``y > f`` and
    ``(1 - alpha)`` when ``y <= f``. The boosting loop fits trees to the *negative*
    gradient via a residual-signed convention (see ``squared_error_gradient`` returning
    ``pred - y``), so this returns the gradient already in that convention:
    ``-alpha`` where the prediction is below the target (push up), ``1-alpha`` where it
    is at/above (push down). Magnitude is constant per side, the standard GBM quantile
    construction; the per-leaf step size comes from the tree/leaf fit and learning rate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.where(y_true > y_pred, -alpha, 1.0 - alpha)


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, *, delta: float = 1.0) -> float:
    """Mean Huber loss with transition point ``delta`` (quadratic within, linear beyond)."""
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta!r}.")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    abs_residual = np.abs(residual)
    quadratic = np.minimum(abs_residual, delta)
    linear = abs_residual - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))


def huber_gradient(y_true: np.ndarray, y_pred: np.ndarray, *, delta: float = 1.0) -> np.ndarray:
    """Per-sample gradient of Huber loss w.r.t. the prediction."""
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta!r}.")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    return np.where(np.abs(residual) <= delta, residual, delta * np.sign(residual))


# ---------------------------------------------------------------------------
# Loss functions (classification)
# ---------------------------------------------------------------------------

#: Clip bound keeping probabilities strictly inside (0, 1) so log/division never hits 0 or 1.
_EPS = 1e-15


def log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean binary cross-entropy for probabilities ``p_pred`` of the positive class.

    ``y_true`` must be 0/1 labels. Probabilities are clipped into ``[_EPS, 1 - _EPS]`` before the
    log, so a confidently-wrong prediction yields a large-but-finite loss rather than ``inf``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p_pred = np.clip(np.asarray(p_pred, dtype=np.float64), _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred)))


def logistic_gradient(y_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    """Per-sample gradient of binary cross-entropy w.r.t. the logit: ``(p_pred - y_true)``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    return p_pred - y_true


def logistic_hessian_diag(p_pred: np.ndarray) -> np.ndarray:
    """Per-sample diagonal Hessian of binary cross-entropy w.r.t. the logit: ``p * (1 - p)``.

    Floored at a tiny positive value so an IRLS/Newton step never divides by zero for a
    saturated probability.
    """
    p_pred = np.asarray(p_pred, dtype=np.float64)
    return np.maximum(p_pred * (1.0 - p_pred), _EPS)


def softmax_cross_entropy(y_true_onehot: np.ndarray, proba: np.ndarray) -> float:
    """Mean multiclass cross-entropy for a one-hot target and a softmax probability matrix."""
    y_true_onehot = np.asarray(y_true_onehot, dtype=np.float64)
    proba = np.clip(np.asarray(proba, dtype=np.float64), _EPS, 1.0)
    return float(-np.mean(np.sum(y_true_onehot * np.log(proba), axis=1)))


def softmax_cross_entropy_gradient(y_true_onehot: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Per-sample, per-class gradient of softmax cross-entropy w.r.t. the logits: ``proba - y``."""
    y_true_onehot = np.asarray(y_true_onehot, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    return proba - y_true_onehot


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of exactly-matching labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape!r} and {y_pred.shape!r}.")
    if y_true.size == 0:
        raise ValueError("accuracy_score requires at least one sample.")
    return float(np.mean(y_true == y_pred))


def log_loss_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Public alias of :func:`log_loss` for symmetry with the other ``*_score`` metrics."""
    return log_loss(y_true, p_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error (alias of :func:`squared_error_loss`)."""
    return squared_error_loss(y_true, y_pred)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(squared_error_loss(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Returns ``0.0`` (not ``nan``) in the degenerate case where ``y_true`` has zero variance and
    the predictions are not perfect, matching the common library convention that an undefined R^2
    is reported as the no-skill baseline rather than propagating a NaN.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def gradient_descent(
    gradient_fn,
    initial_params: np.ndarray,
    *,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Batch gradient descent on an arbitrary differentiable objective.

    Iterates ``params -= learning_rate * gradient_fn(params)`` until either the update's L2 norm
    drops below ``tol`` (converged) or ``max_iter`` iterations elapse. The objective itself is
    never evaluated here -- only its gradient -- so the caller is free to use whatever loss its
    ``gradient_fn`` is the gradient of.

    Args:
        gradient_fn: callable mapping a parameter vector to its gradient (same shape).
        initial_params: starting parameter vector.
        learning_rate: step size (must be positive).
        max_iter: maximum number of iterations (must be positive).
        tol: convergence tolerance on the L2 norm of the parameter update.

    Returns:
        ``(params, n_iter)`` -- the final parameters and how many iterations ran.

    Raises:
        ValueError: if ``learning_rate <= 0``, ``max_iter < 1``, or ``tol < 0``.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate!r}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}.")
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol!r}.")
    params = np.array(initial_params, dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        gradient = np.asarray(gradient_fn(params), dtype=np.float64)
        update = learning_rate * gradient
        params = params - update
        if np.linalg.norm(update) < tol:
            return params, iteration
    return params, max_iter


def newton_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Fit binary logistic regression weights by Newton / iteratively-reweighted least squares.

    Solves for the weight vector ``w`` (including an intercept as ``X``'s first column if the
    caller prepended one) minimizing L2-regularized binary cross-entropy, using the exact Newton
    step ``w <- w - H^{-1} g`` where ``g`` is the gradient and ``H`` the Hessian of the penalized
    log-loss. IRLS is the standard, quadratically-convergent solver for this objective; it reaches
    a tighter optimum in far fewer iterations than plain gradient descent, which is why logistic
    regression uses this path while the GD-based linear regressor uses :func:`gradient_descent`.

    The regularization term ``l2`` penalizes every weight *except* the intercept (assumed to be
    column 0), matching the standard convention that the bias term is not shrunk.

    Args:
        X: ``(n_samples, n_features)`` design matrix (caller prepends an all-ones intercept column
            if an intercept is wanted).
        y: ``(n_samples,)`` array of 0/1 labels.
        l2: L2 penalty strength (``>= 0``); ``0`` means unpenalized.
        max_iter: maximum Newton iterations.
        tol: convergence tolerance on the L2 norm of the weight update.

    Returns:
        ``(weights, n_iter)``.

    Raises:
        ValueError: if ``l2 < 0``, ``max_iter < 1``, or shapes are inconsistent.
    """
    if l2 < 0:
        raise ValueError(f"l2 must be non-negative, got {l2!r}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}.")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape!r}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X and y must share a sample axis, got {X.shape[0]} and {y.shape[0]}.")

    n_features = X.shape[1]
    weights = np.zeros(n_features, dtype=np.float64)
    # Penalty matrix: l2 on every weight except the intercept (column 0).
    penalty = l2 * np.eye(n_features)
    if n_features > 0:
        penalty[0, 0] = 0.0

    for iteration in range(1, max_iter + 1):
        scores = X @ weights
        proba = sigmoid(scores)
        gradient = X.T @ (proba - y) + penalty @ weights
        hessian_weights = logistic_hessian_diag(proba)
        hessian = (X * hessian_weights[:, None]).T @ X + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            # Singular Hessian (e.g. perfectly separable data): fall back to a
            # least-squares solve so the fit degrades gracefully instead of raising.
            step, *_ = np.linalg.lstsq(hessian, gradient, rcond=None)
        weights = weights - step
        if np.linalg.norm(step) < tol:
            return weights, iteration
    return weights, max_iter
