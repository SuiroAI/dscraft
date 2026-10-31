"""First-party outlier / anomaly detection for ``dscraft.eda``.

This internal module implements outlier detection from scratch on ``numpy`` -- the deferred
LazyEDA "outlier/anomaly detection" capability (see the architecture doc's LazyEDA module entry,
and ``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``), now built first-party. It
provides both univariate (per-column) screens and a multivariate (whole-row) screen, so an analyst
can ask either "which values in this column are anomalous" or "which rows look anomalous across all
their features at once":

Univariate screens (each flags a boolean mask over a 1-D numeric array):

- :func:`iqr_outliers` -- Tukey's fences: flag values outside
  ``[Q1 - k*IQR, Q3 + k*IQR]`` (``k = 1.5`` is the classic "outlier" fence, ``3.0`` "far out").
  Distribution-free and the most widely-recognised univariate rule.
- :func:`mad_outliers` -- robust z-score: flag values whose distance from the median, scaled by
  the median absolute deviation (MAD), exceeds a threshold. More robust than a mean/std z-score
  because neither the center (median) nor the scale (MAD) is itself distorted by the outliers it is
  trying to find.

Multivariate screen:

- :class:`IsolationForest` -- an isolation-forest anomaly detector: build an ensemble of random
  "isolation trees" that recursively split on a random feature at a random threshold; a point's
  anomaly score reflects how *shallow* its average isolation depth is across the forest (anomalies
  are isolated in fewer splits). Implemented directly from Liu et al. (2008), using the standard
  path-length normalization ``c(n)``. This is the whole-row screen ``LazyEDA`` uses.

All randomness flows through an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "iqr_outliers",
    "mad_outliers",
    "IsolationForest",
    "ColumnOutlierReport",
]


@dataclass(frozen=True)
class ColumnOutlierReport:
    """Per-column univariate outlier summary (used by ``LazyEDA``).

    Attributes:
        method: which univariate rule produced the mask (``"iqr"`` or ``"mad"``).
        n_outliers: number of flagged values.
        fraction: ``n_outliers / count`` of finite values screened.
        lower_bound / upper_bound: the inclusive bounds outside which a value was flagged (for
            ``"iqr"``), or ``None`` for methods without a simple two-sided bound.
        outlier_indices: indices (into the column's finite-value array) of the flagged values.
    """

    method: str
    n_outliers: int
    fraction: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    outlier_indices: np.ndarray


def iqr_outliers(values, *, k: float = 1.5) -> ColumnOutlierReport:
    """Flag univariate outliers by Tukey's IQR fences.

    Values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` are flagged. ``k=1.5`` is the standard outlier
    fence; ``k=3.0`` flags only "far out" values.

    Raises:
        ValueError: if ``values`` is empty or ``k`` is negative.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k!r}.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("iqr_outliers requires at least one value.")
    q1, q3 = (float(v) for v in np.quantile(arr, [0.25, 0.75]))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (arr < lower) | (arr > upper)
    indices = np.nonzero(mask)[0]
    return ColumnOutlierReport(
        method="iqr",
        n_outliers=int(indices.size),
        fraction=float(indices.size / arr.size),
        lower_bound=lower,
        upper_bound=upper,
        outlier_indices=indices,
    )


def mad_outliers(values, *, threshold: float = 3.5) -> ColumnOutlierReport:
    """Flag univariate outliers by a robust (median/MAD) z-score.

    Computes the modified z-score ``0.6745 * (x - median) / MAD`` (the ``0.6745`` makes the MAD a
    consistent estimator of the standard deviation for normal data) and flags values whose absolute
    modified z-score exceeds ``threshold`` (``3.5`` is Iglewicz & Hoaglin's recommended default).
    When the MAD is zero (more than half the values are identical), falls back to flagging only
    values that differ from the median at all.

    Raises:
        ValueError: if ``values`` is empty or ``threshold`` is non-positive.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold!r}.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mad_outliers requires at least one value.")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0.0:
        mask = arr != median
    else:
        modified_z = 0.6745 * (arr - median) / mad
        mask = np.abs(modified_z) > threshold
    indices = np.nonzero(mask)[0]
    return ColumnOutlierReport(
        method="mad",
        n_outliers=int(indices.size),
        fraction=float(indices.size / arr.size),
        lower_bound=None,
        upper_bound=None,
        outlier_indices=indices,
    )


def _harmonic(n: int) -> float:
    """Harmonic number H(n-1) approximation used in the isolation-forest path normalization."""
    if n <= 1:
        return 0.0
    return float(np.log(n - 1) + 0.5772156649)  # ln + Euler-Mascheroni constant


def _c(n: int) -> float:
    """Average unsuccessful-search path length in a BST of ``n`` points (Liu et al. normalization)."""
    if n <= 1:
        return 1.0
    return 2.0 * _harmonic(n) - (2.0 * (n - 1) / n)


class _IsolationTree:
    """One random isolation tree (internal to :class:`IsolationForest`)."""

    __slots__ = ("feature", "threshold", "left", "right", "size", "depth")

    def __init__(self) -> None:
        self.feature: Optional[int] = None
        self.threshold: float = 0.0
        self.left: Optional["_IsolationTree"] = None
        self.right: Optional["_IsolationTree"] = None
        self.size: int = 0
        self.depth: int = 0

    def path_length(self, x: np.ndarray, current_depth: int = 0) -> float:
        """Isolation depth of point ``x`` in this tree (external-node depth + ``c(size)``)."""
        if self.feature is None:  # external node
            return current_depth + _c(self.size)
        if x[self.feature] < self.threshold:
            return self.left.path_length(x, current_depth + 1)
        return self.right.path_length(x, current_depth + 1)


class IsolationForest:
    """First-party isolation-forest multivariate anomaly detector (Liu et al. 2008).

    Builds ``n_estimators`` isolation trees, each on a random subsample of ``max_samples`` rows,
    splitting on a random feature at a random threshold down to ``max_depth``. A point's anomaly
    score in ``[0, 1]`` is ``2 ** (-avg_path_length / c(max_samples))``: scores near 1 are strong
    anomalies (isolated in few splits), scores well below 0.5 are inliers.

    Args:
        n_estimators: number of isolation trees.
        max_samples: subsample size per tree (capped at the dataset size).
        contamination: expected anomaly fraction, used only to derive the
            :attr:`threshold_`/:meth:`predict` cutoff (the top ``contamination`` fraction by score
            are labelled anomalies). ``"auto"`` uses the classic score cutoff of 0.5.
        random_state: seed / ``numpy.random.Generator``.

    Attributes (after :meth:`fit`):
        threshold_: the score cutoff separating inliers from anomalies.
        n_features_in_: number of input features.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination=0.1,
        random_state=None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X) -> "IsolationForest":
        """Fit the forest on ``X``. Returns ``self``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape!r}.")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        self._sample_size = min(self.max_samples, n)
        self._max_depth = max(1, int(np.ceil(np.log2(max(2, self._sample_size)))))
        rng = _as_generator(self.random_state)

        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=self._sample_size, replace=False) if self._sample_size < n else np.arange(n)
            self._trees.append(self._build(X[idx], 0, rng))

        scores = self.score_samples(X)
        if self.contamination == "auto":
            self.threshold_ = 0.5
        else:
            if not (0.0 < float(self.contamination) < 0.5):
                raise ValueError(f"contamination must be in (0, 0.5) or 'auto', got {self.contamination!r}.")
            # The top `contamination` fraction by score are anomalies.
            self.threshold_ = float(np.quantile(scores, 1.0 - float(self.contamination)))
        return self

    def _build(self, X: np.ndarray, depth: int, rng) -> _IsolationTree:
        node = _IsolationTree()
        node.size = X.shape[0]
        node.depth = depth
        if depth >= self._max_depth or X.shape[0] <= 1:
            return node  # external node
        feature = int(rng.integers(0, X.shape[1]))
        column = X[:, feature]
        lo, hi = column.min(), column.max()
        if lo == hi:
            return node  # cannot split a constant feature -> external node
        threshold = float(rng.uniform(lo, hi))
        left_mask = column < threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], depth + 1, rng)
        node.right = self._build(X[~left_mask], depth + 1, rng)
        return node

    def score_samples(self, X) -> np.ndarray:
        """Anomaly score in ``[0, 1]`` per row (higher = more anomalous)."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        norm = _c(self._sample_size)
        scores = np.empty(X.shape[0], dtype=np.float64)
        for i, row in enumerate(X):
            avg_path = np.mean([tree.path_length(row) for tree in self._trees])
            scores[i] = 2.0 ** (-avg_path / norm) if norm > 0 else 0.5
        return scores

    def predict(self, X) -> np.ndarray:
        """Label each row ``1`` (anomaly) or ``0`` (inlier) using :attr:`threshold_`."""
        return (self.score_samples(X) >= self.threshold_).astype(np.int64)

    def fit_predict(self, X) -> np.ndarray:
        """Fit and label in one call."""
        return self.fit(X).predict(X)


def _as_generator(random_state):
    """Local Generator coercion (kept here so this module has no cross-import for one helper)."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
