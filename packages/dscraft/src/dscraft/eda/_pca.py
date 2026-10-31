"""First-party PCA for exploratory dimensionality analysis in ``dscraft.eda``.

This internal module implements Principal Component Analysis from scratch on ``numpy`` -- a
first-party EDA *breadth* capability (dimensionality-reduction / multivariate structure) added by
the automl/eda rebuild. It is pure linear algebra (mean-centering + the singular value
decomposition), so it adds no dependency beyond ``numpy``, which ``eda`` already uses.

PCA answers the exploratory question "how many dimensions does this numeric data really live in,
and which directions carry the variance" -- the explained-variance-ratio curve is the single most
useful EDA readout (a sharp elbow means the data is effectively low-dimensional). ``LazyEDA`` uses
it to produce a compact :class:`PCAResult` over a dataset's numeric columns.

Implementation notes:

- The fit mean-centers the data (and optionally standardizes each feature to unit variance, which
  is usually what you want when features are on different scales) and takes the SVD of the centered
  matrix. The principal components are the right singular vectors; the explained variance per
  component is ``s**2 / (n - 1)``.
- The SVD route (rather than forming the covariance matrix and eigendecomposing it) is the
  numerically-stable standard and avoids squaring the condition number.
- :meth:`transform` projects data onto the retained components; :meth:`fit_transform` does both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PCAResult",
    "PCA",
]


@dataclass(frozen=True)
class PCAResult:
    """Summary of a fitted PCA (the compact readout ``LazyEDA`` surfaces).

    Attributes:
        n_components: number of components retained.
        explained_variance: per-component variance (length ``n_components``).
        explained_variance_ratio: per-component fraction of total variance (sums to <= 1).
        cumulative_variance_ratio: running sum of ``explained_variance_ratio``.
        n_components_for_90pct: smallest number of leading components whose cumulative explained
            variance reaches 90% (a quick "effective dimensionality" readout), or ``None`` if even
            all retained components fall short.
        feature_names: the numeric feature names the PCA was fit over, in order.
    """

    n_components: int
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    n_components_for_90pct: Optional[int]
    feature_names: list[str]


class PCA:
    """First-party Principal Component Analysis via the SVD.

    Args:
        n_components: number of components to keep; ``None`` keeps ``min(n_samples, n_features)``.
        standardize: if ``True`` (default), scale each feature to unit variance before the SVD
            (correlation-PCA) -- appropriate when features are on different scales; if ``False``,
            only mean-center (covariance-PCA).

    Attributes (after :meth:`fit`):
        components_: ``(n_components, n_features)`` principal axes (right singular vectors).
        explained_variance_ / explained_variance_ratio_: per-component variance and its fraction.
        mean_ / scale_: the centering mean and (if standardizing) per-feature scale.
        n_features_in_: number of input features.
    """

    def __init__(self, n_components: Optional[int] = None, *, standardize: bool = True) -> None:
        self.n_components = n_components
        self.standardize = standardize

    def fit(self, X) -> "PCA":
        """Fit PCA on ``X``. Returns ``self``."""
        Xc = self._preprocess_fit(X)
        n_samples, n_features = Xc.shape
        max_components = min(n_samples, n_features)
        k = max_components if self.n_components is None else min(self.n_components, max_components)
        if k < 1:
            raise ValueError(f"n_components resolves to {k}; need at least 1.")

        # Economy SVD of the centered (and optionally scaled) data.
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[:k]
        explained = (s**2) / (n_samples - 1) if n_samples > 1 else s**2
        total = explained.sum()
        self.explained_variance_ = explained[:k]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros(k)
        )
        self._k = k
        return self

    def _preprocess_fit(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape!r}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("PCA requires finite input (no NaN/inf); drop or impute first.")
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.standardize:
            scale = Xc.std(axis=0, ddof=0)
            scale[scale == 0] = 1.0  # leave constant features unscaled rather than divide by zero
            self.scale_ = scale
            Xc = Xc / scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return Xc

    def transform(self, X) -> np.ndarray:
        """Project ``X`` onto the fitted principal components."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        Xc = (X - self.mean_) / self.scale_
        return Xc @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        """Fit then project ``X`` in one call."""
        self.fit(X)
        return self.transform(X)

    def summary(self, feature_names: list[str]) -> PCAResult:
        """Build the compact :class:`PCAResult` readout over ``feature_names``."""
        ratio = self.explained_variance_ratio_
        cumulative = np.cumsum(ratio)
        reached = np.nonzero(cumulative >= 0.90)[0]
        n_for_90 = int(reached[0] + 1) if reached.size > 0 else None
        return PCAResult(
            n_components=self._k,
            explained_variance=self.explained_variance_,
            explained_variance_ratio=ratio,
            cumulative_variance_ratio=cumulative,
            n_components_for_90pct=n_for_90,
            feature_names=list(feature_names),
        )
