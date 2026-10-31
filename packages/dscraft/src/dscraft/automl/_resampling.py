"""First-party imbalanced-class resamplers for ``dscraft.automl``.

This internal module implements the class-rebalancing techniques from scratch on ``numpy`` -- the
first-party replacement for the ``imbalanced-learn`` resamplers ``resampling.py`` used to wrap.
Each resampler keeps the same caller-facing contract the wrapped library used and the existing
public ``build_resampler`` surface documents: a ``fit_resample(X, y) -> (X_resampled, y_resampled)``
method returning a rebalanced copy of the training set (the originals are never mutated).

Techniques provided:

- :class:`RandomOverSampler` -- duplicate randomly-chosen minority-class rows (with replacement)
  until every class matches the majority count. The simplest rebalancer; adds no synthetic points,
  only exact copies.
- :class:`RandomUnderSampler` -- drop randomly-chosen majority-class rows (without replacement)
  until every class matches the minority count. Discards data; the counterpart to over-sampling.
- :class:`SMOTE` -- Synthetic Minority Over-sampling TEchnique: for each minority class, repeatedly
  pick a minority sample and one of its ``k`` minority-class nearest neighbours, and synthesize a
  new point at a random position on the segment between them. This is the one genuinely non-trivial
  resampling *algorithm* here -- it manufactures new, interpolated feature vectors rather than
  copying or dropping existing rows -- and is implemented directly from its published definition
  (Chawla et al. 2002), using an exact brute-force Euclidean k-NN search over each class.
- :class:`BorderlineSMOTE` -- a SMOTE variant that only synthesizes from minority samples on the
  *borderline* (those whose majority-class neighbours outnumber their minority-class neighbours,
  but not so overwhelmingly that they look like noise), concentrating synthetic points where the
  decision boundary is actually contested.

Scope: like the module it replaces, these build and return a resampled ``(X, y)`` for the caller
to fit a model on themselves -- they do not fit or wrap a classifier. All randomness flows through
an explicit ``numpy.random.Generator`` (via :func:`dscraft.automl._primitives.as_generator`), so a
``random_state`` gives reproducible resampling without touching the global RNG.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dscraft.automl import _primitives as _p

__all__ = [
    "RandomOverSampler",
    "RandomUnderSampler",
    "SMOTE",
    "BorderlineSMOTE",
]


def _class_counts(y: np.ndarray) -> dict:
    """Map each class label to its count."""
    values, counts = np.unique(y, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def _validate_Xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = _p.check_array_2d(X, name="X")
    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1-D with one label per row of X.")
    if np.unique(y).shape[0] < 2:
        raise ValueError("Resampling requires at least 2 classes in y.")
    return X, y


def _knn_indices(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """Indices (into ``pool``) of the ``k`` nearest rows to each row of ``query`` (Euclidean).

    Exact brute-force search: computes the full ``(n_query, n_pool)`` squared-distance matrix and
    takes the ``k`` smallest per row. For the modest, per-class sizes resampling operates on this
    is simple and correct; an approximate-NN index is a future optimization, not needed here.
    """
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
    q_sq = np.sum(query**2, axis=1, keepdims=True)
    p_sq = np.sum(pool**2, axis=1, keepdims=True).T
    dist_sq = q_sq - 2.0 * query @ pool.T + p_sq
    np.maximum(dist_sq, 0.0, out=dist_sq)  # clip tiny negatives from round-off
    return np.argsort(dist_sq, axis=1, kind="mergesort")[:, :k]


class _BaseResampler:
    """Shared validation + target-count resolution for the resamplers."""

    def __init__(self, *, random_state=None) -> None:
        self.random_state = random_state

    def fit_resample(self, X, y):  # pragma: no cover - overridden
        raise NotImplementedError


class RandomOverSampler(_BaseResampler):
    """Random over-sampling: duplicate minority rows until classes are balanced.

    Every class below the majority count is topped up by sampling its own rows *with replacement*
    until it reaches the majority count. No synthetic points are created -- only exact copies.
    """

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Return a class-balanced ``(X_resampled, y_resampled)``."""
        X, y = _validate_Xy(X, y)
        rng = _p.as_generator(self.random_state)
        counts = _class_counts(y)
        target = max(counts.values())
        parts_X = [X]
        parts_y = [y]
        for label, count in counts.items():
            if count < target:
                idx = np.nonzero(y == label)[0]
                extra = rng.choice(idx, size=target - count, replace=True)
                parts_X.append(X[extra])
                parts_y.append(y[extra])
        return np.vstack(parts_X), np.concatenate(parts_y)


class RandomUnderSampler(_BaseResampler):
    """Random under-sampling: drop majority rows until classes are balanced.

    Every class above the minority count is cut down by sampling its own rows *without
    replacement* down to the minority count. Discards data by design.
    """

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Return a class-balanced ``(X_resampled, y_resampled)``."""
        X, y = _validate_Xy(X, y)
        rng = _p.as_generator(self.random_state)
        counts = _class_counts(y)
        target = min(counts.values())
        keep_idx = []
        for label in counts:
            idx = np.nonzero(y == label)[0]
            chosen = rng.choice(idx, size=target, replace=False) if idx.shape[0] > target else idx
            keep_idx.append(chosen)
        keep = np.concatenate(keep_idx)
        keep.sort()  # preserve original row order among the kept rows
        return X[keep], y[keep]


class SMOTE(_BaseResampler):
    """Synthetic Minority Over-sampling TEchnique (Chawla et al. 2002).

    For each class below the majority count, synthesizes new feature vectors by interpolating
    between a minority sample and one of its ``k_neighbors`` minority-class nearest neighbours, at
    a random fraction along the connecting segment, until that class reaches the majority count.

    Args:
        k_neighbors: number of minority-class nearest neighbours to interpolate toward (must be
            ``>= 1``; a class with fewer than ``k_neighbors + 1`` members falls back to random
            over-sampling for that class, since there aren't enough neighbours to interpolate).
        random_state: seed / Generator for neighbour and gap selection.
    """

    def __init__(self, *, k_neighbors: int = 5, random_state=None) -> None:
        super().__init__(random_state=random_state)
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Return a class-balanced ``(X_resampled, y_resampled)`` with synthetic minority points."""
        if self.k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {self.k_neighbors}.")
        X, y = _validate_Xy(X, y)
        rng = _p.as_generator(self.random_state)
        counts = _class_counts(y)
        target = max(counts.values())
        parts_X = [X]
        parts_y = [y]
        for label, count in counts.items():
            n_needed = target - count
            if n_needed <= 0:
                continue
            synthetic = self._synthesize(X[y == label], n_needed, rng)
            parts_X.append(synthetic)
            parts_y.append(np.full(n_needed, label, dtype=y.dtype))
        return np.vstack(parts_X), np.concatenate(parts_y)

    def _synthesize(self, minority: np.ndarray, n_needed: int, rng) -> np.ndarray:
        n_minority = minority.shape[0]
        # Need at least 2 points to interpolate; fall back to plain duplication otherwise.
        if n_minority < 2:
            idx = rng.integers(0, n_minority, size=n_needed)
            return minority[idx]
        k = min(self.k_neighbors, n_minority - 1)
        # Nearest neighbours within the class, excluding self (column 0 of the sorted result).
        neighbor_idx = _knn_indices(minority, minority, k + 1)[:, 1:]
        base_rows = rng.integers(0, n_minority, size=n_needed)
        neighbor_choice = rng.integers(0, k, size=n_needed)
        gaps = rng.random(n_needed)[:, None]
        base = minority[base_rows]
        neighbors = minority[neighbor_idx[base_rows, neighbor_choice]]
        return base + gaps * (neighbors - base)


class BorderlineSMOTE(SMOTE):
    """Borderline-SMOTE: synthesize only from minority samples near the decision boundary.

    Classifies each minority sample by how many of its ``m_neighbors`` nearest neighbours (across
    *all* classes) belong to a different class: "noise" (all neighbours are other-class -- skipped),
    "danger"/borderline (a majority but not all are other-class -- the synthesis seeds), or "safe"
    (mostly same-class -- not used as seeds). Synthetic points are then generated the same way as
    plain :class:`SMOTE`, but only interpolating from the borderline seeds, concentrating new points
    where the boundary is actually contested.

    Args:
        k_neighbors: as in :class:`SMOTE` -- the within-minority neighbours synthesis interpolates
            toward.
        m_neighbors: the all-class neighbourhood size used to classify each minority sample as
            noise / borderline / safe.
        random_state: seed / Generator.
    """

    def __init__(self, *, k_neighbors: int = 5, m_neighbors: int = 10, random_state=None) -> None:
        super().__init__(k_neighbors=k_neighbors, random_state=random_state)
        self.m_neighbors = m_neighbors

    def fit_resample(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Return a class-balanced ``(X_resampled, y_resampled)`` from borderline minority seeds."""
        if self.k_neighbors < 1 or self.m_neighbors < 1:
            raise ValueError("k_neighbors and m_neighbors must both be >= 1.")
        X, y = _validate_Xy(X, y)
        rng = _p.as_generator(self.random_state)
        counts = _class_counts(y)
        target = max(counts.values())
        parts_X = [X]
        parts_y = [y]
        for label, count in counts.items():
            n_needed = target - count
            if n_needed <= 0:
                continue
            minority = X[y == label]
            seeds = self._borderline_seeds(X, y, label, minority)
            if seeds.shape[0] < 2:
                # Not enough borderline seeds to interpolate -> fall back to plain SMOTE
                # over the whole minority class for this label.
                seeds = minority
            synthetic = self._synthesize(seeds, n_needed, rng)
            parts_X.append(synthetic)
            parts_y.append(np.full(n_needed, label, dtype=y.dtype))
        return np.vstack(parts_X), np.concatenate(parts_y)

    def _borderline_seeds(self, X, y, label, minority: np.ndarray) -> np.ndarray:
        """Return the subset of ``minority`` classified as borderline ("in danger")."""
        m = min(self.m_neighbors, X.shape[0] - 1)
        # Neighbours over the WHOLE dataset (excluding self).
        nn = _knn_indices(minority, X, m + 1)[:, 1:]
        neighbor_labels = y[nn]
        other_class_counts = np.sum(neighbor_labels != label, axis=1)
        # Borderline: at least half but not all neighbours are a different class.
        half = m / 2.0
        danger = (other_class_counts >= half) & (other_class_counts < m)
        return minority[danger]
