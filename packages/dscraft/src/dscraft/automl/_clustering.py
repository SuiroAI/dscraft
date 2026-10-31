"""First-party clustering algorithms for ``dscraft.automl``.

This internal module implements unsupervised clustering from scratch on ``numpy`` -- the
first-party replacement for the ``sklearn.cluster.HDBSCAN`` backend ``clustering.py`` currently
wraps, and a broadening of the clustering surface to cover *all the major styles* of clustering
rather than a single backend:

- :class:`DSKMeans` -- centroid-based clustering via Lloyd's algorithm with k-means++ seeding.
  Partitions the data into exactly ``n_clusters`` groups by iteratively assigning each point to
  its nearest centroid and recomputing centroids as cluster means.
- :class:`DSDBSCAN` -- density-based clustering: expands clusters from "core" points (those with
  at least ``min_samples`` neighbours within ``eps``), labelling low-density points as noise
  (``-1``). Finds an arbitrary number of arbitrarily-shaped clusters and needs no ``n_clusters``.

The agglomerative and HDBSCAN-equivalent (hierarchical density-based) clusterers that complete the
"all styles" coverage, and the swap of the public ``build_clusterer`` allowlist off
``sklearn.cluster.HDBSCAN``, land in the next step of the first-party automl/eda rebuild (see
``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``); this module is where they will live.

Each clusterer is **scikit-learn API-compatible** -- it subclasses ``sklearn.base.ClusterMixin``/
``BaseEstimator`` and implements ``fit``/``fit_predict``/``get_params``/``set_params`` and sets the
standard ``labels_`` fitted attribute -- so it is a drop-in for the existing
``build_clusterer`` contract (which returned sklearn-compatible clusterers callers ``.fit_predict``
themselves). Only sklearn's estimator-protocol plumbing is used; all clustering math is ours. All
randomness flows through an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from dscraft.automl import _primitives as _p

__all__ = [
    "DSKMeans",
    "DSDBSCAN",
]


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``(len(A), len(B))`` matrix of squared Euclidean distances, clipped non-negative."""
    a_sq = np.sum(A**2, axis=1, keepdims=True)
    b_sq = np.sum(B**2, axis=1, keepdims=True).T
    d = a_sq - 2.0 * A @ B.T + b_sq
    np.maximum(d, 0.0, out=d)
    return d


class DSKMeans(ClusterMixin, BaseEstimator):
    """First-party k-means clustering (Lloyd's algorithm, k-means++ seeding).

    Args:
        n_clusters: number of clusters ``k`` (``>= 1``).
        max_iter: maximum Lloyd iterations.
        tol: convergence tolerance on the centroid shift (Frobenius norm) between iterations.
        n_init: number of independent k-means++ restarts; the run with the lowest final inertia
            (sum of squared point-to-centroid distances) is kept, guarding against a single bad
            seeding.
        random_state: seed / Generator for k-means++ seeding and restarts.

    Attributes (after :meth:`fit`):
        cluster_centers_: ``(n_clusters, n_features)`` final centroids.
        labels_: ``(n_samples,)`` cluster index per training point.
        inertia_: sum of squared distances of points to their assigned centroid.
        n_iter_: iterations run in the best restart.
        n_features_in_: number of input features.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state=None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None) -> "DSKMeans":
        """Cluster ``X``. ``y`` is ignored (unsupervised). Returns ``self``."""
        X = _p.check_array_2d(X, name="X")
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}.")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed n_samples={X.shape[0]}."
            )
        self.n_features_in_ = X.shape[1]
        rng = _p.as_generator(self.random_state)

        best = None  # (inertia, centers, labels, n_iter)
        for _ in range(max(1, self.n_init)):
            centers = self._kmeanspp_init(X, rng)
            centers, labels, inertia, n_iter = self._lloyd(X, centers)
            if best is None or inertia < best[0]:
                best = (inertia, centers, labels, n_iter)

        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        return self

    def _kmeanspp_init(self, X: np.ndarray, rng) -> np.ndarray:
        """k-means++ seeding: spread initial centroids proportional to squared distance."""
        n = X.shape[0]
        centers = [X[rng.integers(0, n)]]
        for _ in range(1, self.n_clusters):
            d_sq = _pairwise_sq_dists(X, np.vstack(centers)).min(axis=1)
            total = d_sq.sum()
            if total == 0:  # all points coincide with a chosen center
                centers.append(X[rng.integers(0, n)])
                continue
            probs = d_sq / total
            centers.append(X[rng.choice(n, p=probs)])
        return np.vstack(centers)

    def _lloyd(self, X: np.ndarray, centers: np.ndarray):
        labels = np.zeros(X.shape[0], dtype=np.int64)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            dist_sq = _pairwise_sq_dists(X, centers)
            labels = np.argmin(dist_sq, axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if members.shape[0] > 0:
                    new_centers[k] = members.mean(axis=0)
                # An empty cluster keeps its previous centroid (a standard, stable choice).
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift < self.tol:
                break
        dist_sq = _pairwise_sq_dists(X, centers)
        labels = np.argmin(dist_sq, axis=1)
        inertia = float(dist_sq[np.arange(X.shape[0]), labels].sum())
        return centers, labels, inertia, n_iter

    def predict(self, X) -> np.ndarray:
        """Assign each row of ``X`` to its nearest fitted centroid."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return np.argmin(_pairwise_sq_dists(X, self.cluster_centers_), axis=1)


class DSDBSCAN(ClusterMixin, BaseEstimator):
    """First-party DBSCAN density-based clustering.

    Groups points that are densely packed (core points have ``>= min_samples`` neighbours within
    ``eps``, counting themselves), expands clusters through connected core points, attaches
    border points to an adjacent cluster, and labels the rest as noise (``-1``). Finds an
    arbitrary number of arbitrarily-shaped clusters with no ``n_clusters`` argument.

    Args:
        eps: neighbourhood radius (Euclidean); ``> 0``.
        min_samples: minimum neighbourhood size (including the point itself) for a core point.

    Attributes (after :meth:`fit`):
        labels_: ``(n_samples,)`` cluster index per point, or ``-1`` for noise.
        core_sample_indices_: indices of the core points found.
        n_features_in_: number of input features.
    """

    _NOISE = -1
    _UNVISITED = -2

    def __init__(self, eps: float = 0.5, *, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None) -> "DSDBSCAN":
        """Cluster ``X``. ``y`` is ignored (unsupervised). Returns ``self``."""
        X = _p.check_array_2d(X, name="X")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}.")
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        eps_sq = self.eps**2

        # Precompute the eps-neighbourhood of every point (brute-force; exact).
        dist_sq = _pairwise_sq_dists(X, X)
        neighborhoods = [np.nonzero(row <= eps_sq)[0] for row in dist_sq]
        is_core = np.array([nb.shape[0] >= self.min_samples for nb in neighborhoods])

        labels = np.full(n, self._UNVISITED, dtype=np.int64)
        cluster_id = -1
        for i in range(n):
            if labels[i] != self._UNVISITED:
                continue
            if not is_core[i]:
                labels[i] = self._NOISE  # may be reclaimed as a border point later
                continue
            cluster_id += 1
            self._expand(i, neighborhoods, is_core, labels, cluster_id)

        self.labels_ = labels
        self.core_sample_indices_ = np.nonzero(is_core)[0]
        return self

    def _expand(self, seed, neighborhoods, is_core, labels, cluster_id) -> None:
        """Grow ``cluster_id`` outward from core point ``seed`` (iterative BFS over core points)."""
        labels[seed] = cluster_id
        queue = list(neighborhoods[seed])
        head = 0
        while head < len(queue):
            point = queue[head]
            head += 1
            if labels[point] == self._NOISE:
                labels[point] = cluster_id  # border point reclaimed into this cluster
            if labels[point] != self._UNVISITED:
                continue
            labels[point] = cluster_id
            if is_core[point]:
                queue.extend(neighborhoods[point])

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_`` in one call (sklearn ``ClusterMixin`` contract)."""
        return self.fit(X).labels_
