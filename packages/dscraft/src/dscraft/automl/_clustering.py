"""First-party clustering algorithms for ``dscraft.automl``.

This internal module implements unsupervised clustering from scratch on ``numpy`` -- the
first-party replacement for the ``sklearn.cluster.HDBSCAN`` backend ``clustering.py`` used to
wrap, and a broadening of the clustering surface to cover *all the major styles* of clustering
rather than a single backend:

- :class:`DSKMeans` -- centroid-based clustering via Lloyd's algorithm with k-means++ seeding.
  Partitions the data into exactly ``n_clusters`` groups by iteratively assigning each point to
  its nearest centroid and recomputing centroids as cluster means.
- :class:`DSDBSCAN` -- density-based clustering: expands clusters from "core" points (those with
  at least ``min_samples`` neighbours within ``eps``), labelling low-density points as noise
  (``-1``). Finds an arbitrary number of arbitrarily-shaped clusters and needs no ``n_clusters``.
- :class:`DSAgglomerative` -- bottom-up hierarchical clustering: repeatedly merge the two closest
  clusters (Ward / complete / average / single linkage) until ``n_clusters`` remain.
- :class:`DSHDBSCAN` -- hierarchical density-based clustering, the first-party replacement for
  ``sklearn.cluster.HDBSCAN``: builds the mutual-reachability graph, its minimum spanning tree, and
  the single-linkage hierarchy, then condenses that hierarchy and selects the most *stable*
  clusters (the Excess-of-Mass selection), labelling the rest as noise. Implemented directly from
  the published HDBSCAN* definition (Campello/McInnes), not adapted from any library's source.

Together these cover the major clustering *styles* -- centroid, density, hierarchical-agglomerative,
and hierarchical-density -- entirely first-party.

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
    "DSAgglomerative",
    "DSHDBSCAN",
]


def _deselect_subtree(child_ids, children_clusters: dict, selected: set) -> None:
    """Remove every condensed cluster in the subtrees rooted at ``child_ids`` from ``selected``.

    Used by HDBSCAN*'s Excess-of-Mass step: when a parent cluster is selected, none of its
    descendants may also be selected, so this walks each child subtree and discards it.
    """
    stack = list(child_ids)
    while stack:
        node = stack.pop()
        selected.discard(node)
        stack.extend(children_clusters.get(node, ()))


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


# ---------------------------------------------------------------------------
# Hierarchical agglomerative
# ---------------------------------------------------------------------------


class DSAgglomerative(ClusterMixin, BaseEstimator):
    """First-party bottom-up (agglomerative) hierarchical clustering.

    Starts with every point its own cluster and repeatedly merges the two closest clusters under
    the chosen ``linkage`` until ``n_clusters`` remain. Linkages supported:

    - ``"single"`` -- minimum pairwise distance between the two clusters (chaining, finds
      non-convex shapes).
    - ``"complete"`` -- maximum pairwise distance (compact, roughly equal-diameter clusters).
    - ``"average"`` -- mean pairwise distance (UPGMA).
    - ``"ward"`` -- merge the pair that minimizes the increase in total within-cluster variance
      (Ward's criterion, via the Lance-Williams update); the usual default for Euclidean data.

    Args:
        n_clusters: number of clusters to stop at (``>= 1``).
        linkage: one of ``"ward"``, ``"complete"``, ``"average"``, ``"single"``.

    Attributes (after :meth:`fit`):
        labels_: ``(n_samples,)`` cluster index per point.
        n_features_in_: number of input features.
    """

    _LINKAGES = frozenset({"ward", "complete", "average", "single"})

    def __init__(self, n_clusters: int = 2, *, linkage: str = "ward") -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None) -> "DSAgglomerative":
        """Cluster ``X``. ``y`` is ignored (unsupervised). Returns ``self``."""
        X = _p.check_array_2d(X, name="X")
        if self.linkage not in self._LINKAGES:
            raise ValueError(f"linkage must be one of {sorted(self._LINKAGES)!r}, got {self.linkage!r}.")
        n = X.shape[0]
        if not (1 <= self.n_clusters <= n):
            raise ValueError(f"n_clusters must be in [1, n_samples={n}], got {self.n_clusters}.")
        self.n_features_in_ = X.shape[1]

        # Active clusters: each a list of member indices, plus their sizes and a live distance map.
        members = {i: [i] for i in range(n)}
        sizes = {i: 1 for i in range(n)}
        dist = _pairwise_sq_dists(X, X)
        np.sqrt(dist, out=dist)  # work in Euclidean distance (ward uses squared internally)

        # Lance-Williams-ready pairwise cluster distances, seeded from point distances.
        cluster_dist: dict = {}
        active = list(range(n))
        for a in range(n):
            for b in range(a + 1, n):
                cluster_dist[(a, b)] = self._initial_distance(dist[a, b])

        next_id = n
        while len(active) > self.n_clusters:
            # Find the closest active pair.
            (i, j), _d = min(
                ((pair, d) for pair, d in cluster_dist.items()
                 if pair[0] in members and pair[1] in members),
                key=lambda kv: kv[1],
            )
            new = next_id
            next_id += 1
            members[new] = members[i] + members[j]
            sizes[new] = sizes[i] + sizes[j]
            # Update distances from the merged cluster to every other active cluster.
            for k in list(members):
                if k in (i, j, new):
                    continue
                d_ik = self._get(cluster_dist, i, k)
                d_jk = self._get(cluster_dist, j, k)
                d_ij = self._get(cluster_dist, i, j)
                merged = self._lance_williams(d_ik, d_jk, d_ij, sizes[i], sizes[j], sizes[k])
                cluster_dist[self._key(new, k)] = merged
            del members[i], members[j], sizes[i], sizes[j]
            active = list(members)

        labels = np.empty(n, dtype=np.int64)
        for cluster_idx, cluster in enumerate(members.values()):
            for point in cluster:
                labels[point] = cluster_idx
        self.labels_ = labels
        return self

    def _initial_distance(self, d: float) -> float:
        # Ward works with squared distances internally; the others use the raw distance.
        return d**2 if self.linkage == "ward" else d

    @staticmethod
    def _key(a: int, b: int) -> tuple:
        return (a, b) if a < b else (b, a)

    def _get(self, cluster_dist, a, b) -> float:
        return cluster_dist[self._key(a, b)]

    def _lance_williams(self, d_ik, d_jk, d_ij, n_i, n_j, n_k) -> float:
        """Lance-Williams recurrence for the distance from merged (i,j) to k, per linkage."""
        if self.linkage == "single":
            return min(d_ik, d_jk)
        if self.linkage == "complete":
            return max(d_ik, d_jk)
        if self.linkage == "average":
            return (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
        # ward (on squared distances)
        total = n_i + n_j + n_k
        return (
            (n_i + n_k) / total * d_ik
            + (n_j + n_k) / total * d_jk
            - n_k / total * d_ij
        )

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_`` in one call."""
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# Hierarchical density-based (HDBSCAN*)
# ---------------------------------------------------------------------------


class DSHDBSCAN(ClusterMixin, BaseEstimator):
    """First-party hierarchical density-based clustering (HDBSCAN*).

    The density-based counterpart to :class:`DSAgglomerative` and the first-party replacement for
    ``sklearn.cluster.HDBSCAN``. The algorithm, implemented directly from the published HDBSCAN*
    definition:

    1. Compute each point's *core distance* -- the distance to its ``min_samples``-th nearest
       neighbour -- and from it the *mutual reachability distance* between every pair,
       ``max(core_i, core_j, d_ij)``.
    2. Build the minimum spanning tree of the mutual-reachability graph (Prim's algorithm).
    3. Turn the MST into a single-linkage hierarchy by adding edges in increasing weight order
       (union-find), recording the density level (``1/weight``) at which clusters merge.
    4. *Condense* that hierarchy: walk it top-down, and whenever a split produces a child smaller
       than ``min_cluster_size``, treat those points as "falling out of" the parent rather than
       forming a new cluster.
    5. Select the final flat clustering by the Excess-of-Mass criterion -- keep the set of
       condensed-tree clusters with the greatest total *stability* (persistence across density
       levels), labelling everything else as noise (``-1``).

    Args:
        min_cluster_size: smallest grouping accepted as a cluster (``>= 2``); smaller groupings
            become noise.
        min_samples: neighbourhood size defining the core distance (density conservatism);
            defaults to ``min_cluster_size`` when ``None``.

    Attributes (after :meth:`fit`):
        labels_: ``(n_samples,)`` cluster index per point, or ``-1`` for noise.
        probabilities_: per-point cluster "membership strength" in ``[0, 1]`` (``0`` for noise).
        n_features_in_: number of input features.
    """

    def __init__(self, *, min_cluster_size: int = 5, min_samples: Optional[int] = None) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit(self, X, y=None) -> "DSHDBSCAN":
        """Cluster ``X``. ``y`` is ignored (unsupervised). Returns ``self``."""
        X = _p.check_array_2d(X, name="X")
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size}.")
        n = X.shape[0]
        min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
        min_samples = max(1, min(min_samples, n - 1))
        self.n_features_in_ = X.shape[1]

        if n < self.min_cluster_size:
            # Too few points to form even one cluster -> all noise.
            self.labels_ = np.full(n, -1, dtype=np.int64)
            self.probabilities_ = np.zeros(n, dtype=np.float64)
            return self

        d = _pairwise_sq_dists(X, X)
        np.sqrt(d, out=d)
        core = np.sort(d, axis=1)[:, min_samples]  # distance to the min_samples-th neighbour
        # Mutual reachability distance matrix.
        mr = np.maximum(np.maximum(core[:, None], core[None, :]), d)

        mst = self._prim_mst(mr)  # list of (weight, u, v), unsorted
        hierarchy = self._single_linkage(mst, n)
        labels, probabilities = self._condense_and_extract(hierarchy, n)
        self.labels_ = labels
        self.probabilities_ = probabilities
        return self

    def _prim_mst(self, mr: np.ndarray) -> list:
        """Minimum spanning tree of the dense mutual-reachability graph via Prim's algorithm."""
        n = mr.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        best = mr[0].copy()
        parent = np.zeros(n, dtype=np.int64)
        in_tree[0] = True
        best[0] = np.inf
        edges = []
        for _ in range(n - 1):
            nxt = int(np.argmin(best))
            edges.append((float(best[nxt]), int(parent[nxt]), nxt))
            in_tree[nxt] = True
            best[nxt] = np.inf
            closer = mr[nxt] < best
            closer &= ~in_tree
            best[closer] = mr[nxt][closer]
            parent[closer] = nxt
        return edges

    def _single_linkage(self, mst: list, n: int) -> list:
        """Build the single-linkage merge hierarchy from the MST.

        Adds MST edges in increasing weight (union-find), creating one internal hierarchy node per
        merge. Returns a list of nodes indexed ``n .. 2n-2``; each node records its two child node
        ids and the distance (``weight``) at which they merged. Leaf nodes ``0 .. n-1`` are the
        original points. This is the standard SLINK dendrogram the condensed tree is built from.
        """
        edges = sorted(mst, key=lambda e: e[0])
        parent = list(range(2 * n - 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        node_of_root = {i: i for i in range(n)}  # union-find root -> hierarchy node id
        nodes = []  # index i-n -> (left_child, right_child, distance, size)
        node_size = {i: 1 for i in range(n)}
        next_node = n
        for weight, u, v in edges:
            ru, rv = find(u), find(v)
            left = node_of_root[ru]
            right = node_of_root[rv]
            new_id = next_node
            next_node += 1
            size = node_size[left] + node_size[right]
            nodes.append((left, right, float(weight), size))
            node_size[new_id] = size
            parent[ru] = new_id
            parent[rv] = new_id
            node_of_root[find(new_id)] = new_id
        return nodes

    def _condense_and_extract(self, nodes: list, n: int):
        """Condense the single-linkage hierarchy and extract flat clusters by Excess of Mass.

        Walks the dendrogram top-down. At each internal node's split, a child subtree with fewer
        than ``min_cluster_size`` leaves "falls out" (its points leave the parent cluster at this
        split's lambda = 1/distance); a child with at least ``min_cluster_size`` leaves becomes (or
        continues) a condensed cluster. Each condensed cluster's *stability* is the sum over its
        fall-out points of ``lambda_fall_out - lambda_birth`` plus the surviving points'
        persistence to ``max_lambda``. Excess of Mass then selects the set of condensed clusters
        maximizing total stability without choosing both an ancestor and a descendant.
        """
        if not nodes:
            return np.full(n, -1, dtype=np.int64), np.zeros(n)
        root = n + len(nodes) - 1
        min_size = self.min_cluster_size

        # Precompute each node's leaves iteratively (avoid recursion on deep trees).
        leaves_cache: dict = {}

        def node_size(node_id: int) -> int:
            return 1 if node_id < n else nodes[node_id - n][3]

        def leaves(node_id: int) -> list:
            if node_id < n:
                return [node_id]
            if node_id in leaves_cache:
                return leaves_cache[node_id]
            out = []
            stack = [node_id]
            while stack:
                nid = stack.pop()
                if nid < n:
                    out.append(nid)
                else:
                    left, right, _d, _s = nodes[nid - n]
                    stack.append(left)
                    stack.append(right)
            leaves_cache[node_id] = out
            return out

        birth_lambda = {root: 0.0}
        children_clusters = {root: []}
        fallouts = {root: []}  # cluster_id -> list of (point, lambda_at_fallout)

        stack = [(root, root)]
        while stack:
            node_id, cluster = stack.pop()
            if node_id < n:
                continue
            left, right, dist, _size = nodes[node_id - n]
            lam = 1.0 / dist if dist > 0 else float("inf")
            left_big = node_size(left) >= min_size
            right_big = node_size(right) >= min_size
            if left_big and right_big:
                for child in (left, right):
                    birth_lambda[child] = lam
                    fallouts[child] = []
                    children_clusters[child] = []
                    children_clusters[cluster].append(child)
                    stack.append((child, child))
            else:
                for child, big in ((left, left_big), (right, right_big)):
                    if big:
                        stack.append((child, cluster))
                    else:
                        for p in leaves(child):
                            fallouts[cluster].append((p, lam))

        finite_lambdas = [1.0 / nodes[c - n][2] for c in birth_lambda if c >= n and nodes[c - n][2] > 0]
        max_lambda = max(finite_lambdas) if finite_lambdas else 0.0

        all_leaves = {c: set(leaves(c)) for c in birth_lambda}
        fallout_lambda = {c: dict(fallouts[c]) for c in birth_lambda}

        # Stability (HDBSCAN* "relative excess of mass"): a cluster accrues, per point, the
        # persistence of that point WHILE IT BELONGS TO THIS CLUSTER -- i.e. from the cluster's
        # birth lambda until the point leaves it. A point leaves either by falling out (its
        # recorded fall-out lambda) OR by descending into a child cluster (that child's birth
        # lambda); points that neither fall out nor descend persist to max_lambda. Crediting
        # descending points only up to the child's birth lambda (rather than max_lambda) is the
        # key correctness point: otherwise a parent -- especially the root -- accumulates the full
        # persistence of every descendant point and always wins Excess of Mass.
        child_birth_of_point: dict = {}
        for c in birth_lambda:
            lam_c = birth_lambda[c]
            for ch in children_clusters[c]:
                lam_ch = birth_lambda[ch]
                for p in all_leaves[ch]:
                    # A point may sit in nested children; record the SHALLOWEST child's birth
                    # lambda as where it leaves THIS cluster c. Since we iterate parents before
                    # we would need it, keep the smallest (earliest) child-birth lambda seen.
                    key = (c, p)
                    prev = child_birth_of_point.get(key)
                    if prev is None or lam_ch < prev:
                        child_birth_of_point[key] = lam_ch

        stability = {}
        for c in birth_lambda:
            lam_birth = birth_lambda[c]
            s = 0.0
            fo = fallout_lambda[c]
            for p in all_leaves[c]:
                if p in fo and fo[p] != float("inf"):
                    lam_leave = fo[p]
                elif (c, p) in child_birth_of_point:
                    lam_leave = child_birth_of_point[(c, p)]
                else:
                    lam_leave = max_lambda
                s += max(0.0, lam_leave - lam_birth)
            stability[c] = s

        # Excess of Mass, bottom-up (deepest birth lambda first). The root ("everything") is never
        # itself a selectable flat cluster, so it must never win on its own stability -- it always
        # propagates its children's selected stability upward. Treating the root like any other
        # node would let its (necessarily huge) whole-dataset stability deselect every real
        # cluster beneath it, collapsing the result to all-noise.
        selected: set = set()
        selected_stability: dict = {}
        order = sorted(birth_lambda, key=lambda c: birth_lambda[c], reverse=True)
        for c in order:
            child_sum = sum(selected_stability.get(ch, 0.0) for ch in children_clusters[c])
            is_root = c == root
            if not is_root and (not children_clusters[c] or stability[c] >= child_sum):
                selected_stability[c] = stability[c]
                _deselect_subtree(children_clusters[c], children_clusters, selected)
                selected.add(c)
            else:
                # Root, or a node whose children are collectively more stable: keep the children.
                selected_stability[c] = child_sum

        selected.discard(root)  # defensive: the root is never a flat cluster label

        labels = np.full(n, -1, dtype=np.int64)
        prob = np.zeros(n, dtype=np.float64)
        for label_idx, c in enumerate(sorted(selected)):
            lam_birth = birth_lambda[c]
            fo = dict(fallouts[c])
            span = max_lambda - lam_birth
            for p in all_leaves[c]:
                labels[p] = label_idx
                lam_p = fo.get(p, max_lambda)
                prob[p] = float(np.clip((lam_p - lam_birth) / span, 0.0, 1.0)) if span > 0 else 1.0
        return labels, prob

    def fit_predict(self, X, y=None) -> np.ndarray:
        """Fit and return ``labels_`` in one call."""
        return self.fit(X).labels_
