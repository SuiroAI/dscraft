"""Tests for dscraft.automl._clustering.

Validates the first-party clusterers: k-means recovers well-separated blobs and computes a sane
inertia, DBSCAN finds density clusters and labels sparse points as noise, both are reproducible /
deterministic, sklearn-API-compatible, and raise on bad arguments. Where sklearn is present, a
parity check on cluster agreement confirms correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from dscraft.automl._clustering import DSDBSCAN, DSKMeans


def _agreement(labels_a, labels_b) -> float:
    """Adjusted-Rand-free cluster agreement: best-matching accuracy via contingency argmax.

    Good enough to assert 'these two labelings mostly agree' without importing a metric: for each
    predicted cluster, credit the most common true-label overlap.
    """
    from collections import Counter

    total = len(labels_a)
    hits = 0
    for cluster in set(labels_a):
        mask = labels_a == cluster
        true_in_cluster = labels_b[mask]
        if len(true_in_cluster) == 0:
            continue
        hits += Counter(true_in_cluster.tolist()).most_common(1)[0][1]
    return hits / total


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


def test_kmeans_recovers_blobs() -> None:
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    km = DSKMeans(n_clusters=4, random_state=0).fit(X)
    assert km.cluster_centers_.shape == (4, 2)
    assert km.labels_.shape == (300,)
    assert _agreement(km.labels_, y) > 0.95  # clusters align with the true blobs


def test_kmeans_inertia_decreases_with_more_clusters() -> None:
    X, _ = make_blobs(n_samples=200, centers=5, random_state=1)
    low_k = DSKMeans(n_clusters=2, random_state=0).fit(X).inertia_
    high_k = DSKMeans(n_clusters=8, random_state=0).fit(X).inertia_
    assert high_k < low_k  # more centroids -> lower within-cluster SS


def test_kmeans_predict_assigns_nearest_centroid() -> None:
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.4, random_state=2)
    km = DSKMeans(n_clusters=3, random_state=0).fit(X)
    # Predicting the training data reproduces labels_.
    np.testing.assert_array_equal(km.predict(X), km.labels_)
    # A point at a centroid is assigned to that centroid's cluster.
    for k in range(3):
        assert km.predict(km.cluster_centers_[k][None, :])[0] == k


def test_kmeans_reproducible() -> None:
    X, _ = make_blobs(n_samples=200, centers=4, random_state=3)
    a = DSKMeans(n_clusters=4, random_state=7).fit(X)
    b = DSKMeans(n_clusters=4, random_state=7).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    assert a.inertia_ == pytest.approx(b.inertia_)


@pytest.mark.parametrize("n_clusters, match", [(0, "n_clusters"), (10_000, "cannot exceed")])
def test_kmeans_validation(n_clusters, match) -> None:
    X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
    with pytest.raises(ValueError, match=match):
        DSKMeans(n_clusters=n_clusters).fit(X)


def test_kmeans_predict_feature_mismatch() -> None:
    X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
    km = DSKMeans(n_clusters=3, random_state=0).fit(X)
    with pytest.raises(ValueError, match="features"):
        km.predict(np.zeros((2, 5)))


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


def test_dbscan_finds_blobs_and_labels_noise() -> None:
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=0)
    # Add a few far-flung noise points.
    noise = np.array([[20.0, 20.0], [-20.0, -20.0], [25.0, -25.0]])
    X_aug = np.vstack([X, noise])
    db = DSDBSCAN(eps=1.0, min_samples=5).fit(X_aug)
    n_clusters = len(set(db.labels_) - {-1})
    assert 2 <= n_clusters <= 4
    # The appended far-away points are labelled noise.
    assert np.all(db.labels_[-3:] == -1)


def test_dbscan_finds_nonconvex_moons() -> None:
    X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
    db = DSDBSCAN(eps=0.2, min_samples=5).fit(X)
    n_clusters = len(set(db.labels_) - {-1})
    # DBSCAN handles the two interleaving crescents that k-means cannot.
    assert n_clusters == 2
    assert _agreement(db.labels_, y) > 0.9


def test_dbscan_all_noise_when_eps_tiny() -> None:
    X, _ = make_blobs(n_samples=100, centers=3, random_state=0)
    db = DSDBSCAN(eps=1e-6, min_samples=5).fit(X)
    assert np.all(db.labels_ == -1)


def test_dbscan_fit_predict_matches_labels() -> None:
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.4, random_state=0)
    db = DSDBSCAN(eps=1.0, min_samples=5)
    labels = db.fit_predict(X)
    np.testing.assert_array_equal(labels, db.labels_)


def test_dbscan_core_sample_indices_are_core() -> None:
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.4, random_state=0)
    db = DSDBSCAN(eps=1.0, min_samples=5).fit(X)
    # Every core index is assigned to a real cluster, not noise.
    assert np.all(db.labels_[db.core_sample_indices_] >= 0)


@pytest.mark.parametrize("kwargs, match", [({"eps": 0.0}, "eps"), ({"min_samples": 0}, "min_samples")])
def test_dbscan_validation(kwargs, match) -> None:
    X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
    with pytest.raises(ValueError, match=match):
        DSDBSCAN(**kwargs).fit(X)


# ---------------------------------------------------------------------------
# Parity with sklearn (correctness cross-check; skipped if absent)
# ---------------------------------------------------------------------------


def test_kmeans_parity_with_sklearn() -> None:
    cluster = pytest.importorskip("sklearn.cluster")
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    ours = DSKMeans(n_clusters=4, random_state=0).fit(X).labels_
    theirs = cluster.KMeans(n_clusters=4, random_state=0, n_init=10).fit(X).labels_
    # Cluster *labels* are arbitrary integers; compare agreement of the partition instead.
    assert _agreement(ours, theirs) > 0.95


def test_dbscan_parity_with_sklearn() -> None:
    cluster = pytest.importorskip("sklearn.cluster")
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
    ours = DSDBSCAN(eps=0.2, min_samples=5).fit(X).labels_
    theirs = cluster.DBSCAN(eps=0.2, min_samples=5).fit(X).labels_
    # Same number of clusters, and the non-noise partition agrees closely.
    assert len(set(ours) - {-1}) == len(set(theirs) - {-1})
    assert _agreement(ours, theirs) > 0.9
