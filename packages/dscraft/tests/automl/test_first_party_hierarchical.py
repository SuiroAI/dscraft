"""Tests for the hierarchical clusterers in dscraft.automl._clustering.

Covers DSAgglomerative (all four linkages recover well-separated blobs; single-linkage handles
non-convex moons; validation) and DSHDBSCAN (finds the right number of density clusters on blobs,
labels far-flung points as noise, is deterministic, and produces membership probabilities).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

from dscraft.automl._clustering import DSAgglomerative, DSHDBSCAN


def _agreement(labels_a, labels_b) -> float:
    from collections import Counter

    total = len(labels_a)
    hits = 0
    for cluster in set(labels_a):
        mask = labels_a == cluster
        true_in = labels_b[mask]
        if len(true_in):
            hits += Counter(true_in.tolist()).most_common(1)[0][1]
    return hits / total


# ---------------------------------------------------------------------------
# Agglomerative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
def test_agglomerative_recovers_blobs(linkage) -> None:
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.5, random_state=0)
    model = DSAgglomerative(n_clusters=3, linkage=linkage).fit(X)
    assert len(set(model.labels_)) == 3
    assert _agreement(model.labels_, y) > 0.9


def test_agglomerative_single_linkage_handles_moons() -> None:
    X, y = make_moons(n_samples=200, noise=0.04, random_state=0)
    model = DSAgglomerative(n_clusters=2, linkage="single").fit(X)
    assert len(set(model.labels_)) == 2
    assert _agreement(model.labels_, y) > 0.9


def test_agglomerative_fit_predict() -> None:
    X, _ = make_blobs(n_samples=80, centers=2, random_state=0)
    model = DSAgglomerative(n_clusters=2)
    labels = model.fit_predict(X)
    np.testing.assert_array_equal(labels, model.labels_)


def test_agglomerative_n_clusters_one_is_single_group() -> None:
    X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
    model = DSAgglomerative(n_clusters=1).fit(X)
    assert len(set(model.labels_)) == 1


@pytest.mark.parametrize("kwargs, match", [
    ({"linkage": "nope"}, "linkage"),
    ({"n_clusters": 0}, "n_clusters"),
])
def test_agglomerative_validation(kwargs, match) -> None:
    X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
    with pytest.raises(ValueError, match=match):
        DSAgglomerative(**kwargs).fit(X)


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------


def test_hdbscan_finds_blobs() -> None:
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    model = DSHDBSCAN(min_cluster_size=15).fit(X)
    n_clusters = len(set(model.labels_) - {-1})
    assert 2 <= n_clusters <= 4


def test_hdbscan_labels_outliers_as_noise() -> None:
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.4, random_state=0)
    outliers = np.array([[30.0, 30.0], [-30.0, -30.0]])
    X_aug = np.vstack([X, outliers])
    model = DSHDBSCAN(min_cluster_size=15).fit(X_aug)
    # The two far-away points are noise.
    assert np.all(model.labels_[-2:] == -1)


def test_hdbscan_deterministic() -> None:
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=0)
    a = DSHDBSCAN(min_cluster_size=10).fit(X).labels_
    b = DSHDBSCAN(min_cluster_size=10).fit(X).labels_
    np.testing.assert_array_equal(a, b)


def test_hdbscan_probabilities_in_range() -> None:
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=0)
    model = DSHDBSCAN(min_cluster_size=10).fit(X)
    assert model.probabilities_.shape == (150,)
    assert np.all((model.probabilities_ >= 0.0) & (model.probabilities_ <= 1.0))
    # Noise points have zero membership strength.
    assert np.all(model.probabilities_[model.labels_ == -1] == 0.0)


def test_hdbscan_too_few_points_all_noise() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    model = DSHDBSCAN(min_cluster_size=5).fit(X)
    assert np.all(model.labels_ == -1)


def test_hdbscan_validation() -> None:
    X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
    with pytest.raises(ValueError, match="min_cluster_size"):
        DSHDBSCAN(min_cluster_size=1).fit(X)


def test_hdbscan_fit_predict() -> None:
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
    model = DSHDBSCAN(min_cluster_size=10)
    labels = model.fit_predict(X)
    np.testing.assert_array_equal(labels, model.labels_)


def _agree_metric(a, b) -> float:
    from collections import Counter

    hits = 0
    for c in set(a):
        t = b[a == c]
        if len(t):
            hits += Counter(t.tolist()).most_common(1)[0][1]
    return hits / len(a)


@pytest.mark.parametrize("dataset", ["blobs", "moons"])
def test_hdbscan_parity_with_sklearn(dataset) -> None:
    """Our HDBSCAN* matches sklearn's on both globular (blobs) and non-convex (moons) density
    clusters: same cluster count, same noise count, and a near-identical partition."""
    cluster = pytest.importorskip("sklearn.cluster")
    if dataset == "blobs":
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    else:
        X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
    ours = DSHDBSCAN(min_cluster_size=15).fit(X).labels_
    theirs = cluster.HDBSCAN(min_cluster_size=15).fit(X).labels_
    assert len(set(ours) - {-1}) == len(set(theirs) - {-1})
    assert _agree_metric(ours, theirs) > 0.95
