"""Tests for dscraft.automl.clustering.

Covers the HDBSCAN clustering allowlist: a smoke test on a small
synthetic blob dataset asserting it finds a sane number of clusters, plus
the unknown-name `ValueError` dispatch path.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_blobs

from dscraft.automl import SUPPORTED_CLUSTERERS, build_clusterer
from dscraft.automl.clustering import build_clusterer as build_clusterer_direct


def test_public_api_surface() -> None:
    """`build_clusterer`/`SUPPORTED_CLUSTERERS` are importable from `dscraft.automl`."""
    assert build_clusterer is build_clusterer_direct
    assert set(SUPPORTED_CLUSTERERS) == {"HDBSCAN"}


def test_hdbscan_clustering_smoke() -> None:
    """HDBSCAN finds a sane number of clusters on a well-separated blob dataset."""
    X, _y = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.5,
        random_state=0,
    )

    clusterer = build_clusterer("HDBSCAN", min_cluster_size=10)
    labels = clusterer.fit_predict(X)

    found_clusters = set(labels) - {-1}  # -1 is HDBSCAN's noise label
    assert 1 <= len(found_clusters) <= 5


def test_build_clusterer_unknown_name_raises() -> None:
    """An unrecognized clusterer `name` raises ValueError listing valid names."""
    with pytest.raises(ValueError, match="Unsupported clusterer"):
        build_clusterer("NotARealClusterer")
