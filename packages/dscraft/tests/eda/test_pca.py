"""Tests for dscraft.eda._pca and its wiring into LazyEDA.

Validates first-party PCA: explained-variance ratios sum to ~1 and are descending, a
low-rank dataset concentrates variance in few components, transform reduces dimensionality and
round-trips the projection, standardize vs. covariance modes behave, and LazyEDA.profile surfaces
a PCAResult over numeric columns (and None when there aren't enough).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dscraft.eda import PCA, PCAResult, LazyEDA


# ---------------------------------------------------------------------------
# PCA core
# ---------------------------------------------------------------------------


def test_explained_variance_ratio_sums_and_orders() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    pca = PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    assert ratio.sum() == pytest.approx(1.0, abs=1e-9)
    # Components are ordered by descending explained variance.
    assert np.all(np.diff(ratio) <= 1e-12)


def test_low_rank_data_concentrates_variance() -> None:
    # Build 4-D data that really lives on a 1-D line (+ tiny noise).
    rng = np.random.default_rng(1)
    t = rng.normal(size=(300, 1))
    direction = np.array([[1.0, 2.0, -1.0, 0.5]])
    X = t @ direction + rng.normal(scale=1e-3, size=(300, 4))
    pca = PCA(standardize=False).fit(X)
    # Almost all variance in the first component.
    assert pca.explained_variance_ratio_[0] > 0.99


def test_transform_reduces_dimensionality() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 6))
    pca = PCA(n_components=3).fit(X)
    scores = pca.transform(X)
    assert scores.shape == (100, 3)
    assert pca.components_.shape == (3, 6)


def test_fit_transform_matches_fit_then_transform() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    a = PCA(n_components=2).fit_transform(X)
    b = PCA(n_components=2).fit(X).transform(X)
    np.testing.assert_allclose(a, b)


def test_standardize_handles_different_scales() -> None:
    rng = np.random.default_rng(4)
    # One feature has a huge scale; standardize should stop it dominating.
    X = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 1000, 200)])
    standardized = PCA(standardize=True).fit(X)
    covariance = PCA(standardize=False).fit(X)
    # Under covariance-PCA the huge-scale feature captures ~all variance; under correlation-PCA
    # the two components are far more balanced.
    assert covariance.explained_variance_ratio_[0] > 0.99
    assert standardized.explained_variance_ratio_[0] < 0.9


def test_constant_feature_does_not_crash_standardize() -> None:
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    pca = PCA(standardize=True).fit(X)  # constant col left unscaled, no divide-by-zero
    assert pca.components_.shape[1] == 2


def test_pca_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        PCA().fit(np.array([[1.0, 2.0], [np.nan, 3.0]]))


def test_summary_readout() -> None:
    rng = np.random.default_rng(5)
    t = rng.normal(size=(200, 1))
    X = t @ np.array([[1.0, 1.0, 1.0]]) + rng.normal(scale=1e-2, size=(200, 3))
    result = PCA(standardize=False).fit(X).summary(["a", "b", "c"])
    assert isinstance(result, PCAResult)
    assert result.feature_names == ["a", "b", "c"]
    assert result.cumulative_variance_ratio[-1] == pytest.approx(1.0, abs=1e-6)
    assert result.n_components_for_90pct == 1  # effectively 1-D


# ---------------------------------------------------------------------------
# LazyEDA integration
# ---------------------------------------------------------------------------


def test_profile_populates_pca_over_numeric_columns() -> None:
    rng = np.random.default_rng(6)
    frame = pl.DataFrame(
        {
            "x1": rng.normal(size=100),
            "x2": rng.normal(size=100),
            "x3": rng.normal(size=100),
            "label": ["a", "b"] * 50,  # non-numeric, excluded from PCA
        }
    )
    profile = LazyEDA().profile(frame)
    assert profile.pca is not None
    assert profile.pca.feature_names == ["x1", "x2", "x3"]  # label excluded
    assert profile.pca.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-9)


def test_profile_pca_none_with_single_numeric_column() -> None:
    frame = pl.DataFrame({"only": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
    profile = LazyEDA().profile(frame)
    assert profile.pca is None  # need >= 2 numeric columns


def test_profile_pca_skips_null_rows() -> None:
    frame = pl.DataFrame(
        {"x1": [1.0, 2.0, None, 4.0, 5.0], "x2": [2.0, 4.0, 6.0, None, 10.0]}
    )
    profile = LazyEDA().profile(frame)
    # 3 complete rows remain (rows 0, 1, 4) -> PCA runs.
    assert profile.pca is not None
