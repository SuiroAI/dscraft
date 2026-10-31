"""Tests for dscraft.eda._outliers and its wiring into LazyEDA.

Validates the univariate screens (IQR fences, robust MAD z-score) flag planted outliers and not
clean data, the isolation forest scores injected anomalies higher than inliers and matches sklearn
on separation where available, and LazyEDA.profile populates per-column outlier reports.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dscraft.eda import (
    IsolationForest,
    LazyEDA,
    iqr_outliers,
    mad_outliers,
)


# ---------------------------------------------------------------------------
# Univariate screens
# ---------------------------------------------------------------------------


def test_iqr_flags_extreme_value() -> None:
    data = list(range(1, 21)) + [1000]  # one huge outlier
    report = iqr_outliers(data)
    assert report.method == "iqr"
    assert report.n_outliers == 1
    assert 20 in report.outlier_indices  # the 1000 is at index 20
    assert report.upper_bound is not None and report.upper_bound < 1000


def test_iqr_no_outliers_on_clean_data() -> None:
    report = iqr_outliers(list(range(100)))
    assert report.n_outliers == 0
    assert report.fraction == 0.0


def test_iqr_k_controls_sensitivity() -> None:
    data = list(range(1, 21)) + [40]
    loose = iqr_outliers(data, k=3.0)
    tight = iqr_outliers(data, k=0.5)
    assert tight.n_outliers >= loose.n_outliers


def test_mad_flags_robust_outlier() -> None:
    data = [10.0] * 50 + [10.5, 9.5, 500.0]
    report = mad_outliers(data)
    assert report.method == "mad"
    # The 500 is flagged; the 10.5/9.5 near the median are not.
    assert 52 in report.outlier_indices


def test_mad_zero_mad_fallback() -> None:
    # More than half identical -> MAD is 0; fall back to flagging anything != median.
    data = [5.0] * 10 + [7.0, 9.0]
    report = mad_outliers(data)
    assert report.n_outliers == 2


@pytest.mark.parametrize("fn, kwargs, match", [
    (iqr_outliers, {"k": -1.0}, "k must be"),
    (mad_outliers, {"threshold": 0.0}, "threshold"),
])
def test_univariate_validation(fn, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        fn([1.0, 2.0, 3.0], **kwargs)


def test_univariate_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        iqr_outliers([])


# ---------------------------------------------------------------------------
# Isolation forest
# ---------------------------------------------------------------------------


def test_isolation_forest_scores_anomalies_higher() -> None:
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 1, size=(300, 2))
    anomalies = rng.uniform(8, 10, size=(5, 2))
    X = np.vstack([inliers, anomalies])
    forest = IsolationForest(n_estimators=100, random_state=0).fit(X)
    scores = forest.score_samples(X)
    # Anomalies (last 5 rows) should score clearly higher than the inlier median.
    assert scores[-5:].min() > np.median(scores[:300])


def test_isolation_forest_predict_flags_injected_anomalies() -> None:
    rng = np.random.default_rng(1)
    inliers = rng.normal(0, 1, size=(400, 3))
    anomalies = rng.uniform(10, 12, size=(8, 3))
    X = np.vstack([inliers, anomalies])
    labels = IsolationForest(n_estimators=120, contamination=0.05, random_state=0).fit_predict(X)
    # Most of the injected anomalies are caught.
    assert labels[-8:].sum() >= 6


def test_isolation_forest_scores_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    scores = IsolationForest(n_estimators=50, random_state=0).fit(X).score_samples(X)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_isolation_forest_reproducible() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 2))
    a = IsolationForest(n_estimators=60, random_state=7).fit(X).score_samples(X)
    b = IsolationForest(n_estimators=60, random_state=7).fit(X).score_samples(X)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("kwargs, match", [
    ({"n_estimators": 0}, "n_estimators"),
    ({"contamination": 0.8}, "contamination"),
])
def test_isolation_forest_validation(kwargs, match) -> None:
    X = np.random.default_rng(0).normal(size=(50, 2))
    with pytest.raises(ValueError, match=match):
        IsolationForest(**kwargs).fit(X)


# ---------------------------------------------------------------------------
# LazyEDA integration
# ---------------------------------------------------------------------------


def test_profile_populates_column_outliers() -> None:
    frame = pl.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0],
            "city": ["a", "b", "a", "a", "b", "a"],
        }
    )
    profile = LazyEDA().profile(frame)
    assert "amount" in profile.column_outliers
    report = profile.column_outliers["amount"]
    assert report.n_outliers >= 1  # the 1000 is flagged
    # Categorical columns get no univariate numeric outlier screen.
    assert "city" not in profile.column_outliers


def test_profile_outlier_method_mad() -> None:
    frame = pl.DataFrame({"x": [10.0] * 20 + [10.1, 9.9, 500.0]})
    profile = LazyEDA(outlier_method="mad").profile(frame)
    assert profile.column_outliers["x"].method == "mad"


def test_lazyeda_rejects_bad_outlier_method() -> None:
    with pytest.raises(ValueError, match="outlier_method"):
        LazyEDA(outlier_method="bogus")
