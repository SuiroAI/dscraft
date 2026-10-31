"""Tests for dscraft.eda._temporal and its wiring into LazyEDA.

Validates temporal-coverage/cadence profiling (range, median gap, regularity, gap detection,
monotonicity, duplicates), the first-party autocorrelation estimator (r_0==1, detects periodicity
and AR(1) decay), and that LazyEDA.profile now populates temporal descriptors for date columns.
"""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from dscraft.eda import LazyEDA, TemporalDescriptor, autocorrelation
from dscraft.eda._temporal import temporal_descriptor


_DAY = 86400.0  # seconds per day


# ---------------------------------------------------------------------------
# Temporal descriptor
# ---------------------------------------------------------------------------


def test_regular_daily_series() -> None:
    epochs = [i * _DAY for i in range(10)]  # 10 evenly-spaced days
    d = temporal_descriptor(epochs)
    assert d.count == 10 and d.distinct == 10 and d.n_duplicates == 0
    assert d.is_sorted is True
    assert d.span_seconds == pytest.approx(9 * _DAY)
    assert d.median_gap_seconds == pytest.approx(_DAY)
    assert d.regularity == pytest.approx(1.0)  # perfectly regular
    assert d.n_large_gaps == 0


def test_detects_large_gap() -> None:
    # Daily for days 0..6, then jump to day 37 (a 31-day gap), then daily.
    epochs = [i * _DAY for i in range(7)] + [37 * _DAY, 38 * _DAY, 39 * _DAY]
    d = temporal_descriptor(epochs)
    assert d.n_large_gaps == 1
    assert d.large_gap_total_seconds == pytest.approx(31 * _DAY)
    assert d.max_gap_seconds == pytest.approx(31 * _DAY)


def test_duplicates_and_unsorted() -> None:
    epochs = [5 * _DAY, 1 * _DAY, 1 * _DAY, 3 * _DAY]
    d = temporal_descriptor(epochs)
    assert d.n_duplicates == 1  # two identical timestamps -> one duplicate
    assert d.is_sorted is False
    assert d.distinct == 3


def test_single_timestamp_degenerate() -> None:
    d = temporal_descriptor([100.0])
    assert d.count == 1 and d.distinct == 1
    assert d.median_gap_seconds is None and d.regularity is None
    assert d.span_seconds == 0.0


def test_irregular_series_low_regularity() -> None:
    epochs = [0.0, 1 * _DAY, 1.1 * _DAY, 10 * _DAY, 10.2 * _DAY]
    d = temporal_descriptor(epochs)
    assert d.regularity is not None and d.regularity < 1.0


@pytest.mark.parametrize("bad", [[], None])
def test_temporal_descriptor_empty_raises(bad) -> None:
    with pytest.raises(ValueError):
        temporal_descriptor([] if bad is None else bad)


def test_temporal_descriptor_bad_factor() -> None:
    with pytest.raises(ValueError, match="large_gap_factor"):
        temporal_descriptor([1.0, 2.0], large_gap_factor=0.0)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def test_acf_lag0_is_one() -> None:
    rng = np.random.default_rng(0)
    acf = autocorrelation(rng.normal(size=200), max_lag=10)
    assert acf[0] == pytest.approx(1.0)
    assert acf.shape == (11,)


def test_acf_detects_periodicity() -> None:
    # A clean period-12 sine -> strong positive autocorrelation at lag 12.
    t = np.arange(240)
    series = np.sin(2 * np.pi * t / 12)
    acf = autocorrelation(series, max_lag=24)
    assert acf[12] > 0.8
    assert acf[6] < 0  # half a period out of phase -> negative


def test_acf_ar1_decays() -> None:
    # An AR(1) with positive phi has a monotonically-decaying positive ACF.
    rng = np.random.default_rng(1)
    n = 2000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    acf = autocorrelation(x, max_lag=5)
    assert acf[1] > acf[2] > acf[3] > 0


def test_acf_caps_lag_at_series_length() -> None:
    acf = autocorrelation([1.0, 2.0, 3.0], max_lag=100)
    assert acf.shape == (3,)  # min(max_lag, n-1)+1


@pytest.mark.parametrize("values, match", [
    ([5.0], "at least 2"),
    ([3.0, 3.0, 3.0], "constant"),
])
def test_acf_validation(values, match) -> None:
    with pytest.raises(ValueError, match=match):
        autocorrelation(values)


def test_acf_bad_max_lag() -> None:
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation([1.0, 2.0, 3.0], max_lag=0)


# ---------------------------------------------------------------------------
# LazyEDA integration
# ---------------------------------------------------------------------------


def test_profile_populates_temporal_descriptor() -> None:
    base = datetime.date(2026, 1, 1)
    dates = [base + datetime.timedelta(days=i) for i in range(10)]
    frame = pl.DataFrame({"event_date": dates, "value": list(range(10))})
    profile = LazyEDA().profile(frame)
    assert "event_date" in profile.temporal_descriptors
    d = profile.temporal_descriptors["event_date"]
    assert isinstance(d, TemporalDescriptor)
    assert d.count == 10
    assert d.median_gap_seconds == pytest.approx(_DAY)
    # The numeric column is unaffected (still gets its numeric descriptor, not a temporal one).
    assert "value" not in profile.temporal_descriptors
    assert "value" in profile.numeric_descriptors


def test_profile_datetime_column() -> None:
    base = datetime.datetime(2026, 1, 1, 0, 0, 0)
    stamps = [base + datetime.timedelta(hours=i) for i in range(24)]
    frame = pl.DataFrame({"ts": stamps})
    profile = LazyEDA().profile(frame)
    d = profile.temporal_descriptors["ts"]
    assert d.median_gap_seconds == pytest.approx(3600.0)  # hourly
