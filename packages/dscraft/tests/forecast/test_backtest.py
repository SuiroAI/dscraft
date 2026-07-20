"""Hermetic tests for dscraft.forecast.backtest.

No network access required. Reuses the same synthetic-panel construction
pattern as test_forecast.py (sine-wave-plus-trend, fixed seed, 2 series).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscraft.forecast import (
    BacktestAlignmentError,
    BacktestReport,
    ForecastConfig,
    backtest,
)


def _make_synthetic_panel(n_points: int = 120, seed: int = 42) -> pd.DataFrame:
    """Build the same two-series sine-wave-plus-trend panel used in test_forecast.py.

    "series_a" and "series_b" each get a distinct trend slope, seasonal
    amplitude/phase, and level, plus small Gaussian noise, over `n_points`
    daily observations with a 7-day season -- reused here (rather than
    imported) so this test module stays self-contained.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)

    frames = []
    series_params = {
        "series_a": {"trend": 0.05, "amplitude": 5.0, "phase": 0.0, "level": 50.0},
        "series_b": {"trend": -0.03, "amplitude": 8.0, "phase": 1.5, "level": 100.0},
    }
    for unique_id, params in series_params.items():
        seasonal = params["amplitude"] * np.sin(2 * np.pi * t / 7 + params["phase"])
        trend = params["trend"] * t
        noise = rng.normal(scale=0.5, size=n_points)
        y = params["level"] + trend + seasonal + noise
        frames.append(pd.DataFrame({"unique_id": unique_id, "ds": dates, "y": y}))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    """Module-scoped synthetic two-series panel, built once and shared across tests."""
    return _make_synthetic_panel()


def test_backtest_returns_metric_per_series_and_model(synthetic_panel: pd.DataFrame) -> None:
    """backtest() must return one SeriesMetric per (series, model) pair with finite, non-negative errors."""
    config = ForecastConfig(horizon=14, freq="D", season_length=7, models=("AutoARIMA",))
    report = backtest(synthetic_panel, config, test_size=14)

    assert isinstance(report, BacktestReport)
    # 2 series * 1 model = 2 SeriesMetric rows.
    assert len(report.metrics) == 2
    series_ids = {m.unique_id for m in report.metrics}
    assert series_ids == {"series_a", "series_b"}
    for metric in report.metrics:
        assert metric.model == "AutoARIMA"
        assert metric.n_points == 14
        assert np.isfinite(metric.mae)
        assert np.isfinite(metric.rmse)
        assert metric.mae >= 0
        assert metric.rmse >= 0


def test_backtest_mean_metrics_below_sanity_threshold(synthetic_panel: pd.DataFrame) -> None:
    """Sanity threshold justification: the synthetic series has level ~50-100,
    seasonal amplitude 5-8, and noise stddev 0.5. A reasonable classical
    seasonal model forecasting 14 days ahead should track the deterministic
    trend+seasonal signal to within a small fraction of the series level --
    an MAE below 15 (comfortably above the noise floor and expected seasonal
    forecast error, but far below the series' own level/amplitude scale)
    is a generous, non-tautological bar that would fail if the forecast
    pipeline were badly broken (e.g. producing near-random or huge/diverging
    values), while still tolerating normal classical-model forecast error on
    a short synthetic series.
    """
    config = ForecastConfig(horizon=14, freq="D", season_length=7, models=("AutoARIMA", "AutoETS"))
    report = backtest(synthetic_panel, config, test_size=14)

    assert report.mean_mae() < 15.0
    assert report.mean_rmse() < 15.0
    assert report.mean_mae(model="AutoARIMA") < 15.0
    assert report.mean_mae(model="AutoETS") < 15.0


def test_backtest_to_frame_shape(synthetic_panel: pd.DataFrame) -> None:
    """BacktestReport.to_frame() must expose the documented columns with one row per (series, model)."""
    config = ForecastConfig(horizon=10, freq="D", season_length=7, models=("AutoARIMA", "AutoETS"))
    report = backtest(synthetic_panel, config, test_size=10)

    frame = report.to_frame()
    assert list(frame.columns) == [
        "unique_id",
        "model",
        "mae",
        "rmse",
        "n_points",
        "expected_points",
    ]
    # 2 series * 2 models = 4 rows.
    assert len(frame) == 4
    assert (frame["n_points"] == frame["expected_points"]).all()


def test_backtest_raises_when_series_too_short() -> None:
    """backtest() must raise ValueError when a series has fewer observations than test_size + 1."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tiny = pd.DataFrame({"unique_id": "only_series", "ds": dates, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    config = ForecastConfig(models=("AutoARIMA",))
    with pytest.raises(ValueError, match="not enough"):
        backtest(tiny, config, test_size=10)


def test_backtest_mean_mae_raises_for_unknown_model(synthetic_panel: pd.DataFrame) -> None:
    """BacktestReport.mean_mae() must raise ValueError when filtered to a model absent from the report."""
    config = ForecastConfig(horizon=7, season_length=7, models=("AutoARIMA",))
    report = backtest(synthetic_panel, config, test_size=7)
    with pytest.raises(ValueError, match="No backtest metrics"):
        report.mean_mae(model="NotAModel")


def test_backtest_raises_alignment_error_when_series_has_zero_overlap() -> None:
    """Regression test for the silent-drop bug: a series whose actual
    observation dates are NOT on the configured "D" frequency (a genuine
    real-world gap -- e.g. missing observations or an irregular collection
    cadence) ends up, after holding out the last ``test_size`` rows by
    position, with held-out ``ds`` values that never coincide with the
    model's forecasted dates (which are always ``last_train_ds + 1..h`` at
    the configured frequency). Before the fix, ``forecasts.merge(test_df,
    how="inner")`` would silently drop this series from the report with no
    signal at all. Now it must raise :class:`BacktestAlignmentError` naming
    the affected series, rather than returning a plausible-looking report
    missing that series.
    """
    n_points = 30
    test_size = 5

    regular_dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)
    y = 50.0 + 0.1 * t + 3.0 * np.sin(2 * np.pi * t / 7)

    # gappy_series is observed every 6 (nominal) days rather than daily, so
    # its held-out ds values sit at +6, +12, +18, +24, +30 days past the
    # last training ds -- while the "D"-frequency forecast only ever
    # produces +1..+5. Zero overlap by construction.
    gappy_dates = pd.date_range("2024-01-01", periods=n_points, freq="6D")

    panel = pd.concat(
        [
            pd.DataFrame({"unique_id": "regular_series", "ds": regular_dates, "y": y}),
            pd.DataFrame({"unique_id": "gappy_series", "ds": gappy_dates, "y": y}),
        ],
        ignore_index=True,
    )

    config = ForecastConfig(horizon=test_size, freq="D", season_length=7, models=("AutoARIMA",))
    with pytest.raises(BacktestAlignmentError, match="gappy_series"):
        backtest(panel, config, test_size=test_size)


def test_backtest_warns_and_flags_partial_overlap() -> None:
    """A series with SOME but not all held-out dates aligning with the
    forecast must be visibly flagged (warning + n_points < expected_points),
    not silently scored over a shrunken window as if nothing were wrong.
    """
    n_points = 30
    test_size = 6

    regular_dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)
    y = 50.0 + 0.1 * t + 3.0 * np.sin(2 * np.pi * t / 7)

    # half_aligned_series is observed every 2 (nominal) days, so its
    # held-out ds values sit at +2, +4, +6, +8, +10, +12 past the last
    # training ds. The "D"-frequency forecast produces +1..+6, which
    # overlaps at +2, +4, +6 -- exactly half of the held-out window.
    sparse_dates = pd.date_range("2024-01-01", periods=n_points, freq="2D")

    panel = pd.concat(
        [
            pd.DataFrame({"unique_id": "regular_series", "ds": regular_dates, "y": y}),
            pd.DataFrame({"unique_id": "half_aligned_series", "ds": sparse_dates, "y": y}),
        ],
        ignore_index=True,
    )

    config = ForecastConfig(horizon=test_size, freq="D", season_length=7, models=("AutoARIMA",))
    with pytest.warns(UserWarning, match="half_aligned_series"):
        report = backtest(panel, config, test_size=test_size)

    by_id = {m.unique_id: m for m in report.metrics}
    assert set(by_id) == {"regular_series", "half_aligned_series"}

    regular = by_id["regular_series"]
    assert regular.n_points == regular.expected_points == test_size

    partial = by_id["half_aligned_series"]
    assert partial.expected_points == test_size
    assert partial.n_points == 3
    assert partial.n_points < partial.expected_points
