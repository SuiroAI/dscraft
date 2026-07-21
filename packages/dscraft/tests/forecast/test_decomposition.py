"""Hermetic tests for dscraft.forecast.decomposition.

No network access required. Builds small synthetic seasonal signals
directly in this file (fixed seed), matching the style of
test_forecast.py/test_backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscraft.forecast import DecompositionResult, ForecastConfig, SUPPORTED_DECOMPOSITION_METHODS, decompose
from dscraft.forecast.decomposition import boxcox_transform, inverse_boxcox_transform


def _dominant_autocorrelation_lag(series: np.ndarray, max_lag: int) -> int:
    """Return the lag (2..max_lag) with the highest autocorrelation.

    Used to verify a decomposed seasonal component's recovered period
    without hardcoding a fragile exact-value comparison against STL's
    internal representation -- the seasonal component of a series with an
    injected period-N signal should autocorrelate most strongly at lag N.
    """
    centered = series - series.mean()

    def autocorr(lag: int) -> float:
        return float(np.corrcoef(centered[:-lag], centered[lag:])[0, 1])

    return max(range(2, max_lag + 1), key=autocorr)


def _make_single_seasonal_panel(
    n_points: int = 140, period: int = 7, seed: int = 7
) -> pd.DataFrame:
    """One series with a single injected seasonal period, trend, and small noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)
    y = 20.0 + 0.05 * t + 4.0 * np.sin(2 * np.pi * t / period) + rng.normal(scale=0.2, size=n_points)
    return pd.DataFrame({"unique_id": "single_seasonal", "ds": dates, "y": y})


def _make_multi_seasonal_panel(
    n_points: int = 400, periods: tuple[int, int] = (7, 30), seed: int = 11
) -> pd.DataFrame:
    """One series with two injected seasonal periods (e.g. weekly + monthly), trend, and small noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)
    short_period, long_period = periods
    y = (
        50.0
        + 0.02 * t
        + 5.0 * np.sin(2 * np.pi * t / short_period)
        + 8.0 * np.sin(2 * np.pi * t / long_period)
        + rng.normal(scale=0.2, size=n_points)
    )
    return pd.DataFrame({"unique_id": "multi_seasonal", "ds": dates, "y": y})


@pytest.fixture(scope="module")
def single_seasonal_panel() -> pd.DataFrame:
    return _make_single_seasonal_panel()


@pytest.fixture(scope="module")
def multi_seasonal_panel() -> pd.DataFrame:
    return _make_multi_seasonal_panel()


def test_supported_decomposition_methods() -> None:
    """SUPPORTED_DECOMPOSITION_METHODS must contain exactly stl and mstl."""
    assert set(SUPPORTED_DECOMPOSITION_METHODS) == {"stl", "mstl"}


def test_decompose_rejects_unsupported_method(single_seasonal_panel: pd.DataFrame) -> None:
    """decompose() must raise ValueError for a method outside SUPPORTED_DECOMPOSITION_METHODS."""
    with pytest.raises(ValueError, match="Unsupported decomposition method"):
        decompose(single_seasonal_panel, method="x11")


def test_decompose_stl_rejects_period_below_two(single_seasonal_panel: pd.DataFrame) -> None:
    """decompose(method='stl') must raise ValueError when period < 2."""
    with pytest.raises(ValueError, match="period must be >= 2"):
        decompose(single_seasonal_panel, method="stl", period=1)


def test_decompose_mstl_rejects_empty_periods(single_seasonal_panel: pd.DataFrame) -> None:
    """decompose(method='mstl') must raise ValueError when periods is empty."""
    with pytest.raises(ValueError, match="periods must not be empty"):
        decompose(single_seasonal_panel, method="mstl", periods=[])


# --- STL: single-series, single seasonal period -----------------------------


def test_stl_decompose_returns_expected_shape_and_columns(single_seasonal_panel: pd.DataFrame) -> None:
    """decompose(method='stl') must return one row per input row with unique_id/ds/trend/seasonal/resid columns."""
    result = decompose(single_seasonal_panel, method="stl", period=7)

    assert isinstance(result, DecompositionResult)
    assert result.method == "stl"
    assert result.boxcox_lambda is None
    assert set(result.components.columns) == {"unique_id", "ds", "trend", "seasonal", "resid"}
    assert len(result.components) == len(single_seasonal_panel)
    assert np.isfinite(result.components["trend"].to_numpy()).all()
    assert np.isfinite(result.components["seasonal"].to_numpy()).all()
    assert np.isfinite(result.components["resid"].to_numpy()).all()


def test_stl_decompose_recovers_injected_seasonal_period(single_seasonal_panel: pd.DataFrame) -> None:
    """The seasonal component's own dominant autocorrelation lag must match the injected period (7), within tolerance."""
    injected_period = 7
    result = decompose(single_seasonal_panel, method="stl", period=injected_period)

    seasonal = result.components["seasonal"].to_numpy()
    recovered_period = _dominant_autocorrelation_lag(seasonal, max_lag=30)
    assert abs(recovered_period - injected_period) <= 1


def test_stl_decompose_uses_config_season_length_by_default(single_seasonal_panel: pd.DataFrame) -> None:
    """When period is not given, decompose(method='stl') must fall back to config.season_length."""
    config = ForecastConfig(season_length=7)
    explicit = decompose(single_seasonal_panel, config, method="stl", period=7)
    defaulted = decompose(single_seasonal_panel, config, method="stl")

    pd.testing.assert_frame_equal(explicit.components, defaulted.components)


def test_stl_decompose_defaults_to_config_default_season_length() -> None:
    """decompose() without an explicit config must use ForecastConfig()'s default season_length (7)."""
    panel = _make_single_seasonal_panel(period=7, seed=3)
    result = decompose(panel, method="stl")
    assert result.method == "stl"
    assert len(result.components) == len(panel)


# --- MSTL: single-series, multiple seasonal periods --------------------------


def test_mstl_decompose_returns_one_seasonal_column_per_period(multi_seasonal_panel: pd.DataFrame) -> None:
    """decompose(method='mstl') must return one seasonal_<period> column per requested period."""
    result = decompose(multi_seasonal_panel, method="mstl", periods=[7, 30])

    assert result.method == "mstl"
    assert set(result.components.columns) == {"unique_id", "ds", "trend", "seasonal_7", "seasonal_30", "resid"}
    assert len(result.components) == len(multi_seasonal_panel)
    assert np.isfinite(result.components["seasonal_7"].to_numpy()).all()
    assert np.isfinite(result.components["seasonal_30"].to_numpy()).all()


def test_mstl_decompose_recovers_both_injected_seasonal_periods(multi_seasonal_panel: pd.DataFrame) -> None:
    """Each MSTL seasonal_<period> column's dominant autocorrelation lag must match its own injected period."""
    result = decompose(multi_seasonal_panel, method="mstl", periods=[7, 30])

    short_recovered = _dominant_autocorrelation_lag(result.components["seasonal_7"].to_numpy(), max_lag=20)
    long_recovered = _dominant_autocorrelation_lag(result.components["seasonal_30"].to_numpy(), max_lag=60)

    assert abs(short_recovered - 7) <= 1
    assert abs(long_recovered - 30) <= 2


def test_mstl_decompose_multi_series(single_seasonal_panel: pd.DataFrame, multi_seasonal_panel: pd.DataFrame) -> None:
    """decompose() must handle multiple series independently, one decomposition per unique_id."""
    combined = pd.concat([single_seasonal_panel, multi_seasonal_panel.rename(columns={})], ignore_index=True)
    # Both panels already have distinct unique_id values ("single_seasonal", "multi_seasonal"),
    # but they have different lengths -- use the shorter panel's period set for both via mstl.
    result = decompose(combined, method="mstl", periods=[7])

    assert set(result.components["unique_id"].unique()) == {"single_seasonal", "multi_seasonal"}
    per_series_counts = result.components.groupby("unique_id").size()
    assert per_series_counts["single_seasonal"] == len(single_seasonal_panel)
    assert per_series_counts["multi_seasonal"] == len(multi_seasonal_panel)


# --- Box-Cox: standalone helpers + decompose(boxcox=True) --------------------


def test_boxcox_transform_round_trip_recovers_original_values() -> None:
    """boxcox_transform() followed by inverse_boxcox_transform() must recover the original values within tolerance."""
    rng = np.random.default_rng(5)
    original = rng.uniform(low=0.5, high=100.0, size=200)

    transformed, lmbda = boxcox_transform(original)
    recovered = inverse_boxcox_transform(transformed, lmbda)

    np.testing.assert_allclose(recovered, original, rtol=1e-8, atol=1e-8)


def test_boxcox_transform_rejects_non_positive_values() -> None:
    """boxcox_transform() must raise ValueError when given a non-positive value (Box-Cox is undefined there)."""
    with pytest.raises(ValueError, match="strictly-positive"):
        boxcox_transform(np.array([1.0, 2.0, 0.0, 3.0]))


def test_boxcox_transform_with_explicit_lambda() -> None:
    """boxcox_transform() with an explicit lmbda must use it as-is (not re-fit) and round-trip correctly."""
    rng = np.random.default_rng(9)
    original = rng.uniform(low=1.0, high=50.0, size=50)

    transformed, lmbda = boxcox_transform(original, lmbda=0.5)
    assert lmbda == 0.5
    recovered = inverse_boxcox_transform(transformed, lmbda)
    np.testing.assert_allclose(recovered, original, rtol=1e-8, atol=1e-8)


def test_decompose_boxcox_true_adds_lambda_column_and_scalar_for_single_series(
    single_seasonal_panel: pd.DataFrame,
) -> None:
    """decompose(boxcox=True) on a single-series input must set the scalar boxcox_lambda and a per-row column."""
    result = decompose(single_seasonal_panel, method="stl", period=7, boxcox=True)

    assert result.boxcox_lambda is not None
    assert "boxcox_lambda" in result.components.columns
    assert (result.components["boxcox_lambda"] == result.boxcox_lambda).all()


def test_decompose_boxcox_true_scalar_is_none_for_multi_series(
    single_seasonal_panel: pd.DataFrame, multi_seasonal_panel: pd.DataFrame
) -> None:
    """decompose(boxcox=True) on a multi-series input must leave the scalar boxcox_lambda None (ambiguous),
    while still carrying each series' own fitted lambda in the per-row boxcox_lambda column."""
    combined = pd.concat([single_seasonal_panel, multi_seasonal_panel], ignore_index=True)
    result = decompose(combined, method="mstl", periods=[7], boxcox=True)

    assert result.boxcox_lambda is None
    assert "boxcox_lambda" in result.components.columns
    per_series_lambdas = result.components.groupby("unique_id")["boxcox_lambda"].nunique()
    # Each series' boxcox_lambda column value must be constant within that series.
    assert (per_series_lambdas == 1).all()


def test_decompose_boxcox_true_rejects_non_positive_series() -> None:
    """decompose(boxcox=True) must raise ValueError when a series contains a non-positive value."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    panel = pd.DataFrame({"unique_id": "has_zero", "ds": dates, "y": np.concatenate([[0.0], np.arange(1.0, 30.0)])})

    with pytest.raises(ValueError, match="non-positive"):
        decompose(panel, method="stl", period=7, boxcox=True)
