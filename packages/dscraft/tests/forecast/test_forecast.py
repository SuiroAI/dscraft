"""Hermetic tests for dscraft.forecast.forecast.

No network access required. Builds a small synthetic multi-series dataset
(sine-wave-plus-trend, fixed seed, >=2 distinct series IDs) directly in this
file, fits AutoARIMA over it, and asserts basic sanity on the resulting
forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscraft.forecast import ForecastConfig, forecast, prepare_frame, validate_input
from dscraft.forecast.forecast import SUPPORTED_MODELS


def _make_synthetic_panel(n_points: int = 120, seed: int = 42) -> pd.DataFrame:
    """Two seasonal-plus-trend series with a fixed seed, for hermetic testing.

    Series "series_a" and "series_b" each have a distinct trend slope and
    seasonal amplitude/phase, plus small Gaussian noise, over `n_points`
    daily observations with a 7-day season.
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


@pytest.fixture(scope="module")
def arrow_backed_panel(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    """The same panel, converted to Tier-1 Arrow-backed pandas (ArrowDtype)."""
    return synthetic_panel.convert_dtypes(dtype_backend="pyarrow")


def test_supported_models_are_classical_only() -> None:
    """SUPPORTED_MODELS must contain exactly the classical statsforecast models in scope for this pass.

    Per issue #20's gap analysis, this now covers the full classical
    catalog statsforecast ships (autofits, simple baselines, Croston
    family) -- but must NOT include anything from the tree-based ML branch
    (MLForecast/LightGBM/XGBoost) or zero-shot TSFMs, which remain
    explicitly out of scope.
    """
    assert set(SUPPORTED_MODELS) == {
        "AutoARIMA",
        "AutoETS",
        "AutoCES",
        "AutoTheta",
        "SeasonalNaive",
        "Naive",
        "RandomWalkWithDrift",
        "HistoricAverage",
        "WindowAverage",
        "SeasonalWindowAverage",
        "CrostonClassic",
        "CrostonOptimized",
        "CrostonSBA",
    }


def test_forecast_config_rejects_unsupported_model() -> None:
    """ForecastConfig must raise ValueError for a model name outside SUPPORTED_MODELS."""
    with pytest.raises(ValueError, match="Unsupported model"):
        ForecastConfig(models=("LightGBM",))


def test_forecast_config_rejects_empty_models() -> None:
    """ForecastConfig must raise ValueError when models is an empty tuple."""
    with pytest.raises(ValueError, match="must not be empty"):
        ForecastConfig(models=())


def test_validate_input_reports_arrow_backed_columns(arrow_backed_panel: pd.DataFrame) -> None:
    """validate_input() on a fully ArrowDtype-backed frame reports no warnings and is_fully_arrow_backed=True."""
    report = validate_input(arrow_backed_panel)
    assert report.input_kind == "pandas"
    assert report.n_series == 2
    assert report.is_fully_arrow_backed is True
    assert report.warnings == []


def test_validate_input_warns_on_non_arrow_backed(synthetic_panel: pd.DataFrame) -> None:
    """validate_input() on a plain numpy-backed pandas frame flags it as not Arrow-backed via a warning."""
    report = validate_input(synthetic_panel)
    assert report.is_fully_arrow_backed is False
    assert any("Arrow-backed" in w for w in report.warnings)


def test_validate_input_missing_column_raises(synthetic_panel: pd.DataFrame) -> None:
    """validate_input() must raise ValueError when a required id/time/value column is missing."""
    broken = synthetic_panel.drop(columns=["y"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_input(broken)


def test_validate_input_rejects_non_dataframe() -> None:
    """validate_input() must raise TypeError for input that is neither a pandas nor a Polars DataFrame."""
    with pytest.raises(TypeError):
        validate_input([1, 2, 3])


def test_prepare_frame_produces_expected_schema(arrow_backed_panel: pd.DataFrame) -> None:
    """prepare_frame() must rename/coerce input into statsforecast's unique_id/ds/y schema with correct dtypes."""
    prepared = prepare_frame(arrow_backed_panel)
    assert list(prepared.columns) == ["unique_id", "ds", "y"]
    assert prepared["ds"].dtype.kind == "M"  # datetime64
    assert prepared["y"].dtype == np.float64
    assert prepared["unique_id"].nunique() == 2
    assert np.isfinite(prepared["y"].to_numpy()).all()


def test_prepare_frame_rejects_nan_values(synthetic_panel: pd.DataFrame) -> None:
    """prepare_frame() must raise ValueError when the value column contains a NaN (no silent imputation)."""
    broken = synthetic_panel.copy()
    broken.loc[0, "y"] = float("nan")
    with pytest.raises(ValueError, match="NaN/inf"):
        prepare_frame(broken)


def test_forecast_autoarima_shape_and_finiteness(arrow_backed_panel: pd.DataFrame) -> None:
    """forecast() with a single model must return exactly horizon rows per series, all finite."""
    config = ForecastConfig(horizon=7, freq="D", season_length=7, models=("AutoARIMA",))
    result = forecast(arrow_backed_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA"}
    # 2 series * 7-step horizon = 14 forecast rows.
    assert len(result) == 2 * config.horizon
    for unique_id in ("series_a", "series_b"):
        subset = result[result["unique_id"] == unique_id]
        assert len(subset) == config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()


def test_forecast_multiple_models(arrow_backed_panel: pd.DataFrame) -> None:
    """forecast() with multiple models must produce one finite forecast column per fitted model."""
    config = ForecastConfig(horizon=5, freq="D", season_length=7, models=("AutoARIMA", "AutoETS"))
    result = forecast(arrow_backed_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA", "AutoETS"}
    assert len(result) == 2 * config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
    assert np.isfinite(result["AutoETS"].to_numpy()).all()


@pytest.mark.parametrize(
    "model_name",
    [
        "AutoCES",
        "AutoTheta",
        "SeasonalNaive",
        "Naive",
        "RandomWalkWithDrift",
        "HistoricAverage",
        "WindowAverage",
        "SeasonalWindowAverage",
        "CrostonClassic",
        "CrostonOptimized",
        "CrostonSBA",
    ],
)
def test_forecast_each_newly_added_model_fits_and_forecasts(
    arrow_backed_panel: pd.DataFrame, model_name: str
) -> None:
    """Each model added to SUPPORTED_MODELS in issue #20 must actually fit
    and forecast on real (synthetic) data, not just be importable -- proving
    build_statsforecast()'s per-model _MODEL_FACTORIES entry constructs a
    usable model instance for every name in SUPPORTED_MODELS.
    """
    config = ForecastConfig(horizon=7, freq="D", season_length=7, models=(model_name,))
    result = forecast(arrow_backed_panel, config)

    assert set(result.columns) == {"unique_id", "ds", model_name}
    assert len(result) == 2 * config.horizon
    assert np.isfinite(result[model_name].to_numpy()).all()


def test_forecast_accepts_plain_pandas_without_arrow_dtype(synthetic_panel: pd.DataFrame) -> None:
    """Non-Arrow-backed pandas input is still usable -- Tier-1 is a
    convention we validate/report on, not a hard gate (see README)."""
    config = ForecastConfig(horizon=5, models=("AutoARIMA",))
    result = forecast(synthetic_panel, config)
    assert len(result) == 2 * config.horizon


def test_forecast_accepts_custom_column_names(synthetic_panel: pd.DataFrame) -> None:
    """forecast() must work when the caller's id/time/value columns are named via ForecastConfig, not the defaults."""
    renamed = synthetic_panel.rename(columns={"unique_id": "series_id", "ds": "timestamp", "y": "value"})
    config = ForecastConfig(
        id_col="series_id", time_col="timestamp", value_col="value", horizon=5, models=("AutoARIMA",)
    )
    result = forecast(renamed, config)
    assert len(result) == 2 * config.horizon


def test_forecast_accepts_polars_input(synthetic_panel: pd.DataFrame) -> None:
    """validate_input() and forecast() must accept a Polars DataFrame directly, routed via from_polars_zero_copy."""
    pl = pytest.importorskip("polars")
    polars_panel = pl.from_pandas(synthetic_panel)

    report = validate_input(polars_panel)
    assert report.input_kind == "polars"
    assert report.n_series == 2

    config = ForecastConfig(horizon=5, models=("AutoARIMA",))
    result = forecast(polars_panel, config)
    assert len(result) == 2 * config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
