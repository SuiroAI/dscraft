"""Runnable end-to-end demo of benchcraft_lazyforecast.

Section 1 generates a synthetic seasonal (sine-wave-plus-trend) multi-series
time series dataset, forecasts it with AutoARIMA + AutoETS via
`benchcraft_lazyforecast.forecast`, backtests it with
`benchcraft_lazyforecast.backtest`, and prints the resulting forecast and
backtest error metrics.

Section 2 runs the exact same public API (forecast()/backtest(), same
ForecastConfig shape) against a **real** dataset -- Mauna Loa atmospheric
CO2 concentration, resampled to monthly means -- bundled inside the
dev-only `statsmodels` dependency (see
tests/test_real_dataset_validation.py for the full rationale on dataset
choice and why loading it makes no network calls). This is the same
validation performed in that test module; this script exists to give a
human-readable side-by-side comparison of synthetic vs. real backtest error,
not to duplicate the test's logic.

This script only imports and calls the real package API -- per CLAUDE.md's
"no net-new scripts" rule, it does not reimplement any forecasting or
backtesting logic inline.

Requires the `dev` extra (for `statsmodels`) to run Section 2:
    pip install -e "packages/lazyforecast[dev]"

Run with:
    python packages/lazyforecast/examples/forecast_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchcraft_lazyforecast import ForecastConfig, backtest, forecast, validate_input


def build_synthetic_seasonal_panel(n_points: int = 150, seed: int = 7) -> pd.DataFrame:
    """Two seasonal-plus-trend series with a fixed seed -- purely a demo
    fixture, not a general-purpose synthetic-data generator for the package.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_points, freq="D")
    t = np.arange(n_points)

    series_params = {
        "store_1_sales": {"trend": 0.08, "amplitude": 12.0, "phase": 0.0, "level": 200.0},
        "store_2_sales": {"trend": -0.04, "amplitude": 6.0, "phase": 2.0, "level": 120.0},
    }
    frames = []
    for unique_id, params in series_params.items():
        seasonal = params["amplitude"] * np.sin(2 * np.pi * t / 7 + params["phase"])
        trend = params["trend"] * t
        noise = rng.normal(scale=1.0, size=n_points)
        y = params["level"] + trend + seasonal + noise
        frames.append(pd.DataFrame({"unique_id": unique_id, "ds": dates, "y": y}))

    panel = pd.concat(frames, ignore_index=True)
    # Convert to Tier-1 Arrow-backed pandas (pandas 2.x ArrowDtype) to
    # exercise the intended Tier-1 input path (architecture doc §2.1).
    return panel.convert_dtypes(dtype_backend="pyarrow")


def build_real_co2_panel() -> pd.DataFrame:
    """Real Mauna Loa atmospheric CO2 concentration, resampled to monthly
    means, reshaped into the unique_id/ds/y schema benchcraft_lazyforecast
    expects. Loads from a CSV file bundled inside the installed
    `statsmodels` package -- no network access. See
    tests/test_real_dataset_validation.py for the full dataset-choice
    rationale (real annual seasonality + long-term trend, a good real-world
    stress test for the seasonal AutoARIMA/AutoETS path).
    """
    from statsmodels.datasets import co2

    raw = co2.load_pandas().data["co2"]
    # The raw weekly series has scattered missing weeks (real-world
    # equipment downtime); prepare_frame() deliberately rejects NaN/inf
    # rather than imputing (see forecast.py), so we interpolate here, in
    # this caller-side data-prep step, not inside the package.
    monthly = raw.resample("MS").mean().interpolate()
    return pd.DataFrame({"unique_id": "co2_monthly", "ds": monthly.index, "y": monthly.to_numpy()})


def main() -> None:
    """Run the full synthetic-then-real demo: validate, forecast, backtest, and print a side-by-side summary.

    Section 1 runs entirely on the synthetic seasonal panel. Section 2 repeats
    the same sequence against the real co2 dataset and is skipped (with a
    message) if the optional `dev` extra (`statsmodels`) isn't installed.
    """
    panel = build_synthetic_seasonal_panel()

    print("=== Section 1: synthetic seasonal panel ===")
    print("=== Tier-1 input validation ===")
    report = validate_input(panel)
    print(f"input_kind={report.input_kind} n_rows={report.n_rows} n_series={report.n_series}")
    print(f"is_fully_arrow_backed={report.is_fully_arrow_backed}")
    if report.warnings:
        for w in report.warnings:
            print(f"warning: {w}")

    config = ForecastConfig(
        horizon=14,
        freq="D",
        season_length=7,
        models=("AutoARIMA", "AutoETS"),
    )

    print("\n=== Forecast (next 14 days) ===")
    forecasts = forecast(panel, config)
    print(forecasts.head(10).to_string(index=False))

    print("\n=== Backtest (holding out the last 14 days per series) ===")
    synthetic_result = backtest(panel, config, test_size=14)
    print(synthetic_result.to_frame().to_string(index=False))

    print("\n=== Section 1 summary (synthetic) ===")
    for model_name in config.models:
        print(
            f"{model_name}: mean MAE = {synthetic_result.mean_mae(model_name):.3f}, "
            f"mean RMSE = {synthetic_result.mean_rmse(model_name):.3f}"
        )
    print(
        f"Overall: mean MAE = {synthetic_result.mean_mae():.3f}, "
        f"mean RMSE = {synthetic_result.mean_rmse():.3f}"
    )

    print("\n\n=== Section 2: real dataset (statsmodels co2, monthly) ===")
    try:
        co2_panel = build_real_co2_panel()
    except ImportError:
        print(
            "statsmodels is not installed -- skipping Section 2. Install the "
            "`dev` extra to run this section: "
            'pip install -e "packages/lazyforecast[dev]"'
        )
        return

    co2_config = ForecastConfig(horizon=12, freq="MS", season_length=12, models=("AutoARIMA", "AutoETS"))

    print("=== Tier-1 input validation ===")
    co2_report = validate_input(co2_panel)
    print(f"input_kind={co2_report.input_kind} n_rows={co2_report.n_rows} n_series={co2_report.n_series}")

    print("\n=== Forecast (next 12 months) ===")
    co2_forecasts = forecast(co2_panel, co2_config)
    print(co2_forecasts.to_string(index=False))

    print("\n=== Backtest (holding out the last 12 months) ===")
    co2_result = backtest(co2_panel, co2_config, test_size=12)
    print(co2_result.to_frame().to_string(index=False))

    print("\n=== Section 2 summary (real co2 dataset) ===")
    for model_name in co2_config.models:
        print(
            f"{model_name}: mean MAE = {co2_result.mean_mae(model_name):.3f}, "
            f"mean RMSE = {co2_result.mean_rmse(model_name):.3f}"
        )
    print(f"Overall: mean MAE = {co2_result.mean_mae():.3f}, mean RMSE = {co2_result.mean_rmse():.3f}")
    print(f"(series std for scale reference: {co2_panel['y'].std():.3f})")

    print("\n\n=== Side-by-side: synthetic vs. real backtest error ===")
    print(
        f"{'dataset':<20}{'mean MAE':>12}{'mean RMSE':>12}"
    )
    print(
        f"{'synthetic (sine)':<20}{synthetic_result.mean_mae():>12.3f}{synthetic_result.mean_rmse():>12.3f}"
    )
    print(f"{'real (co2, monthly)':<20}{co2_result.mean_mae():>12.3f}{co2_result.mean_rmse():>12.3f}")


if __name__ == "__main__":
    main()
