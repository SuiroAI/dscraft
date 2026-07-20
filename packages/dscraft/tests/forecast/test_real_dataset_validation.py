"""Real-dataset validation for dscraft.forecast.

The rest of this package's test suite (test_forecast.py, test_backtest.py)
is deliberately hermetic and synthetic: a hand-generated sine-wave-plus-trend
panel with a fixed seed. That's the right tool for testing plumbing (schema
coercion, column renaming, Arrow-dtype handling, error paths), but it proves
nothing about whether the forecasting pipeline behaves sensibly on a real
time series with real seasonality, real noise, and no guaranteed clean
periodicity.

This module closes that gap using datasets **bundled inside the already
test-only `statsmodels` dependency** -- per the stakeholder's explicit
preference for "ships as a local package data file" over "fetch something
from the internet." Both datasets used below load from a CSV file physically
installed under `site-packages/statsmodels/datasets/<name>/<name>.csv` (see
e.g. `statsmodels.datasets.co2.__file__`'s directory) -- loading them makes
zero network calls and works fully offline, confirmed by inspecting the
installed package layout during development of this test.

Two real datasets, two different validation angles:

- ``statsmodels.datasets.co2`` (weekly Mauna Loa atmospheric CO2
  concentration, 1958-2001): resampled to monthly means, this is a real
  series with unambiguous annual seasonality (season_length=12) riding on a
  long-term upward trend -- a good test of the seasonal AutoARIMA/AutoETS
  path this package's SUPPORTED_MODELS targets. The raw weekly series has
  scattered missing weeks (equipment downtime), which is realistic
  real-world messiness the synthetic panel never exercises; `y` values are
  linearly interpolated before handing the frame to `prepare_frame()`
  because `prepare_frame()` intentionally *rejects* NaN/inf rather than
  silently imputing (see forecast.py) -- that's this package's real,
  documented contract, so the test respects it rather than working around
  it.
- ``statsmodels.datasets.nile`` (annual Nile river flow at Aswan,
  1871-1970): a real *non-seasonal* series (season_length=1) containing a
  well-known structural break (a 1898 dam-construction-era shift in mean
  flow). This is a deliberately harder stress test than the clean synthetic
  panel: classical AutoARIMA/AutoETS have no seasonal signal to lean on and
  must track a noisier, regime-shifted series. It exercises the same public
  API (forecast/backtest) on a shape of data (short, annual, structural
  break) very different from co2's long, high-frequency, cleanly seasonal
  shape.

Both datasets are reshaped into the exact `unique_id`/`ds`/`y` long-format
schema `dscraft.forecast` expects, then run through the package's
*existing* `validate_input`/`prepare_frame`/`forecast`/`backtest` -- the same
public API and the same canonical path the synthetic tests use. No parallel
data-prep or forecasting logic is introduced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.datasets import co2, nile

from dscraft.forecast import ForecastConfig, backtest, forecast, prepare_frame, validate_input


def _load_co2_monthly_panel() -> pd.DataFrame:
    """Real Mauna Loa CO2 concentration, resampled to monthly means.

    Reshaped into the unique_id/ds/y long format `dscraft.forecast`
    expects. The raw weekly series has ~59 missing weeks out of 2284; those
    become NaNs after resampling to monthly, so we `interpolate()` them here
    -- in the caller's own data-prep step, not inside the package -- because
    `prepare_frame()` deliberately rejects NaN/inf rather than imputing
    (architecture doc: the self-healing preprocessing/imputation engine is
    explicitly out of scope for this pass; see forecast.py/README).
    """
    raw = co2.load_pandas().data["co2"]
    monthly = raw.resample("MS").mean().interpolate()
    assert not monthly.isna().any(), "unexpected residual NaNs after interpolation"
    return pd.DataFrame({"unique_id": "co2_monthly", "ds": monthly.index, "y": monthly.to_numpy()})


def _load_nile_annual_panel() -> pd.DataFrame:
    """Real annual Nile river flow (with a known 1898 structural break).

    Reshaped into the unique_id/ds/y long format `dscraft.forecast`
    expects; the dataset's `year` column (a float) is converted to a
    datetime column via pandas.
    """
    raw = nile.load_pandas().data
    ds = pd.to_datetime(raw["year"].astype(int).astype(str), format="%Y")
    return pd.DataFrame({"unique_id": "nile_flow", "ds": ds, "y": raw["volume"].astype(float).to_numpy()})


@pytest.fixture(scope="module")
def co2_panel() -> pd.DataFrame:
    """Module-scoped real Mauna Loa CO2 monthly panel, loaded once and shared across co2 tests."""
    return _load_co2_monthly_panel()


@pytest.fixture(scope="module")
def nile_panel() -> pd.DataFrame:
    """Module-scoped real Nile annual flow panel, loaded once and shared across nile tests."""
    return _load_nile_annual_panel()


# --- co2: real seasonal series ----------------------------------------------


def test_co2_dataset_loads_locally_with_expected_schema(co2_panel: pd.DataFrame) -> None:
    """Sanity-check the reshaped real dataset before running it through the
    package -- this is a real series, so we assert shape/columns, not values.
    """
    assert list(co2_panel.columns) == ["unique_id", "ds", "y"]
    assert len(co2_panel) > 100  # ~43 years of monthly data
    assert np.isfinite(co2_panel["y"].to_numpy()).all()


def test_co2_validate_input_and_prepare_frame_reuse_existing_helpers(co2_panel: pd.DataFrame) -> None:
    """Exercise the package's *existing* validate_input/prepare_frame path
    (not a parallel real-data-specific prep path) against real data."""
    report = validate_input(co2_panel)
    assert report.input_kind == "pandas"
    assert report.n_series == 1
    assert report.n_rows == len(co2_panel)

    prepared = prepare_frame(co2_panel)
    assert list(prepared.columns) == ["unique_id", "ds", "y"]
    assert np.isfinite(prepared["y"].to_numpy()).all()


def test_co2_forecast_shape_and_finiteness(co2_panel: pd.DataFrame) -> None:
    """forecast() on the real co2 series must produce finite, physically-plausible values (positive, not wildly above history)."""
    config = ForecastConfig(horizon=12, freq="MS", season_length=12, models=("AutoARIMA", "AutoETS"))
    result = forecast(co2_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA", "AutoETS"}
    assert len(result) == config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
    assert np.isfinite(result["AutoETS"].to_numpy()).all()
    # A real, sanity-checkable expectation for a monotonically-rising real
    # series: the forecasted CO2 level should stay within a plausible band
    # around the series' own historical range, not diverge to something
    # physically nonsensical (e.g. negative concentration, or an order of
    # magnitude off).
    history_max = co2_panel["y"].max()
    assert (result["AutoARIMA"] > 0).all()
    assert (result["AutoARIMA"] < history_max * 1.5).all()


def test_co2_backtest_error_is_small_relative_to_series_scale(co2_panel: pd.DataFrame) -> None:
    """The real sanity bound: backtest MAE should be meaningfully smaller
    than the series' own standard deviation, computed here (not a hardcoded
    magic number that only happens to work for one seed). CO2 is a strongly
    seasonal + trending real series that a seasonal classical model should
    track quite closely over a 12-month held-out window.
    """
    config = ForecastConfig(horizon=12, freq="MS", season_length=12, models=("AutoARIMA", "AutoETS"))
    report = backtest(co2_panel, config, test_size=12)

    series_std = float(co2_panel["y"].std())
    assert report.mean_mae() < 0.25 * series_std
    assert report.mean_rmse() < 0.25 * series_std
    for model_name in config.models:
        assert report.mean_mae(model_name) < 0.25 * series_std


# --- nile: real non-seasonal series with a structural break -----------------


def test_nile_dataset_loads_locally_with_expected_schema(nile_panel: pd.DataFrame) -> None:
    """Sanity-check the reshaped real Nile dataset's shape/columns before running it through the package."""
    assert list(nile_panel.columns) == ["unique_id", "ds", "y"]
    assert len(nile_panel) == 100  # 1871-1970 inclusive
    assert np.isfinite(nile_panel["y"].to_numpy()).all()


def test_nile_forecast_shape_and_finiteness(nile_panel: pd.DataFrame) -> None:
    """forecast() on the real, non-seasonal, structurally-broken Nile series must still produce finite values."""
    config = ForecastConfig(horizon=10, freq="YS", season_length=1, models=("AutoARIMA", "AutoETS"))
    result = forecast(nile_panel, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA", "AutoETS"}
    assert len(result) == config.horizon
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
    assert np.isfinite(result["AutoETS"].to_numpy()).all()


def test_nile_backtest_error_is_smaller_than_series_scale(nile_panel: pd.DataFrame) -> None:
    """Nile flow is a genuinely harder real series (structural break, no
    seasonality) than co2, so this is a looser sanity bound than co2's --
    but it must still hold: a working classical model forecasting the last
    10 years should do meaningfully better than the series' own spread,
    computed here rather than hardcoded.
    """
    config = ForecastConfig(horizon=10, freq="YS", season_length=1, models=("AutoARIMA", "AutoETS"))
    report = backtest(nile_panel, config, test_size=10)

    series_std = float(nile_panel["y"].std())
    assert report.mean_mae() < series_std
    assert report.mean_rmse() < series_std
