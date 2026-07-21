"""Standard-benchmark validation for dscraft.forecast via `datasetsforecast`.

Per issue #31's gap analysis: the rest of this package's real-dataset
validation (test_real_dataset_validation.py) uses datasets bundled inside
the `statsmodels` package (co2, nile) -- real data, but not a standard
forecasting-competition benchmark. `datasetsforecast` (Nixtla, Apache-2.0/
Tier 1) is the standard way the field validates forecasting methods
against published competition data (M3/M4/M5), and is used throughout
*Forecasting: Principles and Practice, the Pythonic Way*'s foundation-
model chapter for exactly this purpose.

Unlike `statsmodels`' co2/nile datasets, `datasetsforecast` genuinely
downloads its data over the network on first use (there is no bundled/
offline copy) -- so, per the task's explicit instruction, this module is
network-optional: it attempts the download and skips (rather than
failing) if the network is unavailable, consistent with this repo's
local-first testing posture not requiring network access to pass the
suite.

Dataset choice: the M3 competition's "Other" group is the smallest group
`datasetsforecast` ships across M3/M4 (174 series, a ~33KB download,
confirmed during development of this test) -- far smaller than any M4
group (the smallest, M4 Weekly, is still 359 series backed by a
multi-megabyte CSV download) or M3's Yearly/Quarterly/Monthly groups
(645-1428 series each). `datasetsforecast` caches the downloaded/
decompressed file to disk (`M3.load`'s own behavior), so within a single
test session (and across runs sharing the pytest tmp path parent, if the
caller pins one) the network call only happens once.

This test validates the *existing* forecast()/backtest() pipeline against
real M3 data -- no parallel data-loading or forecasting logic is
introduced here -- and only asserts the pipeline runs end-to-end and
produces sane (finite, correctly-shaped) output, not a specific accuracy
number (per the task's explicit acceptance criteria: this is a benchmark-
validation smoke test, not an accuracy regression test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscraft.forecast import ForecastConfig, backtest, forecast

datasetsforecast = pytest.importorskip("datasetsforecast")

#: Number of M3 "Other" series to actually run forecast()/backtest() over.
#: The full "Other" group (174 series) runs in well under a second locally,
#: but this keeps the test both fast and insulated from any one series'
#: idiosyncrasies (a small, fixed, deterministic subset rather than "all of
#: them" or a random sample).
_N_SERIES = 8

#: M3 "Other" group metadata (see datasetsforecast.m3.M3Info): daily
#: frequency, non-seasonal (seasonality=1), official competition horizon 8.
_HORIZON = 8
_FREQ = "D"
_SEASON_LENGTH = 1


def _load_m3_other_panel(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Download (or reuse a cached copy of) the M3 "Other" competition group.

    Skips the test (rather than failing it) if the dataset can't be
    downloaded -- e.g. no network access -- per this repo's local-first
    testing posture: the rest of the suite must not require network access
    to pass.
    """
    from datasetsforecast.m3 import M3

    directory = tmp_path_factory.mktemp("datasetsforecast_m3", numbered=False)
    try:
        df, _x_df, _s_df = M3.load(str(directory), "Other")
    except Exception as exc:  # noqa: BLE001 - genuinely any network/IO failure should skip, not fail
        pytest.skip(f"datasetsforecast M3 'Other' download unavailable (offline?): {exc!r}")
    return df


@pytest.fixture(scope="module")
def m3_other_panel(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Module-scoped full M3 'Other' panel (174 series), downloaded/cached once."""
    return _load_m3_other_panel(tmp_path_factory)


@pytest.fixture(scope="module")
def m3_other_subset(m3_other_panel: pd.DataFrame) -> pd.DataFrame:
    """A small, fixed subset of `_N_SERIES` M3 'Other' series for the actual forecast()/backtest() runs."""
    subset_ids = sorted(m3_other_panel["unique_id"].unique())[:_N_SERIES]
    subset = m3_other_panel[m3_other_panel["unique_id"].isin(subset_ids)]
    return subset.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def test_m3_other_dataset_loads_with_expected_schema(m3_other_panel: pd.DataFrame) -> None:
    """Sanity-check the real M3 'Other' panel's shape/columns/finiteness before running it through the package."""
    assert list(m3_other_panel.columns) == ["unique_id", "ds", "y"]
    assert m3_other_panel["unique_id"].nunique() == 174  # per datasetsforecast.m3.M3Info's Other.n_ts
    assert np.isfinite(m3_other_panel["y"].to_numpy()).all()


def test_m3_other_forecast_shape_and_finiteness(m3_other_subset: pd.DataFrame) -> None:
    """forecast() on a real M3 'Other' subset must produce finite, correctly-shaped output via the existing pipeline."""
    n_series = m3_other_subset["unique_id"].nunique()
    config = ForecastConfig(
        horizon=_HORIZON, freq=_FREQ, season_length=_SEASON_LENGTH, models=("AutoARIMA", "AutoETS")
    )
    result = forecast(m3_other_subset, config)

    assert set(result.columns) == {"unique_id", "ds", "AutoARIMA", "AutoETS"}
    assert len(result) == n_series * _HORIZON
    assert np.isfinite(result["AutoARIMA"].to_numpy()).all()
    assert np.isfinite(result["AutoETS"].to_numpy()).all()


def test_m3_other_backtest_runs_end_to_end_and_produces_sane_metrics(m3_other_subset: pd.DataFrame) -> None:
    """backtest() on a real M3 'Other' subset must run end-to-end and report finite, non-negative MAE/RMSE.

    This is a pipeline-validation smoke test against a real published
    competition benchmark, not an accuracy regression test -- per the
    task's acceptance criteria, no specific accuracy threshold is asserted.
    """
    n_series = m3_other_subset["unique_id"].nunique()
    config = ForecastConfig(
        horizon=_HORIZON, freq=_FREQ, season_length=_SEASON_LENGTH, models=("AutoARIMA", "AutoETS")
    )
    report = backtest(m3_other_subset, config, test_size=_HORIZON)

    assert len(report.metrics) == n_series * len(config.models)
    for metric in report.metrics:
        assert metric.n_points == metric.expected_points == _HORIZON
        assert np.isfinite(metric.mae)
        assert np.isfinite(metric.rmse)
        assert metric.mae >= 0
        assert metric.rmse >= 0
    assert np.isfinite(report.mean_mae())
    assert np.isfinite(report.mean_rmse())
