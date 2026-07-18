# benchcraft-forecast

A scaffold-depth implementation of one signature capability from
Benchcraft's LazyForecast module (architecture doc Part 3, "Module 3:
LazyForecast"): **classical statistical forecasting (AutoARIMA/AutoETS via
Nixtla's `statsforecast`) running over a Tier-1 Arrow-backed input
pipeline**, plus a basic backtest/evaluation report.

## What this package is (and isn't) right now

The full LazyForecast module, per the architecture doc, reconciles three
forecasting paradigms under one Polars/Arrow zero-copy pipeline: classical
statistical models, tree-based ML models, and zero-shot Time Series
Foundation Models -- plus a self-healing preprocessing engine and a
conformal-prediction uncertainty-quantification layer scored on a unified
PICP/MPIW/Gneiting-Raftery leaderboard.

**This package currently implements exactly one branch: classical
statistical forecasting**, over a Tier-1 Arrow-backed input, with a basic
train/test-split backtest reporting MAE/RMSE. Everything else is
**explicitly out of scope for this pass** -- future work, not partially
stubbed out here:

- **Tree-based ML branch (MLForecast/LightGBM/XGBoost).** This is a
  materially different code path (rolling-window/lag feature
  tabularization, gradient-boosted tree fitting) with its own dependency
  surface (LightGBM/XGBoost) and its own failure modes. Bundling it into
  this pass would have doubled the scope without adding depth to either
  branch. Deferred to a follow-up pass that can give it the same
  scaffold-depth treatment.
- **Zero-shot Time Series Foundation Models (TimesFM 2.5, Chronos-Bolt,
  Lag-Llama, PatchTST).** These require a PyTorch/JAX tensor-export branch
  (DLPack/IPC handoff, per architecture doc §2.1's Tier-1 note on the
  "direct tensor-export branch"), a fundamentally different inference model
  (zero-shot forward pass vs. per-series model fitting), and their own Tier
  1/Tier 2 model-licensing allowlist work (§2.10 -- Apache-2.0 TimesFM/
  Chronos-Bolt as Tier 1, CC BY-NC MOIRAI as Tier 2 opt-in). None of that
  licensing/allowlist work is needed for the classical branch, since
  `statsforecast` ships no bundled model weights at all -- it's pure
  algorithmic code, Apache-2.0, with nothing to gate.
- **The self-healing preprocessing engine** (UTC timezone normalization,
  vectorized irregular-series upsampling, paradigm-aware imputation). This
  package's `prepare_frame` does the minimum necessary coercion to feed
  `statsforecast` a valid input (column renaming, dtype coercion, sorting)
  and explicitly **rejects** NaN/inf values with a clear error rather than
  silently imputing them -- imputation strategy is a real design decision
  the architecture doc treats as its own subsystem, not something to bolt
  on as a side effect of forecasting.
- **Conformal-prediction uncertainty quantification (MSCP/EnbPI)** and the
  full **PICP/MPIW/Gneiting-Raftery leaderboard**. Those score *interval*
  forecasts (upper/lower bounds with a coverage guarantee); this pass's
  `forecast()` only produces point forecasts, and `backtest()` only reports
  point-forecast error (MAE/RMSE) -- the "basic backtest/evaluation report"
  called for in this task's scope, not the full uncertainty-quantification
  leaderboard.

## Validation: synthetic and real datasets

The core test suite (`test_forecast.py`, `test_backtest.py`) is hermetic and
synthetic-only -- a hand-generated sine-wave-plus-trend panel, fixed seed.
That's sufficient to test plumbing, but says nothing about whether
`forecast()`/`backtest()` behave sensibly on a real series with real
seasonality, real noise, and no guaranteed clean periodicity.

`tests/test_real_dataset_validation.py` closes that gap by running the
*exact same* public API (`ForecastConfig`, `validate_input`,
`prepare_frame`, `forecast`, `backtest` -- no parallel data-prep path)
against two real, `statsmodels`-bundled datasets:

- **`statsmodels.datasets.co2`** (weekly Mauna Loa atmospheric CO2
  concentration, 1958-2001, resampled here to monthly means): a real series
  with unambiguous annual seasonality riding on a long-term trend -- exercises
  the seasonal AutoARIMA/AutoETS path (`season_length=12`) this package
  targets, and includes realistic missing-data gaps (interpolated by the
  test/example before reaching `prepare_frame()`, since `prepare_frame()`
  intentionally rejects NaN/inf rather than imputing).
- **`statsmodels.datasets.nile`** (annual Nile river flow at Aswan,
  1871-1970): a real *non-seasonal* series with a well-known 1898 structural
  break -- a harder stress test than co2's clean seasonality, since classical
  AutoARIMA/AutoETS have no seasonal signal to lean on.

Both datasets load from a CSV file physically bundled inside the installed
`statsmodels` package (e.g.
`site-packages/statsmodels/datasets/co2/co2.csv`) -- **zero network calls**,
confirmed by inspecting the installed package layout. `statsmodels` is a
**dev/test-only dependency** (see "Dependency surface" below): it supplies
real validation data, it is never imported by
`benchcraft_lazyforecast`'s own forecasting/backtest logic.

`examples/forecast_example.py` runs both the synthetic panel and the real
co2 dataset back-to-back and prints their backtest MAE/RMSE side by side, so
a reader can see the tool validated on both.

## The signature capability

### 1. Classical forecasting: `benchcraft_lazyforecast.forecast`

```python
import pandas as pd
from benchcraft_lazyforecast import ForecastConfig, forecast

df = pd.DataFrame({
    "unique_id": [...],   # series ID, string or int
    "ds": [...],          # datetime column
    "y": [...],           # numeric value column
})

config = ForecastConfig(horizon=14, freq="D", season_length=7, models=("AutoARIMA", "AutoETS"))
forecasts = forecast(df, config)
# columns: unique_id, ds, AutoARIMA, AutoETS
```

`forecast()` accepts a pandas DataFrame (ideally Tier-1 Arrow-backed via
pandas 2.x `ArrowDtype` columns, per architecture doc §2.1) or a Polars
DataFrame, with configurable ID/time/value column names. Under the hood it:

1. Validates/reports on the input via `validate_input()`, reusing
   `lazycore.data.is_arrow_backed_pandas`/`pandas_arrow_dtypes` rather than
   re-implementing an Arrow-dtype check.
2. Coerces the input into the plain, numpy-backed `unique_id`/`ds`/`y`
   schema `statsforecast`'s numba-jitted models require, via
   `prepare_frame()`. Polars input is routed through
   `lazycore.data.from_polars_zero_copy` -- **not** a hand-rolled Polars
   conversion. Polars itself is only imported lazily, at the point a caller
   actually passes a `polars.DataFrame`; it is never a hard dependency of
   this package.
3. Fits `statsforecast.StatsForecast` with the configured classical models
   (`AutoARIMA` and/or `AutoETS`) and forecasts `config.horizon` steps
   ahead.

`ForecastConfig.models` only accepts `"AutoARIMA"`/`"AutoETS"` (see
`SUPPORTED_MODELS`) -- passing any other model name (e.g. a tree-based or
TSFM model name) raises `ValueError` immediately, rather than silently
being ignored, so scope violations fail loudly.

### 2. Backtest/evaluation: `benchcraft_lazyforecast.backtest`

```python
from benchcraft_lazyforecast import backtest

report = backtest(df, config, test_size=14)
report.to_frame()      # per-(series, model) MAE/RMSE
report.mean_mae()      # averaged across all series and models
report.mean_mae("AutoARIMA")  # averaged across series, one model
```

For each series, the last `test_size` observations are held out, the
configured model(s) are fit on the remaining (earlier) observations, a
`test_size`-step-ahead forecast is produced, and MAE/RMSE are computed
against the held-out actuals. This is the "basic backtest/evaluation
report" called for in this task's scope -- a single train/test split per
series, not the architecture doc's full rolling-origin
cross-validation-plus-conformal-interval leaderboard.

## Dependency surface

- **Core (always installed):** `numpy`, `pandas`, `statsforecast`.
  `statsforecast` is Apache-2.0 and runs fully local/CPU -- no network
  calls, no bundled model checkpoints, so no Tier 1/Tier 2 licensing
  isolation work is needed for this branch (unlike the deferred TSFM
  branch, which does need it). `statsforecast` pulls in its own
  numba/scipy dependency tree; that's expected and accepted, not accidental
  bloat.
- **Polars input:** supported, but **not** a hard dependency -- only
  imported lazily when a caller actually passes a `polars.DataFrame` (see
  `forecast.py`'s `_is_polars_dataframe`), mirroring the lazy-import
  pattern already used in `lazycore.data`.
- **Optional `dev` extra:** `pytest`, `statsmodels`. `statsmodels` is used
  *only* to supply bundled, real, offline time-series datasets (co2, nile)
  for `tests/test_real_dataset_validation.py` and the second section of
  `examples/forecast_example.py` -- it is not a runtime dependency of
  `benchcraft_lazyforecast`'s forecasting logic (that's `statsforecast`,
  listed above).

This package also uses `lazycore.data`'s Tier-1 Arrow-tabular helpers
(`is_arrow_backed_pandas`, `pandas_arrow_dtypes`, `to_polars_zero_copy`,
`from_polars_zero_copy`) rather than reimplementing pandas/Polars/Arrow
interop -- per CLAUDE.md's "fix what's there before adding new" rule.

`lazycore` is a local sibling package (`packages/lazycore`) and is
**installed separately, not as a formal pyproject dependency of this
package** -- hatchling/pip don't have a portable, idiomatic way to express
a relative-path dependency in `pyproject.toml` metadata the way e.g.
Poetry's `path = "../lazycore"` does. This matches the exact convention
already established in `packages/automl` and `packages/lazyclean`. Install
it first (see below).

## Installation (local dev)

```bash
# from the repo root
pip install -e packages/lazycore
pip install -e "packages/lazyforecast[dev]"
```

## Running tests

```bash
pytest packages/lazyforecast/tests
```

Fully hermetic -- no network access required, including the real-dataset
tests (see "Validation" above): `statsmodels`'s bundled datasets load from
local package data files. Tests build a small synthetic multi-series dataset
(sine-wave-plus-trend, fixed seed, 2 series) directly in-file and fit against
it, and separately validate against the real co2/nile datasets. The
Polars-input test is skipped (via `pytest.importorskip`), not failed, if
`polars` isn't installed in the test environment.

## Running the example

```bash
python packages/lazyforecast/examples/forecast_example.py
```

Section 1 generates a synthetic two-store seasonal-sales dataset, forecasts
the next 14 days with AutoARIMA + AutoETS, backtests against the last 14
known days, and prints the forecast table plus the mean MAE/RMSE per model.
Section 2 repeats the same forecast/backtest calls against the real
`statsmodels` co2 dataset (see "Validation" above) and prints a side-by-side
synthetic-vs-real backtest error comparison. Section 2 requires the `dev`
extra (for `statsmodels`); if it isn't installed, the script prints a note
and skips that section rather than failing.
