"""Classical statistical forecasting (architecture doc Part 3, "Module 3: LazyForecast").

This module implements the **one signature capability** in scope for this
scaffold-depth pass: fitting Nixtla's ``statsforecast`` classical models
(``AutoARIMA`` and ``AutoETS``) over a Tier-1 Arrow-backed pandas or Polars
input, per the shared data-tier convention in architecture doc §2.1.

Explicitly out of scope for this pass (see package README for the full
rationale): the tree-based ML branch (MLForecast/LightGBM/XGBoost), the
zero-shot Time Series Foundation Models (TimesFM/Chronos-Bolt/Lag-Llama/
PatchTST), the self-healing preprocessing engine, and conformal-prediction
uncertainty quantification (MSCP/EnbPI).

This is the **one canonical forecasting path** in this package -- there is
no parallel/alternate fit-and-forecast implementation anywhere else here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

from lazycore.data import from_polars_zero_copy, is_arrow_backed_pandas, pandas_arrow_dtypes

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import polars as pl

__all__ = [
    "ForecastConfig",
    "Tier1ValidationReport",
    "SUPPORTED_MODELS",
    "validate_input",
    "prepare_frame",
    "build_statsforecast",
    "forecast",
]

#: Model names this scaffold-depth pass supports, per the architecture doc's
#: explicit scope: classical statistical forecasting only (AutoARIMA and
#: AutoETS via Nixtla's statsforecast). Nixtla also ships AutoTheta/AutoCES/
#: etc., but adding those is not part of this pass's signature capability.
SUPPORTED_MODELS: dict[str, type] = {
    "AutoARIMA": AutoARIMA,
    "AutoETS": AutoETS,
}

#: statsforecast's required long-format schema (architecture doc §2.1's
#: Tier-1 convention is about the *storage* format -- Arrow-backed pandas /
#: Polars -- not this column-naming convention, which is statsforecast's own
#: API contract). We rename the caller's columns into this shape internally
#: and never expose it as something the caller must match themselves.
_SF_ID_COL = "unique_id"
_SF_TIME_COL = "ds"
_SF_VALUE_COL = "y"


@dataclass
class ForecastConfig:
    """Configuration for :func:`forecast` (and reused by :mod:`backtest`).

    Attributes:
        id_col: name of the caller's series-identifier column.
        time_col: name of the caller's datetime column.
        value_col: name of the caller's numeric value column.
        horizon: number of future steps to forecast.
        freq: pandas frequency alias (e.g. ``"D"``, ``"W"``, ``"ME"``),
            passed straight through to ``statsforecast.StatsForecast``.
        season_length: seasonal period passed to each model that accepts one
            (e.g. 7 for daily data with a weekly cycle, 12 for monthly data
            with a yearly cycle).
        models: which of :data:`SUPPORTED_MODELS` to fit. Defaults to both
            classical models named in the architecture doc.
        n_jobs: parallelism passed to ``StatsForecast`` (1 = no
            multiprocessing; safe default for small scaffold-depth runs and
            for hermetic tests).
    """

    id_col: str = "unique_id"
    time_col: str = "ds"
    value_col: str = "y"
    horizon: int = 14
    freq: str = "D"
    season_length: int = 7
    models: tuple[str, ...] = ("AutoARIMA", "AutoETS")
    n_jobs: int = 1

    def __post_init__(self) -> None:
        """Validate ``models``/``horizon`` immediately at construction time.

        Raises:
            ValueError: ``models`` contains a name outside
                :data:`SUPPORTED_MODELS`, ``models`` is empty, or
                ``horizon`` is less than 1.
        """
        unknown = set(self.models) - set(SUPPORTED_MODELS)
        if unknown:
            raise ValueError(
                f"Unsupported model(s) {sorted(unknown)!r}. This scaffold-depth "
                f"pass only supports the classical models in SUPPORTED_MODELS: "
                f"{sorted(SUPPORTED_MODELS)!r}. Tree-based ML models and "
                "zero-shot TSFMs are explicitly out of scope for this pass "
                "(see README)."
            )
        if not self.models:
            raise ValueError("ForecastConfig.models must not be empty.")
        if self.horizon < 1:
            raise ValueError("ForecastConfig.horizon must be >= 1.")


@dataclass
class Tier1ValidationReport:
    """Result of :func:`validate_input` -- a Tier-1 "validate/report" pass.

    This is deliberately a lightweight report, not a hard gate: an input
    that is not yet Arrow-backed is still usable (statsforecast needs plain
    numpy-backed columns internally anyway, see ``prepare_frame``), but the
    report surfaces the Tier-1 posture of the *original* input the caller
    handed in, per architecture doc §2.1.
    """

    input_kind: str  # "pandas" or "polars"
    n_rows: int
    n_series: int
    arrow_backed_columns: dict[str, str] = field(default_factory=dict)
    is_fully_arrow_backed: bool = False
    warnings: list[str] = field(default_factory=list)


def _is_polars_dataframe(data: Any) -> bool:
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(data, pl.DataFrame)


def validate_input(data: Any, config: ForecastConfig | None = None) -> Tier1ValidationReport:
    """Validate and report on a Tier-1 input frame, per architecture doc §2.1.

    Accepts a pandas DataFrame (ideally Arrow-backed via pandas 2.x
    ``ArrowDtype`` columns) or a Polars DataFrame. Uses
    ``lazycore.data.is_arrow_backed_pandas``/``pandas_arrow_dtypes`` to
    report on pandas input rather than re-implementing that check -- per
    CLAUDE.md's "fix what's there, don't duplicate lazycore" rule.

    Raises:
        TypeError: ``data`` is neither a pandas nor a Polars DataFrame.
        ValueError: required columns (id/time/value) are missing.
    """
    config = config or ForecastConfig()
    warns: list[str] = []

    if _is_polars_dataframe(data):
        required = {config.id_col, config.time_col, config.value_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Input Polars DataFrame is missing required column(s): {missing!r}")
        n_series = data[config.id_col].n_unique()
        return Tier1ValidationReport(
            input_kind="polars",
            n_rows=data.height,
            n_series=n_series,
            arrow_backed_columns={},
            is_fully_arrow_backed=True,  # Polars is always Arrow-backed by construction
            warnings=warns,
        )

    if isinstance(data, pd.DataFrame):
        required = {config.id_col, config.time_col, config.value_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Input pandas DataFrame is missing required column(s): {missing!r}")

        arrow_cols = pandas_arrow_dtypes(data)
        fully_arrow = is_arrow_backed_pandas(data)
        if not fully_arrow:
            warns.append(
                "Input pandas DataFrame is not fully Arrow-backed (pandas 2.x "
                "ArrowDtype). benchcraft_lazyforecast follows lazycore's "
                "Tier-1 convention (architecture doc §2.1); consider "
                "`frame.convert_dtypes(dtype_backend='pyarrow')`. Proceeding "
                "anyway -- statsforecast itself requires plain numpy-backed "
                "columns internally, so this package converts either way."
            )

        return Tier1ValidationReport(
            input_kind="pandas",
            n_rows=len(data),
            n_series=data[config.id_col].nunique(),
            arrow_backed_columns=arrow_cols,
            is_fully_arrow_backed=fully_arrow,
            warnings=warns,
        )

    raise TypeError(
        f"Expected a pandas.DataFrame or polars.DataFrame, got {type(data)!r}. "
        "benchcraft_lazyforecast follows lazycore's Tier-1 Arrow-backed "
        "convention (architecture doc §2.1)."
    )


def prepare_frame(data: Any, config: ForecastConfig | None = None) -> pd.DataFrame:
    """Coerce a Tier-1 input frame into the plain pandas frame statsforecast needs.

    ``statsforecast``'s numba-jitted model implementations require plain
    numpy-backed ``float64``/``datetime64[ns]`` columns, not pandas'
    ArrowDtype columns -- so after validating/reporting on the Tier-1 input
    via :func:`validate_input`, this function performs the one necessary
    downstream conversion for that third-party library. This is *not* a
    duplicate of lazycore's Arrow<->Polars interop helpers: Polars input is
    still routed through ``lazycore.data.from_polars_zero_copy`` (reused, not
    reimplemented); the numpy-materialization step below is specific to
    statsforecast's own dtype requirements, which lazycore has no opinion on.

    Returns:
        A pandas DataFrame with exactly three columns, renamed to
        statsforecast's expected schema: ``unique_id`` (str), ``ds``
        (``datetime64[ns]``), ``y`` (``float64``), sorted by
        ``(unique_id, ds)``.
    """
    config = config or ForecastConfig()

    if _is_polars_dataframe(data):
        # Reuse lazycore's zero-copy Polars -> Arrow-backed pandas helper
        # rather than writing a parallel Polars->pandas conversion here.
        frame = from_polars_zero_copy(data)
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        raise TypeError(
            f"Expected a pandas.DataFrame or polars.DataFrame, got {type(data)!r}."
        )

    required = {config.id_col, config.time_col, config.value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input frame is missing required column(s): {missing!r}")

    prepared = pd.DataFrame(
        {
            _SF_ID_COL: frame[config.id_col].astype(str).to_numpy(),
            _SF_TIME_COL: pd.to_datetime(pd.Series(frame[config.time_col]).to_numpy()),
            _SF_VALUE_COL: pd.Series(frame[config.value_col]).astype("float64").to_numpy(),
        }
    )
    prepared = prepared.sort_values([_SF_ID_COL, _SF_TIME_COL]).reset_index(drop=True)

    if not np.isfinite(prepared[_SF_VALUE_COL].to_numpy()).all():
        raise ValueError(
            "Input value column contains NaN/inf values. This scaffold-depth "
            "pass does not implement the architecture doc's self-healing "
            "preprocessing engine (paradigm-aware imputation) -- clean/impute "
            "missing values before calling forecast()/backtest()."
        )

    return prepared


def build_statsforecast(config: ForecastConfig | None = None) -> StatsForecast:
    """Build a ``statsforecast.StatsForecast`` instance from a :class:`ForecastConfig`.

    Instantiates exactly the classical models named in the architecture doc
    for this pass (``AutoARIMA``/``AutoETS``), each parameterized with
    ``config.season_length``.
    """
    config = config or ForecastConfig()
    models = [SUPPORTED_MODELS[name](season_length=config.season_length) for name in config.models]
    return StatsForecast(models=models, freq=config.freq, n_jobs=config.n_jobs)


def forecast(data: Any, config: ForecastConfig | None = None) -> pd.DataFrame:
    """Fit classical statistical model(s) and forecast ``config.horizon`` steps ahead.

    This is the **one canonical entrypoint** for the forecasting path in
    this package.

    Args:
        data: a Tier-1 Arrow-backed pandas DataFrame or a Polars DataFrame,
            with an ID column, a datetime column, and a numeric value
            column (see :class:`ForecastConfig` for the expected column
            names, defaulting to ``unique_id``/``ds``/``y``).
        config: a :class:`ForecastConfig`. Defaults to
            ``ForecastConfig()`` (AutoARIMA + AutoETS, horizon 14, daily
            weekly-seasonal).

    Returns:
        A pandas DataFrame with columns ``unique_id``, ``ds``, and one
        column per fitted model (e.g. ``AutoARIMA``, ``AutoETS``) holding
        that model's point forecast for each of the next ``horizon`` steps
        of each series.
    """
    config = config or ForecastConfig()
    prepared = prepare_frame(data, config)
    sf = build_statsforecast(config)
    with warnings.catch_warnings():
        # statsforecast/numba emit a variety of benign performance/tuning
        # warnings on small synthetic series; this scaffold doesn't want
        # those cluttering caller output.
        warnings.simplefilter("ignore", category=UserWarning)
        forecasts = sf.forecast(df=prepared, h=config.horizon)
    return forecasts.reset_index(drop=True)
