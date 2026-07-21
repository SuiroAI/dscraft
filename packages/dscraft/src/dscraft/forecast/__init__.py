"""dscraft.forecast -- LazyForecast scaffold: classical statistical forecasting.

Public API surface for the classical-statistical-forecasting branch of
LazyForecast (architecture doc Part 3, "Module 3: LazyForecast"): fit any
of Nixtla's ``statsforecast`` classical models (see ``SUPPORTED_MODELS`` --
autoregressive/exponential-smoothing autofits, simple baselines, and the
Croston intermittent-demand family) over a Tier-1 Arrow-backed pandas or
Polars input, produce a horizon forecast, and score it with a basic
backtest/evaluation report (per-series and averaged MAE/RMSE).

This subpackage also exposes standalone time series decomposition
(STL/MSTL, optional Box-Cox) via :func:`decompose` -- a diagnostic/
preprocessing capability in its own right, not folded into ``forecast()``.

Everything else described for LazyForecast in the architecture doc -- the
tree-based ML branch (MLForecast/LightGBM/XGBoost), the zero-shot Time
Series Foundation Models (TimesFM/Chronos-Bolt/Lag-Llama/PatchTST), the
self-healing preprocessing engine, and conformal-prediction uncertainty
quantification (MSCP/EnbPI) -- is explicitly out of scope for this pass.
See the package README for the full rationale.

    >>> from dscraft.forecast import ForecastConfig, forecast, backtest, decompose
    >>> config = ForecastConfig(horizon=14, freq="D", season_length=7)
    >>> forecasts = forecast(df, config)
    >>> report = backtest(df, config, test_size=14)
    >>> report.mean_mae()
    >>> decomposition = decompose(df, config, method="stl")
    >>> decomposition.components
"""

from __future__ import annotations

from .backtest import BacktestAlignmentError, BacktestReport, SeriesMetric, backtest
from .decomposition import SUPPORTED_DECOMPOSITION_METHODS, DecompositionResult, decompose
from .forecast import (
    SUPPORTED_MODELS,
    ForecastConfig,
    Tier1ValidationReport,
    build_statsforecast,
    forecast,
    prepare_frame,
    validate_input,
)

__all__ = [
    "ForecastConfig",
    "Tier1ValidationReport",
    "SUPPORTED_MODELS",
    "validate_input",
    "prepare_frame",
    "build_statsforecast",
    "forecast",
    "SeriesMetric",
    "BacktestReport",
    "BacktestAlignmentError",
    "backtest",
    "SUPPORTED_DECOMPOSITION_METHODS",
    "DecompositionResult",
    "decompose",
]

__version__ = "0.1.0"
