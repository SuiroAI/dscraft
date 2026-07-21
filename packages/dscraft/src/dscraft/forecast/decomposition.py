"""Time series decomposition -- STL/MSTL/Box-Cox (architecture doc Part 3, "Module 3: LazyForecast").

Per issue #22's gap analysis against *Forecasting: Principles and
Practice, the Pythonic Way* (chapters 3 and 12), `dscraft.forecast` had no
decomposition capability before this module: it went straight from raw
input to `StatsForecast` model fitting. This module adds decomposition as
a **standalone, inspectable operation** -- not folded silently into
:func:`dscraft.forecast.forecast.forecast` -- exposing:

- ``method="stl"``: single-seasonal-period decomposition via
  ``statsmodels.tsa.seasonal.STL`` (trend + one seasonal component +
  residual).
- ``method="mstl"``: multiple-seasonal-period decomposition via
  ``statsmodels.tsa.seasonal.MSTL`` (trend + one seasonal component per
  requested period + residual).
- an optional Box-Cox variance-stabilizing transform (``scipy.stats.boxcox``
  / ``scipy.special.inv_boxcox``) applied per-series before decomposition.

Zero new dependencies: ``statsmodels`` is already a `forecast`-extra
dependency (see pyproject.toml's comment on why), and ``scipy`` is already
one of *its* transitive dependencies (`pip show statsmodels` ->
Requires: ... scipy).

This module reuses :func:`dscraft.forecast.forecast.prepare_frame` and
:class:`dscraft.forecast.forecast.ForecastConfig` for Tier-1 input
validation/coercion rather than duplicating that logic -- per CLAUDE.md's
"fix what's there, don't duplicate" rule, this is the same canonical
Tier-1 entry path `forecast()`/`backtest()` already use.

Per CLAUDE.md, `dscraft.eda` and `dscraft.forecast` stay independent
subpackages (no formal inter-module contracts yet) -- this decomposition
capability is `forecast`-specific, not a shared `dscraft.core` utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox as _scipy_boxcox
from statsmodels.tsa.seasonal import MSTL, STL

from .forecast import ForecastConfig, prepare_frame

__all__ = [
    "SUPPORTED_DECOMPOSITION_METHODS",
    "DecompositionResult",
    "decompose",
    "boxcox_transform",
    "inverse_boxcox_transform",
]

#: The two decomposition methods this module supports, per issue #22's
#: scope: single-seasonal-period STL and multi-seasonal-period MSTL. X-11/
#: SEATS decomposition (also named in the issue's evidence) is not
#: implemented -- `statsmodels` doesn't ship it, and adding a new
#: dependency for it is out of scope for this zero-new-dependency pass.
SUPPORTED_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"stl", "mstl"})


@dataclass
class DecompositionResult:
    """Result of :func:`decompose` -- per-series trend/seasonal/residual components.

    Attributes:
        components: a tidy long-format DataFrame with the same
            ``unique_id``/``ds`` shape used everywhere else in this
            subpackage, plus ``trend``, ``resid``, and one seasonal column
            per decomposed period: ``seasonal`` for ``method="stl"``
            (exactly one seasonal component), or ``seasonal_<period>`` for
            each period in ``method="mstl"`` (e.g. ``seasonal_7``,
            ``seasonal_365`` for a series with both weekly and yearly
            seasonality). If ``boxcox=True`` was passed to
            :func:`decompose`, an additional ``boxcox_lambda`` column
            holds each row's series' fitted lambda (constant within a
            ``unique_id``, since Box-Cox is fit per series).
        method: which of :data:`SUPPORTED_DECOMPOSITION_METHODS` produced
            this result.
        boxcox_lambda: the fitted Box-Cox lambda, when ``boxcox=True`` was
            passed to :func:`decompose` **and** the input contained
            exactly one series (a per-series lambda is unambiguous only in
            that case). For a multi-series input decomposed with
            ``boxcox=True``, this is ``None`` -- consult the per-row
            ``boxcox_lambda`` column in ``components`` instead, since each
            series generally has its own fitted lambda. ``None`` when
            ``boxcox=False`` (the default).
    """

    components: pd.DataFrame
    method: str
    boxcox_lambda: float | None = None


def boxcox_transform(values: np.ndarray, lmbda: float | None = None) -> tuple[np.ndarray, float]:
    """Apply a Box-Cox transform to a 1-D array of strictly-positive values.

    Thin wrapper over ``scipy.stats.boxcox``: if ``lmbda`` is ``None``
    (the default), the transform parameter is fit via ``scipy``'s maximum-
    likelihood estimator; otherwise the given ``lmbda`` is applied
    directly. Exposed as a standalone helper (in addition to
    :func:`decompose`'s ``boxcox=True`` opt-in) so callers can apply/
    invert the same transform outside of a decomposition call -- e.g. to
    round-trip a forecast produced on Box-Cox-transformed data back to the
    original scale via :func:`inverse_boxcox_transform`.

    Args:
        values: a 1-D array of strictly-positive floats.
        lmbda: the Box-Cox parameter to apply, or ``None`` to fit it via
            MLE.

    Returns:
        A ``(transformed, lmbda)`` tuple: the transformed array and the
        (fitted or given) lambda used.

    Raises:
        ValueError: ``values`` contains a non-positive entry (Box-Cox is
            only defined for strictly-positive data).
    """
    values = np.asarray(values, dtype="float64")
    if not np.all(values > 0):
        raise ValueError(
            "boxcox_transform() requires strictly-positive values (Box-Cox "
            "is undefined for zero/negative data)."
        )
    if lmbda is None:
        transformed, fitted_lmbda = _scipy_boxcox(values)
        return transformed, float(fitted_lmbda)
    transformed = _scipy_boxcox(values, lmbda=lmbda)
    return transformed, float(lmbda)


def inverse_boxcox_transform(values: np.ndarray, lmbda: float) -> np.ndarray:
    """Invert a Box-Cox transform, recovering the original-scale values.

    Thin wrapper over ``scipy.special.inv_boxcox``, paired with
    :func:`boxcox_transform`.

    Args:
        values: a 1-D array of Box-Cox-transformed values.
        lmbda: the lambda originally used to produce ``values`` (e.g. the
            second element of :func:`boxcox_transform`'s return, or a
            :class:`DecompositionResult`'s ``boxcox_lambda``).

    Returns:
        The original-scale array. Recovers the pre-transform values within
        floating-point tolerance for any ``values`` produced by
        :func:`boxcox_transform` with the same ``lmbda``.
    """
    values = np.asarray(values, dtype="float64")
    return np.asarray(inv_boxcox(values, lmbda), dtype="float64")


def decompose(
    data: Any,
    config: ForecastConfig | None = None,
    *,
    method: str = "stl",
    period: int | None = None,
    periods: Sequence[int] | None = None,
    boxcox: bool = False,
) -> DecompositionResult:
    """Decompose each series in ``data`` into trend/seasonal/residual components.

    A standalone, inspectable diagnostic operation -- this is *not* wired
    into :func:`dscraft.forecast.forecast.forecast`'s fit-and-forecast
    path; callers who want decomposition call this function directly.

    Args:
        data: a Tier-1 Arrow-backed pandas DataFrame or a Polars
            DataFrame, in the same shape :func:`dscraft.forecast.forecast.forecast`
            accepts (an ID column, a datetime column, and a numeric value
            column -- see :class:`ForecastConfig` for the expected column
            names).
        config: a :class:`ForecastConfig`. Only ``id_col``/``time_col``/
            ``value_col`` (for input coercion) and ``season_length``
            (the default seasonal period, see ``period``/``periods``
            below) are used; ``horizon``/``models``/``freq``/``n_jobs`` are
            ignored. Defaults to ``ForecastConfig()``.
        method: ``"stl"`` (single seasonal period) or ``"mstl"`` (multiple
            seasonal periods). Must be one of
            :data:`SUPPORTED_DECOMPOSITION_METHODS`.
        period: the seasonal period for ``method="stl"``. Defaults to
            ``config.season_length`` if not given. Ignored for
            ``method="mstl"``.
        periods: the list of seasonal periods for ``method="mstl"`` (e.g.
            ``[7, 365]`` for a daily series with both weekly and yearly
            seasonality). Defaults to ``[config.season_length]`` if not
            given. Ignored for ``method="stl"``.
        boxcox: if ``True``, apply a Box-Cox transform (fit independently
            per series via :func:`boxcox_transform`) to each series before
            decomposing it. The decomposition's trend/seasonal/residual
            components are then on the Box-Cox-transformed scale, not the
            original scale -- use :func:`inverse_boxcox_transform` with
            the returned lambda(s) to map values back if needed.

    Returns:
        A :class:`DecompositionResult` holding the tidy long-format
        component DataFrame (see its docstring for the exact columns) and
        the fitted Box-Cox lambda(s), if requested.

    Raises:
        ValueError: ``method`` is not one of
            :data:`SUPPORTED_DECOMPOSITION_METHODS`; ``period``/``periods``
            resolves to a value less than 2; ``periods`` is empty for
            ``method="mstl"``; or (when ``boxcox=True``) a series contains
            a non-positive value.
    """
    config = config or ForecastConfig()
    if method not in SUPPORTED_DECOMPOSITION_METHODS:
        raise ValueError(
            f"Unsupported decomposition method {method!r}. Must be one of "
            f"{sorted(SUPPORTED_DECOMPOSITION_METHODS)!r}."
        )

    prepared = prepare_frame(data, config)

    if method == "stl":
        resolved_period = period if period is not None else config.season_length
        if resolved_period < 2:
            raise ValueError(f"period must be >= 2 for STL decomposition, got {resolved_period!r}.")
        resolved_periods: list[int] = [resolved_period]
    else:
        resolved_periods = list(periods) if periods is not None else [config.season_length]
        if not resolved_periods:
            raise ValueError("periods must not be empty for MSTL decomposition.")
        if any(p < 2 for p in resolved_periods):
            raise ValueError(f"All entries in periods must be >= 2 for MSTL decomposition, got {resolved_periods!r}.")

    per_series_frames: list[pd.DataFrame] = []
    boxcox_lambdas: dict[str, float] = {}

    for unique_id, group in prepared.groupby("unique_id", sort=False):
        group = group.sort_values("ds")
        ds = pd.DatetimeIndex(group["ds"].to_numpy())
        y = group["y"].to_numpy(dtype="float64")

        lmbda: float | None = None
        if boxcox:
            if not np.all(y > 0):
                raise ValueError(
                    f"Series {unique_id!r} contains non-positive value(s); Box-Cox "
                    "requires strictly-positive data. Pass boxcox=False, or clean/"
                    "shift the series before calling decompose()."
                )
            y, lmbda = boxcox_transform(y)
            boxcox_lambdas[str(unique_id)] = lmbda

        series = pd.Series(y, index=ds)

        if method == "stl":
            fit = STL(series, period=resolved_periods[0]).fit()
            component_frame = pd.DataFrame(
                {
                    "unique_id": str(unique_id),
                    "ds": ds,
                    "trend": np.asarray(fit.trend),
                    "seasonal": np.asarray(fit.seasonal),
                    "resid": np.asarray(fit.resid),
                }
            )
        else:
            fit = MSTL(series, periods=resolved_periods).fit()
            component_frame = pd.DataFrame(
                {
                    "unique_id": str(unique_id),
                    "ds": ds,
                    "trend": np.asarray(fit.trend),
                }
            )
            # statsmodels' MSTL returns `seasonal` as a DataFrame with one
            # `seasonal_<period>` column per requested period when len(periods)
            # > 1, but as a single plain Series (named "seasonal", not
            # "seasonal_<period>") when only one period is requested. Normalize
            # both shapes into the same per-period `seasonal_<period>` columns
            # this function always produces, regardless of how many periods
            # were requested.
            if len(resolved_periods) == 1:
                component_frame[f"seasonal_{resolved_periods[0]}"] = np.asarray(fit.seasonal)
            else:
                for p in resolved_periods:
                    component_frame[f"seasonal_{p}"] = np.asarray(fit.seasonal[f"seasonal_{p}"])
            component_frame["resid"] = np.asarray(fit.resid)

        if boxcox:
            component_frame["boxcox_lambda"] = lmbda

        per_series_frames.append(component_frame)

    components = pd.concat(per_series_frames, ignore_index=True)

    # A single scalar boxcox_lambda is only unambiguous when the input had
    # exactly one series -- for multi-series input each series generally
    # has its own fitted lambda, already carried per-row in the
    # `boxcox_lambda` column above.
    overall_lambda = next(iter(boxcox_lambdas.values())) if len(boxcox_lambdas) == 1 else None

    return DecompositionResult(components=components, method=method, boxcox_lambda=overall_lambda)
