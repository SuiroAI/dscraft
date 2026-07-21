"""Basic backtest/evaluation helper (architecture doc Part 3, "Module 3: LazyForecast").

Per the task scope, this is deliberately **not** the full leaderboard
machinery described in the architecture doc (Gneiting-Raftery interval
score, PICP, MPIW -- those score *interval* forecasts produced by the
conformal-prediction layer, which is out of scope for this pass). This
module implements the plain, "basic backtest/evaluation report" version:
hold out the last N points of each series, forecast them with
:func:`dscraft.forecast.forecast.forecast`, and report MAE/RMSE per
series and averaged across series.

Per issue #29's adoption decision, the actual MAE/RMSE calculation is
delegated to Nixtla's ``utilsforecast.losses.mae``/``rmse`` rather than
hand-rolled with raw numpy -- ``utilsforecast`` is the shared metrics
layer the rest of the nixtlaverse (StatsForecast/MLForecast/
NeuralForecast/HierarchicalForecast) already assumes, so future backends
added to this subpackage (MLForecast in #21, NeuralForecast in #24,
conformal prediction in #26, hierarchical reconciliation in #27) can all
score against the same canonical metric implementation instead of each
reinventing one. This module still owns the *backtest* orchestration
(train/test split, forecasting, alignment checking, reporting) -- only
the per-series metric arithmetic itself is now delegated.

This is the **one canonical backtest path** in this package -- there is no
parallel/alternate evaluation implementation anywhere else here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from utilsforecast.losses import mae as uf_mae
from utilsforecast.losses import rmse as uf_rmse

from .forecast import ForecastConfig, build_statsforecast, prepare_frame

__all__ = [
    "SeriesMetric",
    "BacktestReport",
    "BacktestAlignmentError",
    "backtest",
]


class BacktestAlignmentError(ValueError):
    """Raised when one or more series have zero overlapping forecast/test dates.

    This happens when the forecasted dates for a series (produced by the
    model over the held-out horizon) don't line up at all with that
    series' actual held-out test dates -- e.g. missing observations,
    irregular frequency, or a forecaster that skips dates. Before this
    check existed, ``backtest()``'s inner join on ``(unique_id, ds)``
    would silently drop such a series from the report entirely, which is
    worse than a loud failure: a metric silently missing is easy to miss,
    a raised exception naming the series is not.
    """


@dataclass
class SeriesMetric:
    """Per-series, per-model backtest error for one held-out window.

    ``n_points`` is how many held-out dates actually had a matching
    forecast (after the inner join on ``(unique_id, ds)``); ``expected_points``
    is how many held-out dates there *should* have been (``test_size``).
    A caller can check ``n_points == expected_points`` to confirm the
    metric was computed over the full held-out window rather than a
    partially-overlapping subset of it (see :func:`backtest`'s docstring).
    """

    unique_id: str
    model: str
    mae: float
    rmse: float
    n_points: int
    expected_points: int


@dataclass
class BacktestReport:
    """Result of :func:`backtest` -- per-series metrics plus overall averages."""

    metrics: list[SeriesMetric] = field(default_factory=list)

    def mean_mae(self, model: str | None = None) -> float:
        """Mean MAE across all (series, model) rows, optionally filtered to one model."""
        values = [m.mae for m in self.metrics if model is None or m.model == model]
        if not values:
            raise ValueError(f"No backtest metrics found for model={model!r}.")
        return float(np.mean(values))

    def mean_rmse(self, model: str | None = None) -> float:
        """Mean RMSE across all (series, model) rows, optionally filtered to one model."""
        values = [m.rmse for m in self.metrics if model is None or m.model == model]
        if not values:
            raise ValueError(f"No backtest metrics found for model={model!r}.")
        return float(np.mean(values))

    def to_frame(self) -> pd.DataFrame:
        """Render the per-series metrics as a plain pandas DataFrame."""
        return pd.DataFrame(
            {
                "unique_id": [m.unique_id for m in self.metrics],
                "model": [m.model for m in self.metrics],
                "mae": [m.mae for m in self.metrics],
                "rmse": [m.rmse for m in self.metrics],
                "n_points": [m.n_points for m in self.metrics],
                "expected_points": [m.expected_points for m in self.metrics],
            }
        )


def backtest(
    data: Any,
    config: ForecastConfig | None = None,
    *,
    test_size: int | None = None,
) -> BacktestReport:
    """Hold out the last ``test_size`` points of each series and score the forecast.

    For each series: the last ``test_size`` observations are held out as the
    test window, the classical model(s) in ``config.models`` are fit on the
    remaining (earlier) observations, a ``test_size``-step-ahead forecast is
    produced, and MAE/RMSE are computed against the held-out actuals.

    Args:
        data: same Tier-1 input accepted by
            :func:`dscraft.forecast.forecast.forecast`.
        config: a :class:`ForecastConfig`. ``config.horizon`` is ignored in
            favor of ``test_size`` (the backtest horizon is exactly the size
            of the held-out window); everything else (models, freq,
            season_length, id/time/value column names) is used as-is.
        test_size: number of trailing points per series to hold out. Defaults
            to ``config.horizon`` if not given.

    Returns:
        A :class:`BacktestReport` with one :class:`SeriesMetric` per
        ``(series, model)`` pair, plus ``mean_mae()``/``mean_rmse()`` helpers
        for the overall average. Each :class:`SeriesMetric` carries both
        ``n_points`` (how many held-out dates actually matched a forecasted
        date) and ``expected_points`` (``test_size``); if a series has
        irregular/missing dates such that the forecast and the held-out
        actuals don't fully align on ``(unique_id, ds)``, ``n_points`` will
        be less than ``expected_points`` and a :class:`UserWarning` is
        emitted naming the affected series.

    Raises:
        ValueError: any series has fewer than ``test_size + 1`` observations
            (nothing left to fit the model on after holding out the test
            window).
        BacktestAlignmentError: after the forecast/test-window join, one or
            more series have *zero* overlapping ``(unique_id, ds)`` pairs
            (e.g. the forecasted dates and the held-out actual dates don't
            align at all -- missing observations, irregular frequency, or a
            forecaster that skips dates). Without this check such a series
            would simply vanish from ``report.metrics`` with no signal.
    """
    config = config or ForecastConfig()
    test_size = test_size if test_size is not None else config.horizon
    if test_size < 1:
        raise ValueError("test_size must be >= 1.")

    prepared = prepare_frame(data, config)

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for unique_id, group in prepared.groupby("unique_id", sort=False):
        group = group.sort_values("ds")
        if len(group) < test_size + 1:
            raise ValueError(
                f"Series {unique_id!r} has only {len(group)} observations, "
                f"which is not enough to hold out test_size={test_size} points "
                "and still have at least one training point."
            )
        train_frames.append(group.iloc[:-test_size])
        test_frames.append(group.iloc[-test_size:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    sf = build_statsforecast(config)
    forecasts = sf.forecast(df=train_df, h=test_size).reset_index(drop=True)

    model_names = [name for name in config.models]
    merged = forecasts.merge(test_df, on=["unique_id", "ds"], how="inner", suffixes=("", "_actual"))

    # Every series in the input must have been placed into test_df above, so
    # this is the authoritative set of series we owe a SeriesMetric to. The
    # inner join above silently drops any (unique_id, ds) pairs that don't
    # align between the forecast and the held-out actuals -- surface that
    # instead of letting affected series quietly vanish from the report.
    expected_unique_ids = test_df["unique_id"].unique().tolist()
    overlap_counts = merged.groupby("unique_id", sort=False)["ds"].nunique()

    missing_entirely = [uid for uid in expected_unique_ids if overlap_counts.get(uid, 0) == 0]
    if missing_entirely:
        raise BacktestAlignmentError(
            "The following series have ZERO overlapping dates between their "
            f"forecast and held-out test window: {sorted(missing_entirely)!r}. "
            "This means the forecasted dates and the actual held-out dates "
            "for these series don't align at all (missing observations, "
            "irregular frequency, or a forecaster that skips dates), so no "
            "backtest metric can be computed for them. Fix the underlying "
            "date misalignment (e.g. ensure the series has a fully regular "
            f"{config.freq!r}-frequency index) before calling backtest()."
        )

    partially_misaligned = {
        uid: int(overlap_counts.get(uid, 0))
        for uid in expected_unique_ids
        if 0 < overlap_counts.get(uid, 0) < test_size
    }
    if partially_misaligned:
        warnings.warn(
            "The following series had PARTIAL overlap between their "
            f"forecast and held-out test window (expected test_size={test_size} "
            f"matching dates): {partially_misaligned!r}. Their SeriesMetric.mae/"
            "rmse were computed over fewer than test_size points -- check "
            "SeriesMetric.n_points vs. SeriesMetric.expected_points before "
            "trusting these numbers as full-window estimates.",
            stacklevel=2,
        )

    # Per issue #29: delegate the actual per-series metric arithmetic to
    # utilsforecast's battle-tested mae()/rmse() rather than hand-rolling it
    # with raw numpy -- see this module's docstring for why. Both functions
    # take the same long-format (unique_id, y, <model columns>) shape
    # `merged` is already in, and return one row per unique_id with one
    # column per model holding that metric.
    mae_by_series = uf_mae(merged, models=model_names, id_col="unique_id", target_col="y")
    rmse_by_series = uf_rmse(merged, models=model_names, id_col="unique_id", target_col="y")
    n_points_by_series = merged.groupby("unique_id", sort=False).size()

    metrics: list[SeriesMetric] = []
    for _, mae_row in mae_by_series.iterrows():
        unique_id = mae_row["unique_id"]
        rmse_row = rmse_by_series.loc[rmse_by_series["unique_id"] == unique_id].iloc[0]
        n_points = int(n_points_by_series.loc[unique_id])
        for model_name in model_names:
            metrics.append(
                SeriesMetric(
                    unique_id=str(unique_id),
                    model=model_name,
                    mae=float(mae_row[model_name]),
                    rmse=float(rmse_row[model_name]),
                    n_points=n_points,
                    expected_points=test_size,
                )
            )

    return BacktestReport(metrics=metrics)
