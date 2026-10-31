"""First-party temporal / time-series column profiling for ``dscraft.eda``.

This internal module profiles temporal columns from scratch on ``numpy`` -- the deferred LazyEDA
"time-series-specific EDA" capability (see the architecture doc's LazyEDA module entry and
``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``). ``LazyEDA`` previously gave
``"temporal"`` columns only a schema/null summary; this adds the temporal-aware descriptors an
analyst wants for a date/timestamp column:

- **Range coverage:** earliest and latest timestamp, and the total span.
- **Cadence:** the dominant spacing between consecutive (sorted, de-duplicated) timestamps -- the
  median inter-arrival gap -- plus the min/max gap and a regularity ratio that says how close to
  evenly-spaced the series is (1.0 = perfectly regular).
- **Gap detection:** the count and total duration of "large" gaps (runs longer than a multiple of
  the median cadence), which flag missing periods in an otherwise-regular series.
- **Monotonicity / duplicates:** whether timestamps are sorted and how many duplicate timestamps
  there are -- both common data-quality issues for event/log columns.

Per the locked architecture decision (§2.9), this is built **standalone** -- it does not import
from or depend on ``dscraft.forecast``, even though time-series forecasting is the obvious future
consumer. It operates on timestamps already converted to numeric epoch values (seconds), which
``LazyEDA`` derives from a Polars temporal column; the unit is carried on the result so durations
can be reported meaningfully.

For value-bearing time series (a numeric column indexed by time), :func:`autocorrelation` computes
the sample autocorrelation function up to a given lag -- the single most useful signal for spotting
serial dependence or seasonality -- from scratch via the standard biased ACF estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "TemporalDescriptor",
    "temporal_descriptor",
    "autocorrelation",
]


@dataclass(frozen=True)
class TemporalDescriptor:
    """Temporal-coverage and cadence summary for a datetime column.

    Timestamps are summarized as numeric epoch seconds; ``*_seconds`` fields are durations in
    seconds. ``regularity`` is in ``[0, 1]`` (1 = perfectly evenly spaced). Fields that need at
    least two distinct timestamps to define are ``None`` for a degenerate column.
    """

    count: int
    distinct: int
    n_duplicates: int
    is_sorted: bool
    min_epoch: float
    max_epoch: float
    span_seconds: float
    median_gap_seconds: Optional[float]
    min_gap_seconds: Optional[float]
    max_gap_seconds: Optional[float]
    regularity: Optional[float]
    n_large_gaps: int
    large_gap_total_seconds: float


def temporal_descriptor(epoch_seconds, *, large_gap_factor: float = 3.0) -> TemporalDescriptor:
    """Profile a temporal column given its values as numeric epoch seconds.

    Args:
        epoch_seconds: 1-D array-like of timestamps expressed as epoch seconds (floats). Must be
            non-empty and finite (the caller drops nulls first).
        large_gap_factor: a gap between consecutive distinct timestamps counts as "large" when it
            exceeds ``large_gap_factor * median_gap``. ``3.0`` flags clearly-missing periods
            without tripping on ordinary jitter.

    Returns:
        A :class:`TemporalDescriptor`.

    Raises:
        ValueError: if ``epoch_seconds`` is empty or ``large_gap_factor <= 0``.
    """
    if large_gap_factor <= 0:
        raise ValueError(f"large_gap_factor must be positive, got {large_gap_factor!r}.")
    arr = np.asarray(epoch_seconds, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("temporal_descriptor requires at least one timestamp.")

    is_sorted = bool(np.all(np.diff(arr) >= 0)) if n > 1 else True
    ordered = np.sort(arr)
    distinct_values = np.unique(ordered)
    distinct = distinct_values.shape[0]
    n_duplicates = n - distinct

    min_epoch = float(ordered[0])
    max_epoch = float(ordered[-1])
    span = max_epoch - min_epoch

    median_gap = min_gap = max_gap = None
    regularity = None
    n_large_gaps = 0
    large_gap_total = 0.0

    if distinct >= 2:
        gaps = np.diff(distinct_values)
        median_gap = float(np.median(gaps))
        min_gap = float(gaps.min())
        max_gap = float(gaps.max())
        # Regularity: how tightly gaps cluster around the median (1 = all equal).
        if median_gap > 0:
            regularity = float(np.clip(1.0 - (np.std(gaps) / median_gap), 0.0, 1.0))
            threshold = large_gap_factor * median_gap
            large = gaps[gaps > threshold]
            n_large_gaps = int(large.shape[0])
            large_gap_total = float(large.sum())
        else:
            regularity = 1.0

    return TemporalDescriptor(
        count=n,
        distinct=distinct,
        n_duplicates=n_duplicates,
        is_sorted=is_sorted,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        span_seconds=span,
        median_gap_seconds=median_gap,
        min_gap_seconds=min_gap,
        max_gap_seconds=max_gap,
        regularity=regularity,
        n_large_gaps=n_large_gaps,
        large_gap_total_seconds=large_gap_total,
    )


def autocorrelation(values, *, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelation function (ACF) of a value series, lags ``0 .. max_lag``.

    Uses the standard biased ACF estimator
    ``r_k = sum_{t} (x_t - xbar)(x_{t+k} - xbar) / sum_t (x_t - xbar)^2`` -- the same normalization
    used throughout time-series analysis, which guarantees ``r_0 == 1`` and keeps the estimates in
    ``[-1, 1]``. Intended for a numeric column that is ordered in time (the caller is responsible
    for passing values already in temporal order).

    Args:
        values: 1-D array-like of finite numeric observations in time order.
        max_lag: largest lag to compute (capped at ``n - 1``). Must be ``>= 1``.

    Returns:
        A ``(min(max_lag, n-1) + 1,)`` array ``[r_0, r_1, ..., r_L]`` with ``r_0 == 1``.

    Raises:
        ValueError: if ``values`` has fewer than 2 points, is constant (zero variance, ACF
            undefined), or ``max_lag < 1``.
    """
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag!r}.")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n < 2:
        raise ValueError("autocorrelation requires at least 2 observations.")
    centered = arr - arr.mean()
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        raise ValueError("autocorrelation is undefined for a constant series (zero variance).")
    lags = min(max_lag, n - 1)
    acf = np.empty(lags + 1, dtype=np.float64)
    for k in range(lags + 1):
        acf[k] = float(np.sum(centered[: n - k] * centered[k:]) / denom)
    return acf
