"""First-party univariate distribution descriptors for ``dscraft.eda``.

This internal module computes per-column distribution statistics from scratch on ``numpy`` -- a
first-party EDA *breadth* capability added by the automl/eda rebuild (see
``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``). It deepens ``LazyEDA``'s per-column
profiling beyond the schema/null/cardinality/quantile summary it already produces, adding the
descriptive statistics an analyst reaches for first when looking at a single column:

- **Central tendency / spread:** mean, variance, standard deviation, min, max, range.
- **Shape:** skewness (Fisher, bias-corrected) and excess kurtosis -- how asymmetric and how
  heavy-tailed the distribution is relative to a normal.
- **Robust statistics:** median, median absolute deviation (MAD), and the interquartile range
  (IQR) -- outlier-resistant analogues of mean/std, which matter precisely because real columns
  often contain the outliers that make mean/std misleading.
- **Coefficient of variation:** std / |mean|, a scale-free spread measure (``None`` when the mean
  is ~0, where it is undefined).

For categorical/string columns, a separate :func:`categorical_descriptor` reports the distinct
count, the mode and its frequency, and the normalized Shannon entropy of the value distribution
(0 = one value dominates, 1 = perfectly uniform) -- a compact "how concentrated is this column"
summary.

Everything here operates on already-materialized, finite values (``LazyEDA`` filters NaN/inf
before calling), returns plain dataclasses of Python floats, and uses only ``numpy``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "NumericDescriptor",
    "CategoricalDescriptor",
    "numeric_descriptor",
    "categorical_descriptor",
]


@dataclass(frozen=True)
class NumericDescriptor:
    """Univariate descriptive statistics for a numeric column.

    All fields are plain floats. ``skewness``/``kurtosis``/``coefficient_of_variation`` may be
    ``None`` when undefined (e.g. zero variance, or a near-zero mean for CV).
    """

    count: int
    mean: float
    variance: float
    std: float
    minimum: float
    maximum: float
    value_range: float
    median: float
    mad: float
    iqr: float
    skewness: Optional[float]
    kurtosis: Optional[float]
    coefficient_of_variation: Optional[float]


@dataclass(frozen=True)
class CategoricalDescriptor:
    """Concentration statistics for a categorical/string column."""

    count: int
    distinct: int
    mode: Optional[str]
    mode_frequency: int
    normalized_entropy: float


def numeric_descriptor(values: Sequence[float]) -> NumericDescriptor:
    """Compute :class:`NumericDescriptor` over a sequence of finite numeric ``values``.

    Skewness is the bias-corrected Fisher-Pearson standardized moment coefficient; kurtosis is the
    bias-corrected *excess* kurtosis (0 for a normal). Both are ``None`` for zero-variance data
    (undefined) or when there are too few points for the bias correction (``n < 3`` for skew,
    ``n < 4`` for kurtosis). MAD is the median of absolute deviations from the median; IQR is
    ``p75 - p25`` via linear-interpolation quantiles.

    Raises:
        ValueError: if ``values`` is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("numeric_descriptor requires at least one value.")

    mean = float(arr.mean())
    variance = float(arr.var(ddof=1)) if n > 1 else 0.0
    std = math.sqrt(variance)
    minimum = float(arr.min())
    maximum = float(arr.max())
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    q25, q75 = (float(v) for v in np.quantile(arr, [0.25, 0.75]))
    iqr = q75 - q25

    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    if std > 0:
        z = (arr - mean) / std
        if n >= 3:
            # Bias-corrected Fisher-Pearson standardized skewness.
            skewness = float((n / ((n - 1) * (n - 2))) * np.sum(z**3))
        if n >= 4:
            # Bias-corrected excess kurtosis.
            m4 = np.sum(z**4)
            kurtosis = float(
                ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * m4
                - (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            )

    cv: Optional[float] = None
    if abs(mean) > 1e-12:
        cv = std / abs(mean)

    return NumericDescriptor(
        count=n,
        mean=mean,
        variance=variance,
        std=std,
        minimum=minimum,
        maximum=maximum,
        value_range=maximum - minimum,
        median=median,
        mad=mad,
        iqr=iqr,
        skewness=skewness,
        kurtosis=kurtosis,
        coefficient_of_variation=cv,
    )


def categorical_descriptor(values: Sequence) -> CategoricalDescriptor:
    """Compute :class:`CategoricalDescriptor` over a sequence of categorical ``values``.

    Values are stringified (matching ``LazyEDA``'s categorical-histogram convention) before
    counting. ``normalized_entropy`` is the Shannon entropy of the empirical value distribution
    divided by ``log(distinct)`` -- in ``[0, 1]``, where 0 means one value dominates and 1 means a
    perfectly uniform spread. A single-distinct-value column has entropy 0 by convention (there is
    no uncertainty to normalize).

    Raises:
        ValueError: if ``values`` is empty.
    """
    strings = [str(v) for v in values]
    n = len(strings)
    if n == 0:
        raise ValueError("categorical_descriptor requires at least one value.")
    counter = Counter(strings)
    distinct = len(counter)
    mode, mode_frequency = counter.most_common(1)[0]

    if distinct <= 1:
        normalized_entropy = 0.0
    else:
        probs = np.array([c / n for c in counter.values()], dtype=np.float64)
        entropy = float(-np.sum(probs * np.log(probs)))
        normalized_entropy = entropy / math.log(distinct)

    return CategoricalDescriptor(
        count=n,
        distinct=distinct,
        mode=mode,
        mode_frequency=mode_frequency,
        normalized_entropy=normalized_entropy,
    )
