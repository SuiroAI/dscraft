"""Correlation/association metrics across continuous, categorical, and mixed column pairs.

This module implements the classical, well-known association-metric suite
described in the architecture doc's EDA research as a "Phik-equivalent"
mixed-type correlation matrix, without attempting to reimplement the actual
`phik <https://github.com/KaveIO/PhiK>`_ library's specific algorithm (a
more statistically sophisticated Pearson-correlation-like formulation built
on maximum-likelihood contingency-table interpolation). That would require
licensing/algorithm-fidelity research beyond what is practical here.
Instead, this module routes each column pair to the classical formula that
already matches its combination of dtypes -- Pearson/Spearman/Kendall for
continuous-continuous, Cramer's V for categorical-categorical, and the
correlation ratio (eta) for mixed pairs -- and assembles those into one
pairwise matrix via :func:`mixed_type_association_matrix`. That function is
deliberately **not** named ``phik_matrix`` to avoid implying algorithmic
equivalence with the real Phik method.

Formula sources (implemented from scratch except where SciPy already
provides the exact well-known statistic directly):

- Continuous-continuous: thin wrappers around ``scipy.stats.pearsonr``,
  ``scipy.stats.spearmanr``, and ``scipy.stats.kendalltau`` -- see
  :func:`continuous_correlation`. Not reimplemented; SciPy's own
  implementations are the canonical ones.
- Cramer's V: built from ``scipy.stats.contingency.crosstab`` (contingency
  table) and ``scipy.stats.chi2_contingency`` (chi-squared statistic),
  combined via the standard formula
  ``V = sqrt(chi2 / (n * min(r - 1, c - 1)))`` -- see :func:`cramers_v`.
  Neither ``crosstab``/``chi2_contingency`` output *is* Cramer's V; that
  combination step is this module's own code.
- Theil's U (uncertainty coefficient): a from-scratch Shannon-entropy
  computation over the empirical joint/marginal distributions of the same
  contingency table -- SciPy has no direct equivalent. See
  :func:`theils_u`.
- Correlation ratio (eta): a from-scratch between-group/total
  sum-of-squares computation -- SciPy has no direct equivalent. See
  :func:`correlation_ratio`.

**NaN policy.** Every function in this module uses pairwise-complete-case
deletion: rows where *either* of the two columns being compared is missing
are dropped before the metric is computed, rather than letting NaN
propagate silently into a nonsensical statistic (e.g. ``pearsonr`` on an
array containing NaN returns ``nan`` with no explanation of why). "Missing"
means ``numpy.nan``/``float('nan')`` for numeric dtypes and ``None`` or a
NaN float for object-dtype categorical arrays -- see
:func:`_categorical_missing_mask`.

**Performance.** This module is deliberately single-threaded. SciPy/NumPy
already internally vectorize and BLAS-parallelize the underlying linear
algebra; wrapping an outer ``multiprocessing``/``joblib`` process pool
around calls into those already-parallel primitives causes thread
oversubscription (nested contention that can make things *slower*, not
faster). :func:`mixed_type_association_matrix` iterates over column pairs
with a plain Python loop for exactly this reason -- do not add
``multiprocessing``/``joblib``/``concurrent.futures`` to this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.stats.contingency import crosstab

__all__ = [
    "CorrelationResult",
    "ColumnKind",
    "AssociationMatrixResult",
    "continuous_correlation",
    "cramers_v",
    "theils_u",
    "correlation_ratio",
    "mixed_type_association_matrix",
]

#: The three continuous-continuous methods :func:`continuous_correlation`
#: supports, each a direct wrapper around the like-named ``scipy.stats``
#: function.
_CONTINUOUS_METHODS = {
    "pearson": stats.pearsonr,
    "spearman": stats.spearmanr,
    "kendall": stats.kendalltau,
}

#: Coarse kind used by :func:`mixed_type_association_matrix` to decide which
#: metric applies to a given column pair. Not a replacement for a column's
#: exact dtype -- just the binary distinction the routing logic needs.
ColumnKind = Literal["continuous", "categorical"]


# ---------------------------------------------------------------------------
# Shared validation / NaN-handling helpers
# ---------------------------------------------------------------------------


def _as_1d_array(values, name: str, *, dtype=None) -> np.ndarray:
    """Convert ``values`` to a 1D ``numpy.ndarray``, rejecting non-1D shapes."""
    arr = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like, got shape {arr.shape!r}.")
    return arr


def _require_same_length(x: np.ndarray, y: np.ndarray, name_x: str, name_y: str) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{name_x} and {name_y} must have the same length, got "
            f"{x.shape[0]} and {y.shape[0]}."
        )


def _to_continuous_array(values, name: str) -> np.ndarray:
    """Validate ``values`` as a 1D array and coerce it to ``float64``.

    Raises a clear ``ValueError`` (rather than letting a cryptic NumPy
    ``TypeError`` propagate) if the values cannot be interpreted as
    floating-point numbers -- e.g. a genuinely categorical/string column
    passed where a continuous one is expected.
    """
    arr = _as_1d_array(values, name)
    try:
        arr = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be convertible to float64 for a continuous correlation "
            f"measure, got dtype {arr.dtype!r}."
        ) from exc
    return arr


def _pairwise_complete_continuous(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where either ``x`` or ``y`` is NaN (pairwise-complete-case)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def _categorical_missing_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of "missing" entries in a categorical array.

    - Float-dtype arrays: missing means ``numpy.isnan``.
    - Integer/unsigned/boolean-dtype arrays: these dtypes cannot represent
      NaN at all, so nothing is missing.
    - Object-dtype arrays (the common case for string/mixed categorical
      data): an element is missing if it is ``None`` or a NaN float (e.g.
      what a pandas column with dtype ``object`` typically holds for a
      missing cell).
    """
    if arr.dtype.kind == "f":
        return np.isnan(arr)
    if arr.dtype.kind in "iub":
        return np.zeros(arr.shape, dtype=bool)
    mask = np.empty(arr.shape, dtype=bool)
    for idx, value in enumerate(arr):
        mask[idx] = value is None or (isinstance(value, float) and np.isnan(value))
    return mask


def _pairwise_complete_categorical(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where either categorical array is missing (pairwise-complete-case)."""
    mask = ~(_categorical_missing_mask(x) | _categorical_missing_mask(y))
    return x[mask], y[mask]


def _require_min_categories(arr: np.ndarray, name: str) -> None:
    if arr.size == 0:
        raise ValueError(
            f"{name} has no rows remaining after dropping missing values; cannot "
            "compute a categorical association measure."
        )
    unique = np.unique(arr)
    if unique.size < 2:
        raise ValueError(
            f"{name} must contain at least 2 unique categories (after dropping "
            f"missing values) for a correlation/association measure to be defined; "
            f"got {unique.size}."
        )


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy (natural-log/nats) of a discrete probability vector.

    Zero-probability entries are excluded from the sum rather than computed
    as ``0 * log(0)`` (which is mathematically defined as ``0`` in the
    entropy limit but is ``nan`` if evaluated literally in floating point).
    The base of the logarithm (nats here, via ``numpy.log``, rather than
    bits via ``numpy.log2``) is an arbitrary but consistent choice -- it
    cancels out entirely in :func:`theils_u`'s ``(H(X) - H(X|Y)) / H(X)``
    ratio, so the choice of base has no effect on Theil's U itself.
    """
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def _contingency_table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build the ``(r, c)`` contingency (count) table for two categorical arrays."""
    return crosstab(x, y).count.astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Continuous-continuous
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationResult:
    """Result of a continuous-continuous correlation test.

    Mirrors the ``.statistic``/``.pvalue``-style result object SciPy's own
    ``pearsonr``/``spearmanr``/``kendalltau`` already return, under the
    stable field names ``statistic``/``p_value`` (SciPy has used both plain
    2-tuples and named result objects across versions for these functions;
    wrapping the result in this dataclass means callers of
    :func:`continuous_correlation` never have to branch on which shape they
    got back).
    """

    statistic: float
    p_value: float
    method: str
    n: int


def continuous_correlation(
    x, y, *, method: Literal["pearson", "spearman", "kendall"] = "pearson"
) -> CorrelationResult:
    """Continuous-continuous correlation between ``x`` and ``y``.

    A thin, consistent wrapper around ``scipy.stats.pearsonr`` (linear
    correlation), ``scipy.stats.spearmanr`` (rank/monotonic correlation), or
    ``scipy.stats.kendalltau`` (rank concordance) -- SciPy's own
    implementations are used directly and are not reimplemented here.

    NaN policy: pairwise-complete-case. Rows where either ``x`` or ``y`` is
    NaN are dropped before the correlation is computed.

    Args:
        x: a 1D array-like of numeric (or numeric-convertible) values.
        y: a 1D array-like of numeric (or numeric-convertible) values, the
            same length as ``x``.
        method: one of ``"pearson"``, ``"spearman"``, ``"kendall"``.

    Returns:
        A :class:`CorrelationResult`.

    Raises:
        ValueError: if ``x``/``y`` are not 1D, have mismatched lengths, are
            not numeric-convertible, fewer than 2 rows remain after
            dropping pairwise-missing values, or ``method`` is not one of
            the three supported values.
    """
    if method not in _CONTINUOUS_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_CONTINUOUS_METHODS)!r}, got {method!r}."
        )
    x_arr = _to_continuous_array(x, "x")
    y_arr = _to_continuous_array(y, "y")
    _require_same_length(x_arr, y_arr, "x", "y")
    x_clean, y_clean = _pairwise_complete_continuous(x_arr, y_arr)
    if x_clean.size < 2:
        raise ValueError(
            "At least 2 rows must remain after dropping pairwise-missing values "
            f"to compute a correlation; got {x_clean.size}."
        )

    result = _CONTINUOUS_METHODS[method](x_clean, y_clean)
    statistic = float(getattr(result, "statistic", result[0]))
    p_value = float(getattr(result, "pvalue", result[1]))
    return CorrelationResult(statistic=statistic, p_value=p_value, method=method, n=x_clean.size)


# ---------------------------------------------------------------------------
# 2. Categorical-categorical
# ---------------------------------------------------------------------------


def cramers_v(x, y, *, bias_correction: bool = False) -> float:
    """Cramer's V association between two categorical arrays.

    Computed from a contingency table (``scipy.stats.contingency.crosstab``)
    and its chi-squared statistic (``scipy.stats.chi2_contingency``, with
    Yates' continuity correction explicitly disabled -- that correction
    exists to make a chi-squared *hypothesis test* more conservative for 2x2
    tables, and applying it here would bias Cramer's V's *magnitude* itself
    downward, which is not what this function is being asked for), combined
    via the standard formula::

        V = sqrt(chi2 / (n * min(r - 1, c - 1)))

    where ``n`` is the total sample count and ``r``/``c`` are the number of
    rows/columns in the contingency table. ``V`` is in ``[0, 1]``: ``0``
    means no association, ``1`` means a perfect association (each category
    of one variable maps to exactly one category of the other).

    ``bias_correction=True`` applies the well-known Bergsma (2013)
    bias-corrected variant, which shrinks both the chi-squared-derived phi^2
    term and the effective ``r``/``c`` toward removing the small-sample
    upward bias plain Cramer's V has (it tends to overstate association for
    small ``n`` or large ``r``/``c`` relative to ``n``)::

        phi2_corrected = max(0, phi2 - (r - 1)(c - 1) / (n - 1))
        r_corrected = r - (r - 1)^2 / (n - 1)
        c_corrected = c - (c - 1)^2 / (n - 1)
        V_corrected = sqrt(phi2_corrected / min(r_corrected - 1, c_corrected - 1))

    Defaults to ``False`` (the plain, uncorrected statistic) since that is
    the more commonly expected value; pass ``bias_correction=True`` for the
    small-sample-adjusted variant.

    NaN policy: pairwise-complete-case. Rows where either ``x`` or ``y`` is
    missing (see :func:`_categorical_missing_mask`) are dropped first.

    Args:
        x: a 1D categorical array-like.
        y: a 1D categorical array-like, the same length as ``x``.
        bias_correction: if ``True``, apply the Bergsma bias-corrected
            variant described above instead of the plain formula.

    Returns:
        Cramer's V, a float in ``[0, 1]``.

    Raises:
        ValueError: if ``x``/``y`` are not 1D, have mismatched lengths, or
            either has fewer than 2 unique categories after dropping
            pairwise-missing values.
    """
    x_arr = _as_1d_array(x, "x", dtype=object)
    y_arr = _as_1d_array(y, "y", dtype=object)
    _require_same_length(x_arr, y_arr, "x", "y")
    x_clean, y_clean = _pairwise_complete_categorical(x_arr, y_arr)
    _require_min_categories(x_clean, "x")
    _require_min_categories(y_clean, "y")

    table = _contingency_table(x_clean, y_clean)
    n = table.sum()
    r, c = table.shape
    chi2, _p_value, _dof, _expected = stats.chi2_contingency(table, correction=False)
    phi2 = chi2 / n

    if not bias_correction:
        denom = min(r - 1, c - 1)
        return float(np.sqrt(phi2 / denom))

    phi2_corrected = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    c_corrected = c - ((c - 1) ** 2) / (n - 1)
    denom_corrected = min(r_corrected - 1, c_corrected - 1)
    if denom_corrected <= 0:
        raise ValueError(
            "Bias-corrected Cramer's V is undefined for this sample size/table "
            "shape (corrected denominator <= 0); use bias_correction=False or a "
            "larger sample."
        )
    return float(np.sqrt(phi2_corrected / denom_corrected))


def theils_u(x, y) -> float:
    """Theil's U (uncertainty coefficient), ``U(X|Y)`` -- asymmetric.

    Measures the fraction of ``x``'s (Shannon) entropy that is "explained
    away" by knowing ``y``::

        U(X|Y) = (H(X) - H(X|Y)) / H(X)

    where ``H(X)`` is the marginal entropy of ``x`` and ``H(X|Y)`` is the
    conditional entropy of ``x`` given ``y``, both computed from the
    empirical joint distribution of the contingency table built from
    ``x``/``y``. Entropy is computed in nats (natural log, via
    :func:`_shannon_entropy`) -- the logarithm base cancels out of the
    ``U`` ratio itself, so this choice does not affect the returned value.

    **Argument order matters -- this is call:** ``theils_u(x, y)`` returns
    ``U(X|Y)``, i.e. "how much does knowing ``y`` reduce uncertainty about
    ``x``". It is asymmetric: ``theils_u(x, y) != theils_u(y, x)`` in
    general. If ``y`` perfectly determines ``x`` (every ``y`` category maps
    to exactly one ``x`` category), ``theils_u(x, y)`` is ``1.0`` (or very
    close to it), even if the reverse is not true. To get the other
    direction, call ``theils_u(y, x)`` explicitly.

    ``U`` is in ``[0, 1]``: ``0`` means ``y`` gives no information about
    ``x``, ``1`` means ``y`` fully determines ``x``.

    NaN policy: pairwise-complete-case, same as :func:`cramers_v`.

    Args:
        x: a 1D categorical array-like -- the variable whose uncertainty is
            being measured/predicted.
        y: a 1D categorical array-like, the same length as ``x`` -- the
            variable being conditioned on / used to predict ``x``.

    Returns:
        ``U(X|Y)``, a float in ``[0, 1]`` (clipped to that range to absorb
        floating-point rounding noise near the boundaries).

    Raises:
        ValueError: if ``x``/``y`` are not 1D, have mismatched lengths, or
            either has fewer than 2 unique categories after dropping
            pairwise-missing values.
    """
    x_arr = _as_1d_array(x, "x", dtype=object)
    y_arr = _as_1d_array(y, "y", dtype=object)
    _require_same_length(x_arr, y_arr, "x", "y")
    x_clean, y_clean = _pairwise_complete_categorical(x_arr, y_arr)
    _require_min_categories(x_clean, "x")
    _require_min_categories(y_clean, "y")

    table = _contingency_table(x_clean, y_clean)
    n = table.sum()
    p_xy = table / n
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    h_x = _shannon_entropy(p_x)
    h_xy = _shannon_entropy(p_xy.ravel())
    h_y = _shannon_entropy(p_y)
    h_x_given_y = h_xy - h_y

    if h_x == 0.0:
        # Only possible if x has a single category after NaN-dropping,
        # which _require_min_categories already rejects above -- defensive
        # guard against a divide-by-zero, not expected to trigger.
        raise ValueError("Theil's U is undefined when x has zero entropy.")

    u = (h_x - h_x_given_y) / h_x
    return float(np.clip(u, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Mixed (continuous vs. categorical)
# ---------------------------------------------------------------------------


def correlation_ratio(categories, values) -> float:
    """Correlation ratio (eta) between a categorical grouping variable and a continuous variable.

    ``eta^2`` is the fraction of ``values``'s total variance explained by
    group membership::

        eta^2 = sum_i n_i * (mean_i - grand_mean)^2 / sum_j (values_j - grand_mean)^2

    where ``n_i`` is the number of observations in category ``i``,
    ``mean_i`` is the sample mean of ``values`` within category ``i``, and
    ``grand_mean`` is the overall mean of ``values``. Returns
    ``eta = sqrt(eta^2)``, in ``[0, 1]``: ``0`` means group membership
    explains none of ``values``'s variance (all group means equal),
    ``1`` means it explains all of it (zero within-group variance).

    NaN policy: pairwise-complete-case. Rows where ``categories`` is missing
    (see :func:`_categorical_missing_mask`) or ``values`` is NaN are dropped
    first.

    Args:
        categories: a 1D categorical array-like (the grouping variable).
        values: a 1D array-like of numeric (or numeric-convertible) values,
            the same length as ``categories``.

    Returns:
        eta, a float in ``[0, 1]``.

    Raises:
        ValueError: if the inputs are not 1D, have mismatched lengths,
            ``categories`` has fewer than 2 unique categories after
            dropping pairwise-missing rows, or ``values`` has zero variance
            (undefined -- there is nothing for group membership to
            explain).
    """
    cat_arr = _as_1d_array(categories, "categories", dtype=object)
    val_arr = _to_continuous_array(values, "values")
    _require_same_length(cat_arr, val_arr, "categories", "values")

    missing = _categorical_missing_mask(cat_arr) | np.isnan(val_arr)
    cat_clean = cat_arr[~missing]
    val_clean = val_arr[~missing]
    _require_min_categories(cat_clean, "categories")

    grand_mean = val_clean.mean()
    ss_total = float(np.sum((val_clean - grand_mean) ** 2))
    if ss_total == 0.0:
        raise ValueError(
            "Correlation ratio is undefined when values has zero variance "
            "(after dropping pairwise-missing rows)."
        )

    ss_between = 0.0
    for category in np.unique(cat_clean):
        group_values = val_clean[cat_clean == category]
        group_mean = group_values.mean()
        ss_between += group_values.size * (group_mean - grand_mean) ** 2

    eta_squared = float(np.clip(ss_between / ss_total, 0.0, 1.0))
    return float(np.sqrt(eta_squared))


# ---------------------------------------------------------------------------
# Mixed-type association matrix
# ---------------------------------------------------------------------------


def _infer_column_kind(array: np.ndarray) -> ColumnKind:
    """Infer whether a column is continuous or categorical from its dtype.

    Integer/unsigned/float dtypes are treated as continuous. Everything
    else -- boolean, object (the common case for strings/mixed Python
    objects), string/unicode -- is treated as categorical. Booleans are
    categorical (only two possible values, more naturally an association
    target than a continuous quantity), matching the same convention
    ``dscraft.eda.engine`` uses to keep booleans out of its ``"numeric"``
    category. Callers can override this inference per-column via
    :func:`mixed_type_association_matrix`'s ``column_kinds`` argument --
    e.g. for an integer column that is actually a categorical code.
    """
    if array.dtype.kind in "iuf":
        return "continuous"
    return "categorical"


def _extract_column(data, name: str) -> np.ndarray:
    """Pull one named column out of a DataFrame-like or mapping-like ``data``.

    Duck-typed rather than importing pandas/polars: if the column object
    has a ``to_numpy`` method (pandas ``Series``, polars ``Series``), that
    is used; otherwise it is passed straight to ``numpy.asarray``. This
    keeps this module import-safe without pandas or polars installed.
    """
    column = data[name]
    to_numpy = getattr(column, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(column)


def _resolve_column_names(data, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    frame_columns = getattr(data, "columns", None)
    if frame_columns is not None:
        return list(frame_columns)
    if isinstance(data, Mapping):
        return list(data.keys())
    raise TypeError(
        "Could not determine column names from data; pass `columns` explicitly, "
        "or provide a DataFrame-like object with a `.columns` attribute or a "
        "Mapping[str, array-like]."
    )


@dataclass
class AssociationMatrixResult:
    """Result of :func:`mixed_type_association_matrix`.

    ``matrix`` is a plain ``(n, n)`` ``numpy.ndarray`` rather than a
    ``pandas.DataFrame`` so this module never requires pandas to be
    installed; wrap it yourself with
    ``pandas.DataFrame(result.matrix, index=result.columns, columns=result.columns)``
    if a labeled frame is more convenient.

    ``symmetric`` is ``True`` unless ``categorical_metric="theils_u"`` was
    requested and at least one categorical-categorical pair was present --
    Theil's U is asymmetric (see :func:`theils_u`), so in that case
    ``matrix[i, j]`` and ``matrix[j, i]`` for a categorical-categorical pair
    are *not* equal, and each holds a different, independently meaningful
    value: ``matrix[i, j] = U(column_i | column_j)``. Every other metric
    used by this function (Pearson/Spearman/Kendall, Cramer's V, the
    correlation ratio) is symmetric, so those cells are always mirrored.

    ``unavailable_pairs`` records column pairs whose metric raised a
    ``ValueError`` (e.g. a categorical column pair whose pairwise-complete
    subset happened to collapse to a single category) -- a third, explicit
    category distinct from "computed and near 0" or "computed and near 1",
    matching this repo's convention (see
    ``dscraft.clean.dedup.DedupReport.zero_vector_row_indices``) of never
    silently guessing a value the underlying computation could not
    actually produce. Those cells hold ``numpy.nan`` in ``matrix``.
    """

    columns: list[str]
    column_kinds: dict[str, ColumnKind]
    matrix: np.ndarray
    metrics: list[list[str]]
    symmetric: bool
    unavailable_pairs: dict[tuple[str, str], str] = field(default_factory=dict)

    def get(self, row: str, col: str) -> float:
        """Look up the association value between two columns by name."""
        i = self.columns.index(row)
        j = self.columns.index(col)
        return float(self.matrix[i, j])


def mixed_type_association_matrix(
    data,
    *,
    columns: Sequence[str] | None = None,
    column_kinds: Mapping[str, ColumnKind] | None = None,
    continuous_method: Literal["pearson", "spearman"] = "pearson",
    categorical_metric: Literal["cramers_v", "theils_u"] = "cramers_v",
) -> AssociationMatrixResult:
    """Build a full pairwise association matrix across mixed-dtype columns.

    For every pair of columns, this routes to the appropriate classical
    metric based on each column's inferred (or overridden) kind:

    - continuous-continuous -> :func:`continuous_correlation` (Pearson by
      default, or Spearman via ``continuous_method="spearman"``).
    - categorical-categorical -> :func:`cramers_v` by default, or
      :func:`theils_u` via ``categorical_metric="theils_u"`` (see
      :class:`AssociationMatrixResult`'s ``symmetric`` field for why that
      changes the matrix's symmetry).
    - mixed (one continuous, one categorical) -> :func:`correlation_ratio`.

    Diagonal entries (a column against itself) are always ``1.0`` by
    definition (perfect self-association) and are not computed via the
    pairwise metrics above.

    This is **not** an implementation of the actual Phik algorithm --  see
    the module docstring for why. It is a simpler, well-defined "route each
    pair to the matching classical formula" matrix.

    Performance: this function iterates over column pairs with a plain
    Python loop. It deliberately does not use ``multiprocessing``/
    ``joblib``/``concurrent.futures`` -- see the module docstring.

    Args:
        data: a DataFrame-like object with a ``.columns`` attribute and
            column access via ``data[name]`` (e.g. a ``pandas.DataFrame`` or
            ``polars.DataFrame``), or a ``Mapping[str, array-like]``.
        columns: which columns to include, and in what order. Defaults to
            all of ``data``'s columns (or ``data``'s keys, for a mapping),
            in their existing order.
        column_kinds: optional per-column override of the inferred
            continuous/categorical kind (see :func:`_infer_column_kind`).
            Only columns needing an override must be present; any column
            not in this mapping falls back to dtype-based inference.
        continuous_method: ``"pearson"`` or ``"spearman"``, passed through
            to :func:`continuous_correlation` for continuous-continuous
            pairs.
        categorical_metric: ``"cramers_v"`` (default, symmetric) or
            ``"theils_u"`` (asymmetric -- see
            :class:`AssociationMatrixResult`), used for
            categorical-categorical pairs.

    Returns:
        An :class:`AssociationMatrixResult`.

    Raises:
        ValueError: if fewer than 1 column is selected, or a selected
            column name is missing from ``data``.
        TypeError: if column names cannot be determined from ``data`` and
            ``columns`` was not passed explicitly.
    """
    resolved_columns = _resolve_column_names(data, columns)
    if len(resolved_columns) == 0:
        raise ValueError("At least 1 column must be selected to build an association matrix.")

    arrays: dict[str, np.ndarray] = {}
    kinds: dict[str, ColumnKind] = {}
    for name in resolved_columns:
        try:
            arrays[name] = _extract_column(data, name)
        except KeyError as exc:
            raise ValueError(f"Column {name!r} not found in data.") from exc
        override = column_kinds.get(name) if column_kinds is not None else None
        kinds[name] = override if override is not None else _infer_column_kind(arrays[name])

    n = len(resolved_columns)
    matrix = np.eye(n, dtype=np.float64)
    metric_labels = [["identity" if i == j else "" for j in range(n)] for i in range(n)]
    unavailable_pairs: dict[tuple[str, str], str] = {}
    is_asymmetric_run = False

    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = resolved_columns[i], resolved_columns[j]
            kind_i, kind_j = kinds[name_i], kinds[name_j]
            array_i, array_j = arrays[name_i], arrays[name_j]

            try:
                if kind_i == "continuous" and kind_j == "continuous":
                    value = continuous_correlation(
                        array_i, array_j, method=continuous_method
                    ).statistic
                    metric_labels[i][j] = metric_labels[j][i] = continuous_method
                    matrix[i, j] = matrix[j, i] = value
                elif kind_i == "categorical" and kind_j == "categorical":
                    if categorical_metric == "cramers_v":
                        value = cramers_v(array_i, array_j)
                        metric_labels[i][j] = metric_labels[j][i] = "cramers_v"
                        matrix[i, j] = matrix[j, i] = value
                    else:
                        is_asymmetric_run = True
                        matrix[i, j] = theils_u(array_i, array_j)
                        matrix[j, i] = theils_u(array_j, array_i)
                        metric_labels[i][j] = metric_labels[j][i] = "theils_u"
                else:
                    # Mixed pair: whichever of the two is categorical is the
                    # grouping variable, whichever is continuous is `values`.
                    if kind_i == "categorical":
                        cat_array, cont_array = array_i, array_j
                    else:
                        cat_array, cont_array = array_j, array_i
                    value = correlation_ratio(cat_array, cont_array)
                    metric_labels[i][j] = metric_labels[j][i] = "correlation_ratio"
                    matrix[i, j] = matrix[j, i] = value
            except ValueError as exc:
                matrix[i, j] = matrix[j, i] = np.nan
                metric_labels[i][j] = metric_labels[j][i] = "unavailable"
                unavailable_pairs[(name_i, name_j)] = str(exc)

    return AssociationMatrixResult(
        columns=resolved_columns,
        column_kinds=kinds,
        matrix=matrix,
        metrics=metric_labels,
        symmetric=not is_asymmetric_run,
        unavailable_pairs=unavailable_pairs,
    )
