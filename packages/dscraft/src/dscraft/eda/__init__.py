"""dscraft.eda -- exploratory data analysis: lazy profiling + sketches + associations + an HTML report.

This is the public API surface for the ``dscraft.eda`` subpackage (per
CLAUDE.md's "one canonical location per capability" / "consistent
per-module layout" conventions -- every subpackage exposes its callable
surface from ``__init__.py``, not from a caller having to import internal
submodules directly).

``dscraft.eda`` composes four independently-built, independently-tested
submodules that otherwise have zero knowledge of each other:

- :mod:`dscraft.eda.engine` -- lazy Polars execution: schema, per-column
  null counts, and row count, computed without materializing a source's
  full data.
- :mod:`dscraft.eda.sketches` -- HyperLogLog (HLL) cardinality estimation
  and KLL quantile estimation, both from the Apache DataSketches library.
- :mod:`dscraft.eda.associations` -- a mixed continuous/categorical
  pairwise correlation/association-matrix suite.
- :mod:`dscraft.eda.report` -- a self-contained, single-file HTML/Canvas
  report renderer that only ever consumes already-aggregated,
  plain-dataclass summary data (never a raw DataFrame).

This module is the wiring step that turns those four independently-useful
pieces into one coherent, single-call workflow:

>>> from dscraft.eda import LazyEDA
>>> profile = LazyEDA().profile("orders.parquet")
>>> profile.null_report.columns_with_nulls()
['discount_code']
>>> profile.export("orders_eda_report.html")

**Column routing heuristic (numeric -> quantiles, string/categorical ->
cardinality).** :func:`dscraft.eda.engine.profile_schema` already buckets
every column into a coarse category (``"numeric"``, ``"string"``,
``"boolean"``, ``"temporal"``, ``"other"`` -- see
:data:`dscraft.eda.engine.ColumnCategory`). :meth:`LazyEDA.profile` reuses
that categorization directly rather than re-deriving its own numeric/
categorical split:

- ``"numeric"`` columns get a :func:`dscraft.eda.sketches.estimate_quantiles`
  pass (min/p25/p50/p75/max by default) plus a histogram (see below).
- ``"string"`` columns (Polars ``Utf8``/``String``/``Categorical``/``Enum``)
  get a :func:`dscraft.eda.sketches.estimate_cardinality` pass (an
  estimated distinct-value count) plus a top-K value-frequency histogram.
- ``"boolean"``/``"temporal"``/``"other"`` columns get neither sketch --
  only their schema/null summary is reported. This is a deliberate v1
  scope line, not an oversight: boolean columns have a trivially small
  (<=2) cardinality that HLL brings no value over an exact count for, and
  temporal-column-specific summarization (date-range histograms, gap
  detection) is exactly the kind of time-series-aware EDA
  ``dscraft.forecast`` would eventually want -- see the architecture doc's
  LazyEDA module entry for why that's deferred, not built here.

**Materialization strategy: one ``.collect()`` for the whole frame, not
one per column.** :func:`dscraft.eda.engine.profile_engine` builds the
schema/null reports from the *lazy* plan (a single aggregation query, per
that module's own docstring) without ever materializing row data. Sketch
computation, however, fundamentally requires iterating real values (HLL/
KLL are streaming algorithms fed one value at a time via ``.update()``),
and the association matrix needs every column's actual values to compute
pairwise statistics -- there is no lazy-query shortcut for either. Rather
than issue one ``.select(col).collect()`` per column (n round trips
through Polars' query engine for an n-column frame) or resort to a
partial/sampled read, this module collects the full frame into memory
**exactly once** (:meth:`polars.LazyFrame.collect`) and reuses that single
eager :class:`polars.DataFrame` for every column's sketch *and* for the
association matrix. This is the simplest correct v1 approach for
"exploratory" profiling of datasets that fit in memory (the reference
hardware's headline spec is 128GB unified memory, per CLAUDE.md) -- a
future version could sample large columns instead of materializing them
in full, but that changes the accuracy semantics of the sketches
themselves (an HLL/KLL sketch fed a sample estimates the *sample's*
distinct-count/quantiles, not the full column's) and is left as a v2
concern, not addressed by this pass.

**Histograms: derived directly from the same materialized column values,
not from the KLL sketch's own quantile output.** ``report.py``'s
:class:`~dscraft.eda.report.ColumnSummary.histogram` field expects
pre-binned ``(label, count)`` pairs. Since this module already
materializes every numeric/string column's full value list to feed the
HLL/KLL sketches (see above), the simplest and most accurate route to a
histogram is to bin that same already-in-memory list directly --
``numpy.histogram`` (equal-width bins) for numeric columns, an exact
:class:`collections.Counter` top-K pass for string/categorical columns --
rather than attempting to reconstruct approximate bin edges from the KLL
sketch's few quantile estimates (which would both be less accurate than
using the real values already in hand, and require the KLL sketch to
report far more quantiles than :meth:`LazyEDA.profile` actually needs for
its stat table). See :func:`_numeric_histogram`/:func:`_categorical_histogram`.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

from dscraft.eda.associations import (
    AssociationMatrixResult,
    mixed_type_association_matrix,
)
from dscraft.eda.engine import (
    ColumnCategory,
    ColumnSchema,
    EngineProfile,
    NullReport,
    SchemaReport,
    profile_engine,
)
from dscraft.eda.plots import (
    PlotnineExtraNotInstalledError,
    association_heatmap,
    column_distribution,
)
from dscraft.eda.report import (
    AssociationMatrix,
    ColumnSummary,
    EDAReportData,
    HistogramBin,
    export_report,
)
from dscraft.eda.sketches import (
    HLLResult,
    KLLResult,
    estimate_cardinality,
    estimate_quantiles,
)

__all__ = [
    "EDAProfile",
    "LazyEDA",
    "PlotnineExtraNotInstalledError",
    "association_heatmap",
    "column_distribution",
]

#: Default quantiles requested per numeric column: min, p25, median, p75,
#: max. Matches the label mapping in :data:`_QUANTILE_LABELS` below.
_DEFAULT_QUANTILES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Display labels for the default quantile set, keyed by the quantile
#: value itself. A quantile outside this mapping (only possible if a
#: caller constructs :class:`LazyEDA` with a custom ``quantiles`` sequence)
#: falls back to a generated ``"pNN"``-style label -- see
#: :meth:`LazyEDA.profile`.
_QUANTILE_LABELS: dict[float, str] = {0.0: "min", 0.25: "p25", 0.5: "p50", 0.75: "p75", 1.0: "max"}

#: Column categories (per :data:`dscraft.eda.engine.ColumnCategory`) that
#: receive a quantile/histogram numeric-sketch pass.
_NUMERIC_CATEGORY: ColumnCategory = "numeric"

#: Column categories that receive a cardinality/top-K-histogram
#: categorical-sketch pass.
_STRING_CATEGORY: ColumnCategory = "string"


def _quantile_label(q: float) -> str:
    """Map a quantile value to its display label, e.g. ``0.5 -> "p50"``."""
    if q in _QUANTILE_LABELS:
        return _QUANTILE_LABELS[q]
    return f"p{q * 100:g}"


def _numeric_histogram(values: Sequence[float], bins: int) -> list[HistogramBin]:
    """Equal-width histogram over already-materialized numeric ``values``.

    Uses ``numpy.histogram`` directly on the real values (already in hand
    -- see the module docstring's "Histograms" section for why this is
    preferred over deriving approximate bins from the KLL sketch's
    quantile output). The final bin's label is inclusive on both ends
    (``"[x, y]"``) to match ``numpy.histogram``'s own convention that the
    last bin includes its right edge; every other bin is half-open
    (``"[x, y)"``).

    Returns an empty list if ``values`` is empty -- callers are expected
    to guard against this themselves (see :meth:`LazyEDA.profile`), but an
    empty list here is a safe, unsurprising fallback rather than an error.
    """
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(array, bins=bins)
    histogram: list[HistogramBin] = []
    last_index = len(counts) - 1
    for index, count in enumerate(counts):
        low, high = edges[index], edges[index + 1]
        closing = "]" if index == last_index else ")"
        label = f"[{low:.4g}, {high:.4g}{closing}"
        histogram.append(HistogramBin(label=label, count=int(count)))
    return histogram


def _categorical_histogram(values: Sequence, top_k: int) -> list[HistogramBin]:
    """Exact top-``top_k`` value-frequency histogram over ``values``.

    Every value is stringified via ``str()`` before counting -- matching
    :func:`dscraft.eda.sketches.estimate_cardinality`'s own str()-fallback
    convention for non-``int``/``float``/``str`` items -- so a histogram
    label is always a plain, displayable string regardless of the
    column's underlying Python value types.

    Unlike :func:`_numeric_histogram`'s equal-width bins (which must cover
    the whole numeric range), this is an *exact* count over the real data
    for the ``top_k`` most frequent distinct values, not an approximation
    from the HLL sketch (HLL estimates only a total distinct count, not
    per-value frequencies -- there is no sketch-derived shortcut here).

    Returns an empty list if ``values`` is empty.
    """
    if not values:
        return []
    counter = Counter(str(value) for value in values)
    return [
        HistogramBin(label=label, count=count) for label, count in counter.most_common(top_k)
    ]


class EDAProfile:
    """The composed result of :meth:`LazyEDA.profile`.

    Bundles every intermediate result produced while building the report
    (schema, nulls, per-column sketches, the association matrix) as plain,
    inspectable attributes -- not just the ability to render/export an
    HTML report -- matching this platform's established convention
    (``dscraft.clean``'s ``SanitizerReport``, ``dscraft.forecast``'s
    backtest report) of returning a result object a caller can inspect
    and act on programmatically, rather than a write-only side-effecting
    exporter.

    Attributes:
        schema_report: the :class:`~dscraft.eda.engine.SchemaReport` for
            the profiled source (column names, Polars dtypes, coarse
            categories).
        null_report: the :class:`~dscraft.eda.engine.NullReport` for the
            profiled source (per-column null counts/percentages, total
            row count).
        row_count: the profiled source's total row count (same value as
            ``null_report.total_rows``, surfaced here too for
            convenience).
        quantile_results: a mapping of column name to
            :class:`~dscraft.eda.sketches.KLLResult`, for every ``"numeric"``
            column that had at least one non-null value. Columns with a
            different category, or with zero non-null values, are absent
            from this mapping (not present with a placeholder value).
        cardinality_results: a mapping of column name to
            :class:`~dscraft.eda.sketches.HLLResult`, for every
            ``"string"`` column that had at least one non-null value.
            Same absent-rather-than-placeholder convention as
            ``quantile_results``.
        association_matrix: the
            :class:`~dscraft.eda.associations.AssociationMatrixResult`
            computed across every profiled column, or ``None`` if the
            source had zero columns (there is nothing to associate).
    """

    def __init__(
        self,
        *,
        schema_report: SchemaReport,
        null_report: NullReport,
        row_count: int,
        quantile_results: dict[str, KLLResult],
        cardinality_results: dict[str, HLLResult],
        association_matrix: Optional[AssociationMatrixResult],
        report_data: EDAReportData,
    ) -> None:
        self.schema_report = schema_report
        self.null_report = null_report
        self.row_count = row_count
        self.quantile_results = quantile_results
        self.cardinality_results = cardinality_results
        self.association_matrix = association_matrix
        self._report_data = report_data

    @property
    def report_data(self) -> EDAReportData:
        """The underlying :class:`~dscraft.eda.report.EDAReportData` this
        profile was built from.

        Exposed for callers who want to call
        :func:`dscraft.eda.report.render_report` themselves (e.g. to get
        the HTML string in-memory without writing a file) rather than
        going through :meth:`export`.
        """
        return self._report_data

    def export(self, path: Union[str, Path]) -> None:
        """Render this profile as a self-contained HTML report and write it to ``path``.

        Equivalent to
        ``dscraft.eda.report.export_report(self.report_data, path)`` --
        see that function's docstring for the exact write semantics
        (UTF-8, parent directory must already exist, overwrites any
        existing file at ``path``).
        """
        export_report(self._report_data, path)


class LazyEDA:
    """Single entry point composing ``engine``/``sketches``/``associations``/``report``.

    ``LazyEDA().profile(source)`` runs the full pipeline described in the
    module docstring (schema + nulls via :mod:`dscraft.eda.engine`,
    per-column sketches via :mod:`dscraft.eda.sketches`, a pairwise
    association matrix via :mod:`dscraft.eda.associations`) and returns an
    :class:`EDAProfile` wrapping both the raw intermediate results and a
    ready-to-export :mod:`dscraft.eda.report` structure.

    All tuning knobs (histogram bin count, top-K category count, which
    quantiles to estimate, sketch precision parameters) are configured
    once at construction time via keyword-only arguments with sensible
    defaults, so the common case is just ``LazyEDA().profile(source)``.
    """

    def __init__(
        self,
        *,
        histogram_bins: int = 10,
        top_k_categories: int = 10,
        quantiles: Sequence[float] = _DEFAULT_QUANTILES,
        hll_log2_k: int = 12,
        kll_k: int = 200,
    ) -> None:
        """
        Args:
            histogram_bins: number of equal-width bins for numeric-column
                histograms (see :func:`_numeric_histogram`).
            top_k_categories: number of most-frequent distinct values kept
                in a string/categorical column's histogram (see
                :func:`_categorical_histogram`).
            quantiles: which quantiles to estimate per numeric column, via
                :func:`dscraft.eda.sketches.estimate_quantiles`. Defaults
                to min/p25/median/p75/max.
            hll_log2_k: precision parameter passed through to every
                :func:`dscraft.eda.sketches.estimate_cardinality` call
                (must be in ``[7, 21]`` -- see that function's docstring).
            kll_k: precision parameter passed through to every
                :func:`dscraft.eda.sketches.estimate_quantiles` call (must
                be in ``[8, 65535]`` -- see that function's docstring).
        """
        self._histogram_bins = histogram_bins
        self._top_k_categories = top_k_categories
        self._quantiles = tuple(quantiles)
        self._hll_log2_k = hll_log2_k
        self._kll_k = kll_k

    def profile(
        self,
        source: Union[pl.LazyFrame, pl.DataFrame, str, Path],
        *,
        title: str = "EDA Report",
        metadata: Optional[dict[str, str]] = None,
    ) -> EDAProfile:
        """Profile ``source`` end-to-end and return a composed :class:`EDAProfile`.

        Args:
            source: anything accepted by
                :func:`dscraft.eda.engine.load_lazy` -- a ``pl.LazyFrame``,
                a ``pl.DataFrame``, or a path (``str``/``Path``) to a
                ``.parquet``/``.csv`` file.
            title: display title embedded in the eventual HTML report
                (see :class:`dscraft.eda.report.EDAReportData.title`).
            metadata: optional extra label/value pairs embedded in the
                eventual HTML report's header (see
                :class:`dscraft.eda.report.EDAReportData.metadata`).

        Returns:
            An :class:`EDAProfile`.

        Raises:
            FileNotFoundError, ValueError, TypeError: see
                :func:`dscraft.eda.engine.load_lazy` -- ``source``
                normalization errors propagate unchanged.
        """
        engine_profile: EngineProfile = profile_engine(source)
        schema_report = engine_profile.schema_report
        null_report = engine_profile.null_report
        row_count = engine_profile.row_count

        # Materialize the full frame exactly once -- see the module
        # docstring's "Materialization strategy" section for why this is
        # a single `.collect()` rather than one per column.
        full_frame: pl.DataFrame = engine_profile.lazyframe.collect()

        quantile_results: dict[str, KLLResult] = {}
        cardinality_results: dict[str, HLLResult] = {}
        column_summaries: list[ColumnSummary] = []

        for column in schema_report.columns:
            column_summaries.append(
                self._summarize_column(
                    column=column,
                    full_frame=full_frame,
                    null_report=null_report,
                    row_count=row_count,
                    quantile_results=quantile_results,
                    cardinality_results=cardinality_results,
                )
            )

        association_result: Optional[AssociationMatrixResult] = None
        report_association_matrix: Optional[AssociationMatrix] = None
        if schema_report.columns:
            # mixed_type_association_matrix duck-types its `data` argument
            # (a `.columns` attribute plus `data[name]` column access) --
            # a `pl.DataFrame` satisfies that directly, so `full_frame` is
            # passed straight through with no conversion. Requires at
            # least 1 column (guarded above); 0 rows or single-category
            # columns are handled internally by that function (recorded
            # as `unavailable_pairs`, not raised).
            association_result = mixed_type_association_matrix(full_frame)
            report_association_matrix = AssociationMatrix(
                column_names=list(association_result.columns),
                values=association_result.matrix.tolist(),
            )

        report_data = EDAReportData(
            column_summaries=column_summaries,
            association_matrix=report_association_matrix,
            title=title,
            row_count=row_count,
            metadata=dict(metadata) if metadata else {},
        )

        return EDAProfile(
            schema_report=schema_report,
            null_report=null_report,
            row_count=row_count,
            quantile_results=quantile_results,
            cardinality_results=cardinality_results,
            association_matrix=association_result,
            report_data=report_data,
        )

    def _summarize_column(
        self,
        *,
        column: ColumnSchema,
        full_frame: pl.DataFrame,
        null_report: NullReport,
        row_count: int,
        quantile_results: dict[str, KLLResult],
        cardinality_results: dict[str, HLLResult],
    ) -> ColumnSummary:
        """Build one column's :class:`~dscraft.eda.report.ColumnSummary`.

        Also populates ``quantile_results``/``cardinality_results`` in
        place for the caller (:meth:`profile`) to bundle into the
        returned :class:`EDAProfile`.
        """
        name = column.name
        null_count = null_report.null_counts.get(name, 0)
        null_percentage = null_report.null_percentages.get(name, 0.0)

        cardinality_estimate: Optional[int] = None
        quantiles: Optional[dict[str, float]] = None
        histogram: Optional[list[HistogramBin]] = None

        # `full_frame[name]` on a `pl.DataFrame` returns a `pl.Series`;
        # `.drop_nulls().to_list()` materializes it as plain Python values
        # -- the same values fed to the HLL/KLL sketches below are reused
        # directly for histogram construction (see module docstring).
        non_null_values = full_frame[name].drop_nulls().to_list()

        if column.category == _NUMERIC_CATEGORY and non_null_values:
            # Polars treats `NaN` as a valid float *value*, not as missing
            # data -- null and NaN are distinct concepts there, so the
            # `.drop_nulls()` call above does NOT drop NaN (nor +/-inf).
            # Left unfiltered, a numeric column containing NaN/inf can
            # still reach quantile estimation / histogram construction
            # below and crash the whole `profile()` call. Filter to only
            # finite values here: quantiles/the histogram are computed
            # over the column's finite values only, with NaN/inf values
            # excluded from sketching (though still reflected in
            # null_count/null_percentage's row-level accounting above,
            # since those are computed independently of this filter).
            finite_values = [value for value in non_null_values if math.isfinite(value)]
            if finite_values:
                kll_result = estimate_quantiles(
                    finite_values, quantiles=self._quantiles, k=self._kll_k
                )
                quantile_results[name] = kll_result
                quantiles = {
                    _quantile_label(q): value
                    for q, value in kll_result.quantile_estimates.items()
                }
                histogram = _numeric_histogram(finite_values, self._histogram_bins)
        elif column.category == _STRING_CATEGORY and non_null_values:
            hll_result = estimate_cardinality(non_null_values, log2_k=self._hll_log2_k)
            cardinality_results[name] = hll_result
            cardinality_estimate = round(hll_result.estimate)
            histogram = _categorical_histogram(non_null_values, self._top_k_categories)
        # "boolean"/"temporal"/"other" columns intentionally get neither
        # sketch -- see the module docstring's routing-heuristic section.

        return ColumnSummary(
            name=name,
            dtype_category=column.category,
            null_count=null_count,
            null_percentage=null_percentage,
            row_count=row_count,
            cardinality_estimate=cardinality_estimate,
            quantiles=quantiles,
            histogram=histogram,
        )
