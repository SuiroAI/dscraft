"""dscraft.eda.plots -- optional grammar-of-graphics plotting via plotnine.

This module is a genuinely additive, narrow-scope capability (see the
architecture doc's RICED-prioritized issue tracker: Effort=1, Impact=1,
"cosmetic/API-breadth value") -- it is **not** a replacement for
:mod:`dscraft.eda.report`'s hand-rolled, dependency-free HTML/Canvas report
renderer, which stays exactly as-is and remains the default path for a
single-file, air-gapped-friendly shareable report. This module solves a
different problem: exposing ``dscraft.eda``'s already-computed result
objects -- :class:`~dscraft.eda.associations.AssociationMatrixResult` and
:class:`~dscraft.eda.report.ColumnSummary` -- as ready-to-plot inputs for
`plotnine <https://plotnine.org/>`_ (BSD-2-Clause, Tier 1 per CLAUDE.md's
licensing policy), a ggplot2-grammar-of-graphics port for Python. This is
for callers who want publication-quality, static, layered exploratory plots
(e.g. for a paper or notebook) rather than the interactive single-file HTML
report -- the two capabilities are deliberately kept distinct, per the
originating issue's explicit scope note, not blurred into one API.

**Optional, separately-gated dependency.** ``plotnine`` (and its own
transitive dependencies, including matplotlib and pandas) is installed via
a *separate* extra, ``eda-plotnine``, not folded into the base ``eda``
extra -- matching this ``pyproject.toml``'s existing ``automl``/
``automl-onnx`` split-extra precedent. ``plotnine`` is therefore never
imported at module level here (this file's own top-level imports must
succeed with only the base ``eda`` extra installed, since
``dscraft/eda/__init__.py`` imports this module unconditionally to expose
its public functions) -- every function below imports it lazily, inside
the function body, via :func:`_require_plotnine`, mirroring
:mod:`dscraft.automl.compile`'s ``_require_onnx_stack()``/
``ONNXExtraNotInstalledError`` lazy-import discipline for its own optional
``automl-onnx`` extra.

**Reuse, don't recompute.** Both functions below accept the actual result
objects ``dscraft.eda.associations``/``dscraft.eda.report`` already
produce (an :class:`~dscraft.eda.associations.AssociationMatrixResult` or a
:class:`~dscraft.eda.report.ColumnSummary`) and never re-derive an
association matrix or a histogram themselves -- see
:func:`association_heatmap` and :func:`column_distribution` below. Neither
function calls ``.save()`` on the ``plotnine.ggplot`` object it returns;
saving (and any further customization -- themes, labels, facets) is left
entirely to the caller, matching plotnine's own idiomatic
``ggplot(...) + geom_...(...); plot.save(path)`` usage pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dscraft.eda.associations import AssociationMatrixResult
from dscraft.eda.report import ColumnSummary

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import plotnine

__all__ = [
    "PlotnineExtraNotInstalledError",
    "association_heatmap",
    "column_distribution",
]

#: `ColumnSummary.dtype_category` values routed to a numeric-style
#: (pre-binned-equal-width, ordered-bin) rendering by
#: :func:`column_distribution`. Matches `dscraft.eda.engine`'s
#: `"numeric"` category and `LazyEDA`'s own numeric/categorical routing
#: heuristic (see `dscraft/eda/__init__.py`'s module docstring).
_NUMERIC_CATEGORY = "numeric"


class PlotnineExtraNotInstalledError(ImportError):
    """Raised when a `dscraft.eda.plots` function is called without `plotnine` installed.

    `plotnine`-based plotting is a lazy-loaded optional extra of this
    package (`eda-plotnine`, on top of the base `eda` extra), not a hard
    dependency of `dscraft.eda` -- `import dscraft.eda` (and
    `import dscraft.eda.plots`) must succeed without `plotnine` installed.
    This error is only raised the moment a plotting function in this
    module is actually called, mirroring
    `dscraft.automl.compile.ONNXExtraNotInstalledError`'s equivalent
    lazy-import contract for AutoML's optional `automl-onnx` extra.
    """


def _require_plotnine() -> Any:
    """Import `plotnine`, or raise a clear, actionable `PlotnineExtraNotInstalledError`.

    Kept as its own function (rather than inline try/except in each public
    function below) so the import boundary -- and therefore the "this is a
    lazy-loaded optional extra" contract -- is in exactly one place, per
    `dscraft.automl.compile._require_onnx_stack`'s own precedent.
    """
    try:
        import plotnine
    except ImportError as exc:
        raise PlotnineExtraNotInstalledError(
            "dscraft.eda.plots requires the optional 'eda-plotnine' extra "
            "(plotnine, and its own transitive matplotlib/pandas "
            "dependencies) on top of the 'eda' extra (polars, datasketches, "
            "scipy) that dscraft.eda's core profiling/report capability "
            "depends on. Install both together with:\n"
            '    pip install "dscraft[eda,eda-plotnine]"\n'
            "This keeps plotnine-based grammar-of-graphics plotting opt-in "
            "and separate from dscraft.eda's core dependency surface, per "
            "the architecture doc's per-subpackage extras convention."
        ) from exc
    return plotnine


def association_heatmap(
    result: AssociationMatrixResult,
    *,
    title: str = "Association matrix",
) -> "plotnine.ggplot":
    """Render an :class:`AssociationMatrixResult` as a plotnine `geom_tile` heatmap.

    Reuses ``result`` exactly as computed by
    :func:`dscraft.eda.associations.mixed_type_association_matrix` --
    this function performs no correlation/association computation of its
    own, only a long-format reshape of ``result.matrix`` (required by
    `plotnine`'s tidy-data-in, grammar-of-graphics-out API; `geom_tile`
    itself has no "give it a raw 2D array" entry point the way the
    hand-rolled Canvas heatmap in `dscraft.eda.report` does).

    Both axes are rendered as ordered categorical columns, in
    ``result.columns``'s original order (not re-sorted alphabetically or by
    value) -- matching the axis order a caller would already expect from
    having built ``result`` themselves. Cells whose value is
    ``float("nan")`` (see :class:`AssociationMatrixResult.unavailable_pairs`
    -- a pair whose metric could not be computed) are rendered as `plotnine`
    /matplotlib's default missing-value grey, not silently dropped or
    zero-filled.

    Args:
        result: an :class:`~dscraft.eda.associations.AssociationMatrixResult`,
            e.g. from :func:`dscraft.eda.associations.mixed_type_association_matrix`
            directly, or via ``EDAProfile.association_matrix``.
        title: plot title, passed to `plotnine.ggtitle`.

    Returns:
        A `plotnine.ggplot` object. The caller may further customize it
        (themes, color scales, labels) and/or call `.save(path)` on it
        themselves -- this function never calls `.save()`.

    Raises:
        PlotnineExtraNotInstalledError: the optional `eda-plotnine` extra
            is not installed. Install it with
            `pip install "dscraft[eda,eda-plotnine]"`.
    """
    plotnine = _require_plotnine()
    import pandas as pd

    columns = list(result.columns)
    records = [
        {
            "row": row_name,
            "column": col_name,
            "value": float(result.matrix[i, j]),
        }
        for i, row_name in enumerate(columns)
        for j, col_name in enumerate(columns)
    ]
    frame = pd.DataFrame.from_records(records)
    # Ordered categoricals so plotnine/matplotlib render axes in `result`'s
    # own column order, not alphabetically -- and reversed on the y axis so
    # the first column appears at the top of the heatmap, matching the
    # top-to-bottom reading order of `result.matrix` itself.
    frame["row"] = pd.Categorical(frame["row"], categories=list(reversed(columns)), ordered=True)
    frame["column"] = pd.Categorical(frame["column"], categories=columns, ordered=True)

    return (
        plotnine.ggplot(frame, plotnine.aes(x="column", y="row", fill="value"))
        + plotnine.geom_tile()
        + plotnine.scale_fill_gradient2(low="#2563eb", mid="#ffffff", high="#dc2626", midpoint=0)
        + plotnine.labs(title=title, x="", y="", fill="association")
        + plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )


def column_distribution(
    summary: ColumnSummary,
    *,
    title: str | None = None,
) -> "plotnine.ggplot":
    """Render a :class:`ColumnSummary`'s pre-binned histogram as a plotnine bar plot.

    Reuses ``summary.histogram`` exactly as already computed by
    `LazyEDA.profile` (equal-width bins for numeric columns via
    `dscraft.eda._numeric_histogram`, top-K value-frequency bins for
    string/categorical columns via `dscraft.eda._categorical_histogram`) --
    this function performs no binning/counting of its own and never touches
    a raw per-row column of values (`ColumnSummary`, per
    `dscraft.eda.report`'s own design constraint, never carries raw row
    data, only already-aggregated summaries).

    Dispatches its rendering style on ``summary.dtype_category``:

    - ``"numeric"``: bars are drawn edge-to-edge (``width=1``, no gaps)
      against ``summary.histogram``'s bin order preserved as an ordered
      categorical axis -- the same continuous, contiguous-bin visual a
      `geom_histogram` produces, without asking plotnine to re-bin already-
      binned data itself (`geom_histogram`'s own binning stat operates on
      raw observations, which this function deliberately does not have or
      request -- see the module docstring's "reuse, don't recompute"
      constraint). This is `geom_col` (`plotnine`'s pre-computed-height bar
      geom, i.e. `geom_bar(stat="identity")`) with histogram-style styling.
    - anything else (``"string"``, ``"boolean"``, ``"temporal"``,
      ``"other"``): bars are drawn with plotnine's default gap/spacing, in
      ``summary.histogram``'s existing (already frequency-sorted, for
      string columns) order -- the conventional discrete-category bar
      chart look via `geom_bar(stat="identity")`.

    Args:
        summary: a :class:`~dscraft.eda.report.ColumnSummary` with a
            non-empty ``histogram``, e.g. from
            ``EDAProfile.report_data.column_summaries``.
        title: plot title. Defaults to ``f"Distribution of {summary.name}"``.

    Returns:
        A `plotnine.ggplot` object. The caller may further customize it
        (themes, color scales, labels) and/or call `.save(path)` on it
        themselves -- this function never calls `.save()`.

    Raises:
        ValueError: if ``summary.histogram`` is `None` or empty -- there is
            nothing to plot.
        PlotnineExtraNotInstalledError: the optional `eda-plotnine` extra
            is not installed. Install it with
            `pip install "dscraft[eda,eda-plotnine]"`.
    """
    if not summary.histogram:
        raise ValueError(
            f"column_distribution() requires summary.histogram to be a "
            f"non-empty list of HistogramBin entries; column {summary.name!r} "
            "has none (LazyEDA.profile only populates a histogram for "
            "'numeric'/'string' columns with at least one non-null value)."
        )

    plotnine = _require_plotnine()
    import pandas as pd

    labels = [bin_.label for bin_ in summary.histogram]
    counts = [bin_.count for bin_ in summary.histogram]
    frame = pd.DataFrame({"label": labels, "count": counts})
    # Preserve the histogram's own bin order (ascending numeric bin edges,
    # or descending top-K frequency for categorical columns) as an ordered
    # categorical axis, rather than letting plotnine/matplotlib re-sort
    # labels alphabetically.
    frame["label"] = pd.Categorical(frame["label"], categories=labels, ordered=True)

    plot_title = title if title is not None else f"Distribution of {summary.name}"
    is_numeric = summary.dtype_category == _NUMERIC_CATEGORY
    bar_width = 1.0 if is_numeric else 0.9

    plot = (
        plotnine.ggplot(frame, plotnine.aes(x="label", y="count"))
        + plotnine.geom_col(width=bar_width, fill="#2563eb")
        + plotnine.labs(title=plot_title, x=summary.name, y="count")
        + plotnine.theme(axis_text_x=plotnine.element_text(rotation=45, hjust=1))
    )
    return plot
