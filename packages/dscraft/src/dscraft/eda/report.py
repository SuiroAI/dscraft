"""Self-contained, single-file HTML report renderer for EDA results.

This module is the final stage of the ``dscraft.eda`` pipeline: it turns
already-summarized profiling output (per-column stats, a null report, a
pairwise association matrix, pre-binned histogram data) into one standalone
HTML document string, and never touches a raw ``polars``/``pandas``
DataFrame itself. Producing those summaries is the job of
``engine.py``/``sketches.py``/``associations.py`` (sibling modules built
concurrently) -- this module's only contract with them is the plain
dataclasses defined here (:class:`ColumnSummary`, :class:`AssociationMatrix`,
:class:`EDAReportData`), which intentionally use only primitive types and
stdlib containers so a caller wiring real profiling output into a report
never needs this package's HTML/JS internals, and this module never needs
theirs.

Four hard design constraints drive everything below (see the architecture
doc's EDA/reporting research notes):

1. **Single self-contained file, no external network calls.** Every byte of
   CSS and JavaScript is inlined into the one HTML document
   :func:`render_report` returns -- no ``<script src="https://...">``, no
   ``<link href="https://...">``, not even to a CDN-hosted charting
   library. This platform is explicitly local-only/air-gapped-friendly
   (CLAUDE.md), and a report that silently phones home (or silently fails
   to render offline) would violate that. See
   :func:`dscraft.eda.report` test suite's CDN-absence check for how this
   is verified mechanically, not just by convention.
2. **Pre-binned input only.** Every visual here is driven by data the
   caller already aggregated -- a null-percentage-per-column list, a
   pairwise association matrix, a list of :class:`HistogramBin` buckets per
   column. This module performs zero aggregation of its own and never
   accepts (or iterates) a raw per-row dataset. That keeps report
   generation O(number of columns / bins), not O(number of rows), which is
   what actually keeps the rendered file under the ~500KB target
   regardless of how large the underlying dataset was.
3. **Progressive level-of-detail.** The initial paint renders only summary
   statistics and small sketches (a null-percentage bar chart, a compact
   association-matrix heatmap). Per-column histogram detail is wrapped in a
   native ``<details>``/``<summary>`` element, collapsed by default -- an
   HTML-only mechanism that needs zero JavaScript to hide/reveal content,
   so the "detail on demand" behavior costs nothing extra in bundle size or
   script complexity.
4. **Hand-rolled Canvas 2D drawing, not a charting library.** The null-chart
   bar chart and the association-matrix heatmap are drawn onto
   ``<canvas>`` elements via plain ``CanvasRenderingContext2D`` calls
   (``fillRect``, ``fillText``) driven by a JSON literal embedded in an
   inline ``<script>`` tag. No charting library is vendored or referenced
   -- one was not specified by the source research, and adding one would
   be a new, unreviewed frontend dependency inside a single HTML file that
   is supposed to stay small and auditable. Axis labels, legends, and the
   per-column stat tables use plain HTML/CSS, not canvas -- per the source
   research, canvas is reserved for the genuinely dense/numerous visual
   marks (bars, heatmap cells), not for text that reads better as real DOM
   text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape as _html_escape
from pathlib import Path
from typing import Optional

__all__ = [
    "HistogramBin",
    "ColumnSummary",
    "AssociationMatrix",
    "EDAReportData",
    "render_report",
    "export_report",
]


# ---------------------------------------------------------------------------
# Input contract -- plain dataclasses, primitives/stdlib containers only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistogramBin:
    """One pre-binned histogram bucket for a single column.

    ``label`` is a caller-chosen display string for the bucket (e.g.
    ``"[0.0, 5.0)"`` for a numeric bin edge pair, or a categorical value
    name like ``"California"`` for a top-K category count) -- this module
    never derives bin edges or labels itself, it only draws whatever
    ``(label, count)`` pairs the caller already computed (see the module
    docstring's "pre-binned input only" constraint). ``count`` must be
    non-negative; this is checked at render time
    (see :func:`render_report`), not here, so that constructing a
    :class:`HistogramBin` directly never requires importing render-time
    validation helpers.
    """

    label: str
    count: int


@dataclass(frozen=True)
class ColumnSummary:
    """Already-aggregated summary statistics for one column.

    Every field here is a scalar or a small fixed-size sequence -- never a
    per-row value or a raw column of data. ``quantiles`` and ``histogram``
    are both optional and independent of each other: a caller may supply
    quantiles without a histogram (or vice versa) depending on what its
    upstream sketching step computed for that particular column.

    Attributes:
        name: column name, used verbatim as a display label (HTML-escaped
            at render time -- never assumed to be safe to interpolate
            directly).
        dtype_category: a short caller-chosen category string, e.g.
            ``"numeric"``, ``"string"``, ``"boolean"``, ``"temporal"``, or
            ``"other"`` -- matches the coarse categories
            :mod:`dscraft.eda.engine` already produces, but this module
            does not import or validate against that module's
            :data:`~dscraft.eda.engine.ColumnCategory` type; any string is
            accepted and displayed as-is.
        null_count: number of null/missing values in this column.
        null_percentage: ``null_count`` expressed as a percentage in
            ``[0.0, 100.0]`` of the column's total row count (matching
            :class:`dscraft.eda.engine.NullReport`'s convention).
        row_count: total number of rows profiled for this column (used to
            contextualize ``null_count``/``null_percentage`` and, for
            categorical columns, ``cardinality_estimate``).
        cardinality_estimate: an estimated count of distinct values, for
            categorical/string columns where that is meaningful. ``None``
            when not computed or not applicable (e.g. for most numeric
            columns).
        quantiles: an optional mapping of quantile label (e.g. ``"min"``,
            ``"p25"``, ``"p50"``, ``"p75"``, ``"max"``) to its numeric
            value, for numeric columns. Keys and their number are entirely
            caller-defined; this module renders whatever is present as a
            simple label/value table, in insertion order.
        histogram: an optional pre-binned distribution for this column, as
            a list of :class:`HistogramBin`. ``None`` (or an empty list)
            means no histogram is rendered for this column -- not an
            error.
    """

    name: str
    dtype_category: str
    null_count: int
    null_percentage: float
    row_count: int
    cardinality_estimate: Optional[int] = None
    quantiles: Optional[dict[str, float]] = None
    histogram: Optional[list[HistogramBin]] = None


@dataclass(frozen=True)
class AssociationMatrix:
    """A pairwise association/correlation matrix over a set of columns.

    ``values`` is a plain ``list[list[float]]`` (rather than a
    ``numpy.ndarray``) so this module's public dataclasses never require a
    caller (or a future wiring step) to have NumPy installed just to
    construct an :class:`EDAReportData` -- see the module docstring's
    "primitives/stdlib containers only" contract. A NumPy 2D array is
    accepted transparently anywhere a nested sequence is expected (e.g.
    ``AssociationMatrix(column_names=[...], values=my_ndarray.tolist())``
    or, since NumPy arrays already support ``len()``/indexing/iteration,
    even ``values=my_ndarray`` directly -- this dataclass does not enforce
    ``list`` specifically, only that ``values`` is square and matches
    ``column_names`` in length, checked below).

    Values are not required to be in any particular range (e.g. Pearson
    correlation in ``[-1, 1]`` vs. an association-strength score in
    ``[0, 1]``) or to be symmetric -- the heatmap renderer
    (:func:`render_report`) color-scales purely off the min/max values
    actually present in a given matrix, so either convention renders
    sensibly.

    Raises:
        ValueError: at construction time, if ``values`` is not square
            (``len(values) == len(values[i]) == len(column_names)`` for
            every row ``i``).
    """

    column_names: list[str]
    values: list

    def __post_init__(self) -> None:
        n = len(self.column_names)
        if len(self.values) != n:
            raise ValueError(
                f"AssociationMatrix.values must have {n} rows to match "
                f"column_names, got {len(self.values)}."
            )
        for row_index, row in enumerate(self.values):
            if len(row) != n:
                raise ValueError(
                    f"AssociationMatrix.values must be square: row {row_index} has "
                    f"{len(row)} entries, expected {n} to match column_names."
                )


@dataclass
class EDAReportData:
    """The complete input contract for :func:`render_report`.

    This is the one structure a wiring step needs to construct from real
    ``engine.py``/``sketches.py``/``associations.py`` output. Every field
    is a plain dataclass, dict, list, or primitive -- no dependency on this
    module's internal HTML/JS rendering code, and no dependency on any
    particular upstream library's return type (a caller converts
    ``polars``/``numpy`` results into these plain fields itself).

    Attributes:
        column_summaries: per-column summaries, in the order they should
            appear in the report. May be empty -- see
            :func:`render_report`'s documented "no data" fallback.
        association_matrix: an optional pairwise association/correlation
            matrix. ``None`` means no association heatmap is rendered.
        title: a display title for the report (HTML-escaped at render
            time).
        row_count: total row count of the profiled dataset, shown in the
            report header. ``None`` if not known/not supplied.
        metadata: an open-ended mapping of extra caller-chosen label/value
            pairs (e.g. ``{"source": "orders.parquet", "generated_by":
            "dscraft.eda v0.1"}``) rendered as a small key/value list under
            the report header. Both keys and values are HTML-escaped.
    """

    column_summaries: list[ColumnSummary]
    association_matrix: Optional[AssociationMatrix] = None
    title: str = "EDA Report"
    row_count: Optional[int] = None
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rendering internals
# ---------------------------------------------------------------------------

#: Byte budget this module targets for a "representative" dataset (see the
#: package's test suite for what counts as representative). This is a
#: design target documented here for readers of this module, not an
#: enforced runtime ceiling -- render_report() never raises solely because
#: its output exceeds this size, since a caller-supplied report with an
#: unusually large number of columns/histogram bins is still valid input
#: worth rendering, just outside this module's normal design envelope.
TARGET_MAX_BYTES = 500_000

_CSS = """
:root {
  color-scheme: light;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #2563eb;
  --bad: #dc2626;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 2rem;
  line-height: 1.4;
}
h1 { font-size: 1.5rem; margin: 0 0 0.25rem 0; }
h2 { font-size: 1.15rem; margin: 2rem 0 0.75rem 0; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }
h3 { font-size: 0.95rem; margin: 1rem 0 0.5rem 0; color: var(--muted); font-weight: 600; }
p.subtitle { color: var(--muted); margin: 0 0 1rem 0; font-size: 0.9rem; }
table { border-collapse: collapse; font-size: 0.85rem; }
table.stat-table td, table.stat-table th { padding: 0.15rem 0.6rem 0.15rem 0; text-align: left; }
table.stat-table th { color: var(--muted); font-weight: 600; }
.metadata-list { font-size: 0.85rem; color: var(--muted); margin: 0 0 1rem 0; padding: 0; list-style: none; }
.metadata-list li { display: inline; margin-right: 1.25rem; }
details.column-detail {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 0.5rem;
  padding: 0.5rem 0.75rem;
}
details.column-detail summary {
  cursor: pointer;
  font-weight: 600;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
}
details.column-detail summary .col-meta {
  font-weight: 400;
  color: var(--muted);
  font-size: 0.8rem;
}
.null-badge {
  display: inline-block;
  min-width: 3.5rem;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.null-badge.high { color: var(--bad); font-weight: 600; }
canvas { display: block; max-width: 100%; }
.empty-state { color: var(--muted); font-style: italic; padding: 2rem 0; }
"""

# Hand-rolled Canvas 2D drawing only -- no charting library. See module
# docstring, design constraint 4. Kept intentionally small and readable:
# a horizontal bar chart (drawBarChart, used for null percentages and
# per-column histograms) and a heatmap grid (drawHeatmap, used for the
# association matrix), both reading their input from the JSON literal
# embedded below as `window.__EDA_REPORT_DATA__`.
_JS = """
(function () {
  "use strict";
  var DATA = window.__EDA_REPORT_DATA__;

  function drawBarChart(canvas, labels, values, opts) {
    opts = opts || {};
    var ctx = canvas.getContext("2d");
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    var n = values.length;
    if (n === 0) { return; }
    var maxVal = opts.maxValue !== undefined ? opts.maxValue : Math.max.apply(null, values.concat([1e-9]));
    var leftPad = opts.leftPad || 90;
    var rightPad = 10;
    var topPad = 6;
    var bottomPad = 6;
    var plotW = w - leftPad - rightPad;
    var plotH = h - topPad - bottomPad;
    var barH = Math.max(2, Math.min(18, plotH / n - 4));
    var gap = plotH / n;
    ctx.font = "11px -apple-system, Helvetica, Arial, sans-serif";
    ctx.textBaseline = "middle";
    for (var i = 0; i < n; i++) {
      var y = topPad + i * gap + (gap - barH) / 2;
      var frac = maxVal > 0 ? (values[i] / maxVal) : 0;
      var barW = Math.max(0, frac * plotW);
      ctx.fillStyle = opts.barColor || "#2563eb";
      ctx.fillRect(leftPad, y, barW, barH);
      ctx.fillStyle = "#1a1a1a";
      ctx.textAlign = "right";
      ctx.fillText(truncateLabel(String(labels[i]), 14), leftPad - 6, y + barH / 2);
      ctx.textAlign = "left";
      ctx.fillStyle = "#6b7280";
      ctx.fillText(formatValue(values[i]), leftPad + barW + 4, y + barH / 2);
    }
  }

  function truncateLabel(label, maxLen) {
    if (label.length <= maxLen) { return label; }
    return label.slice(0, maxLen - 1) + "\\u2026";
  }

  function formatValue(v) {
    if (Math.abs(v - Math.round(v)) < 1e-9) { return String(Math.round(v)); }
    return v.toFixed(2);
  }

  function heatColor(value, min, max) {
    var span = max - min;
    var t = span > 0 ? (value - min) / span : 0.5;
    t = Math.max(0, Math.min(1, t));
    // Simple blue (low) -> white -> red (high) diverging scale.
    var r, g, b;
    if (t < 0.5) {
      var u = t / 0.5;
      r = Math.round(37 + u * (255 - 37));
      g = Math.round(99 + u * (255 - 99));
      b = Math.round(235 + u * (255 - 235));
    } else {
      var v2 = (t - 0.5) / 0.5;
      r = 255;
      g = Math.round(255 - v2 * (255 - 38));
      b = Math.round(255 - v2 * (255 - 38));
    }
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function drawHeatmap(canvas, labels, matrix) {
    var ctx = canvas.getContext("2d");
    var n = labels.length;
    if (n === 0) { return; }
    var leftPad = 110, topPad = 110, rightPad = 10, bottomPad = 10;
    var w = canvas.width, h = canvas.height;
    var plotW = w - leftPad - rightPad;
    var plotH = h - topPad - bottomPad;
    var cell = Math.min(plotW / n, plotH / n);
    ctx.clearRect(0, 0, w, h);

    var min = Infinity, max = -Infinity;
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var v = matrix[i][j];
        if (v < min) { min = v; }
        if (v > max) { max = v; }
      }
    }
    if (!isFinite(min) || !isFinite(max)) { min = 0; max = 1; }

    ctx.font = "10px -apple-system, Helvetica, Arial, sans-serif";
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        var x = leftPad + j * cell;
        var y = topPad + i * cell;
        ctx.fillStyle = heatColor(matrix[i][j], min, max);
        ctx.fillRect(x, y, Math.ceil(cell), Math.ceil(cell));
      }
    }
    // Row labels (left) and column labels (top, rotated).
    ctx.fillStyle = "#1a1a1a";
    ctx.textBaseline = "middle";
    for (i = 0; i < n; i++) {
      ctx.textAlign = "right";
      ctx.fillText(truncateLabel(String(labels[i]), 16), leftPad - 6, topPad + i * cell + cell / 2);
    }
    for (j = 0; j < n; j++) {
      ctx.save();
      ctx.translate(leftPad + j * cell + cell / 2, topPad - 6);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = "right";
      ctx.fillText(truncateLabel(String(labels[j]), 16), 0, 0);
      ctx.restore();
    }
  }

  function init() {
    if (!DATA) { return; }
    if (DATA.nullChart) {
      var nullCanvas = document.getElementById("null-chart");
      if (nullCanvas) {
        drawBarChart(nullCanvas, DATA.nullChart.labels, DATA.nullChart.values, {
          maxValue: 100, barColor: "#dc2626"
        });
      }
    }
    if (DATA.associationMatrix) {
      var heatCanvas = document.getElementById("association-heatmap");
      if (heatCanvas) {
        drawHeatmap(heatCanvas, DATA.associationMatrix.labels, DATA.associationMatrix.values);
      }
    }
    if (DATA.histograms) {
      for (var colId in DATA.histograms) {
        if (!Object.prototype.hasOwnProperty.call(DATA.histograms, colId)) { continue; }
        var histCanvas = document.getElementById(colId);
        if (!histCanvas) { continue; }
        var hist = DATA.histograms[colId];
        drawBarChart(histCanvas, hist.labels, hist.counts, { leftPad: 110, barColor: "#2563eb" });
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""


def _safe_dom_id(prefix: str, index: int) -> str:
    """Build a deterministic, collision-free DOM id for a per-column element.

    Column names are arbitrary caller-supplied strings and may contain
    spaces, punctuation, or duplicate across columns -- none of which are
    valid/reliable as an HTML ``id``. This derives the id from ``prefix``
    and the column's positional ``index`` only, never from the column
    name itself, so it is always a valid id and always unique within one
    report regardless of what the column is named.
    """
    return f"{prefix}-{index}"


def _round(value: float, digits: int = 4) -> float:
    """Round a float for compact JSON embedding, tolerating non-finite values.

    NaN/inf pass through ``round()`` unchanged (``round(float("nan"), 4)``
    is itself ``nan``), and ``json.dumps`` would otherwise emit the
    non-JSON-standard bare tokens ``NaN``/``Infinity``/``-Infinity``. Those
    tokens *are* accepted by JavaScript's own lenient JSON-literal context
    (this module embeds data as a JS literal via ``<script>``, not via
    ``JSON.parse`` of a strictly-conforming string), so no substitution is
    needed here beyond rounding -- see :func:`_json_literal`.
    """
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return float("nan")


def _json_literal(data: dict) -> str:
    """Serialize ``data`` as a compact JS literal for embedding in a ``<script>`` tag.

    Uses the smallest separators (no extra whitespace) to minimize output
    size -- this is the dominant contributor to report size for a
    wide/detailed dataset, so trimming incidental whitespace here directly
    helps stay under this module's ~500KB target. The literal is embedded
    directly as ``window.__EDA_REPORT_DATA__ = <literal>;`` rather than as
    a JSON string parsed via ``JSON.parse`` at runtime, so it tolerates
    ``NaN``/``Infinity`` (which real association scores or quantiles can
    legitimately contain) without extra sentinel-value encoding.

    Every literal ``<`` character in the serialized output is replaced
    with the JS/JSON unicode escape sequence for it. This is not cosmetic:
    caller-supplied column names or metadata values are string data that
    flows straight into this JSON literal, and a value containing the
    literal substring ``</script>`` (e.g. a column literally named
    ``<script>alert(1)</script>``) would otherwise prematurely close the
    surrounding ``<script>`` tag when the browser's HTML parser scans for
    that exact byte sequence -- HTML parsing happens before JavaScript
    parsing, so no amount of JS-level string escaping (backslashes,
    quotes) protects against this; only guaranteeing the literal ``<``
    character never appears verbatim in the surrounding markup does. The
    unicode escape sequence is valid inside a JS string literal and,
    unlike a raw ``<``, can never be misread as the start of an HTML tag
    by the browser's HTML tokenizer, which the embedding ``<script>`` tag
    is scanned by first.
    """
    return json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")


def _render_empty_report(data: "EDAReportData") -> str:
    """Render a minimal, valid HTML report for an ``EDAReportData`` with no columns.

    See :func:`render_report`'s documented behavior for empty/missing
    ``column_summaries``: this is the "handle gracefully" branch, not an
    error -- a caller profiling a zero-column dataset (or one where every
    upstream sketch failed) still gets a well-formed, if uninformative,
    HTML file back.
    """
    title = _html_escape(data.title)
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f"<style>{_CSS}</style></head>"
        f"<body><h1>{title}</h1>"
        '<p class="empty-state">No column data available for this report.</p>'
        "</body></html>\n"
    )


def _render_metadata_list(metadata: dict[str, str]) -> str:
    if not metadata:
        return ""
    items = "".join(
        f"<li><strong>{_html_escape(str(key))}:</strong> {_html_escape(str(value))}</li>"
        for key, value in metadata.items()
    )
    return f'<ul class="metadata-list">{items}</ul>'


def _render_stat_table(summary: ColumnSummary) -> str:
    rows = [
        ("dtype", _html_escape(summary.dtype_category)),
        ("rows", f"{summary.row_count:,}"),
        ("nulls", f"{summary.null_count:,} ({summary.null_percentage:.1f}%)"),
    ]
    if summary.cardinality_estimate is not None:
        rows.append(("distinct (est.)", f"{summary.cardinality_estimate:,}"))
    if summary.quantiles:
        for label, value in summary.quantiles.items():
            rows.append((_html_escape(str(label)), formatted_quantile(value)))
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f'<table class="stat-table">{body}</table>'


def formatted_quantile(value: float) -> str:
    """Format a numeric quantile value for display in the per-column stat table."""
    if value != value:  # NaN check without importing math for one comparison
        return "NaN"
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"


def _render_column_detail(summary: ColumnSummary, index: int) -> tuple[str, Optional[dict]]:
    """Render one column's collapsible ``<details>`` block.

    Returns the HTML fragment plus (if ``summary.histogram`` is non-empty)
    a ``{"labels": [...], "counts": [...]}`` dict keyed for embedding into
    the page's JSON data blob, so :func:`render_report` can collect every
    column's histogram data into one flat mapping without this function
    needing to know about the surrounding document structure.
    """
    null_class = "null-badge high" if summary.null_percentage >= 20.0 else "null-badge"
    name = _html_escape(summary.name)
    stat_table = _render_stat_table(summary)

    histogram_payload: Optional[dict] = None
    histogram_html = ""
    if summary.histogram:
        canvas_id = _safe_dom_id("hist", index)
        histogram_payload = {
            "labels": [bin_.label for bin_ in summary.histogram],
            "counts": [bin_.count for bin_ in summary.histogram],
        }
        canvas_height = max(60, min(320, 26 * len(summary.histogram)))
        histogram_html = (
            '<h3 style="margin-top:0.75rem;">Distribution</h3>'
            f'<canvas id="{canvas_id}" width="640" height="{canvas_height}"></canvas>'
        )

    detail_id = _safe_dom_id("column", index)
    return (
        f'<details class="column-detail" id="{detail_id}">'
        f"<summary><span>{name}</span>"
        f'<span class="col-meta">'
        f"{_html_escape(summary.dtype_category)} &middot; "
        f'<span class="{null_class}">{summary.null_percentage:.1f}% null</span>'
        "</span></summary>"
        f"{stat_table}{histogram_html}"
        "</details>",
        histogram_payload,
    )


def render_report(data: EDAReportData) -> str:
    """Render ``data`` into a complete, self-contained HTML document string.

    The returned string is a single ``<!DOCTYPE html>`` document with all
    CSS and JavaScript inlined in ``<style>``/``<script>`` tags -- no
    external resource references anywhere (see module docstring, design
    constraint 1). It can be written directly to a ``.html`` file (see
    :func:`export_report`) and opened in a browser with no network access
    and no other files present.

    **Empty-data handling.** If ``data.column_summaries`` is empty, this
    function does **not** raise -- it renders a minimal, valid HTML
    document with a "no data available" message instead (see
    :func:`_render_empty_report`). This is a deliberate choice: an EDA
    pipeline run against a genuinely empty or fully-failed-to-profile
    dataset is a normal (if uninteresting) outcome the caller should still
    get a report for, not an exception to catch.

    **Progressive level-of-detail.** The document's initial paint is the
    "Overview" section only: a null-percentage-per-column bar chart and
    (if ``data.association_matrix`` is supplied) a correlation/association
    heatmap, both drawn on ``<canvas>`` elements. Every column's detailed
    per-column stats (and its histogram, if supplied) live inside a
    ``<details>`` element that is collapsed by default -- expanding it
    is the only user action needed to see that column's detail, and no
    JavaScript is required for the collapse/expand behavior itself (native
    HTML). Every histogram canvas is still drawn eagerly on page load
    (cheap: it is driven by already-binned data, not raw rows), so
    expanding a ``<details>`` never needs to trigger a fresh render.

    Args:
        data: the report's input data. See :class:`EDAReportData`.

    Returns:
        A complete HTML document as a single ``str``.

    Raises:
        ValueError: never raised by this function itself for
            empty/missing ``column_summaries`` (see above) -- but
            constructing an invalid :class:`AssociationMatrix` (non-square
            ``values``) raises ``ValueError`` at :class:`EDAReportData`
            construction time, before ``render_report`` is ever called.
    """
    if not data.column_summaries:
        return _render_empty_report(data)

    title = _html_escape(data.title)
    subtitle_parts = []
    if data.row_count is not None:
        subtitle_parts.append(f"{data.row_count:,} rows")
    subtitle_parts.append(f"{len(data.column_summaries)} columns")
    subtitle = " &middot; ".join(subtitle_parts)

    null_labels = [summary.name for summary in data.column_summaries]
    null_values = [_round(summary.null_percentage, 2) for summary in data.column_summaries]

    json_payload: dict = {
        "nullChart": {"labels": null_labels, "values": null_values},
    }

    if data.association_matrix is not None:
        matrix = data.association_matrix
        json_payload["associationMatrix"] = {
            "labels": list(matrix.column_names),
            "values": [[_round(v) for v in row] for row in matrix.values],
        }
        heatmap_size = max(240, min(900, 40 * len(matrix.column_names) + 120))
        heatmap_html = (
            "<h3>Association matrix</h3>"
            f'<canvas id="association-heatmap" width="{heatmap_size}" height="{heatmap_size}"></canvas>'
        )
    else:
        heatmap_html = ""

    column_details_html = []
    histograms_payload: dict = {}
    for index, summary in enumerate(data.column_summaries):
        html_fragment, histogram_payload = _render_column_detail(summary, index)
        column_details_html.append(html_fragment)
        if histogram_payload is not None:
            histograms_payload[_safe_dom_id("hist", index)] = histogram_payload
    if histograms_payload:
        json_payload["histograms"] = histograms_payload

    null_chart_height = max(120, min(900, 22 * len(data.column_summaries) + 20))

    document = (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{title}</h1>"
        f'<p class="subtitle">{subtitle}</p>'
        f"{_render_metadata_list(data.metadata)}"
        '<section id="overview"><h2>Overview</h2>'
        "<h3>Null percentage by column</h3>"
        f'<canvas id="null-chart" width="760" height="{null_chart_height}"></canvas>'
        f"{heatmap_html}"
        "</section>"
        '<section id="columns"><h2>Columns</h2>'
        f"{''.join(column_details_html)}"
        "</section>"
        "<script>"
        f"window.__EDA_REPORT_DATA__ = {_json_literal(json_payload)};"
        f"{_JS}"
        "</script>"
        "</body></html>\n"
    )
    return document


def export_report(data: EDAReportData, path: str | Path) -> None:
    """Render ``data`` and write the resulting HTML document to ``path``.

    Equivalent to ``Path(path).write_text(render_report(data),
    encoding="utf-8")``, provided as a named convenience so callers do not
    need to remember the UTF-8 encoding requirement (the document may
    contain non-ASCII characters from column names/metadata values) or
    import :func:`render_report` themselves for the common
    render-then-write case. Parent directories are not created
    automatically -- ``path``'s parent must already exist, matching plain
    ``Path.write_text`` semantics.

    Args:
        data: the report's input data. See :class:`EDAReportData`.
        path: destination file path (``str`` or ``pathlib.Path``). Any
            existing file at this path is overwritten.

    Raises:
        FileNotFoundError: if ``path``'s parent directory does not exist.
        ValueError: see :func:`render_report`/:class:`AssociationMatrix`.
    """
    html = render_report(data)
    Path(path).write_text(html, encoding="utf-8")
