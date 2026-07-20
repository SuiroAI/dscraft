"""dscraft.eda.engine -- lazy execution foundation for exploratory data analysis.

This module is the single entry point every other ``dscraft.eda`` submodule
(e.g. ``sketches.py``, ``associations.py``, ``report.py``) is expected to
build on top of for turning a caller-supplied data source into a
``polars.LazyFrame`` plus basic metadata, without each of them
re-implementing source-loading, null-analysis, or schema-reading from
scratch. Per the Tier-1 convention documented in ``dscraft.core.data``
(architecture doc §2.1), dense tabular data's canonical local representation
is Apache Arrow, fronted here by Polars' *lazy* API rather than its eager
one.

**Why lazy, specifically.** ``pl.scan_parquet``/``pl.scan_csv`` build a query
plan without materializing any data. Polars' own query optimizer applies
projection pruning and predicate pushdown against that plan only when it is
actually executed (``.collect()``), so a caller that only ever needs, say,
row counts and null percentages for three columns of a 50-column file never
pays to read the other 47 columns off disk. Eagerly loading the whole file
first (``pl.read_parquet``/``pl.read_csv``) would defeat that entirely, so
this module never does so -- see :func:`load_lazy`.

This module intentionally does not import or depend on ``sketches.py``,
``associations.py``, or ``report.py`` (those are sibling modules built
concurrently); it has zero knowledge of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ColumnCategory",
    "ColumnSchema",
    "SchemaReport",
    "NullReport",
    "EngineProfile",
    "load_lazy",
    "profile_schema",
    "profile_nulls",
    "profile_engine",
]

#: File extensions this module knows how to scan lazily. Anything else
#: raises a ``ValueError`` in :func:`load_lazy` rather than silently
#: guessing a format.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".parquet", ".csv")

#: Coarse, dtype-derived buckets used by :func:`profile_schema` for
#: convenience -- these are *not* a replacement for the exact Polars dtype
#: (still available per-column on :class:`ColumnSchema`), just a cheap
#: grouping other eda modules (e.g. a sketching module deciding whether a
#: numeric summary or a string-length summary applies to a column) can
#: switch on without each re-deriving the same dtype-to-category mapping.
ColumnCategory = str
_CATEGORY_NUMERIC: ColumnCategory = "numeric"
_CATEGORY_STRING: ColumnCategory = "string"
_CATEGORY_TEMPORAL: ColumnCategory = "temporal"
_CATEGORY_BOOLEAN: ColumnCategory = "boolean"
_CATEGORY_OTHER: ColumnCategory = "other"


def _categorize_dtype(dtype: pl.DataType) -> ColumnCategory:
    """Map a Polars dtype to a coarse category string.

    Order matters: ``pl.Boolean`` is checked before the numeric family
    since some dtype-classification helpers elsewhere treat booleans as a
    degenerate integer -- here they are always their own category, never
    folded into ``"numeric"``.
    """
    if dtype == pl.Boolean:
        return _CATEGORY_BOOLEAN
    if dtype.is_numeric():
        return _CATEGORY_NUMERIC
    if dtype.is_temporal():
        return _CATEGORY_TEMPORAL
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return _CATEGORY_STRING
    return _CATEGORY_OTHER


@dataclass(frozen=True)
class ColumnSchema:
    """Schema summary for a single column."""

    name: str
    dtype: pl.DataType
    category: ColumnCategory


@dataclass(frozen=True)
class SchemaReport:
    """Schema summary for an entire frame.

    ``columns`` preserves the frame's original column order (the order
    ``pl.LazyFrame.schema`` itself yields, which is insertion order, not a
    re-sorted one), so callers can zip it against other order-sensitive
    per-column results without re-deriving column order themselves.
    """

    columns: list[ColumnSchema]

    def by_name(self) -> dict[str, ColumnSchema]:
        """Convenience lookup: column name -> its :class:`ColumnSchema`."""
        return {column.name: column for column in self.columns}

    def names_in_category(self, category: ColumnCategory) -> list[str]:
        """Column names whose coarse category equals ``category``, in
        original column order."""
        return [column.name for column in self.columns if column.category == category]


@dataclass(frozen=True)
class NullReport:
    """Per-column null count and null percentage for a frame.

    ``null_percentages`` is expressed as a value in ``[0.0, 100.0]``, not a
    ``[0.0, 1.0]`` fraction. ``total_rows`` is included so callers can
    re-derive percentages themselves if a different convention is needed,
    without re-running the underlying lazy query.
    """

    null_counts: dict[str, int]
    null_percentages: dict[str, float]
    total_rows: int

    def columns_with_nulls(self) -> list[str]:
        """Column names with at least one null value, in schema order."""
        return [name for name, count in self.null_counts.items() if count > 0]


@dataclass(frozen=True)
class EngineProfile:
    """Bundled basic metadata for a data source: schema + nulls + row count.

    This is the return type of :func:`profile_engine`, the single entry
    point other ``dscraft.eda`` modules are expected to call to get a
    normalized ``pl.LazyFrame`` plus this metadata, rather than each
    re-implementing source normalization, schema reading, or null-counting
    themselves.
    """

    lazyframe: pl.LazyFrame
    schema_report: SchemaReport
    null_report: NullReport
    row_count: int


def load_lazy(source: pl.LazyFrame | pl.DataFrame | str | Path) -> pl.LazyFrame:
    """Normalize a data source into a single ``pl.LazyFrame``.

    Accepted input forms:

    - ``pl.LazyFrame`` -- returned as-is (no-op).
    - ``pl.DataFrame`` -- an already-materialized eager frame. Accepted and
      converted via ``.lazy()`` rather than rejected: a caller that already
      has an eager frame in hand (e.g. the small result of some upstream
      computation) gets a uniform lazy-execution entry point without having
      to remember to call ``.lazy()`` themselves first. Note that this does
      *not* retroactively make the data source itself lazy -- whatever
      caused ``source`` to be eager already happened -- it only makes the
      *rest* of the pipeline built on top of the returned ``LazyFrame``
      (projection pruning, predicate pushdown, etc. for downstream
      ``.select``/``.filter`` calls) benefit from Polars' lazy optimizer.
    - ``str`` / ``pathlib.Path`` -- a file path, scanned lazily via
      ``pl.scan_parquet`` (``.parquet``) or ``pl.scan_csv`` (``.csv``). Only
      these two extensions are supported; see :data:`SUPPORTED_EXTENSIONS`.
      The file is *not* read into memory here -- ``scan_*`` only builds a
      query plan, per this module's docstring.

    Args:
        source: a ``pl.LazyFrame``, a ``pl.DataFrame``, or a path (string or
            ``Path``) to a ``.parquet`` or ``.csv`` file.

    Returns:
        A ``pl.LazyFrame`` wrapping ``source``.

    Raises:
        FileNotFoundError: if ``source`` is a path-like value that does not
            exist on disk.
        ValueError: if ``source`` is a path-like value that exists but does
            not have a supported extension (``.parquet``/``.csv``).
        TypeError: if ``source`` is none of ``pl.LazyFrame``, ``pl.DataFrame``,
            ``str``, or ``Path`` -- the error message names the actual type
            received.
    """
    if isinstance(source, pl.LazyFrame):
        return source
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path!r}")
        if not path.is_file():
            raise ValueError(f"Expected a file, got a directory or special file: {path!r}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.scan_parquet(path)
        if suffix == ".csv":
            return pl.scan_csv(path)
        raise ValueError(
            f"Unsupported file extension {suffix!r} for {path!r}. Only "
            f"{SUPPORTED_EXTENSIONS!r} are supported by dscraft.eda.engine.load_lazy "
            "at this time."
        )
    raise TypeError(
        "dscraft.eda.engine.load_lazy expected a pl.LazyFrame, pl.DataFrame, "
        f"str, or pathlib.Path, got {type(source).__name__!r} ({source!r})."
    )


def profile_schema(frame: pl.LazyFrame) -> SchemaReport:
    """Summarize ``frame``'s schema: column names, Polars dtypes, and a
    coarse category per column.

    Reads ``frame.collect_schema()`` directly -- Polars' own schema
    resolution at the query-plan level, which inspects the plan's metadata
    without executing it (i.e. no row data is scanned or materialized) --
    rather than running a separate eager type-inference pass. This is
    intentional: for
    Parquet/CSV sources loaded via :func:`load_lazy`, Polars has already
    determined (or, for CSV, inferred from a sample) the dtype of every
    column as part of building the scan's query plan, and re-deriving that
    information a second way here would both duplicate work and risk
    disagreeing with what Polars itself will actually produce on
    ``.collect()``.

    Args:
        frame: a ``pl.LazyFrame`` (see :func:`load_lazy` to normalize other
            input forms into one first).

    Returns:
        A :class:`SchemaReport` in the frame's original column order.
    """
    schema = frame.collect_schema()
    columns = [
        ColumnSchema(name=name, dtype=dtype, category=_categorize_dtype(dtype))
        for name, dtype in schema.items()
    ]
    return SchemaReport(columns=columns)


def profile_nulls(frame: pl.LazyFrame) -> NullReport:
    """Compute per-column null count and null percentage for ``frame``.

    Executes exactly one lazy aggregation query (one ``.collect()`` call)
    that computes every column's null count in a single pass, then derives
    percentages from that same result plus one ``pl.len()`` row count --
    not one ``.collect()`` per column, which would force ``n_columns``
    separate full scans of the source and defeat the point of using
    Polars' lazy engine at all.

    An empty frame (zero columns) returns a :class:`NullReport` with empty
    ``null_counts``/``null_percentages`` dicts. A frame with columns but
    zero rows reports ``null_percentages`` of ``0.0`` for every column
    (there are no rows to be null), never a division-by-zero error.

    Args:
        frame: a ``pl.LazyFrame`` (see :func:`load_lazy` to normalize other
            input forms into one first).

    Returns:
        A :class:`NullReport` keyed by column name, in the frame's original
        column order.
    """
    columns = list(frame.collect_schema().keys())
    if not columns:
        return NullReport(null_counts={}, null_percentages={}, total_rows=0)

    result = frame.select(
        [pl.col(name).null_count().alias(name) for name in columns] + [pl.len().alias("__dscraft_eda_row_count__")]
    ).collect()
    row = result.row(0, named=True)
    total_rows = int(row["__dscraft_eda_row_count__"])

    null_counts: dict[str, int] = {name: int(row[name]) for name in columns}
    if total_rows == 0:
        null_percentages: dict[str, float] = {name: 0.0 for name in columns}
    else:
        null_percentages = {name: (null_counts[name] / total_rows) * 100.0 for name in columns}

    return NullReport(null_counts=null_counts, null_percentages=null_percentages, total_rows=total_rows)


def profile_engine(source: pl.LazyFrame | pl.DataFrame | str | Path) -> EngineProfile:
    """Single entry point: normalize ``source`` and compute its basic profile.

    This is the function other ``dscraft.eda`` modules (``sketches.py``,
    ``associations.py``, ``report.py``) are expected to call to get a
    ``pl.LazyFrame`` plus schema/null/row-count metadata in one step,
    rather than each calling :func:`load_lazy` themselves and
    re-implementing null-counting or schema-reading on top of it.

    Internally this issues exactly one ``.collect()`` call against the
    normalized lazy plan, inside :func:`profile_nulls`, for the null counts
    (which also yields the row count via the same query, at no extra scan
    cost). The schema comes from ``collect_schema()``, which resolves plan
    metadata without executing the plan or scanning any row data. No step
    here materializes the full frame.

    Args:
        source: anything accepted by :func:`load_lazy` -- a ``pl.LazyFrame``,
            a ``pl.DataFrame``, or a path (string or ``Path``) to a
            ``.parquet``/``.csv`` file.

    Returns:
        An :class:`EngineProfile` bundling the normalized ``pl.LazyFrame``,
        its :class:`SchemaReport`, its :class:`NullReport`, and its row
        count (``null_report.total_rows``, surfaced again at the top level
        for convenience since it is the single most commonly needed piece
        of metadata).

    Raises:
        FileNotFoundError, ValueError, TypeError: see :func:`load_lazy`.
    """
    frame = load_lazy(source)
    schema_report = profile_schema(frame)
    null_report = profile_nulls(frame)
    return EngineProfile(
        lazyframe=frame,
        schema_report=schema_report,
        null_report=null_report,
        row_count=null_report.total_rows,
    )
