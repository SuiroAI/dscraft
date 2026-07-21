"""Tests for dscraft.eda.plots -- optional plotnine grammar-of-graphics plotting.

`plotnine`-dependent tests require the optional `eda-plotnine` extra; they
are skipped (not failed) via `_import_plotnine()` if it isn't installed,
mirroring `dscraft/tests/automl/test_compile.py`'s own per-test
`pytest.importorskip` pattern for its optional `automl-onnx` extra. The
clear-error path (`PlotnineExtraNotInstalledError`) is instead simulated by
monkeypatching `sys.modules["plotnine"]` to `None` -- which makes Python's
own import machinery raise `ImportError` for `import plotnine` -- so that
test exercises the real error path without needing a second, separate
plotnine-free virtualenv.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest

from dscraft.eda import LazyEDA
from dscraft.eda.associations import mixed_type_association_matrix
from dscraft.eda.plots import (
    PlotnineExtraNotInstalledError,
    association_heatmap,
    column_distribution,
)
from dscraft.eda.report import ColumnSummary, HistogramBin


def _import_plotnine():
    """Import `plotnine`, skipping the calling test if unavailable.

    Kept as a per-test helper (rather than a module-level
    `pytest.importorskip`) so `test_plotnine_extra_not_installed_error`
    -- which doesn't actually need `plotnine` installed -- always runs.
    """
    return pytest.importorskip(
        "plotnine", reason="plotnine not installed; skipping plotnine-dependent tests"
    )


def _mixed_dataframe(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    countries = np.array(["US", "CA", "MX", "DE"])
    return pl.DataFrame(
        {
            "country": rng.choice(countries, size=n).tolist(),
            "amount": rng.normal(loc=50.0, scale=10.0, size=n).tolist(),
            "score": rng.normal(loc=0.0, scale=1.0, size=n).tolist(),
        }
    )


def test_association_heatmap_returns_ggplot() -> None:
    """`association_heatmap` returns a `plotnine.ggplot` built from
    `result`'s own data (reused directly, not recomputed by
    `dscraft.eda.plots`), and actually inspects the rendered plot's
    structure rather than just its type:

    - a `geom_tile` layer is present (the heatmap-specific geom
      `association_heatmap` documents using, not e.g. `geom_col`);
    - `plot.data` (the long-format frame `association_heatmap` builds
      from `result.matrix`) contains one row per (row, column) pair, with
      the same column/row labels and association values as `result`
      itself -- proving the reshape didn't drop or garble any cell;
    - the default title ("Association matrix") propagates into the
      plot's `labs`.
    """
    plotnine = _import_plotnine()
    result = mixed_type_association_matrix(_mixed_dataframe())
    plot = association_heatmap(result)
    assert isinstance(plot, plotnine.ggplot)

    # A `geom_tile` layer is present -- the heatmap-specific geom this
    # function documents itself as using.
    assert any(isinstance(layer.geom, plotnine.geoms.geom_tile) for layer in plot.layers)

    # The long-format `plot.data` frame has one row per (row, column) pair
    # and reproduces `result.matrix`'s actual values -- not just some
    # frame of the right shape.
    columns = list(result.columns)
    assert len(plot.data) == len(columns) * len(columns)
    assert set(plot.data["row"].astype(str)) == set(columns)
    assert set(plot.data["column"].astype(str)) == set(columns)
    for _, record in plot.data.iterrows():
        i = columns.index(str(record["row"]))
        j = columns.index(str(record["column"]))
        expected = float(result.matrix[i, j])
        if np.isnan(expected):
            assert np.isnan(record["value"])
        else:
            assert record["value"] == pytest.approx(expected)

    # The default title propagates through `plotnine.labs(title=...)`.
    assert plot.labels.title == "Association matrix"


def test_association_heatmap_from_lazyeda_profile() -> None:
    """`association_heatmap` also accepts the `AssociationMatrixResult`
    surfaced on a real `EDAProfile` from `LazyEDA().profile(...)` --
    proving the end-to-end wiring, not just a hand-built result object."""
    plotnine = _import_plotnine()
    profile = LazyEDA().profile(_mixed_dataframe())
    assert profile.association_matrix is not None
    plot = association_heatmap(profile.association_matrix, title="Custom title")
    assert isinstance(plot, plotnine.ggplot)


def test_column_distribution_numeric_returns_ggplot() -> None:
    """`column_distribution` returns a `plotnine.ggplot` for a numeric
    column's pre-binned histogram (`geom_col`-based histogram styling)."""
    plotnine = _import_plotnine()
    summary = ColumnSummary(
        name="amount",
        dtype_category="numeric",
        null_count=0,
        null_percentage=0.0,
        row_count=100,
        histogram=[
            HistogramBin(label="[0, 10)", count=5),
            HistogramBin(label="[10, 20)", count=40),
            HistogramBin(label="[20, 30]", count=55),
        ],
    )
    plot = column_distribution(summary)
    assert isinstance(plot, plotnine.ggplot)


def test_column_distribution_categorical_returns_ggplot() -> None:
    """`column_distribution` returns a `plotnine.ggplot` for a
    string/categorical column's top-K value-frequency histogram
    (`geom_col`-based bar-chart styling)."""
    plotnine = _import_plotnine()
    summary = ColumnSummary(
        name="country",
        dtype_category="string",
        null_count=0,
        null_percentage=0.0,
        row_count=100,
        cardinality_estimate=4,
        histogram=[
            HistogramBin(label="US", count=60),
            HistogramBin(label="CA", count=25),
            HistogramBin(label="MX", count=15),
        ],
    )
    plot = column_distribution(summary, title="Country frequency")
    assert isinstance(plot, plotnine.ggplot)


def test_column_distribution_from_real_lazyeda_profile() -> None:
    """`column_distribution` accepts a real `ColumnSummary` straight out of
    `EDAProfile.report_data.column_summaries` (reused directly, not
    recomputed by `dscraft.eda.plots`)."""
    plotnine = _import_plotnine()
    profile = LazyEDA().profile(_mixed_dataframe())
    summaries = {s.name: s for s in profile.report_data.column_summaries}
    plot = column_distribution(summaries["amount"])
    assert isinstance(plot, plotnine.ggplot)


def test_column_distribution_rejects_empty_histogram() -> None:
    """`column_distribution` raises `ValueError` (not a `plotnine`-level
    error) for a `ColumnSummary` with no histogram -- there is nothing to
    plot, and this must be caught before ever trying to import plotnine."""
    summary = ColumnSummary(
        name="flag",
        dtype_category="boolean",
        null_count=0,
        null_percentage=0.0,
        row_count=10,
    )
    with pytest.raises(ValueError):
        column_distribution(summary)


def test_plotnine_extra_not_installed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both plotting functions raise a clear `PlotnineExtraNotInstalledError`
    when `plotnine` cannot be imported, simulated here by forcing
    `import plotnine` to fail via `sys.modules["plotnine"] = None` (Python's
    import machinery raises `ImportError` for any module whose
    `sys.modules` entry is `None`), rather than requiring a second,
    separate plotnine-free virtualenv."""
    monkeypatch.setitem(sys.modules, "plotnine", None)

    result = mixed_type_association_matrix(_mixed_dataframe(n=20))
    with pytest.raises(PlotnineExtraNotInstalledError):
        association_heatmap(result)

    summary = ColumnSummary(
        name="amount",
        dtype_category="numeric",
        null_count=0,
        null_percentage=0.0,
        row_count=20,
        histogram=[HistogramBin(label="[0, 1)", count=5)],
    )
    with pytest.raises(PlotnineExtraNotInstalledError):
        column_distribution(summary)


def test_public_api_surface() -> None:
    """`association_heatmap`, `column_distribution`, and
    `PlotnineExtraNotInstalledError` are all importable from the top-level
    `dscraft.eda` package (one canonical export path), matching
    `dscraft.automl`'s equivalent `test_public_api_surface` convention for
    its own lazy-loaded-extra error type."""
    import dscraft.eda

    assert dscraft.eda.association_heatmap is association_heatmap
    assert dscraft.eda.column_distribution is column_distribution
    assert dscraft.eda.PlotnineExtraNotInstalledError is PlotnineExtraNotInstalledError
