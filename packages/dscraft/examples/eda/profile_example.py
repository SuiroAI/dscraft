"""Runnable demo: profile a small synthetic dataset with `dscraft.eda.LazyEDA`.

This script only imports and calls the real `dscraft.eda` package API -- it
does not reimplement any profiling/sketching/association logic inline (per
CLAUDE.md's "no net-new scripts" rule). Run it with:

    python packages/dscraft/examples/eda/profile_example.py

It builds a small synthetic `polars.DataFrame` in memory (no network access
or bundled dataset file required) with a deliberate mix of column shapes --
a numeric column with a couple of nulls, a low-cardinality categorical
column, a high-cardinality "id"-style string column, and a boolean column --
so the printed summary below exercises every column-category branch
`LazyEDA.profile` supports (see `dscraft/eda/__init__.py`'s module
docstring for the numeric-vs-categorical routing heuristic).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from dscraft.eda import LazyEDA

ROWS = 200


def _build_synthetic_dataframe() -> pl.DataFrame:
    """A small, deliberately mixed-dtype synthetic dataset."""
    countries = ["US", "US", "US", "CA", "CA", "MX", "DE", "FR"]
    ages = [float(20 + (i * 7) % 60) for i in range(ROWS)]
    # Sprinkle in a few nulls so the null-percentage summary has something
    # to report.
    ages = [None if i % 37 == 0 else age for i, age in enumerate(ages)]

    return pl.DataFrame(
        {
            "customer_id": [f"cust-{i:06d}" for i in range(ROWS)],  # high-cardinality string
            "country": [countries[i % len(countries)] for i in range(ROWS)],  # low-cardinality
            "age": ages,  # numeric, with nulls
            "is_subscribed": [i % 3 == 0 for i in range(ROWS)],  # boolean
        }
    )


def main() -> None:
    """Profile the synthetic dataset, print a summary, and export an HTML report."""
    df = _build_synthetic_dataframe()

    print(f"Profiling a synthetic {df.shape[0]}-row x {df.shape[1]}-column dataset...")
    profile = LazyEDA().profile(df, title="Synthetic Customer Dataset EDA")

    print()
    print(f"Row count: {profile.row_count}")
    print(f"Columns with at least one null: {profile.null_report.columns_with_nulls()}")
    print()

    print("Per-column summary:")
    for column in profile.schema_report.columns:
        name = column.name
        null_pct = profile.null_report.null_percentages[name]
        line = f"  {name:15s} category={column.category:9s} null%={null_pct:5.1f}"
        if name in profile.quantile_results:
            median = profile.quantile_results[name].quantile_estimates[0.5]
            line += f"  median~={median:.1f}"
        if name in profile.cardinality_results:
            estimate = profile.cardinality_results[name].estimate
            line += f"  distinct~={estimate:.0f}"
        print(line)

    if profile.association_matrix is not None:
        print()
        print(f"Association matrix computed across columns: {profile.association_matrix.columns}")
        if profile.association_matrix.unavailable_pairs:
            print(f"  Unavailable pairs: {list(profile.association_matrix.unavailable_pairs)}")

    output_path = Path(tempfile.gettempdir()) / "dscraft_eda_profile_example.html"
    profile.export(output_path)
    print()
    print(f"Wrote self-contained HTML EDA report to: {output_path}")
    print("Open that file directly in a browser to view it (no server needed).")


if __name__ == "__main__":
    main()
