"""End-to-end wiring tests for `dscraft.eda.LazyEDA`/`EDAProfile`.

Unlike `test_engine.py`/`test_sketches.py`/`test_associations.py`/
`test_report.py` (which each test one submodule in isolation against its
own real API), these tests exercise the composed public entry point --
`LazyEDA().profile(source)` -- to prove the wiring in
`dscraft/eda/__init__.py` actually calls through every submodule
correctly and produces a coherent, exportable result.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from dscraft.eda import EDAProfile, LazyEDA
from dscraft.eda.associations import AssociationMatrixResult
from dscraft.eda.engine import NullReport, SchemaReport
from dscraft.eda.report import EDAReportData
from dscraft.eda.sketches import HLLResult, KLLResult


def _mixed_dataframe(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    countries = np.array(["US", "CA", "MX", "DE"])
    return pl.DataFrame(
        {
            "id": [f"row-{i:05d}" for i in range(n)],  # high-cardinality string
            "country": rng.choice(countries, size=n).tolist(),  # low-cardinality string
            "amount": [
                None if i % 41 == 0 else float(rng.normal(loc=50.0, scale=10.0))
                for i in range(n)
            ],  # numeric, with nulls
            "is_active": rng.choice([True, False], size=n).tolist(),  # boolean
        }
    )


# ---------------------------------------------------------------------------
# Basic end-to-end shape
# ---------------------------------------------------------------------------


class TestProfileBasics:
    def test_returns_eda_profile(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        assert isinstance(profile, EDAProfile)

    def test_schema_and_null_reports_are_real_engine_types(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        assert isinstance(profile.schema_report, SchemaReport)
        assert isinstance(profile.null_report, NullReport)

    def test_row_count_matches_source(self) -> None:
        df = _mixed_dataframe(n=123)
        profile = LazyEDA().profile(df)
        assert profile.row_count == 123
        assert profile.null_report.total_rows == 123

    def test_accepts_path_source(self, tmp_path) -> None:
        df = _mixed_dataframe(n=50)
        path = tmp_path / "mixed.parquet"
        df.write_parquet(path)
        profile = LazyEDA().profile(path)
        assert profile.row_count == 50


# ---------------------------------------------------------------------------
# Column routing: numeric -> quantiles, string -> cardinality
# ---------------------------------------------------------------------------


class TestColumnRouting:
    def test_numeric_column_gets_quantile_result_and_no_cardinality(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        assert "amount" in profile.quantile_results
        assert isinstance(profile.quantile_results["amount"], KLLResult)
        assert "amount" not in profile.cardinality_results

    def test_string_columns_get_cardinality_result_and_no_quantiles(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        for name in ("id", "country"):
            assert name in profile.cardinality_results
            assert isinstance(profile.cardinality_results[name], HLLResult)
            assert name not in profile.quantile_results

    def test_high_cardinality_column_estimate_roughly_matches_row_count(self) -> None:
        df = _mixed_dataframe(n=300)
        profile = LazyEDA().profile(df)
        # Every "id" value is unique by construction.
        assert profile.cardinality_results["id"].estimate == pytest.approx(300, rel=0.2)

    def test_low_cardinality_column_estimate_is_small(self) -> None:
        df = _mixed_dataframe(n=300)
        profile = LazyEDA().profile(df)
        # Only 4 distinct countries by construction.
        assert profile.cardinality_results["country"].estimate < 20

    def test_boolean_column_gets_neither_sketch(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        assert "is_active" not in profile.quantile_results
        assert "is_active" not in profile.cardinality_results

    def test_quantile_estimates_include_default_min_median_max(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        kll = profile.quantile_results["amount"]
        assert set(kll.quantile_estimates.keys()) == {0.0, 0.25, 0.5, 0.75, 1.0}


# ---------------------------------------------------------------------------
# Association matrix wiring
# ---------------------------------------------------------------------------


class TestAssociationMatrixWiring:
    def test_association_matrix_present_and_covers_all_columns(self) -> None:
        df = _mixed_dataframe()
        profile = LazyEDA().profile(df)
        assert isinstance(profile.association_matrix, AssociationMatrixResult)
        assert set(profile.association_matrix.columns) == set(df.columns)

    def test_single_column_source_still_produces_identity_matrix(self) -> None:
        df = pl.DataFrame({"only": [1.0, 2.0, 3.0, 4.0, 5.0]})
        profile = LazyEDA().profile(df)
        assert profile.association_matrix is not None
        assert profile.association_matrix.matrix.shape == (1, 1)
        assert profile.association_matrix.matrix[0, 0] == pytest.approx(1.0)

    def test_zero_column_source_has_no_association_matrix(self) -> None:
        df = pl.DataFrame()
        profile = LazyEDA().profile(df)
        assert profile.association_matrix is None


# ---------------------------------------------------------------------------
# Report composition + export
# ---------------------------------------------------------------------------


class TestReportComposition:
    def test_report_data_is_real_eda_report_data(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        assert isinstance(profile.report_data, EDAReportData)
        assert len(profile.report_data.column_summaries) == 4

    def test_report_data_column_summary_names_match_schema_order(self) -> None:
        df = _mixed_dataframe()
        profile = LazyEDA().profile(df)
        names = [c.name for c in profile.report_data.column_summaries]
        assert names == df.columns

    def test_numeric_column_summary_has_quantiles_and_histogram(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        summary = next(c for c in profile.report_data.column_summaries if c.name == "amount")
        assert summary.dtype_category == "numeric"
        assert summary.quantiles is not None
        assert set(summary.quantiles.keys()) == {"min", "p25", "p50", "p75", "max"}
        assert summary.histogram
        assert sum(bin_.count for bin_ in summary.histogram) <= summary.row_count

    def test_string_column_summary_has_cardinality_and_top_k_histogram(self) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        summary = next(c for c in profile.report_data.column_summaries if c.name == "country")
        assert summary.dtype_category == "string"
        assert summary.cardinality_estimate is not None
        assert summary.histogram
        assert len(summary.histogram) <= 10

    def test_title_and_metadata_pass_through(self) -> None:
        profile = LazyEDA().profile(
            _mixed_dataframe(), title="My Custom Title", metadata={"source": "unit-test"}
        )
        assert profile.report_data.title == "My Custom Title"
        assert profile.report_data.metadata == {"source": "unit-test"}

    def test_export_writes_a_real_html_file(self, tmp_path) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        out_path = tmp_path / "report.html"
        profile.export(out_path)

        assert out_path.exists()
        html = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "country" in html
        assert "amount" in html

    def test_export_accepts_string_path(self, tmp_path) -> None:
        profile = LazyEDA().profile(_mixed_dataframe())
        out_path = tmp_path / "report_str.html"
        profile.export(str(out_path))
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_null_numeric_column_gets_no_sketch_but_still_summarized(self) -> None:
        df = pl.DataFrame({"all_null": pl.Series("all_null", [None, None, None], dtype=pl.Float64)})
        profile = LazyEDA().profile(df)
        assert "all_null" not in profile.quantile_results
        summary = profile.report_data.column_summaries[0]
        assert summary.null_count == 3
        assert summary.quantiles is None
        assert summary.histogram is None

    def test_zero_row_source_does_not_raise(self, tmp_path) -> None:
        df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int64), "b": pl.Series("b", [], dtype=pl.Utf8)})
        profile = LazyEDA().profile(df)
        assert profile.row_count == 0
        # Zero non-null values -> no sketches computed for either column.
        assert profile.quantile_results == {}
        assert profile.cardinality_results == {}
        # export must still succeed and produce valid HTML (report.py's
        # own empty/near-empty handling), not raise.
        out_path = tmp_path / "empty.html"
        profile.export(out_path)
        assert out_path.exists()
        assert "<!DOCTYPE html>" in out_path.read_text(encoding="utf-8")

    def test_custom_tuning_knobs_are_respected(self) -> None:
        df = _mixed_dataframe(n=300)
        lazy_eda = LazyEDA(histogram_bins=4, top_k_categories=2, quantiles=(0.5,), kll_k=64, hll_log2_k=8)
        profile = lazy_eda.profile(df)

        assert set(profile.quantile_results["amount"].quantile_estimates.keys()) == {0.5}
        assert profile.quantile_results["amount"].k == 64
        assert profile.cardinality_results["country"].log2_k == 8

        country_summary = next(
            c for c in profile.report_data.column_summaries if c.name == "country"
        )
        assert len(country_summary.histogram) <= 2

        amount_summary = next(
            c for c in profile.report_data.column_summaries if c.name == "amount"
        )
        assert len(amount_summary.histogram) <= 4

    def test_numeric_column_with_nan_and_inf_does_not_raise(self) -> None:
        """Polars' `drop_nulls()` does NOT drop float NaN (null and NaN are
        distinct concepts there), so a numeric column containing NaN/+-inf
        used to reach quantile estimation / histogram construction
        unfiltered and crash `profile()`. This proves such a column is now
        handled gracefully: quantiles/the histogram are computed only over
        the column's finite values.
        """
        df = pl.DataFrame(
            {
                "amount": pl.Series(
                    "amount",
                    [1.0, 2.0, float("nan"), 3.0, float("inf"), float("-inf"), 4.0, None],
                    dtype=pl.Float64,
                )
            }
        )
        profile = LazyEDA().profile(df)

        summary = profile.report_data.column_summaries[0]
        # 1 explicit null -> reflected in null accounting independently of
        # the finite-value filter applied for sketching.
        assert summary.null_count == 1
        # Quantiles/histogram must exist (finite values [1.0, 2.0, 3.0,
        # 4.0] remain) and must not include NaN/inf as the min/max.
        assert summary.quantiles is not None
        assert math.isfinite(summary.quantiles["min"])
        assert math.isfinite(summary.quantiles["max"])
        assert summary.histogram is not None
