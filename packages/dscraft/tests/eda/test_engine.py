"""Tests for dscraft.eda.engine -- the lazy execution foundation."""

from __future__ import annotations

import polars as pl
import pytest

from dscraft.eda.engine import (
    EngineProfile,
    NullReport,
    SchemaReport,
    load_lazy,
    profile_engine,
    profile_nulls,
    profile_schema,
)


def _sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alice", "bob", None, "dana"],
            "score": [1.5, None, 3.5, None],
        }
    )


# ---------------------------------------------------------------------------
# load_lazy: input normalization across all four accepted forms
# ---------------------------------------------------------------------------


class TestLoadLazyInputForms:
    def test_from_lazyframe_returns_equivalent_frame(self) -> None:
        lf = _sample_dataframe().lazy()
        result = load_lazy(lf)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().equals(lf.collect())

    def test_from_dataframe_converts_via_lazy(self) -> None:
        df = _sample_dataframe()
        result = load_lazy(df)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().equals(df)

    def test_from_parquet_path(self, tmp_path) -> None:
        df = _sample_dataframe()
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        result = load_lazy(path)
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().equals(df)

    def test_from_parquet_path_as_string(self, tmp_path) -> None:
        df = _sample_dataframe()
        path = tmp_path / "test.parquet"
        df.write_parquet(path)

        result = load_lazy(str(path))
        assert isinstance(result, pl.LazyFrame)
        assert result.collect().equals(df)

    def test_from_csv_path(self, tmp_path) -> None:
        df = _sample_dataframe()
        path = tmp_path / "test.csv"
        df.write_csv(path)

        result = load_lazy(path)
        assert isinstance(result, pl.LazyFrame)
        # CSV round-trips nulls but not necessarily identical dtypes to the
        # in-memory frame (e.g. int64 stays int64 here since there are no
        # missing ints), so compare via profile_nulls-relevant behavior:
        # equal values, same shape.
        collected = result.collect()
        assert collected.shape == df.shape
        assert collected["id"].to_list() == df["id"].to_list()
        assert collected["name"].to_list() == df["name"].to_list()

    def test_all_four_input_forms_produce_equivalent_null_reports(self, tmp_path) -> None:
        df = _sample_dataframe()
        parquet_path = tmp_path / "equiv.parquet"
        csv_path = tmp_path / "equiv.csv"
        df.write_parquet(parquet_path)
        df.write_csv(csv_path)

        sources = [df.lazy(), df, parquet_path, csv_path]
        reports = [profile_nulls(load_lazy(source)) for source in sources]

        first = reports[0]
        for report in reports[1:]:
            assert report.null_counts == first.null_counts
            assert report.total_rows == first.total_rows
            assert report.null_percentages == pytest.approx(first.null_percentages)


# ---------------------------------------------------------------------------
# load_lazy: error cases
# ---------------------------------------------------------------------------


class TestLoadLazyErrors:
    def test_nonexistent_path_raises_file_not_found(self, tmp_path) -> None:
        missing = tmp_path / "does_not_exist.parquet"
        with pytest.raises(FileNotFoundError):
            load_lazy(missing)

    def test_unsupported_extension_raises_value_error(self, tmp_path) -> None:
        path = tmp_path / "data.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match=r"\.json"):
            load_lazy(path)

    def test_wrong_input_type_raises_type_error_naming_the_type(self) -> None:
        with pytest.raises(TypeError, match="int"):
            load_lazy(42)  # type: ignore[arg-type]

    def test_directory_path_raises_value_error(self, tmp_path) -> None:
        directory = tmp_path / "a_directory.parquet"
        directory.mkdir()
        with pytest.raises(ValueError):
            load_lazy(directory)


# ---------------------------------------------------------------------------
# profile_nulls: correctness against known null counts
# ---------------------------------------------------------------------------


class TestProfileNulls:
    def test_known_null_counts_and_percentages(self) -> None:
        df = pl.DataFrame(
            {
                "no_nulls": [1, 2, 3, 4],
                "half_null": [1, None, 3, None],
                "all_null": [None, None, None, None],
            }
        )
        report = profile_nulls(df.lazy())

        assert isinstance(report, NullReport)
        assert report.total_rows == 4
        assert report.null_counts == {"no_nulls": 0, "half_null": 2, "all_null": 4}
        assert report.null_percentages["no_nulls"] == pytest.approx(0.0)
        assert report.null_percentages["half_null"] == pytest.approx(50.0)
        assert report.null_percentages["all_null"] == pytest.approx(100.0)
        assert report.columns_with_nulls() == ["half_null", "all_null"]

    def test_empty_dataframe_zero_rows_no_division_error(self) -> None:
        df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int64)})
        report = profile_nulls(df.lazy())
        assert report.total_rows == 0
        assert report.null_counts == {"a": 0}
        assert report.null_percentages == {"a": 0.0}

    def test_no_columns_returns_empty_report(self) -> None:
        df = pl.DataFrame()
        report = profile_nulls(df.lazy())
        assert report.null_counts == {}
        assert report.null_percentages == {}
        assert report.total_rows == 0


# ---------------------------------------------------------------------------
# profile_schema: dtype + coarse category classification
# ---------------------------------------------------------------------------


class TestProfileSchema:
    def test_mixed_dtype_categories(self) -> None:
        import datetime as dt

        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "date_col": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
                "datetime_col": [
                    dt.datetime(2024, 1, 1, 0, 0),
                    dt.datetime(2024, 1, 2, 0, 0),
                    dt.datetime(2024, 1, 3, 0, 0),
                ],
            }
        )
        report = profile_schema(df.lazy())
        assert isinstance(report, SchemaReport)

        by_name = report.by_name()
        assert by_name["int_col"].category == "numeric"
        assert by_name["float_col"].category == "numeric"
        assert by_name["str_col"].category == "string"
        assert by_name["bool_col"].category == "boolean"
        assert by_name["date_col"].category == "temporal"
        assert by_name["datetime_col"].category == "temporal"

    def test_column_order_preserved(self) -> None:
        df = pl.DataFrame({"z": [1], "a": [2], "m": [3]})
        report = profile_schema(df.lazy())
        assert [c.name for c in report.columns] == ["z", "a", "m"]

    def test_names_in_category(self) -> None:
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [3, 4]})
        report = profile_schema(df.lazy())
        assert report.names_in_category("numeric") == ["a", "c"]
        assert report.names_in_category("string") == ["b"]


# ---------------------------------------------------------------------------
# profile_engine: the combined entry point
# ---------------------------------------------------------------------------


class TestProfileEngine:
    def test_row_count_correct(self) -> None:
        df = pl.DataFrame({"a": list(range(37))})
        profile = profile_engine(df)
        assert isinstance(profile, EngineProfile)
        assert profile.row_count == 37
        assert profile.null_report.total_rows == 37

    def test_bundles_schema_and_nulls_consistently(self) -> None:
        df = _sample_dataframe()
        profile = profile_engine(df)

        assert set(profile.null_report.null_counts.keys()) == {"id", "name", "score"}
        assert {c.name for c in profile.schema_report.columns} == {"id", "name", "score"}
        assert profile.null_report.null_counts["name"] == 1
        assert profile.null_report.null_counts["score"] == 2

    def test_accepts_path_source_directly(self, tmp_path) -> None:
        df = _sample_dataframe()
        path = tmp_path / "profile_source.parquet"
        df.write_parquet(path)

        profile = profile_engine(path)
        assert profile.row_count == 4
        assert profile.lazyframe.collect().equals(df)

    def test_returned_lazyframe_is_still_lazy(self) -> None:
        df = _sample_dataframe()
        profile = profile_engine(df)
        assert isinstance(profile.lazyframe, pl.LazyFrame)
