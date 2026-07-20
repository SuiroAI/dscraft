"""Tests for dscraft.eda.report -- the self-contained HTML report renderer."""

from __future__ import annotations

import re

import pytest

from dscraft.eda.report import (
    AssociationMatrix,
    ColumnSummary,
    EDAReportData,
    HistogramBin,
    export_report,
    render_report,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _small_report_data() -> EDAReportData:
    """A handful of columns mixing numeric/categorical summaries, plus a
    small association matrix -- representative of the minimal input shape
    a caller would construct."""
    columns = [
        ColumnSummary(
            name="age",
            dtype_category="numeric",
            null_count=2,
            null_percentage=2.0,
            row_count=100,
            quantiles={"min": 18.0, "p25": 29.0, "p50": 41.0, "p75": 55.0, "max": 92.0},
            histogram=[
                HistogramBin(label="[18, 30)", count=25),
                HistogramBin(label="[30, 45)", count=40),
                HistogramBin(label="[45, 92]", count=33),
            ],
        ),
        ColumnSummary(
            name="country",
            dtype_category="string",
            null_count=0,
            null_percentage=0.0,
            row_count=100,
            cardinality_estimate=12,
            histogram=[
                HistogramBin(label="US", count=60),
                HistogramBin(label="CA", count=25),
                HistogramBin(label="MX", count=15),
            ],
        ),
        ColumnSummary(
            name="is_active",
            dtype_category="boolean",
            null_count=50,
            null_percentage=50.0,
            row_count=100,
        ),
    ]
    association_matrix = AssociationMatrix(
        column_names=["age", "country", "is_active"],
        values=[
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.05],
            [-0.1, 0.05, 1.0],
        ],
    )
    return EDAReportData(
        column_summaries=columns,
        association_matrix=association_matrix,
        title="Sample Dataset EDA",
        row_count=100,
        metadata={"source": "synthetic_test_fixture.parquet"},
    )


def _large_report_data(num_columns: int = 50, num_bins: int = 12) -> EDAReportData:
    """A larger, but not absurd, synthetic dataset for the size-budget test:
    50 columns of summary stats (half numeric-with-histogram, half
    categorical-with-histogram) plus a 50x50 association matrix."""
    columns: list[ColumnSummary] = []
    for i in range(num_columns):
        name = f"feature_{i:03d}_with_a_moderately_descriptive_name"
        if i % 2 == 0:
            columns.append(
                ColumnSummary(
                    name=name,
                    dtype_category="numeric",
                    null_count=i,
                    null_percentage=round(i / num_columns * 37.5, 2),
                    row_count=10_000,
                    quantiles={
                        "min": float(i),
                        "p25": float(i * 2),
                        "p50": float(i * 3),
                        "p75": float(i * 4),
                        "max": float(i * 5),
                    },
                    histogram=[
                        HistogramBin(label=f"[{b * 10}, {(b + 1) * 10})", count=(i + 1) * (b + 1))
                        for b in range(num_bins)
                    ],
                )
            )
        else:
            columns.append(
                ColumnSummary(
                    name=name,
                    dtype_category="string",
                    null_count=i,
                    null_percentage=round(i / num_columns * 12.5, 2),
                    row_count=10_000,
                    cardinality_estimate=i + 3,
                    histogram=[
                        HistogramBin(label=f"category_value_{b:02d}", count=(i + 1) * (b + 1))
                        for b in range(num_bins)
                    ],
                )
            )

    names = [c.name for c in columns]
    values = [
        [1.0 if row == col else round(((row + 1) * (col + 1) % 97) / 97.0 - 0.5, 4) for col in range(num_columns)]
        for row in range(num_columns)
    ]
    association_matrix = AssociationMatrix(column_names=names, values=values)

    return EDAReportData(
        column_summaries=columns,
        association_matrix=association_matrix,
        title="Large Synthetic EDA Report",
        row_count=10_000,
        metadata={"source": "large_synthetic_fixture.parquet", "generated_by": "test_report.py"},
    )


# ---------------------------------------------------------------------------
# Basic rendering sanity
# ---------------------------------------------------------------------------


class TestRenderReportBasics:
    def test_returns_non_empty_string(self) -> None:
        html = render_report(_small_report_data())
        assert isinstance(html, str)
        assert len(html) > 0

    def test_contains_basic_html_structure(self) -> None:
        html = render_report(_small_report_data())
        assert "<html" in html
        assert "<head" in html
        assert "<body>" in html
        assert "</html>" in html
        assert "<!DOCTYPE html>" in html

    def test_contains_column_names_and_title(self) -> None:
        data = _small_report_data()
        html = render_report(data)
        assert "Sample Dataset EDA" in html
        assert "age" in html
        assert "country" in html
        assert "is_active" in html

    def test_html_escapes_column_names(self) -> None:
        data = EDAReportData(
            column_summaries=[
                ColumnSummary(
                    name="<script>alert(1)</script>",
                    dtype_category="string",
                    null_count=0,
                    null_percentage=0.0,
                    row_count=10,
                )
            ],
            title="Escaping Test",
        )
        html = render_report(data)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_collapsible_details_present_and_collapsed_by_default(self) -> None:
        html = render_report(_small_report_data())
        assert "<details" in html
        assert "<summary>" in html
        # Collapsed by default means no `open` attribute on the <details> tag.
        assert not re.search(r"<details[^>]*\bopen\b", html)


# ---------------------------------------------------------------------------
# Size budget
# ---------------------------------------------------------------------------


class TestSizeBudget:
    def test_representative_dataset_under_500kb(self) -> None:
        data = _large_report_data(num_columns=50, num_bins=12)
        html = render_report(data)
        size_bytes = len(html.encode("utf-8"))
        assert size_bytes < 500_000, (
            f"Rendered report for 50 columns / 50x50 association matrix was "
            f"{size_bytes} bytes, expected under 500,000."
        )


# ---------------------------------------------------------------------------
# No external CDN references
# ---------------------------------------------------------------------------


class TestNoExternalReferences:
    def test_no_external_script_src(self) -> None:
        html = render_report(_large_report_data(num_columns=10, num_bins=5))
        assert not re.search(r'<script[^>]+src\s*=\s*"https?://', html, re.IGNORECASE)

    def test_no_external_link_href(self) -> None:
        html = render_report(_large_report_data(num_columns=10, num_bins=5))
        assert not re.search(r'<link[^>]+href\s*=\s*"https?://', html, re.IGNORECASE)

    def test_no_false_positive_from_column_name_containing_http_substring(self) -> None:
        """A column name that happens to contain 'http' must not be mistaken
        for an external resource reference by the check above."""
        data = EDAReportData(
            column_summaries=[
                ColumnSummary(
                    name="https://this-looks-like-a-url-but-is-just-a-column-name",
                    dtype_category="string",
                    null_count=0,
                    null_percentage=0.0,
                    row_count=5,
                )
            ],
            title="URL-like column name test",
        )
        html = render_report(data)
        # The literal substring is present somewhere (as escaped text content)...
        assert "https://this-looks-like-a-url-but-is-just-a-column-name" in html
        # ...but never inside a <script src=...> or <link href=...> attribute.
        assert not re.search(r'<script[^>]+src\s*=\s*"https?://', html, re.IGNORECASE)
        assert not re.search(r'<link[^>]+href\s*=\s*"https?://', html, re.IGNORECASE)

    def test_no_script_tags_have_src_attribute_at_all(self) -> None:
        """Stronger check: every <script ...> opening tag in the document has
        no src attribute whatsoever (all script content is inlined)."""
        html = render_report(_small_report_data())
        script_open_tags = re.findall(r"<script\b[^>]*>", html, re.IGNORECASE)
        assert script_open_tags, "expected at least one <script> tag in the report"
        for tag in script_open_tags:
            assert "src=" not in tag.lower()

    def test_no_link_tags_at_all(self) -> None:
        """This module never needs a <link> tag (no external stylesheets,
        no favicon) -- CSS is inlined via <style> only."""
        html = render_report(_small_report_data())
        assert "<link" not in html.lower()


# ---------------------------------------------------------------------------
# export_report
# ---------------------------------------------------------------------------


class TestExportReport:
    def test_writes_file_matching_render_report_output(self, tmp_path) -> None:
        data = _small_report_data()
        out_path = tmp_path / "report.html"
        export_report(data, out_path)

        assert out_path.exists()
        written = out_path.read_text(encoding="utf-8")
        assert written == render_report(data)

    def test_accepts_string_path(self, tmp_path) -> None:
        data = _small_report_data()
        out_path = tmp_path / "report_str_path.html"
        export_report(data, str(out_path))
        assert out_path.exists()
        assert len(out_path.read_text(encoding="utf-8")) > 0


# ---------------------------------------------------------------------------
# Empty / missing data handling
# ---------------------------------------------------------------------------


class TestEmptyDataHandling:
    def test_empty_column_summaries_renders_no_data_message_instead_of_raising(self) -> None:
        data = EDAReportData(column_summaries=[], title="Empty Report")
        html = render_report(data)
        assert isinstance(html, str)
        assert "<html" in html and "<body>" in html
        assert "No column data available" in html
        # Still a self-contained, CDN-free document even in the empty case.
        assert not re.search(r'<script[^>]+src\s*=\s*"https?://', html, re.IGNORECASE)
        assert not re.search(r'<link[^>]+href\s*=\s*"https?://', html, re.IGNORECASE)

    def test_report_without_association_matrix(self) -> None:
        data = EDAReportData(
            column_summaries=[
                ColumnSummary(
                    name="only_col",
                    dtype_category="numeric",
                    null_count=0,
                    null_percentage=0.0,
                    row_count=5,
                )
            ],
            association_matrix=None,
            title="No Association Matrix",
        )
        html = render_report(data)
        # The canvas element itself must not be rendered when there is no
        # association matrix, even though the shared inline JS (which
        # unconditionally defines drawHeatmap/getElementById("association-heatmap")
        # for the case where a matrix *is* present) still mentions the id by name.
        assert '<canvas id="association-heatmap"' not in html
        assert "only_col" in html

    def test_report_without_metadata(self) -> None:
        data = EDAReportData(
            column_summaries=[
                ColumnSummary(
                    name="c",
                    dtype_category="numeric",
                    null_count=0,
                    null_percentage=0.0,
                    row_count=1,
                )
            ]
        )
        html = render_report(data)
        assert "<html" in html


# ---------------------------------------------------------------------------
# AssociationMatrix validation
# ---------------------------------------------------------------------------


class TestAssociationMatrixValidation:
    def test_non_square_values_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AssociationMatrix(column_names=["a", "b"], values=[[1.0, 0.5]])

    def test_mismatched_row_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            AssociationMatrix(column_names=["a", "b"], values=[[1.0, 0.5], [0.5]])

    def test_valid_square_matrix_constructs(self) -> None:
        matrix = AssociationMatrix(column_names=["a", "b"], values=[[1.0, 0.5], [0.5, 1.0]])
        assert matrix.column_names == ["a", "b"]
