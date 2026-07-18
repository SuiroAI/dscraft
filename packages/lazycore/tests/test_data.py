"""Tests for lazycore.data (three-tier data/tensor conventions, §2.1)."""

from __future__ import annotations

import abc

import pytest

from lazycore.data import (
    DenseMediaPipeline,
    SparseFormat,
    SparseGraphTensorAdapter,
    from_polars_zero_copy,
    is_arrow_backed_pandas,
    pandas_arrow_dtypes,
    to_polars_zero_copy,
)

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Tier 1: Arrow-backed pandas/polars helpers
# ---------------------------------------------------------------------------


def _arrow_backed_frame() -> "pd.DataFrame":
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"]}
    ).convert_dtypes(dtype_backend="pyarrow")


def _plain_numpy_frame() -> "pd.DataFrame":
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_is_arrow_backed_pandas_true_for_arrow_dtype_frame():
    """A DataFrame with all columns converted to the pyarrow dtype backend is reported as Arrow-backed."""
    assert is_arrow_backed_pandas(_arrow_backed_frame()) is True


def test_is_arrow_backed_pandas_false_for_plain_numpy_frame():
    """A DataFrame with plain numpy-backed columns is reported as not Arrow-backed."""
    assert is_arrow_backed_pandas(_plain_numpy_frame()) is False


def test_is_arrow_backed_pandas_true_for_empty_frame():
    """An empty (columnless) DataFrame is trivially considered Arrow-backed."""
    assert is_arrow_backed_pandas(pd.DataFrame()) is True


def test_pandas_arrow_dtypes_reports_only_arrow_columns():
    """pandas_arrow_dtypes() returns only the names of columns backed by ArrowDtype, excluding a plain numpy float64 column added to the same frame."""
    mixed = _arrow_backed_frame()
    mixed["c"] = [1.0, 2.0, 3.0]  # plain numpy float64 column
    dtypes = pandas_arrow_dtypes(mixed)
    assert set(dtypes) == {"a", "b"}
    assert "c" not in dtypes


def test_roundtrip_pandas_to_polars_and_back_preserves_data():
    """Converting an Arrow-backed pandas frame to Polars and back preserves all values and lands the result back on the Arrow-backed tier."""
    original = _arrow_backed_frame()
    as_polars = to_polars_zero_copy(original)
    assert isinstance(as_polars, pl.DataFrame)
    assert as_polars.to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": ["x", "y", "z"],
    }

    back_to_pandas = from_polars_zero_copy(as_polars)
    assert list(back_to_pandas["a"]) == [1, 2, 3]
    assert list(back_to_pandas["b"]) == ["x", "y", "z"]
    # Round-tripping through this helper should land back on the Arrow tier.
    assert is_arrow_backed_pandas(back_to_pandas) is True


# ---------------------------------------------------------------------------
# Tier 2: sparse graph tensor adapter is an interface, not an implementation
# ---------------------------------------------------------------------------


def test_sparse_graph_tensor_adapter_is_abstract():
    """SparseGraphTensorAdapter is an ABC and cannot be instantiated directly since it has unimplemented abstract methods."""
    assert issubclass(SparseGraphTensorAdapter, abc.ABC)
    with pytest.raises(TypeError):
        SparseGraphTensorAdapter()  # abstract methods not implemented


def test_sparse_format_constants():
    """SparseFormat exposes the expected COO/CSR/CSC string constants."""
    assert SparseFormat.COO == "coo"
    assert SparseFormat.CSR == "csr"
    assert SparseFormat.CSC == "csc"


def test_concrete_sparse_adapter_subclass_satisfies_interface():
    """A minimal concrete subclass implementing all abstract members can be instantiated and its native_format/shape/to_coo() behave as defined."""
    class ToyCooAdapter(SparseGraphTensorAdapter):
        def __init__(self, shape: tuple[int, int]) -> None:
            self._shape = shape

        @property
        def native_format(self) -> str:
            return SparseFormat.COO

        @property
        def shape(self) -> tuple[int, int]:
            return self._shape

        def to_coo(self) -> "ToyCooAdapter":
            return self

        def to_csr(self) -> "ToyCooAdapter":
            raise NotImplementedError("toy adapter only supports COO")

        def to_csc(self) -> "ToyCooAdapter":
            raise NotImplementedError("toy adapter only supports COO")

    adapter = ToyCooAdapter((4, 4))
    assert adapter.native_format == SparseFormat.COO
    assert adapter.shape == (4, 4)
    assert adapter.to_coo() is adapter


# ---------------------------------------------------------------------------
# Tier 3: dense media pipeline is an interface, not an implementation
# ---------------------------------------------------------------------------


def test_dense_media_pipeline_is_abstract():
    """DenseMediaPipeline cannot be instantiated directly since it has unimplemented abstract methods."""
    with pytest.raises(TypeError):
        DenseMediaPipeline()  # abstract methods not implemented


def test_concrete_dense_media_pipeline_runs_decode_augment_export():
    """run() drives a concrete pipeline through decode -> augment -> to_dense_tensor in order, and the resulting tensor genuinely supports the DLPack protocol."""
    calls: list[str] = []

    class ToyTensor:
        def __dlpack__(self, stream=None):
            calls.append("dlpack")
            return object()

        def __dlpack_device__(self):
            return (1, 0)

    class ToyPipeline(DenseMediaPipeline):
        def decode(self, raw: bytes):
            calls.append("decode")
            return raw

        def augment(self, decoded):
            calls.append("augment")
            return decoded

        def to_dense_tensor(self, augmented):
            calls.append("to_dense_tensor")
            return ToyTensor()

    pipeline = ToyPipeline()
    result = pipeline.run(b"fake-image-bytes")

    assert calls == ["decode", "augment", "to_dense_tensor"]
    # The result must genuinely support the DLPack protocol.
    assert result.__dlpack_device__() == (1, 0)
    result.__dlpack__()
    assert calls[-1] == "dlpack"
