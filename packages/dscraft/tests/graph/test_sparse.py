"""Tests for `PyGSparseAdapter` (Tier-2 sparse graph tensor adapter).

Constructs a small synthetic graph (8 nodes, a handful of directed edges)
directly, and verifies that `.to_coo()` / `.to_csr()` / `.to_csc()` all
round-trip to a structurally equivalent dense adjacency matrix -- i.e. the
conversion is real (produces correct data), not just a metadata relabel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dscraft.graph import PyGSparseAdapter
from dscraft.core.data import SparseFormat, SparseGraphTensorAdapter


NUM_NODES = 8
# A small ring plus a couple of chords, directed edges (src, dst).
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
    (0, 4), (2, 6),
]


def _make_adapter() -> PyGSparseAdapter:
    """Build the module's fixed COO-native ring-plus-chords `PyGSparseAdapter`."""
    src = [e[0] for e in EDGES]
    dst = [e[1] for e in EDGES]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def _expected_dense() -> np.ndarray:
    """Build the dense adjacency matrix that `EDGES` is expected to produce."""
    dense = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for s, d in EDGES:
        dense[s, d] = 1.0
    return dense


def test_is_concrete_subclass_of_lazycore_adapter():
    """`PyGSparseAdapter` must be a real subclass of the lazycore abstract adapter."""
    adapter = _make_adapter()
    assert isinstance(adapter, SparseGraphTensorAdapter)


def test_native_format_and_shape_on_construction():
    """A freshly constructed adapter reports COO format and the expected dense shape."""
    adapter = _make_adapter()
    assert adapter.native_format == SparseFormat.COO
    assert adapter.shape == (NUM_NODES, NUM_NODES)


def test_coo_round_trip_matches_expected_dense():
    """Calling `.to_coo()` on an already-COO adapter preserves the correct structure."""
    adapter = _make_adapter()
    coo_adapter = adapter.to_coo()
    assert coo_adapter.native_format == SparseFormat.COO
    np.testing.assert_array_equal(coo_adapter.to_dense_numpy(), _expected_dense())


def test_csr_conversion_matches_expected_dense():
    """`.to_csr()` produces a real `scipy.sparse.csr_matrix` with correct structure."""
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    assert csr_adapter.native_format == SparseFormat.CSR
    np.testing.assert_array_equal(csr_adapter.to_dense_numpy(), _expected_dense())
    # Confirm it's a real scipy CSR matrix, not a relabel.
    import scipy.sparse as sp

    assert isinstance(csr_adapter.scipy_matrix, sp.csr_matrix)


def test_csc_conversion_matches_expected_dense():
    """`.to_csc()` produces a real `scipy.sparse.csc_matrix` with correct structure."""
    adapter = _make_adapter()
    csc_adapter = adapter.to_csc()
    assert csc_adapter.native_format == SparseFormat.CSC
    np.testing.assert_array_equal(csc_adapter.to_dense_numpy(), _expected_dense())
    import scipy.sparse as sp

    assert isinstance(csc_adapter.scipy_matrix, sp.csc_matrix)


def test_csr_to_coo_round_trip_preserves_structure():
    """Converting CSR back to COO reconstructs the original adjacency structure."""
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    back_to_coo = csr_adapter.to_coo()
    assert back_to_coo.native_format == SparseFormat.COO
    np.testing.assert_array_equal(back_to_coo.to_dense_numpy(), _expected_dense())


def test_csc_to_coo_round_trip_preserves_structure():
    """Converting CSC back to COO reconstructs the original adjacency structure."""
    adapter = _make_adapter()
    csc_adapter = adapter.to_csc()
    back_to_coo = csc_adapter.to_coo()
    assert back_to_coo.native_format == SparseFormat.COO
    np.testing.assert_array_equal(back_to_coo.to_dense_numpy(), _expected_dense())


def test_all_three_formats_agree_on_dense_reconstruction():
    """COO, CSR, and CSC conversions of the same adapter all yield an identical dense matrix."""
    adapter = _make_adapter()
    dense_coo = adapter.to_coo().to_dense_numpy()
    dense_csr = adapter.to_csr().to_dense_numpy()
    dense_csc = adapter.to_csc().to_dense_numpy()
    np.testing.assert_array_equal(dense_coo, dense_csr)
    np.testing.assert_array_equal(dense_coo, dense_csc)


def test_invalid_edge_index_shape_rejected():
    """`from_edge_index` raises `ValueError` for an edge_index that isn't shape `[2, num_edges]`."""
    bad_edge_index = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError):
        PyGSparseAdapter.from_edge_index(bad_edge_index, num_nodes=NUM_NODES)


def test_edge_index_accessor_requires_coo_format():
    """Accessing `.edge_index` on a CSR-backed adapter raises `RuntimeError`."""
    adapter = _make_adapter()
    csr_adapter = adapter.to_csr()
    with pytest.raises(RuntimeError):
        _ = csr_adapter.edge_index


def test_scipy_matrix_accessor_requires_csr_or_csc_format():
    """Accessing `.scipy_matrix` on a COO-backed adapter raises `RuntimeError`."""
    adapter = _make_adapter()
    with pytest.raises(RuntimeError):
        _ = adapter.scipy_matrix


# -- edge-index contract validation (construction-time rejection) ---------


def _valid_edge_index() -> torch.Tensor:
    src = [e[0] for e in EDGES]
    dst = [e[1] for e in EDGES]
    return torch.tensor([src, dst], dtype=torch.long)


def test_negative_num_nodes_rejected():
    """`num_nodes <= 0` must raise `ValueError` at construction time."""
    edge_index = _valid_edge_index()
    with pytest.raises(ValueError, match="num_nodes"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=-1)


def test_zero_num_nodes_rejected():
    """`num_nodes == 0` must raise `ValueError` at construction time."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="num_nodes"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=0)


def test_non_integer_num_nodes_rejected():
    """A non-integer `num_nodes` (e.g. a float) must raise `ValueError`."""
    edge_index = _valid_edge_index()
    with pytest.raises(ValueError, match="num_nodes"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=8.5)  # type: ignore[arg-type]


def test_out_of_range_node_index_rejected():
    """An edge_index value >= num_nodes must raise a clear `ValueError`
    naming the offending value, not silently corrupt the graph."""
    edge_index = torch.tensor([[0, 1], [1, 100]], dtype=torch.long)
    with pytest.raises(ValueError, match="100"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def test_negative_node_index_rejected():
    """A negative edge_index value must raise a clear `ValueError` naming
    the offending value."""
    edge_index = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="-1"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def test_non_integer_edge_index_dtype_rejected():
    """A float-dtype edge_index must be rejected explicitly rather than
    silently truncated by downstream SciPy/torch_geometric conversion."""
    edge_index = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def test_mismatched_edge_weight_length_rejected():
    """An `edge_weight` vector whose length differs from the number of
    edges must raise a clear `ValueError` at construction time."""
    edge_index = _valid_edge_index()
    wrong_length_weight = torch.ones(len(EDGES) - 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="edge_weight"):
        PyGSparseAdapter.from_edge_index(
            edge_index, num_nodes=NUM_NODES, edge_weight=wrong_length_weight
        )


def test_non_1d_edge_weight_rejected():
    """An `edge_weight` tensor with the right length but the wrong number
    of dimensions (e.g. shape `(num_edges, 1)`) must be rejected at
    construction time rather than silently broadcasting during SciPy
    conversion or `GCNConv` execution."""
    edge_index = _valid_edge_index()
    two_d_weight = torch.ones(len(EDGES), 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="edge_weight"):
        PyGSparseAdapter.from_edge_index(
            edge_index, num_nodes=NUM_NODES, edge_weight=two_d_weight
        )


def test_matching_edge_weight_length_accepted():
    """A correctly-sized `edge_weight` is accepted and preserved."""
    edge_index = _valid_edge_index()
    edge_weight = torch.arange(1, len(EDGES) + 1, dtype=torch.float32)
    adapter = PyGSparseAdapter.from_edge_index(
        edge_index, num_nodes=NUM_NODES, edge_weight=edge_weight
    )
    torch.testing.assert_close(adapter.edge_weight, edge_weight)


def test_malformed_graph_never_reaches_to_csr_or_to_csc():
    """Regression guard: an out-of-range edge index must be rejected at
    `from_edge_index` construction, not survive until `.to_csr()` builds a
    malformed SciPy matrix or a GCN forward pass runs on it."""
    edge_index = torch.tensor([[0, 1], [1, 999]], dtype=torch.long)
    with pytest.raises(ValueError):
        adapter = PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)
        adapter.to_csr()  # should never be reached
