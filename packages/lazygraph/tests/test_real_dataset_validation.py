"""Real-dataset validation for `PyGSparseAdapter` + `GCN`.

The existing test suite (`test_sparse.py`, `test_gcn.py`) only exercises
the adapter/GCN against small hand-built synthetic graphs. This module
validates the *same* public API against a real graph with real structure:
`torch_geometric.datasets.KarateClub`, Zachary's Karate Club social
network.

Why `KarateClub` specifically: it is generated in-process from a hardcoded
edge list baked directly into `torch_geometric/datasets/karate.py` (no
`download()`/`raw_dir`/URL fetch anywhere in its implementation) -- it
ships fully bundled inside the already-required `torch_geometric`
dependency. No network access occurs when instantiating it, and no new
third-party dependency is introduced.

This test reuses `PyGSparseAdapter`/`GCN` exactly as the synthetic tests
do (per CLAUDE.md's "one canonical adapter/GCN path" rule) -- it does not
introduce a second graph-loading path.
"""

from __future__ import annotations

import torch
from torch_geometric.datasets import KarateClub

from benchcraft_lazygraph import GCN, PyGSparseAdapter, resolve_device
from lazycore.data import SparseFormat

# Zachary's Karate Club is a fixed, well-known real graph: these are its
# own reported/canonical stats (see torch_geometric.datasets.KarateClub's
# docstring "STATS" table), not values derived from our own computation --
# we assert the adapter/dataset agree with these independently-known facts.
EXPECTED_NUM_NODES = 34
EXPECTED_NUM_EDGES = 156  # directed-both-ways count of 78 undirected edges
EXPECTED_NUM_FEATURES = 34  # identity-matrix node features
EXPECTED_NUM_CLASSES = 4  # 4-faction community split


def _load_karate_club():
    """Load the bundled `KarateClub` dataset and return `(dataset, data)`."""
    dataset = KarateClub()
    data = dataset[0]
    return dataset, data


def test_karate_club_reports_known_real_graph_stats():
    """Sanity-check the dataset itself before trusting the adapter with it."""
    dataset, data = _load_karate_club()

    assert data.num_nodes == EXPECTED_NUM_NODES
    assert data.num_edges == EXPECTED_NUM_EDGES
    assert data.num_node_features == EXPECTED_NUM_FEATURES
    assert int(data.y.max().item()) + 1 == EXPECTED_NUM_CLASSES
    assert dataset.num_classes == EXPECTED_NUM_CLASSES


def test_adapter_from_real_edge_index_matches_known_shape_and_edge_count():
    """Wrap the real KarateClub edge_index in the package's canonical
    adapter and confirm structural facts against the dataset's own known,
    fixed values -- not just "some finite valid graph"."""
    _, data = _load_karate_club()

    adapter = PyGSparseAdapter.from_edge_index(
        data.edge_index, num_nodes=data.num_nodes
    )

    assert adapter.native_format == SparseFormat.COO
    assert adapter.shape == (EXPECTED_NUM_NODES, EXPECTED_NUM_NODES)
    assert adapter.edge_index.shape == (2, EXPECTED_NUM_EDGES)


def test_real_graph_round_trips_through_coo_csr_csc_consistently():
    """The adapter's COO/CSR/CSC conversions must all agree with each other
    and with the real graph's known edge count on the real KarateClub
    structure (not just the synthetic ring graph the other tests use)."""
    _, data = _load_karate_club()
    adapter = PyGSparseAdapter.from_edge_index(
        data.edge_index, num_nodes=data.num_nodes
    )

    csr_adapter = adapter.to_csr()
    csc_adapter = adapter.to_csc()

    assert csr_adapter.native_format == SparseFormat.CSR
    assert csc_adapter.native_format == SparseFormat.CSC
    assert csr_adapter.scipy_matrix.nnz == EXPECTED_NUM_EDGES
    assert csc_adapter.scipy_matrix.nnz == EXPECTED_NUM_EDGES

    dense_coo = adapter.to_coo().to_dense_numpy()
    dense_csr = csr_adapter.to_dense_numpy()
    dense_csc = csc_adapter.to_dense_numpy()

    assert dense_coo.shape == (EXPECTED_NUM_NODES, EXPECTED_NUM_NODES)
    assert (dense_coo == dense_csr).all()
    assert (dense_coo == dense_csc).all()
    assert dense_coo.sum() == EXPECTED_NUM_EDGES

    # Round-trip CSR/CSC back to COO and confirm structure is preserved.
    back_from_csr = csr_adapter.to_coo()
    back_from_csc = csc_adapter.to_coo()
    assert back_from_csr.edge_index.shape == (2, EXPECTED_NUM_EDGES)
    assert back_from_csc.edge_index.shape == (2, EXPECTED_NUM_EDGES)


def test_gcn_forward_pass_on_real_karate_club_features():
    """Run the package's one canonical `GCN` forward pass on the real
    KarateClub node features/structure and confirm finite, correctly
    shaped output. This is a forward-pass sanity check, not a training or
    accuracy claim -- the model is untrained."""
    torch.manual_seed(0)
    device = resolve_device()

    _, data = _load_karate_club()
    adapter = PyGSparseAdapter.from_edge_index(
        data.edge_index, num_nodes=data.num_nodes
    )

    x = data.x.to(device)
    hidden_channels = 8
    out_channels = EXPECTED_NUM_CLASSES  # configure to match the real 4-class split

    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    ).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    # Shape matches (num_nodes, num_classes) when configured for the real
    # faction split -- not an accuracy claim about an untrained network.
    assert out.shape == (EXPECTED_NUM_NODES, out_channels)
    assert torch.isfinite(out).all()


def test_gcn_accepts_csr_backed_real_graph_adapter_too():
    """Confirm the GCN's internal `.to_coo()` conversion also works
    correctly when starting from a CSR-backed real-graph adapter."""
    torch.manual_seed(1)
    _, data = _load_karate_club()
    adapter = PyGSparseAdapter.from_edge_index(
        data.edge_index, num_nodes=data.num_nodes
    ).to_csr()

    model = GCN(in_channels=data.num_node_features, hidden_channels=8, out_channels=4)
    model.eval()

    with torch.no_grad():
        out = model(data.x, adapter)

    assert out.shape == (EXPECTED_NUM_NODES, 4)
    assert torch.isfinite(out).all()
