"""Runnable demo of dscraft.graph's sparse adapter + GCN capability.

Section 1 builds a small synthetic Erdos-Renyi random graph, wraps its
edge list in `PyGSparseAdapter`, converts it to CSR (`scipy.sparse`
bridge) purely to demonstrate the format conversion, converts back to COO
(PyG's required input format), and runs a two-layer `GCN` forward pass
over random node features -- printing the resulting output shape.

Section 2 runs the exact same adapter/GCN pipeline against a real graph:
`torch_geometric.datasets.KarateClub` (Zachary's Karate Club), which ships
fully bundled inside the already-required `torch_geometric` dependency --
no network access, no new dependency. See
`tests/test_real_dataset_validation.py` for the full correctness
assertions against this dataset's known real stats; this section just
demonstrates and prints them.

This script only *calls* the package's public API -- it does not
reimplement any adapter/GCN logic inline (per CLAUDE.md's "no net-new
scripts" rule).

Run with:
    python packages/dscraft/examples/graph/gcn_example.py
"""

from __future__ import annotations

import random

import torch
from torch_geometric.datasets import KarateClub

from dscraft.graph import GCN, PyGSparseAdapter, resolve_device


def make_erdos_renyi_edge_index(num_nodes: int, p: float, seed: int = 0) -> torch.Tensor:
    """Build a small directed Erdos-Renyi random graph's edge_index.

    Each ordered pair (i, j) with i != j is included independently with
    probability `p`. Self-loops are not added here -- `GCNConv` adds its
    own self-loops internally as part of its symmetric normalization.
    """
    rng = random.Random(seed)
    src: list[int] = []
    dst: list[int] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and rng.random() < p:
                src.append(i)
                dst.append(j)
    if not src:
        # Guarantee at least a minimal ring so the graph isn't empty.
        src = list(range(num_nodes))
        dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    return torch.tensor([src, dst], dtype=torch.long)


def run_synthetic_section(device: torch.device) -> None:
    """Build a synthetic Erdos-Renyi graph and run it through the adapter/GCN.

    Wraps the generated edge list in `PyGSparseAdapter`, converts it to CSR
    then CSC purely to demonstrate the real `scipy.sparse` format bridge,
    then runs a two-layer `GCN` forward pass over random node features and
    asserts the output has the expected shape and is entirely finite.
    """
    print("=" * 70)
    print("Section 1: synthetic Erdos-Renyi graph")
    print("=" * 70)

    num_nodes = 12
    in_channels = 6
    hidden_channels = 16
    out_channels = 4

    edge_index = make_erdos_renyi_edge_index(num_nodes, p=0.2, seed=42)
    print(f"Synthetic Erdos-Renyi graph: {num_nodes} nodes, {edge_index.shape[1]} directed edges")

    adapter = PyGSparseAdapter.from_edge_index(edge_index, num_nodes=num_nodes)
    print(f"Adapter native format: {adapter.native_format}, shape: {adapter.shape}")

    # Demonstrate the real SciPy sparse bridge (§2.1's Tier-2 conversion).
    csr_adapter = adapter.to_csr()
    print(
        f"Converted to CSR via scipy.sparse: "
        f"{type(csr_adapter.scipy_matrix).__name__}, nnz={csr_adapter.scipy_matrix.nnz}"
    )

    csc_adapter = adapter.to_csc()
    print(
        f"Converted to CSC via scipy.sparse: "
        f"{type(csc_adapter.scipy_matrix).__name__}, nnz={csc_adapter.scipy_matrix.nnz}"
    )

    # GCNConv needs COO input (PyG's native format) -- the adapter handles
    # this conversion internally regardless of which format is passed in.
    torch.manual_seed(42)
    node_features = torch.randn(num_nodes, in_channels, device=device)

    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    model.eval()

    with torch.no_grad():
        output = model(node_features, csr_adapter)

    print(f"GCN forward pass output shape: {tuple(output.shape)}")
    assert output.shape == (num_nodes, out_channels)
    assert torch.isfinite(output).all(), "GCN produced non-finite output"
    print("GCN forward pass produced finite output of the expected shape.")


def run_real_dataset_section(device: torch.device) -> None:
    """Run the same adapter/GCN pipeline against the real KarateClub graph.

    Loads `torch_geometric.datasets.KarateClub` (bundled in-process, no
    network access), wraps its edge index in the same `PyGSparseAdapter`
    used for the synthetic section, converts to CSR/CSC to show the format
    bridge still works on real data, and runs a `GCN` forward pass
    configured for the dataset's real 4-class faction split.
    """
    print()
    print("=" * 70)
    print("Section 2: real graph -- torch_geometric.datasets.KarateClub")
    print("=" * 70)

    # KarateClub is generated in-process from a hardcoded edge list baked
    # directly into torch_geometric -- no download(), no raw_dir, no URL
    # fetch. It ships fully bundled inside the already-required
    # torch_geometric dependency, so this requires zero network access and
    # zero new dependencies.
    dataset = KarateClub()
    data = dataset[0]

    print(
        f"Real KarateClub graph: {data.num_nodes} nodes, {data.num_edges} "
        f"directed edges, {data.num_node_features} node features, "
        f"{dataset.num_classes} classes (ground-truth factions)"
    )

    # Reuses the exact same PyGSparseAdapter/GCN public API as Section 1 --
    # no parallel graph-loading path for real data.
    adapter = PyGSparseAdapter.from_edge_index(data.edge_index, num_nodes=data.num_nodes)
    print(f"Adapter native format: {adapter.native_format}, shape: {adapter.shape}")

    csr_adapter = adapter.to_csr()
    csc_adapter = adapter.to_csc()
    print(
        f"Converted to CSR via scipy.sparse: nnz={csr_adapter.scipy_matrix.nnz}; "
        f"converted to CSC via scipy.sparse: nnz={csc_adapter.scipy_matrix.nnz}"
    )

    x = data.x.to(device)
    # Configure out_channels to match the real 4-class faction split. This
    # is a forward-pass shape/finiteness check only -- the model below is
    # untrained, so its output is not evaluated for classification accuracy.
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
    ).to(device)
    model.eval()

    with torch.no_grad():
        output = model(x, csr_adapter)

    print(f"GCN forward pass output shape: {tuple(output.shape)}")
    assert output.shape == (data.num_nodes, dataset.num_classes)
    assert torch.isfinite(output).all(), "GCN produced non-finite output"
    print("GCN forward pass produced finite output of the expected shape on the real graph.")


def main() -> None:
    """Resolve a compute device and run both the synthetic and real demo sections."""
    device = resolve_device()
    print(f"Using device: {device}")
    print()

    run_synthetic_section(device)
    run_real_dataset_section(device)


if __name__ == "__main__":
    main()
