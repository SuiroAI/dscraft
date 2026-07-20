"""Tests for the minimal GCN forward pass (`dscraft.graph.gcn.GCN`).

Builds a small synthetic graph via `PyGSparseAdapter`, feeds it random
node features through `GCN`, and verifies the output has the expected
shape and is entirely finite (no NaN/Inf from a broken forward pass).
"""

from __future__ import annotations

import torch

from dscraft.graph import GCN, PyGSparseAdapter, resolve_device


NUM_NODES = 10
IN_CHANNELS = 5
HIDDEN_CHANNELS = 8
OUT_CHANNELS = 3


def _make_adapter() -> PyGSparseAdapter:
    """Build a small bidirectional ring-graph `PyGSparseAdapter` for tests."""
    # Small ring graph, made bidirectional (undirected) so every node has
    # at least one neighbor for message passing.
    src = list(range(NUM_NODES)) + list(range(1, NUM_NODES)) + [0]
    dst = list(range(1, NUM_NODES)) + [0] + list(range(NUM_NODES - 1))
    # (src -> next node) and (dst -> previous node), i.e. both directions.
    forward_src = list(range(NUM_NODES))
    forward_dst = [(i + 1) % NUM_NODES for i in range(NUM_NODES)]
    all_src = forward_src + forward_dst
    all_dst = forward_dst + forward_src
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    return PyGSparseAdapter.from_edge_index(edge_index, num_nodes=NUM_NODES)


def test_gcn_forward_pass_output_shape_and_finiteness():
    """A COO-backed adapter through `GCN` yields finite, correctly shaped output."""
    torch.manual_seed(0)
    device = resolve_device()

    adapter = _make_adapter()
    x = torch.randn(NUM_NODES, IN_CHANNELS, device=device)

    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS).to(device)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


def test_gcn_accepts_csr_backed_adapter_too():
    """The adapter's native format shouldn't matter to the GCN -- it
    converts to COO internally via `.to_coo()`."""
    torch.manual_seed(1)
    adapter = _make_adapter().to_csr()

    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()


def test_resolve_device_falls_back_cleanly_for_bogus_preference():
    """Passing an unavailable/invalid preferred device string should not raise --
    `resolve_device` should fall through to auto-detection instead."""
    device = resolve_device(preferred="not-a-real-device")
    assert device.type in ("cpu", "mps", "cuda")


def test_resolve_device_default_never_raises():
    """Calling `resolve_device` with no arguments always returns a valid `torch.device`."""
    device = resolve_device()
    assert isinstance(device, torch.device)


def _make_weighted_adapter(edge_weight: torch.Tensor | None) -> PyGSparseAdapter:
    """Build the same fixed ring-graph adapter as `_make_adapter`, optionally
    carrying the given `edge_weight`."""
    forward_src = list(range(NUM_NODES))
    forward_dst = [(i + 1) % NUM_NODES for i in range(NUM_NODES)]
    all_src = forward_src + forward_dst
    all_dst = forward_dst + forward_src
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    return PyGSparseAdapter.from_edge_index(
        edge_index, num_nodes=NUM_NODES, edge_weight=edge_weight
    )


def test_gcn_forward_actually_uses_edge_weight_from_adapter():
    """Regression test: `GCN.forward` must read `edge_weight` off the
    adapter and pass it through to `GCNConv`, not silently degrade a
    weighted graph to unweighted.

    Builds the identical graph structure/features twice -- once with
    distinct, non-uniform edge weights and once unweighted (`edge_weight=
    None`) -- runs both through the same untrained `GCN`, and asserts the
    outputs are numerically different. If `edge_weight` were being ignored
    (the bug being regression-tested), these two outputs would be
    bit-for-bit identical.
    """
    torch.manual_seed(0)
    num_edges = 2 * NUM_NODES
    # Distinct, non-uniform weights (not all-ones/uniform, which would be
    # numerically indistinguishable from the unweighted default).
    edge_weight = torch.linspace(0.1, 3.0, steps=num_edges, dtype=torch.float32)

    weighted_adapter = _make_weighted_adapter(edge_weight)
    unweighted_adapter = _make_weighted_adapter(None)

    x = torch.randn(NUM_NODES, IN_CHANNELS)

    torch.manual_seed(42)
    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    model.eval()

    with torch.no_grad():
        out_weighted = model(x, weighted_adapter)
        out_unweighted = model(x, unweighted_adapter)

    assert out_weighted.shape == out_unweighted.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out_weighted).all()
    assert torch.isfinite(out_unweighted).all()
    # The real numerical assertion: weighted output must differ from
    # unweighted output given the same model/features/structure.
    assert not torch.allclose(out_weighted, out_unweighted), (
        "GCN.forward produced identical output for weighted vs. unweighted "
        "adapters -- edge_weight is being ignored, not passed through to "
        "GCNConv."
    )


def test_gcn_forward_unweighted_adapter_unchanged_behavior():
    """An adapter with `edge_weight=None` must still pass `None` through to
    `GCNConv` (preserving exact prior unweighted behavior) -- calling
    `GCN.forward` on it must not raise and must produce finite output."""
    torch.manual_seed(7)
    adapter = _make_weighted_adapter(None)
    assert adapter.edge_weight is None

    x = torch.randn(NUM_NODES, IN_CHANNELS)
    model = GCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)
    model.eval()

    with torch.no_grad():
        out = model(x, adapter)

    assert out.shape == (NUM_NODES, OUT_CHANNELS)
    assert torch.isfinite(out).all()
