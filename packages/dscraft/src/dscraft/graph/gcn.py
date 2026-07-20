"""Minimal GCN forward pass consuming a :class:`PyGSparseAdapter`.

This is the *one* canonical GCN path in this package (per CLAUDE.md's
"one canonical location per capability" rule) -- there is no second/
parallel message-passing implementation anywhere else here. It uses the
real `torch_geometric.nn.GCNConv` op (per the architecture doc's naming of
`GCNConv` directly and its instruction to prefer using the real library
over hand-rolling message-passing), not a reimplementation of
normalized-adjacency aggregation.

Scope (see package README for the full list of deferred capabilities):
this module implements a single-layer-configurable stack of `GCNConv`
layers only. GAT, GraphSAGE, Graph Transformers/Laplacian Positional
Encodings, neighborhood sampling, and oversmoothing/oversquashing
monitoring are explicitly out of scope for this pass.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from dscraft.graph.sparse import PyGSparseAdapter

_logger = logging.getLogger(__name__)

__all__ = ["GCN", "resolve_device"]


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick a compute device, defaulting to CPU with a clean fallback.

    Per CLAUDE.md: never hardcode ``cuda``. This picks MPS if available
    (the primary backend for this platform per CLAUDE.md), else CUDA if
    available, else CPU -- but always falls back cleanly rather than
    raising if a caller passes an unavailable ``preferred`` device.
    """
    if preferred is not None:
        try:
            device = torch.device(preferred)
            # Smoke-test the device is actually usable.
            torch.zeros(1, device=device)
            return device
        except (RuntimeError, AssertionError) as exc:
            # RuntimeError: malformed device string (e.g. torch.device("not-a-device"))
            # or an unavailable backend at allocation time. AssertionError: PyTorch's
            # own "not compiled with <backend> enabled" check (e.g. CUDA on a
            # CPU/MPS-only build). Both are the expected "this device string is not
            # usable on this machine" case, so fall through to auto-detection. Any
            # other exception type is unexpected and should propagate rather than be
            # silently swallowed.
            _logger.warning(
                "Preferred device %r is not usable (%s: %s); falling back to "
                "auto-detected device.",
                preferred,
                type(exc).__name__,
                exc,
            )

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GCN(nn.Module):
    """A minimal 2-layer Graph Convolutional Network.

    Consumes node features plus a :class:`PyGSparseAdapter` (converted
    internally to COO, PyG's native input format for `GCNConv`) and
    produces per-node output embeddings/logits.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        """Build the two `GCNConv` layers.

        Args:
            in_channels: Size of each input node feature vector.
            hidden_channels: Size of the hidden representation produced by
                the first `GCNConv` layer (after the ReLU).
            out_channels: Size of the final per-node output (e.g. number of
                classes for node classification, or embedding dimension).
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, adapter: PyGSparseAdapter) -> torch.Tensor:
        """Run the forward pass.

        Args:
            x: Node feature matrix of shape ``[num_nodes, in_channels]``.
            adapter: A :class:`PyGSparseAdapter` describing the graph
                structure (any native format -- this converts to COO,
                `GCNConv`'s required input format, internally). If the
                adapter carries per-edge weights, they are passed through
                to each `GCNConv` layer; an unweighted adapter (``None``
                edge weights) preserves `GCNConv`'s default unweighted
                behavior exactly.

        Returns:
            Node output tensor of shape ``[num_nodes, out_channels]``.
        """
        coo_adapter = adapter.to_coo()
        edge_index = coo_adapter.edge_index.to(x.device)
        edge_weight = coo_adapter.edge_weight
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)

        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h
