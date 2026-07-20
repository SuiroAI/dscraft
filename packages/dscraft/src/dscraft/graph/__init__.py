"""dscraft.graph: DSCraft's graph ML module.

This scaffold-depth pass implements exactly one signature capability from
the architecture doc (Part 3, "Module 4: LazyGraph"): a concrete Tier-2
sparse graph tensor adapter (`PyGSparseAdapter`, bridging PyTorch
Geometric's native COO `edge_index` and `scipy.sparse`'s CSR/CSC formats)
plus a minimal GCN forward pass (`GCN`, built on `torch_geometric.nn.GCNConv`)
that consumes it.

GAT/GraphSAGE, Graph Transformers/Laplacian Positional Encodings, the
SQL-to-graph mapping engine, neighborhood-sampling paradigms (node-wise/
layer-wise/LADIES), and oversmoothing/oversquashing monitoring are
explicitly out of scope for this pass -- see the package README's
"Deferred" section.

Public API surface (the one canonical sparse adapter and one canonical
GCN path in this package -- no parallel implementations exist elsewhere):
    >>> from dscraft.graph import PyGSparseAdapter, GCN, resolve_device
"""

from dscraft.graph.gcn import GCN, resolve_device
from dscraft.graph.sparse import PyGSparseAdapter

__all__ = [
    "PyGSparseAdapter",
    "GCN",
    "resolve_device",
]

__version__ = "0.1.0"
