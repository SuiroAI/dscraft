# benchcraft-graph

Benchcraft's graph ML module (internal codename "LazyGraph", architecture
doc Part 3 "Module 4: LazyGraph"). This is a **scaffold-depth pass**, not a
full implementation of the module's eventual scope.

## What this package is (and isn't) right now

The full LazyGraph module is designed around a much larger scope: MPNNs
(GCN/GAT/GraphSAGE) and Global Graph Transformers with Laplacian
Positional Encodings, unified across PyTorch Geometric (COO-native) and
DGL (CSR/CSC-native) via a zero-copy Universal Sparse Tensor layer; an
automated SQL-to-graph mapping engine; three neighborhood-sampling
paradigms (node-wise, layer-wise/FastGCN, LADIES); and training-pathology
monitoring for oversmoothing (Dirichlet-Energy/Rayleigh-Coefficient) and
oversquashing (Jacobian-sensitivity/discrete-curvature).

**This package implements exactly one signature capability: a concrete
Tier-2 sparse graph tensor adapter (COO <-> CSR/CSC bridge) plus a minimal
GCN forward pass built on top of it.** Everything else listed above is
explicitly out of scope for this pass -- see "Deferred" below.

## The signature capability

### 1. `PyGSparseAdapter` — the first concrete Tier-2 adapter

`lazycore.data.SparseGraphTensorAdapter` (in `packages/lazycore`) defines
only the *abstract shape* of the COO/CSR-CSC conversion boundary named in
architecture doc §2.1 ("Universal Sparse Tensor" concept) -- it
deliberately depends on nothing graph-related. `benchcraft_lazygraph.PyGSparseAdapter`
is the first concrete subclass: it wraps a PyTorch Geometric-native COO
`edge_index` tensor (`[2, num_edges]`) and bridges it to `scipy.sparse`'s
CSR/CSC formats.

This is a **real conversion**, not a metadata relabel -- DLPack cannot
represent sparsity, so every `.to_coo()`/`.to_csr()`/`.to_csc()` call
actually (re)builds a sparse structure in the target format:

```python
import torch
from benchcraft_lazygraph import PyGSparseAdapter

edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
adapter = PyGSparseAdapter.from_edge_index(edge_index, num_nodes=4)

adapter.native_format   # "coo"
csr = adapter.to_csr()  # real scipy.sparse.csr_matrix build
csc = adapter.to_csc()  # real scipy.sparse.csc_matrix build
coo_again = csr.to_coo()  # reconstructs a fresh edge_index from SciPy's COO view
```

### 2. `GCN` — a minimal GCN forward pass consuming the adapter

`benchcraft_lazygraph.GCN` is a small two-layer Graph Convolutional
Network built on `torch_geometric.nn.GCNConv` (the real library op named
directly in the architecture doc, not a hand-rolled reimplementation of
normalized-adjacency message passing). It takes node features plus a
`PyGSparseAdapter` in *any* native format and converts to COO internally
(PyG's required input format for `GCNConv`) before running the forward
pass -- demonstrating that the adapter is genuinely load-bearing, not just
a data-shuffling exercise:

```python
from benchcraft_lazygraph import GCN, resolve_device
import torch

device = resolve_device()  # MPS > CUDA > CPU, always falls back cleanly
model = GCN(in_channels=6, hidden_channels=16, out_channels=4).to(device)

x = torch.randn(12, 6, device=device)
out = model(x, adapter)  # adapter can be COO-, CSR-, or CSC-backed
```

See `examples/gcn_example.py` for a complete runnable demo (builds a
synthetic Erdos-Renyi graph, converts through CSR/CSC, runs the GCN
forward pass, prints the output shape), and `tests/test_sparse.py` /
`tests/test_gcn.py` for the correctness test suite.

## Validation: synthetic graph and a real, bundled dataset

The test suite validates the adapter/GCN pipeline against two kinds of
graphs:

- **Synthetic**: a small hand-built ring/chord graph (`tests/test_sparse.py`)
  and a small synthetic ring graph (`tests/test_gcn.py`), proving the code
  runs correctly on constructed, controlled structure.
- **Real**: `torch_geometric.datasets.KarateClub` -- Zachary's Karate Club,
  a real 34-node/156-directed-edge social network with a known 4-faction
  ground-truth community split (`tests/test_real_dataset_validation.py`).
  `KarateClub` is generated in-process from a hardcoded edge list baked
  directly into `torch_geometric`'s own source (no `download()`, no
  `raw_dir`, no URL fetch) -- it is genuinely bundled inside the
  already-required `torch_geometric` dependency, so this validation
  requires **no network access and no new dependency**. The test wraps its
  real `edge_index` in `PyGSparseAdapter.from_edge_index(...)` (the same
  public API the synthetic tests use), round-trips it through COO/CSR/CSC,
  and asserts the adapter's shape/edge counts match the dataset's own
  known, fixed values -- then runs the existing `GCN` forward pass on the
  real node features and asserts finite, correctly shaped output. This
  confirms the adapter and GCN forward pass behave correctly on a real
  graph's real structure; it is not a training run and makes no accuracy
  claim about an untrained network.
  `examples/gcn_example.py` runs the same real-dataset section alongside
  the synthetic one and prints its real structural stats.

## Licensing constraints (§2.2, §2.4 of Module 4's description)

Per CLAUDE.md's licensing policy and the architecture doc's two mandatory
build-time exclusions for this module:

- **No METIS, no `torch-sparse`.** METIS carries a restrictive University
  of Minnesota academic license; its optional carrier in the PyG ecosystem
  is `torch-sparse`. Neither is a dependency of this package, and neither
  is imported or referenced anywhere in its code. (The architecture doc's
  prescribed replacement, `torch-cluster`'s Graclus algorithm, is used for
  graph coarsening/sampling -- out of scope for this narrow pass, so the
  simplest compliant path here is to depend on none of them.)
- **No SuiteSparse/CHOLMOD** (GPLv2+). The CPU sparse-format bridge in
  `sparse.py` uses only `scipy.sparse` and `numpy` -- never CHOLMOD, never
  any GPL-licensed sparse-linear-algebra library.

## Deferred (explicitly out of scope for this pass)

These are real future-work items from the architecture doc's Module 4
section, not partially-stubbed-out code in this package:

- **GAT, GraphSAGE.** Only GCN (via `torch_geometric.nn.GCNConv`) is
  implemented.
- **Global Graph Transformers / Laplacian Positional Encodings.**
- **SQL-to-graph mapping engine** (schema reflection, FK-to-edge
  conversion, semantic-type-aware featurization, temporal-leakage
  filtering).
- **Neighborhood-sampling paradigms** (node-wise/GraphSAGE-style,
  layer-wise/FastGCN-style, LADIES).
- **Oversmoothing/oversquashing monitoring** (Dirichlet-Energy/
  Rayleigh-Coefficient tracking, Jacobian-sensitivity/discrete-curvature
  analysis).
- **DGL support.** Only the PyG (COO-native) side of the Universal Sparse
  Tensor layer is implemented; a DGL CSR/CSC-native adapter is future work.

## Dependency surface

Per the architecture doc's §2.7 framing (PyTorch-heavy modules have their
own, expected-to-conflict dependency universe) and Part 3's description of
LazyGraph as explicitly PyG/DGL-based: `torch` and `torch_geometric` are
**core dependencies**, not optional extras, because this package's one
signature capability cannot exist without them.

- **Core (always installed):** `numpy`, `scipy`, `torch`, `torch_geometric`.
- **Optional `dev` extra:** `pytest`.

This package subclasses `lazycore.data.SparseGraphTensorAdapter` directly
(does not redefine or duplicate it). `lazycore` is a local sibling package
(`packages/lazycore`) and is **installed separately, not as a formal
pyproject dependency of this package** -- hatchling/pip don't have a
portable, idiomatic way to express a relative-path dependency in
`pyproject.toml` metadata. Install it first (see below), matching the
convention established in `packages/automl` and `packages/lazyclean`.

## Installation (local dev)

```bash
# from the repo root
pip install -e packages/lazycore
pip install -e "packages/lazygraph[dev]"
```

Note: `torch`/`torch_geometric` installs are large and can take several
minutes -- this is expected for this module.

## Running tests

```bash
pytest packages/lazygraph/tests
```

## Running the example

```bash
python packages/lazygraph/examples/gcn_example.py
```

Runs two sections: (1) a small synthetic Erdos-Renyi random graph, wrapped
in `PyGSparseAdapter`, converted through CSR and CSC (`scipy.sparse`
bridge), with the `GCN` forward pass run over random node features; (2)
the real, bundled `torch_geometric.datasets.KarateClub` graph run through
the identical adapter/GCN pipeline. Both sections print their structural
stats (node count, edge count, output shape).

## Platform note

This package never hardcodes `device="cuda"`. `resolve_device()` prefers
MPS (the primary backend per CLAUDE.md), falls back to CUDA, then CPU, and
degrades cleanly if a preferred device string turns out to be unavailable.
