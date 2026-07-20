"""Concrete Tier-2 sparse graph tensor adapter (architecture doc §2.1).

``dscraft.core.data.SparseGraphTensorAdapter`` defines only the *shape* of the
COO / CSR-CSC conversion boundary and deliberately depends on nothing
graph-related. This module provides the first concrete implementation of
that interface for DSCraft: a bridge between PyTorch Geometric's native
COO ``edge_index`` representation and ``scipy.sparse``'s CSR/CSC formats.

This is the "SciPy sparse bridge" named in §2.1's Tier-2 row -- a genuine,
unavoidable conversion step (DLPack cannot represent sparsity), not a
metadata relabel. Every ``to_*`` conversion here actually builds a new
in-memory sparse structure in the target format.

Per CLAUDE.md's "fix what's there / one canonical implementation" rule:
this is the *only* sparse-tensor adapter in this package. Do not add a
second/parallel adapter class or reimplement
``dscraft.core.data.SparseGraphTensorAdapter`` here -- subclass it, as done
below.

Licensing note (§2.2, §2.4 of Module 4's description): this module uses
only ``scipy.sparse`` and ``numpy`` for the CSR/CSC bridge. It never
imports or depends on SuiteSparse/CHOLMOD (GPLv2+) or `torch-sparse`
(the optional METIS-linked carrier) -- see the package README's licensing
section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from dscraft.core.data import SparseFormat, SparseGraphTensorAdapter

if TYPE_CHECKING:  # pragma: no cover - type-checking-only import
    import torch

__all__ = ["PyGSparseAdapter"]


class PyGSparseAdapter(SparseGraphTensorAdapter):
    """Concrete Tier-2 adapter bridging PyG-native COO and SciPy CSR/CSC.

    Holds a PyTorch Geometric-style ``edge_index`` tensor (``torch.Tensor``
    of shape ``[2, num_edges]``, row 0 = source node indices, row 1 =
    target node indices) plus the dense ``(num_rows, num_cols)`` shape of
    the adjacency matrix it represents. Optionally carries per-edge
    weights; unweighted graphs are treated as all-ones edge weights.

    This adapter is intentionally format-tagged: an instance constructed
    via :meth:`from_edge_index` starts life in COO format (PyG's native
    representation), and calling :meth:`to_csr`/:meth:`to_csc` returns a
    *new* adapter instance holding a ``scipy.sparse`` matrix in that format
    instead, with :attr:`native_format` updated accordingly. Converting
    back to COO from a CSR/CSC-backed instance reconstructs a fresh
    ``edge_index`` tensor from the SciPy matrix's COO view.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        native_format: str,
        edge_index: "torch.Tensor | None" = None,
        edge_weight: "torch.Tensor | None" = None,
        scipy_matrix: sp.spmatrix | None = None,
    ) -> None:
        """Construct an adapter directly in a given format.

        Prefer :meth:`from_edge_index` for the common COO-construction
        case; this constructor is also used internally by :meth:`to_coo`,
        :meth:`to_csr`, and :meth:`to_csc` to build the converted instance.

        Args:
            shape: Dense ``(num_rows, num_cols)`` shape of the adjacency
                matrix represented by this adapter.
            native_format: One of ``SparseFormat.COO``/``CSR``/``CSC``,
                identifying which backing representation is populated.
            edge_index: PyG-style ``[2, num_edges]`` COO edge index.
                Required (and only meaningful) when ``native_format`` is
                ``SparseFormat.COO``.
            edge_weight: Optional per-edge weights aligned with
                ``edge_index``; unweighted COO graphs are treated as
                all-ones weights.
            scipy_matrix: A ``scipy.sparse`` matrix already in CSR or CSC
                format. Required (and only meaningful) when
                ``native_format`` is ``SparseFormat.CSR``/``CSC``.

        Raises:
            ValueError: If ``native_format`` is unrecognized, or if the
                format-specific required data (``edge_index`` for COO,
                ``scipy_matrix`` for CSR/CSC) is missing.
        """
        if native_format not in (SparseFormat.COO, SparseFormat.CSR, SparseFormat.CSC):
            raise ValueError(f"Unknown sparse format: {native_format!r}")
        if native_format == SparseFormat.COO and edge_index is None:
            raise ValueError("COO-format PyGSparseAdapter requires edge_index.")
        if native_format in (SparseFormat.CSR, SparseFormat.CSC) and scipy_matrix is None:
            raise ValueError(
                f"{native_format.upper()}-format PyGSparseAdapter requires scipy_matrix."
            )

        self._shape = shape
        self._native_format = native_format
        self._edge_index = edge_index
        self._edge_weight = edge_weight
        self._scipy_matrix = scipy_matrix

    # -- construction -------------------------------------------------

    @classmethod
    def from_edge_index(
        cls,
        edge_index: "torch.Tensor",
        num_nodes: int,
        edge_weight: "torch.Tensor | None" = None,
    ) -> "PyGSparseAdapter":
        """Build a COO-native adapter from a PyG-style ``edge_index``.

        ``edge_index`` must be a ``[2, num_edges]`` integer tensor whose
        values all lie in ``[0, num_nodes)``. The resulting adapter
        represents a square ``(num_nodes, num_nodes)`` adjacency matrix,
        which is the common case for MPNN inputs.

        This validates the full edge-index contract at construction time
        (not just its shape) so malformed graphs cannot survive until a
        ``.to_csr()``/``.to_csc()`` SciPy conversion or a GCN forward pass:

        Raises:
            ValueError: If ``num_nodes`` is not a positive integer; if
                ``edge_index`` does not have shape ``[2, num_edges]``; if
                ``edge_index`` has a non-integer dtype; if any node index in
                ``edge_index`` falls outside ``[0, num_nodes)``; or if
                ``edge_weight`` is provided and is not a 1-D tensor, or its
                length does not match the number of edges.
        """
        import torch

        if not isinstance(num_nodes, int) or isinstance(num_nodes, bool) or num_nodes <= 0:
            raise ValueError(f"num_nodes must be a positive integer, got {num_nodes!r}")

        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
            )

        _integer_dtypes = (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
        if edge_index.dtype not in _integer_dtypes:
            raise ValueError(
                f"edge_index must have an integer dtype, got {edge_index.dtype}"
            )

        num_edges = edge_index.shape[1]
        if num_edges > 0:
            min_index = int(edge_index.min().item())
            max_index = int(edge_index.max().item())
            if min_index < 0 or max_index >= num_nodes:
                offending = min_index if min_index < 0 else max_index
                raise ValueError(
                    f"edge_index values must be within [0, {num_nodes}) for "
                    f"num_nodes={num_nodes}, but found offending value "
                    f"{offending}"
                )

        if edge_weight is not None and edge_weight.dim() != 1:
            raise ValueError(
                f"edge_weight must be a 1-D tensor of per-edge weights, got "
                f"shape {tuple(edge_weight.shape)}"
            )

        if edge_weight is not None and edge_weight.shape[0] != num_edges:
            raise ValueError(
                f"edge_weight length ({edge_weight.shape[0]}) must match the "
                f"number of edges ({num_edges})"
            )

        return cls(
            shape=(num_nodes, num_nodes),
            native_format=SparseFormat.COO,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    # -- SparseGraphTensorAdapter interface ----------------------------

    @property
    def native_format(self) -> str:
        """Which backing representation this adapter currently holds.

        One of ``SparseFormat.COO``, ``SparseFormat.CSR``, or
        ``SparseFormat.CSC``. Changes across calls to :meth:`to_coo`,
        :meth:`to_csr`, or :meth:`to_csc`, each of which returns a new
        adapter instance rather than mutating this one in place.
        """
        return self._native_format

    @property
    def shape(self) -> tuple[int, int]:
        """Dense ``(num_rows, num_cols)`` shape of the adjacency matrix."""
        return self._shape

    @property
    def edge_index(self) -> "torch.Tensor":
        """The PyG-native ``[2, num_edges]`` COO edge index.

        Only meaningful when :attr:`native_format` is ``SparseFormat.COO``;
        raises if the adapter currently holds CSR/CSC data (call
        :meth:`to_coo` first).
        """
        if self._native_format != SparseFormat.COO or self._edge_index is None:
            raise RuntimeError(
                "edge_index is only available on a COO-format adapter; "
                "call .to_coo() first."
            )
        return self._edge_index

    @property
    def edge_weight(self) -> "torch.Tensor | None":
        """Optional per-edge weights aligned with :attr:`edge_index`.

        Only meaningful when :attr:`native_format` is ``SparseFormat.COO``;
        raises if the adapter currently holds CSR/CSC data (call
        :meth:`to_coo` first). Returns ``None`` for an unweighted graph --
        callers (e.g. `GCNConv`) should treat that as "no edge weights",
        not as an error.
        """
        if self._native_format != SparseFormat.COO:
            raise RuntimeError(
                "edge_weight is only available on a COO-format adapter; "
                "call .to_coo() first."
            )
        return self._edge_weight

    @property
    def scipy_matrix(self) -> sp.spmatrix:
        """The underlying ``scipy.sparse`` matrix for a CSR/CSC adapter."""
        if self._scipy_matrix is None:
            raise RuntimeError(
                "scipy_matrix is only available on a CSR/CSC-format "
                "adapter; call .to_csr() or .to_csc() first."
            )
        return self._scipy_matrix

    def to_coo(self) -> "PyGSparseAdapter":
        """Return a COO-format adapter (PyG-native)."""
        import torch

        if self._native_format == SparseFormat.COO:
            return self

        coo = self._scipy_matrix.tocoo()
        edge_index = torch.as_tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long
        )
        edge_weight = None
        if not np.allclose(coo.data, 1.0):
            edge_weight = torch.as_tensor(coo.data, dtype=torch.float32)
        return PyGSparseAdapter(
            shape=self._shape,
            native_format=SparseFormat.COO,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def to_csr(self) -> "PyGSparseAdapter":
        """Return a CSR-format adapter via a real ``scipy.sparse`` build."""
        if self._native_format == SparseFormat.CSR:
            return self
        return PyGSparseAdapter(
            shape=self._shape,
            native_format=SparseFormat.CSR,
            scipy_matrix=self._to_scipy_coo().tocsr(),
        )

    def to_csc(self) -> "PyGSparseAdapter":
        """Return a CSC-format adapter via a real ``scipy.sparse`` build."""
        if self._native_format == SparseFormat.CSC:
            return self
        return PyGSparseAdapter(
            shape=self._shape,
            native_format=SparseFormat.CSC,
            scipy_matrix=self._to_scipy_coo().tocsc(),
        )

    # -- helpers --------------------------------------------------------

    def _to_scipy_coo(self) -> sp.coo_matrix:
        """Build a ``scipy.sparse.coo_matrix`` from whatever this adapter
        currently holds. This is the real conversion step -- it always
        materializes a new SciPy structure, it never just relabels."""
        if self._native_format == SparseFormat.COO:
            edge_index = self._edge_index.detach().cpu().numpy()
            if self._edge_weight is not None:
                data = self._edge_weight.detach().cpu().numpy()
            else:
                data = np.ones(edge_index.shape[1], dtype=np.float32)
            return sp.coo_matrix(
                (data, (edge_index[0], edge_index[1])), shape=self._shape
            )
        # CSR/CSC-backed: SciPy handles the reformat internally.
        return self._scipy_matrix.tocoo()

    def to_dense_numpy(self) -> np.ndarray:
        """Convenience: materialize the full dense adjacency matrix.

        Used by tests to verify structural equivalence across formats; not
        part of the abstract interface.
        """
        return self._to_scipy_coo().toarray()
