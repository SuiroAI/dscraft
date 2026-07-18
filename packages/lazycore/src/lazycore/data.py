"""Shared three-tier data/tensor conventions (architecture doc §2.1).

LazyCore does not ship a data-processing library. It documents and lightly
implements the three-tier convention that every Benchcraft module agrees to
follow so that modules sharing "the same underlying data" don't each invent
their own representation:

Tier 1 - Dense tabular / text / time-series
    Canonical format: Apache Arrow, fronted by pandas 2.x ``ArrowDtype``
    columns and/or Polars DataFrames. pandas 2.x and Polars are both
    zero-copy front-ends over the same Arrow buffers, so this tier is a set
    of small conversion/validation helpers, not a wrapping library. Neither
    pandas nor polars is a hard dependency of ``lazycore`` -- consumers
    import them themselves, and these helpers import them lazily so that a
    PyTorch-free or otherwise minimal module never pays for that import
    unless it actually calls into this tier.

Tier 2 - Sparse graph tensors
    DLPack's spec is explicitly limited to dense/strided arrays and cannot
    represent sparsity, so there is no zero-copy shortcut here -- a real
    adapter is required. LazyCore only defines the *shape* of that adapter
    (an abstract interface for COO / CSR-CSC exchange) via
    :class:`SparseGraphTensorAdapter`. Concrete implementations bridging
    PyTorch Geometric's native COO and DGL's native CSR/CSC formats belong
    in LazyGraph, not here -- lazycore never depends on torch, PyG, or DGL.

Tier 3 - Dense image / audio
    FFCV-style workloads are bottlenecked by decode+augmentation *compute*,
    not by data-copy/serialization overhead, so the shared convention is a
    pipeline shape (decode -> augment -> DLPack handoff only at the final
    dense-tensor stage), defined here as :class:`DenseMediaPipeline`.
    Concrete decode/augment implementations belong in LazyVision.

Nothing in this module imports pandas, polars, torch, or any other heavy
runtime dependency at module import time. Where such libraries are needed
for type-checking only, imports are guarded behind ``TYPE_CHECKING``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import pandas as pd
    import polars as pl

__all__ = [
    "ArrowBackedFrame",
    "is_arrow_backed_pandas",
    "pandas_arrow_dtypes",
    "to_polars_zero_copy",
    "from_polars_zero_copy",
    "SparseGraphTensorAdapter",
    "SparseFormat",
    "DenseMediaPipeline",
]


# ---------------------------------------------------------------------------
# Tier 1: Arrow-backed dense tabular / text / time-series helpers
# ---------------------------------------------------------------------------

#: Duck-typed alias for "something Arrow-shaped": a pandas DataFrame backed
#: by ArrowDtype columns, or a Polars DataFrame. Not a runtime-enforced type
#: (that would require importing pandas/polars eagerly); it exists purely
#: for readability in module type hints via ``TYPE_CHECKING``.
if TYPE_CHECKING:  # pragma: no cover
    ArrowBackedFrame = "pd.DataFrame | pl.DataFrame"
else:
    ArrowBackedFrame = Any


def _require_pandas() -> "pd":  # pragma: no cover - trivial import guard
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This lazycore.data helper requires pandas to be installed in "
            "the calling module's environment. lazycore itself does not "
            "depend on pandas."
        ) from exc
    return pd


def _require_polars() -> "pl":  # pragma: no cover - trivial import guard
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This lazycore.data helper requires polars to be installed in "
            "the calling module's environment. lazycore itself does not "
            "depend on polars."
        ) from exc
    return pl


def pandas_arrow_dtypes(frame: "pd.DataFrame") -> dict[str, str]:
    """Return the subset of ``frame``'s columns backed by ``ArrowDtype``.

    Maps column name -> the string form of its pyarrow type. Useful for a
    module to quickly check which columns are already on the Tier-1
    zero-copy path versus which are still plain numpy-backed pandas columns
    that would force a conversion.
    """
    pd = _require_pandas()
    result: dict[str, str] = {}
    for name, dtype in frame.dtypes.items():
        if isinstance(dtype, pd.ArrowDtype):
            result[str(name)] = str(dtype.pyarrow_dtype)
    return result


def is_arrow_backed_pandas(frame: "pd.DataFrame") -> bool:
    """True if every column in ``frame`` uses a pandas 2.x ``ArrowDtype``.

    An empty DataFrame (no columns) is considered trivially Arrow-backed.
    """
    pd = _require_pandas()
    dtypes = list(frame.dtypes)
    if not dtypes:
        return True
    return all(isinstance(dtype, pd.ArrowDtype) for dtype in dtypes)


def to_polars_zero_copy(frame: "pd.DataFrame") -> "pl.DataFrame":
    """Convert an Arrow-backed pandas DataFrame to Polars.

    Per §2.1, pandas 2.x (``ArrowDtype``) and Polars are interchangeable
    front-ends over the same Arrow buffers, so this conversion is
    near-zero-cost when ``frame`` is already Arrow-backed. If it is not,
    this still works correctly but pandas/polars will do the necessary
    materialization themselves -- this helper does not silently hide that
    cost, it just delegates to ``pl.from_pandas`` either way.
    """
    pl = _require_polars()
    return pl.from_pandas(frame)


def from_polars_zero_copy(frame: "pl.DataFrame") -> "pd.DataFrame":
    """Convert a Polars DataFrame to an Arrow-backed pandas DataFrame.

    Uses pandas 2.x's ``dtype_backend="pyarrow"`` so the resulting frame
    stays on the Tier-1 Arrow representation rather than falling back to
    numpy-backed columns.
    """
    _require_pandas()  # ensure a clear error if pandas is missing at all
    return frame.to_pandas(use_pyarrow_extension_array=True)


# ---------------------------------------------------------------------------
# Tier 2: Sparse graph tensor adapter (interface only)
# ---------------------------------------------------------------------------


class SparseFormat:
    """Sparse tensor storage format names used across the Tier-2 adapter.

    These correspond to the "COO / CSR-CSC / SciPy sparse bridge" family
    named in §2.1 (modeled on NVIDIA's "Universal Sparse Tensor" concept).
    """

    COO = "coo"
    CSR = "csr"
    CSC = "csc"


class SparseGraphTensorAdapter(abc.ABC):
    """Abstract interface for a Tier-2 sparse graph tensor adapter.

    DLPack cannot represent sparsity, so there is no zero-copy shortcut for
    graph tensors -- a real conversion step between sparse formats (COO,
    CSR/CSC) is unavoidable. LazyCore defines only the *shape* of that
    conversion boundary here; it does not implement it and does not depend
    on any graph library (PyTorch Geometric, DGL, SciPy, etc.). LazyGraph
    is expected to provide concrete subclasses that actually bridge PyG's
    native COO representation and DGL's native CSR/CSC representation.

    Implementations should treat every method below as doing real work
    (not just re-labelling metadata) since a genuine conversion step is
    happening.
    """

    @property
    @abc.abstractmethod
    def native_format(self) -> str:
        """The sparse format (one of :class:`SparseFormat`) this adapter
        instance currently holds its data in."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        """The dense (rows, cols) shape represented by this sparse tensor."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_coo(self) -> "SparseGraphTensorAdapter":
        """Return an adapter view/copy in COO format (PyG-native)."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_csr(self) -> "SparseGraphTensorAdapter":
        """Return an adapter view/copy in CSR format (DGL-native for many
        aggregation ops)."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_csc(self) -> "SparseGraphTensorAdapter":
        """Return an adapter view/copy in CSC format."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tier 3: Dense image / audio pipeline (interface only)
# ---------------------------------------------------------------------------


@runtime_checkable
class _SupportsDLPack(Protocol):
    """Structural type for "something that can hand off via DLPack" --
    i.e. exposes ``__dlpack__``/``__dlpack_device__`` per the DLPack
    protocol, without lazycore needing to depend on torch/numpy to say so.
    """

    def __dlpack__(self, stream: Any | None = ...) -> Any:
        """Export this tensor via the DLPack protocol for zero-copy handoff."""
        ...  # pragma: no cover

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the ``(device_type, device_id)`` pair per the DLPack spec."""
        ...  # pragma: no cover


class DenseMediaPipeline(abc.ABC):
    """Abstract interface for a Tier-3 FFCV-style decode+augment pipeline.

    Image/audio workloads are bottlenecked by decode+augmentation
    *compute*, not by data-copy/serialization overhead (this is why FFCV
    outperforms both the PyTorch DataLoader and NVIDIA DALI). The shared
    convention is therefore about pipeline *shape*, not about a shared
    zero-copy buffer: decode and augment happen natively (however the
    concrete implementation wants), and only the final, already-dense
    tensor is handed off via DLPack.

    LazyCore does not implement decode/augment logic and does not depend on
    any image/audio/tensor library. LazyVision is expected to provide
    concrete subclasses.
    """

    @abc.abstractmethod
    def decode(self, raw: bytes) -> Any:
        """Decode a raw media payload (e.g. JPEG/PNG/WAV bytes) into an
        intermediate in-memory representation. Not required to be a dense
        tensor yet -- this stage is decode-compute-bound, and
        implementations are free to use whatever intermediate
        representation is fastest for their decoder."""
        raise NotImplementedError

    @abc.abstractmethod
    def augment(self, decoded: Any) -> Any:
        """Apply augmentation to a decoded sample. Still not required to be
        a DLPack-compatible dense tensor -- augmentation is also
        compute-bound, not copy-bound."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_dense_tensor(self, augmented: Any) -> _SupportsDLPack:
        """Produce the final dense tensor for this sample. The return value
        must support the DLPack protocol (``__dlpack__`` /
        ``__dlpack_device__``) -- this is the one and only point in the
        Tier-3 pipeline where a zero-copy handoff (e.g. to PyTorch) is
        expected to occur."""
        raise NotImplementedError

    def run(self, raw: bytes) -> _SupportsDLPack:
        """Convenience driver: decode -> augment -> to_dense_tensor.

        Concrete pipelines may override this if they need to fuse stages
        for performance, but the default composition documents the
        expected shape of the pipeline.
        """
        decoded = self.decode(raw)
        augmented = self.augment(decoded)
        return self.to_dense_tensor(augmented)
