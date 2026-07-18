"""benchcraft_lazyclean -- LazyClean scaffold: ONNX Runtime embeddings + near-duplicate detection.

Public API surface for the one signature capability implemented at this
scaffold depth (architecture doc Part 3, "Module 2: LazyClean"): embed a
batch of text rows via native ONNX Runtime (no PyTorch, no `transformers`),
then flag near-duplicate row pairs via cosine-similarity thresholding over
those embeddings -- a minimal stand-in for the Density-Based Semantic
Deduplication (D4) idea. See ``dedup.py`` for the explicit naive-O(n^2)
scope boundary and ``embeddings.py`` for the PyTorch-free embedding path.

Everything else described for LazyClean in the architecture doc (the
IVF-HNSW ANN index, spherical mini-batch k-means, the DeCoLe tabular
label-error detector, train/test contamination auditing, the aggregate
"Dataset Integrity Score") is out of scope for this pass -- see README.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from lazycore.data import is_arrow_backed_pandas

from .dedup import DedupReport, DuplicatePair, cosine_similarity_matrix, find_near_duplicates
from .embeddings import (
    MODEL_ALLOWLIST,
    RECOMMENDED_MODEL_NAME,
    EmbeddingModel,
    build_synthetic_embedding_model,
    build_synthetic_embedding_onnx,
    download_recommended_model,
    hashing_bag_of_words_vectorizer,
)

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import pandas as pd
    import polars as pl

__all__ = [
    "EmbeddingModel",
    "MODEL_ALLOWLIST",
    "RECOMMENDED_MODEL_NAME",
    "build_synthetic_embedding_model",
    "build_synthetic_embedding_onnx",
    "download_recommended_model",
    "hashing_bag_of_words_vectorizer",
    "DedupReport",
    "DuplicatePair",
    "cosine_similarity_matrix",
    "find_near_duplicates",
    "detect_near_duplicate_text",
]

#: Text rows may be supplied as a plain iterable of strings, or as a
#: Tier-1 Arrow-backed column (a pandas Series or a Polars Series), per
#: lazycore.data's §2.1 conventions.
TextRows = "Iterable[str] | pd.Series | pl.Series"


def _coerce_text_rows(rows: object) -> list[str]:
    """Normalize the accepted input shapes down to a plain ``list[str]``.

    Accepts a plain iterable of strings, or a Tier-1 Arrow-backed pandas
    Series / Polars Series (see ``lazycore.data``, architecture doc §2.1).
    A pandas Series that is *not* Arrow-backed is still accepted (this
    package does not hard-require Tier-1 storage to function), but emits a
    warning pointing at the convention, using
    ``lazycore.data.is_arrow_backed_pandas`` for the check -- reusing
    lazycore's Tier-1 helper rather than re-implementing an Arrow-dtype
    check here.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(rows, pd.DataFrame):
        raise TypeError(
            "Expected a single text column (a pandas Series), not a full "
            "DataFrame. Pass e.g. `frame['text_column']`."
        )

    if pd is not None and isinstance(rows, pd.Series):
        if not is_arrow_backed_pandas(rows.to_frame()):
            warnings.warn(
                "Input pandas Series is not Arrow-backed (pandas 2.x "
                "ArrowDtype). benchcraft_lazyclean follows lazycore's Tier-1 "
                "convention (architecture doc §2.1) for zero-copy interop "
                "across Benchcraft modules; consider "
                "`series.convert_dtypes(dtype_backend='pyarrow')`. "
                "Proceeding anyway -- this is not a hard requirement.",
                stacklevel=3,
            )
        return [str(value) for value in rows.tolist()]

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None and isinstance(rows, pl.Series):
        return [str(value) for value in rows.to_list()]

    return [str(value) for value in rows]  # type: ignore[union-attr]


def detect_near_duplicate_text(
    rows: object,
    model: EmbeddingModel,
    *,
    threshold: float = 0.92,
) -> tuple[np.ndarray, DedupReport]:
    """Embed ``rows`` via ONNX Runtime and flag near-duplicate row pairs.

    This is the one canonical entrypoint tying the embedding path
    (``embeddings.py``) to the dedup path (``dedup.py``) together -- prefer
    it over calling ``model.embed`` + ``find_near_duplicates`` separately
    unless you need the embeddings for something else in between.

    Args:
        rows: an iterable of text strings, or a Tier-1 Arrow-backed pandas
            Series / Polars Series (a single text column).
        model: an :class:`EmbeddingModel` (e.g. from
            :func:`build_synthetic_embedding_model` for hermetic use, or a
            real production model wired via
            :meth:`EmbeddingModel.from_onnx_file`).
        threshold: cosine-similarity cutoff in ``(0.0, 1.0]`` -- see
            :func:`find_near_duplicates`.

    Returns:
        ``(embeddings, report)`` -- the ``(n_rows, embedding_dim)`` float32
        embedding array and the :class:`DedupReport` of flagged pairs.
    """
    texts = _coerce_text_rows(rows)
    embeddings = model.embed(texts)
    report = find_near_duplicates(embeddings, threshold=threshold)
    return embeddings, report
