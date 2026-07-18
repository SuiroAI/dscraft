"""Cosine-similarity near-duplicate detection over embeddings.

This is a minimal stand-in for the Density-Based Semantic Deduplication
(D4) capability described in the architecture doc's Part 3 ("Module 2:
LazyClean"). The real D4 design uses spherical mini-batch k-means to bucket
embeddings into clusters, then an IVF-HNSW approximate-nearest-neighbor
index within/across clusters, specifically to *avoid* O(n^2) pairwise
cosine-similarity cost at scale.

**This module deliberately implements only the naive O(n^2) brute-force
version.** For a small batch this is a perfectly correct and simple
stand-in for the ANN index; it does not scale past a few thousand rows
before the pairwise similarity matrix becomes the bottleneck the
architecture doc calls out IVF-HNSW as solving. Replacing
:func:`cosine_similarity_matrix` (and the O(n^2) loop in
:func:`find_near_duplicates`) with a real IVF-HNSW index is tracked as
follow-up work, not implemented in this scaffold-depth pass -- see the
package README for the explicit scope boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DuplicatePair",
    "DedupReport",
    "cosine_similarity_matrix",
    "find_near_duplicates",
]


@dataclass(frozen=True)
class DuplicatePair:
    """One flagged near-duplicate row pair."""

    index_a: int
    index_b: int
    similarity: float


@dataclass
class DedupReport:
    """Result of a near-duplicate scan over a batch of embedded rows.

    ``zero_vector_row_indices`` is a distinct third category alongside
    "flagged as a duplicate pair" and "not flagged" -- see
    :func:`find_near_duplicates` for why an all-zero embedding row cannot be
    scored as either a confirmed duplicate or a confirmed distinct row.
    """

    pairs: list[DuplicatePair]
    threshold: float
    num_rows: int
    zero_vector_row_indices: list[int] = field(default_factory=list)

    def flagged_indices(self) -> set[int]:
        """The set of row indices that appear in at least one flagged pair.

        Rows in :attr:`zero_vector_row_indices` never appear here: a
        zero-vector row's similarity to *anything* (including another
        zero-vector row) is undefined, so it can never be scored above
        ``threshold`` and flagged as a duplicate. Check
        ``zero_vector_row_indices`` separately to see which rows produced no
        extractable features and could not be compared at all.
        """
        flagged: set[int] = set()
        for pair in self.pairs:
            flagged.add(pair.index_a)
            flagged.add(pair.index_b)
        return flagged


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the full pairwise cosine-similarity matrix for ``embeddings``.

    **Naive O(n^2) implementation.** This materializes an ``(n, n)`` dense
    similarity matrix, which is the exact cost profile the architecture
    doc's IVF-HNSW approximate-nearest-neighbor index (Part 3, "Module 2:
    LazyClean") exists to avoid at scale. Acceptable and simple for the
    small batches this scaffold-depth pass targets; not a substitute for
    that index once real dataset sizes are in play.

    **Zero-vector rows are undefined, not 0.0 or 1.0.** A genuinely all-zero
    embedding row (e.g. ``hashing_bag_of_words_vectorizer`` on text with no
    regex-matching tokens at all -- empty/whitespace-only text, but also
    punctuation-only text like ``"!!!"``/``"???"`` or non-ASCII text the
    tokenizer's ``[a-z0-9]+`` regex can't match) has no defined direction. A
    zero embedding means the vectorizer extracted *no features*, not that
    the source rows are equal or that they are distinct -- two different
    zero-feature texts (``"!!!"`` and ``"???"``) are not duplicates of each
    other just because they hash to the same all-zero vector, and treating
    that as similarity 1.0 falsely flags unrelated rows as duplicates at
    every valid threshold. Silently reading it as 0.0 (the plain
    normalized-dot-product result) is equally wrong in the other direction:
    it silently misses genuinely-identical zero-feature rows (e.g. two
    empty strings) as duplicates. Both are silent guesses this vectorizer
    has no basis for. This function reports the honest answer instead:
    every entry involving at least one zero-vector row (including a
    zero-vector row against itself) is ``np.nan`` -- "not comparable" --
    rather than a guessed 0.0 or 1.0. Callers that need pairwise duplicate
    decisions should treat ``nan`` as "cannot say" and handle zero-vector
    rows as a separate category (see :func:`find_near_duplicates` and
    :attr:`DedupReport.zero_vector_row_indices`).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D (n_rows, dim) array, got shape {embeddings.shape!r}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    is_zero_vector = (norms == 0.0).reshape(-1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = embeddings / safe_norms
    similarities = normalized @ normalized.T

    if is_zero_vector.any():
        # Any pair where *either* row is a zero vector is undefined -- not
        # just the zero-vector-against-zero-vector case. See the docstring
        # above for why silently choosing 0.0 or 1.0 here is wrong either
        # way.
        undefined_mask = np.logical_or.outer(is_zero_vector, is_zero_vector)
        similarities = np.where(undefined_mask, np.nan, similarities)
    return similarities


def find_near_duplicates(embeddings: np.ndarray, *, threshold: float = 0.92) -> DedupReport:
    """Flag row-index pairs whose cosine similarity is at or above ``threshold``.

    Brute-force O(n^2): computes the full pairwise similarity matrix via
    :func:`cosine_similarity_matrix` and scans its upper triangle. See the
    module docstring for why this is an intentional, documented scope
    boundary rather than the production-scale IVF-HNSW path.

    **Zero-vector rows are never flagged by similarity score.** A row whose
    embedding is all-zero (see :func:`cosine_similarity_matrix`) has an
    undefined (``nan``) similarity to every other row, including another
    zero-vector row, so it is skipped when scanning for duplicate pairs --
    it can never be silently swept in as a "duplicate" of an unrelated
    zero-vector row, nor silently cleared as "distinct". Instead, every such
    row's index is reported separately in
    :attr:`DedupReport.zero_vector_row_indices`, meaning "this row produced
    no extractable features and could not be compared" -- a third category
    distinct from both "confirmed duplicate" and "confirmed distinct".

    Args:
        embeddings: ``(n_rows, dim)`` array, typically from
            :meth:`benchcraft_lazyclean.embeddings.EmbeddingModel.embed`.
        threshold: cosine-similarity cutoff in ``(0.0, 1.0]`` above which a
            pair is flagged as a near-duplicate.

    Returns:
        A :class:`DedupReport` with pairs sorted by descending similarity,
        plus ``zero_vector_row_indices`` for rows that could not be
        compared at all.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0.0, 1.0], got {threshold!r}.")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D (n_rows, dim) array, got shape {embeddings.shape!r}.")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")

    num_rows = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1)
    zero_vector_row_indices = [i for i in range(num_rows) if norms[i] == 0.0]
    similarities = cosine_similarity_matrix(embeddings)

    pairs: list[DuplicatePair] = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            similarity = similarities[i, j]
            if np.isnan(similarity):
                # Undefined -- at least one of these two rows is a zero
                # vector. Not comparable, so never flagged as a duplicate
                # here; see zero_vector_row_indices instead.
                continue
            similarity = float(similarity)
            if similarity >= threshold:
                pairs.append(DuplicatePair(index_a=i, index_b=j, similarity=similarity))

    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return DedupReport(
        pairs=pairs,
        threshold=threshold,
        num_rows=num_rows,
        zero_vector_row_indices=zero_vector_row_indices,
    )
