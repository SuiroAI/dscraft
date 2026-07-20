"""Runnable demo: embed a handful of text rows and report near-duplicate pairs.

This script only imports and calls the real `dscraft.clean` package
API -- it does not reimplement any embedding or dedup logic inline (per
CLAUDE.md's "no net-new scripts" rule). Run it with:

    python packages/dscraft/examples/clean/dedup_example.py

It uses the fully hermetic synthetic ONNX embedding model
(`build_synthetic_embedding_model`), so no network access or bundled model
file is required. See the package README for how a real production
sentence-embedding model would be wired in via the same
`EmbeddingModel`/`detect_near_duplicate_text` API.
"""

from __future__ import annotations

from dscraft.clean import build_synthetic_embedding_model, detect_near_duplicate_text

ROWS = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog",  # near-duplicate of row 0 (punctuation only)
    "the QUICK brown fox JUMPS over the lazy dog!!!",  # near-duplicate of row 0 (case/punctuation)
    "Quantum entanglement enables non-local correlations between particles.",  # distinct
    "Sourdough bread requires a long, slow fermentation of the starter.",  # distinct
]


def main() -> None:
    """Build the synthetic embedding model, run the dedup pipeline over
    ``ROWS``, and print the flagged near-duplicate pairs plus distinct rows."""
    model = build_synthetic_embedding_model(vocab_dim=128, embedding_dim=32)

    print(f"Embedding {len(ROWS)} rows via ONNX Runtime (no PyTorch/transformers)...")
    embeddings, report = detect_near_duplicate_text(ROWS, model, threshold=0.9)
    print(f"Embeddings shape: {embeddings.shape}")
    print(
        f"Near-duplicate scan: {len(report.pairs)} pair(s) flagged "
        f"at cosine-similarity threshold {report.threshold} "
        f"(naive O(n^2) brute-force check -- see dedup.py)."
    )
    print()

    if not report.pairs:
        print("No near-duplicate pairs found.")
        return

    for pair in report.pairs:
        print(
            f"  rows [{pair.index_a}] <-> [{pair.index_b}]  "
            f"similarity={pair.similarity:.4f}"
        )
        print(f"    [{pair.index_a}] {ROWS[pair.index_a]!r}")
        print(f"    [{pair.index_b}] {ROWS[pair.index_b]!r}")

    flagged = report.flagged_indices()
    distinct_rows = [
        i
        for i in range(len(ROWS))
        if i not in flagged and i not in report.zero_vector_row_indices
    ]
    print()
    print(f"Rows not flagged as near-duplicates: {distinct_rows}")
    if report.zero_vector_row_indices:
        # None of ROWS above hit this in practice (they're all real
        # sentences), but this field exists precisely because a
        # hashing-vectorizer row that extracts zero features (e.g. empty,
        # whitespace-only, punctuation-only, or non-ASCII text the
        # tokenizer's [a-z0-9]+ regex can't match) cannot be honestly
        # scored as either a duplicate or distinct -- see the README's
        # "Zero-vector rows" section.
        print(
            "Rows that produced no extractable features (not comparable): "
            f"{report.zero_vector_row_indices}"
        )


if __name__ == "__main__":
    main()
