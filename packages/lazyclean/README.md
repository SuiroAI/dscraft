# benchcraft-clean

A scaffold-depth implementation of one signature capability from
Benchcraft's LazyClean module (architecture doc Part 3, "Module 2:
LazyClean"): **embedding generation via native ONNX Runtime feeding a
near-duplicate detection check** -- a minimal version of the
Density-Based Semantic Deduplication (D4) idea.

## What this package does (and doesn't) implement

This is a **scaffold-depth pass, not a full implementation** of LazyClean.
In scope:

1. Embed a batch of text rows via an ONNX Runtime session
   (`benchcraft_lazyclean.embeddings`).
2. Flag near-duplicate row-index pairs via cosine-similarity thresholding
   over those embeddings (`benchcraft_lazyclean.dedup`).

Explicitly **out of scope** for this pass (tracked as future work per the
architecture doc, not silently dropped):

- The **IVF-HNSW approximate-nearest-neighbor index** and **spherical
  mini-batch k-means** clustering step that the real D4 design uses to
  avoid O(n²) pairwise cosine-similarity cost at scale.
- The **DeCoLe tabular label-error detector** (per-subpopulation confident
  learning).
- The **train/test contamination auditor**.
- The aggregate **"Dataset Integrity Score"**.

## Zero-vector rows: "not comparable", not a silent duplicate or distinct call

`hashing_bag_of_words_vectorizer` tokenizes with a simple `[a-z0-9]+` regex.
Any text with **zero regex-matching tokens** -- not just genuinely empty or
whitespace-only strings, but also punctuation-only text (`"!!!"`, `"???"`)
and non-ASCII text the regex can't match (`"日本語"`) -- embeds to the
identical all-zero vector. A zero embedding means *the vectorizer extracted
no features*, not that the source rows are equal, and not that they're
distinct either -- a hashing bag-of-words vectorizer with zero extracted
features genuinely has no basis to compare two such rows.

`cosine_similarity_matrix` and `find_near_duplicates` treat this honestly:
every pairwise entry involving at least one zero-vector row (including a
zero-vector row against itself, and against a genuinely non-zero row) is
`nan` -- undefined, not silently `0.0` and not silently `1.0` -- and
`find_near_duplicates` never flags such a pair as a duplicate by score.
Instead, `DedupReport.zero_vector_row_indices` lists every row that
produced no extractable features and could not be compared at all, as a
third category distinct from both "confirmed duplicate" and "confirmed
distinct" rows in `report.pairs`/`report.flagged_indices()`.

This is itself the fix for two earlier bugs that got this wrong in
opposite directions: originally, two genuinely-empty rows read similarity
`0.0` against each other and were silently missed as duplicates; a
follow-up fix over-corrected by making *any* two zero-vector rows read
`1.0`, which falsely flagged unrelated zero-feature rows (e.g. `"!!!"` and
`"???"`) as duplicates of each other at every valid threshold. Neither
silent guess is right -- reporting "not comparable" separately is the
honest answer given what this vectorizer can actually tell you. Genuinely
identical non-empty text is unaffected by any of this and is still
correctly flagged as a duplicate via its (non-zero) embedding vector.

## The naive O(n²) caveat

`dedup.py`'s `find_near_duplicates` computes the **full pairwise
cosine-similarity matrix** over the embedded batch (`cosine_similarity_matrix`)
and scans it directly. This is O(n²) in both time and memory. It is a
correct, simple stand-in for the ANN index at this scaffold's small-batch
depth, but it is **not** the production path -- the architecture doc calls
out IVF-HNSW specifically to avoid this cost once dataset sizes grow past a
small batch. Replacing this brute-force check with a real IVF-HNSW index is
explicit follow-up work, not implemented here.

## The PyTorch-free / <100MB constraint, and why

Per the architecture doc (Part 3, Appendix A) and `CLAUDE.md`'s packaging
section (§2.7): LazyClean is deliberately **PyTorch-free**. This package's
runtime dependencies are exactly `onnxruntime`, `onnx`, `numpy`, and
`lazycore` -- **no `torch`, no `transformers`, anywhere, including optional
extras.** This is enforced as a hard rule (see `tests/test_embeddings.py::test_no_pytorch_or_transformers_imported`),
not a soft preference, because:

- PyTorch + HuggingFace `transformers` together pull in a multi-hundred-MB
  to multi-GB dependency tree (CUDA/MPS backends, tokenizer binaries,
  model-hub client code). LazyClean's design target is to stay under
  **~100MB** total footprint so a data-cleaning pass doesn't require
  installing the platform's heaviest dependency stack just to check for
  duplicate rows.
- It is also this module's specific differentiator against the PyTorch/HF
  stack every other embedding-generation tool in this space defaults to.

Embeddings are produced by loading a `.onnx` model file directly via the
`onnxruntime` Python package and running lightweight, tokenizer-adjacent
preprocessing ourselves in plain Python/NumPy (see
`hashing_bag_of_words_vectorizer` in `embeddings.py`) -- never via
`AutoTokenizer`/`AutoModel` or any `transformers` call.

## Solving "we need a real `.onnx` model" without network access or a bundled checkpoint

Real sentence-embedding ONNX models are tens to hundreds of MB, which is
wrong to check into this repo and wrong to require for `pytest` to pass in
an offline CI environment. This package handles that with two paths:

1. **Hermetic (used by tests and the example):**
   `embeddings.build_synthetic_embedding_onnx` hand-builds a tiny ONNX
   graph on the fly, directly via the `onnx` package's graph-builder API
   (`onnx.helper`/`onnx.numpy_helper`) -- a linear projection plus L2
   normalization over a hashed bag-of-words feature vector. No network
   access, no multi-hundred-MB file, deterministic given a seed.
   `embeddings.build_synthetic_embedding_model()` wraps this plus the
   default preprocessor into a ready-to-use `EmbeddingModel` in one call.
   This is **not** a semantically meaningful sentence embedding -- it exists
   solely to exercise the embed → cosine-similarity dedup pipeline
   end-to-end in tests and the example without any external dependency.
2. **Production (documented, not exercised by tests):** see "Wiring in a
   real production model" below.

### Wiring in a real production model

`benchcraft_lazyclean.embeddings.MODEL_ALLOWLIST` (a per-module
`lazycore.licensing.Allowlist` instance, per architecture doc §2.10)
registers one recommended Tier-1 checkpoint:

```python
from benchcraft_lazyclean.embeddings import MODEL_ALLOWLIST
entry = MODEL_ALLOWLIST.check("Xenova/all-MiniLM-L6-v2")
# ModelLicenseEntry(name='Xenova/all-MiniLM-L6-v2', tier=ModelTier.TIER_1,
#                    license_identifier='Apache-2.0', ...)
```

**Xenova/all-MiniLM-L6-v2** is an ONNX-exported build of
`sentence-transformers/all-MiniLM-L6-v2` (384-dim mean-pooled embeddings,
~90MB fp32 / ~23MB int8-quantized -- comfortably under the <100MB target),
Apache-2.0 licensed, auto-usable under the Tier-1 policy with no opt-in
gate. It is **not bundled** with this package. `embeddings.download_recommended_model()`
is the optional, lazy download path (never called by tests or the example,
matching the platform's local-only-by-default posture) that fetches and
caches its `.onnx` graph.

To actually use it for real inference, pair it with a real tokenizer
instead of the default `hashing_bag_of_words_vectorizer`:

```python
from pathlib import Path
from tokenizers import Tokenizer  # standalone Rust-backed tokenizer, no torch/transformers
from benchcraft_lazyclean.embeddings import EmbeddingModel, download_recommended_model

onnx_path = download_recommended_model()  # requires network access; opt-in
tokenizer = Tokenizer.from_pretrained("Xenova/all-MiniLM-L6-v2")

def real_preprocessor(text: str):
    encoding = tokenizer.encode(text)
    # Feed input_ids/attention_mask into the model's actual ONNX inputs and
    # mean-pool the token embeddings over attention_mask here -- the exact
    # shape depends on the checkpoint's input signature (use
    # onnxruntime.InferenceSession(...).get_inputs() to inspect it).
    ...

model = EmbeddingModel.from_onnx_file(
    onnx_path, preprocessor=real_preprocessor, embedding_dim=384,
)
```

`tokenizers` (the standalone HuggingFace tokenizer library, Rust-backed,
**no PyTorch/`transformers` dependency**) is a fine choice here per the
task's constraints, but it is deliberately **not** a hard dependency of
this package -- it is only needed if you wire in a real subword-tokenized
model. The default hashing preprocessor has zero extra dependencies.

## Public API

```python
from benchcraft_lazyclean import (
    EmbeddingModel,               # wraps an onnxruntime.InferenceSession + preprocessor
    build_synthetic_embedding_model,  # hermetic test/example fixture
    detect_near_duplicate_text,   # rows -> (embeddings, DedupReport)
    find_near_duplicates,         # embeddings -> DedupReport (naive O(n^2))
    DedupReport, DuplicatePair,
    MODEL_ALLOWLIST,              # lazycore.licensing.Allowlist for this module
)

model = build_synthetic_embedding_model()
embeddings, report = detect_near_duplicate_text(
    ["some text", "some txt", "totally different"], model, threshold=0.9,
)
for pair in report.pairs:
    print(pair.index_a, pair.index_b, pair.similarity)

# Rows that produced no extractable features (e.g. empty/punctuation-only/
# non-ASCII text under the hashing vectorizer) are reported separately --
# see "Zero-vector rows" above -- rather than silently folded into `pairs`.
print("could not compare:", report.zero_vector_row_indices)
```

`detect_near_duplicate_text` accepts a plain `Iterable[str]`, or a Tier-1
Arrow-backed `pandas.Series` / `polars.Series` (a single text column), per
`lazycore.data`'s §2.1 conventions -- it uses
`lazycore.data.is_arrow_backed_pandas` to check (and warn, not fail, if
not) rather than re-implementing that check.

## Installation

`lazycore` is a local sibling package under `packages/lazycore` and is not
declared as a formal `pyproject.toml` dependency (hatchling/pip have no
portable relative-path dependency syntax) -- install it first, matching the
convention already established by `packages/automl`:

```bash
pip install -e packages/lazycore
pip install -e "packages/lazyclean[dev]"
```

## Running tests

```bash
pytest packages/lazyclean/tests
```

Fully hermetic -- no network access required. Uses
`build_synthetic_embedding_model()` throughout.

## Running the example

```bash
python packages/lazyclean/examples/dedup_example.py
```
