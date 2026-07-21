# Decision: shared PyTorch-free embeddings helper in `dscraft.core`?

- **Status:** Decided — wait, do not promote yet.
- **Date:** 2026-07-21
- **Issue:** [gr3enarr0w/dscraft#14](https://github.com/gr3enarr0w/dscraft/issues/14)
- **Related:** [gr3enarr0w/dscraft#12](https://github.com/gr3enarr0w/dscraft/issues/12) (`dscraft.agent` RAG pipeline — planning only, no code)

## Summary

`dscraft.clean.embeddings` is a self-contained, PyTorch-free ONNX Runtime
text-embedding module. Issue #14 asks whether it should be promoted to
`dscraft.core` now that a second consumer — `dscraft.agent`'s planned RAG
pipeline (#12) — has been identified. **Recommendation: do not promote now.**
#12 is a filed, unimplemented issue with no landed code and no settled
embeddings-backend choice (it explicitly proposes evaluating *at least two*
backends, e.g. sentence-transformers and fastembed, not committing to the
ONNX-Runtime-from-scratch shape `clean` built). CLAUDE.md's inter-module
rule is "deferred until two real modules need to exchange data" — a single
real module (`clean`) plus one *planned* module is not two real modules yet.
Revisit when #12 lands with actual embeddings code.

## (a) What `dscraft.clean/embeddings.py` currently does

Read in full at `packages/dscraft/src/dscraft/clean/embeddings.py` (444
lines). Structure:

1. **Model loading** — `EmbeddingModel.from_onnx_file()` wraps
   `onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])`
   directly. No PyTorch, no `transformers` — the module docstring makes this
   an explicit, hard constraint ("Do not import `torch` or `transformers`
   anywhere in this package, including for type hints").
2. **Tokenization/preprocessing** — `hashing_bag_of_words_vectorizer()` is a
   pure-Python/NumPy SHA-256 feature-hashing tokenizer used only by the
   synthetic test/example model. It is explicitly documented as a stand-in
   ("not intended to produce semantically meaningful embeddings for
   production use"); a real deployment would swap in a proper subword
   tokenizer (e.g. the standalone `tokenizers` library) feeding
   `input_ids`/`attention_mask`/`token_type_ids` into a transformer ONNX
   graph. `EmbeddingModel.embed()` supports both preprocessor output shapes
   (single array for a single-input graph, or a `{input_name: array}`
   mapping for a multi-input sentence-transformer graph) via the
   `PreprocessorOutput` union type — this dispatch logic is generic, not
   dedup-specific.
3. **Inference/output shape** — `EmbeddingModel.embed(texts)` returns a
   `(n, embedding_dim)` `float32` NumPy array. Empty input returns
   `(0, embedding_dim)`. No pooling/normalization decisions are hard-coded
   in `EmbeddingModel` itself — the synthetic model bakes L2-normalization
   into its own ONNX graph (`build_synthetic_embedding_onnx`), but that is a
   property of the *model graph*, not of the wrapper class. A real
   sentence-transformer ONNX export would similarly bake its own
   mean-pooling/normalization into the graph or require the caller's
   preprocessor/postprocessing to do it — `EmbeddingModel` doesn't care.
4. **Model provenance/licensing** — `MODEL_ALLOWLIST` is a
   `dscraft.core.licensing.Allowlist` instance populated with exactly one
   entry (`Xenova/all-MiniLM-L6-v2`, Apache-2.0/Tier-1), plus
   `download_recommended_model()`, a lazy, never-called-by-default download
   path pinned to an immutable commit SHA.
5. **Synthetic fixture builder** — `build_synthetic_embedding_onnx()` +
   `build_synthetic_embedding_model()` hand-build a tiny linear-projection +
   L2-normalize ONNX graph via the `onnx` package's graph-builder API. This
   is explicitly a test/example fixture, not a production embedding model.

### Genuinely generic vs. `clean`-specific

The runtime embedding *infrastructure* in `embeddings.py` — specifically
the `EmbeddingModel` class, the `PreprocessorOutput` union type, and the
`hashing_bag_of_words_vectorizer()` preprocessor — is generic: there is
**no dedup-specific post-processing baked into any of it**. Confirmed by
reading `dedup.py` (188 lines) and `clean/__init__.py`: `dedup.py` never
imports from `embeddings.py` at all (only references it in a docstring
type-hint comment); the only place they're composed is
`clean.detect_near_duplicate_text()` in `clean/__init__.py`, which is a
~10-line convenience wrapper that calls `model.embed(texts)` then
`find_near_duplicates(embeddings, threshold=...)` — two independently
testable, independently reusable pieces glued together at the call site,
not inside `embeddings.py` itself. So *if* promotion happened, the
promotable surface is narrower than "the whole module": it is these three
generic names only, not everything currently in the file.

Two things stay `clean`-owned and must **not** be described as
promotable, even though they live in the same file today:

1. **Policy/config, not infrastructure — `MODEL_ALLOWLIST`,
   `RECOMMENDED_MODEL_NAME`, and `download_recommended_model()`'s specific
   pinned checkpoint choice** (`Xenova/all-MiniLM-L6-v2`). This is `clean`'s
   own model recommendation, not a canonical "the one embedding model
   dscraft ships." Per `dscraft.core.licensing`'s documented pattern, each
   module owns its own `Allowlist` instance — a future `dscraft.agent`
   would very plausibly want a *different* recommended checkpoint (e.g. one
   tuned for retrieval rather than dedup similarity, or a multilingual
   model), so this piece should stay per-module even if the generic
   `EmbeddingModel`/ONNX-loading machinery moves.
2. **Test/example fixtures, not production infrastructure —
   `build_synthetic_embedding_onnx()` and `build_synthetic_embedding_model()`**.
   These hand-build a tiny, non-semantic ONNX graph purely so `clean`'s own
   test suite and `examples/dedup_example.py` can run hermetically with no
   network access and no bundled model file. They are not generic embedding
   *infrastructure* in the same sense as `EmbeddingModel` — they are
   `clean`-specific test/example scaffolding that happens to use the
   generic `EmbeddingModel`/ONNX machinery, the same way any other module's
   own hermetic test fixtures would. Per this package's per-module test
   layout convention, these belong in `clean`'s own tests/examples
   permanently, regardless of whether `EmbeddingModel` itself is ever
   promoted.

## (b) Does promoting now satisfy CLAUDE.md's "two real modules" rule?

CLAUDE.md is explicit: *"No formal inter-module data contracts yet... don't
build a shared typed interface between modules preemptively; that's
deferred until two real modules need to exchange data."* The architecture
doc's §2.9 (quoted in issue #14 itself) uses the same "two real modules"
language.

Issue #12 (`dscraft.agent`'s RAG pipeline) is, by its own text, **"Planning
only... No implementation yet"**, and its acceptance criteria say the
future implementation must select *"at least two embeddings backends"* —
its evidence section cites real local usage of `sentence-transformers`,
`fastembed`, *and* `qdrant-client`'s built-in embedding helpers, not a
single settled choice. Concretely, `dscraft.agent`'s extra in
`pyproject.toml` is `agent = []` today — stdlib + core only, zero
embeddings-related code, zero embeddings-related dependency.

This matters for the promotion question specifically because:

- We do not yet know if `agent`'s eventual embeddings backend will even be
  ONNX-Runtime-shaped. `fastembed` (cited in #12's evidence) happens to be
  ONNX-based internally, but it exposes its *own* higher-level Python API
  (`fastembed.TextEmbedding(...).embed(documents)`) — a caller composing
  `fastembed` would not touch `dscraft.clean.embeddings.EmbeddingModel` at
  all; they'd depend on the `fastembed` package directly. If `agent` instead
  picks `sentence-transformers` (also cited), that specific choice would
  **not by itself establish a shared interface** with `clean`'s embeddings:
  `sentence-transformers` is PyTorch-based, a different backend and a
  different loading mechanism than `clean`'s ONNX-Runtime-only
  `EmbeddingModel`, so `agent` couldn't reuse a promoted-from-`clean` helper
  *if it went that route*. That is a statement about one specific backend
  choice, not a categorical claim that `agent` and `clean` can never share
  embeddings infrastructure — `agent` could just as plausibly land on its
  own separate ONNX-backed implementation (e.g. via `fastembed`'s ONNX
  internals, or a bespoke `onnxruntime.InferenceSession` wrapper of its
  own) that *would* share `EmbeddingModel`'s shape, if that turns out to be
  the better fit for its RAG pipeline. Which of these `agent` actually picks
  is exactly the unresolved, code-not-yet-written question this evaluation
  cannot answer — "not automatically compatible" is the accurate framing,
  not "impossible."
- A "concrete filed issue that says it will need embeddings" is not the
  same claim as "a concrete, landed piece of code with a specific,
  demonstrated interface requirement." CLAUDE.md's rule exists specifically
  to prevent designing a shared interface off of *assumed* future
  requirements instead of *actual, observed* ones — which is exactly the
  failure mode of promoting now: we would be guessing at what shape
  `agent`'s RAG layer needs (raw ONNX session wrapper vs. a higher-level
  "embed documents, get chunked/pooled vectors" API vs. batch-embedding
  with async/streaming support for a vector-store ingestion pipeline) before
  a single line of `agent` code exists to tell us.
- Issue #14 itself hedges on this exact point in its "Proposed scope"
  section: *"the answer might legitimately be 'no, they're similar in shape
  but different enough in requirements to stay separate.'"* The evidence
  available at evaluation time doesn't resolve that hedge — it can't, since
  `agent` has no code to compare against yet.

**Conclusion: a filed, unimplemented issue does not count as a second real
module for CLAUDE.md's purposes.** The rule requires actual landed code
from a second module demonstrating the same interface need, not a
plausible future requirement, however well-evidenced the eventual need is.
Promoting now would be exactly the "shared typed interface built
preemptively" CLAUDE.md and architecture doc §2.9 warn against — the
interface would be designed by extrapolation from `clean`'s needs alone,
dressed up as being for two modules.

## (c) Recommendation: wait

**Do not promote `embeddings.py` to `dscraft.core` in this pass.** Revisit
when `dscraft.agent`'s RAG pipeline (#12) lands with real code that
actually calls an embeddings backend. At that point:

1. **Compare the two real implementations.** Does `agent`'s landed
   embeddings code use the same `onnxruntime.InferenceSession`-plus-
   preprocessor-callable shape as `clean.embeddings.EmbeddingModel`, or a
   materially different one (e.g. wrapping `fastembed`'s own API, or
   supporting multiple backend libraries behind one interface — which would
   itself be a small "embeddings router," arguably a different and larger
   piece of infrastructure than what `clean` has today)?
2. **If the shapes genuinely converge** (e.g. `agent` also settles on a
   bare ONNX Runtime session + pluggable preprocessor, PyTorch-free, for at
   least one of its backend options): promote at that point. The concrete,
   scoped move would be — narrower than "the whole file," per the (a)
   section above:
   - Move only `EmbeddingModel`, `PreprocessorOutput`, and
     `hashing_bag_of_words_vectorizer` to a new
     `packages/dscraft/src/dscraft/core/embeddings.py`. These are the
     generic runtime embedding *infrastructure*; nothing else in the file
     qualifies.
   - Leave `MODEL_ALLOWLIST`, `RECOMMENDED_MODEL_NAME`,
     `_RECOMMENDED_MODEL_REVISION`, `_RECOMMENDED_MODEL_ONNX_URL`, and
     `download_recommended_model()` in `dscraft/clean/embeddings.py` — these
     stay per-module by design (see part (a) above), and `agent` would
     define its own `Allowlist`/recommended-checkpoint entry alongside its
     own recommended model, per `dscraft.core.licensing`'s existing
     per-module-ownership pattern.
   - Also leave `build_synthetic_embedding_onnx()` and
     `build_synthetic_embedding_model()` in `dscraft/clean/embeddings.py`
     (or move them into `clean`'s own `tests/`/`examples/` tree) — these are
     `clean`-specific test/example fixtures, not infrastructure other
     modules need, and should not follow `EmbeddingModel` into `core` even
     if the promotion happens. A future `agent` promotion candidate would
     bring its own hermetic test/example fixtures the same way, not reuse
     `clean`'s synthetic ONNX graph builder.
   - `dscraft/clean/embeddings.py` becomes a thin re-export
     (`from dscraft.core.embeddings import EmbeddingModel, ...`) so
     `dscraft.clean.embeddings.EmbeddingModel` keeps working for existing
     callers/tests without a breaking import-path change, while
     `build_synthetic_embedding_onnx`/`build_synthetic_embedding_model`
     remain defined directly in `clean` (they are not re-exports of
     anything in `core`).
   - `dscraft.core`'s `<extra>` situation: the promoted module still needs
     `onnxruntime`/`numpy` to *import*, since `EmbeddingModel` references
     them at module level, not lazily. (The `onnx` package's graph-builder
     API, used only by `build_synthetic_embedding_onnx`, stays a `clean`-only
     import under this narrower scope — it would not need to become a
     `dscraft.core` dependency at all.) Under this package's "core stays
     thin, near-zero deps" contract (`dscraft.core`'s only unconditional
     dependency today is `opentelemetry-api`), that would still force
     `onnxruntime`+`numpy` onto every `dscraft.core` import unless the
     promoted module is instead
     gated behind a new `core`-scoped extra (e.g. `dscraft[core-embeddings]`)
     with lazy internal imports, mirroring how `automl-onnx` is already
     split out from `automl`'s base extra specifically to avoid forcing the
     ONNX stack onto every `automl` install. **This dependency-footprint
     problem is itself a second, independent reason to wait**: promoting
     today would either (i) silently add onnxruntime+numpy as new
     unconditional `dscraft.core` runtime dependencies — directly
     contradicting CLAUDE.md's "`dscraft.core` stays thin" rule — or (ii)
     require inventing a new core-scoped extra/lazy-import convention that
     doesn't exist in `dscraft.core` today, which is exactly the kind of
     new shared-infrastructure surface area that should wait for a second
     real consumer to justify its shape.
3. **If the shapes diverge** (e.g. `agent` needs multi-backend selection,
   async batch embedding for vector-store ingestion, or a PyTorch-based
   option that `clean` can never use): keep them separate permanently, and
   record that outcome as the settled answer so a third module's embeddings
   need doesn't re-litigate this same question from scratch — per issue
   #14's own acceptance criteria ("If not shared: the reasoning is
   documented so it isn't re-litigated next time").

## What was NOT done in this pass

No code was moved. `packages/dscraft/src/dscraft/clean/embeddings.py`,
`packages/dscraft/src/dscraft/core/`, and their respective test suites are
unchanged by this evaluation. This is deliberate: per this document's own
conclusion, the promotion is not yet justified, and per the task
instructions for issue #14, implementation was only warranted if the
evaluation itself concluded "promote now."
