# 2026-07 image-based near-duplicate detection evaluation

- **Status:** Accepted — implemented a minimal first step (`clean`, ONNX-based)
- **Issue:** [gr3enarr0w/dscraft#33](https://github.com/gr3enarr0w/dscraft/issues/33) — "[clean/vision] Evaluate
  image-based near-duplicate detection (CLIP embeddings, face detection)"
- **Scope:** Effort 0.5, evaluation-only per the roadmap; implement only a "genuinely small, safe first step" if the
  evidence supports it now.

## The architectural tension this evaluation resolves

`dscraft.clean` already has near-duplicate detection (`detect_near_duplicate_text`, LSHBloom contamination
screening), but only for text. `photo_dedupe_project` (the real local project cited in the issue) does image dedup
via `open-clip-torch` — a PyTorch-based CLIP embedding model. `dscraft.clean` has a hard, deliberate PyTorch-free
constraint (ONNX Runtime only, <100MB target — CLAUDE.md's shared-architecture section); `dscraft.vision` has no
such constraint but also has no dedup capability. The question this issue asks: does a Tier-1-licensed,
ONNX-exportable, <100MB-class CLIP-family image encoder exist? If yes, this belongs in `clean`, preserving the
PyTorch-free constraint. If no, it belongs in `vision`, or shouldn't be built yet.

## Research findings

### 1. A candidate, ONNX-exportable, <100MB CLIP vision encoder exists — licensing is pending verification, not settled Tier 1

`openai/clip-vit-base-patch32`'s own GitHub repository (`github.com/openai/CLIP`) is **MIT-licensed**
(`LICENSE` file: "MIT License · Copyright (c) 2021 OpenAI"). The Hugging Face community re-export
`Xenova/clip-vit-base-patch32` (a Transformers.js-compatible ONNX conversion of that same checkpoint) publishes
the vision tower and text tower as **separate** ONNX graphs, in multiple quantization levels:

| File | Size |
|---|---|
| `vision_model.onnx` (fp32) | 352 MB |
| `vision_model_fp16.onnx` | 176 MB |
| `vision_model_quantized.onnx` | 89.1 MB |
| **`vision_model_int8.onnx`** | **88.6 MB** |
| `vision_model_uint8.onnx` | 88.6 MB |
| `vision_model_q4.onnx` | 63.6 MB |
| `vision_model_bnb4.onnx` | 58.3 MB |

`vision_model_int8.onnx` at **88.6MB** is directly comparable to `embeddings.py`'s already-accepted text-embedding
precedent (the recommended `all-MiniLM-L6-v2` checkpoint's fp32 export is documented there as "~90MB fp32 /
~23MB int8-quantized"), and sits comfortably under this module's <100MB target. This is exactly the
"the ONNX file exists, is Tier-1, and is ~90MB" scenario the issue's own acceptance criteria anticipated as a
"genuinely small first step" trigger.

**License caveat, documented plainly rather than glossed over — and, per 2026-07 CodeRabbit review, decisive
enough to require Tier 2, not Tier 1:** OpenAI's own Hugging Face model card for `clip-vit-base-patch32` does not
carry an explicit SPDX `license:` tag in its metadata. The initial pass through this evaluation classified the
checkpoint Tier 1 on the theory that the widely-held community inheritance from the MIT-licensed `openai/CLIP`
*code* repository (the same repository that produced and distributes these pretrained weights) was sufficient
evidence. On review, that reasoning does not hold up against CLAUDE.md's LazyIsolate policy: evidence about the
*code's* license is not a first-party license declaration on the *weights/checkpoint* themselves, and an
unverified checkpoint must not be auto-usable as Tier 1 — that is exactly the scenario Tier 2
(`accept_restricted_licenses=True`) exists for. **Conclusion revised: this checkpoint is registered Tier 2
(`ModelTier.TIER_2`) in `image_dedup.py`, pending verification, not a settled Tier-1 conclusion.** This is
analogous to — but weaker than — `embeddings.py`'s existing `Xenova/all-MiniLM-L6-v2` entry, which cites an
explicit `Apache-2.0` tag and *is* Tier 1. Per CLAUDE.md's "maintaining and re-verifying allowlists is an ongoing
task, not one-time" policy, this caveat (and the Tier 2 classification) is recorded directly in the allowlist
entry's `notes` field (see `image_dedup.py`), not just in this document, so it travels with the code.
`download_recommended_clip_vision_model()` requires `accept_restricted_licenses=True` before it will download or
reuse a cached copy of this checkpoint, via the same `dscraft.core.licensing.Allowlist.check()` gate every other
Tier 2 entry in this codebase uses.

### 2. Face detection (`insightface`) and image-quality scoring (`pyiqa`) — deferred, per the issue's own instructions

The issue frames face-detection-based identity dedup and image-quality scoring as "natural companion pieces, not
separate capabilities," but its constraints section is explicit that InsightFace's model weights are "mixed
MIT/Apache... some non-commercial" and require a documented per-model check before adoption, and its acceptance
criteria only ask for a *decision on which module owns this capability* plus *a documented per-model license check
if InsightFace is adopted* — not an implementation. This evaluation makes the CLIP-embedding decision (below) but
does **not** evaluate or adopt InsightFace/pyiqa in this pass — that would be new capability scope (identity-aware
dedup, quality-based cluster-representative selection) beyond "does a viable embedding model exist," and neither
`pyiqa` nor `insightface` bears on the PyTorch-free-vs-not architectural question this issue exists to resolve
(both are companion features to a dedup capability, not the dedup capability itself). If face-detection/
quality-scoring dedup is wanted, it should be scoped as its own follow-up issue once the embedding-based dedup
foundation this evaluation adds is in place.

## Decision

**Image near-duplicate detection belongs in `dscraft.clean`, not `dscraft.vision`**, via the ONNX-exported CLIP
vision tower — preserving the PyTorch-free constraint exactly as the issue's option 1 describes, and exactly
mirroring `embeddings.py`'s existing ONNX Runtime pattern (synthetic hermetic fixture for tests + a documented,
lazy, optional production-download path, never bundled, never network-required at import/test time).

**Implemented this pass** (not merely proposed) — `packages/dscraft/src/dscraft/clean/image_dedup.py`:

- `ImageEmbeddingModel` — the image-modality analogue of `embeddings.EmbeddingModel`, wrapping an
  `onnxruntime.InferenceSession` plus a preprocessor callable. No `pillow`, no `torch`, no CLIP-specific Python
  package — callers pass already-decoded `(H, W, 3)` uint8 numpy arrays; `clean` does not take on an image-decoding
  dependency (that stays `dscraft.vision`'s job, or the caller's).
- `resize_and_normalize()` — a dependency-free (`numpy`-only) nearest-neighbor resize + `[0, 1]` normalize, the
  image analogue of `hashing_bag_of_words_vectorizer`. Explicitly documented as *not* a real CLIP preprocessing
  pipeline (no bicubic resize, no CLIP-specific mean/std normalization) — a placeholder preprocessor for the
  synthetic test/example fixture, exactly matching the existing text path's scope boundary.
- `build_synthetic_image_embedding_onnx()` / `build_synthetic_image_embedding_model()` — hermetic ONNX graph
  fixtures (linear projection + L2 normalize over a downsampled-pixel feature vector), used by tests and any future
  example — no network access, no bundled multi-hundred-MB file, mirroring
  `embeddings.build_synthetic_embedding_onnx`/`build_synthetic_embedding_model` exactly.
- `download_recommended_clip_vision_model()` — documents (and, given network access and
  `accept_restricted_licenses=True`, performs) the production wiring to `Xenova/clip-vit-base-patch32`'s
  `vision_model_int8.onnx`, pinned to an immutable commit SHA (not `main`), with SHA-256 integrity verification
  (of both freshly-downloaded and previously-cached files) and an atomic download-then-rename, allowlist-gated
  exactly like `embeddings.download_recommended_model`. **Never called by tests, the example, or any import-time
  code.**
- `RECOMMENDED_IMAGE_MODEL_NAME` registered into the **same** `dscraft.clean.embeddings.MODEL_ALLOWLIST` instance
  (not a second allowlist) as **Tier 2** (pending verification — see the licensing caveat above), with the
  license caveat recorded in its `notes` field — per `dscraft.core.licensing.Allowlist`'s documented
  per-*module* (not per-file) ownership contract.
- `detect_near_duplicate_images()` — the one canonical entrypoint, mirroring `detect_near_duplicate_text`.
  **Deliberately reuses `dscraft.clean.dedup.find_near_duplicates` as-is** rather than reimplementing any
  near-duplicate-scanning logic: that function already operates on any `(n, dim)` embedding array regardless of
  modality, so no new dedup algorithm was needed, only the image-specific embedding step (per CLAUDE.md's "one
  canonical location per capability" rule — `dedup.py`'s near-duplicate scan is that one location, for every
  modality this package supports).

Public API exported from `dscraft.clean.__init__` alongside the existing text-dedup surface. No new dependency was
added to `pyproject.toml` — `image_dedup.py` uses only `numpy`/`onnx`/`onnxruntime`, all already present in the
`clean` extra.

Tests: `packages/dscraft/tests/clean/test_image_dedup.py` (13 tests) — covers the PyTorch-free static-source-scan
guarantee, the allowlist registration, `resize_and_normalize`'s shape/range/grayscale/error-handling behavior, the
synthetic ONNX model's determinism/L2-normalization/empty-input behavior, and `detect_near_duplicate_images`
end-to-end (including an explicit "produces the same report `dedup.find_near_duplicates` would" regression test,
enforcing the no-reimplementation decision above).

## What was deliberately *not* done in this pass

- **Real CLIP preprocessing** (bicubic 224×224 resize, CLIP's specific per-channel mean/std normalization) is not
  implemented — `resize_and_normalize` is a synthetic-fixture stand-in, exactly like `hashing_bag_of_words_vectorizer`
  is for text. A caller wiring in the real `vision_model_int8.onnx` checkpoint needs to supply a real CLIP
  preprocessor as their `ImageEmbeddingModel.preprocessor`; this module does not ship one. Building a real,
  dependency-free (no Pillow/torchvision) CLIP preprocessor is a reasonable, scoped follow-up but was not required
  to answer this issue's core "does a viable model exist and where does it belong" question, and risked scope creep
  beyond "genuinely small."
- **Face detection (InsightFace) and image-quality scoring (pyiqa)** — explicitly deferred, per the "Research
  findings" section above; not evaluated, not adopted, not implemented.
- **The `dscraft.vision` alternative** (PyTorch-based CLIP via `open-clip-torch`, reusing `vision`'s existing
  PyTorch dependency) was considered and rejected for this capability specifically because a viable ONNX path
  exists and the issue's own constraint is explicit: "if CLIP can't be cleanly ONNX-exported for this use case,
  this capability belongs in vision, not clean" — the converse holds here: it *can* be cleanly ONNX-exported, so it
  belongs in `clean`. (This ONNX-exportability finding is independent of the Tier 1/Tier 2 licensing question
  above — Tier 2 gating governs *whether the checkpoint auto-downloads*, not which subpackage owns the
  capability.)

## Bottom line

A candidate, ONNX-exportable, <100MB-class (88.6MB int8) CLIP vision encoder exists and is architecturally viable
for `dscraft.clean`. **Its licensing is Tier 2 (pending verification), not a settled Tier 1 conclusion** — see the
revised licensing caveat above, added during 2026-07 CodeRabbit review. This evaluation implements the minimal,
safe first step directly — the `ImageEmbeddingModel`/`resize_and_normalize`/`detect_near_duplicate_images`
scaffold in `image_dedup.py`, following `embeddings.py`'s exact ONNX Runtime pattern, gated behind
`accept_restricted_licenses=True` and SHA-256 integrity verification — while explicitly deferring real CLIP
preprocessing and the face-detection/image-quality companion features to separate future scope.
