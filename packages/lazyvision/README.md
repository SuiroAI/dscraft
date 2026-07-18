# benchcraft-vision

Benchcraft's computer-vision module (internal codename "LazyVision",
architecture doc Part 3 "Module 5: LazyVision"). This is a **scaffold-depth
pass**, not a full implementation of the module's eventual scope.

## What this package is (and isn't) right now

The full LazyVision module is designed to unify CNN classifiers, Vision
Transformers, real-time object detectors (YOLO-family, D-FINE, RT-DETR),
and acoustic/spectrogram models under one preprocessing abstraction, backed
by a native Rust/PyO3 data-loading layer.

**This package currently implements exactly one signature capability slice:
a small CNN image classifier, captured via `torch.export()` and exported to
ONNX, plus the first concrete `lazycore.data.DenseMediaPipeline` subclass
handling decode → augment → to-dense-tensor preprocessing.** Everything
else named above is future work, not partially stubbed out here (see
"Deferred" below).

## The signature capability

### 1. `SimpleImagePipeline` — the first concrete Tier-3 pipeline

`lazycore.data.DenseMediaPipeline` (architecture doc §2.1, Tier 3: dense
image/audio) is an abstract interface: LazyCore defines the pipeline
*shape* (decode → augment → to-dense-tensor, with a DLPack handoff only at
the final dense-tensor stage) but implements none of it and depends on no
image/tensor library. `benchcraft_lazyvision.SimpleImagePipeline` is the
first concrete subclass:

```python
from benchcraft_lazyvision import SimpleImagePipeline, PipelineConfig

pipeline = SimpleImagePipeline(PipelineConfig(image_size=32, horizontal_flip_prob=0.5))
dense_tensor = pipeline.run(raw_image_bytes)  # decode -> augment -> to_dense_tensor
# dense_tensor is a torch.Tensor, shape (3, 32, 32), float32, range [0, 1]
```

- **`decode`** uses Pillow (`PIL.Image.open`) to turn raw encoded bytes
  (PNG/JPEG/...) into an RGB `PIL.Image`.
- **`augment`** resizes to a fixed square size and, with a configurable
  probability, applies a horizontal flip — a real, if simple, augmentation
  (not a stub), and one that also anchors every sample to a fixed,
  ONNX-export-friendly static shape.
- **`to_dense_tensor`** converts to a `(C, H, W)` float32 `torch.Tensor`
  normalized to `[0, 1]`. `torch.Tensor` implements `__dlpack__` /
  `__dlpack_device__` natively, so this return value already satisfies
  `lazycore.data`'s `_SupportsDLPack` protocol with no extra conversion
  step — the "DLPack handoff at the final dense-tensor stage" the
  architecture doc describes is simply "return a `torch.Tensor`" here.

This subclasses `lazycore.data.DenseMediaPipeline` directly; it does not
redefine a parallel interface.

### 2. `TinyCNN` — a small CNN classifier

`benchcraft_lazyvision.TinyCNN` is a minimal LeNet-style classifier (two
`conv → relu → maxpool` blocks, one linear head) — deliberately small,
since the point of this pass is proving the export path works correctly,
not achieving accuracy. `build_model()` returns a deterministically
(seed-controlled), untrained model:

```python
from benchcraft_lazyvision import ModelConfig, build_model

model = build_model(ModelConfig(in_channels=3, image_size=32, num_classes=10), seed=0)
```

Training a real model on real data is **explicitly not required** at this
scope (per the task brief) — an initialized-but-untrained model is
sufficient to validate correctness of the export mechanism. For
input/output-shape-compatible synthetic data (random tensors standing in
for a small image-classification task), use
`synthetic_classification_batch()` — fully in-memory, no network access, no
dataset download.

### 3. `export_to_onnx` / `verify_export` — the `torch.export` → ONNX path

```python
from benchcraft_lazyvision import export_to_onnx, verify_export

export_to_onnx(model, example_input, "tiny_cnn.onnx")
result = verify_export(model, "tiny_cnn.onnx", fresh_random_input, atol=1e-4, rtol=1e-3)
assert result.matched
```

**Exact mechanism used, and why:** this module calls
`torch.export.export()` to capture the model as a functional
`ExportedProgram` (structural/shape tracing via TorchDynamo — the same
capture frontend the architecture doc calls out, in §2.5, as a legitimate
shared step with the future, deferred edge-compilation module), then passes
that `ExportedProgram` directly into `torch.onnx.export(..., dynamo=True)`
to lower it to ONNX, and saves the result via the returned
`torch.onnx.ONNXProgram.save(path)`.

This corner of the PyTorch API has moved around across versions, which is
worth documenting explicitly rather than leaving implicit:

- `torch.onnx.dynamo_export()` was the original entrypoint for
  dynamo/`torch.export`-based ONNX export (introduced around PyTorch 2.1).
- It was **deprecated starting in PyTorch 2.5** in favor of
  `torch.onnx.export(model, args, dynamo=True)` — the same top-level
  `torch.onnx.export` function used by the legacy TorchScript-tracing
  exporter, now with a `dynamo=True` flag that routes through the
  `torch.export`/FX-graph-based lowering path instead.
- `dynamo_export` was fully **removed in later PyTorch releases**.

This package therefore uses **`torch.onnx.export(exported_program, (example_input,), dynamo=True)`**
— not `dynamo_export`, and not the legacy tracing-based exporter without
`dynamo=True` — because it is the current, non-deprecated,
actively-maintained API as of the PyTorch version this package pins
(`torch>=2.5`, see `pyproject.toml`). Passing the already-captured
`ExportedProgram` (rather than the raw `nn.Module`) makes the two-step
"capture via `torch.export`, then lower to ONNX" structure explicit in
code, matching the architecture doc's phrasing of the capability
("`torch.export` → onnx-graphsurgeon"-style lowering) rather than
collapsing both steps into a single opaque call.

Correctness is verified by running the same input through both the
original PyTorch model and `onnxruntime.InferenceSession`, then comparing
outputs with `numpy.allclose` within a numerical tolerance — mirroring the
pattern already used by `packages/automl`'s `.compile()` tests. See
`tests/test_export.py` for the full suite (including a batch of *fresh*
random inputs never seen during tracing, and a negative test confirming
`verify_export` actually detects a real mismatch rather than trivially
passing).

## MPS note

Per CLAUDE.md, **MPS (`torch.device("mps")`) is this platform's primary
backend**, not CUDA. `resolve_device()` is this package's one canonical
device-selection helper — it mirrors
`benchcraft_lazygraph.gcn.resolve_device`'s exact pattern (MPS available? ->
use it; else CUDA available? -> use it; else CPU; an explicit but
unavailable/invalid `preferred` device falls back to auto-detection rather
than raising). `build_model`, `synthetic_classification_batch`, and
`PipelineConfig`/`SimpleImagePipeline` all default their `device` argument
to `None`, which routes through `resolve_device()` — so a caller who does
not specify a device gets **MPS-first-with-CPU-fallback**, not an implicit
CPU default. `TinyCNN` itself accepts an arbitrary device and contains no
CUDA-only or CPU-only assumptions.

This package's own **tests and the example script explicitly pin
`device="cpu"`** (the example resolves a device once via `resolve_device()`
and then propagates it consistently everywhere, so it exercises whatever
this package considers the canonical default — MPS on Apple Silicon, CPU
elsewhere). Tests instead pin `device="cpu"` explicitly, purely because CPU
is the fastest, most portable choice for automated, hermetic, deterministic
verification (no GPU required in CI, no MPS-availability check needed) —
not because MPS is unsupported. `export_to_onnx`/`verify_export` operate on
whatever device the model and example input already live on (internally
converting to CPU/numpy only where `onnxruntime`'s `CPUExecutionProvider`
requires it). This package does not yet include the architecture doc's
called-out v1 work item of auditing MPS kernel-support maturity for
attention/NMS layers, since neither is in scope for a plain CNN classifier.

## Deferred (explicitly out of scope for this pass)

Per the task brief and the architecture doc's full Module 5 description,
the following are **not** implemented here and are real future work, not
partial stubs:

- **Vision Transformers** and hybrid local-convolutional spatial-locality
  modules.
- **Real-time object detectors** (YOLO-family, D-FINE, RT-DETR) and their
  NMS-layer ONNX export concerns (`onnx-graphsurgeon` custom-operator
  mapping).
- **Acoustic/spectrogram models** (Mel-spectrogram STFT preprocessing).
- **The native Rust/PyO3 data-loading layer** with DLPack zero-copy
  handoff across process boundaries — `SimpleImagePipeline` is a pure
  in-process Python/Pillow/PyTorch implementation.
- **Sharpness-Aware Minimization and Layer-wise Learning Rate Decay**
  training-time regularization for small-dataset ViT stability.
- **The AGPL/GPL-detector subprocess-isolation plugin architecture.**
  Per CLAUDE.md's licensing policy and the architecture doc §2.2: this
  package contains a plain CNN classifier only, no object-detection model
  of any kind, so the AGPL-3.0/GPL-3.0 concern around Ultralytics
  YOLOv8/v11, Tsinghua YOLOv12, or GPL-3.0 RTMDet does not apply here.
  **`ultralytics` and no other AGPL/GPL-licensed detector package is
  anywhere in this package's dependency tree.**

## Dependency surface

Per architecture doc §2.7 ("PyTorch-heavy modules... have genuinely
conflicting dependency universes") and matching the precedent set by
`packages/lazygraph`: this module's whole signature capability requires
`torch` — there is no meaningful degraded mode of either the pipeline
(needs a tensor type) or the export path (needs `torch.export`) without
it — so `torch`/`onnx`/`onnxruntime` are **core dependencies**, not an
optional extra the way AutoML's ONNX path is.

- **Core (always installed):** `numpy`, `pillow`, `torch`, `onnx`,
  `onnxruntime`.
- **Optional `dev` extra:** `pytest`, `scikit-learn`. `scikit-learn` is
  included purely as a source of real, locally-bundled image data
  (`sklearn.datasets.load_digits()`) for real-dataset validation (see
  "Running tests" above) — it is **not** a core runtime dependency of this
  package and plays no role in `SimpleImagePipeline`, `TinyCNN`, or the
  export path themselves.

`torchvision` is deliberately **not** a dependency — this pass uses
synthetic in-memory images (Pillow-encoded bytes, or random tensors via
`synthetic_classification_batch`) for hermetic, network-free tests and
examples; no `torchvision` dataset download is needed.

`lazycore` is a local sibling package (`packages/lazycore`) and is
**installed separately, not as a formal pyproject dependency of this
package** — hatchling/pip don't have a portable, idiomatic way to express a
relative-path dependency in `pyproject.toml` metadata (unlike e.g. Poetry's
`path = "../lazycore"`). Install it first (see below), matching the
convention established in `packages/automl`, `packages/lazyclean`, and
`packages/lazygraph`.

## Installation (local dev)

```bash
# from the repo root
pip install -e packages/lazycore
pip install -e "packages/lazyvision[dev]"
```

## Running tests

```bash
pytest packages/lazyvision/tests
```

All tests are hermetic (no network access, no dataset download): synthetic
in-memory image bytes and random tensors are the primary stand-in for real
data (`test_pipeline.py`, `test_export.py`), **plus** a dedicated real-data
validation module, `tests/test_real_dataset_validation.py`, which runs the
exact same `SimpleImagePipeline` → `TinyCNN` → `export_to_onnx` →
`verify_export` path against a real, genuinely bundled-as-package-data
dataset — `sklearn.datasets.load_digits()`'s 8x8 grayscale handwritten
digit images — confirming the ONNX export correctness check (PyTorch vs.
ONNX Runtime output) holds on real image data, not just random tensors.
This still requires no network access: `load_digits()` reads a small CSV
shipped inside the installed `scikit-learn` package, no download involved.
`scikit-learn` is a **dev/test-only** dependency for this purpose (see
"Dependency surface" below) — it plays no role in this package's runtime
logic.

## Running the example

```bash
python packages/lazyvision/examples/export_cnn_example.py
```

Runs the full pipeline → model → export → verify flow twice, printed as two
clearly separated sections:

1. **Synthetic section:** builds a `SimpleImagePipeline`, runs a synthetic
   gradient-pattern image through it to produce a dense tensor, builds a
   `TinyCNN` sized to match, exports it via `torch.export` → ONNX, and
   verifies the ONNX Runtime output matches the original PyTorch model's
   output (on a fresh batch of random inputs, not just the tracing example)
   within tolerance.
2. **Real-dataset section:** repeats the identical flow — same
   `SimpleImagePipeline`, same `TinyCNN`, same `export_to_onnx`/
   `verify_export` calls — on one real handwritten-digit image from
   scikit-learn's bundled `sklearn.datasets.load_digits()` dataset, printing
   the same PyTorch-vs-ONNX correctness numbers for that real image. This
   section requires the `dev` extra installed (`pip install -e
   "packages/lazyvision[dev]"`), since scikit-learn is dev/test-only here.
