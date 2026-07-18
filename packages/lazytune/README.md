# benchcraft-tune

A scaffold-depth implementation of one signature capability from
Benchcraft's LazyTune module (architecture doc Part 3, "Module 6:
LazyTune"): the **Adapter-Factory pattern's `BaseTrainingAdapter`
interface, with one concrete `ProgrammaticAdapter` implementation that
performs a real (tiny) LoRA fine-tuning step** on a small local causal
language model, via the standalone `peft` + `transformers` libraries.

## Scope

LazyTune's whole reason for existing (per the architecture doc) is to
unify PyTorch/HuggingFace-based fine-tuning tooling (Axolotl,
LLaMA-Factory, Unsloth, TRL, torchtune) behind one adapter interface. The
"no PyTorch/HuggingFace" constraint that shapes `benchcraft_lazyclean`
belongs to *that* module specifically (it exists to stay under a ~100MB,
non-PyTorch footprint for embeddings); it does not apply here. `torch`,
`transformers`, and `peft` are intentional, expected dependencies of this
package.

In scope for this pass:

1. `BaseTrainingAdapter` -- a minimal abstract interface (`prepare`,
   `train_step`, `save_adapter`) representing the Adapter-Factory pattern's
   shared shape (`adapter.py`).
2. `ProgrammaticAdapter` -- the one concrete, in-process implementation
   (the architecture doc's "`ProgrammaticAdapter`s (Unsloth, TRL)" family,
   as opposed to the subprocess-isolated `SubprocessAdapter` family for
   torchtune/Axolotl via `torchrun`). It loads a small causal LM, wraps it
   with `peft` LoRA, and runs **real** forward + backward + optimizer-step
   training -- not a mock.
3. `export_gguf_stub` / `export_mlx_stub` -- documented stubs for the
   narrowed export interface shape (see "GGUF/MLX export stub rationale").

## What's deferred, and why

Everything below is out of scope for this pass, tracked as future work per
the architecture doc, not silently dropped:

- **`SubprocessAdapter` (torchtune/Axolotl via `torchrun`)**. The
  architecture doc's Adapter-Factory pattern pairs `ProgrammaticAdapter`s
  with subprocess-isolated `SubprocessAdapter`s for recipe-driven trainers
  invoked via `torchrun`. That requires real subprocess orchestration,
  config-file generation, and process-lifecycle management this scaffold
  pass does not attempt. `BaseTrainingAdapter`'s interface is intentionally
  general enough that a future `SubprocessAdapter` could implement it
  without changing the interface.
- **Multi-fidelity BOHB micro-tuning + Multi-Power-Law/Shifted-Power-Law
  scale translation.** The architecture doc's hyperparameter-search system
  (train small proxy models on a data subsample via BOHB, extrapolate
  full-scale convergence) is a substantial system in its own right,
  unrelated to the adapter interface itself.
- **KL-penalization / reward-shaping for RL** (catastrophic-forgetting and
  reward-hacking mitigations for RL fine-tuning). This pass only
  demonstrates supervised LoRA fine-tuning (next-token loss), not RL.
- **Frobenius-norm-based early stopping and multi-GPU dependency-breakage
  risk mitigations** (Appendix A). Both are explicitly informational/
  deferred per the task -- not in scope for this pass, and lower priority
  given v1's single-device, MPS-primary scope.
- **Real GGUF/MLX export conversion.** See below.

## The tiny-model / licensing decision

This package needs "a real-ish causal LM" to demonstrate LoRA fine-tuning
without either bundling a multi-hundred-MB checkpoint or requiring network
access at test time -- the exact problem `benchcraft_lazyclean.embeddings`
already solved for ONNX embedding models (`build_synthetic_embedding_model`
plus a documented, lazy, optional real-model download path). This package
mirrors that same pattern:

- **Hermetic default (used by tests and the example):**
  `build_hermetic_causal_lm()` constructs a tiny GPT-2-architecture model
  **from scratch** via `transformers.GPT2Config` +
  `AutoModelForCausalLM.from_config` -- random weights, no download, no
  network access, no bundled file. This was chosen over pointing at a tiny
  HuggingFace Hub test-fixture checkpoint (e.g. `sshleifer/tiny-gpt2`,
  `hf-internal-testing/tiny-random-gpt2`) for two reasons: (1) those
  fixture repos' license fields are typically unset/unspecified on the Hub,
  which fails the task's own licensing bar cleanly rather than
  ambiguously; and (2) `from_config` construction is *more* hermetic than
  even a tiny download -- it needs no HTTP call, no Hub cache, and no
  `HF_HUB_OFFLINE`-style environment concern at all, ever. It sidesteps the
  licensing question entirely because there is no external checkpoint to
  license-check.
- **Real production path (documented, not exercised by tests, and requires
  one adjustment to run):** `MODEL_ALLOWLIST` (a per-module
  `lazycore.licensing.Allowlist` instance, per architecture doc §2.10)
  registers `openai-community/gpt2` (OpenAI's original GPT-2 "small", 124M
  params) as Tier 1, MIT-licensed. **Important:**
  `ProgrammaticAdapter.train_step` and `compute_loss` call
  `tokenizer.batch_encode(batch)` unconditionally -- a method that exists on
  `TinyTokenizer` but **not** on a real HuggingFace tokenizer (a real
  `AutoTokenizer`/`PreTrainedTokenizerBase` has no `batch_encode` method at
  all; the closest equivalent is calling the tokenizer directly, e.g.
  `tokenizer(texts, padding=True, return_tensors="pt")`). Passing a real
  `(model, tokenizer)` pair to `prepare()` and then calling `train_step()`
  as shown below will raise `AttributeError: 'GPT2TokenizerFast' object has
  no attribute 'batch_encode'` -- it is **not** a drop-in replacement yet.
  To actually fine-tune a real checkpoint, wrap the real tokenizer in a
  small adapter object that exposes a `batch_encode(texts) -> {"input_ids":
  ..., "attention_mask": ..., "labels": ...}` method (mirroring
  `TinyTokenizer.batch_encode`'s contract, including setting padded
  positions in `labels` to `-100`) and pass `(model, wrapped_tokenizer)` to
  `prepare()` instead:

  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from benchcraft_lazytune import ProgrammaticAdapter, MODEL_ALLOWLIST, RECOMMENDED_BASE_MODEL_NAME

  MODEL_ALLOWLIST.check(RECOMMENDED_BASE_MODEL_NAME)  # confirms Tier 1, no opt-in flag needed

  model = AutoModelForCausalLM.from_pretrained(RECOMMENDED_BASE_MODEL_NAME)  # network access on first use
  tokenizer = AutoTokenizer.from_pretrained(RECOMMENDED_BASE_MODEL_NAME)

  class RealTokenizerAdapter:
      """Minimal shim so a real HF tokenizer satisfies TinyTokenizer's batch_encode contract."""

      def __init__(self, tokenizer):
          self._tokenizer = tokenizer

      def batch_encode(self, texts):
          encoded = self._tokenizer(texts, padding=True, return_tensors="pt")
          labels = encoded["input_ids"].clone()
          labels[encoded["attention_mask"] == 0] = -100
          return {**encoded, "labels": labels}

  adapter = ProgrammaticAdapter()
  adapter.prepare((model, RealTokenizerAdapter(tokenizer)), dataset=["your real training text..."])
  result = adapter.train_step(["a training batch..."])
  ```

  This shim (and the `train_step`/`compute_loss` call sites that assume a
  `TinyTokenizer`-shaped `batch_encode`) is scaffold-depth, illustrative
  code, not a tested/shipped part of this package -- see "What's deferred
  and why" above.

## MoE-over-dense guidance (informational, no code impact at this scope)

Per architecture doc §Part 1/§Part 4/CLAUDE.md: default local-LLM guidance
across the platform favors Mixture-of-Experts architectures over dense
models at large parameter counts (e.g. a ~70B-total/~3B-active MoE is
genuinely fast on the 128GB unified-memory reference hardware; a dense 70B
model is not). This tiny-LoRA-demo scaffold operates on a model with a few
thousand parameters, so MoE-vs-dense selection has no code-level relevance
here -- it's noted for context. A future model-selection helper for this
module (out of scope for this pass) should default-recommend MoE
architectures once real, larger base models are wired in via the
`(model, tokenizer)` `model_ref` path.

## GGUF/MLX export stub rationale

Per architecture doc §2.5 (export backend 2), LazyTune's v1 export scope
is narrowed to local-only serving formats -- GGUF (llama.cpp) and
MLX-native (Apple Silicon) -- with the original
vLLM/SGLang/TensorRT-LLM/AutoAWQ-Marlin cloud-serving pipeline deferred.
`export.py`'s `export_gguf_stub` / `export_mlx_stub` document that
narrowed interface's shape (what arguments a real export call would take)
without performing real conversion, because:

- Real GGUF export requires llama.cpp's own conversion scripts
  (`convert_hf_to_gguf.py`, `llama-quantize`), which track a fast-moving,
  architecture-specific HF-to-GGUF tensor-layout mapping.
- Real MLX export requires Apple's `mlx-lm` conversion tooling
  (`mlx_lm.convert`), tied to a specific MLX/Apple Silicon toolchain.

Both are heavyweight, environment-specific external build dependencies --
vendoring or reimplementing either is explicitly out of scope for this
pass. Both functions raise `NotImplementedError` with a docstring/message
explaining exactly why and what real dependency would be needed.

## Public API

```python
from benchcraft_lazytune import (
    BaseTrainingAdapter,       # abstract interface: prepare / train_step / save_adapter
    ProgrammaticAdapter,       # the one concrete in-process LoRA fine-tuning adapter
    TrainStepResult,           # dataclass: loss (float), step (int)
    TinyTokenizer,             # hermetic, corpus-derived word-level tokenizer
    build_hermetic_causal_lm,  # from-scratch tiny GPT-2-architecture model + TinyTokenizer
    default_lora_config,       # the LoraConfig ProgrammaticAdapter uses by default
    MODEL_ALLOWLIST,           # lazycore.licensing.Allowlist for this module
    RECOMMENDED_BASE_MODEL_NAME,  # "openai-community/gpt2" (Tier 1, MIT)
    export_gguf_stub,          # NotImplementedError stub documenting GGUF export shape
    export_mlx_stub,           # NotImplementedError stub documenting MLX export shape
)

adapter = ProgrammaticAdapter()
adapter.prepare(None, dataset=["some training text...", "more training text..."])
result = adapter.train_step(["a training batch of text rows..."])
print(result.loss, result.step)
adapter.save_adapter("/tmp/my-lora-adapter")
```

`prepare(model_ref, dataset)` accepts `model_ref=None` to build the
hermetic from-scratch model (sized off `dataset`'s vocabulary), or an
explicit `(model, tokenizer)` tuple to fine-tune a real base model (see
"The tiny-model / licensing decision" above).

## Installation

`lazycore` is a local sibling package under `packages/lazycore`, declared as
a bare (unpinned) `pyproject.toml` dependency -- matching the convention
already established by `packages/automl`, `packages/lazyforecast`, and
`packages/lazygraph`. Since it isn't published to a package index, install
it first so pip can resolve the dependency from the local editable install:

```bash
pip install -e packages/lazycore
pip install -e "packages/lazytune[dev]"
```

`torch`/`transformers`/`peft` installs can take a while (CPU wheels are
still sizeable) -- this is expected.

## Running tests

```bash
pytest packages/lazytune/tests
```

Fully hermetic -- no network access required. Uses
`build_hermetic_causal_lm()` throughout; tests assert the training loss is
a finite float, that LoRA parameters actually change after real training
steps, and that `save_adapter` writes real, reloadable adapter files to
disk.

## Running the example

```bash
python packages/lazytune/examples/lora_finetune_example.py
```

Prints parameter counts, a before/after loss comparison across real LoRA
training steps, and the path the trained adapter was saved to.
