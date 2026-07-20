"""Tests for the BaseTrainingAdapter interface + ProgrammaticAdapter LoRA fine-tuning.

Fully hermetic: uses build_hermetic_causal_lm() (a from-scratch,
randomly-initialized GPT-2-architecture model via transformers.GPT2Config +
AutoModelForCausalLM.from_config), never touches the network, and never
bundles/downloads a real checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from dscraft.tune import (
    MODEL_ALLOWLIST,
    RECOMMENDED_BASE_MODEL_NAME,
    BaseTrainingAdapter,
    ProgrammaticAdapter,
    TinyTokenizer,
    TrainStepResult,
    build_hermetic_causal_lm,
)
from dscraft.tune.export import export_gguf_stub, export_mlx_stub
from dscraft.core.licensing import ModelTier

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a slow green turtle naps under the warm sun",
    "the lazy dog sleeps all afternoon in the shade",
    "quick foxes and lazy dogs share the same field",
]


def test_programmatic_adapter_is_a_base_training_adapter():
    """ProgrammaticAdapter must implement the Adapter-Factory base interface."""
    assert issubclass(ProgrammaticAdapter, BaseTrainingAdapter)


def test_base_training_adapter_cannot_be_instantiated_directly():
    """BaseTrainingAdapter is abstract (via abc.ABC) and cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseTrainingAdapter()  # type: ignore[abstract]


def test_build_hermetic_causal_lm_requires_no_network_and_is_tiny():
    """The from-scratch GPT-2-architecture model builds without network access and stays tiny (<200K params)."""
    model, tokenizer = build_hermetic_causal_lm(CORPUS, n_embd=16, n_layer=1, n_head=1)
    assert isinstance(tokenizer, TinyTokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    # A genuinely tiny model -- order of tens of thousands of params, not
    # anywhere near a production-scale checkpoint.
    assert 0 < n_params < 200_000


def test_tiny_tokenizer_encode_and_batch_encode_shapes():
    """encode() returns plain ints; batch_encode() pads a batch to equal length and tracks each row's true (unpadded) length via attention_mask."""
    tokenizer = TinyTokenizer.build_from_corpus(CORPUS)
    encoded = tokenizer.encode("the quick brown fox")
    assert all(isinstance(i, int) for i in encoded)

    batch = tokenizer.batch_encode(["the quick fox", "a slow turtle naps"])
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].dtype == torch.long
    # Shorter sequence padded to match the longer one.
    assert batch["attention_mask"][0].sum().item() == 3
    assert batch["attention_mask"][1].sum().item() == 4


def test_prepare_requires_call_before_train_step_or_save():
    """train_step() and save_adapter() must raise RuntimeError if prepare() was never called."""
    adapter = ProgrammaticAdapter()
    with pytest.raises(RuntimeError):
        adapter.train_step(["the quick brown fox"])
    with pytest.raises(RuntimeError):
        adapter.save_adapter("/tmp/should-not-be-created")


def test_train_step_runs_and_returns_finite_loss():
    """A single train_step() after prepare() returns a TrainStepResult with step=1 and a finite, positive loss."""
    adapter = ProgrammaticAdapter()
    adapter.prepare(None, CORPUS)

    result = adapter.train_step(CORPUS[:2])
    assert isinstance(result, TrainStepResult)
    assert result.step == 1
    assert torch.isfinite(torch.tensor(result.loss))
    assert result.loss > 0.0


def test_train_step_actually_changes_lora_parameters():
    """Multiple train_step() calls must produce a genuine gradient update: at least one LoRA parameter tensor changes value."""
    adapter = ProgrammaticAdapter()
    adapter.prepare(None, CORPUS)

    lora_params_before = {
        name: param.detach().clone()
        for name, param in adapter.model.named_parameters()
        if param.requires_grad
    }
    assert lora_params_before, "expected at least one trainable LoRA parameter"

    for _ in range(3):
        adapter.train_step(CORPUS)

    changed = False
    for name, param in adapter.model.named_parameters():
        if name in lora_params_before and not torch.equal(param.detach(), lora_params_before[name]):
            changed = True
            break
    assert changed, "expected LoRA parameters to change after real training steps"


def test_multiple_train_steps_move_the_loss_and_increment_step_count():
    """10 train_step() calls should increment the step counter to 10 and measurably move the loss (a real optimization signal, not a no-op)."""
    adapter = ProgrammaticAdapter(learning_rate=1e-2)
    adapter.prepare(None, CORPUS)

    loss_before = adapter.compute_loss(CORPUS)
    for _ in range(10):
        result = adapter.train_step(CORPUS)
    loss_after = adapter.compute_loss(CORPUS)

    assert result.step == 10
    assert torch.isfinite(torch.tensor(loss_before))
    assert torch.isfinite(torch.tensor(loss_after))
    # A real, if tiny, optimization signal: loss should move measurably
    # after 10 real gradient steps at a deliberately punchy learning rate.
    assert loss_after != pytest.approx(loss_before)


def test_save_adapter_writes_reloadable_files(tmp_path):
    """save_adapter() writes adapter_config.json + adapter weights, and the saved directory can be reloaded onto a fresh matching base model via PeftModel.from_pretrained."""
    from peft import PeftModel

    adapter = ProgrammaticAdapter()
    adapter.prepare(None, CORPUS)
    adapter.train_step(CORPUS)

    save_dir = tmp_path / "lora-adapter"
    adapter.save_adapter(save_dir)

    assert save_dir.exists()
    written_files = list(save_dir.iterdir())
    assert any(f.name == "adapter_config.json" for f in written_files)
    assert any("adapter_model" in f.name for f in written_files)

    # Reload: build a fresh matching base model + wrap with the saved adapter.
    fresh_base_model, tokenizer = build_hermetic_causal_lm(CORPUS)
    reloaded = PeftModel.from_pretrained(fresh_base_model, str(save_dir))
    assert isinstance(reloaded, PeftModel)


def test_recommended_base_model_registered_as_tier_1():
    """The documented real base-model checkpoint (openai-community/gpt2) is registered in MODEL_ALLOWLIST as Tier 1 (MIT, auto-usable)."""
    entry = MODEL_ALLOWLIST.get(RECOMMENDED_BASE_MODEL_NAME)
    assert entry is not None
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "MIT"


def test_export_stubs_raise_not_implemented(tmp_path):
    """Both export stubs must always raise NotImplementedError with a message identifying the intended format (GGUF / MLX)."""
    with pytest.raises(NotImplementedError, match="GGUF"):
        export_gguf_stub(tmp_path / "adapter", tmp_path / "model.gguf")
    with pytest.raises(NotImplementedError, match="MLX"):
        export_mlx_stub(tmp_path / "adapter", tmp_path / "model-mlx")
