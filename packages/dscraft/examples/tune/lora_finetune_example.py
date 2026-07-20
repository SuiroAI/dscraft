"""Runnable demo: real (tiny) LoRA fine-tuning via the ProgrammaticAdapter.

This script only imports and calls the real `dscraft.tune` package
API -- it does not reimplement any model-building or training logic inline
(per CLAUDE.md's "no net-new scripts" rule). Run it with:

    python packages/dscraft/examples/tune/lora_finetune_example.py

It uses the fully hermetic from-scratch model
(`build_hermetic_causal_lm`, wired in automatically by
`ProgrammaticAdapter.prepare(None, dataset)`), so no network access or
bundled checkpoint file is required. See the package README for how a real
production base model (`openai-community/gpt2`, registered in
`MODEL_ALLOWLIST`) would be wired in via the same `ProgrammaticAdapter` API.
"""

from __future__ import annotations

import tempfile

from dscraft.tune import ProgrammaticAdapter

# A tiny synthetic text corpus -- just enough repeated structure for a few
# real LoRA gradient steps to visibly move the loss on a model this small.
DATASET = [
    "the quick brown fox jumps over the lazy dog",
    "a slow green turtle naps under the warm sun",
    "the lazy dog sleeps all afternoon in the shade",
    "quick foxes and lazy dogs share the same field",
    "the warm sun helps the green turtle nap longer",
]

TRAIN_STEPS = 25


def main() -> None:
    """Prepare a hermetic ProgrammaticAdapter, run real LoRA training steps, and report the loss before/after plus the saved adapter path."""
    print(f"Preparing ProgrammaticAdapter with a hermetic from-scratch base model")
    print(f"(no network access, no bundled checkpoint) over {len(DATASET)} training rows...")
    adapter = ProgrammaticAdapter(learning_rate=1e-2)
    adapter.prepare(None, DATASET)

    n_trainable = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in adapter.model.parameters())
    print(f"Base model + LoRA adapter: {n_total:,} total params, {n_trainable:,} trainable (LoRA-only).")
    print()

    loss_before = adapter.compute_loss(DATASET)
    print(f"Loss before fine-tuning: {loss_before:.4f}")

    print(f"Running {TRAIN_STEPS} real LoRA training steps (forward + backward + optimizer step)...")
    for _ in range(TRAIN_STEPS):
        result = adapter.train_step(DATASET)
    print(f"Loss at final training step ({result.step}): {result.loss:.4f}")

    loss_after = adapter.compute_loss(DATASET)
    print(f"Loss after fine-tuning:  {loss_after:.4f}")
    print()
    delta = loss_before - loss_after
    direction = "decreased" if delta > 0 else "increased"
    print(f"Loss {direction} by {abs(delta):.4f} over {TRAIN_STEPS} LoRA training steps.")

    save_dir = tempfile.mkdtemp(prefix="dscraft_tune_lora_adapter_")
    adapter.save_adapter(save_dir)
    print()
    print(f"Saved LoRA adapter weights to: {save_dir}")


if __name__ == "__main__":
    main()
