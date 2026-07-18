"""benchcraft_lazytune -- LazyTune scaffold: Adapter-Factory LoRA fine-tuning.

Public API surface for the one signature capability implemented at this
scaffold depth (architecture doc Part 3, "Module 6: LazyTune"): the
Adapter-Factory pattern's ``BaseTrainingAdapter`` interface, plus one
concrete in-process ``ProgrammaticAdapter`` implementation that performs a
real (tiny) LoRA fine-tuning step via the standalone `peft` + `transformers`
libraries.

Everything else described for LazyTune in the architecture doc -- the
subprocess-isolated ``SubprocessAdapter`` family (torchtune/Axolotl via
`torchrun`), the multi-fidelity BOHB micro-tuning system, Multi-Power-Law
fitting, KL-penalization/reward-shaping for RL, and real GGUF/MLX export
conversion -- is out of scope for this pass. See README.
"""

from __future__ import annotations

from .adapter import (
    MODEL_ALLOWLIST,
    RECOMMENDED_BASE_MODEL_NAME,
    BaseTrainingAdapter,
    ProgrammaticAdapter,
    TinyTokenizer,
    TrainStepResult,
    build_hermetic_causal_lm,
    default_lora_config,
)
from .export import export_gguf_stub, export_mlx_stub

__all__ = [
    "MODEL_ALLOWLIST",
    "RECOMMENDED_BASE_MODEL_NAME",
    "BaseTrainingAdapter",
    "ProgrammaticAdapter",
    "TinyTokenizer",
    "TrainStepResult",
    "build_hermetic_causal_lm",
    "default_lora_config",
    "export_gguf_stub",
    "export_mlx_stub",
]
