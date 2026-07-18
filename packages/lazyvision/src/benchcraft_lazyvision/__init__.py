"""benchcraft-lazyvision: Benchcraft's computer-vision module.

This scaffold-depth pass implements exactly one signature capability from
the architecture doc (Part 3, "Module 5: LazyVision", §2.1 Tier 3, §2.5
export backend 1): a small CNN image classifier, captured via
`torch.export()` and exported to ONNX, plus the first concrete
`lazycore.data.DenseMediaPipeline` subclass handling decode+augment+
to-dense-tensor preprocessing.

Vision Transformers, real-time object detectors (YOLO/D-FINE/RT-DETR),
acoustic/spectrogram models, the Rust/PyO3 data-loading layer,
Sharpness-Aware Minimization/Layer-wise LR Decay, and the AGPL-detector
subprocess-isolation plugin architecture from the same architecture-doc
section are explicitly out of scope for this pass -- future work, not
partially stubbed out here. See the package README's "Deferred" section.

Public API surface (this package's one canonical pipeline and one
canonical export path -- no parallel implementations exist elsewhere in
this codebase):

    >>> from benchcraft_lazyvision import (
    ...     SimpleImagePipeline,
    ...     PipelineConfig,
    ...     TinyCNN,
    ...     ModelConfig,
    ...     build_model,
    ...     synthetic_classification_batch,
    ...     export_to_onnx,
    ...     verify_export,
    ...     ExportResult,
    ...     resolve_device,
    ... )
"""

from benchcraft_lazyvision.export import ExportResult, export_to_onnx, verify_export
from benchcraft_lazyvision.model import (
    ModelConfig,
    TinyCNN,
    build_model,
    resolve_device,
    synthetic_classification_batch,
)
from benchcraft_lazyvision.pipeline import PipelineConfig, SimpleImagePipeline

__all__ = [
    "SimpleImagePipeline",
    "PipelineConfig",
    "TinyCNN",
    "ModelConfig",
    "build_model",
    "synthetic_classification_batch",
    "export_to_onnx",
    "verify_export",
    "ExportResult",
    "resolve_device",
]

__version__ = "0.1.0"
