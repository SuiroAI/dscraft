"""dscraft-vision: DSCraft's computer-vision module.

This scaffold-depth pass implements exactly one signature capability from
the architecture doc (Part 3, "Module 5: LazyVision", §2.1 Tier 3, §2.5
export backend 1): a small CNN image classifier, captured via
`torch.export()` and exported to ONNX, plus the first concrete
`dscraft.core.data.DenseMediaPipeline` subclass handling decode+augment+
to-dense-tensor preprocessing.

Vision Transformers, real-time object detectors (YOLO/D-FINE/RT-DETR),
acoustic/spectrogram models, the Rust/PyO3 data-loading layer,
Sharpness-Aware Minimization/Layer-wise LR Decay, and the AGPL-detector
subprocess-isolation plugin architecture from the same architecture-doc
section are explicitly out of scope for this pass -- future work, not
partially stubbed out here. See the package README's "Deferred" section.

A later pass (issue #11) added OCR as a second, independent capability:
:func:`run_ocr` dispatches to a selectable backend (EasyOCR or Tesseract,
per the multi-backend design principle -- see `dscraft.vision.ocr`'s module
docstring) rather than reusing/extending the CNN/export pipeline above; it
does not depend on `SimpleImagePipeline`/`TinyCNN`/the export path, and
they do not depend on it.

Public API surface (this package's one canonical dense-image pipeline, one
canonical export path, and one canonical OCR entry point -- no parallel
implementations exist elsewhere in this codebase):

    >>> from dscraft.vision import (
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
    ...     run_ocr,
    ...     OCRResult,
    ...     OCRDetection,
    ...     SUPPORTED_OCR_BACKENDS,
    ...     TesseractNotInstalledError,
    ... )
"""

from dscraft.vision.export import ExportResult, export_to_onnx, verify_export
from dscraft.vision.model import (
    ModelConfig,
    TinyCNN,
    build_model,
    resolve_device,
    synthetic_classification_batch,
)
from dscraft.vision.ocr import (
    OCRDetection,
    OCRResult,
    SUPPORTED_OCR_BACKENDS,
    TesseractNotInstalledError,
    run_ocr,
)
from dscraft.vision.pipeline import PipelineConfig, SimpleImagePipeline

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
    "run_ocr",
    "OCRResult",
    "OCRDetection",
    "SUPPORTED_OCR_BACKENDS",
    "TesseractNotInstalledError",
]

__version__ = "0.1.0"
