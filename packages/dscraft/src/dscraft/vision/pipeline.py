"""``SimpleImagePipeline``: the first concrete Tier-3 ``DenseMediaPipeline``.

Per the architecture doc §2.1 (Tier 3: dense image/audio) and
`dscraft.core.data.DenseMediaPipeline`'s own docstring: dscraft.core defines only
the *shape* of the decode -> augment -> to-dense-tensor pipeline and
explicitly depends on nothing image/tensor-related; LazyVision is expected
to provide the first concrete decode/augment implementation. This module is
that implementation -- it subclasses `dscraft.core.data.DenseMediaPipeline`
directly rather than redefining a parallel interface, per CLAUDE.md's
"fix what's there, don't duplicate" rule.

Scope for this pass is deliberately narrow (a "signature capability" slice,
not the full Tier-3 pipeline the architecture doc envisions):

- ``decode`` uses Pillow to turn raw encoded image bytes (PNG/JPEG/...)
  into a PIL ``Image`` -- not the eventual Rust/PyO3 native decoder
  described in the architecture doc (explicitly deferred; see README).
- ``augment`` applies one real, simple augmentation (a seeded random
  horizontal flip) plus a deterministic resize -- not the full FFCV-style
  augmentation surface, SAM/LLRD training-time regularization, or
  hybrid-local-convolutional spatial-locality modules described elsewhere
  in the architecture doc (all deferred).
- ``to_dense_tensor`` converts to a `torch.Tensor` in ``(C, H, W)`` layout,
  float32, normalized to ``[0, 1]``. PyTorch tensors implement
  ``__dlpack__``/``__dlpack_device__`` natively, so this return value
  already satisfies `dscraft.core.data`'s ``_SupportsDLPack`` protocol without
  this module needing to do anything special -- the DLPack handoff the
  architecture doc describes ("zero-copy handoff only at the final
  dense-tensor stage") is simply "return a torch.Tensor" here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from dscraft.core.data import DenseMediaPipeline

from dscraft.vision.model import resolve_device

__all__ = ["SimpleImagePipeline", "PipelineConfig"]


@dataclass(frozen=True)
class PipelineConfig:
    """Shape/behavior knobs for :class:`SimpleImagePipeline`.

    Attributes:
        image_size: Output height/width (square) after resize.
        horizontal_flip_prob: Probability of a random horizontal flip
            during :meth:`SimpleImagePipeline.augment`. Set to ``0.0`` for
            a deterministic no-op augmentation.
        seed: Seed for the pipeline's own random generator, so augmentation
            is reproducible across runs -- useful for hermetic tests.
        device: Where :meth:`SimpleImagePipeline.to_dense_tensor` places
            the resulting tensor. Defaults to ``None``, which resolves via
            `dscraft.vision.model.resolve_device` -- this package's
            one canonical device-selection helper (MPS -> CUDA -> CPU,
            mirroring `dscraft.graph.gcn.resolve_device`) -- so a
            caller who does not specify a device gets MPS-first rather
            than an implicit CPU default. Pass ``"cpu"`` explicitly for
            hermetic/deterministic tests.
    """

    image_size: int = 32
    horizontal_flip_prob: float = 0.5
    seed: int = 0
    device: str | torch.device | None = None


class SimpleImagePipeline(DenseMediaPipeline):
    """Concrete Tier-3 pipeline: decode (Pillow) -> augment (flip+resize)
    -> to_dense_tensor (torch.Tensor).

    This is the **one canonical** preprocessing pipeline in this package --
    there is no second/parallel decode+augment implementation elsewhere in
    this codebase.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Create the pipeline and seed its private augmentation RNG.

        Args:
            config: Behavior knobs (image size, flip probability, seed).
                Defaults to ``PipelineConfig()``. The RNG used by
                :meth:`augment` is seeded from ``config.seed`` here, once,
                so repeated calls to :meth:`augment`/:meth:`run` on the same
                pipeline instance are reproducible but not identical (the
                RNG's internal state advances across calls).
        """
        self.config = config or PipelineConfig()
        self._rng = np.random.default_rng(self.config.seed)
        preferred = (
            str(self.config.device) if self.config.device is not None else None
        )
        self._device = resolve_device(preferred)

    def decode(self, raw: bytes) -> Image.Image:
        """Decode raw encoded image bytes (PNG/JPEG/...) into a PIL Image.

        Converts to RGB unconditionally so downstream stages always see a
        3-channel image regardless of the source encoding (e.g. a
        grayscale PNG or an RGBA PNG with an alpha channel).
        """
        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("RGB")

    def augment(self, decoded: Image.Image) -> Image.Image:
        """Resize to ``config.image_size`` and, with probability
        ``config.horizontal_flip_prob``, apply a horizontal flip.

        This is a real, if simple, augmentation -- not a stub -- matching
        the task's "trivial augmentation... keep it simple and real, not a
        stub" guidance. The resize step also anchors every sample to a
        fixed, ONNX-export-friendly static shape.
        """
        resized = decoded.resize(
            (self.config.image_size, self.config.image_size),
            resample=Image.BILINEAR,
        )
        if self._rng.random() < self.config.horizontal_flip_prob:
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
        return resized

    def to_dense_tensor(self, augmented: Image.Image) -> torch.Tensor:
        """Convert a PIL Image to a ``(C, H, W)`` float32 tensor in [0, 1].

        The returned `torch.Tensor` implements ``__dlpack__`` /
        ``__dlpack_device__`` natively, satisfying
        `dscraft.core.data`'s ``_SupportsDLPack`` protocol -- no extra
        conversion step is needed to comply with the Tier-3 contract. The
        tensor is placed on ``self._device`` (resolved from
        ``config.device`` via :func:`resolve_device` in ``__init__`` --
        MPS-first by default, not a hardcoded CPU placement).
        """
        array = np.asarray(augmented, dtype=np.float32) / 255.0  # (H, W, C)
        chw = np.transpose(array, (2, 0, 1)).copy()  # (C, H, W), contiguous
        return torch.from_numpy(chw).to(self._device)
