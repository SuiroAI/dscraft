"""A small CNN image classifier (architecture doc Part 3, "Module 5: LazyVision").

Scope for this scaffold-depth pass is deliberately narrow: a tiny,
LeNet-style CNN (a few conv layers, one linear head), used purely to prove
the `torch.export` -> ONNX export path in :mod:`benchcraft_lazyvision.export`
works end-to-end and correctly. Training a real model on real data is
explicitly **not** required at this scope -- the point is validating the
export mechanism, not model accuracy -- so :func:`build_model` returns an
initialized-but-untrained model. Vision Transformers, real-time object
detectors (YOLO/D-FINE/RT-DETR), and acoustic/spectrogram models from the
same architecture-doc section are out of scope for this pass; see the
package README's "Deferred" section.

Per CLAUDE.md's MPS-primary constraint: nothing here hardcodes
``device="cpu"`` in a way that would prevent later running on MPS --
:func:`build_model` accepts an arbitrary ``torch.device``/device string and
the model itself is plain `nn.Module` layers with no CUDA-only or
CPU-only assumptions.

**Device resolution.** :func:`resolve_device` is the one canonical
device-selection helper in this package (mirroring
`benchcraft_lazygraph.gcn.resolve_device`'s exact MPS -> CUDA -> CPU
auto-detect-with-fallback pattern, per CLAUDE.md's "one canonical location
per capability" rule and its own "MPS is the primary backend" constraint).
:func:`build_model`, :func:`synthetic_classification_batch`, and
`benchcraft_lazyvision.pipeline.SimpleImagePipeline` all default their
``device`` to ``None``, which routes through this same helper -- so a
caller who does not specify a device gets MPS-first-with-CPU-fallback,
not a hardcoded CPU default. This module's own *tests* and the *example
script* still explicitly pin ``device="cpu"`` for hermetic, portable,
deterministic automated verification -- not because MPS is unsupported,
but because CPU-for-tests was an intentional prior design choice (see the
package README's "MPS note") that this change preserves rather than
overrides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

_logger = logging.getLogger(__name__)

__all__ = [
    "TinyCNN",
    "ModelConfig",
    "build_model",
    "synthetic_classification_batch",
    "resolve_device",
]


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick a compute device, defaulting to MPS with a clean fallback.

    This is the one canonical device-resolution helper in this package --
    mirrors `benchcraft_lazygraph.gcn.resolve_device` exactly, for
    consistency across Benchcraft modules. Per CLAUDE.md: MPS is the
    primary backend for this platform, so this picks MPS if available,
    else CUDA if available, else CPU -- but always falls back cleanly
    rather than raising if a caller passes an unavailable ``preferred``
    device.
    """
    if preferred is not None:
        try:
            device = torch.device(preferred)
            # Smoke-test the device is actually usable.
            torch.zeros(1, device=device)
            return device
        except (RuntimeError, AssertionError) as exc:
            # RuntimeError: malformed device string (e.g. torch.device("not-a-device"))
            # or an unavailable backend at allocation time. AssertionError: PyTorch's
            # own "not compiled with <backend> enabled" check (e.g. CUDA on a
            # CPU/MPS-only build). Both are the expected "this device string is not
            # usable on this machine" case, so fall through to auto-detection. Any
            # other exception type is unexpected and should propagate rather than be
            # silently swallowed.
            _logger.warning(
                "Preferred device %r is not usable (%s: %s); falling back to "
                "auto-detected device.",
                preferred,
                type(exc).__name__,
                exc,
            )

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class ModelConfig:
    """Shape configuration shared between :class:`TinyCNN` and its callers.

    Kept as a small dataclass (rather than scattering magic numbers across
    ``model.py``, ``export.py``, and the tests/example) so the CNN's input
    contract has exactly one definition.
    """

    in_channels: int = 3
    image_size: int = 32
    num_classes: int = 10


class TinyCNN(nn.Module):
    """A minimal LeNet-style CNN: two conv+pool blocks, one linear head.

    Deliberately small (a "few conv layers", per the task's scope) --
    this is a scaffold proving the export path, not a competitive image
    classifier. Input is expected to be ``(N, in_channels, image_size,
    image_size)`` float32, e.g. the dense tensor produced by
    :class:`benchcraft_lazyvision.pipeline.SimpleImagePipeline`.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Build the conv/pool/linear layers for the given ``config``.

        Args:
            config: Shape configuration (input channels, image size, number
                of output classes). Defaults to ``ModelConfig()``.

        Raises:
            ValueError: ``config.image_size`` is too small for two 2x2
                max-pools to leave at least a 1x1 feature map (i.e. less
                than 4).
        """
        super().__init__()
        self.config = config or ModelConfig()

        self.conv1 = nn.Conv2d(self.config.in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two 2x2 max-pools halve the spatial size twice.
        reduced = self.config.image_size // 4
        if reduced < 1:
            raise ValueError(
                "ModelConfig.image_size must be >= 4 so that two 2x2 "
                f"max-pools leave at least a 1x1 feature map; got "
                f"{self.config.image_size}."
            )
        self.fc = nn.Linear(16 * reduced * reduced, self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two conv+relu+pool blocks and the linear head.

        Args:
            x: Input batch of shape ``(N, in_channels, image_size,
                image_size)``, float32.

        Returns:
            Unnormalized class logits of shape ``(N, num_classes)``.
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def build_model(
    config: ModelConfig | None = None,
    *,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> TinyCNN:
    """Construct a :class:`TinyCNN` with deterministic (seeded) init weights.

    Not trained -- per this pass's scope, an initialized-but-untrained model
    is sufficient to validate the export path (see module docstring).

    Args:
        config: Optional :class:`ModelConfig`. Defaults to ``ModelConfig()``
            (3x32x32 input, 10 classes).
        seed: Seed for ``torch.manual_seed`` so weight init is reproducible
            across test runs.
        device: Where to place the model. Defaults to ``None``, which
            resolves via :func:`resolve_device` -- MPS first (this
            platform's primary backend per CLAUDE.md), then CUDA, then CPU.
            Callers may pass an explicit device/string (e.g. ``"cpu"``,
            ``"mps"``) to override; this module's own tests explicitly pass
            ``device="cpu"`` for hermetic, deterministic verification.

    Returns:
        A :class:`TinyCNN` in ``eval()`` mode on ``device``.
    """
    resolved_device = resolve_device(str(device) if device is not None else None)
    generator_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = TinyCNN(config)
    finally:
        torch.set_rng_state(generator_state)
    return model.to(resolved_device).eval()


def synthetic_classification_batch(
    config: ModelConfig | None = None,
    *,
    batch_size: int = 4,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small, fully-synthetic (images, labels) batch.

    Stands in for a real image-classification dataset so that tests and the
    example script stay hermetic and fast (no network access, no dataset
    download), per the task's "prefer whichever keeps tests hermetic/fast"
    guidance. Images are random float32 tensors in ``[0, 1)``; labels are
    random integers in ``[0, num_classes)``. This is not meant to represent
    a learnable task -- it is purely input/output-shape-compatible stand-in
    data for exercising the pipeline and export path.

    Args:
        config: Optional :class:`ModelConfig`.
        batch_size: Number of samples in the batch.
        seed: Seed for the batch's own CPU generator (kept on CPU
            regardless of ``device`` -- this only affects reproducibility
            of the random values, not where the resulting tensors live).
        device: Where to place the returned tensors. Defaults to ``None``,
            which resolves via :func:`resolve_device` (MPS -> CUDA -> CPU).

    Returns:
        ``(images, labels)`` where ``images`` has shape
        ``(batch_size, in_channels, image_size, image_size)`` and ``labels``
        has shape ``(batch_size,)``.
    """
    cfg = config or ModelConfig()
    resolved_device = resolve_device(str(device) if device is not None else None)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.rand(
        (batch_size, cfg.in_channels, cfg.image_size, cfg.image_size),
        generator=generator,
        dtype=torch.float32,
    ).to(resolved_device)
    labels = torch.randint(
        low=0,
        high=cfg.num_classes,
        size=(batch_size,),
        generator=generator,
    ).to(resolved_device)
    return images, labels
