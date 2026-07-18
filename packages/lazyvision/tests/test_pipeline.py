"""Tests for `benchcraft_lazyvision.pipeline.SimpleImagePipeline`.

Hermetic: builds synthetic image bytes in-memory via Pillow (no network
access, no dataset download) and exercises decode/augment/to_dense_tensor
individually as well as the inherited `DenseMediaPipeline.run` driver.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from benchcraft_lazyvision import PipelineConfig, SimpleImagePipeline
from lazycore.data import DenseMediaPipeline


def _make_raw_image_bytes(
    size: tuple[int, int] = (16, 20), mode: str = "RGB", fmt: str = "PNG"
) -> bytes:
    """Build raw encoded image bytes for a small synthetic test image.

    Uses a non-square size and a gradient fill (not a flat color) so that
    resize and horizontal-flip behavior are actually exercised/observable,
    rather than being no-ops on a uniform image.
    """
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    # Horizontal gradient in the red channel so a flip is observable.
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[:, :, 1] = 64
    array[:, :, 2] = 128
    img = Image.fromarray(array, mode="RGB")
    if mode != "RGB":
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_pipeline_is_a_dense_media_pipeline_subclass() -> None:
    """SimpleImagePipeline must be a real subclass of
    `lazycore.data.DenseMediaPipeline`, not a parallel interface, per
    CLAUDE.md's "fix what's there, don't duplicate" rule."""
    # Per CLAUDE.md's "fix what's there, don't duplicate" rule: this must
    # be a real subclass of lazycore.data.DenseMediaPipeline, not a
    # parallel interface.
    pipeline = SimpleImagePipeline()
    assert isinstance(pipeline, DenseMediaPipeline)


def test_decode_returns_rgb_pil_image() -> None:
    """decode() must convert a non-RGB source (grayscale) to a 3-channel
    RGB PIL Image while preserving the original pixel dimensions."""
    pipeline = SimpleImagePipeline()
    raw = _make_raw_image_bytes(size=(16, 20), mode="L")  # grayscale source
    decoded = pipeline.decode(raw)
    assert isinstance(decoded, Image.Image)
    assert decoded.mode == "RGB"
    assert decoded.size == (16, 20)


def test_augment_resizes_to_configured_size() -> None:
    """augment() must resize a non-square source image to the square
    ``config.image_size`` regardless of the source's original dimensions."""
    config = PipelineConfig(image_size=24, horizontal_flip_prob=0.0)
    pipeline = SimpleImagePipeline(config)
    raw = _make_raw_image_bytes(size=(16, 20))
    decoded = pipeline.decode(raw)
    augmented = pipeline.augment(decoded)
    assert augmented.size == (24, 24)


def test_augment_flip_is_deterministic_under_seed() -> None:
    """With a fixed seed, horizontal_flip_prob=1.0 must always flip the
    image and horizontal_flip_prob=0.0 must never flip it -- augmentation
    is not left to unseeded randomness."""
    raw = _make_raw_image_bytes(size=(16, 16))

    # flip_prob=1.0 must always flip.
    always_flip = SimpleImagePipeline(
        PipelineConfig(image_size=16, horizontal_flip_prob=1.0, seed=1)
    )
    decoded = always_flip.decode(raw)
    flipped = always_flip.augment(decoded)
    expected_flip = decoded.transpose(Image.FLIP_LEFT_RIGHT)
    assert np.array_equal(np.asarray(flipped), np.asarray(expected_flip))
    assert not np.array_equal(np.asarray(flipped), np.asarray(decoded))

    # flip_prob=0.0 must never flip.
    never_flip = SimpleImagePipeline(
        PipelineConfig(image_size=16, horizontal_flip_prob=0.0, seed=1)
    )
    decoded2 = never_flip.decode(raw)
    not_flipped = never_flip.augment(decoded2)
    assert np.array_equal(np.asarray(not_flipped), np.asarray(decoded2))


def test_to_dense_tensor_shape_dtype_and_range() -> None:
    """to_dense_tensor() must produce a ``(C, H, W)`` float32 tensor with
    all values normalized into the ``[0, 1]`` range."""
    # device explicitly pinned to CPU (rather than relying on the default
    # MPS-first resolution) for hermetic, deterministic, portable
    # automated verification -- see PipelineConfig's docstring.
    config = PipelineConfig(image_size=32, horizontal_flip_prob=0.0, device="cpu")
    pipeline = SimpleImagePipeline(config)
    raw = _make_raw_image_bytes(size=(32, 32))
    decoded = pipeline.decode(raw)
    augmented = pipeline.augment(decoded)
    tensor = pipeline.to_dense_tensor(augmented)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)  # (C, H, W)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0)


def test_to_dense_tensor_satisfies_dlpack_protocol() -> None:
    """The pipeline's output must expose ``__dlpack__``/
    ``__dlpack_device__`` (torch.Tensor implements this natively), so a
    future change to to_dense_tensor's return type can't silently break
    the Tier-3 DLPack-handoff contract without this test failing."""
    # lazycore.data.DenseMediaPipeline.to_dense_tensor's contract requires
    # the return value to support the DLPack protocol. torch.Tensor
    # implements this natively; assert it explicitly so a future change to
    # to_dense_tensor's return type can't silently break the Tier-3
    # contract without a test failing.
    pipeline = SimpleImagePipeline(PipelineConfig(device="cpu"))
    raw = _make_raw_image_bytes()
    tensor = pipeline.run(raw)
    assert hasattr(tensor, "__dlpack__")
    assert hasattr(tensor, "__dlpack_device__")
    capsule = tensor.__dlpack__()
    assert capsule is not None


def test_run_driver_matches_manual_composition() -> None:
    """The inherited `DenseMediaPipeline.run` driver must produce the exact
    same tensor as manually chaining decode -> augment -> to_dense_tensor,
    for a fresh pipeline instance with matching config/seed."""
    config = PipelineConfig(image_size=28, horizontal_flip_prob=0.0, seed=42, device="cpu")
    pipeline = SimpleImagePipeline(config)
    raw = _make_raw_image_bytes(size=(28, 28))

    manual = pipeline.to_dense_tensor(pipeline.augment(pipeline.decode(raw)))

    # A second pipeline instance with the same config/seed and the same
    # deterministic (no-flip) augmentation should match `run`'s output
    # exactly.
    pipeline2 = SimpleImagePipeline(config)
    via_run = pipeline2.run(raw)

    assert torch.equal(manual, via_run)


def test_decode_rejects_garbage_bytes() -> None:
    """decode() must raise rather than silently return a bogus image when
    handed bytes that are not a valid encoded image.

    Narrowed to `PIL.UnidentifiedImageError` (rather than the broad
    `Exception`) since that is the specific exception `PIL.Image.open`
    raises for undecodable bytes -- a bare `Exception` would also pass if
    an unrelated `decode()` regression raised some other error, hiding a
    real bug behind a spuriously-green test."""
    pipeline = SimpleImagePipeline()
    with pytest.raises(UnidentifiedImageError):
        pipeline.decode(b"not an image")
