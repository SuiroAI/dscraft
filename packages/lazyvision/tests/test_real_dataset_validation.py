"""Real-dataset validation: same pipeline/export path, real image data.

The rest of this package's test suite (`test_pipeline.py`, `test_export.py`)
exercises `SimpleImagePipeline` / `TinyCNN` / `export_to_onnx` /
`verify_export` exclusively against synthetic, in-memory-generated data
(gradient-pattern PNGs, random tensors). That proves the mechanical
decode -> augment -> to_dense_tensor -> export -> verify machinery works,
but says nothing about behavior on an actual real-world image.

This module closes that gap using `sklearn.datasets.load_digits()`: a real,
small (8x8 grayscale) dataset of genuine handwritten digit images (a
downsampled version of the UCI Optical Recognition of Handwritten Digits
dataset). It ships as package data *inside* the `scikit-learn` wheel/sdist
(under `sklearn/datasets/data/`) and is loaded from local disk by
`load_digits()` -- no network access, no external download -- which is why
it was chosen over fetching an image dataset from the internet, per the
stakeholder's explicit preference for a dependency-bundled real dataset.

`scikit-learn` is a **dev/test-only** dependency of this package (see
`pyproject.toml`'s `dev` extra) -- it plays no role in
`benchcraft_lazyvision`'s actual runtime logic -- the same "add a
validation-only dependency without touching core runtime deps" pattern.

Bridging real data into the real API surface:

`SimpleImagePipeline.decode()` (see `pipeline.py`) accepts raw *encoded*
image bytes (``PIL.Image.open(io.BytesIO(raw))`` -- i.e. actual PNG/JPEG/...
file bytes), not a raw pixel array. `load_digits()` instead returns each
digit as a bare ``(8, 8)`` float array with values in ``[0, 16]``. To
genuinely exercise `decode()` on real image-shaped data (rather than
bypassing it and jumping straight to `to_dense_tensor`, which the task
explicitly calls out as not real validation), this module does the same
bridging step a real caller would have to do: rescale the ``[0, 16]`` pixel
values to ``[0, 255]`` uint8, build a real grayscale `PIL.Image` from that
array, and PNG-encode it to bytes via `Image.save(buf, format="PNG")`. Those
PNG bytes are then handed to `SimpleImagePipeline.run()` exactly like any
other raw image payload -- the same `decode()` entry point, with no
special-casing for "the real-data case."
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from benchcraft_lazyvision import (
    ExportResult,
    ModelConfig,
    PipelineConfig,
    SimpleImagePipeline,
    build_model,
    export_to_onnx,
    verify_export,
)

IMAGE_SIZE = 32
NUM_CLASSES = 10  # load_digits() has 10 classes (digits 0-9), same as TinyCNN's default.


def _load_real_digit_png_bytes(index: int = 0) -> tuple[bytes, int]:
    """Load one real handwritten-digit image from sklearn's bundled dataset
    and encode it as real PNG bytes, i.e. exactly the raw-bytes input shape
    `SimpleImagePipeline.decode()` expects.

    Returns:
        ``(png_bytes, label)`` where ``label`` is the digit's true class
        (0-9), included for context/debugging even though this test does
        not assert on classification accuracy (the untrained `TinyCNN` is
        not expected to classify correctly -- see `model.py`'s docstring).
    """
    from sklearn.datasets import load_digits  # local import: dev-only dependency

    digits = load_digits()
    image_0_16 = digits.images[index]  # (8, 8) float64, values in [0, 16]
    label = int(digits.target[index])

    # Real bridging step: rescale real pixel values to a real uint8 image,
    # matching what any caller handing `load_digits()` data to an
    # image-encoding library would have to do -- not a synthetic stand-in.
    image_uint8 = np.clip(image_0_16 / 16.0 * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L")  # real 8x8 grayscale image

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")  # real PNG encoding, exercised by decode()
    return buf.getvalue(), label


def test_load_digits_is_bundled_and_real() -> None:
    """Sanity check on the data source itself: real images, real pixel
    range, no network access required (load_digits() reads a bundled .csv
    from the installed sklearn package's own data directory)."""
    from sklearn.datasets import load_digits

    digits = load_digits()
    assert digits.images.shape[1:] == (8, 8)
    assert digits.images.min() >= 0.0
    assert digits.images.max() <= 16.0
    # 1797 samples is load_digits()'s well-known fixed dataset size.
    assert digits.images.shape[0] == 1797


def test_real_digit_bytes_are_a_genuine_decodable_png() -> None:
    """The bridging step must produce bytes that `Image.open` (i.e. what
    `SimpleImagePipeline.decode()` calls) can actually decode as an image
    -- not a fake/placeholder payload."""
    png_bytes, label = _load_real_digit_png_bytes(index=0)
    assert isinstance(label, int)
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.size == (8, 8)


def test_real_digit_runs_through_the_full_pipeline() -> None:
    """The real digit image must flow through the exact same
    decode -> augment -> to_dense_tensor pipeline the synthetic tests use,
    with no shortcut around `decode()`."""
    png_bytes, _label = _load_real_digit_png_bytes(index=0)

    config = PipelineConfig(
        image_size=IMAGE_SIZE, horizontal_flip_prob=0.0, seed=0, device="cpu"
    )
    pipeline = SimpleImagePipeline(config)

    dense_tensor = pipeline.run(png_bytes)  # decode() really parses the PNG bytes above.

    assert isinstance(dense_tensor, torch.Tensor)
    assert dense_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)  # decode() converts to RGB
    assert dense_tensor.dtype == torch.float32
    assert torch.all(dense_tensor >= 0.0) and torch.all(dense_tensor <= 1.0)
    # A real digit stroke is not a blank image -- some signal must survive
    # decode -> resize.
    assert dense_tensor.max() > 0.0


@pytest.mark.parametrize("digit_index", [0, 1, 42, 100])
def test_real_digit_export_matches_pytorch_output(digit_index: int) -> None:
    """The full, unmodified pipeline -> TinyCNN -> export_to_onnx ->
    verify_export flow, run on a real handwritten-digit image instead of a
    random synthetic tensor. This is the core acceptance check: the ONNX
    export correctness guarantee must hold on real image data, not just on
    random tensors, using the exact same public API as the synthetic tests
    (no forked/parallel pipeline or export path for "the real-data case").
    """
    png_bytes, label = _load_real_digit_png_bytes(index=digit_index)

    pipeline_config = PipelineConfig(
        image_size=IMAGE_SIZE, horizontal_flip_prob=0.0, seed=digit_index, device="cpu"
    )
    pipeline = SimpleImagePipeline(pipeline_config)
    dense_tensor = pipeline.run(png_bytes)

    model_config = ModelConfig(
        in_channels=dense_tensor.shape[0], image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )
    model = build_model(model_config, seed=0, device="cpu")

    # torch.export's dynamic batch dimension requires a tracing batch size
    # >= 2 (see export_to_onnx's docstring) -- build a real batch out of the
    # same real digit image, repeated, rather than padding with synthetic
    # filler that would dilute what's being verified.
    trace_batch = dense_tensor.unsqueeze(0).repeat(2, 1, 1, 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / f"tiny_cnn_digit_{digit_index}.onnx"
        export_to_onnx(model, trace_batch, onnx_path)

        # Verify on the real digit image itself (batch size 1 is fine here
        # since export_to_onnx already marked the batch dim dynamic).
        verification_batch = dense_tensor.unsqueeze(0)
        result = verify_export(model, onnx_path, verification_batch, atol=1e-4, rtol=1e-3)

        assert isinstance(result, ExportResult)
        assert result.matched
        assert result.max_abs_diff < 1e-3
        assert 0 <= label <= 9
