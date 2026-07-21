"""Tests for `dscraft.vision.ocr`.

Hermetic: renders a small synthetic image containing real text via
`PIL.ImageDraw`/`PIL.ImageFont` in-memory (no network access, no bundled
test asset needed -- matching this package's existing "generate synthetic
test data in-memory" discipline, e.g. `test_pipeline.py`'s gradient PNGs
and `test_real_dataset_validation.py`'s bundled-`sklearn`-dataset PNGs).

The EasyOCR-backend test runs unconditionally (pure pip dependency, already
required by the `vision` extra per issue #11). The Tesseract-backend test
is skipped (not failed) via `pytest.skip(...)` if the system `tesseract`
binary genuinely isn't installed on the machine running the tests -- see
`dscraft.vision.ocr._run_tesseract`'s up-front `shutil.which` check, which
this test reuses to decide skip-vs-run.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from dscraft.vision import (
    OCRDetection,
    OCRResult,
    SUPPORTED_OCR_BACKENDS,
    TesseractNotInstalledError,
    run_ocr,
)

TEXT = "HELLO"


def _make_text_image(text: str = TEXT, size: tuple[int, int] = (240, 80)) -> Image.Image:
    """Render a real, synthetic image containing rendered text.

    White background, large black text -- high-contrast and large enough
    for both EasyOCR and Tesseract to reliably recognize without any
    dataset download or bundled asset, per the task's "no network, no
    bundled asset needed" guidance. Uses a truetype font if one is
    discoverable on this system, else falls back to Pillow's built-in
    default bitmap font -- either renders real, recognizable glyphs.
    """
    image = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 48)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 48
            )
        except OSError:
            # Pillow's built-in bitmap font is fixed-size (~11px) on Pillow
            # <10.1; passing `size=` requires Pillow>=10.1 (see the `vision`
            # extra's floor in pyproject.toml) to get a legible, scaled
            # fallback glyph rather than a tiny bitmap font that neither
            # backend can reliably recognize.
            font = ImageFont.load_default(size=48)
    draw.text((10, 10), text, fill="black", font=font)
    return image


def test_supported_ocr_backends_allowlist() -> None:
    """The allowlist advertises exactly the two backends issue #11 requires."""
    assert set(SUPPORTED_OCR_BACKENDS) == {"easyocr", "tesseract"}


def test_run_ocr_rejects_unknown_backend() -> None:
    image = _make_text_image()
    with pytest.raises(ValueError, match="Unsupported OCR backend"):
        run_ocr(image, backend="not-a-real-backend")


def test_run_ocr_accepts_pil_numpy_and_path(tmp_path, monkeypatch) -> None:
    """`run_ocr`'s image argument accepts the same shapes
    `SimpleImagePipeline`'s convention implies: a decoded PIL Image, a
    numpy array, or a file path -- not a single hard-coded input type.

    This is purely an input-shape acceptance test for `_to_pil_image`, not
    a recognition-quality test (that's `test_easyocr_backend_extracts_
    recognizable_text` below) -- so `easyocr.Reader` is mocked here to
    avoid 4 redundant real model-load/inference calls per test run.
    """
    import easyocr

    class _FakeReader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def readtext(self, array, detail=1):
            return [([[0, 0], [10, 0], [10, 10], [0, 10]], TEXT, 0.99)]

    monkeypatch.setattr(easyocr, "Reader", _FakeReader)

    image = _make_text_image()
    array = np.asarray(image)
    file_path = tmp_path / "text.png"
    image.save(file_path)

    for candidate in (image, array, file_path, str(file_path)):
        result = run_ocr(candidate, backend="easyocr")
        assert isinstance(result, OCRResult)
        assert result.text == TEXT


def test_easyocr_backend_extracts_recognizable_text() -> None:
    """EasyOCR is a pure pip dependency (already required by the `vision`
    extra) -- this test runs unconditionally, no skip condition."""
    image = _make_text_image()

    result = run_ocr(image, backend="easyocr")

    assert isinstance(result, OCRResult)
    assert result.backend == "easyocr"
    assert TEXT.lower() in result.text.lower().replace(" ", "")
    assert isinstance(result.detections, list)
    assert len(result.detections) >= 1
    for detection in result.detections:
        assert isinstance(detection, OCRDetection)
        assert isinstance(detection.text, str)
        assert 0.0 <= detection.confidence <= 1.0
        left, top, right, bottom = detection.bbox
        assert right > left
        assert bottom > top


def test_tesseract_backend_extracts_recognizable_text() -> None:
    """Skip (don't fail) if the system `tesseract` binary isn't installed --
    it is a separate, non-pip dependency (see pyproject.toml's `vision`
    extra comment) that may genuinely be absent on the machine running
    these tests."""
    if shutil.which("tesseract") is None:
        pytest.skip(
            "System `tesseract` binary not found on PATH; install it "
            "separately (e.g. `brew install tesseract` on macOS) to run "
            "this test."
        )

    image = _make_text_image()

    result = run_ocr(image, backend="tesseract")

    assert isinstance(result, OCRResult)
    assert result.backend == "tesseract"
    assert TEXT.lower() in result.text.lower().replace(" ", "")
    assert isinstance(result.detections, list)
    assert len(result.detections) >= 1
    for detection in result.detections:
        assert isinstance(detection, OCRDetection)
        assert isinstance(detection.text, str)
        assert 0.0 <= detection.confidence <= 1.0
        left, top, right, bottom = detection.bbox
        assert right > left
        assert bottom > top


def test_tesseract_backend_raises_clear_error_when_binary_missing(monkeypatch) -> None:
    """Simulate the missing-binary case regardless of whether `tesseract`
    is actually installed on the machine running this test, so this
    behavior is always verified (not conditionally skipped like the
    happy-path test above)."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    image = _make_text_image()

    with pytest.raises(TesseractNotInstalledError, match="brew install tesseract"):
        run_ocr(image, backend="tesseract")
