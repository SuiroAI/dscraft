"""OCR as an optional, pluggable capability inside `dscraft.vision`.

Per the architecture doc's multi-backend design principle (never pick a
single winner between competing frameworks with genuinely different
tradeoffs) and issue #11 (scoped from a machine-wide DS/ML tooling audit
citing two real local evidence projects -- ``mcp-screen-recorder-docs`` and
``public-tasks`` -- that both already use EasyOCR/pytesseract/OpenCV in
practice): this module exposes **two** selectable OCR backends rather than
hard-coding one.

- ``"easyocr"``: PyTorch-based, MPS/GPU-capable, no external system binary
  to install. Apache-2.0.
- ``"tesseract"``: a thin wrapper (`pytesseract`) around the system
  ``tesseract`` binary -- CPU-only, much smaller Python-side footprint, but
  requires that binary to be installed separately (it is *not* a pip
  package and cannot be declared in ``pyproject.toml``; see that file's
  ``vision`` extra for the install-time note). Also Apache-2.0 (both the
  wrapper and Tesseract itself), so -- per the architecture doc's
  LazyIsolate policy -- both backends are Tier 1 permissive and need no
  license-isolation gating (unlike, e.g., LazyVision's AGPL YOLO detector
  plugin, which does).

Design: an allowlist-dispatch pattern mirroring
`dscraft.forecast.forecast.SUPPORTED_MODELS`'s exact shape (a public
name -> implementation mapping, validated up front, with one canonical
entry point that dispatches into one private per-backend function). Each
backend's dependency (`easyocr` / `pytesseract`) is imported lazily *inside
its own private function only* -- never at module level -- following the
same lazy-import discipline as `dscraft.automl.compile._require_onnx_stack`.
This keeps ``dscraft.vision``'s base import surface unaffected by whichever
OCR backend(s) a given environment does or doesn't have installed.

**Image input convention.** Matches `dscraft.vision.pipeline`'s own
convention as closely as fits: :func:`run_ocr` accepts a `PIL.Image.Image`
directly (the type `SimpleImagePipeline.decode()` already produces), so a
caller can feed the exact same decoded image object into both the existing
dense-tensor pipeline and OCR without re-decoding. It also accepts a
`numpy.ndarray` (both backends' native input shape) and a file path
(`str`/`pathlib.Path`) for convenience, since OCR is commonly invoked
directly against a file rather than through the augmentation pipeline (OCR
has no use for :meth:`SimpleImagePipeline.augment`'s resize/flip step --
resizing before recognition can *hurt* text legibility, and a fixed square
aspect ratio is actively wrong for most document/screenshot inputs -- so
this module deliberately does not reuse the ``augment`` stage, only the
"give me a decoded image" idea from ``decode``).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

__all__ = [
    "OCRDetection",
    "OCRResult",
    "SUPPORTED_OCR_BACKENDS",
    "TesseractNotInstalledError",
    "run_ocr",
]

#: Backend names this module supports, mapped to a short human-readable
#: description -- mirrors `dscraft.forecast.forecast.SUPPORTED_MODELS`'s
#: exact "public allowlist dict, validated up front" shape. Values are
#: descriptive only (unlike SUPPORTED_MODELS, which maps to constructible
#: classes) since each backend's actual entry point is a private
#: module-level function, not a class to instantiate.
SUPPORTED_OCR_BACKENDS: dict[str, str] = {
    "easyocr": "PyTorch-based OCR (MPS/GPU-capable, no external binary).",
    "tesseract": "pytesseract wrapper around the system `tesseract` binary "
    "(CPU-only, requires separate binary install).",
}


class TesseractNotInstalledError(RuntimeError):
    """Raised when the ``tesseract`` backend is selected but the system
    ``tesseract`` binary is not installed/discoverable.

    A dedicated exception type (rather than letting `pytesseract` raise its
    own `pytesseract.TesseractNotFoundError`, or letting a bare
    `FileNotFoundError` propagate) so callers get one clear, actionable
    error message regardless of exactly how the missing-binary condition
    was detected (an up-front `shutil.which` check, or a caught
    `pytesseract.TesseractNotFoundError` from an actual invocation
    attempt) -- both paths raise this same type with the same
    install-instructions message.
    """


@dataclass(frozen=True)
class OCRDetection:
    """One per-detection (typically per-word) OCR result.

    Attributes:
        text: The recognized text for this detection.
        confidence: Backend-reported confidence, normalized to ``[0.0,
            1.0]`` (EasyOCR already reports in this range; Tesseract's
            ``image_to_data`` reports ``0-100`` and is divided by 100 here
            so both backends are directly comparable).
        bbox: Axis-aligned bounding box as ``(left, top, right, bottom)``
            in pixel coordinates. EasyOCR natively returns a four-point
            polygon; it is normalized here to the same axis-aligned
            ``(left, top, right, bottom)`` shape Tesseract's
            ``image_to_data`` provides, so both backends return
            comparably-shaped results per the task's "comparably-structured
            results" requirement.
    """

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OCRResult:
    """Result of :func:`run_ocr`: full extracted text plus per-detection detail.

    Attributes:
        text: The full extracted text, i.e. every detection's ``text``
            joined with a single space, in the order the backend returned
            them.
        detections: Per-detection bounding boxes + confidence scores (see
            :class:`OCRDetection`). Empty if the backend found no text.
        backend: Which backend (from :data:`SUPPORTED_OCR_BACKENDS`)
            produced this result -- kept on the result itself so a caller
            comparing multiple backends' output doesn't need to track it
            separately.
    """

    text: str
    detections: list[OCRDetection] = field(default_factory=list)
    backend: str = ""


def _to_pil_image(image: "Image.Image | np.ndarray | str | Path") -> Image.Image:
    """Normalize any of this module's accepted image input shapes to a PIL Image.

    Mirrors `dscraft.vision.pipeline.SimpleImagePipeline.decode`'s
    "always convert to RGB" discipline, so downstream backends always see a
    consistent 3-channel image regardless of the source (grayscale array,
    RGBA file, etc.) -- not a second/parallel decode convention.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return img.convert("RGB")
    raise TypeError(
        "run_ocr's `image` argument must be a PIL.Image.Image, a "
        "numpy.ndarray, or a file path (str/pathlib.Path); got "
        f"{type(image).__name__!r}."
    )


def _run_easyocr(image: Image.Image, **kwargs: Any) -> OCRResult:
    """Run OCR via EasyOCR.

    ``easyocr`` (and its transitive ``torch``/``torchvision`` dependency,
    already required unconditionally by the base ``vision`` extra) is
    imported lazily here, inside this function only -- never at module
    level -- so importing `dscraft.vision.ocr` does not force an eager
    `easyocr.Reader` construction (which downloads/loads detector+recognizer
    model weights) on every caller, even ones who only want the
    ``tesseract`` backend.

    Args:
        image: A decoded, RGB `PIL.Image.Image`.
        **kwargs: Forwarded to `easyocr.Reader` (e.g. ``lang_list``,
            ``gpu``). Defaults to English, GPU auto-detection left to
            EasyOCR's own default.

    Returns:
        An :class:`OCRResult` with ``backend="easyocr"``.
    """
    import easyocr  # local import: keeps the base `vision` extra torch-only-if-needed

    lang_list = kwargs.pop("lang_list", ["en"])
    reader = easyocr.Reader(lang_list, **kwargs)

    array = np.asarray(image)
    # detail=1 returns (bbox, text, confidence) triples -- the structured
    # form this module needs to build per-detection OCRDetection entries,
    # as opposed to detail=0's plain-text-only list.
    raw_detections = reader.readtext(array, detail=1)

    detections: list[OCRDetection] = []
    for polygon, text, confidence in raw_detections:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        detections.append(
            OCRDetection(
                text=text,
                confidence=float(confidence),
                bbox=(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
            )
        )

    full_text = " ".join(detection.text for detection in detections)
    return OCRResult(text=full_text, detections=detections, backend="easyocr")


def _run_tesseract(image: Image.Image, **kwargs: Any) -> OCRResult:
    """Run OCR via Tesseract (through the `pytesseract` wrapper).

    ``pytesseract`` is imported lazily here, inside this function only --
    never at module level -- matching the same discipline as
    :func:`_run_easyocr` and `dscraft.automl.compile._require_onnx_stack`.

    Since Tesseract requires an external system binary (not just the
    ``pytesseract`` pip package), this function checks for the binary's
    presence *up front* via `shutil.which` before attempting any OCR call,
    so a missing binary always raises the same clear,
    :class:`TesseractNotInstalledError` -- never an opaque traceback from
    deep inside `pytesseract`/`subprocess`. As a second line of defense
    (e.g. ``tesseract`` is on PATH but broken/misconfigured), a
    `pytesseract.TesseractNotFoundError` raised by the actual call is also
    caught and re-raised as the same clear error type.

    Args:
        image: A decoded, RGB `PIL.Image.Image`.
        **kwargs: Forwarded to `pytesseract.image_to_data` (e.g. ``lang``,
            ``config``).

    Returns:
        An :class:`OCRResult` with ``backend="tesseract"``.

    Raises:
        TesseractNotInstalledError: The system ``tesseract`` binary is not
            installed/discoverable.
    """
    if shutil.which("tesseract") is None:
        raise TesseractNotInstalledError(
            "The 'tesseract' OCR backend requires the system `tesseract` "
            "binary, which is not installed or not on PATH. This binary is "
            "NOT a pip package and cannot be installed via "
            "`pip install dscraft[vision]` -- install it separately:\n"
            "    macOS:         brew install tesseract\n"
            "    Debian/Ubuntu: apt-get install tesseract-ocr\n"
            "Then retry run_ocr(..., backend='tesseract')."
        )

    import pytesseract  # local import: pip package only, needs the system binary above
    from pytesseract import Output

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT, **kwargs)
    except pytesseract.TesseractNotFoundError as exc:
        # Second line of defense: `tesseract` was on PATH (the shutil.which
        # check above passed) but pytesseract still couldn't invoke it
        # (e.g. broken symlink, permissions). Re-raise as the same clear,
        # actionable error type rather than letting this propagate raw.
        raise TesseractNotInstalledError(
            "The system `tesseract` binary was found on PATH but could not "
            "be invoked. Verify the installation (e.g. `tesseract "
            "--version`) and reinstall if needed:\n"
            "    macOS:         brew install tesseract\n"
            "    Debian/Ubuntu: apt-get install tesseract-ocr"
        ) from exc

    detections: list[OCRDetection] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            # image_to_data returns one row per detected box at every
            # granularity level (page/block/paragraph/line/word); rows
            # above word level have empty text and are not real detections.
            continue
        left, top = float(data["left"][i]), float(data["top"][i])
        width, height = float(data["width"][i]), float(data["height"][i])
        # Tesseract reports confidence 0-100 (and -1 for non-text rows,
        # already filtered out above by the empty-text check); normalize to
        # [0, 1] to match EasyOCR's native range for comparable results.
        confidence = max(0.0, float(data["conf"][i])) / 100.0
        detections.append(
            OCRDetection(
                text=text,
                confidence=confidence,
                bbox=(left, top, left + width, top + height),
            )
        )

    full_text = " ".join(detection.text for detection in detections)
    return OCRResult(text=full_text, detections=detections, backend="tesseract")


_BACKEND_DISPATCH = {
    "easyocr": _run_easyocr,
    "tesseract": _run_tesseract,
}


def run_ocr(
    image: "Image.Image | np.ndarray | str | Path",
    backend: str = "easyocr",
    **kwargs: Any,
) -> OCRResult:
    """Extract text from an image using the requested OCR backend.

    This is the **one canonical** OCR entry point in this package -- both
    backends are reached exclusively through this function's allowlist
    dispatch, mirroring `dscraft.forecast.forecast`'s
    ``SUPPORTED_MODELS``-driven dispatch shape. Per the architecture doc's
    multi-backend design principle, neither backend is treated as the
    default "winner" beyond this function's ``backend`` parameter default
    (chosen for zero-external-binary convenience, not technical
    superiority) -- callers needing Tesseract's smaller footprint or
    EasyOCR's GPU/MPS path simply pass ``backend=...`` explicitly.

    Args:
        image: A `PIL.Image.Image`, `numpy.ndarray`, or file path
            (str/pathlib.Path). Matches
            `dscraft.vision.pipeline.SimpleImagePipeline.decode`'s
            "always normalize to RGB" convention internally.
        backend: One of :data:`SUPPORTED_OCR_BACKENDS`'s keys
            (``"easyocr"`` or ``"tesseract"``). Defaults to ``"easyocr"``.
        **kwargs: Backend-specific keyword arguments, forwarded to
            `easyocr.Reader` (for the ``"easyocr"`` backend) or
            `pytesseract.image_to_data` (for the ``"tesseract"`` backend).

    Returns:
        An :class:`OCRResult` with the extracted text and per-detection
        bounding boxes/confidence scores.

    Raises:
        ValueError: ``backend`` is not one of :data:`SUPPORTED_OCR_BACKENDS`.
        TesseractNotInstalledError: ``backend="tesseract"`` was requested
            but the system ``tesseract`` binary is not installed.
    """
    if backend not in SUPPORTED_OCR_BACKENDS:
        raise ValueError(
            f"Unsupported OCR backend {backend!r}. Supported backends: "
            f"{sorted(SUPPORTED_OCR_BACKENDS)!r}."
        )

    pil_image = _to_pil_image(image)
    return _BACKEND_DISPATCH[backend](pil_image, **kwargs)
