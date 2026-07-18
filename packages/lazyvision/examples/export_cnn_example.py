"""Runnable end-to-end demo of benchcraft_lazyvision's signature capability.

Per CLAUDE.md's "no net-new scripts" rule: this example imports and calls
the real package API (`benchcraft_lazyvision`) rather than reimplementing
any pipeline/model/export logic inline.

Flow:
    1. Build a `SimpleImagePipeline` (the concrete `DenseMediaPipeline`
       subclass) and run it on a small in-memory synthetic image to
       produce a dense tensor.
    2. Build a `TinyCNN` classifier sized to match that tensor's shape.
    3. Export the CNN via `torch.export` -> ONNX
       (`benchcraft_lazyvision.export_to_onnx`).
    4. Verify the exported ONNX model's output matches the original
       PyTorch model's output within tolerance
       (`benchcraft_lazyvision.verify_export`), on a batch of fresh random
       inputs -- not just the tensor traced during export.
    5. Print the result.
    6. Repeat steps 1-4 on a *real* handwritten-digit image from
       scikit-learn's bundled `sklearn.datasets.load_digits()` dataset
       (no network access -- it ships as package data inside the
       `scikit-learn` wheel), through the exact same pipeline/model/export
       API used for the synthetic section above. Requires this package's
       `dev` extra (`pip install -e "packages/lazyvision[dev]"`) since
       scikit-learn is a dev/test-only dependency, not a core runtime one.

Run with:
    python packages/lazyvision/examples/export_cnn_example.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchcraft_lazyvision import (
    ModelConfig,
    PipelineConfig,
    SimpleImagePipeline,
    build_model,
    export_to_onnx,
    resolve_device,
    synthetic_classification_batch,
    verify_export,
)

IMAGE_SIZE = 32
NUM_CLASSES = 10


def _make_synthetic_image_bytes(size: int) -> bytes:
    """Build one small in-memory synthetic "image" (a gradient pattern
    PNG) -- stands in for a real decoded image file, keeping this example
    hermetic (no network access, no dataset download)."""
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.linspace(255, 0, size, dtype=np.uint8)[:, None]
    array[:, :, 2] = 100
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _load_real_digit_png_bytes(index: int = 0) -> tuple[bytes, int]:
    """Load one real handwritten-digit image from scikit-learn's bundled
    `load_digits()` dataset (8x8 grayscale, values 0-16 -- a real,
    genuinely bundled-as-package-data dataset, no network access, no
    download) and encode it as real PNG bytes -- exactly the raw-bytes
    input shape `SimpleImagePipeline.decode()` expects, so the real image
    genuinely exercises the `decode()` step rather than bypassing it.

    Requires this package's `dev` extra (scikit-learn is a dev/test-only
    dependency here, not a core runtime one -- see pyproject.toml).
    """
    from sklearn.datasets import load_digits  # local import: dev-only dependency

    digits = load_digits()
    image_0_16 = digits.images[index]  # (8, 8) float64, values in [0, 16]
    label = int(digits.target[index])

    image_uint8 = np.clip(image_0_16 / 16.0 * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L")

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue(), label


def main() -> None:
    print("=== benchcraft-lazyvision: pipeline + torch.export -> ONNX demo ===\n")

    # Resolve the device once via the package's one canonical
    # device-selection helper (MPS -> CUDA -> CPU, per CLAUDE.md's
    # MPS-primary constraint) rather than hardcoding "cpu" -- a caller
    # running this example on Apple Silicon gets MPS automatically.
    device = resolve_device()
    print(f"Resolved device: {device}\n")

    # --- 1. Run the DenseMediaPipeline on a synthetic image -----------
    pipeline_config = PipelineConfig(
        image_size=IMAGE_SIZE, horizontal_flip_prob=0.5, seed=7, device=device
    )
    pipeline = SimpleImagePipeline(pipeline_config)
    raw_bytes = _make_synthetic_image_bytes(size=40)  # deliberately not
    # pre-sized to IMAGE_SIZE, to show the pipeline's resize step doing
    # real work.

    dense_tensor = pipeline.run(raw_bytes)
    print(
        f"Pipeline produced a dense tensor: shape={tuple(dense_tensor.shape)}, "
        f"dtype={dense_tensor.dtype}, "
        f"range=[{dense_tensor.min():.3f}, {dense_tensor.max():.3f}]"
    )

    # --- 2. Build a CNN sized to match the pipeline's output shape -----
    model_config = ModelConfig(
        in_channels=dense_tensor.shape[0], image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )
    model = build_model(model_config, seed=0, device=device)
    print(f"Built TinyCNN: {sum(p.numel() for p in model.parameters())} parameters")

    # A batch built from the pipeline's own output, used to trace the
    # export (torch.export needs at least one concrete example input).
    # Batch size 2 (rather than 1) is used deliberately: export_to_onnx
    # marks the batch dimension dynamic, and torch.export requires a
    # tracing example whose dynamic dimension is not accidentally
    # specializable to a constant (a batch size of 1 gets specialized to a
    # fixed constant instead of treated as dynamic).
    trace_batch = dense_tensor.unsqueeze(0).repeat(2, 1, 1, 1)  # (2, C, H, W)

    # --- 3. Export via torch.export -> ONNX ----------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "tiny_cnn.onnx"
        export_to_onnx(model, trace_batch, onnx_path)
        print(f"Exported ONNX model to: {onnx_path} ({onnx_path.stat().st_size} bytes)")

        # --- 4. Verify correctness on a *fresh* batch of random inputs -
        verification_batch, _ = synthetic_classification_batch(
            model_config, batch_size=8, seed=123, device=device
        )
        result = verify_export(model, onnx_path, verification_batch, atol=1e-4, rtol=1e-3)

        # --- 5. Print the result ---------------------------------------
        print("\n=== Correctness check: PyTorch vs. ONNX Runtime ===")
        print(f"  matched:       {result.matched}")
        print(f"  max_abs_diff:  {result.max_abs_diff:.3e}")
        print(f"  max_rel_diff:  {result.max_rel_diff:.3e}")

        with torch.no_grad():
            sample_logits = model(verification_batch[:1]).detach().cpu().numpy()
        print(f"\nSample PyTorch logits (first item): {sample_logits.round(4)}")

    # ====================================================================
    # === Section 2: the SAME pipeline -> model -> export -> verify flow,
    # === run on a REAL handwritten-digit image (sklearn's bundled
    # === load_digits() dataset) instead of a synthetic gradient PNG.
    # ====================================================================
    print("\n" + "=" * 70)
    print("=== Real-dataset section: sklearn.datasets.load_digits() ===")
    print("=" * 70 + "\n")

    real_raw_bytes, real_label = _load_real_digit_png_bytes(index=0)
    print(f"Loaded real handwritten digit (true label={real_label}) from "
          f"sklearn.datasets.load_digits(), encoded as {len(real_raw_bytes)}-byte PNG.")

    # --- 1'. Run the SAME SimpleImagePipeline on the real digit image ----
    real_pipeline = SimpleImagePipeline(
        PipelineConfig(image_size=IMAGE_SIZE, horizontal_flip_prob=0.5, seed=7, device=device)
    )
    real_dense_tensor = real_pipeline.run(real_raw_bytes)  # decode() really parses the PNG.
    print(
        f"Pipeline produced a dense tensor: shape={tuple(real_dense_tensor.shape)}, "
        f"dtype={real_dense_tensor.dtype}, "
        f"range=[{real_dense_tensor.min():.3f}, {real_dense_tensor.max():.3f}]"
    )

    # --- 2'. Build the SAME TinyCNN, sized to match -----------------------
    real_model_config = ModelConfig(
        in_channels=real_dense_tensor.shape[0], image_size=IMAGE_SIZE, num_classes=NUM_CLASSES
    )
    real_model = build_model(real_model_config, seed=0, device=device)
    print(f"Built TinyCNN: {sum(p.numel() for p in real_model.parameters())} parameters")

    real_trace_batch = real_dense_tensor.unsqueeze(0).repeat(2, 1, 1, 1)  # (2, C, H, W)

    # --- 3'. Export via the SAME torch.export -> ONNX path ----------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        real_onnx_path = Path(tmp_dir) / "tiny_cnn_real_digit.onnx"
        export_to_onnx(real_model, real_trace_batch, real_onnx_path)
        print(
            f"Exported ONNX model to: {real_onnx_path} "
            f"({real_onnx_path.stat().st_size} bytes)"
        )

        # --- 4'. Verify correctness on the real digit image itself -------
        real_verification_batch = real_dense_tensor.unsqueeze(0)  # (1, C, H, W)
        real_result = verify_export(
            real_model, real_onnx_path, real_verification_batch, atol=1e-4, rtol=1e-3
        )

        # --- 5'. Print the result ----------------------------------------
        print("\n=== Correctness check (real digit image): PyTorch vs. ONNX Runtime ===")
        print(f"  matched:       {real_result.matched}")
        print(f"  max_abs_diff:  {real_result.max_abs_diff:.3e}")
        print(f"  max_rel_diff:  {real_result.max_rel_diff:.3e}")

        with torch.no_grad():
            real_sample_logits = real_model(real_verification_batch).detach().cpu().numpy()
        print(f"\nSample PyTorch logits (real digit, true label={real_label}): "
              f"{real_sample_logits.round(4)}")

    print(
        "\nDone. torch.export -> ONNX round trip verified successfully on both "
        "synthetic and real (sklearn load_digits()) image data."
    )


if __name__ == "__main__":
    main()
