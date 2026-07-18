"""Tests for `benchcraft_lazyvision.export`: torch.export -> ONNX correctness.

Hermetic: uses `benchcraft_lazyvision.model.build_model` (deterministic,
untrained weight init) and `synthetic_classification_batch` (random
tensors, no network/dataset download) so the whole suite runs offline and
fast, mirroring the correctness-verification pattern already used by
`packages/automl`'s `.compile()` tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchcraft_lazyvision import (
    ExportResult,
    ModelConfig,
    build_model,
    export_to_onnx,
    synthetic_classification_batch,
    verify_export,
)


@pytest.fixture()
def small_model_and_config() -> tuple[torch.nn.Module, ModelConfig]:
    """A small, deterministically-initialized TinyCNN (16x16 input, 4
    classes) shared by this module's tests, kept small purely to keep
    `torch.export`/ONNX export fast."""
    config = ModelConfig(in_channels=3, image_size=16, num_classes=4)
    model = build_model(config, seed=0, device="cpu")
    return model, config


def test_export_to_onnx_writes_a_file(tmp_path, small_model_and_config) -> None:
    """export_to_onnx() must write a non-empty .onnx file to the given path."""
    model, config = small_model_and_config
    example_input, _ = synthetic_classification_batch(config, batch_size=2, seed=1, device="cpu")
    onnx_path = tmp_path / "tiny_cnn.onnx"

    export_to_onnx(model, example_input, onnx_path)

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


def test_exported_onnx_matches_pytorch_on_training_input(
    tmp_path, small_model_and_config
) -> None:
    """verify_export() must report a match when run on the exact same
    example input that was used to trace the export."""
    model, config = small_model_and_config
    example_input, _ = synthetic_classification_batch(config, batch_size=2, seed=1, device="cpu")
    onnx_path = tmp_path / "tiny_cnn.onnx"
    export_to_onnx(model, example_input, onnx_path)

    result = verify_export(model, onnx_path, example_input, atol=1e-4, rtol=1e-3)

    assert isinstance(result, ExportResult)
    assert result.matched
    assert result.max_abs_diff < 1e-3


def test_exported_onnx_matches_pytorch_on_a_batch_of_fresh_random_inputs(
    tmp_path, small_model_and_config
) -> None:
    """Verify correctness on inputs *different* from the one used to trace
    the export -- the acceptance criteria call for checking a batch of
    random inputs, not just the exact tracing example."""
    model, config = small_model_and_config
    trace_input, _ = synthetic_classification_batch(config, batch_size=2, seed=1, device="cpu")
    onnx_path = tmp_path / "tiny_cnn.onnx"
    export_to_onnx(model, trace_input, onnx_path)

    fresh_input, _ = synthetic_classification_batch(config, batch_size=8, seed=999, device="cpu")

    result = verify_export(model, onnx_path, fresh_input, atol=1e-4, rtol=1e-3)

    assert result.matched
    assert result.max_abs_diff < 1e-3


def test_verify_export_raises_on_real_mismatch(tmp_path, small_model_and_config) -> None:
    """Sanity-check that verify_export actually detects a mismatch, rather
    than trivially passing: export one model, then verify a *different*
    model's output against that ONNX file, which must not match."""
    model, config = small_model_and_config
    example_input, _ = synthetic_classification_batch(config, batch_size=2, seed=1, device="cpu")
    onnx_path = tmp_path / "tiny_cnn.onnx"
    export_to_onnx(model, example_input, onnx_path)

    different_model = build_model(config, seed=12345, device="cpu")

    with pytest.raises(ValueError, match="does not match"):
        verify_export(different_model, onnx_path, example_input, atol=1e-4, rtol=1e-3)


def test_onnxruntime_output_is_a_real_numpy_computation(
    tmp_path, small_model_and_config
) -> None:
    """Independently exercise onnxruntime.InferenceSession directly (not
    just through verify_export) to confirm the exported graph is a real,
    runnable ONNX model, not just a file that happens to exist."""
    import onnxruntime

    model, config = small_model_and_config
    example_input, _ = synthetic_classification_batch(config, batch_size=3, seed=2, device="cpu")
    onnx_path = tmp_path / "tiny_cnn.onnx"
    export_to_onnx(model, example_input, onnx_path)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    (onnx_output,) = session.run(None, {input_name: example_input.numpy()})

    with torch.no_grad():
        torch_output = model(example_input).numpy()

    assert onnx_output.shape == torch_output.shape == (3, config.num_classes)
    assert np.allclose(onnx_output, torch_output, atol=1e-4, rtol=1e-3)
