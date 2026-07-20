"""The export path: capture a CNN via `torch.export`, terminal format ONNX.

Per the architecture doc §2.5 ("Export/Compilation Architecture") and Part 3
("Module 5: LazyVision"): backend 1 (ONNX/ONNX Runtime) is the shared
terminal execution format for LazyVision, reached via `torch.export` ->
onnx-graphsurgeon-style lowering, as distinct from AutoML's `skl2onnx` path
and LazyTune's GGUF/MLX path -- these three backends are permanently
separate and are not being unified here.

**Exact mechanism used, and why:** this module calls
``torch.export.export()`` to capture the model as a functional
``ExportedProgram`` (structural/shape tracing via TorchDynamo, the same
frontend the architecture doc calls out as a legitimate shared capture step
with the future, deferred edge-compilation module), then passes that
``ExportedProgram`` straight into ``torch.onnx.export(..., dynamo=True)`` to
lower it to ONNX. This corner of the PyTorch API has moved around across
versions -- ``torch.onnx.dynamo_export`` was the original entrypoint for
dynamo-based ONNX export, then deprecated in favor of
``torch.onnx.export(..., dynamo=True)`` from PyTorch 2.5 onward, with
``dynamo_export`` itself removed in later releases. This module deliberately
uses the ``torch.onnx.export(..., dynamo=True)`` form (not
``dynamo_export``, and not the legacy TorchScript-tracing exporter) because
it is the current, non-deprecated, actively-maintained API as of the
PyTorch version pinned in this package's ``pyproject.toml`` (``torch>=2.5``).
See the package README for more detail on this API-churn note.

This is the **one canonical** export path in this package -- there is no
second/parallel ONNX export function anywhere else in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["ExportResult", "export_to_onnx", "verify_export"]


@dataclass(frozen=True)
class ExportResult:
    """Result of :func:`verify_export`: how closely the exported ONNX model's
    output matched the original PyTorch model's output.

    Attributes:
        onnx_path: Path the ONNX model was loaded from for verification.
        max_abs_diff: Largest absolute difference between the PyTorch and
            ONNX Runtime outputs, elementwise, over the verification batch.
        max_rel_diff: Largest relative difference (relative to the
            PyTorch output's magnitude), elementwise.
        matched: Whether ``max_abs_diff``/``max_rel_diff`` were within the
            requested ``atol``/``rtol`` tolerance (i.e. `numpy.allclose`
            returned ``True``).
    """

    onnx_path: Path
    max_abs_diff: float
    max_rel_diff: float
    matched: bool


def export_to_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    onnx_path: str | Path,
) -> "torch.onnx.ONNXProgram":
    """Capture ``model`` via `torch.export.export` and lower it to ONNX.

    Args:
        model: The PyTorch module to export. Should already be in
            ``eval()`` mode (this function does not change training/eval
            state, callers are expected to control that -- e.g.
            `dscraft.vision.model.build_model` already returns an
            ``eval()`` model).
        example_input: A representative input tensor (batch included, e.g.
            shape ``(N, C, H, W)``) used both to trace the graph via
            `torch.export.export` and to determine the ONNX graph's
            input shape/dtype. Its batch dimension (dim 0) is marked
            dynamic (see below), so ``example_input``'s batch size does
            not need to match the batch size used at verification/
            inference time. Use a batch size of 2 or more here -- a batch
            size of exactly 1 causes `torch.export` to specialize the
            dimension to a fixed constant instead of treating it as
            dynamic, which raises a `ConstraintViolationError`.
        onnx_path: Where to write the resulting ``.onnx`` file.

    Returns:
        The `torch.onnx.ONNXProgram` produced by the dynamo-based exporter
        (see module docstring for exactly which API this is and why). The
        file at ``onnx_path`` has already been written by the time this
        function returns.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # `torch.export.export` captures static shapes by default -- the
    # traced graph would otherwise only ever accept inputs with exactly
    # `example_input`'s batch size. Marking dim 0 dynamic keeps the
    # exported ONNX model usable across batch sizes (e.g. verifying
    # against a differently-sized batch than the one used to trace),
    # which is both a realistic deployment requirement and what this
    # package's own test suite exercises.
    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = ({0: batch_dim},)

    with torch.no_grad():
        exported_program = torch.export.export(
            model, (example_input,), dynamic_shapes=dynamic_shapes
        )
        onnx_program = torch.onnx.export(exported_program, (example_input,), dynamo=True)

    # Saved explicitly (rather than relying on an `f=...` kwarg) so this
    # code path is robust to the exact `f=`-handling behavior of whichever
    # torch.onnx.export version is installed -- see module docstring's note
    # on API churn in this corner of PyTorch.
    onnx_program.save(str(onnx_path))
    return onnx_program


def verify_export(
    model: nn.Module,
    onnx_path: str | Path,
    example_input: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> ExportResult:
    """Run ``model`` and the exported ONNX model on the same input and
    compare outputs, mirroring the correctness-verification pattern already
    used by `dscraft.automl`'s `.compile()` tests.

    Args:
        model: The original PyTorch module (same one passed to
            :func:`export_to_onnx`).
        onnx_path: Path to the ``.onnx`` file written by
            :func:`export_to_onnx`.
        example_input: Input tensor to run through both models. Does not
            need to be the same tensor used to trace the export (a batch of
            random inputs of the same shape/dtype is the intended usage,
            per this package's test suite), as long as its shape is
            compatible with the traced graph.
        atol: Absolute tolerance passed to `numpy.allclose`.
        rtol: Relative tolerance passed to `numpy.allclose`.

    Returns:
        An :class:`ExportResult` describing the comparison.

    Raises:
        ValueError: The ONNX output does not match the PyTorch output
            within ``atol``/``rtol``.
    """
    import onnxruntime  # local import: keeps this an explicit, visible dep

    onnx_path = Path(onnx_path)

    model = model.eval()
    with torch.no_grad():
        torch_output = model(example_input).detach().cpu().numpy()

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_input = example_input.detach().cpu().numpy()
    (onnx_output,) = session.run(None, {input_name: onnx_input})

    abs_diff = np.abs(torch_output - onnx_output)
    max_abs_diff = float(np.max(abs_diff))
    rel_diff = abs_diff / (np.abs(torch_output) + 1e-12)
    max_rel_diff = float(np.max(rel_diff))
    matched = bool(np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol))

    result = ExportResult(
        onnx_path=onnx_path,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        matched=matched,
    )
    if not matched:
        raise ValueError(
            "ONNX export output does not match the original PyTorch "
            f"model's output within atol={atol}, rtol={rtol}: "
            f"max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff}."
        )
    return result
