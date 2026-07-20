"""dscraft.automl: DSCraft's clean-room tabular AutoML module.

This scaffold-depth pass implements exactly one signature capability from
the architecture doc (Part 3, "Module 1: AutoML"): the `.compile()` path
that fuses a fitted `sklearn.pipeline.Pipeline` into a single ONNX graph
via `skl2onnx`, replacing brittle pickle-based serialization. See
`compile.py` and the package README for details and clean-room provenance.

The streaming/incremental `partial_fit` fading-factor evaluator and the
PSI drift-detection feature from the same architecture doc section are
explicitly out of scope for this pass -- future work, not implemented
here.

Public API surface (this is the one canonical entrypoint -- no parallel
export path exists elsewhere in this package):
    >>> from dscraft.automl import compile, CompileOptions
"""

from dscraft.automl.compile import (
    CompileOptions,
    ONNXExtraNotInstalledError,
    compile,  # noqa: A004 -- intentional public API name, see architecture doc
)

__all__ = [
    "compile",
    "CompileOptions",
    "ONNXExtraNotInstalledError",
]

__version__ = "0.1.0"
