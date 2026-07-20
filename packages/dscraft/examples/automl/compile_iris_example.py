"""Runnable demo of dscraft.automl's `.compile()` capability on Iris.

Fits a real sklearn Pipeline (StandardScaler + LogisticRegression) on the
classic Iris toy dataset, compiles it to a single ONNX graph via
`dscraft.automl.compile`, runs inference through `onnxruntime`, and
asserts the ONNX output matches the original sklearn pipeline's own
predictions within tolerance.

This script only *calls* the package's public API -- it does not
reimplement any `.compile()` logic inline (per CLAUDE.md's "no net-new
scripts" rule).

Run with:
    python packages/dscraft/examples/automl/compile_iris_example.py

Requires the `onnx` extra:
    pip install -e "packages/dscraft[onnx]"
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dscraft.automl import CompileOptions, ONNXExtraNotInstalledError, compile


def main() -> None:
    """Fit, compile, and cross-check a sklearn pipeline on the Iris dataset.

    Trains a StandardScaler + LogisticRegression pipeline, compiles it to
    ONNX via `dscraft.automl.compile`, runs the graph through
    `onnxruntime`, and asserts the ONNX labels/probabilities match the
    original sklearn pipeline's own `predict`/`predict_proba` output.
    Exits with status 1 (printing an install hint) if the optional `onnx`
    extra is not installed.
    """
    try:
        import onnxruntime
    except ImportError:
        print(
            "This example requires the 'onnx' extra. Install with:\n"
            '    pip install -e "packages/dscraft[onnx]"',
            file=sys.stderr,
        )
        raise SystemExit(1)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    sklearn_accuracy = pipeline.score(X_test, y_test)
    print(f"Fitted sklearn Pipeline test accuracy: {sklearn_accuracy:.4f}")

    try:
        onnx_model = compile(pipeline, X_test, options=CompileOptions(zipmap=False))
    except ONNXExtraNotInstalledError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    print(f"Compiled ONNX graph with {len(onnx_model.graph.node)} nodes.")

    session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    X_test_f32 = X_test.astype(np.float32)
    onnx_labels, onnx_proba = session.run(None, {input_name: X_test_f32})

    sklearn_labels = pipeline.predict(X_test)
    sklearn_proba = pipeline.predict_proba(X_test)

    labels_match = np.array_equal(np.asarray(onnx_labels).astype(sklearn_labels.dtype), sklearn_labels)
    proba_close = np.allclose(np.asarray(onnx_proba), sklearn_proba, atol=1e-4)

    print(f"ONNX predicted labels match sklearn exactly: {labels_match}")
    print(f"ONNX predicted probabilities match sklearn within 1e-4: {proba_close}")

    assert labels_match, "ONNX label predictions diverged from the sklearn pipeline."
    assert proba_close, "ONNX predict_proba output diverged from the sklearn pipeline."

    print("compile() correctness check passed: ONNX graph matches sklearn pipeline.")


if __name__ == "__main__":
    main()
