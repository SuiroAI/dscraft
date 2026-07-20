"""Tests for dscraft.automl.compile.

Covers the package's one signature capability for this scaffold-depth
pass: fusing a fitted sklearn Pipeline into a single ONNX graph, and
verifying that graph produces the same predictions as the original
pipeline when run through onnxruntime.

Most tests in this module exercise `compile()`'s actual ONNX conversion
and therefore require the `automl-onnx` extra (skl2onnx, onnx,
onnxruntime); those are skipped (not failed) via `_import_onnx_stack()`
if it isn't installed, since that extra is optional by design. A handful
of tests -- `test_public_api_surface` and the tests that assert `compile()`
raises before ever reaching ONNX conversion (bad pipeline type, unfitted
pipeline, non-numeric input, wrong feature count) -- need only the base
`automl` extra and deliberately do NOT call `_import_onnx_stack()`, so
they still run (and prove real coverage) in an `automl`-only environment
that never installed `automl-onnx`.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dscraft.automl
from dscraft.automl import CompileOptions, compile


def _import_onnx_stack():
    """Import `onnx`/`onnxruntime`, skipping the calling test if unavailable.

    Kept as a per-test helper (rather than a module-level
    `pytest.importorskip`) so tests that don't actually exercise ONNX
    conversion -- e.g. `test_public_api_surface`, which only needs the
    base `automl` extra -- are not silently skipped just because
    `automl-onnx` isn't installed in the current environment.
    """
    onnxruntime = pytest.importorskip(
        "onnxruntime", reason="onnxruntime not installed; skipping onnx-dependent tests"
    )
    onnx = pytest.importorskip("onnx")
    return onnx, onnxruntime


def _fit_scaler_logreg_pipeline(random_state: int = 0) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Build and fit a StandardScaler + LogisticRegression pipeline for tests.

    Returns the fitted pipeline along with the training features (unused by
    most callers, but returned for the pandas-DataFrame test that needs
    matching column counts) and a float32 test split ready to feed straight
    into `compile()` or an ONNX Runtime session.
    """
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test.astype(np.float32)


def _run_onnx(onnx_model, X: np.ndarray) -> np.ndarray:
    """Run `X` through a compiled ONNX model on the CPU execution provider.

    Returns the raw list of `onnxruntime` session outputs (typically
    `[labels, probabilities]` for a classifier), letting each test assert on
    whichever output index it cares about.
    """
    _, onnxruntime = _import_onnx_stack()
    session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.astype(np.float32)})
    return outputs


def test_public_api_surface():
    """`compile`, `CompileOptions`, and `ONNXExtraNotInstalledError` are all
    importable from the `dscraft.automl` package, and
    `dscraft.automl.compile` is the same object as
    `dscraft.automl.compile.compile` (one canonical export path)."""
    assert dscraft.automl.compile is compile
    assert hasattr(dscraft.automl, "CompileOptions")
    assert hasattr(dscraft.automl, "ONNXExtraNotInstalledError")


def test_compile_rejects_non_pipeline():
    """`compile()` raises `TypeError` when given a bare estimator instead of
    an `sklearn.pipeline.Pipeline`."""
    with pytest.raises(TypeError):
        compile(LogisticRegression(), np.zeros((1, 3)))


def test_compile_rejects_unfitted_pipeline():
    """`compile()` raises `NotFittedError` for a `Pipeline` whose steps have
    never been fit, before any ONNX conversion is attempted."""
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("clf", LogisticRegression())])
    with pytest.raises(NotFittedError):
        compile(pipeline, np.zeros((1, 3)))


def test_compile_produces_valid_onnx_model():
    """`compile()` returns an `onnx.ModelProto` that passes
    `onnx.checker.check_model`, i.e. a structurally valid ONNX graph."""
    onnx, _ = _import_onnx_stack()
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test)
    assert isinstance(onnx_model, onnx.ModelProto)
    onnx.checker.check_model(onnx_model)


def test_compiled_onnx_predictions_match_sklearn_predict():
    """Running the compiled ONNX graph through `onnxruntime` yields the same
    class labels as the original fitted pipeline's `predict()`."""
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test)

    sklearn_preds = pipeline.predict(X_test)
    onnx_outputs = _run_onnx(onnx_model, X_test)
    onnx_labels = onnx_outputs[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)


def test_compiled_onnx_predictions_match_sklearn_predict_proba():
    """With `zipmap=False`, the compiled ONNX graph's second output is a
    plain probability tensor that matches the original pipeline's
    `predict_proba()` within `1e-4` absolute tolerance."""
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    onnx_model = compile(pipeline, X_test, options=CompileOptions(zipmap=False))

    sklearn_proba = pipeline.predict_proba(X_test)
    onnx_outputs = _run_onnx(onnx_model, X_test)
    onnx_proba = np.asarray(onnx_outputs[1])

    assert np.allclose(onnx_proba, sklearn_proba, atol=1e-4)


def test_compile_accepts_pandas_dataframe_input():
    """`compile()` accepts a pandas DataFrame as `sample_input` (coercing it
    to a numeric array internally) and still produces an ONNX graph whose
    predictions match the sklearn pipeline run on the equivalent numpy
    array."""
    pd = pytest.importorskip("pandas")
    pipeline, X_train, X_test = _fit_scaler_logreg_pipeline()
    df_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])

    onnx_model = compile(pipeline, df_test)
    sklearn_preds = pipeline.predict(X_test)
    onnx_labels = _run_onnx(onnx_model, X_test)[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)


def test_compile_raises_typeerror_for_non_numeric_dataframe_column():
    """`compile()` raises `TypeError` (not a raw pandas/numpy `ValueError`)
    when `sample_input` is a DataFrame with a non-numeric column, and the
    error message names the offending column and mentions the float32
    target dtype."""
    pd = pytest.importorskip("pandas")
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()

    # A DataFrame with a non-numeric ("string") column cannot be coerced to
    # a float32 array. The documented contract (compile()'s docstring) says
    # this raises TypeError, not a raw ValueError from pandas/numpy.
    bad_df = pd.DataFrame({"f": ["a", "b", "c"]})

    with pytest.raises(TypeError) as exc_info:
        compile(pipeline, bad_df)

    message = str(exc_info.value)
    assert "float32" in message
    assert "f" in message


def test_compile_rejects_sample_input_with_wrong_feature_count():
    """`compile()` raises a clear `ValueError` (not a confusing downstream
    skl2onnx/ONNX error) when `sample_input`'s column count doesn't match
    `pipeline.n_features_in_`, whether the sample has too many or too few
    columns."""
    pipeline, _, X_test = _fit_scaler_logreg_pipeline()
    assert pipeline.n_features_in_ == X_test.shape[1]

    too_many = np.zeros((X_test.shape[0], X_test.shape[1] + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="feature"):
        compile(pipeline, too_many)

    too_few = np.zeros((X_test.shape[0], X_test.shape[1] - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="feature"):
        compile(pipeline, too_few)


def test_compile_on_breast_cancer_dataset_end_to_end():
    """End-to-end smoke test on a real (non-synthetic) dataset: a
    StandardScaler + LogisticRegression pipeline fit on
    `load_breast_cancer()` compiles to ONNX and the ONNX graph's predicted
    labels exactly match the sklearn pipeline's `predict()` output."""
    data = load_breast_cancer()
    X_train, X_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )
    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )
    pipeline.fit(X_train, y_train)

    onnx_model = compile(pipeline, X_test)
    sklearn_preds = pipeline.predict(X_test)
    onnx_labels = _run_onnx(onnx_model, X_test.astype(np.float32))[0]

    assert np.array_equal(onnx_labels.astype(sklearn_preds.dtype), sklearn_preds)
