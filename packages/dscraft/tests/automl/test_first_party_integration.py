"""Integration tests for the first-party automl backends end to end.

These tie the rebuilt pieces together through the public API and pin down two contracts:

1. A first-party estimator (built by `build_model`) works as a real `sklearn.pipeline.Pipeline`
   step -- fit/predict through the pipeline, exposing `n_features_in_` -- which is the
   sklearn-compatibility contract the rebuild promised (see the ADR).
2. `dscraft.automl.compile()` still compiles a *sklearn-native* pipeline to ONNX unchanged (a
   regression guard that the rebuild didn't disturb the export path), AND -- honestly documented
   here -- `compile()` does NOT support the first-party `DS*` estimators, because `skl2onnx` has
   no shape calculator / converter registered for them. ONNX export of the first-party estimators
   is a separate, not-yet-built capability, not a silent partial behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dscraft.automl import build_clusterer, build_model, build_resampler, compile
from dscraft.automl import _primitives as _p


def test_first_party_classifier_in_sklearn_pipeline() -> None:
    """A build_model estimator fits/predicts as a real sklearn Pipeline step."""
    X, y = load_iris(return_X_y=True)
    pipe = Pipeline(
        [("scale", StandardScaler()), ("clf", build_model("DSBoost", "classification", n_estimators=20, random_state=0))]
    )
    pipe.fit(X, y)
    assert _p.accuracy_score(y, pipe.predict(X)) > 0.9
    assert pipe.named_steps["clf"].n_features_in_ == X.shape[1]


def test_resampler_then_first_party_classifier_end_to_end() -> None:
    """A full imbalanced workflow: first-party SMOTE -> first-party classifier, all in-house."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, weights=[0.9, 0.1], random_state=0
    )
    X_res, y_res = build_resampler("SMOTE", random_state=0).fit_resample(X, y)
    # Classes are balanced after resampling.
    _vals, counts = np.unique(y_res, return_counts=True)
    assert counts.min() == counts.max()
    clf = build_model("DSRandomForest", "classification", n_estimators=20, random_state=0)
    clf.fit(X_res, y_res)
    assert len(clf.predict(X)) == len(X)


def test_first_party_clusterer_end_to_end() -> None:
    """A build_clusterer estimator clusters a standardized feature matrix."""
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=0)
    Xs = StandardScaler().fit_transform(X)
    labels = build_clusterer("DSKMeans", n_clusters=3, random_state=0).fit_predict(Xs)
    assert len(set(labels)) == 3


def test_compile_still_works_for_sklearn_native_pipeline() -> None:
    """Regression guard: the rebuild left the sklearn-native -> ONNX compile path intact."""
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression

    from dscraft.automl import CompileOptions

    X, y = load_iris(return_X_y=True)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    onnx_model = compile(pipe, X.astype(np.float32), options=CompileOptions(zipmap=False))
    session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    onnx_labels = session.run(None, {session.get_inputs()[0].name: X.astype(np.float32)})[0]
    assert np.array_equal(np.asarray(onnx_labels).astype(y.dtype), pipe.predict(X))


def test_compile_rejects_first_party_estimator_clearly() -> None:
    """compile() does NOT (yet) support the first-party DS* estimators -- documented, not silent.

    skl2onnx has no converter registered for the from-scratch estimators, so it raises a clear
    error naming the unconvertible type rather than producing a wrong graph. ONNX export of the
    first-party estimators is a separate future capability.
    """
    pytest.importorskip("skl2onnx")
    from dscraft.automl import CompileOptions

    X, y = load_iris(return_X_y=True)
    pipe = Pipeline(
        [("scale", StandardScaler()), ("clf", build_model("DSLogistic", "classification"))]
    ).fit(X, y)
    with pytest.raises(Exception) as exc_info:  # skl2onnx raises its own MissingShapeCalculator
        compile(pipe, X.astype(np.float32), options=CompileOptions(zipmap=False))
    # The error should name the unconvertible first-party type, not fail opaquely.
    assert "DSLogistic" in str(exc_info.value) or "shape calculator" in str(exc_info.value).lower()
