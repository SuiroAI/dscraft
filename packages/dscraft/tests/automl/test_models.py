"""Tests for dscraft.automl.models.

Covers the first-party multi-backend model factory: fit+predict smoke tests for each of the
first-party classification and regression backends (linear/logistic, gradient boosting, random
forest, extra-trees), the public-API surface, and the unknown-name/unknown-task `ValueError`
dispatch paths. The third-party XGBoost/LightGBM/CatBoost backends were removed by the first-party
automl/eda rebuild (see docs/decisions/2026-10-first-party-automl-eda-rebuild.md); their
capability is now provided by the first-party `DSBoost` backend.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from dscraft.automl import SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS, build_model
from dscraft.automl.models import build_model as build_model_direct

#: Every first-party classification / regression backend name the allowlists expose.
_CLASSIFIER_NAMES = ["DSLogistic", "DSBoost", "DSRandomForest", "DSExtraTrees"]
_REGRESSOR_NAMES = ["DSLinear", "DSBoost", "DSRandomForest", "DSExtraTrees"]


def test_public_api_surface() -> None:
    """`build_model`/`SUPPORTED_*` are importable from `dscraft.automl`, all first-party now."""
    assert build_model is build_model_direct
    assert set(SUPPORTED_CLASSIFIERS) == set(_CLASSIFIER_NAMES)
    assert set(SUPPORTED_REGRESSORS) == set(_REGRESSOR_NAMES)
    # No third-party backend names survive the rebuild.
    assert "XGBoost" not in SUPPORTED_CLASSIFIERS
    assert "LightGBM" not in SUPPORTED_REGRESSORS
    assert "CatBoost" not in SUPPORTED_CLASSIFIERS


def _small_kwargs(name: str) -> dict:
    """Keep the ensemble backends tiny so the smoke tests stay fast."""
    if name in ("DSBoost", "DSRandomForest", "DSExtraTrees"):
        return {"n_estimators": 15, "random_state": 0}
    if name == "DSLogistic":
        return {"C": 10.0}
    return {}  # DSLinear


@pytest.mark.parametrize("name", _CLASSIFIER_NAMES)
def test_classifier_fit_predict_smoke(name: str) -> None:
    """Each first-party classifier backend fits and predicts on the bundled iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = build_model(name, "classification", **_small_kwargs(name))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert np.isin(np.asarray(predictions).ravel(), np.unique(y)).all()


@pytest.mark.parametrize("name", _REGRESSOR_NAMES)
def test_regressor_fit_predict_smoke(name: str) -> None:
    """Each first-party regressor backend fits and predicts on the bundled diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = build_model(name, "regression", **_small_kwargs(name))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)


def test_build_model_unknown_task_raises() -> None:
    """An unrecognized `task` raises ValueError listing valid task names."""
    with pytest.raises(ValueError, match="Unsupported task"):
        build_model("DSBoost", "not-a-real-task")


def test_build_model_unknown_name_raises() -> None:
    """An unrecognized model `name` raises ValueError listing valid names."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("NotARealBackend", "classification")


def test_removed_third_party_backend_name_raises() -> None:
    """A removed third-party backend name is now an unknown name, not a silent fallback."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("XGBoost", "classification")
