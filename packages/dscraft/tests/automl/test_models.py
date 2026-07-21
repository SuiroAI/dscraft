"""Tests for dscraft.automl.models.

Covers the multi-backend gradient-boosted-tree model factory:
fit+predict smoke tests for each of the 6 (3 libraries x 2 task types)
supported backends, plus the unknown-name `ValueError` dispatch path.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from dscraft.automl import SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS, build_model
from dscraft.automl.models import build_model as build_model_direct


def test_public_api_surface() -> None:
    """`build_model`/`SUPPORTED_*` are importable from `dscraft.automl` itself."""
    assert build_model is build_model_direct
    assert set(SUPPORTED_CLASSIFIERS) == {"XGBoost", "LightGBM", "CatBoost"}
    assert set(SUPPORTED_REGRESSORS) == {"XGBoost", "LightGBM", "CatBoost"}


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
def test_classifier_fit_predict_smoke(name: str) -> None:
    """Each classifier backend fits and predicts on the bundled iris dataset."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # CatBoost defaults to writing a `catboost_info/` training-artifact
    # directory into the current working directory; disable that for
    # tests (and silence its otherwise-verbose training log).
    kwargs = {"verbose": False, "allow_writing_files": False} if name == "CatBoost" else {}
    model = build_model(name, "classification", n_estimators=10, **kwargs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert np.isin(np.asarray(predictions).ravel(), np.unique(y)).all()


@pytest.mark.parametrize("name", ["XGBoost", "LightGBM", "CatBoost"])
def test_regressor_fit_predict_smoke(name: str) -> None:
    """Each regressor backend fits and predicts on the bundled diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # CatBoost defaults to writing a `catboost_info/` training-artifact
    # directory into the current working directory; disable that for
    # tests (and silence its otherwise-verbose training log).
    kwargs = {"verbose": False, "allow_writing_files": False} if name == "CatBoost" else {}
    model = build_model(name, "regression", n_estimators=10, **kwargs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_build_model_unknown_task_raises() -> None:
    """An unrecognized `task` raises ValueError listing valid task names."""
    with pytest.raises(ValueError, match="Unsupported task"):
        build_model("XGBoost", "not-a-real-task")


def test_build_model_unknown_name_raises() -> None:
    """An unrecognized model `name` raises ValueError listing valid names."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("NotARealLibrary", "classification")
