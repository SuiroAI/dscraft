"""Tests for dscraft.automl._forest.

Validates the first-party bagging ensembles (random forest + extra-trees): they learn standard
classification/regression datasets, soft-voting probabilities are well-formed, the ensemble
reduces variance relative to a single tree, extra-trees uses the random splitter on un-resampled
data while the forest bootstraps, results are reproducible from one seed, and the estimators are
sklearn-API-compatible.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dscraft.automl import _primitives as _p
from dscraft.automl._forest import (
    DSExtraTreesClassifier,
    DSExtraTreesRegressor,
    DSRandomForestClassifier,
    DSRandomForestRegressor,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [DSRandomForestClassifier, DSExtraTreesClassifier])
def test_forest_classifier_learns_iris(cls) -> None:
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = cls(n_estimators=50, random_state=0).fit(X_tr, y_tr)
    assert _p.accuracy_score(y_te, model.predict(X_te)) > 0.9
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_te)), atol=1e-9)


@pytest.mark.parametrize("cls", [DSRandomForestClassifier, DSExtraTreesClassifier])
def test_forest_classifier_breast_cancer(cls) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = cls(n_estimators=60, random_state=0).fit(X_tr, y_tr)
    assert _p.accuracy_score(y_te, model.predict(X_te)) > 0.93


def test_forest_preserves_string_labels() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, size=(50, 2)), rng.normal(2, 0.5, size=(50, 2))])
    y = np.array(["a"] * 50 + ["b"] * 50)
    model = DSRandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({"a", "b"})


def test_forest_classifier_requires_two_classes() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        DSRandomForestClassifier().fit(np.zeros((4, 1)), np.zeros(4))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, sk_cls",
    [
        (DSRandomForestRegressor, "RandomForestRegressor"),
        (DSExtraTreesRegressor, "ExtraTreesRegressor"),
    ],
)
def test_forest_regressor_learns_diabetes(cls, sk_cls) -> None:
    # Diabetes is noisy; a sqrt-feature forest over 10 features lands in the high-0.2s/low-0.3s
    # R^2 range. Rather than guess a bare threshold, assert PARITY: our forest must land within a
    # small margin of sklearn's same-family ensemble with matched settings (a correctness check,
    # not byte-identical output).
    from sklearn import ensemble

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    ours = cls(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    theirs = getattr(ensemble, sk_cls)(n_estimators=100, random_state=0, max_features="sqrt").fit(X_tr, y_tr)
    ours_r2 = _p.r2_score(y_te, ours.predict(X_te))
    theirs_r2 = _p.r2_score(y_te, theirs.predict(X_te))
    assert ours_r2 > 0.25
    assert ours_r2 > theirs_r2 - 0.1


def test_forest_reduces_variance_vs_single_tree() -> None:
    # A bagged ensemble should generalize at least as well as a single deep tree on a
    # noisy regression target (lower held-out error on average).
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(400, 3))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2 + rng.normal(scale=0.3, size=400)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    from dscraft.automl._tree import DecisionTree

    single = DecisionTree(criterion="mse", max_depth=None).fit(X_tr, y_tr)
    forest = DSRandomForestRegressor(n_estimators=80, random_state=0).fit(X_tr, y_tr)
    single_mse = _p.mean_squared_error(y_te, single.predict(X_te))
    forest_mse = _p.mean_squared_error(y_te, forest.predict(X_te))
    assert forest_mse < single_mse


# ---------------------------------------------------------------------------
# Bootstrap vs. extra-trees behaviour + reproducibility
# ---------------------------------------------------------------------------


def test_extra_trees_defaults_differ_from_forest() -> None:
    et = DSExtraTreesRegressor()
    rf = DSRandomForestRegressor()
    assert et._resolved_bootstrap() is False and et._default_splitter == "random"
    assert rf._resolved_bootstrap() is True and rf._default_splitter == "best"


def test_bootstrap_override_is_respected() -> None:
    # A random forest with bootstrap explicitly disabled should not resample rows.
    forest = DSRandomForestRegressor(bootstrap=False)
    assert forest._resolved_bootstrap() is False


def test_fit_is_reproducible_with_seed() -> None:
    X, y = load_iris(return_X_y=True)
    a = DSRandomForestClassifier(n_estimators=25, random_state=7).fit(X, y)
    b = DSRandomForestClassifier(n_estimators=25, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_n_estimators_validation() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        DSRandomForestRegressor(n_estimators=0).fit(np.zeros((4, 1)), np.arange(4.0))


# ---------------------------------------------------------------------------
# sklearn API compatibility
# ---------------------------------------------------------------------------


def test_forest_clonable_and_pipeline() -> None:
    assert clone(DSRandomForestClassifier(n_estimators=10)).get_params()["n_estimators"] == 10
    X, y = load_breast_cancer(return_X_y=True)
    pipe = Pipeline([("scale", StandardScaler()), ("rf", DSRandomForestClassifier(n_estimators=30, random_state=0))])
    pipe.fit(X, y)
    assert _p.accuracy_score(y, pipe.predict(X)) > 0.95
    assert pipe.named_steps["rf"].n_features_in_ == X.shape[1]


def test_forest_predict_feature_mismatch() -> None:
    model = DSRandomForestRegressor(n_estimators=5, random_state=0).fit(np.zeros((6, 2)), np.arange(6.0))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))
