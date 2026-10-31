"""Tests for dscraft.automl._boosting.

Validates the first-party gradient boosting: it learns regression, binary, and multiclass
problems to a competitive accuracy, honours shrinkage / subsampling / early stopping, is
reproducible under a seed, is sklearn-API-compatible (clone + Pipeline + compile), and reaches
accuracy *parity* with the third-party XGBoost backend on standard datasets (a correctness check
on our from-scratch implementation, not an attempt at byte-identical output).
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
from dscraft.automl._boosting import DSBoostClassifier, DSBoostRegressor


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regressor_learns_diabetes() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    # Gentler depth/rate than the overfitting default: diabetes is noisy, and deep/aggressive
    # boosting overfits it (XGBoost scores the same ~0.24 with depth=3/lr=0.1/100 rounds -- see
    # the parity test). These params generalize, so the held-out R^2 is a meaningful floor.
    model = DSBoostRegressor(n_estimators=60, learning_rate=0.05, max_depth=2, random_state=0)
    model.fit(X_tr, y_tr)
    assert model.n_estimators_ == 60
    assert _p.r2_score(y_te, model.predict(X_te)) > 0.3


def test_regressor_fits_training_signal() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(300, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    model = DSBoostRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    assert _p.r2_score(y, model.predict(X)) > 0.9


def test_regressor_shrinkage_slows_fit() -> None:
    X, y = load_diabetes(return_X_y=True)
    fast = DSBoostRegressor(n_estimators=20, learning_rate=0.5, random_state=0).fit(X, y)
    slow = DSBoostRegressor(n_estimators=20, learning_rate=0.01, random_state=0).fit(X, y)
    # With the same few rounds, the tiny learning rate fits the training set less well.
    assert _p.mean_squared_error(y, slow.predict(X)) > _p.mean_squared_error(y, fast.predict(X))


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


def test_classifier_learns_breast_cancer_binary() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = DSBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
    assert _p.accuracy_score(y_te, model.predict(X_te)) > 0.93
    proba = model.predict_proba(X_te)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_te)))


def test_classifier_preserves_string_labels_binary() -> None:
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 0.5, size=(60, 2)), rng.normal(2, 0.5, size=(60, 2))])
    y = np.array(["no"] * 60 + ["yes"] * 60)
    model = DSBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
    assert set(np.unique(model.predict(X))).issubset({"no", "yes"})


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------


def test_classifier_learns_iris_multiclass() -> None:
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = DSBoostClassifier(n_estimators=80, learning_rate=0.2, max_depth=3, random_state=0).fit(X_tr, y_tr)
    assert model._multiclass
    assert _p.accuracy_score(y_te, model.predict(X_te)) > 0.9
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_te)), atol=1e-9)


# ---------------------------------------------------------------------------
# Subsampling / early stopping / reproducibility / validation
# ---------------------------------------------------------------------------


def test_subsample_runs_and_learns() -> None:
    X, y = load_diabetes(return_X_y=True)
    model = DSBoostRegressor(n_estimators=60, subsample=0.6, random_state=0).fit(X, y)
    assert _p.r2_score(y, model.predict(X)) > 0.5


def test_early_stopping_can_stop_before_cap() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = DSBoostClassifier(
        n_estimators=500, learning_rate=0.3, validation_fraction=0.2,
        n_iter_no_change=3, tol=1e-4, random_state=0,
    ).fit(X, y)
    # It should converge and stop well before 500 rounds on this easy dataset.
    assert model.n_estimators_ < 500


def test_fit_is_reproducible_with_seed() -> None:
    X, y = load_iris(return_X_y=True)
    a = DSBoostClassifier(n_estimators=30, subsample=0.7, random_state=42).fit(X, y)
    b = DSBoostClassifier(n_estimators=30, subsample=0.7, random_state=42).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("kwargs, match", [
    ({"n_estimators": 0}, "n_estimators"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"subsample": 0.0}, "subsample"),
    ({"subsample": 1.5}, "subsample"),
    ({"validation_fraction": 1.0}, "validation_fraction"),
])
def test_booster_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DSBoostRegressor(**kwargs).fit(np.zeros((5, 1)), np.arange(5.0))


def test_classifier_requires_two_classes() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        DSBoostClassifier().fit(np.zeros((4, 1)), np.zeros(4))


# ---------------------------------------------------------------------------
# sklearn API compatibility
# ---------------------------------------------------------------------------


def test_boosters_clonable_and_pipeline_and_feature_count() -> None:
    reg = DSBoostRegressor(n_estimators=10, max_depth=2)
    assert clone(reg).get_params()["n_estimators"] == 10

    X, y = load_breast_cancer(return_X_y=True)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", DSBoostClassifier(n_estimators=30, random_state=0))])
    pipe.fit(X, y)
    assert _p.accuracy_score(y, pipe.predict(X)) > 0.95
    assert pipe.named_steps["clf"].n_features_in_ == X.shape[1]


def test_predict_feature_mismatch_raises() -> None:
    model = DSBoostRegressor(n_estimators=5).fit(np.zeros((6, 2)), np.arange(6.0))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Parity with the (still-installed) third-party backend -- correctness check
# ---------------------------------------------------------------------------


def test_regression_parity_with_xgboost() -> None:
    """Our booster lands within a reasonable margin of XGBoost on diabetes (correctness check).

    XGBoost is no longer a declared dependency of dscraft (the first-party rebuild removed it),
    so this cross-check is skipped when xgboost is not installed -- it is a development-time
    correctness validation against the library we replaced, not a runtime requirement.
    """
    XGBRegressor = pytest.importorskip("xgboost").XGBRegressor

    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    ours = DSBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
    theirs = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
    ours_r2 = _p.r2_score(y_te, ours.predict(X_te))
    theirs_r2 = _p.r2_score(y_te, theirs.predict(X_te))
    # Not identical output -- but our R^2 should be in the same ballpark, not dramatically worse.
    assert ours_r2 > theirs_r2 - 0.15


def test_classification_parity_with_xgboost() -> None:
    """Our classifier is competitive with XGBoost on breast cancer (correctness check).

    Skipped when xgboost is absent -- see :func:`test_regression_parity_with_xgboost`.
    """
    XGBClassifier = pytest.importorskip("xgboost").XGBClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    ours = DSBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
    theirs = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
    ours_acc = _p.accuracy_score(y_te, ours.predict(X_te))
    theirs_acc = _p.accuracy_score(y_te, np.asarray(theirs.predict(X_te)))
    assert ours_acc > theirs_acc - 0.05
