"""Tests for dscraft.automl._linear.

Validates the first-party linear models against known/recoverable structure (no scikit-learn
estimator is used as an oracle): OLS recovers planted coefficients, ridge shrinks them, lasso
drives irrelevant ones to exactly zero, the models are sklearn-API-compatible (get_params /
set_params / clone), and logistic regression learns separable binary and multiclass problems.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dscraft.automl._linear import DSLinearRegressor, DSLogisticClassifier
from dscraft.automl import _primitives as _p


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def test_ols_recovers_planted_coefficients() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true_coef = np.array([2.0, -1.0, 0.5])
    y = X @ true_coef + 4.0  # intercept 4
    model = DSLinearRegressor(alpha=0.0).fit(X, y)
    np.testing.assert_allclose(model.coef_, true_coef, atol=1e-6)
    assert model.intercept_ == pytest.approx(4.0, abs=1e-6)


def test_ols_no_intercept() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([1.5, -2.0])
    model = DSLinearRegressor(alpha=0.0, fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, [1.5, -2.0], atol=1e-6)


def test_ridge_shrinks_coefficients_toward_zero() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 4))
    y = X @ np.array([3.0, 3.0, 3.0, 3.0]) + 1.0
    unpenalized = DSLinearRegressor(alpha=0.0).fit(X, y)
    ridged = DSLinearRegressor(alpha=50.0, l1_ratio=0.0).fit(X, y)
    assert np.linalg.norm(ridged.coef_) < np.linalg.norm(unpenalized.coef_)


def test_lasso_zeros_out_irrelevant_feature() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 3))
    # Feature 2 is pure noise w.r.t. y.
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + 3.0
    model = DSLinearRegressor(
        alpha=0.1, l1_ratio=1.0, max_iter=5000, learning_rate=0.05, tol=1e-8
    ).fit(X, y)
    # The irrelevant coefficient is driven to (near-)exactly zero by L1.
    assert abs(model.coef_[2]) < 1e-2
    # The real coefficients survive (shrunk, but clearly non-zero and correct-signed).
    assert model.coef_[0] > 1.0 and model.coef_[1] < -0.5


def test_elastic_net_runs_and_predicts() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(150, 5))
    y = X @ rng.normal(size=5) + rng.normal(scale=0.1, size=150)
    model = DSLinearRegressor(alpha=0.05, l1_ratio=0.5, max_iter=2000).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (150,)
    assert _p.r2_score(y, preds) > 0.8


@pytest.mark.parametrize("kwargs, match", [
    ({"alpha": -1.0}, "alpha"),
    ({"l1_ratio": 2.0}, "l1_ratio"),
])
def test_linear_regressor_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DSLinearRegressor(**kwargs).fit(np.zeros((3, 1)), np.array([0.0, 1.0, 2.0]))


def test_linear_predict_feature_mismatch() -> None:
    model = DSLinearRegressor().fit(np.zeros((4, 2)), np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Logistic classification
# ---------------------------------------------------------------------------


def test_logistic_learns_separable_binary() -> None:
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(-2, 0.5, size=(100, 2)), rng.normal(2, 0.5, size=(100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    model = DSLogisticClassifier(C=1.0).fit(X, y)
    assert _p.accuracy_score(y, model.predict(X)) > 0.97
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(200))


def test_logistic_preserves_string_labels() -> None:
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(-2, 0.5, size=(60, 2)), rng.normal(2, 0.5, size=(60, 2))])
    y = np.array(["neg"] * 60 + ["pos"] * 60)
    model = DSLogisticClassifier().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({"neg", "pos"})
    np.testing.assert_array_equal(model.classes_, ["neg", "pos"])


def test_logistic_multiclass_one_vs_rest() -> None:
    rng = np.random.default_rng(7)
    centers = np.array([[0, 0], [6, 6], [0, 6]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(80, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 80)
    model = DSLogisticClassifier(C=10.0).fit(X, y)
    assert model.classes_.shape[0] == 3
    assert model.coef_.shape == (3, 2)
    assert _p.accuracy_score(y, model.predict(X)) > 0.95
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(240), atol=1e-9)


def test_logistic_C_controls_regularization() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 3))
    y = (X @ np.array([3.0, -3.0, 2.0]) > 0).astype(int)
    weak_reg = DSLogisticClassifier(C=100.0).fit(X, y)   # large C -> weak penalty
    strong_reg = DSLogisticClassifier(C=0.01).fit(X, y)  # small C -> strong penalty
    assert np.linalg.norm(strong_reg.coef_) < np.linalg.norm(weak_reg.coef_)


def test_logistic_validation() -> None:
    with pytest.raises(ValueError, match="C must be positive"):
        DSLogisticClassifier(C=0.0).fit(np.zeros((2, 1)), np.array([0, 1]))
    with pytest.raises(ValueError, match="at least 2 classes"):
        DSLogisticClassifier().fit(np.zeros((3, 1)), np.array([0, 0, 0]))


# ---------------------------------------------------------------------------
# scikit-learn API compatibility (the contract that keeps compile() working)
# ---------------------------------------------------------------------------


def test_estimators_are_clonable_and_expose_params() -> None:
    reg = DSLinearRegressor(alpha=0.3, l1_ratio=0.5)
    params = reg.get_params()
    assert params["alpha"] == 0.3 and params["l1_ratio"] == 0.5
    reg2 = clone(reg)  # requires get_params/set_params to round-trip
    assert reg2.get_params() == params

    clf = DSLogisticClassifier(C=2.0)
    assert clone(clf).get_params()["C"] == 2.0


def test_linear_regressor_works_inside_sklearn_pipeline() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + 0.5
    pipe = Pipeline([("scale", StandardScaler()), ("model", DSLinearRegressor(alpha=0.0))])
    pipe.fit(X, y)
    assert _p.r2_score(y, pipe.predict(X)) > 0.99
    # The pipeline exposes n_features_in_ (needed by dscraft.automl.compile).
    assert pipe.named_steps["model"].n_features_in_ == 3


def test_logistic_works_inside_sklearn_pipeline() -> None:
    rng = np.random.default_rng(10)
    X = np.vstack([rng.normal(-2, 0.5, size=(80, 2)), rng.normal(2, 0.5, size=(80, 2))])
    y = np.array([0] * 80 + [1] * 80)
    pipe = Pipeline([("scale", StandardScaler()), ("model", DSLogisticClassifier())])
    pipe.fit(X, y)
    assert _p.accuracy_score(y, pipe.predict(X)) > 0.97
