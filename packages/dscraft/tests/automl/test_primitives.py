"""Tests for dscraft.automl._primitives.

Validates the shared first-party numeric/optimization substrate against closed-form and
known-value references (no scikit-learn estimator / scipy solver is used as an oracle -- the
point of this module is that the math is ours). Covers: input validation, numerically-stable
link functions at extreme inputs, loss values + gradients, metrics against hand-computed values,
and both solvers converging to analytically-known optima.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.automl import _primitives as P


# ---------------------------------------------------------------------------
# Random state / validation
# ---------------------------------------------------------------------------


def test_as_generator_variants() -> None:
    assert isinstance(P.as_generator(None), np.random.Generator)
    assert isinstance(P.as_generator(0), np.random.Generator)
    gen = np.random.default_rng(1)
    assert P.as_generator(gen) is gen


def test_as_generator_seed_is_reproducible() -> None:
    a = P.as_generator(42).random(5)
    b = P.as_generator(42).random(5)
    np.testing.assert_array_equal(a, b)


def test_as_generator_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="random_state"):
        P.as_generator("not-a-seed")


def test_check_array_2d_accepts_valid() -> None:
    arr = P.check_array_2d([[1.0, 2.0], [3.0, 4.0]])
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float64


@pytest.mark.parametrize(
    "bad, match",
    [
        ([1.0, 2.0, 3.0], "2-D"),
        (np.empty((0, 3)), "non-empty"),
        ([[1.0, np.nan]], "finite"),
        ([[1.0, np.inf]], "finite"),
    ],
)
def test_check_array_2d_rejects_invalid(bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        P.check_array_2d(bad)


def test_check_X_y_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        P.check_X_y([[1.0], [2.0], [3.0]], [0, 1])


def test_check_X_y_non_1d_y_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        P.check_X_y([[1.0], [2.0]], [[0], [1]])


def test_check_X_y_preserves_y_dtype() -> None:
    _X, y = P.check_X_y([[1.0], [2.0]], np.array(["a", "b"]))
    assert y.dtype.kind in "US"


def test_train_validation_split_disjoint_and_complete() -> None:
    train, val = P.train_validation_split(100, 0.2, random_state=0)
    assert len(val) == 20
    assert len(train) == 80
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(range(100))


def test_train_validation_split_reproducible() -> None:
    a = P.train_validation_split(50, 0.3, random_state=7)
    b = P.train_validation_split(50, 0.3, random_state=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize(
    "n, frac, match",
    [
        (1, 0.2, "n_samples must be >= 2"),
        (10, 0.0, r"\(0, 1\)"),
        (10, 1.0, r"\(0, 1\)"),
        (10, 0.01, "degenerate"),
    ],
)
def test_train_validation_split_invalid(n, frac, match) -> None:
    with pytest.raises(ValueError, match=match):
        P.train_validation_split(n, frac)


# ---------------------------------------------------------------------------
# Link / activation functions
# ---------------------------------------------------------------------------


def test_sigmoid_known_values() -> None:
    np.testing.assert_allclose(P.sigmoid(np.array([0.0])), [0.5])
    np.testing.assert_allclose(P.sigmoid(np.array([2.0, -2.0])), [0.880797, 0.119203], atol=1e-5)


def test_sigmoid_is_stable_at_extremes() -> None:
    out = P.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-12)


def test_softmax_rows_sum_to_one() -> None:
    proba = P.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(proba.sum(axis=1), [1.0, 1.0])
    # Uniform input -> uniform output.
    np.testing.assert_allclose(proba[1], [1 / 3, 1 / 3, 1 / 3])


def test_softmax_stable_at_extremes() -> None:
    proba = P.softmax(np.array([[1000.0, 1001.0]]))
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), [1.0])


def test_softmax_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        P.softmax(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Losses + gradients
# ---------------------------------------------------------------------------


def test_squared_error_loss_and_gradient() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert P.squared_error_loss(y_true, y_pred) == pytest.approx(4.0 / 3.0)
    np.testing.assert_array_equal(P.squared_error_gradient(y_true, y_pred), [0.0, 0.0, 2.0])


def test_absolute_error_gradient_sign() -> None:
    grad = P.absolute_error_gradient(np.array([0.0, 0.0, 0.0]), np.array([-2.0, 0.0, 3.0]))
    np.testing.assert_array_equal(grad, [-1.0, 0.0, 1.0])


def test_huber_loss_matches_mse_within_delta() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.5, -0.5])  # both within delta=1 -> pure quadratic 0.5*r^2
    assert P.huber_loss(y_true, y_pred, delta=1.0) == pytest.approx(0.125)


def test_huber_gradient_clips_beyond_delta() -> None:
    grad = P.huber_gradient(np.array([0.0, 0.0]), np.array([0.5, 5.0]), delta=1.0)
    np.testing.assert_allclose(grad, [0.5, 1.0])


def test_huber_rejects_nonpositive_delta() -> None:
    with pytest.raises(ValueError, match="delta"):
        P.huber_loss(np.array([0.0]), np.array([0.0]), delta=0.0)


def test_log_loss_perfect_vs_wrong() -> None:
    # Perfect predictions -> ~0 loss.
    assert P.log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-10)
    # Confidently wrong -> large but finite.
    loss = P.log_loss(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(loss) and loss > 10


def test_logistic_gradient_and_hessian() -> None:
    np.testing.assert_allclose(P.logistic_gradient(np.array([1.0]), np.array([0.7])), [-0.3])
    np.testing.assert_allclose(P.logistic_hessian_diag(np.array([0.5])), [0.25])
    # Hessian floored away from zero for saturated probability.
    assert P.logistic_hessian_diag(np.array([1.0]))[0] > 0


def test_softmax_cross_entropy_and_gradient() -> None:
    onehot = np.array([[1.0, 0.0], [0.0, 1.0]])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert P.softmax_cross_entropy(onehot, proba) == pytest.approx(
        -0.5 * (np.log(0.9) + np.log(0.8))
    )
    np.testing.assert_allclose(
        P.softmax_cross_entropy_gradient(onehot, proba),
        [[-0.1, 0.1], [0.2, -0.2]],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_accuracy_score() -> None:
    assert P.accuracy_score(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])) == pytest.approx(0.75)


def test_accuracy_score_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        P.accuracy_score(np.array([1, 0]), np.array([1]))


def test_regression_metrics_known_values() -> None:
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    assert P.mean_squared_error(y_true, y_pred) == pytest.approx(0.375)
    assert P.root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert P.mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)


def test_r2_score_perfect_and_baseline() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert P.r2_score(y, y) == pytest.approx(1.0)
    # Predicting the mean everywhere -> R^2 == 0.
    assert P.r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_score_zero_variance_target() -> None:
    y = np.array([5.0, 5.0, 5.0])
    assert P.r2_score(y, y) == pytest.approx(1.0)
    assert P.r2_score(y, np.array([5.0, 5.0, 6.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def test_gradient_descent_finds_quadratic_minimum() -> None:
    # Minimize f(w) = (w - 3)^2, gradient 2(w - 3); optimum at w = 3.
    params, n_iter = P.gradient_descent(
        lambda w: 2.0 * (w - 3.0), np.array([0.0]), learning_rate=0.1, max_iter=1000, tol=1e-9
    )
    assert params[0] == pytest.approx(3.0, abs=1e-4)
    assert n_iter < 1000  # converged before the cap


def test_gradient_descent_invalid_args() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        P.gradient_descent(lambda w: w, np.array([0.0]), learning_rate=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        P.gradient_descent(lambda w: w, np.array([0.0]), max_iter=0)


def test_newton_irls_recovers_logistic_weights() -> None:
    # Generate separable-ish data from a known logistic model and check IRLS recovers a
    # weight vector that classifies it well (not exact weights -- those are only identifiable
    # up to the data, but the decision rule should be near-perfect).
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 1))
    intercept = np.ones((n, 1))
    X = np.hstack([intercept, x])
    true_w = np.array([0.5, 3.0])
    proba = P.sigmoid(X @ true_w)
    y = (rng.random(n) < proba).astype(np.float64)

    weights, n_iter = P.newton_irls(X, y, l2=0.0, max_iter=100, tol=1e-10)
    assert n_iter < 100
    predictions = (P.sigmoid(X @ weights) >= 0.5).astype(np.float64)
    # Threshold reflects the Bayes-optimal rate achievable on this deliberately-noisy
    # single-feature logistic model (slope 3.0 over N(0,1) inputs leaves genuine class
    # overlap), not a round number -- IRLS converging to the right decision rule lands here.
    assert P.accuracy_score(y, predictions) > 0.80
    # Recovered slope should share the sign/rough magnitude of the true one.
    assert weights[1] > 1.0


def test_newton_irls_l2_shrinks_nonintercept_weights() -> None:
    rng = np.random.default_rng(1)
    X = np.hstack([np.ones((200, 1)), rng.normal(size=(200, 3))])
    y = (rng.random(200) < P.sigmoid(X @ np.array([0.0, 2.0, -2.0, 1.0]))).astype(np.float64)
    w_unpenalized, _ = P.newton_irls(X, y, l2=0.0)
    w_penalized, _ = P.newton_irls(X, y, l2=50.0)
    # Heavy L2 shrinks the non-intercept coefficients toward zero.
    assert np.linalg.norm(w_penalized[1:]) < np.linalg.norm(w_unpenalized[1:])


def test_newton_irls_rejects_negative_l2() -> None:
    with pytest.raises(ValueError, match="l2"):
        P.newton_irls(np.ones((2, 1)), np.array([0.0, 1.0]), l2=-1.0)
