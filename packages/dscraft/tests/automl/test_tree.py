"""Tests for dscraft.automl._tree.

Validates the first-party CART implementation: it learns separable classification and simple
regression patterns, honours its depth/leaf/impurity constraints, implements the gradient-boosting
second-order criterion correctly (leaf = Newton step), supports random-forest feature subsampling
and the extra-trees random splitter, and reproduces runs from a fixed seed. No scikit-learn tree
is used as an oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.automl._tree import DecisionTree


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"criterion": "nonsense"}, "criterion"),
        ({"splitter": "nope"}, "splitter"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_depth": 0}, "max_depth"),
    ],
)
def test_constructor_validation(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DecisionTree(**kwargs)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_learns_axis_aligned_rule() -> None:
    # A perfectly axis-aligned rule: class 1 iff feature 0 > 0.5.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 2))
    y = (X[:, 0] > 0.5).astype(np.int64)
    tree = DecisionTree(criterion="gini", max_depth=3, random_state=0)
    tree.fit(X, y, n_classes=2)
    assert tree.n_classes_ == 2
    assert np.mean(tree.predict(X) == y) == 1.0


def test_classification_predict_proba_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(100, 3))
    y = (X[:, 1] > 0.5).astype(np.int64)
    tree = DecisionTree(criterion="entropy", max_depth=4, random_state=1).fit(X, y, n_classes=2)
    proba = tree.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(100))


def test_entropy_and_gini_both_fit_three_class_blobs() -> None:
    rng = np.random.default_rng(2)
    centers = np.array([[0, 0], [5, 5], [0, 5]])
    X = np.vstack([c + rng.normal(scale=0.3, size=(60, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 60).astype(np.int64)
    for criterion in ("gini", "entropy"):
        tree = DecisionTree(criterion=criterion, max_depth=5, random_state=0).fit(X, y, n_classes=3)
        assert np.mean(tree.predict(X) == y) > 0.98


def test_classification_requires_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        DecisionTree(criterion="gini").fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regression_learns_step_function() -> None:
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.where(X[:, 0] < 5, 1.0, 3.0)
    tree = DecisionTree(criterion="mse", max_depth=2).fit(X, y)
    preds = tree.predict(X)
    # A depth-2 tree easily captures a single step; predictions land on the two levels.
    np.testing.assert_allclose(np.sort(np.unique(np.round(preds, 6))), [1.0, 3.0])


def test_regression_deep_tree_memorizes() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(80, 2))
    y = rng.normal(size=80)
    tree = DecisionTree(criterion="mse", max_depth=None, min_samples_leaf=1).fit(X, y)
    # With unlimited depth and distinct X rows, the tree can fit the training targets closely.
    assert np.mean((tree.predict(X) - y) ** 2) < 1e-6


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_max_depth_limits_leaf_depth() -> None:
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
    tree = DecisionTree(criterion="gini", max_depth=2, random_state=0).fit(X, y, n_classes=2)

    def max_leaf_depth(node, depth=0):
        if node.is_leaf:
            return depth
        return max(max_leaf_depth(node.left, depth + 1), max_leaf_depth(node.right, depth + 1))

    assert max_leaf_depth(tree.root_) <= 2


def test_min_samples_leaf_respected() -> None:
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(100, 2))
    y = rng.normal(size=100)
    tree = DecisionTree(criterion="mse", min_samples_leaf=20).fit(X, y)

    def check(node):
        if node.is_leaf:
            assert node.n_samples >= 1
            return
        assert node.left.n_samples >= 20 and node.right.n_samples >= 20
        check(node.left)
        check(node.right)

    check(tree.root_)


def test_min_samples_split_makes_small_node_a_leaf() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    tree = DecisionTree(criterion="mse", min_samples_split=10).fit(X, y)
    assert tree.root_.is_leaf
    assert tree.predict(X)[0] == pytest.approx(y.mean())


def test_constant_feature_yields_leaf() -> None:
    X = np.ones((10, 1))
    y = np.array([0.0] * 5 + [1.0] * 5)
    tree = DecisionTree(criterion="mse").fit(X, y)
    assert tree.root_.is_leaf  # no split possible on a constant feature


# ---------------------------------------------------------------------------
# Gradient (boosting) criterion
# ---------------------------------------------------------------------------


def test_gradient_leaf_value_is_newton_step() -> None:
    # Single leaf (max_depth guards against splitting a tiny constant-feature node): the leaf
    # value must equal -sum(g) / (sum(h) + lambda).
    X = np.ones((6, 1))  # constant feature -> forces a single leaf
    g = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 1.0])
    h = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lam = 2.0
    tree = DecisionTree(criterion="gradient", l2_regularization=lam).fit(
        X, np.zeros(6), gradients=g, hessians=h
    )
    expected = -g.sum() / (h.sum() + lam)
    assert tree.predict(X)[0] == pytest.approx(expected)


def test_gradient_criterion_splits_on_gain() -> None:
    # Gradients that clearly separate on feature 0: the tree should split and the two leaves
    # should have opposite-signed Newton values.
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    g = np.array([2.0, 2.0, 2.0, -2.0, -2.0, -2.0])
    h = np.ones(6)
    tree = DecisionTree(criterion="gradient", max_depth=1, l2_regularization=0.0).fit(
        X, np.zeros(6), gradients=g, hessians=h
    )
    assert not tree.root_.is_leaf
    left_val = tree.predict(np.array([[0.0]]))[0]
    right_val = tree.predict(np.array([[1.0]]))[0]
    assert left_val < 0 < right_val  # -mean(g) per leaf: -2 and +2


def test_gradient_requires_gradients() -> None:
    with pytest.raises(ValueError, match="gradients are required"):
        DecisionTree(criterion="gradient").fit(np.zeros((3, 1)), np.zeros(3))


def test_gradient_defaults_hessians_to_ones() -> None:
    X = np.ones((4, 1))
    g = np.array([1.0, 1.0, 1.0, 1.0])
    tree = DecisionTree(criterion="gradient").fit(X, np.zeros(4), gradients=g)  # no hessians
    # Newton step with h=1 each: -sum(g) / (n) = -4/4 = -1.
    assert tree.predict(X)[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Randomness: feature subsampling + extra-trees splitter
# ---------------------------------------------------------------------------


def test_max_features_sqrt_and_log2_resolve() -> None:
    tree = DecisionTree(criterion="gini", max_features="sqrt")
    assert tree._resolve_max_features(16) == 4
    tree2 = DecisionTree(criterion="gini", max_features="log2")
    assert tree2._resolve_max_features(8) == 3
    tree3 = DecisionTree(criterion="gini", max_features=3)
    assert tree3._resolve_max_features(10) == 3
    assert tree3._resolve_max_features(2) == 2  # capped at n_features


def test_random_splitter_still_learns_signal() -> None:
    rng = np.random.default_rng(6)
    X = rng.uniform(size=(300, 2))
    y = (X[:, 0] > 0.5).astype(np.int64)
    tree = DecisionTree(criterion="gini", splitter="random", max_depth=6, random_state=0)
    tree.fit(X, y, n_classes=2)
    # Random thresholds are weaker per-split, but a reasonably deep extra-tree still learns it.
    assert np.mean(tree.predict(X) == y) > 0.85


def test_fit_is_reproducible_with_seed() -> None:
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(150, 5))
    y = (X[:, 2] > 0.5).astype(np.int64)
    a = DecisionTree(criterion="gini", max_features="sqrt", random_state=123).fit(X, y, n_classes=2)
    b = DecisionTree(criterion="gini", max_features="sqrt", random_state=123).fit(X, y, n_classes=2)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# ---------------------------------------------------------------------------
# Prediction-time validation
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        DecisionTree(criterion="mse").predict(np.zeros((2, 1)))


def test_predict_feature_count_mismatch_raises() -> None:
    tree = DecisionTree(criterion="mse").fit(np.zeros((4, 2)), np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="features"):
        tree.predict(np.zeros((2, 3)))


def test_predict_proba_only_for_classification() -> None:
    tree = DecisionTree(criterion="mse").fit(np.zeros((4, 1)), np.zeros(4))
    with pytest.raises(ValueError, match="classification"):
        tree.predict_proba(np.zeros((2, 1)))
