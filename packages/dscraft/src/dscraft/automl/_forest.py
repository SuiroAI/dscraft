"""First-party bagging-ensemble (random forest / extra-trees) estimators for ``dscraft.automl``.

This internal module implements the bagging family of tree ensembles from scratch, as the
first-party counterpart to the boosting family in :mod:`dscraft.automl._boosting`. Together they
give ``dscraft.automl`` both major tree-ensemble *styles* -- sequential boosting and parallel
bagging -- entirely in-house, built on the same first-party CART (:mod:`dscraft.automl._tree`) and
primitives (:mod:`dscraft.automl._primitives`), with no third-party ensemble library.

Two variants, selected by the ``bootstrap``/``splitter`` combination:

- **Random forest** (``bootstrap=True``, ``splitter="best"``): each tree is grown on a bootstrap
  resample of the rows and considers a random ``max_features`` subset at every split, evaluating
  the *best* threshold among them -- Breiman's random forest.
- **Extra-trees** (``bootstrap=False``, ``splitter="random"``): each tree is grown on the full
  (un-resampled) data but picks a *random* threshold per considered feature -- the
  extremely-randomized-trees rule, which trades a little per-tree fit for lower variance and
  faster training.

Both predict by averaging across trees: class-probability averaging (soft voting) for
classification, mean prediction for regression. Each tree gets its own child ``Generator`` derived
from the ensemble's ``random_state`` so the whole forest is reproducible from one seed while every
tree still sees independent randomness.

Both estimators are **scikit-learn API-compatible** (same contract as the linear and boosting
estimators), so they drop into an ``sklearn.pipeline.Pipeline`` and ``dscraft.automl.compile()``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from dscraft.automl import _primitives as _p
from dscraft.automl._tree import DecisionTree

__all__ = [
    "DSRandomForestClassifier",
    "DSRandomForestRegressor",
    "DSExtraTreesClassifier",
    "DSExtraTreesRegressor",
]


class _BaseForest(BaseEstimator):
    """Shared machinery for the bagging-ensemble estimators.

    Holds the common hyperparameters and the per-tree build loop; subclasses set the
    ``_is_classifier``/``_default_bootstrap``/``_default_splitter`` class attributes and the
    tree criterion, and implement aggregation.
    """

    _is_classifier: bool = False
    _default_bootstrap: bool = True
    _default_splitter: str = "best"
    _criterion: str = "mse"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features="sqrt",
        bootstrap: Optional[bool] = None,
        random_state=None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _resolved_bootstrap(self) -> bool:
        return self._default_bootstrap if self.bootstrap is None else self.bootstrap

    def _fit_forest(self, X, y, *, n_classes: Optional[int]) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        rng = _p.as_generator(self.random_state)
        n_samples = X.shape[0]
        bootstrap = self._resolved_bootstrap()
        # One independent child seed per tree, derived from the ensemble RNG -> reproducible.
        child_seeds = rng.integers(0, 2**32 - 1, size=self.n_estimators)

        self.estimators_ = []
        for seed in child_seeds:
            tree_rng = np.random.default_rng(int(seed))
            if bootstrap:
                sample_idx = tree_rng.integers(0, n_samples, size=n_samples)  # with replacement
            else:
                sample_idx = np.arange(n_samples)
            tree = DecisionTree(
                criterion=self._criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                splitter=self._default_splitter,
                random_state=tree_rng,
            )
            if self._is_classifier:
                tree.fit(X[sample_idx], y[sample_idx], n_classes=n_classes)
            else:
                tree.fit(X[sample_idx], y[sample_idx])
            self.estimators_.append(tree)


class _BaseForestClassifier(ClassifierMixin, _BaseForest):
    _is_classifier = True
    _criterion = "gini"

    def fit(self, X, y) -> "_BaseForestClassifier":
        """Fit the ensemble. Returns ``self``."""
        X, y = _p.check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("Classifier needs at least 2 classes in y.")
        self.n_features_in_ = X.shape[1]
        class_index = {cls: i for i, cls in enumerate(self.classes_)}
        y_idx = np.array([class_index[v] for v in y])
        self._fit_forest(X, y_idx, n_classes=self.classes_.shape[0])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Average the per-tree class-probability vectors (soft voting)."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        proba_sum = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.estimators_)

    def predict(self, X) -> np.ndarray:
        """Predict class labels (argmax of the averaged probabilities)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _BaseForestRegressor(RegressorMixin, _BaseForest):
    _is_classifier = False
    _criterion = "mse"

    def fit(self, X, y) -> "_BaseForestRegressor":
        """Fit the ensemble. Returns ``self``."""
        X, y = _p.check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]
        self._fit_forest(X, y, n_classes=None)
        return self

    def predict(self, X) -> np.ndarray:
        """Average the per-tree scalar predictions."""
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        total = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.estimators_:
            total += tree.predict(X)
        return total / len(self.estimators_)


class DSRandomForestClassifier(_BaseForestClassifier):
    """First-party random forest classifier (bootstrap rows + best split over a feature subset)."""

    _default_bootstrap = True
    _default_splitter = "best"


class DSRandomForestRegressor(_BaseForestRegressor):
    """First-party random forest regressor (bootstrap rows + best split over a feature subset)."""

    _default_bootstrap = True
    _default_splitter = "best"


class DSExtraTreesClassifier(_BaseForestClassifier):
    """First-party extra-trees classifier (full data + one random threshold per considered feature)."""

    _default_bootstrap = False
    _default_splitter = "random"


class DSExtraTreesRegressor(_BaseForestRegressor):
    """First-party extra-trees regressor (full data + one random threshold per considered feature)."""

    _default_bootstrap = False
    _default_splitter = "random"
