"""First-party CART decision tree for ``dscraft.automl``.

This internal module implements Classification And Regression Trees (CART) from scratch on
``numpy`` only -- no ``scikit-learn`` tree, no third-party tree library. It is the shared
substrate two later first-party ensemble families build on:

- the random-forest / extra-trees ensembles (bagging many of these trees), and
- gradient boosting, which fits a *regression* tree at each round to the (negative) gradient of
  the loss -- optionally using second-order (Hessian) information for the leaf values.

To serve both the plain-fit and the boosting use cases from one implementation, the tree is
built around a single growth routine, :class:`DecisionTree`, parameterized by a *split criterion*:

- ``"gini"`` / ``"entropy"`` -- classification impurity over class labels.
- ``"mse"`` -- regression variance reduction (unweighted squared error).
- ``"gradient"`` -- the Newton/second-order objective gradient boosting needs: given per-sample
  gradients ``g`` and Hessians ``h``, each candidate split is scored by the regularized
  structure-score gain ``G^2 / (H + lambda)`` (the XGBoost-style gain formula, implemented here
  from the public objective, not adapted from any library's source), and a leaf's value is the
  Newton step ``-sum(g) / (sum(h) + lambda)``. This is what lets the gradient-boosting PR reuse
  this exact class rather than growing a second, parallel tree implementation.

Design choices:

- **Exact, not histogram, split finding (for v1).** For each candidate feature the samples are
  sorted once and every between-distinct-value threshold is evaluated via a single left-to-right
  prefix-sum sweep, so a feature's best split is found in ``O(n log n)`` (the sort) + ``O(n)`` (the
  sweep). Histogram binning (bucketing feature values to evaluate ``O(bins)`` thresholds instead of
  ``O(n)``) is the standard large-``n`` speedup and is noted as a future optimization in the
  rebuild ADR, but exact split finding is simpler, has no bin-count accuracy knob, and is correct
  by construction -- the right default for a first correct implementation.
- **Explicit, reproducible randomness.** The per-split random feature subsample (used by random
  forests) and the random-threshold mode (used by extra-trees) both draw from a caller-supplied
  ``numpy.random.Generator`` (see :mod:`dscraft.automl._primitives`'s ``as_generator``), never the
  global RNG.
- **Iterative (stack-based) growth, not recursion.** The tree is grown with an explicit work stack
  so a deep tree can never hit Python's recursion limit.

This is an internal module (leading-underscore name), not part of ``dscraft.automl``'s public
surface; the caller-facing estimators that wrap it live in ``models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dscraft.automl import _primitives as _p

__all__ = [
    "TreeNode",
    "DecisionTree",
]

#: Split criteria this tree understands. ``"gini"``/``"entropy"`` are classification criteria
#: (require integer-encoded class labels and ``n_classes``); ``"mse"`` is unweighted regression;
#: ``"gradient"`` is the second-order boosting objective (requires per-sample gradients/Hessians).
_CLASSIFICATION_CRITERIA = frozenset({"gini", "entropy"})
_REGRESSION_CRITERIA = frozenset({"mse", "gradient"})
_ALL_CRITERIA = _CLASSIFICATION_CRITERIA | _REGRESSION_CRITERIA


@dataclass
class TreeNode:
    """One node of a fitted :class:`DecisionTree`.

    A node is either internal (``feature_index``/``threshold`` set, ``left``/``right`` children)
    or a leaf (``value`` set, children ``None``). Leaf ``value`` is a class-probability vector for
    classification criteria, or a scalar prediction for regression/gradient criteria.
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[np.ndarray] = None
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (i.e. holds a prediction ``value``)."""
        return self.left is None and self.right is None


# Internal frame pushed on the growth stack: the sample indices at a node, the node object to
# fill in, and its depth.
@dataclass
class _GrowFrame:
    indices: np.ndarray
    node: TreeNode
    depth: int


class DecisionTree:
    """A from-scratch CART decision tree, usable for classification, regression, or boosting.

    One class serves every first-party tree-based estimator in ``dscraft.automl`` (random forest,
    extra-trees, gradient boosting) by selecting a split ``criterion`` and, for the boosting path,
    passing per-sample gradients/Hessians to :meth:`fit`.

    Args:
        criterion: one of ``"gini"``, ``"entropy"`` (classification), ``"mse"`` (regression), or
            ``"gradient"`` (second-order boosting objective).
        max_depth: maximum tree depth; ``None`` for unlimited (bounded only by the sample/leaf
            constraints below).
        min_samples_split: a node with fewer samples than this is never split (becomes a leaf).
        min_samples_leaf: a split is only accepted if *both* children have at least this many
            samples.
        min_impurity_decrease: a split is only accepted if it reduces impurity (or increases the
            boosting gain) by at least this much.
        max_features: number of features to consider at each split -- an int (exact count),
            ``"sqrt"``/``"log2"`` (the random-forest conventions), or ``None`` (all features).
            When fewer than all features are considered, the subset is drawn fresh per node from
            ``random_state``.
        splitter: ``"best"`` (evaluate every candidate threshold -- CART/random-forest) or
            ``"random"`` (pick one random threshold per considered feature -- the extra-trees
            rule). ``"random"`` still picks the *best among the randomly-chosen* per-feature
            thresholds.
        l2_regularization: the ``lambda`` in the ``"gradient"`` criterion's gain/leaf formulas;
            ignored by the other criteria.
        random_state: seed / ``numpy.random.Generator`` for feature subsampling and the random
            splitter (see :func:`dscraft.automl._primitives.as_generator`).

    Attributes (after :meth:`fit`):
        root_: the root :class:`TreeNode`.
        n_features_in_: number of input features.
        n_classes_: number of classes (classification criteria only; ``None`` otherwise).
    """

    def __init__(
        self,
        *,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        splitter: str = "best",
        l2_regularization: float = 0.0,
        random_state=None,
    ) -> None:
        if criterion not in _ALL_CRITERIA:
            raise ValueError(f"criterion must be one of {sorted(_ALL_CRITERIA)!r}, got {criterion!r}.")
        if splitter not in ("best", "random"):
            raise ValueError(f"splitter must be 'best' or 'random', got {splitter!r}.")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}.")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}.")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.splitter = splitter
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.root_: Optional[TreeNode] = None
        self.n_features_in_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_classes: Optional[int] = None,
        gradients: Optional[np.ndarray] = None,
        hessians: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        """Grow the tree.

        For ``"gini"``/``"entropy"``: ``y`` must be integer class labels in ``[0, n_classes)`` and
        ``n_classes`` must be given. For ``"mse"``: ``y`` is the regression target. For
        ``"gradient"``: ``gradients`` (and optionally ``hessians``) give the per-sample first-
        (and second-) order loss derivatives the tree fits against; ``y`` is ignored for leaf
        values (the leaf is the Newton step over the gradients/Hessians) but is still accepted for
        a uniform call signature.

        Returns ``self``.
        """
        X = _p.check_array_2d(X, name="X")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.criterion in _CLASSIFICATION_CRITERIA:
            if n_classes is None:
                raise ValueError("n_classes is required for a classification criterion.")
            y = np.asarray(y)
            if y.shape[0] != n_samples:
                raise ValueError("y must have one label per sample.")
            self.n_classes_ = int(n_classes)
            targets = y.astype(np.int64)
            grad = hess = None
        elif self.criterion == "mse":
            targets = np.asarray(y, dtype=np.float64)
            if targets.shape[0] != n_samples:
                raise ValueError("y must have one target per sample.")
            grad = hess = None
        else:  # "gradient"
            if gradients is None:
                raise ValueError("gradients are required for the 'gradient' criterion.")
            grad = np.asarray(gradients, dtype=np.float64)
            if grad.shape[0] != n_samples:
                raise ValueError("gradients must have one entry per sample.")
            hess = (
                np.ones(n_samples, dtype=np.float64)
                if hessians is None
                else np.asarray(hessians, dtype=np.float64)
            )
            if hess.shape[0] != n_samples:
                raise ValueError("hessians must have one entry per sample.")
            targets = None

        rng = _p.as_generator(self.random_state)
        n_features_to_try = self._resolve_max_features(n_features)

        self.root_ = TreeNode()
        stack = [_GrowFrame(indices=np.arange(n_samples), node=self.root_, depth=0)]
        while stack:
            frame = stack.pop()
            self._grow_node(frame, X, targets, grad, hess, rng, n_features_to_try, stack)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for ``X``.

        Returns a 1-D array of class indices (classification) or scalar predictions
        (regression/gradient). For classification, use :meth:`predict_proba` for probabilities.
        """
        proba_or_value = self._route(X)
        if self.criterion in _CLASSIFICATION_CRITERIA:
            return np.argmax(proba_or_value, axis=1)
        return proba_or_value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class-probability vectors for ``X`` (classification criteria only)."""
        if self.criterion not in _CLASSIFICATION_CRITERIA:
            raise ValueError("predict_proba is only defined for classification criteria.")
        return self._route(X)

    # -- growth internals ---------------------------------------------------

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"max_features string must be 'sqrt' or 'log2', got {mf!r}.")
        if isinstance(mf, (int, np.integer)):
            if mf < 1:
                raise ValueError(f"max_features int must be >= 1, got {mf}.")
            return min(int(mf), n_features)
        raise TypeError(f"max_features must be int, 'sqrt', 'log2', or None, got {type(mf).__name__}.")

    def _grow_node(self, frame, X, targets, grad, hess, rng, n_features_to_try, stack) -> None:
        indices = frame.indices
        node = frame.node
        node.n_samples = int(indices.shape[0])

        depth_ok = self.max_depth is None or frame.depth < self.max_depth
        enough_to_split = indices.shape[0] >= self.min_samples_split

        if not (depth_ok and enough_to_split):
            node.value = self._leaf_value(indices, targets, grad, hess)
            return

        split = self._find_best_split(indices, X, targets, grad, hess, rng, n_features_to_try)
        if split is None:
            node.value = self._leaf_value(indices, targets, grad, hess)
            return

        feature_index, threshold, left_indices, right_indices = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = TreeNode()
        node.right = TreeNode()
        stack.append(_GrowFrame(indices=left_indices, node=node.left, depth=frame.depth + 1))
        stack.append(_GrowFrame(indices=right_indices, node=node.right, depth=frame.depth + 1))

    def _leaf_value(self, indices, targets, grad, hess) -> np.ndarray:
        if self.criterion in _CLASSIFICATION_CRITERIA:
            counts = np.bincount(targets[indices], minlength=self.n_classes_).astype(np.float64)
            total = counts.sum()
            return counts / total if total > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
        if self.criterion == "mse":
            return np.asarray(float(np.mean(targets[indices])))
        # gradient: Newton leaf value -sum(g) / (sum(h) + lambda).
        g_sum = float(np.sum(grad[indices]))
        h_sum = float(np.sum(hess[indices]))
        return np.asarray(-g_sum / (h_sum + self.l2_regularization))

    def _find_best_split(self, indices, X, targets, grad, hess, rng, n_features_to_try):
        n_features = X.shape[1]
        if n_features_to_try < n_features:
            feature_candidates = rng.choice(n_features, size=n_features_to_try, replace=False)
        else:
            feature_candidates = np.arange(n_features)

        parent_impurity = self._node_impurity(indices, targets, grad, hess)
        best = None  # (gain, feature, threshold, left_idx, right_idx)

        for feature_index in feature_candidates:
            column = X[indices, feature_index]
            order = np.argsort(column, kind="mergesort")
            sorted_column = column[order]
            sorted_local = indices[order]

            # Candidate thresholds are midpoints between consecutive distinct values.
            distinct_boundaries = np.nonzero(np.diff(sorted_column) > 0)[0]
            if distinct_boundaries.size == 0:
                continue  # constant feature over this node -- no split possible

            if self.splitter == "random":
                # Extra-trees: one random threshold in the feature's (min, max) range.
                low, high = sorted_column[0], sorted_column[-1]
                rand_threshold = rng.uniform(low, high)
                # Find the split position implied by that threshold.
                split_positions = [int(np.searchsorted(sorted_column, rand_threshold, side="right")) - 1]
                split_positions = [p for p in split_positions if 0 <= p < sorted_column.size - 1]
            else:
                split_positions = distinct_boundaries.tolist()

            for position in split_positions:
                left_count = position + 1
                right_count = indices.shape[0] - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_local = sorted_local[: position + 1]
                right_local = sorted_local[position + 1 :]
                gain = parent_impurity - self._split_impurity(
                    left_local, right_local, targets, grad, hess
                )
                if gain > self.min_impurity_decrease and (best is None or gain > best[0]):
                    threshold = float((sorted_column[position] + sorted_column[position + 1]) / 2.0)
                    best = (gain, int(feature_index), threshold, left_local, right_local)

        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    # -- impurity / gain ----------------------------------------------------

    def _node_impurity(self, indices, targets, grad, hess) -> float:
        """Impurity (classification/mse) or negative structure-score (gradient) of one node.

        For the ``"gradient"`` criterion this returns the *negative* structure score so that the
        same ``parent - children`` "gain = reduction" convention used by the impurity criteria
        yields the XGBoost-style gain for a split.
        """
        if self.criterion in _CLASSIFICATION_CRITERIA:
            counts = np.bincount(targets[indices], minlength=self.n_classes_).astype(np.float64)
            proba = counts / counts.sum()
            if self.criterion == "gini":
                return float(1.0 - np.sum(proba**2))
            nonzero = proba[proba > 0]
            return float(-np.sum(nonzero * np.log2(nonzero)))
        if self.criterion == "mse":
            values = targets[indices]
            return float(np.var(values)) * values.shape[0]  # weighted by count for additive gain
        g_sum = float(np.sum(grad[indices]))
        h_sum = float(np.sum(hess[indices]))
        return -(g_sum**2) / (h_sum + self.l2_regularization)

    def _split_impurity(self, left_indices, right_indices, targets, grad, hess) -> float:
        if self.criterion in _CLASSIFICATION_CRITERIA:
            n_left, n_right = left_indices.shape[0], right_indices.shape[0]
            total = n_left + n_right
            left_imp = self._node_impurity(left_indices, targets, grad, hess)
            right_imp = self._node_impurity(right_indices, targets, grad, hess)
            return (n_left / total) * left_imp + (n_right / total) * right_imp
        # mse/gradient already return count-weighted / structure-score quantities that add.
        return self._node_impurity(left_indices, targets, grad, hess) + self._node_impurity(
            right_indices, targets, grad, hess
        )

    # -- prediction routing -------------------------------------------------

    def _route(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("This DecisionTree is not fitted yet; call fit() first.")
        X = _p.check_array_2d(X, name="X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this tree was fit on {self.n_features_in_}."
            )
        outputs = [self._route_one(row) for row in X]
        return np.asarray(outputs)

    def _route_one(self, row: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
