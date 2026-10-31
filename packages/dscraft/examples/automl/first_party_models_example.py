"""Runnable demo of dscraft.automl's first-party model/resampler/clusterer backends.

Shows the first-party automl backends -- all built from scratch on numpy, no xgboost/lightgbm/
catboost/imbalanced-learn/sklearn-cluster dependency -- working end to end through the public
`build_model` / `build_resampler` / `build_clusterer` factories:

1. Train each first-party classifier (DSLogistic / DSBoost / DSRandomForest / DSExtraTrees) on
   Iris and report test accuracy.
2. Rebalance a synthetic imbalanced set with the first-party SMOTE, then fit a first-party
   classifier on the balanced data.
3. Cluster blobs with the first-party DSKMeans and DSHDBSCAN.

This script only *calls* the package's public API -- it does not reimplement any model, resampling,
or clustering logic inline (per CLAUDE.md's "no net-new scripts" rule).

Run with:
    python packages/dscraft/examples/automl/first_party_models_example.py

Requires the `automl` extra (numpy/pandas/scikit-learn only -- no boosting library):
    pip install -e "packages/dscraft[automl]"
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.model_selection import train_test_split

from dscraft.automl import build_clusterer, build_model, build_resampler


def demo_classifiers() -> None:
    """Train every first-party classifier backend on Iris and print test accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        *load_iris(return_X_y=True), test_size=0.3, random_state=0, stratify=load_iris().target
    )
    print("First-party classifier backends on Iris (test accuracy):")
    for name in ("DSLogistic", "DSBoost", "DSRandomForest", "DSExtraTrees"):
        kwargs = {"n_estimators": 50, "random_state": 0} if name != "DSLogistic" else {"C": 10.0}
        model = build_model(name, "classification", **kwargs)
        model.fit(X_train, y_train)
        acc = np.mean(model.predict(X_test) == y_test)
        print(f"  {name:16s} accuracy={acc:.3f}")


def demo_resampling() -> None:
    """Rebalance a skewed dataset with first-party SMOTE, then fit a classifier on it."""
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, weights=[0.9, 0.1], random_state=0
    )
    _vals, before = np.unique(y, return_counts=True)
    X_res, y_res = build_resampler("SMOTE", random_state=0).fit_resample(X, y)
    _vals, after = np.unique(y_res, return_counts=True)
    print("\nFirst-party SMOTE resampling:")
    print(f"  class counts before: {before.tolist()}  after: {after.tolist()}")
    clf = build_model("DSBoost", "classification", n_estimators=40, random_state=0)
    clf.fit(X_res, y_res)
    print(f"  trained DSBoost on the balanced set ({len(X_res)} rows)")


def demo_clustering() -> None:
    """Cluster well-separated blobs with first-party DSKMeans and DSHDBSCAN."""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    kmeans_labels = build_clusterer("DSKMeans", n_clusters=3, random_state=0).fit_predict(X)
    hdbscan_labels = build_clusterer("DSHDBSCAN", min_cluster_size=15).fit_predict(X)
    print("\nFirst-party clustering on 3 blobs:")
    print(f"  DSKMeans  -> {len(set(kmeans_labels))} clusters")
    print(
        f"  DSHDBSCAN -> {len(set(hdbscan_labels) - {-1})} clusters, "
        f"{int((hdbscan_labels == -1).sum())} noise points"
    )


def main() -> None:
    demo_classifiers()
    demo_resampling()
    demo_clustering()
    print("\nAll first-party automl backends ran with no third-party ML-algorithm library.")


if __name__ == "__main__":
    main()
