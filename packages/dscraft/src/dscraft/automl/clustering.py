"""Unsupervised clustering backends for `dscraft.automl`.

`dscraft.automl` is otherwise scoped as supervised tabular AutoML (model
backends in `models.py`, ONNX export in `compile.py`); this module adds an
unsupervised clustering allowlist, independent of the supervised
model-selection surface -- clustering doesn't need a fitted classifier or
regressor to make sense on its own.

Per this repo's multi-backend design principle (CLAUDE.md: "when multiple
options serve the same purpose, expose ALL of them as selectable options
via an allowlist-style dispatch, never hard-code or pick a single
'winner'"), this uses the same `SUPPORTED_*`-dict-allowlist pattern used by
`models.py`/`dscraft.forecast.forecast.SUPPORTED_MODELS`, and now spans the
major clustering *styles* rather than a single backend.

Every clusterer is a **first-party** implementation
(`dscraft.automl._clustering`, built from scratch on numpy): `DSKMeans`
(centroid-based), `DSDBSCAN` (density-based), `DSAgglomerative`
(hierarchical agglomerative), and `DSHDBSCAN` (hierarchical density-based,
HDBSCAN*). The first-party automl/eda rebuild (see
`docs/decisions/2026-10-first-party-automl-eda-rebuild.md`) replaced the
`sklearn.cluster.HDBSCAN` backend this module originally wrapped with the
first-party `DSHDBSCAN`; the old `"HDBSCAN"` allowlist name is kept as a
backward-compatible alias of `"DSHDBSCAN"`. No clustering algorithm library
is a dependency here (scikit-learn remains only for its estimator-protocol
plumbing, `ClusterMixin`).
"""

from __future__ import annotations

from typing import Any

from sklearn.base import ClusterMixin

from dscraft.automl._clustering import DSAgglomerative, DSDBSCAN, DSHDBSCAN, DSKMeans

__all__ = [
    "SUPPORTED_CLUSTERERS",
    "build_clusterer",
]

#: Clustering backends this module supports, keyed by caller-facing name.
#: All are **first-party** implementations (`dscraft.automl._clustering`,
#: built from scratch on numpy), covering the major clustering *styles* per
#: the first-party automl/eda rebuild (see
#: docs/decisions/2026-10-first-party-automl-eda-rebuild.md):
#:
#: - ``"DSKMeans"`` -- centroid-based (Lloyd + k-means++)
#: - ``"DSDBSCAN"`` -- density-based (core/border/noise)
#: - ``"DSAgglomerative"`` -- hierarchical agglomerative (ward/complete/average/single linkage)
#: - ``"DSHDBSCAN"`` -- hierarchical density-based (HDBSCAN*), the first-party
#:   replacement for the `sklearn.cluster.HDBSCAN` backend this allowlist
#:   originally wrapped. ``"HDBSCAN"`` is kept as a backward-compatible alias
#:   of ``"DSHDBSCAN"`` so existing callers of the old name keep working.
SUPPORTED_CLUSTERERS: dict[str, type] = {
    "DSKMeans": DSKMeans,
    "DSDBSCAN": DSDBSCAN,
    "DSAgglomerative": DSAgglomerative,
    "DSHDBSCAN": DSHDBSCAN,
    "HDBSCAN": DSHDBSCAN,  # backward-compatible alias of the first-party DSHDBSCAN
}


def build_clusterer(name: str, **kwargs: Any) -> ClusterMixin:
    """Build an unfitted, sklearn-compatible clustering estimator.

    This is the one canonical clustering-backend factory for
    `dscraft.automl` -- no parallel clustering entrypoint exists elsewhere
    in this package.

    Args:
        name: which backend to build -- a key of :data:`SUPPORTED_CLUSTERERS`
            (currently only ``"HDBSCAN"``).
        **kwargs: forwarded straight through to the selected estimator's
            constructor (e.g. ``min_cluster_size``, ``min_samples``).

    Returns:
        An unfitted instance of the selected clusterer class. Call
        `.fit(X)` / `.fit_predict(X)` on it yourself.

    Raises:
        ValueError: ``name`` is not a key of :data:`SUPPORTED_CLUSTERERS`.
    """
    clusterer_cls = SUPPORTED_CLUSTERERS.get(name)
    if clusterer_cls is None:
        raise ValueError(
            f"Unsupported clusterer {name!r}. "
            f"Supported: {sorted(SUPPORTED_CLUSTERERS)!r}."
        )

    return clusterer_cls(**kwargs)
