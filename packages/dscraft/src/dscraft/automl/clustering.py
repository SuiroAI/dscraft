"""Density-based clustering backends for `dscraft.automl`.

`dscraft.automl` is otherwise scoped purely as supervised tabular AutoML
(model backends in `models.py`, ONNX export in `compile.py`); this module
adds an unsupervised clustering allowlist, independent of the supervised
model-selection surface -- clustering doesn't need a fitted classifier or
regressor to make sense on its own.

Per this repo's multi-backend design principle (CLAUDE.md: "when multiple
libraries serve the same purpose, expose ALL of them as selectable
options via an allowlist-style dispatch, never hard-code or pick a single
'winner'"), this establishes the same `SUPPORTED_*`-dict-allowlist pattern
used by `models.py`/`dscraft.forecast.forecast.SUPPORTED_MODELS`, even
though HDBSCAN is (for now) its only entry -- the point is that the
allowlist dispatch mechanism is in place for future clustering backends,
not that every possible clusterer ships today.

HDBSCAN is exposed via scikit-learn's own built-in
`sklearn.cluster.HDBSCAN` (available since scikit-learn 1.3, already this
extra's floor -- see `pyproject.toml`'s `automl` extra), not the separate
`hdbscan` PyPI package, since the built-in implementation is present and
working with no additional dependency required.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import ClusterMixin
from sklearn.cluster import HDBSCAN

__all__ = [
    "SUPPORTED_CLUSTERERS",
    "build_clusterer",
]

#: Clustering backends this module supports, keyed by caller-facing name.
#: HDBSCAN is the only entry today; the allowlist pattern itself (not an
#: exhaustive set of backends) is what this module establishes, per the
#: multi-backend design principle.
SUPPORTED_CLUSTERERS: dict[str, type] = {
    "HDBSCAN": HDBSCAN,
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
