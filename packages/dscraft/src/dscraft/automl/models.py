"""First-party model backends for `dscraft.automl`.

This module exposes `dscraft.automl`'s caller-selectable model backends,
all of which are now **first-party** implementations built from scratch on
numpy (see the `DS*` classes in `_linear.py`, `_boosting.py`, and
`_forest.py`, which in turn build on `_tree.py` and `_primitives.py`). The
third-party XGBoost / LightGBM / CatBoost backends this module originally
wrapped have been removed as part of the first-party automl/eda rebuild
(see `docs/decisions/2026-10-first-party-automl-eda-rebuild.md`): the
modeling, boosting, and techniques are now our own end to end.

This keeps this repo's multi-backend design principle (CLAUDE.md: "when
multiple options serve the same purpose, expose ALL of them as selectable
options via an allowlist-style dispatch, never hard-code or pick a single
'winner'") -- the allowlist still offers several equally-supported model
*families* (linear, gradient boosting, random forest, extra-trees) with no
single default; the rebuild only changed the *implementations* behind the
allowlist from external libraries to in-house ones.

This mirrors `dscraft.forecast.forecast`'s established
`SUPPORTED_MODELS`-dict-allowlist pattern (see that module's
`SUPPORTED_MODELS` and `ForecastConfig.__post_init__`): a module-level
dict constant maps a caller-facing model name to a class, and a factory
function raises a clear `ValueError` listing the valid names on an
unknown one.

Scope: this module only builds an unfitted, sklearn-compatible estimator
instance for the caller to `.fit()` themselves -- it does not implement
model selection, hyperparameter search, or the streaming `partial_fit`
evaluator / PSI drift detection named elsewhere in the architecture doc
(both explicitly out of scope for this package, per
`dscraft/automl/__init__.py`'s module docstring).
"""

from __future__ import annotations

from typing import Any

from dscraft.automl._boosting import (
    DSBoostClassifier,
    DSBoostRegressor,
    DSQuantileBoostRegressor,
)
from dscraft.automl._forest import (
    DSExtraTreesClassifier,
    DSExtraTreesRegressor,
    DSRandomForestClassifier,
    DSRandomForestRegressor,
)
from dscraft.automl._linear import DSLinearRegressor, DSLogisticClassifier

__all__ = [
    "SUPPORTED_CLASSIFIERS",
    "SUPPORTED_REGRESSORS",
    "build_model",
]

#: Concrete classifier estimator types this module builds -- all first-party
#: `DS*` classes (see `_linear.py`/`_boosting.py`/`_forest.py`).
_ClassifierType = (
    DSLogisticClassifier
    | DSBoostClassifier
    | DSRandomForestClassifier
    | DSExtraTreesClassifier
)
_RegressorType = (
    DSLinearRegressor
    | DSBoostRegressor
    | DSQuantileBoostRegressor
    | DSRandomForestRegressor
    | DSExtraTreesRegressor
)

#: Classification backends this module supports, keyed by caller-facing
#: name. All are equally-supported, first-party options (multi-backend
#: design principle) -- none is a "default"; the caller names one
#: explicitly via :func:`build_model`:
#:
#: - ``"DSLogistic"`` -- logistic regression (`_linear.DSLogisticClassifier`)
#: - ``"DSBoost"`` -- gradient-boosted trees (`_boosting.DSBoostClassifier`)
#: - ``"DSRandomForest"`` -- random forest (`_forest.DSRandomForestClassifier`)
#: - ``"DSExtraTrees"`` -- extremely-randomized trees (`_forest.DSExtraTreesClassifier`)
#:
#: The third-party XGBoost/LightGBM/CatBoost backends this allowlist
#: originally wrapped were removed by the first-party automl/eda rebuild
#: (see docs/decisions/2026-10-first-party-automl-eda-rebuild.md), their
#: capability now provided by the first-party `"DSBoost"` backend.
SUPPORTED_CLASSIFIERS: dict[str, type[_ClassifierType]] = {
    "DSLogistic": DSLogisticClassifier,
    "DSBoost": DSBoostClassifier,
    "DSRandomForest": DSRandomForestClassifier,
    "DSExtraTrees": DSExtraTreesClassifier,
}

#: Regression backends this module supports, keyed by caller-facing name.
#: Same first-party, multi-backend posture as :data:`SUPPORTED_CLASSIFIERS`:
#:
#: - ``"DSLinear"`` -- OLS/ridge/lasso/elastic-net (`_linear.DSLinearRegressor`)
#: - ``"DSBoost"`` -- gradient-boosted trees (`_boosting.DSBoostRegressor`)
#: - ``"DSQuantileBoost"`` -- first-party gradient-boosted *quantile* regressor
#:   (`_boosting.DSQuantileBoostRegressor`), pinball loss at a caller-supplied
#:   ``alpha`` in (0, 1). This is the first-party replacement for the removed
#:   third-party ``objective="quantile"`` backend; SuiroAI's seeded quantile
#:   forecasting consumes it via this public factory.
#: - ``"DSRandomForest"`` -- random forest (`_forest.DSRandomForestRegressor`)
#: - ``"DSExtraTrees"`` -- extremely-randomized trees (`_forest.DSExtraTreesRegressor`)
SUPPORTED_REGRESSORS: dict[str, type[_RegressorType]] = {
    "DSLinear": DSLinearRegressor,
    "DSBoost": DSBoostRegressor,
    "DSQuantileBoost": DSQuantileBoostRegressor,
    "DSRandomForest": DSRandomForestRegressor,
    "DSExtraTrees": DSExtraTreesRegressor,
}

#: Valid ``task`` values for :func:`build_model`, mapped to the allowlist
#: dict each selects from.
_TASK_ALLOWLISTS: dict[str, dict[str, type[_ClassifierType]] | dict[str, type[_RegressorType]]] = {
    "classification": SUPPORTED_CLASSIFIERS,
    "regression": SUPPORTED_REGRESSORS,
}


def build_model(name: str, task: str, **kwargs: Any) -> _ClassifierType | _RegressorType:
    """Build an unfitted, sklearn-compatible first-party estimator.

    This is the one canonical model-backend factory for `dscraft.automl`
    (no parallel `AutoML`/`AutoMLBoosted` class exists elsewhere in this
    package, per the architecture doc's AutoML scope).

    Args:
        name: which backend to build -- one of :data:`SUPPORTED_CLASSIFIERS`
            / :data:`SUPPORTED_REGRESSORS`'s keys, depending on ``task``.
            For ``task="classification"``: ``"DSLogistic"``, ``"DSBoost"``,
            ``"DSRandomForest"``, or ``"DSExtraTrees"``. For
            ``task="regression"``: ``"DSLinear"``, ``"DSBoost"``,
            ``"DSRandomForest"``, or ``"DSExtraTrees"``. All are first-party
            backends (see the module docstring).
        task: ``"classification"`` or ``"regression"``.
        **kwargs: forwarded straight through to the selected estimator's
            constructor (e.g. ``n_estimators``, ``max_depth``,
            ``learning_rate`` for the ensemble backends; ``alpha``/
            ``l1_ratio`` for ``"DSLinear"``; ``C`` for ``"DSLogistic"``).

    Returns:
        An unfitted instance of the selected estimator class. Call
        `.fit(X, y)` on it yourself -- this module does not implement a
        training/evaluation loop.

    Raises:
        ValueError: ``task`` is not ``"classification"`` or
            ``"regression"``, or ``name`` is not a key of the allowlist
            selected by ``task``.
    """
    allowlist = _TASK_ALLOWLISTS.get(task)
    if allowlist is None:
        raise ValueError(
            f"Unsupported task {task!r}. Supported: {sorted(_TASK_ALLOWLISTS)!r}."
        )

    model_cls = allowlist.get(name)
    if model_cls is None:
        raise ValueError(
            f"Unsupported model {name!r} for task {task!r}. "
            f"Supported: {sorted(allowlist)!r}."
        )

    return model_cls(**kwargs)
