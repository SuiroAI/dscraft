"""Pluggable gradient-boosted-tree model backends for `dscraft.automl`.

Per the architecture doc's AutoML module entry and this repo's
multi-backend design principle (CLAUDE.md: "when multiple libraries serve
the same purpose, expose ALL of them as selectable options via an
allowlist-style dispatch, never hard-code or pick a single 'winner'"),
this module adds XGBoost, LightGBM, and CatBoost as equally-supported,
caller-selectable model backends alongside scikit-learn's own estimators
-- it does not pick a "default" or "recommended" one among the three.

This mirrors `dscraft.forecast.forecast`'s established
`SUPPORTED_MODELS`-dict-allowlist pattern (see that module's
`SUPPORTED_MODELS` and `ForecastConfig.__post_init__`): a module-level
dict constant maps a caller-facing model name to a class, and a factory
function raises a clear `ValueError` listing the valid names on an
unknown one.

Unlike `compile.py`'s ONNX stack, XGBoost/LightGBM/CatBoost are base
runtime dependencies of the `automl` extra (see `pyproject.toml`'s
`automl` extra and its inline comment), so they are imported at module
level here, the same way `compile.py` imports `sklearn.pipeline.Pipeline`
directly rather than lazily.

Scope: this module only builds an unfitted, sklearn-compatible estimator
instance for the caller to `.fit()` themselves -- it does not implement
model selection, hyperparameter search, or the streaming `partial_fit`
evaluator / PSI drift detection named elsewhere in the architecture doc
(both explicitly out of scope for this package, per
`dscraft/automl/__init__.py`'s module docstring).
"""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

__all__ = [
    "SUPPORTED_CLASSIFIERS",
    "SUPPORTED_REGRESSORS",
    "build_model",
]

#: Concrete estimator types this module ever builds. CatBoost's estimators
#: do not reliably inherit from `sklearn.base.BaseEstimator` across
#: catboost versions, so this module widens to an explicit union of the
#: six concrete classes actually returned, rather than annotating against
#: `BaseEstimator`.
_ClassifierType = XGBClassifier | LGBMClassifier | CatBoostClassifier
_RegressorType = XGBRegressor | LGBMRegressor | CatBoostRegressor

#: Classification backends this module supports, keyed by caller-facing
#: name. All three are equally-supported options (multi-backend design
#: principle) -- none is a "default"; the caller must name one explicitly
#: via :func:`build_model`.
SUPPORTED_CLASSIFIERS: dict[str, type[_ClassifierType]] = {
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "CatBoost": CatBoostClassifier,
}

#: Regression backends this module supports, keyed by caller-facing name.
#: Same multi-backend posture as :data:`SUPPORTED_CLASSIFIERS`.
SUPPORTED_REGRESSORS: dict[str, type[_RegressorType]] = {
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "CatBoost": CatBoostRegressor,
}

#: Valid ``task`` values for :func:`build_model`, mapped to the allowlist
#: dict each selects from.
_TASK_ALLOWLISTS: dict[str, dict[str, type[_ClassifierType]] | dict[str, type[_RegressorType]]] = {
    "classification": SUPPORTED_CLASSIFIERS,
    "regression": SUPPORTED_REGRESSORS,
}


def build_model(name: str, task: str, **kwargs: Any) -> _ClassifierType | _RegressorType:
    """Build an unfitted, sklearn-compatible gradient-boosted-tree estimator.

    This is the one canonical model-backend factory for `dscraft.automl`
    (no parallel `AutoML`/`AutoMLBoosted` class exists elsewhere in this
    package, per the architecture doc's AutoML scope).

    Args:
        name: which backend to build -- one of :data:`SUPPORTED_CLASSIFIERS`
            / :data:`SUPPORTED_REGRESSORS`'s keys (``"XGBoost"``,
            ``"LightGBM"``, or ``"CatBoost"``), depending on ``task``.
        task: ``"classification"`` or ``"regression"``.
        **kwargs: forwarded straight through to the selected estimator's
            constructor (e.g. ``n_estimators``, ``max_depth``,
            ``learning_rate``).

    Returns:
        An unfitted instance of the selected estimator class. Call
        `.fit(X, y)` on it yourself -- this scaffold-depth pass does not
        implement a training/evaluation loop.

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
