"""Imbalanced-class resampling backends for `dscraft.automl`.

`dscraft.automl` handles skewed class distributions -- which degrade a
naively-fit classifier -- via this optional preprocessing step, ahead of
the classifier path (`models.py`), that resamples a training set toward
better class balance. The resampling *techniques* are now **first-party**
(`dscraft.automl._resampling`, built from scratch on numpy): the
third-party `imbalanced-learn` backends this module originally wrapped were
replaced by the first-party automl/eda rebuild (see
`docs/decisions/2026-10-first-party-automl-eda-rebuild.md`), so no
resampling library is a dependency here.

Per this repo's multi-backend design principle, this uses the same
`SUPPORTED_*`-dict-allowlist pattern used elsewhere in this package
(`models.py`'s `SUPPORTED_CLASSIFIERS`/`SUPPORTED_REGRESSORS`,
`clustering.py`'s `SUPPORTED_CLUSTERERS`,
`dscraft.forecast.forecast.SUPPORTED_MODELS`), covering both over-sampling
(`RandomOverSampler`, `SMOTE`, `BorderlineSMOTE`) and under-sampling
(`RandomUnderSampler`) strategies as equally-selectable options -- none is
a "default" resampling strategy.
"""

from __future__ import annotations

from typing import Any

from dscraft.automl._resampling import (
    BorderlineSMOTE,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTE,
)

__all__ = [
    "SUPPORTED_RESAMPLERS",
    "build_resampler",
]

#: Resampling backends this module supports, keyed by caller-facing name.
#: Covers over-sampling (random + SMOTE + borderline-SMOTE) and
#: under-sampling (random) strategies as equally-supported options, per the
#: multi-backend design principle. All are **first-party** implementations
#: (`dscraft.automl._resampling`, built from scratch on numpy) -- the
#: third-party `imbalanced-learn` backends this allowlist originally wrapped
#: were replaced by the first-party automl/eda rebuild (see
#: docs/decisions/2026-10-first-party-automl-eda-rebuild.md). The
#: first-party `RandomOverSampler`/`SMOTE`/`RandomUnderSampler` classes keep
#: the same caller-facing names and the same `fit_resample(X, y)` contract,
#: so this is a drop-in replacement; `BorderlineSMOTE` is an additional
#: first-party strategy this rebuild adds.
SUPPORTED_RESAMPLERS: dict[str, type] = {
    "RandomOverSampler": RandomOverSampler,
    "SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "RandomUnderSampler": RandomUnderSampler,
}


def build_resampler(name: str, **kwargs: Any):
    """Build an unfitted, first-party resampler exposing ``fit_resample(X, y)``.

    This is the one canonical resampling-backend factory for
    `dscraft.automl` -- no parallel resampling entrypoint exists elsewhere
    in this package.

    Args:
        name: which backend to build -- a key of :data:`SUPPORTED_RESAMPLERS`
            (``"RandomOverSampler"``, ``"SMOTE"``, ``"BorderlineSMOTE"``, or
            ``"RandomUnderSampler"``).
        **kwargs: forwarded straight through to the selected resampler's
            constructor (e.g. ``random_state``, ``k_neighbors`` for the
            SMOTE variants, ``m_neighbors`` for ``"BorderlineSMOTE"``).

    Returns:
        An unfitted instance of the selected resampler class. Call
        `.fit_resample(X, y)` on it yourself to get the resampled
        ``(X, y)`` pair.

    Raises:
        ValueError: ``name`` is not a key of :data:`SUPPORTED_RESAMPLERS`.
    """
    resampler_cls = SUPPORTED_RESAMPLERS.get(name)
    if resampler_cls is None:
        raise ValueError(
            f"Unsupported resampler {name!r}. "
            f"Supported: {sorted(SUPPORTED_RESAMPLERS)!r}."
        )

    return resampler_cls(**kwargs)
