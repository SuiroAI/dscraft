"""Imbalanced-class resampling backends for `dscraft.automl`.

`dscraft.automl` had no imbalanced-class handling before this module --
real classification workloads often have skewed class distributions that
degrade a naively-fit classifier. This module adds an optional
preprocessing step, ahead of the existing classifier path (`models.py`),
that resamples a training set toward better class balance via
`imbalanced-learn`.

Per this repo's multi-backend design principle, this establishes the
same `SUPPORTED_*`-dict-allowlist pattern used elsewhere in this package
(`models.py`'s `SUPPORTED_CLASSIFIERS`/`SUPPORTED_REGRESSORS`,
`clustering.py`'s `SUPPORTED_CLUSTERERS`,
`dscraft.forecast.forecast.SUPPORTED_MODELS`), covering both
over-sampling (`RandomOverSampler`, `SMOTE`) and under-sampling
(`RandomUnderSampler`) strategies as equally-selectable options -- none
is a "default" resampling strategy.

Kept as an independently-selectable capability from `clustering.py`'s
HDBSCAN support (both new in the same architecture-doc gap-analysis
issue, but deliberately not coupled into one feature flag just because
they came from the same source project).
"""

from __future__ import annotations

from typing import Any

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

__all__ = [
    "SUPPORTED_RESAMPLERS",
    "build_resampler",
]

#: Resampling backends this module supports, keyed by caller-facing name.
#: Covers both over-sampling and under-sampling strategies as equally-
#: supported options, per the multi-backend design principle.
SUPPORTED_RESAMPLERS: dict[str, type] = {
    "RandomOverSampler": RandomOverSampler,
    "SMOTE": SMOTE,
    "RandomUnderSampler": RandomUnderSampler,
}


def build_resampler(name: str, **kwargs: Any) -> BaseSampler:
    """Build an unfitted, imbalanced-learn-compatible resampler.

    This is the one canonical resampling-backend factory for
    `dscraft.automl` -- no parallel resampling entrypoint exists elsewhere
    in this package.

    Args:
        name: which backend to build -- a key of :data:`SUPPORTED_RESAMPLERS`
            (``"RandomOverSampler"``, ``"SMOTE"``, or
            ``"RandomUnderSampler"``).
        **kwargs: forwarded straight through to the selected resampler's
            constructor (e.g. ``sampling_strategy``, ``random_state``).

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
