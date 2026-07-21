"""Tests for dscraft.automl.resampling.

Covers the imbalanced-class resampling allowlist: a smoke test per
resampler on a small synthetic imbalanced classification dataset,
asserting the class balance actually improves after resampling, plus the
unknown-name `ValueError` dispatch path.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from dscraft.automl import SUPPORTED_RESAMPLERS, build_resampler
from dscraft.automl.resampling import build_resampler as build_resampler_direct


def test_public_api_surface() -> None:
    """`build_resampler`/`SUPPORTED_RESAMPLERS` are importable from `dscraft.automl`."""
    assert build_resampler is build_resampler_direct
    assert set(SUPPORTED_RESAMPLERS) == {"RandomOverSampler", "SMOTE", "RandomUnderSampler"}


def _imbalance_ratio(y: np.ndarray) -> float:
    """Minority-class fraction of the smaller of the two classes in a binary `y`."""
    _values, counts = np.unique(y, return_counts=True)
    return counts.min() / counts.sum()


@pytest.mark.parametrize("name", ["RandomOverSampler", "SMOTE", "RandomUnderSampler"])
def test_resampler_improves_class_balance_smoke(name: str) -> None:
    """Each resampler improves class balance on a synthetic imbalanced dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        weights=[0.9, 0.1],
        random_state=0,
    )
    before_ratio = _imbalance_ratio(y)

    resampler = build_resampler(name, random_state=0)
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    after_ratio = _imbalance_ratio(y_resampled)

    assert after_ratio > before_ratio
    assert len(X_resampled) == len(y_resampled)


def test_build_resampler_unknown_name_raises() -> None:
    """An unrecognized resampler `name` raises ValueError listing valid names."""
    with pytest.raises(ValueError, match="Unsupported resampler"):
        build_resampler("NotARealResampler")
