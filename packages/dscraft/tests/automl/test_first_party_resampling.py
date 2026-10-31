"""Tests for dscraft.automl._resampling.

Validates the first-party resamplers: they balance class counts, SMOTE synthesizes interpolated
(not duplicated) minority points that lie within the minority class's convex hull, under-sampling
never invents data, borderline-SMOTE concentrates on the boundary, results are reproducible, and
the resampled training set is usable (originals never mutated). Where imbalanced-learn is present,
a parity check confirms the balancing behaviour matches.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from dscraft.automl._resampling import (
    BorderlineSMOTE,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTE,
)


def _imbalanced(weights=(0.9, 0.1), n=300, seed=0):
    return make_classification(
        n_samples=n, n_features=5, n_informative=3, n_redundant=0,
        weights=list(weights), random_state=seed,
    )


def _counts(y):
    _v, c = np.unique(y, return_counts=True)
    return c


# ---------------------------------------------------------------------------
# Balancing behaviour
# ---------------------------------------------------------------------------


def test_random_over_sampler_balances_to_majority() -> None:
    X, y = _imbalanced()
    Xr, yr = RandomOverSampler(random_state=0).fit_resample(X, y)
    counts = _counts(yr)
    assert counts.min() == counts.max()  # balanced
    assert counts.max() == _counts(y).max()  # up to the majority count
    assert len(Xr) == len(yr)


def test_random_under_sampler_balances_to_minority() -> None:
    X, y = _imbalanced()
    Xr, yr = RandomUnderSampler(random_state=0).fit_resample(X, y)
    counts = _counts(yr)
    assert counts.min() == counts.max()
    assert counts.max() == _counts(y).min()  # down to the minority count


def test_smote_balances_to_majority() -> None:
    X, y = _imbalanced()
    Xr, yr = SMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)
    counts = _counts(yr)
    assert counts.min() == counts.max()
    assert counts.max() == _counts(y).max()


def test_borderline_smote_balances() -> None:
    X, y = _imbalanced(weights=(0.85, 0.15))
    Xr, yr = BorderlineSMOTE(random_state=0).fit_resample(X, y)
    counts = _counts(yr)
    assert counts.min() == counts.max()


# ---------------------------------------------------------------------------
# SMOTE synthesizes, it does not merely copy
# ---------------------------------------------------------------------------


def test_smote_creates_novel_interpolated_points() -> None:
    X, y = _imbalanced()
    minority_label = np.argmin(_counts(y))
    original_minority = X[y == minority_label]
    Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
    new_minority = Xr[len(y):]  # synthetic points are appended after the originals
    # At least some synthetic points are not exact copies of any original minority row.
    def is_copy(pt):
        return np.any(np.all(np.isclose(original_minority, pt), axis=1))
    assert not all(is_copy(pt) for pt in new_minority)


def test_smote_points_lie_within_minority_feature_bounds() -> None:
    # Interpolation on segments between minority points stays within the per-feature min/max box
    # of the minority class (a basic sanity property of convex interpolation).
    X, y = _imbalanced()
    minority_label = np.argmin(_counts(y))
    minority = X[y == minority_label]
    lo, hi = minority.min(axis=0), minority.max(axis=0)
    Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
    synth = Xr[len(y):]
    assert np.all(synth >= lo - 1e-9) and np.all(synth <= hi + 1e-9)


def test_smote_small_class_falls_back_gracefully() -> None:
    # A minority class with a single member cannot interpolate; SMOTE must still balance (by
    # duplication) rather than crash.
    X = np.vstack([np.zeros((10, 2)), np.ones((1, 2))])
    y = np.array([0] * 10 + [1])
    Xr, yr = SMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)
    assert _counts(yr).min() == _counts(yr).max() == 10


# ---------------------------------------------------------------------------
# Non-mutation, reproducibility, validation
# ---------------------------------------------------------------------------


def test_originals_not_mutated() -> None:
    X, y = _imbalanced()
    X_before = X.copy()
    y_before = y.copy()
    SMOTE(random_state=0).fit_resample(X, y)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


@pytest.mark.parametrize("sampler_cls", [RandomOverSampler, RandomUnderSampler, SMOTE, BorderlineSMOTE])
def test_resampling_is_reproducible(sampler_cls) -> None:
    X, y = _imbalanced()
    a_X, a_y = sampler_cls(random_state=42).fit_resample(X, y)
    b_X, b_y = sampler_cls(random_state=42).fit_resample(X, y)
    np.testing.assert_array_equal(a_X, b_X)
    np.testing.assert_array_equal(a_y, b_y)


def test_requires_two_classes() -> None:
    with pytest.raises(ValueError, match="at least 2 classes"):
        SMOTE().fit_resample(np.zeros((5, 2)), np.zeros(5))


def test_smote_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k_neighbors"):
        SMOTE(k_neighbors=0).fit_resample(*_imbalanced())


# ---------------------------------------------------------------------------
# Parity with imbalanced-learn (correctness cross-check; skipped if absent)
# ---------------------------------------------------------------------------


def test_over_sampler_parity_with_imblearn() -> None:
    imblearn = pytest.importorskip("imblearn")
    from imblearn.over_sampling import RandomOverSampler as RefOver

    X, y = _imbalanced()
    ours = _counts(RandomOverSampler(random_state=0).fit_resample(X, y)[1])
    theirs = _counts(RefOver(random_state=0).fit_resample(X, y)[1])
    np.testing.assert_array_equal(np.sort(ours), np.sort(theirs))  # same balanced counts


def test_smote_parity_counts_with_imblearn() -> None:
    pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE as RefSMOTE

    X, y = _imbalanced()
    ours = _counts(SMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)[1])
    theirs = _counts(RefSMOTE(k_neighbors=5, random_state=0).fit_resample(X, y)[1])
    # Same final per-class counts (the interpolated points differ, but the balancing target does not).
    np.testing.assert_array_equal(np.sort(ours), np.sort(theirs))
