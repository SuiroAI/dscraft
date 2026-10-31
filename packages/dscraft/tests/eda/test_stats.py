"""Tests for the first-party statistics in dscraft.eda._stats.

Validates Pearson/Spearman/Kendall correlation and the chi-squared contingency statistic against
closed-form/known values, and (where scipy is available) cross-checks them against scipy.stats as
an oracle to confirm the from-scratch implementations agree to tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.eda import _stats


# ---------------------------------------------------------------------------
# Known-value correctness
# ---------------------------------------------------------------------------


def test_pearson_perfect_positive_and_negative() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _stats.pearsonr(x, 2 * x + 1).statistic == pytest.approx(1.0)
    assert _stats.pearsonr(x, -3 * x).statistic == pytest.approx(-1.0)


def test_pearson_zero_for_constant() -> None:
    x = np.array([1.0, 2.0, 3.0])
    r = _stats.pearsonr(x, np.array([5.0, 5.0, 5.0]))
    assert r.statistic == 0.0 and r.p_value == 1.0


def test_spearman_is_monotonic_invariant() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # A monotonic (non-linear) transform -> Spearman 1.0 even though Pearson < 1.
    y = x**3
    assert _stats.spearmanr(x, y).statistic == pytest.approx(1.0)


def test_kendall_perfect_and_reversed() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert _stats.kendalltau(x, x).statistic == pytest.approx(1.0)
    assert _stats.kendalltau(x, x[::-1]).statistic == pytest.approx(-1.0)


def test_chi2_independent_table_is_near_zero() -> None:
    # Perfectly independent table -> chi2 == 0.
    table = np.array([[10.0, 10.0], [10.0, 10.0]])
    chi2, dof = _stats.chi2_contingency(table)
    assert chi2 == pytest.approx(0.0)
    assert dof == 1


def test_chi2_dependent_table_is_large() -> None:
    table = np.array([[40.0, 0.0], [0.0, 40.0]])  # perfect association
    chi2, dof = _stats.chi2_contingency(table)
    assert chi2 > 50  # strongly non-independent
    assert dof == 1


def test_rankdata_averages_ties() -> None:
    ranks = _stats._rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    # The two tied 20s share the average of ranks 2 and 3 -> 2.5.
    np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])


# ---------------------------------------------------------------------------
# Cross-check against scipy (oracle; skipped if scipy absent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pearsonr", "spearmanr", "kendalltau"])
def test_correlation_matches_scipy(method) -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(scale=0.5, size=200)
    ours = getattr(_stats, method)(x, y)
    theirs = getattr(stats, method)(x, y)
    assert ours.statistic == pytest.approx(float(theirs.statistic), abs=1e-6)
    # p-values use asymptotic approximations -> looser tolerance.
    assert ours.p_value == pytest.approx(float(theirs.pvalue), abs=1e-3)


def test_chi2_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    table = rng.integers(5, 50, size=(3, 4)).astype(float)
    ours_chi2, ours_dof = _stats.chi2_contingency(table)
    their_chi2, _p, their_dof, _exp = stats.chi2_contingency(table, correction=False)
    assert ours_chi2 == pytest.approx(float(their_chi2), rel=1e-9)
    assert ours_dof == their_dof


def test_incomplete_beta_symmetry() -> None:
    # I_x(a,b) + I_{1-x}(b,a) == 1 (the symmetry the t-tail relies on).
    assert _stats._betai(2.5, 3.5, 0.4) + _stats._betai(3.5, 2.5, 0.6) == pytest.approx(1.0, abs=1e-9)
