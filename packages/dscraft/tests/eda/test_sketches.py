"""Tests for HyperLogLog cardinality and KLL quantile sketches.

Accuracy tolerances are derived from each sketch's own documented error
bound rather than picked arbitrarily:

- HLL: :attr:`~dscraft.eda.sketches.HLLResult.relative_error` is the
  family's theoretical *standard* error (one standard deviation) for the
  configured ``log2_k``. A single point estimate is allowed to land several
  standard errors away from the true value without indicating a bug --
  asserting within a tight 1-sigma band would make this test flaky by
  construction. We assert within 5 standard errors (>99.9999% confidence
  under a normal approximation), which is loose enough to be robust across
  runs/seeds while still failing loudly if the estimate is systematically
  wrong (e.g. off by 2x, which would be tens of standard errors away).
- KLL: :attr:`~dscraft.eda.sketches.KLLResult.normalized_rank_error` is a
  *rank*-space error bound (fraction of the sorted stream), not a
  value-space one. We convert it to an approximate value-space tolerance by
  multiplying by the sample's observed range (max - min): a rank error of
  ``e`` means the estimated quantile's true rank could be off by ``e`` in
  normalized-rank terms, which for a reasonably smooth distribution
  translates to roughly ``e * range`` in value terms near the middle of the
  distribution. We use a generous 8x safety multiplier on top of that
  (rather than the library's tighter, distribution-dependent guarantee) to
  keep this robust across the normal/uniform/constant fixtures below
  without being so loose it stops proving anything.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dscraft.eda.sketches import (
    HLLResult,
    KLLResult,
    estimate_cardinality,
    estimate_quantiles,
)

# ---------------------------------------------------------------------------
# HyperLogLog (cardinality)
# ---------------------------------------------------------------------------


def test_hll_result_is_frozen_dataclass_with_expected_fields():
    result = estimate_cardinality([1, 2, 3], log2_k=10)
    assert isinstance(result, HLLResult)
    assert result.log2_k == 10
    assert result.num_values_processed == 3
    with pytest.raises(AttributeError):
        result.estimate = 999.0  # type: ignore[misc]


@pytest.mark.parametrize("log2_k", [8, 10, 12, 14])
def test_hll_accuracy_against_known_exact_cardinality_integers(log2_k):
    """`list(range(n))` has an exactly known distinct count of `n`."""
    true_cardinality = 100_000
    result = estimate_cardinality(range(true_cardinality), log2_k=log2_k)

    # Theoretical bound check: within 5 standard errors of the true value.
    # See module docstring for why 5 (not 1) sigma is the right band for a
    # single, non-flaky point-estimate assertion.
    tolerance = 5 * result.relative_error * true_cardinality
    assert abs(result.estimate - true_cardinality) <= tolerance, (
        f"log2_k={log2_k}: estimate {result.estimate} vs true {true_cardinality}, "
        f"tolerance {tolerance} (relative_error={result.relative_error})"
    )


def test_hll_accuracy_against_known_exact_cardinality_random_strings():
    """Random strings with a separately-tracked true distinct count via `set()`."""
    rng = random.Random(42)
    values = [f"user-{rng.randrange(0, 10**9)}" for _ in range(50_000)]
    true_cardinality = len(set(values))  # ground truth, tracked independently

    result = estimate_cardinality(values, log2_k=12)
    tolerance = 5 * result.relative_error * true_cardinality
    assert abs(result.estimate - true_cardinality) <= tolerance


def test_hll_higher_precision_gives_tighter_relative_error():
    """Higher log2_k must report a strictly smaller (or equal) theoretical RSE."""
    low = estimate_cardinality(range(10_000), log2_k=8)
    mid = estimate_cardinality(range(10_000), log2_k=12)
    high = estimate_cardinality(range(10_000), log2_k=16)
    assert low.relative_error > mid.relative_error > high.relative_error


def test_hll_relative_error_matches_documented_formula():
    """1.04 / sqrt(2**log2_k), per the DataSketches HLL family's documented RSE."""
    result = estimate_cardinality([1, 2, 3], log2_k=12)
    expected = 1.04 / math.sqrt(2**12)
    assert result.relative_error == pytest.approx(expected)


def test_hll_accepts_non_primitive_hashable_items_via_str_fallback():
    """Non-int/float/str items (e.g. tuples) are converted via str() -- see
    the estimate_cardinality docstring's documented collision caveat."""
    result = estimate_cardinality([(1, 2), (3, 4), (1, 2)], log2_k=10)
    assert result.num_values_processed == 3
    assert result.estimate > 0


def test_hll_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        estimate_cardinality([], log2_k=10)


def test_hll_single_value_input():
    result = estimate_cardinality(["only-one"], log2_k=10)
    assert result.num_values_processed == 1
    assert result.estimate == pytest.approx(1.0, abs=0.5)


def test_hll_all_identical_values_estimates_cardinality_one():
    result = estimate_cardinality(["dup"] * 1000, log2_k=10)
    assert result.num_values_processed == 1000
    assert result.estimate == pytest.approx(1.0, abs=0.5)


@pytest.mark.parametrize("log2_k", [0, 1, 6, 22, 100, -5])
def test_hll_log2_k_out_of_range_raises_value_error(log2_k):
    with pytest.raises(ValueError):
        estimate_cardinality([1, 2, 3], log2_k=log2_k)


# ---------------------------------------------------------------------------
# KLL (quantiles)
# ---------------------------------------------------------------------------


def test_kll_result_is_frozen_dataclass_with_expected_fields():
    result = estimate_quantiles([1.0, 2.0, 3.0], quantiles=(0.5,), k=200)
    assert isinstance(result, KLLResult)
    assert result.k == 200
    assert result.num_values_processed == 3
    assert set(result.quantile_estimates.keys()) == {0.5}
    with pytest.raises(AttributeError):
        result.k = 999  # type: ignore[misc]


def test_kll_accuracy_against_known_normal_distribution():
    """np.random.default_rng(seed).normal(...) gives a fixed, reproducible
    sample whose TRUE quantiles (via np.quantile) are the ground truth."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=100_000)
    requested = [0.1, 0.25, 0.5, 0.75, 0.9]

    result = estimate_quantiles(data.tolist(), quantiles=requested, k=200)

    true_quantiles = {q: float(np.quantile(data, q)) for q in requested}
    data_range = float(data.max() - data.min())
    # See module docstring for the rank-error -> value-space tolerance
    # derivation and the 8x safety multiplier.
    tolerance = 8 * result.normalized_rank_error * data_range

    for q in requested:
        estimate = result.quantile_estimates[q]
        true_value = true_quantiles[q]
        assert abs(estimate - true_value) <= tolerance, (
            f"quantile={q}: estimate {estimate} vs true {true_value}, tolerance {tolerance}"
        )


def test_kll_accuracy_against_known_uniform_distribution():
    rng = np.random.default_rng(7)
    data = rng.uniform(low=0.0, high=100.0, size=50_000)
    result = estimate_quantiles(data.tolist(), quantiles=(0.25, 0.5, 0.75), k=200)

    # Uniform[0, 100] has closed-form quantiles: q -> q * 100.
    for q, estimate in result.quantile_estimates.items():
        expected = q * 100.0
        assert estimate == pytest.approx(expected, abs=8 * result.normalized_rank_error * 100.0)


def test_kll_default_quantiles_are_quartiles_and_median():
    result = estimate_quantiles([float(i) for i in range(1, 1001)])
    assert set(result.quantile_estimates.keys()) == {0.25, 0.5, 0.75}


def test_kll_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        estimate_quantiles([])


def test_kll_single_value_input():
    result = estimate_quantiles([42.0], quantiles=(0.5,))
    assert result.num_values_processed == 1
    assert result.quantile_estimates[0.5] == pytest.approx(42.0)


def test_kll_all_identical_values():
    result = estimate_quantiles([5.0] * 1000, quantiles=(0.1, 0.5, 0.9))
    for estimate in result.quantile_estimates.values():
        assert estimate == pytest.approx(5.0)


def test_kll_non_numeric_input_raises_type_error():
    with pytest.raises(TypeError):
        estimate_quantiles(["a", "b", "c"])


def test_kll_non_numeric_input_mixed_with_numeric_raises_type_error():
    with pytest.raises(TypeError):
        estimate_quantiles([1.0, 2.0, "oops", 4.0])


def test_kll_bool_input_raises_type_error():
    """bool is technically an int subclass in Python, but treating
    True/False as 1.0/0.0 in a numeric-quantile context is almost always an
    upstream typing bug -- see the estimate_quantiles docstring."""
    with pytest.raises(TypeError):
        estimate_quantiles([1.0, True, 3.0])


def test_kll_non_finite_input_raises_value_error():
    with pytest.raises(ValueError):
        estimate_quantiles([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError):
        estimate_quantiles([1.0, float("inf"), 3.0])


@pytest.mark.parametrize("bad_quantile", [-0.1, 1.1, 2.0, -5.0])
def test_kll_out_of_range_quantile_raises_value_error(bad_quantile):
    with pytest.raises(ValueError):
        estimate_quantiles([1.0, 2.0, 3.0], quantiles=(bad_quantile,))


def test_kll_empty_quantiles_sequence_raises_value_error():
    with pytest.raises(ValueError):
        estimate_quantiles([1.0, 2.0, 3.0], quantiles=())


@pytest.mark.parametrize("k", [0, 1, 7, 65536, 100000, -10])
def test_kll_k_out_of_range_raises_value_error(k):
    with pytest.raises(ValueError):
        estimate_quantiles([1.0, 2.0, 3.0], k=k)
