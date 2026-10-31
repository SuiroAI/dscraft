"""Tests for the first-party KLL quantile sketch in dscraft.eda._kll.

Validates the from-scratch KLL directly (the public estimate_quantiles contract is covered by
test_sketches.py): quantile estimates are close to the true quantiles within the sketch's
rank-error bound on known distributions, min/median/max behave, the rank-error formula holds,
larger k is more accurate, and reproducibility holds under a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.eda._kll import KLL


def _true_quantile(data, q):
    return float(np.quantile(data, q))


@pytest.mark.parametrize("k", [100, 200, 400])
def test_quantiles_within_rank_error(k) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=50_000)
    sketch = KLL(k=k, random_state=0)
    sketch.update_many(data)
    eps = sketch.normalized_rank_error()
    sorted_data = np.sort(data)
    n = len(data)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        est = sketch.quantiles([q])[0]
        # The estimate's TRUE rank should be within ~eps (a few x for probabilistic slack) of q.
        true_rank = np.searchsorted(sorted_data, est) / n
        assert abs(true_rank - q) <= 6 * eps + 0.02


def test_min_and_max_are_exact() -> None:
    data = list(range(1000))
    sketch = KLL(k=200, random_state=0)
    sketch.update_many(data)
    assert sketch.quantiles([0.0])[0] == 0.0
    assert sketch.quantiles([1.0])[0] == 999.0


def test_median_of_uniform() -> None:
    data = np.linspace(0, 100, 20_000)
    sketch = KLL(k=200, random_state=0)
    sketch.update_many(data)
    median = sketch.quantiles([0.5])[0]
    assert abs(median - 50.0) < 3.0


def test_larger_k_is_more_accurate_on_average() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=40_000)
    true_median = float(np.median(data))
    small = KLL(k=50, random_state=0)
    large = KLL(k=800, random_state=0)
    small.update_many(data)
    large.update_many(data)
    # Larger k -> tighter rank-error bound.
    assert large.normalized_rank_error() < small.normalized_rank_error()


def test_rank_error_formula() -> None:
    for k in (8, 200, 1000):
        assert KLL(k=k).normalized_rank_error() == pytest.approx(2.296 / k)


def test_reproducible_with_seed() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(size=20_000)
    a = KLL(k=200, random_state=7)
    b = KLL(k=200, random_state=7)
    a.update_many(data)
    b.update_many(data)
    assert a.quantiles([0.25, 0.5, 0.75]) == b.quantiles([0.25, 0.5, 0.75])


def test_sketch_stays_sublinear() -> None:
    # After many inserts the retained item count is far below n (the whole point of a sketch).
    sketch = KLL(k=200, random_state=0)
    sketch.update_many(range(100_000))
    retained = sum(len(level) for level in sketch._levels)
    assert retained < 100_000 // 10


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        KLL(k=200).quantiles([0.5])


def test_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        KLL(k=1)
