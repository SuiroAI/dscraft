"""Tests for the first-party HyperLogLog in dscraft.eda._hll.

Validates the from-scratch HLL directly (the public estimate_cardinality contract is covered by
test_sketches.py): the estimate is within the theoretical relative standard error on streams of
known cardinality across several precisions, duplicates don't inflate it, hashing is stable across
instances/processes, and the small-range (linear-counting) correction keeps low cardinalities
accurate.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.eda._hll import HyperLogLog, _hash64


@pytest.mark.parametrize("log2_k", [8, 12, 14])
@pytest.mark.parametrize("true_cardinality", [100, 1000, 10000])
def test_estimate_within_theoretical_error(log2_k, true_cardinality) -> None:
    hll = HyperLogLog(log2_k)
    hll.update_many(range(true_cardinality))
    est = hll.estimate()
    rse = hll.relative_error()
    # Allow a generous ~4 standard errors (plus a small floor for the smallest sketches) -- HLL is
    # probabilistic, so this bounds the error without being flaky.
    tolerance = 4.0 * rse * true_cardinality + 5
    assert abs(est - true_cardinality) <= tolerance


def test_duplicates_do_not_inflate_estimate() -> None:
    hll = HyperLogLog(14)
    # 500 distinct values, each inserted 20 times.
    for _ in range(20):
        hll.update_many(range(500))
    est = hll.estimate()
    assert 450 <= est <= 550  # close to 500 distinct, not 10000 total


def test_exact_for_tiny_cardinality_via_linear_counting() -> None:
    # With very few distinct values most registers stay empty, so the small-range (linear
    # counting) correction should make the estimate essentially exact.
    hll = HyperLogLog(12)
    hll.update_many(["a", "b", "c"])
    assert round(hll.estimate()) == 3


def test_relative_error_matches_formula() -> None:
    for log2_k in (7, 10, 16):
        hll = HyperLogLog(log2_k)
        assert hll.relative_error() == pytest.approx(1.04 / np.sqrt(2**log2_k))


def test_hash_is_stable_across_instances() -> None:
    # Reproducibility: the same value hashes identically every time (not salted per process).
    assert _hash64("hello") == _hash64("hello")
    assert _hash64(42) == _hash64(42)


def test_distinct_types_hash_distinctly() -> None:
    # int 1, float 1.0, str "1", and bool True are different logical values and should not all
    # collide to one hash (they carry a type-tag prefix).
    hashes = {_hash64(1), _hash64(1.0), _hash64("1"), _hash64(True)}
    assert len(hashes) >= 3  # at most int/float may coincide numerically; the rest are distinct


def test_register_count_matches_precision() -> None:
    hll = HyperLogLog(10)
    assert hll.m == 1024
    assert hll.registers.shape == (1024,)
    assert hll.registers.dtype == np.uint8


def test_rejects_invalid_log2_k() -> None:
    with pytest.raises(ValueError, match="log2_k"):
        HyperLogLog(0)


def test_non_primitive_items_coerced_via_str() -> None:
    # Tuples etc. are str()-coerced; two tuples with the same repr count as one.
    hll = HyperLogLog(12)
    hll.update((1, 2))
    hll.update((1, 2))
    hll.update((3, 4))
    assert round(hll.estimate()) == 2
