"""Tests for dscraft.eda._univariate and its wiring into LazyEDA.

Validates the first-party univariate descriptors against known/closed-form values (moments,
robust stats, shape, entropy) and confirms LazyEDA.profile populates numeric/categorical
descriptors for the right columns.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from dscraft.eda import LazyEDA, NumericDescriptor, CategoricalDescriptor
from dscraft.eda._univariate import categorical_descriptor, numeric_descriptor


# ---------------------------------------------------------------------------
# Numeric descriptor
# ---------------------------------------------------------------------------


def test_numeric_descriptor_known_values() -> None:
    d = numeric_descriptor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d.count == 5
    assert d.mean == pytest.approx(3.0)
    assert d.median == pytest.approx(3.0)
    assert d.minimum == 1.0 and d.maximum == 5.0 and d.value_range == 4.0
    assert d.variance == pytest.approx(2.5)  # sample variance (ddof=1)
    assert d.std == pytest.approx(math.sqrt(2.5))
    assert d.iqr == pytest.approx(2.0)  # p75=4, p25=2
    assert d.mad == pytest.approx(1.0)  # median of |x - 3| = median(2,1,0,1,2)


def test_numeric_descriptor_symmetric_has_zero_skew() -> None:
    d = numeric_descriptor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert d.skewness == pytest.approx(0.0, abs=1e-9)


def test_numeric_descriptor_right_skewed_positive() -> None:
    # A long right tail -> positive skew.
    d = numeric_descriptor([1, 1, 1, 1, 2, 2, 3, 10])
    assert d.skewness is not None and d.skewness > 0


def test_numeric_descriptor_kurtosis_sign() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(size=5000)
    d = numeric_descriptor(normal.tolist())
    # Excess kurtosis of a normal is ~0.
    assert d.kurtosis is not None and abs(d.kurtosis) < 0.5


def test_numeric_descriptor_constant_has_no_shape_or_cv_edge() -> None:
    d = numeric_descriptor([5.0, 5.0, 5.0, 5.0])
    assert d.variance == 0.0 and d.std == 0.0
    assert d.skewness is None and d.kurtosis is None  # undefined for zero variance
    assert d.coefficient_of_variation == pytest.approx(0.0)  # std 0 / |mean| 5


def test_numeric_descriptor_cv_none_near_zero_mean() -> None:
    d = numeric_descriptor([-1.0, 1.0])  # mean ~0
    assert d.coefficient_of_variation is None


def test_numeric_descriptor_too_few_for_shape() -> None:
    d = numeric_descriptor([1.0, 2.0])  # n < 3 -> no skew; n < 4 -> no kurtosis
    assert d.skewness is None and d.kurtosis is None


def test_numeric_descriptor_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        numeric_descriptor([])


# ---------------------------------------------------------------------------
# Categorical descriptor
# ---------------------------------------------------------------------------


def test_categorical_descriptor_mode_and_distinct() -> None:
    d = categorical_descriptor(["a", "a", "a", "b", "c"])
    assert d.count == 5 and d.distinct == 3
    assert d.mode == "a" and d.mode_frequency == 3


def test_categorical_descriptor_uniform_max_entropy() -> None:
    d = categorical_descriptor(["a", "b", "c", "d"])
    assert d.normalized_entropy == pytest.approx(1.0)  # perfectly uniform


def test_categorical_descriptor_single_value_zero_entropy() -> None:
    d = categorical_descriptor(["x", "x", "x"])
    assert d.distinct == 1 and d.normalized_entropy == 0.0


def test_categorical_descriptor_stringifies() -> None:
    d = categorical_descriptor([1, 1, 2])
    assert d.mode == "1" and d.distinct == 2


def test_categorical_descriptor_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        categorical_descriptor([])


# ---------------------------------------------------------------------------
# LazyEDA integration
# ---------------------------------------------------------------------------


def test_profile_populates_descriptors() -> None:
    frame = pl.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "city": ["NYC", "LA", "NYC", "NYC", "SF", "LA"],
            "flag": [True, False, True, True, False, True],
        }
    )
    profile = LazyEDA().profile(frame)
    assert "amount" in profile.numeric_descriptors
    assert isinstance(profile.numeric_descriptors["amount"], NumericDescriptor)
    assert "city" in profile.categorical_descriptors
    assert isinstance(profile.categorical_descriptors["city"], CategoricalDescriptor)
    # Boolean columns get neither sketch nor descriptor (unchanged routing).
    assert "flag" not in profile.numeric_descriptors
    assert "flag" not in profile.categorical_descriptors
    # Sanity: the numeric descriptor reflects the real data.
    assert profile.numeric_descriptors["amount"].maximum == 100.0
    assert profile.categorical_descriptors["city"].mode == "NYC"


def test_profile_descriptors_absent_for_empty_numeric() -> None:
    frame = pl.DataFrame({"empty": [None, None, None]}, schema={"empty": pl.Float64})
    profile = LazyEDA().profile(frame)
    assert "empty" not in profile.numeric_descriptors
