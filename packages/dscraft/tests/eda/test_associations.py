"""Tests for dscraft.eda.associations -- continuous/categorical/mixed association metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.eda.associations import (
    AssociationMatrixResult,
    CorrelationResult,
    continuous_correlation,
    correlation_ratio,
    cramers_v,
    mixed_type_association_matrix,
    theils_u,
)


# ---------------------------------------------------------------------------
# 1. Continuous-continuous
# ---------------------------------------------------------------------------


def test_continuous_correlation_perfect_positive_linear():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2.0 * x + 1.0
    result = continuous_correlation(x, y, method="pearson")
    assert isinstance(result, CorrelationResult)
    assert result.statistic == pytest.approx(1.0, abs=1e-9)
    assert result.method == "pearson"
    assert result.n == 200


def test_continuous_correlation_perfect_negative_linear():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = -3.0 * x + 5.0
    result = continuous_correlation(x, y, method="pearson")
    assert result.statistic == pytest.approx(-1.0, abs=1e-9)


def test_continuous_correlation_independent_random_arrays_near_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    result = continuous_correlation(x, y, method="pearson")
    assert abs(result.statistic) < 0.05


def test_continuous_correlation_spearman_and_kendall_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # monotonic but nonlinear
    spearman = continuous_correlation(x, y, method="spearman")
    kendall = continuous_correlation(x, y, method="kendall")
    assert spearman.statistic == pytest.approx(1.0, abs=1e-9)
    assert kendall.statistic == pytest.approx(1.0, abs=1e-9)


def test_continuous_correlation_nan_handling_pairwise_complete():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, np.nan, 10.0])
    # Rows 2 and 3 dropped (either x or y is NaN) -> 3 rows remain: (1,2),(2,4),(5,10)
    result = continuous_correlation(x, y, method="pearson")
    assert result.n == 3
    assert result.statistic == pytest.approx(1.0, abs=1e-9)


def test_continuous_correlation_rejects_non_1d():
    with pytest.raises(ValueError):
        continuous_correlation(np.zeros((2, 2)), np.zeros((2, 2)))


def test_continuous_correlation_rejects_mismatched_length():
    with pytest.raises(ValueError):
        continuous_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_continuous_correlation_rejects_unknown_method():
    with pytest.raises(ValueError):
        continuous_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0]), method="bogus")


def test_continuous_correlation_rejects_too_few_rows_after_nan_drop():
    x = np.array([1.0, np.nan])
    y = np.array([np.nan, 2.0])
    with pytest.raises(ValueError):
        continuous_correlation(x, y)


# ---------------------------------------------------------------------------
# 2. Categorical-categorical
# ---------------------------------------------------------------------------


def test_cramers_v_independent_categoricals_near_zero():
    rng = np.random.default_rng(7)
    n = 20000
    x = rng.choice(["a", "b", "c"], size=n)
    y = rng.choice(["x", "y", "z"], size=n)  # generated independently of x
    v = cramers_v(x, y)
    assert v < 0.05


def test_cramers_v_perfect_association_is_one():
    # Each category of x maps to exactly one category of y.
    x = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    y = np.array(["x", "x", "x", "y", "y", "y", "z", "z", "z"])
    v = cramers_v(x, y)
    assert v == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_identical_column_is_one():
    x = np.array(["a", "b", "a", "c", "b", "c", "a"])
    v = cramers_v(x, x.copy())
    assert v == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_bias_correction_reduces_or_equals_plain():
    rng = np.random.default_rng(3)
    x = rng.choice(["a", "b", "c", "d"], size=40)
    y = rng.choice(["p", "q", "r"], size=40)
    plain = cramers_v(x, y, bias_correction=False)
    corrected = cramers_v(x, y, bias_correction=True)
    assert 0.0 <= corrected
    assert corrected <= plain + 1e-9


def test_cramers_v_rejects_fewer_than_two_categories():
    x = np.array(["a", "a", "a"])
    y = np.array(["x", "y", "z"])
    with pytest.raises(ValueError):
        cramers_v(x, y)


def test_cramers_v_rejects_mismatched_length():
    with pytest.raises(ValueError):
        cramers_v(np.array(["a", "b"]), np.array(["x", "y", "z"]))


def test_cramers_v_nan_handling_pairwise_complete():
    # Row index 3 has y=None and row index 4 has x=None; pairwise-complete
    # deletion drops both, leaving 3 usable rows: (a,x), (a,x), (b,y).
    x = np.array(["a", "a", "b", "b", None], dtype=object)
    y = np.array(["x", "x", "y", None, "y"], dtype=object)
    v = cramers_v(x, y)
    assert v == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Theil's U -- asymmetry
# ---------------------------------------------------------------------------


def test_theils_u_asymmetry_y_determines_x_but_not_reverse():
    # y perfectly determines x (each y maps to exactly one x), but x does
    # NOT perfectly determine y (each x maps to multiple y values).
    x = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    y = np.array(["p", "p", "q", "q", "r", "r", "s", "s"])
    u_x_given_y = theils_u(x, y)
    u_y_given_x = theils_u(y, x)
    assert u_x_given_y == pytest.approx(1.0, abs=1e-9)
    assert u_y_given_x < u_x_given_y
    assert u_y_given_x < 1.0


def test_theils_u_independent_categoricals_near_zero():
    rng = np.random.default_rng(11)
    n = 20000
    x = rng.choice(["a", "b", "c"], size=n)
    y = rng.choice(["x", "y", "z"], size=n)
    u = theils_u(x, y)
    assert u < 0.05


def test_theils_u_bounds_and_rejects_degenerate():
    with pytest.raises(ValueError):
        theils_u(np.array(["a", "a", "a"]), np.array(["x", "y", "z"]))


def test_theils_u_rejects_mismatched_length():
    with pytest.raises(ValueError):
        theils_u(np.array(["a", "b"]), np.array(["x", "y", "z"]))


# ---------------------------------------------------------------------------
# 3. Correlation ratio (eta)
# ---------------------------------------------------------------------------


def test_correlation_ratio_known_high_eta_well_separated_zero_variance_groups():
    categories = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"])
    # Zero within-group variance, well-separated means -> eta ~= 1.0
    values = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
    eta = correlation_ratio(categories, values)
    assert eta == pytest.approx(1.0, abs=1e-9)


def test_correlation_ratio_known_low_eta_identical_means():
    rng = np.random.default_rng(5)
    categories = np.array(["a"] * 500 + ["b"] * 500 + ["c"] * 500)
    # All groups share the same mean (0.0) with substantial within-group
    # variance -> group membership explains ~none of the variance.
    values = rng.normal(loc=0.0, scale=5.0, size=1500)
    eta = correlation_ratio(categories, values)
    assert eta < 0.15


def test_correlation_ratio_nan_handling_pairwise_complete():
    categories = np.array(["a", "a", "b", "b", None], dtype=object)
    values = np.array([1.0, 1.0, 10.0, 10.0, np.nan])
    eta = correlation_ratio(categories, values)
    assert eta == pytest.approx(1.0, abs=1e-9)


def test_correlation_ratio_rejects_fewer_than_two_categories():
    with pytest.raises(ValueError):
        correlation_ratio(np.array(["a", "a", "a"]), np.array([1.0, 2.0, 3.0]))


def test_correlation_ratio_rejects_zero_variance_values():
    with pytest.raises(ValueError):
        correlation_ratio(np.array(["a", "b", "a", "b"]), np.array([1.0, 1.0, 1.0, 1.0]))


def test_correlation_ratio_rejects_mismatched_length():
    with pytest.raises(ValueError):
        correlation_ratio(np.array(["a", "b"]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Mixed-type association matrix
# ---------------------------------------------------------------------------


def test_mixed_type_association_matrix_routes_correct_metric_per_pair():
    rng = np.random.default_rng(21)
    n = 300
    cont_a = rng.normal(size=n)
    cont_b = 2.0 * cont_a + rng.normal(scale=0.01, size=n)  # strongly correlated with cont_a
    cat_a = rng.choice(["p", "q"], size=n)
    cat_b = rng.choice(["u", "v", "w"], size=n)  # independent of cat_a

    data = {
        "cont_a": cont_a,
        "cont_b": cont_b,
        "cat_a": cat_a,
        "cat_b": cat_b,
    }
    result = mixed_type_association_matrix(data)

    assert isinstance(result, AssociationMatrixResult)
    assert result.columns == ["cont_a", "cont_b", "cat_a", "cat_b"]
    assert result.column_kinds == {
        "cont_a": "continuous",
        "cont_b": "continuous",
        "cat_a": "categorical",
        "cat_b": "categorical",
    }
    assert result.symmetric is True

    # Diagonal is always 1.0 by definition.
    np.testing.assert_allclose(np.diag(result.matrix), 1.0)

    # cont_a vs cont_b: continuous-continuous -> pearson, strongly correlated.
    i, j = result.columns.index("cont_a"), result.columns.index("cont_b")
    assert result.metrics[i][j] == "pearson"
    assert result.matrix[i, j] > 0.99

    # cat_a vs cat_b: categorical-categorical -> cramers_v, independent -> near 0.
    i, j = result.columns.index("cat_a"), result.columns.index("cat_b")
    assert result.metrics[i][j] == "cramers_v"
    assert result.matrix[i, j] < 0.2

    # cont_a vs cat_a: mixed -> correlation_ratio, symmetric regardless of order.
    i, j = result.columns.index("cont_a"), result.columns.index("cat_a")
    assert result.metrics[i][j] == "correlation_ratio"
    assert result.matrix[i, j] == result.matrix[j, i]

    # Matrix is symmetric overall.
    np.testing.assert_allclose(result.matrix, result.matrix.T)


def test_mixed_type_association_matrix_theils_u_mode_is_asymmetric():
    x = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    y = np.array(["p", "p", "q", "q", "r", "r", "s", "s"])
    data = {"x": x, "y": y}
    result = mixed_type_association_matrix(data, categorical_metric="theils_u")
    assert result.symmetric is False
    i, j = result.columns.index("x"), result.columns.index("y")
    assert result.matrix[i, j] != pytest.approx(result.matrix[j, i])
    assert result.metrics[i][j] == "theils_u"


def test_mixed_type_association_matrix_column_kinds_override():
    # An integer column that is actually a categorical code.
    data = {
        "code": np.array([1, 1, 2, 2, 3, 3]),
        "value": np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]),
    }
    result = mixed_type_association_matrix(data, column_kinds={"code": "categorical"})
    assert result.column_kinds["code"] == "categorical"
    i, j = result.columns.index("code"), result.columns.index("value")
    assert result.metrics[i][j] == "correlation_ratio"


def test_mixed_type_association_matrix_rejects_empty_column_selection():
    with pytest.raises(ValueError):
        mixed_type_association_matrix({"a": [1, 2, 3]}, columns=[])


def test_mixed_type_association_matrix_unavailable_pair_recorded_not_raised():
    # After NaN-dropping there's only one shared row between "a" and "b",
    # not enough for cat_a to have >= 2 categories in the overlap.
    data = {
        "a": np.array(["x", "x", "x", "x"], dtype=object),
        "b": np.array(["p", "q", "r", "s"], dtype=object),
    }
    result = mixed_type_association_matrix(data)
    i, j = result.columns.index("a"), result.columns.index("b")
    assert ("a", "b") in result.unavailable_pairs
    assert np.isnan(result.matrix[i, j])
    assert np.isnan(result.matrix[j, i])
    assert result.metrics[i][j] == "unavailable"
