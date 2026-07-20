"""Tests for the aggregate Dataset Integrity Score (``dscraft.clean.integrity``).

Score convention reminder (see module docstring in ``integrity.py``): each
component is a "badness" rate in ``[0.0, 1.0]`` (0 = perfect, 1 = worst),
and the final ``I_data = 1 - weighted_sum_of_bad_things`` -- so **higher
I_data is better**, and a perfectly clean/uncontaminated/zero-drift dataset
scores ``I_data == 1.0``.

``jensen_shannon_divergence`` uses the natural-log convention, bounded in
``[0, ln(2)]`` (not ``[0, 1]``) -- see that function's docstring. The
composed :func:`demographic_drift_component` normalizes by ``ln(2)`` so its
own output stays in ``[0.0, 1.0]`` alongside the other two components.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.clean.integrity import (
    IntegrityReport,
    contamination_rate_component,
    dataset_integrity_score,
    demographic_drift_component,
    jensen_shannon_divergence,
    label_error_rate_component,
)

LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# jensen_shannon_divergence
# ---------------------------------------------------------------------------


def test_jsd_identical_distributions_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jsd_disjoint_support_equals_ln2_max():
    """Maximally different (disjoint support) distributions hit the known
    natural-log-convention maximum of ln(2), not 1.0."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p, q) == pytest.approx(LN2, abs=1e-9)


def test_jsd_known_worked_example():
    """Hand-computable example: p = [1, 0], q = [0.5, 0.5].

    m = [0.75, 0.25].
    KL(p || m) = 1 * log(1 / 0.75) = log(4/3).
    KL(q || m) = 0.5 * log(0.5 / 0.75) + 0.5 * log(0.5 / 0.25)
               = 0.5 * log(2/3) + 0.5 * log(2).
    JS = 0.5 * KL(p||m) + 0.5 * KL(q||m).
    """
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    m = np.array([0.75, 0.25])

    kl_p_m = 1.0 * np.log(1.0 / 0.75)
    kl_q_m = 0.5 * np.log(0.5 / 0.75) + 0.5 * np.log(0.5 / 0.25)
    expected = 0.5 * kl_p_m + 0.5 * kl_q_m

    assert jensen_shannon_divergence(p, q) == pytest.approx(expected, abs=1e-9)
    assert m.sum() == pytest.approx(1.0)  # sanity check on the worked example itself


def test_jsd_is_symmetric():
    p = np.array([0.1, 0.9])
    q = np.array([0.6, 0.4])
    assert jensen_shannon_divergence(p, q) == pytest.approx(jensen_shannon_divergence(q, p), abs=1e-12)


def test_jsd_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([0.5, 0.5], [1.0, 0.0, 0.0])


def test_jsd_rejects_negative_values():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([1.5, -0.5], [0.5, 0.5])


def test_jsd_rejects_non_normalized_distributions():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([0.5, 0.9], [0.5, 0.5])


def test_jsd_rejects_non_finite_values():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([np.nan, 1.0], [0.5, 0.5])


# ---------------------------------------------------------------------------
# label_error_rate_component
# ---------------------------------------------------------------------------


def test_label_error_rate_component_from_scalar_rate():
    assert label_error_rate_component(0.25) == pytest.approx(0.25)


def test_label_error_rate_component_from_boolean_mask():
    mask = np.array([True, False, True, False, False])  # 2/5 flagged
    assert label_error_rate_component(mask) == pytest.approx(0.4)


def test_label_error_rate_component_all_clean_mask_is_zero():
    mask = np.array([False, False, False])
    assert label_error_rate_component(mask) == pytest.approx(0.0)


def test_label_error_rate_component_all_flagged_mask_is_one():
    mask = np.array([True, True])
    assert label_error_rate_component(mask) == pytest.approx(1.0)


def test_label_error_rate_component_rejects_empty_mask():
    with pytest.raises(ValueError):
        label_error_rate_component(np.array([], dtype=bool))


def test_label_error_rate_component_rejects_out_of_range_scalar():
    with pytest.raises(ValueError):
        label_error_rate_component(1.5)


# ---------------------------------------------------------------------------
# contamination_rate_component
# ---------------------------------------------------------------------------


def test_contamination_rate_component_from_scalar_rate():
    assert contamination_rate_component(0.1) == pytest.approx(0.1)


def test_contamination_rate_component_from_three_state_sequence():
    states = [
        "clean",
        "validated_contaminated",
        "candidate_unvalidated",  # not counted -- unvalidated, per module contract
        "validated_contaminated",
        "clean",
    ]
    # 2 validated_contaminated out of 5 total.
    assert contamination_rate_component(states) == pytest.approx(0.4)


def test_contamination_rate_component_from_boolean_mask():
    mask = np.array([True, False, True, True])
    assert contamination_rate_component(mask) == pytest.approx(0.75)


def test_contamination_rate_component_all_clean_is_zero():
    states = ["clean", "clean", "candidate_unvalidated"]
    assert contamination_rate_component(states) == pytest.approx(0.0)


def test_contamination_rate_component_rejects_empty_sequence():
    with pytest.raises(ValueError):
        contamination_rate_component([])


def test_contamination_rate_component_rejects_out_of_range_scalar():
    with pytest.raises(ValueError):
        contamination_rate_component(-0.2)


# ---------------------------------------------------------------------------
# Regression tests: a plain Python list[bool] (or a 0/1 int list/array) used
# to be silently miscomputed as a rate of 0.0, because a plain (non-ndarray)
# input was force-cast to `dtype=object`, defeating NumPy's automatic bool
# dtype inference and falling through to a string comparison that is always
# False for a bool/int operand. See the module docstring's "Score
# convention" section is unaffected -- this is purely about
# contamination_rate_component's input coercion.
# ---------------------------------------------------------------------------


def test_contamination_rate_component_from_plain_list_of_bools_not_silently_zero():
    """A plain `list[bool]` (not a numpy array) must be computed correctly,
    not silently coerced into a wrong (always ~0.0) rate."""
    mask = [True, False, True, True]  # 3/4 flagged
    assert contamination_rate_component(mask) == pytest.approx(0.75)


def test_contamination_rate_component_from_plain_tuple_of_bools():
    mask = (True, False, False, False)  # 1/4 flagged
    assert contamination_rate_component(mask) == pytest.approx(0.25)


def test_contamination_rate_component_from_plain_list_of_zero_one_ints():
    """A plain `list[int]` of 0/1 values is also boolean-like and must not
    be silently miscomputed as 0.0 either."""
    mask = [1, 0, 1, 1]  # 3/4 flagged
    assert contamination_rate_component(mask) == pytest.approx(0.75)


def test_contamination_rate_component_from_numpy_integer_zero_one_array():
    """A genuine numpy integer-dtype (not bool-dtype) 0/1 array is also
    boolean-like and was previously miscomputed even though it IS an
    ndarray (the original bug wasn't only about non-ndarray inputs)."""
    mask = np.array([1, 0, 1, 1])  # int64 dtype, not bool_
    assert contamination_rate_component(mask) == pytest.approx(0.75)


def test_contamination_rate_component_all_true_list_of_bools_is_one():
    assert contamination_rate_component([True, True, True]) == pytest.approx(1.0)


def test_contamination_rate_component_all_false_list_of_bools_is_zero():
    assert contamination_rate_component([False, False, False]) == pytest.approx(0.0)


def test_contamination_rate_component_rejects_invalid_integer_values():
    """An integer sequence with a value other than 0/1 is a caller mistake
    (e.g. a count or index array) and must raise, not be silently
    misinterpreted as either a boolean mask or a (never-matching) sequence
    of state labels."""
    with pytest.raises(ValueError):
        contamination_rate_component([0, 1, 2])


def test_contamination_rate_component_rejects_invalid_numpy_integer_values():
    with pytest.raises(ValueError):
        contamination_rate_component(np.array([0, 1, 2]))


def test_contamination_rate_component_string_states_still_work_after_fix():
    """Non-regression: the original three-state string-sequence path (a
    genuinely non-boolean-like sequence) must still work exactly as before
    the fix."""
    states = ["clean", "validated_contaminated", "validated_contaminated", "clean"]
    assert contamination_rate_component(states) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Regression test: VALIDATED_CONTAMINATED_STATE must exactly match the real
# sibling `contamination.py` module's `ContaminationStatus.VALIDATED_CONTAMINATED`
# enum member value. This module's docstring documents accepting "the
# sibling contamination.py module's three-state ... result" as a sequence of
# state strings -- a mismatched hardcoded constant (e.g. a hyphenated
# "validated-contaminated" vs. the real underscored "validated_contaminated")
# would silently produce a wrong (always-0.0-contributing) contamination
# rate for any caller who actually wires in real ContaminationStatus values,
# without ever raising an error. This test proves the documented contract
# works end-to-end using the sibling module's *real* enum values (converted
# to their string form, exactly as a real caller would do), not a
# hand-typed guess at the string.
# ---------------------------------------------------------------------------


def test_contamination_rate_component_matches_real_contamination_status_value():
    """Real `ContaminationStatus` enum values (as a real caller would supply
    them, via `.value`) must be correctly counted by
    `contamination_rate_component` -- proving the documented API contract
    with the sibling `contamination.py` module actually holds, not just
    that a hand-typed string literal happens to match another hand-typed
    string literal in this test file."""
    from dscraft.clean.contamination import ContaminationStatus

    states = [
        ContaminationStatus.CLEAN.value,
        ContaminationStatus.VALIDATED_CONTAMINATED.value,
        ContaminationStatus.CANDIDATE_UNVALIDATED.value,  # not counted
        ContaminationStatus.VALIDATED_CONTAMINATED.value,
        ContaminationStatus.CLEAN.value,
    ]
    # 2 validated_contaminated out of 5 total.
    assert contamination_rate_component(states) == pytest.approx(0.4)


def test_contamination_rate_component_matches_real_contamination_status_enum_members_directly():
    """Same as above, but passing the raw `ContaminationStatus` enum members
    themselves (not pre-extracted `.value` strings) through a plain Python
    list -- exercising the `dtype=object` fallback-to-string-comparison path
    in `_coerce_boolean_like_or_state_mask`, since `Enum` members are neither
    bool-like nor int-like. `Enum.__eq__` deliberately does not equal its own
    `.value` by default, so this only passes if `contamination_rate_component`
    is comparing against `.value`-equivalent strings correctly and the enum
    members compare unequal to `VALIDATED_CONTAMINATED_STATE` -- included to
    document that raw enum members are NOT directly supported and a caller
    must pass `.value` (or `str`) forms, consistent with this module's
    documented "plain inputs" contract."""
    from dscraft.clean.contamination import ContaminationStatus

    states = [
        ContaminationStatus.CLEAN,
        ContaminationStatus.VALIDATED_CONTAMINATED,
        ContaminationStatus.CANDIDATE_UNVALIDATED,
        ContaminationStatus.VALIDATED_CONTAMINATED,
        ContaminationStatus.CLEAN,
    ]
    # Raw enum members are not strings, so none compare equal to
    # VALIDATED_CONTAMINATED_STATE -- rate is 0.0 (not the 0.4 you'd get from
    # .value strings). This documents the boundary of the contract.
    assert contamination_rate_component(states) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# demographic_drift_component
# ---------------------------------------------------------------------------


def test_demographic_drift_component_identical_distributions_is_zero():
    train_groups = ["a", "a", "b", "b"]
    test_groups = ["a", "a", "b", "b"]
    assert demographic_drift_component(train_groups, test_groups) == pytest.approx(0.0, abs=1e-9)


def test_demographic_drift_component_disjoint_groups_is_one():
    """A group present in train but entirely absent from test (and vice
    versa) is the disjoint-support case -- normalized drift hits its max
    of 1.0."""
    train_groups = ["a", "a", "a"]
    test_groups = ["b", "b", "b"]
    assert demographic_drift_component(train_groups, test_groups) == pytest.approx(1.0, abs=1e-9)


def test_demographic_drift_component_partial_overlap_is_between_bounds():
    train_groups = ["a", "a", "b", "b"]
    test_groups = ["a", "b", "b", "b"]
    drift = demographic_drift_component(train_groups, test_groups)
    assert 0.0 < drift < 1.0


def test_demographic_drift_component_rejects_empty_train_groups():
    with pytest.raises(ValueError):
        demographic_drift_component([], ["a", "b"])


def test_demographic_drift_component_rejects_empty_test_groups():
    with pytest.raises(ValueError):
        demographic_drift_component(["a", "b"], [])


# ---------------------------------------------------------------------------
# dataset_integrity_score
# ---------------------------------------------------------------------------


def test_dataset_integrity_score_perfect_data_scores_one():
    """Zero label errors, zero contamination, identical group
    distributions -- the best possible score under the higher-is-better
    convention is exactly 1.0."""
    report = dataset_integrity_score(
        label_error_input=0.0,
        contamination_input=0.0,
        train_groups=["a", "a", "b", "b"],
        test_groups=["a", "a", "b", "b"],
    )
    assert isinstance(report, IntegrityReport)
    assert report.score == pytest.approx(1.0, abs=1e-9)
    assert report.label_error_rate == pytest.approx(0.0)
    assert report.contamination_rate == pytest.approx(0.0)
    assert report.drift == pytest.approx(0.0, abs=1e-9)


def test_dataset_integrity_score_worst_case_scores_zero():
    """Every component maxed out (all labels flagged, all test items
    validated-contaminated, fully disjoint group support) drives the score
    down to the worst possible 0.0."""
    report = dataset_integrity_score(
        label_error_input=1.0,
        contamination_input=1.0,
        train_groups=["a", "a"],
        test_groups=["b", "b"],
    )
    assert report.score == pytest.approx(0.0, abs=1e-9)


def test_dataset_integrity_score_default_equal_weights():
    report = dataset_integrity_score(
        label_error_input=0.3,
        contamination_input=0.3,
        train_groups=["a", "a", "b", "b"],
        test_groups=["a", "a", "b", "b"],  # zero drift
    )
    expected_bad = (1.0 / 3.0) * 0.3 + (1.0 / 3.0) * 0.3 + (1.0 / 3.0) * 0.0
    assert report.score == pytest.approx(1.0 - expected_bad)
    assert report.weights == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))


def test_dataset_integrity_score_custom_weights():
    weights = (0.5, 0.25, 0.25)
    report = dataset_integrity_score(
        label_error_input=0.2,
        contamination_input=0.4,
        train_groups=["a", "a", "b", "b"],
        test_groups=["a", "a", "b", "b"],  # zero drift
        weights=weights,
    )
    expected_bad = 0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.0
    assert report.score == pytest.approx(1.0 - expected_bad)
    assert report.weights == weights


def test_dataset_integrity_score_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        dataset_integrity_score(
            label_error_input=0.1,
            contamination_input=0.1,
            train_groups=["a", "b"],
            test_groups=["a", "b"],
            weights=(0.5, 0.5, 0.5),
        )


def test_dataset_integrity_score_report_exposes_breakdown():
    report = dataset_integrity_score(
        label_error_input=np.array([True, False, False, False]),  # 0.25
        contamination_input=["validated_contaminated", "clean"],  # 0.5
        train_groups=["a", "a", "b"],
        test_groups=["a", "b", "b"],
    )
    assert hasattr(report, "score")
    assert report.label_error_rate == pytest.approx(0.25)
    assert report.contamination_rate == pytest.approx(0.5)
    assert 0.0 <= report.drift <= 1.0
    assert report.score == pytest.approx(
        1.0 - (report.weights[0] * 0.25 + report.weights[1] * 0.5 + report.weights[2] * report.drift)
    )
