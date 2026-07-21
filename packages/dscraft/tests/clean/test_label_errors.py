"""Tests for DeCoLe: group-conditioned Confident Learning label-error detection.

The core scenario this file exists to prove (see
``test_prune_by_class_group_conditioning_avoids_naive_global_over_pruning``
below) is the exact failure mode described in the module docstring: a
single global per-class threshold, calibrated across two groups with very
different typical confidence levels, systematically flags a
low-confidence-but-correctly-labeled group's examples as likely mislabeled,
while DeCoLe's per-group thresholds do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.clean.label_errors import (
    GroupClassThresholds,
    GroupConfidentJoint,
    build_group_confident_joints,
    compute_group_class_thresholds,
    detect_label_errors,
    prune_by_both,
    prune_by_class,
    prune_by_noise_rate,
)


def _make_probs(class0_probs: list[float]) -> np.ndarray:
    """Build a 2-class probability matrix from a list of class-0 probabilities."""
    class0 = np.array(class0_probs, dtype=float)
    return np.stack([class0, 1.0 - class0], axis=1)


# ---------------------------------------------------------------------------
# Core claim: per-group thresholds avoid the naive-global-threshold
# over-pruning failure mode.
# ---------------------------------------------------------------------------

# Symmetric offsets (sum to exactly zero) applied around each group's own
# mean confidence, so each group's own per-group threshold equals exactly
# its construction mean and induces exactly 5 negative-scoring and 5
# positive-scoring members under group-conditioned thresholding.
_OFFSETS = [-0.025, -0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02, 0.025]

# Group A: high-confidence correct predictions (mean own-label confidence 0.95).
_GROUP_A_PROBS = [0.95 + offset for offset in _OFFSETS]
# Group B: systematically lower-confidence but still CORRECT predictions
# (mean own-label confidence 0.60, i.e. still > 0.5 -- these are not
# mislabeled examples, just examples the model is less sure about).
_GROUP_B_PROBS = [0.60 + offset for offset in _OFFSETS]


def _naive_vs_grouped_setup():
    """Build the 20-example, 2-group dataset described in the module docstring:
    group A is high-confidence-correct, group B is low-confidence-correct.
    All 20 labels are observed as class 0 and are, in fact, correctly labeled
    (no genuine label errors anywhere in this fixture) -- the only question
    is whether a pruning strategy incorrectly flags any of them.
    """
    labels = np.zeros(20, dtype=int)
    probs = _make_probs(_GROUP_A_PROBS + _GROUP_B_PROBS)
    true_groups = np.array(["A"] * 10 + ["B"] * 10, dtype=object)
    return labels, probs, true_groups


def test_prune_by_class_group_conditioning_avoids_naive_global_over_pruning():
    """A naive single global threshold over-prunes group B's correct labels;
    DeCoLe's per-group thresholds do not.

    Using a single fake group value for all 20 examples reduces
    ``compute_group_class_thresholds``/``build_group_confident_joints`` to
    exactly standard (non-group-conditioned) Confident Learning's global
    per-class threshold -- the naive baseline this test compares against,
    computed with this package's own (correct) machinery rather than a
    hand-rolled parallel implementation.
    """
    labels, probs, true_groups = _naive_vs_grouped_setup()
    fake_single_group = np.array(["all"] * 20, dtype=object)

    # --- Naive baseline: one global threshold for everyone. ---
    naive_thresholds = compute_group_class_thresholds(labels, probs, fake_single_group)
    # The single global threshold sits between the two groups' true means
    # (0.95 and 0.60), calibrated by the pooled average -- exactly the
    # "systematically over-prunes low-confidence minority group" scenario.
    assert naive_thresholds.per_group[("all", 0)] == pytest.approx(0.775, abs=1e-9)

    naive_joints = build_group_confident_joints(
        labels, probs, fake_single_group, thresholds=naive_thresholds
    )
    naive_mask = prune_by_class(naive_joints, n_samples=20, frac=0.5)

    naive_flagged_group_a = naive_mask[:10].sum()
    naive_flagged_group_b = naive_mask[10:].sum()
    # Naive global thresholding flags ALL of group B (whose correct labels
    # sit well below the pooled-average threshold) and NONE of group A.
    assert naive_flagged_group_a == 0
    assert naive_flagged_group_b == 10

    # --- DeCoLe: per-group thresholds. ---
    grouped_thresholds = compute_group_class_thresholds(labels, probs, true_groups)
    assert grouped_thresholds.per_group[("A", 0)] == pytest.approx(0.95, abs=1e-9)
    assert grouped_thresholds.per_group[("B", 0)] == pytest.approx(0.60, abs=1e-9)

    grouped_joints = build_group_confident_joints(
        labels, probs, true_groups, thresholds=grouped_thresholds
    )
    grouped_mask = prune_by_class(grouped_joints, n_samples=20, frac=0.5)

    grouped_flagged_group_a = grouped_mask[:10].sum()
    grouped_flagged_group_b = grouped_mask[10:].sum()
    # Per-group thresholding compares each example only against its OWN
    # group's typical confidence, so the "most anomalous half" is split
    # evenly across both groups (5 lowest-offset members of each) instead
    # of being concentrated entirely in group B.
    assert grouped_flagged_group_a == 5
    assert grouped_flagged_group_b == 5
    # The headline claim: DeCoLe flags far fewer of group B's genuinely
    # correct labels than the naive global-threshold baseline does.
    assert grouped_flagged_group_b < naive_flagged_group_b


def test_detect_label_errors_end_to_end_avoids_naive_over_pruning():
    """The composed detect_label_errors() convenience function reproduces
    the same group-conditioning fix as the manual three-step call above."""
    labels, probs, true_groups = _naive_vs_grouped_setup()
    mask = detect_label_errors(labels, probs, true_groups, strategy="class", frac=0.5)
    assert mask[:10].sum() == 5
    assert mask[10:].sum() == 5


# ---------------------------------------------------------------------------
# Step 1: compute_group_class_thresholds
# ---------------------------------------------------------------------------


def test_compute_group_class_thresholds_basic_means():
    """Each (group, class) threshold is the mean predicted probability for
    that class among that group's examples observed as that class."""
    labels = np.array([0, 0, 1, 1], dtype=int)
    probs = _make_probs([0.8, 0.6, 0.3, 0.1])
    groups = np.array(["x", "x", "x", "x"], dtype=object)

    thresholds = compute_group_class_thresholds(labels, probs, groups)
    assert isinstance(thresholds, GroupClassThresholds)
    assert thresholds.per_group[("x", 0)] == pytest.approx((0.8 + 0.6) / 2)
    # class 1's own probability column is 1 - class0_probs for rows labeled 1.
    assert thresholds.per_group[("x", 1)] == pytest.approx((0.7 + 0.9) / 2)
    assert thresholds.fallback_pairs == frozenset()


def test_compute_group_class_thresholds_zero_examples_pair_falls_back_to_global():
    """A (group, class) pair with zero examples falls back to the global
    per-class threshold, and is recorded in fallback_pairs."""
    labels = np.array([0, 0, 0, 1], dtype=int)
    probs = _make_probs([0.9, 0.8, 0.7, 0.2])
    # Group "A" has both classes; group "B" only ever has class 0 --
    # (B, 1) has zero examples.
    groups = np.array(["A", "A", "B", "A"], dtype=object)

    thresholds = compute_group_class_thresholds(labels, probs, groups)
    assert ("B", 1) in thresholds.fallback_pairs
    # Global class-1 threshold: mean of class-1 probability among the one
    # example labeled 1 (index 3, prob1 = 0.8).
    assert thresholds.global_per_class[1] == pytest.approx(0.8)
    assert thresholds.per_group[("B", 1)] == pytest.approx(0.8)
    # (B, 0) DID have an example (index 2), so it should NOT be a fallback.
    assert ("B", 0) not in thresholds.fallback_pairs
    assert thresholds.per_group[("B", 0)] == pytest.approx(0.7)


def test_compute_group_class_thresholds_group_with_only_one_class_present():
    """A group that only ever has one class observed still produces a valid
    threshold for that class, and a documented fallback for the other."""
    labels = np.array([0, 0, 0], dtype=int)
    probs = _make_probs([0.9, 0.85, 0.95])
    groups = np.array(["only-zero"] * 3, dtype=object)

    thresholds = compute_group_class_thresholds(labels, probs, groups)
    assert thresholds.per_group[("only-zero", 0)] == pytest.approx((0.9 + 0.85 + 0.95) / 3)
    assert ("only-zero", 1) in thresholds.fallback_pairs
    # Class 1 was never observed anywhere in this tiny dataset, so even the
    # global fallback is undefined (NaN) -- documented, not a crash.
    assert np.isnan(thresholds.global_per_class[1])
    assert np.isnan(thresholds.per_group[("only-zero", 1)])


def test_group_class_thresholds_get_falls_back_for_unseen_group():
    """GroupClassThresholds.get() falls back to the global per-class
    threshold for a group value that was never seen at all."""
    labels = np.array([0, 0, 1, 1], dtype=int)
    probs = _make_probs([0.9, 0.8, 0.3, 0.2])
    groups = np.array(["x", "x", "x", "x"], dtype=object)
    thresholds = compute_group_class_thresholds(labels, probs, groups)

    assert thresholds.get("never-seen-group", 0) == pytest.approx(thresholds.global_per_class[0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_rows_that_do_not_sum_to_one():
    labels = np.array([0, 1])
    probs = np.array([[0.5, 0.6], [0.4, 0.6]])  # row 0 sums to 1.1
    groups = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="sum to ~1.0"):
        compute_group_class_thresholds(labels, probs, groups)


def test_rejects_mismatched_lengths():
    labels = np.array([0, 1, 0])
    probs = _make_probs([0.9, 0.1])  # only 2 rows, labels has 3
    groups = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="same length"):
        compute_group_class_thresholds(labels, probs, groups)

    labels2 = np.array([0, 1])
    probs2 = _make_probs([0.9, 0.1])
    groups2 = np.array(["a", "b", "c"], dtype=object)  # groups has 3, others have 2
    with pytest.raises(ValueError, match="same length"):
        compute_group_class_thresholds(labels2, probs2, groups2)


def test_rejects_non_finite_probs():
    labels = np.array([0, 1, 0])
    probs = np.array([[0.9, 0.1], [np.nan, 1.0], [0.5, 0.5]])
    groups = np.array(["a", "b", "c"], dtype=object)
    with pytest.raises(ValueError, match="finite"):
        compute_group_class_thresholds(labels, probs, groups)

    probs_inf = np.array([[0.9, 0.1], [np.inf, -np.inf], [0.5, 0.5]])
    with pytest.raises(ValueError, match="finite"):
        compute_group_class_thresholds(labels, probs_inf, groups)


def test_rejects_2d_probs_shape_violation():
    labels = np.array([0, 1])
    probs_1d = np.array([0.9, 0.1])
    groups = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="2D"):
        compute_group_class_thresholds(labels, probs_1d, groups)


def test_rejects_out_of_range_integer_labels():
    labels = np.array([0, 2])  # 2 is out of range for a 2-class probs matrix
    probs = _make_probs([0.9, 0.1])
    groups = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="out-of-range"):
        compute_group_class_thresholds(labels, probs, groups)


def test_categorical_labels_are_mapped_to_indices():
    """String labels are accepted and internally mapped to indices by
    sorted order of the unique observed values."""
    labels = np.array(["cat", "dog", "cat", "dog"], dtype=object)
    probs = _make_probs([0.9, 0.2, 0.8, 0.1])  # column 0 = P(class "cat") since "cat" < "dog"
    groups = np.array(["g", "g", "g", "g"], dtype=object)

    thresholds = compute_group_class_thresholds(labels, probs, groups)
    # "cat" sorts before "dog" -> "cat" is class index 0, "dog" is class index 1.
    assert thresholds.per_group[("g", 0)] == pytest.approx((0.9 + 0.8) / 2)
    assert thresholds.per_group[("g", 1)] == pytest.approx((0.8 + 0.9) / 2)


def test_categorical_labels_reject_class_count_mismatch():
    """A categorical label set whose unique-value count doesn't match
    probs.shape[1] raises ValueError rather than guessing a mapping."""
    labels = np.array(["cat", "dog", "bird"], dtype=object)  # 3 unique values
    probs = _make_probs([0.9, 0.2, 0.5])  # only 2 columns
    groups = np.array(["g", "g", "g"], dtype=object)
    with pytest.raises(ValueError, match="unique categorical value"):
        compute_group_class_thresholds(labels, probs, groups)


# ---------------------------------------------------------------------------
# Step 2: build_group_confident_joints
# ---------------------------------------------------------------------------


def _noise_rate_fixture():
    """One group, 10 examples, one deliberately mislabeled example (index 9:
    observed label 0, but confidently predicted as class 1)."""
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=int)
    class0_probs = [
        0.90,
        0.90,
        0.90,
        0.90,
        0.90,  # indices 0-4: label 0, confidently correct
        0.15,
        0.12,
        0.08,
        0.05,  # indices 5-8: label 1, class-1 probs 0.85/0.88/0.92/0.95
        0.05,  # index 9: label 0 (observed), but predicted class 1 at 0.95
    ]
    probs = _make_probs(class0_probs)
    groups = np.array(["g"] * 10, dtype=object)
    return labels, probs, groups


def test_build_group_confident_joints_assigns_off_diagonal_for_mislabeled_example():
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)
    assert len(joints) == 1
    joint = joints[0]
    assert isinstance(joint, GroupConfidentJoint)
    assert joint.group == "g"

    # Index 9 (observed label 0) is confidently assigned to class 1 --
    # its own predicted probability for class 1 (0.95) exceeds class 1's
    # threshold (mean of indices 5-8's class-1 probs = 0.90), while its
    # class-0 probability (0.05) does not exceed class 0's threshold.
    local_pos = list(joint.global_indices).index(9)
    assert joint.assigned_class[local_pos] == 1
    assert joint.observed_labels[local_pos] == 0

    # That example lands in an off-diagonal confident-joint cell.
    assert joint.counts[0, 1] == 1
    assert joint.joint[0, 1] > 0.0
    assert not np.isnan(joint.assigned_class_joint_prob[local_pos])


def test_build_group_confident_joints_reuses_precomputed_thresholds():
    """Passing a precomputed GroupClassThresholds produces the same result
    as letting build_group_confident_joints compute it internally."""
    labels, probs, groups = _noise_rate_fixture()
    thresholds = compute_group_class_thresholds(labels, probs, groups)
    joints_precomputed = build_group_confident_joints(labels, probs, groups, thresholds=thresholds)
    joints_internal = build_group_confident_joints(labels, probs, groups)
    np.testing.assert_array_equal(joints_precomputed[0].counts, joints_internal[0].counts)
    np.testing.assert_allclose(joints_precomputed[0].joint, joints_internal[0].joint)


# ---------------------------------------------------------------------------
# Step 3: pruning strategies (tested independently)
# ---------------------------------------------------------------------------


def test_prune_by_noise_rate_flags_the_confidently_mislabeled_example():
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)

    mask = prune_by_noise_rate(joints, n_samples=10, frac=0.1)  # flag exactly 1 example
    assert mask.dtype == bool
    assert mask.sum() == 1
    assert mask[9]  # the deliberately mislabeled example
    assert not mask[:9].any()


def test_prune_by_noise_rate_flags_fewer_than_requested_when_no_candidates_exist():
    """If there are no off-diagonal confident-joint assignments at all,
    prune_by_noise_rate flags nothing, even if a large frac is requested --
    it never invents candidates."""
    labels = np.array([0, 0, 0], dtype=int)
    probs = _make_probs([0.9, 0.85, 0.95])
    groups = np.array(["g", "g", "g"], dtype=object)
    joints = build_group_confident_joints(labels, probs, groups)

    mask = prune_by_noise_rate(joints, n_samples=3, frac=1.0)
    assert mask.sum() == 0


def test_prune_by_class_flags_the_anomalously_low_confidence_example():
    """One example (index 4) has an own-label confidence far below its
    group/class's typical (mean) behavior -- prune_by_class flags exactly
    that example, regardless of whether it has any confident-joint
    assignment at all (this group only ever has class 0, so class 1's
    threshold is an undefined global fallback and index 4 gets
    assigned_class == -1)."""
    labels = np.array([0, 0, 0, 0, 0], dtype=int)
    probs = _make_probs([0.90, 0.92, 0.88, 0.91, 0.30])
    groups = np.array(["g"] * 5, dtype=object)
    joints = build_group_confident_joints(labels, probs, groups)

    # Sanity check the "only one class present" + "no assignment" premise.
    local_pos = list(joints[0].global_indices).index(4)
    assert joints[0].assigned_class[local_pos] == -1

    mask = prune_by_class(joints, n_samples=5, n=1)
    assert mask.sum() == 1
    assert mask[4]
    assert not mask[:4].any()


def test_prune_by_class_excludes_examples_with_undefined_threshold():
    """An example whose (group, observed-label) threshold is NaN (that
    class is never observed anywhere in the dataset) is excluded from
    prune_by_class ranking entirely, even under a large frac."""
    labels = np.array([0, 0, 0], dtype=int)
    probs = _make_probs([0.9, 0.85, 0.95])
    groups = np.array(["g", "g", "g"], dtype=object)
    joints = build_group_confident_joints(labels, probs, groups)

    mask = prune_by_class(joints, n_samples=3, frac=1.0)
    # All 3 examples have a well-defined class-0 threshold (they ARE the
    # class-0 examples), so all 3 are eligible and, with frac=1.0, all
    # get flagged -- this asserts the eligible pool is exactly {0, 1, 2},
    # not silently empty or silently including a phantom NaN entry.
    assert mask.sum() == 3


def test_pruning_functions_reject_invalid_frac_and_n():
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)

    with pytest.raises(ValueError):
        prune_by_noise_rate(joints, n_samples=10, frac=1.5)
    with pytest.raises(ValueError):
        prune_by_class(joints, n_samples=10, frac=-0.1)
    with pytest.raises(ValueError):
        prune_by_noise_rate(joints, n_samples=10, n=11)  # exceeds n_samples
    with pytest.raises(ValueError):
        prune_by_class(joints, n_samples=10, n=-1)


def test_n_takes_precedence_over_frac():
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)
    # frac would resolve to 5 (frac=0.5 * 10), but n=1 must win.
    mask = prune_by_noise_rate(joints, n_samples=10, frac=0.5, n=1)
    assert mask.sum() == 1


def test_prune_by_both_is_the_intersection_of_noise_rate_and_class():
    """prune_by_both is a pure composition -- it must always equal the
    element-wise AND of prune_by_noise_rate and prune_by_class run
    independently with the same frac/n budget, mirroring cleanlab's
    find_label_issues(filter_by="both")."""
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)

    noise_rate_mask = prune_by_noise_rate(joints, n_samples=10, frac=0.5)
    class_mask = prune_by_class(joints, n_samples=10, frac=0.5)
    both_mask = prune_by_both(joints, n_samples=10, frac=0.5)

    assert both_mask.dtype == bool
    np.testing.assert_array_equal(both_mask, noise_rate_mask & class_mask)
    # The deliberately mislabeled example (index 9) is the strongest
    # candidate under both strategies, so it must survive the intersection.
    assert both_mask[9]


def test_prune_by_both_can_be_empty_even_when_class_alone_flags_examples():
    """When every observed label belongs to a single class, prune_by_noise_rate
    can never find an off-diagonal confident-joint assignment (there is no
    "other class" to be confidently reassigned to), so its mask is always
    empty -- and intersecting with prune_by_class (which *does* flag
    something here, see test_prune_by_class_flags_the_anomalously_low_confidence_example)
    must therefore also be empty. This is the genuinely-stricter behavior
    "both" is expected to have relative to either strategy alone."""
    labels = np.array([0, 0, 0, 0, 0], dtype=int)
    probs = _make_probs([0.90, 0.92, 0.88, 0.91, 0.30])
    groups = np.array(["g"] * 5, dtype=object)
    joints = build_group_confident_joints(labels, probs, groups)

    class_mask = prune_by_class(joints, n_samples=5, frac=1.0)
    assert class_mask.sum() == 5  # every example is eligible and flagged at frac=1.0

    both_mask = prune_by_both(joints, n_samples=5, frac=1.0)
    assert both_mask.sum() == 0


def test_prune_by_both_rejects_invalid_frac_and_n():
    labels, probs, groups = _noise_rate_fixture()
    joints = build_group_confident_joints(labels, probs, groups)

    with pytest.raises(ValueError):
        prune_by_both(joints, n_samples=10, frac=1.5)
    with pytest.raises(ValueError):
        prune_by_both(joints, n_samples=10, n=-1)


# ---------------------------------------------------------------------------
# Step 4: detect_label_errors
# ---------------------------------------------------------------------------


def test_detect_label_errors_noise_rate_strategy():
    labels, probs, groups = _noise_rate_fixture()
    mask = detect_label_errors(labels, probs, groups, strategy="noise_rate", frac=0.1)
    assert mask.sum() == 1
    assert mask[9]


def test_detect_label_errors_class_strategy():
    labels = np.array([0, 0, 0, 0, 0], dtype=int)
    probs = _make_probs([0.90, 0.92, 0.88, 0.91, 0.30])
    groups = np.array(["g"] * 5, dtype=object)
    mask = detect_label_errors(labels, probs, groups, strategy="class", n=1)
    assert mask.sum() == 1
    assert mask[4]


def test_detect_label_errors_both_strategy():
    labels, probs, groups = _noise_rate_fixture()
    mask = detect_label_errors(labels, probs, groups, strategy="both", frac=0.5)
    noise_rate_mask = detect_label_errors(labels, probs, groups, strategy="noise_rate", frac=0.5)
    class_mask = detect_label_errors(labels, probs, groups, strategy="class", frac=0.5)
    np.testing.assert_array_equal(mask, noise_rate_mask & class_mask)
    assert mask[9]


def test_detect_label_errors_rejects_unknown_strategy():
    labels, probs, groups = _noise_rate_fixture()
    with pytest.raises(ValueError, match="strategy"):
        detect_label_errors(labels, probs, groups, strategy="bogus")


def test_detect_label_errors_default_strategy_is_noise_rate():
    labels, probs, groups = _noise_rate_fixture()
    default_mask = detect_label_errors(labels, probs, groups, frac=0.1)
    explicit_mask = detect_label_errors(labels, probs, groups, strategy="noise_rate", frac=0.1)
    np.testing.assert_array_equal(default_mask, explicit_mask)
