"""DeCoLe: group-conditioned Confident Learning for tabular label-error detection.

Standard Confident Learning (Northcutt et al.'s "confident joint" framework)
estimates a single, global, per-class self-confidence threshold from
out-of-sample predicted probabilities, uses it to build one global "confident
joint" count matrix estimating the latent (true-label, observed-label) joint
distribution, and prunes the examples that matrix says are most likely
mislabeled. That approach implicitly assumes the probability of a label
being flipped depends only on the *true class*, never on the example's
group/demographic attribute or features.

**DeCoLe's fix, implemented in this module:** partition examples by a
sensitive/demographic group attribute and compute fully independent
per-group thresholds and per-group confident joints (see
:func:`compute_group_class_thresholds` and
:func:`build_group_confident_joints`). A single global threshold, calibrated
mostly by whichever group dominates the dataset (or simply has better
per-class model confidence), systematically over-prunes correct-but-
low-confidence minority-group examples: model confidence is frequently
data-sparsity-biased against minority groups even when their labels are
correct, which is exactly the real-world group-conditioned annotation-bias
scenario the standard Confident Learning assumption above breaks under.
Computing one threshold *per (group, class) pair* instead means a group
that is systematically less confident (even on correctly-labeled examples)
gets its own, lower bar -- it is compared against its own typical
confidence, not against a different group's.

This module is a from-scratch NumPy implementation. It never imports or
wraps ``cleanlab`` (AGPL-3.0) -- see CLAUDE.md's licensing policy (LazyRed
Part 5 / "Licensing policy (LazyIsolate)"): AGPL/GPL code is never an
acceptable runtime dependency of this 100%-MIT platform, and confident-joint
label-error detection is a public algorithm (not cleanlab-proprietary code),
so this is a clean, independent implementation of that public algorithm
rather than a derivative of any specific package.

No PyTorch, no scikit-learn training -- this module only does array math
over already-computed out-of-sample predicted probabilities (``probs``),
which callers are expected to have produced themselves (e.g. via k-fold
cross-validation with any classifier). Depends only on NumPy, matching
``dscraft.clean``'s existing dependency footprint (see ``embeddings.py``'s
module docstring and this package's ``clean`` extra in ``pyproject.toml``).

Public entrypoints, in the order a caller would typically use them:

1. :func:`compute_group_class_thresholds` -- per-(group, class) thresholds.
2. :func:`build_group_confident_joints` -- per-group confident-joint
   matrices ``Q_g``.
3. :func:`prune_by_noise_rate` / :func:`prune_by_class` / :func:`prune_by_both`
   -- pruning strategies over the confident joints, each returning a
   boolean mask. ``prune_by_both`` is a thin composition of the first two
   (their intersection), mirroring cleanlab's ``find_label_issues(filter_by=
   "both")`` mode -- see its own docstring below.
4. :func:`detect_label_errors` -- the one-call convenience entrypoint that
   composes 1-3, matching this package's "one clear entrypoint" pattern
   (see ``dscraft.clean.detect_near_duplicate_text`` for the analogous
   entrypoint in ``dedup.py``/``embeddings.py``).

**Expected input form for ``labels``.** ``labels`` (the *observed*, possibly
noisy labels) may be given as either integer class indices in
``[0, n_classes)`` matching ``probs``'s column order, or as string/object/
categorical values, which are internally mapped to indices by sorting the
unique observed values and assigning indices in that sorted order. See
:func:`_validate_and_encode` for the exact rule and its failure mode (a
class that is never observed in ``labels`` at all cannot be safely mapped to
a ``probs`` column, so that case raises ``ValueError`` rather than guessing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "GroupClassThresholds",
    "GroupConfidentJoint",
    "compute_group_class_thresholds",
    "build_group_confident_joints",
    "prune_by_noise_rate",
    "prune_by_class",
    "prune_by_both",
    "detect_label_errors",
]


# ---------------------------------------------------------------------------
# Shared validation / label encoding
# ---------------------------------------------------------------------------


def _sorted_unique(values: np.ndarray) -> list[Any]:
    """Return the distinct values in ``values`` in sorted order.

    Falls back to first-seen (insertion) order if the values are not
    mutually orderable (``sorted()`` raises ``TypeError``, e.g. genuinely
    mixed-type group/label values) -- still fully deterministic given a
    fixed input array, just not sorted.
    """
    items = values.tolist()
    try:
        return sorted(set(items))
    except TypeError:
        seen: dict[Any, None] = {}
        for item in items:
            seen.setdefault(item, None)
        return list(seen.keys())


def _validate_and_encode(
    labels: Any, probs: Any, groups: Any
) -> tuple[np.ndarray, np.ndarray, int, list[Any]]:
    """Validate ``labels``/``probs``/``groups`` and encode labels as class indices.

    ``probs`` must be a dense ``(n_samples, n_classes)`` array of
    out-of-sample predicted class probabilities (e.g. from k-fold
    cross-validation), with every row finite and summing to ~1.0 (within
    ``atol=1e-3``, to tolerate ordinary floating-point roundoff without
    silently accepting probabilities that don't actually form a
    distribution).

    ``labels`` may be given as either:

    - an integer (or boolean) dtype array -- used directly as class indices,
      each required to satisfy ``0 <= label < n_classes`` where
      ``n_classes = probs.shape[1]``; or
    - any other dtype (string, object, float, ...) -- treated as
      categorical: the unique observed values are sorted and mapped to
      indices ``0..n_classes-1`` in that sorted order. This requires the
      number of unique observed values to exactly equal ``n_classes``; a
      class that is never observed anywhere in ``labels`` cannot be safely
      matched up to a specific ``probs`` column, so a mismatch raises
      ``ValueError`` naming the two counts rather than guessing an
      arbitrary mapping.

    ``groups`` is a 1D array-like of any hashable group/demographic
    attribute value (e.g. strings), the same length as ``labels``/``probs``.

    Returns ``(labels_idx, groups_arr, n_classes, class_names)``, where
    ``labels_idx`` is an ``int`` array of encoded class indices and
    ``class_names`` records what each index ``0..n_classes-1`` originally
    meant (``list(range(n_classes))`` for the integer-label case, or the
    sorted unique observed values for the categorical case).
    """
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim != 2:
        raise ValueError(
            f"probs must be a 2D (n_samples, n_classes) array, got shape {probs_arr.shape!r}."
        )
    if not np.isfinite(probs_arr).all():
        raise ValueError("probs must contain only finite values (no NaN/Inf).")

    row_sums = probs_arr.sum(axis=1)
    close_to_one = np.isclose(row_sums, 1.0, atol=1e-3)
    if not close_to_one.all():
        bad_rows = np.where(~close_to_one)[0]
        raise ValueError(
            "probs rows must each sum to ~1.0 (within atol=1e-3); "
            f"{bad_rows.size} row(s) violate this, e.g. row {int(bad_rows[0])} sums to "
            f"{row_sums[bad_rows[0]]!r}."
        )

    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups, dtype=object)
    n_samples = probs_arr.shape[0]
    if labels_arr.shape[0] != n_samples or groups_arr.shape[0] != n_samples:
        raise ValueError(
            "labels, probs, and groups must all have the same length along axis 0; got "
            f"len(labels)={labels_arr.shape[0]}, probs.shape[0]={n_samples}, "
            f"len(groups)={groups_arr.shape[0]}."
        )

    n_classes = probs_arr.shape[1]
    class_names: list[Any]
    if np.issubdtype(labels_arr.dtype, np.integer) or labels_arr.dtype == np.bool_:
        labels_idx = labels_arr.astype(int)
        out_of_range = (labels_idx < 0) | (labels_idx >= n_classes)
        if out_of_range.any():
            bad_pos = int(np.where(out_of_range)[0][0])
            raise ValueError(
                f"labels contains an out-of-range class index {labels_idx[bad_pos]!r} at "
                f"position {bad_pos}; expected 0 <= label < n_classes ({n_classes}), since "
                "probs has that many columns."
            )
        class_names = list(range(n_classes))
    else:
        unique_values = _sorted_unique(labels_arr)
        if len(unique_values) != n_classes:
            raise ValueError(
                f"labels has {len(unique_values)} unique categorical value(s) "
                f"({unique_values!r}) but probs has {n_classes} column(s); these counts must "
                "match exactly so each unique observed label maps to exactly one probs column."
            )
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        labels_idx = np.array(
            [value_to_index[value] for value in labels_arr.tolist()], dtype=int
        )
        class_names = unique_values

    return labels_idx, groups_arr, n_classes, class_names


def _resolve_count(n_samples: int, *, frac: float, n: int | None) -> int:
    """Resolve a ``frac``/``n`` pair into a concrete flag count.

    ``n`` takes precedence over ``frac`` when both are given (``frac`` has a
    non-``None`` default, ``n`` does not, so an explicit ``n`` always
    reflects deliberate caller intent). ``n`` must be a non-negative integer
    no larger than ``n_samples``; ``frac`` must be in ``[0.0, 1.0]``.
    """
    if n is not None:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n!r}.")
        if n > n_samples:
            raise ValueError(f"n ({n}) cannot exceed n_samples ({n_samples}).")
        return int(n)
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"frac must be in [0.0, 1.0], got {frac!r}.")
    return int(round(frac * n_samples))


# ---------------------------------------------------------------------------
# Step 1: per-group, per-class confidence thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupClassThresholds:
    """Per-(group, class) self-confidence thresholds ``t_{j,g}``.

    ``per_group`` maps ``(group, class_index) -> threshold`` for every
    observed group and every class ``0..n_classes-1``, including pairs that
    had zero examples (see :func:`compute_group_class_thresholds` for the
    documented fallback). ``fallback_pairs`` records exactly which
    ``(group, class_index)`` pairs used the global fallback rather than
    their own per-group mean, so callers/tests can distinguish "computed
    from real per-group data" from "fell back" without re-deriving it.
    """

    per_group: dict[tuple[Any, int], float]
    global_per_class: np.ndarray
    fallback_pairs: frozenset[tuple[Any, int]]
    class_names: list[Any]

    def get(self, group: Any, class_index: int) -> float:
        """Return ``t_{class_index, group}``.

        Falls back to ``global_per_class[class_index]`` if ``(group,
        class_index)`` is not a key in :attr:`per_group` at all (e.g. a
        group value not seen when these thresholds were computed) -- this
        mirrors, and is consistent with, the zero-examples fallback already
        baked into :attr:`per_group` itself by
        :func:`compute_group_class_thresholds`.
        """
        return self.per_group.get((group, class_index), self.global_per_class[class_index])


def compute_group_class_thresholds(labels: Any, probs: Any, groups: Any) -> GroupClassThresholds:
    """Compute per-(group, class) self-confidence thresholds ``t_{j,g}``.

    For each unique group ``g`` and each class ``j``, ``t_{j,g}`` is the
    mean predicted probability for class ``j`` (i.e. ``probs[:, j]``) among
    examples in group ``g`` whose *observed* (possibly noisy) label is
    ``j``. This is the group-conditioned analogue of standard Confident
    Learning's single global per-class threshold -- see the module
    docstring for why conditioning on group matters.

    **Zero-examples fallback (documented, consistent, testable choice):** if
    a ``(group, class)`` pair has zero examples where group is ``g`` and the
    observed label is ``j``, no per-group mean can be computed for it. This
    function falls back to the **global per-class threshold** for that
    class (the same quantity standard, non-group-conditioned Confident
    Learning would use: the mean of ``probs[:, j]`` over *all* examples
    whose observed label is ``j``, regardless of group). That fallback pair
    is recorded in the returned :class:`GroupClassThresholds`'s
    ``fallback_pairs``. If a class ``j`` is never observed *anywhere* in
    ``labels`` (not even globally), its global fallback threshold is itself
    ``NaN`` -- there is no data anywhere to estimate a threshold from, and
    that ``NaN`` is intentionally left as-is (see
    :func:`build_group_confident_joints`, where a ``NaN`` threshold means
    that class can never be selected as "exceeded" for any example).

    Args:
        labels: 1D array-like of observed (possibly noisy) labels -- see
            :func:`_validate_and_encode` for the accepted integer-index vs.
            categorical input forms.
        probs: ``(n_samples, n_classes)`` array of out-of-sample predicted
            class probabilities; each row must be finite and sum to ~1.0.
        groups: 1D array-like of group/demographic attribute values, same
            length as ``labels``/``probs``.

    Returns:
        A :class:`GroupClassThresholds` with a threshold for every
        ``(group, class)`` pair (falling back where documented above), plus
        the global per-class thresholds and the set of pairs that used the
        fallback.
    """
    labels_idx, groups_arr, n_classes, class_names = _validate_and_encode(labels, probs, groups)
    probs_arr = np.asarray(probs, dtype=float)

    global_per_class = np.full(n_classes, np.nan, dtype=float)
    for j in range(n_classes):
        class_mask = labels_idx == j
        if class_mask.any():
            global_per_class[j] = probs_arr[class_mask, j].mean()

    unique_groups = _sorted_unique(groups_arr)
    per_group: dict[tuple[Any, int], float] = {}
    fallback_pairs: set[tuple[Any, int]] = set()
    for group in unique_groups:
        group_mask = groups_arr == group
        for j in range(n_classes):
            pair_mask = group_mask & (labels_idx == j)
            if pair_mask.any():
                per_group[(group, j)] = float(probs_arr[pair_mask, j].mean())
            else:
                fallback_pairs.add((group, j))
                per_group[(group, j)] = float(global_per_class[j])

    return GroupClassThresholds(
        per_group=per_group,
        global_per_class=global_per_class,
        fallback_pairs=frozenset(fallback_pairs),
        class_names=class_names,
    )


# ---------------------------------------------------------------------------
# Step 2: per-group confident-joint matrices
# ---------------------------------------------------------------------------


@dataclass
class GroupConfidentJoint:
    """One group's confident-joint matrix plus per-member scoring detail.

    ``counts`` is the raw, unnormalized ``[n_classes, n_classes]`` integer
    confident-joint matrix ``C_g`` (see :func:`build_group_confident_joints`
    for its construction rule). ``joint`` is ``Q_g``, ``counts`` calibrated
    against this group's observed noisy-label marginal and normalized to sum
    to 1.0 (or all-zero if this group produced no confidently-assigned
    examples at all).

    The remaining fields carry one entry per member of this group (in the
    same order as ``global_indices``), letting :func:`prune_by_noise_rate`
    and :func:`prune_by_class` rank individual examples without
    recomputing anything from ``counts``/``joint``:

    - ``global_indices``: this member's row index into the original
      ``labels``/``probs``/``groups`` arrays.
    - ``observed_labels``: this member's observed label, as a class index.
    - ``assigned_class``: the confident-joint-assigned class index (the
      ``argmax`` over classes whose per-group threshold this member's
      predicted probability exceeded), or ``-1`` if no class's threshold
      was exceeded at all (this member contributed to no cell of
      ``counts``).
    - ``assigned_class_joint_prob``: ``joint[observed_label,
      assigned_class]`` for this member, or ``NaN`` if ``assigned_class``
      is ``-1``.
    - ``own_label_prob``: ``probs[row, observed_label]`` -- this member's
      predicted probability for its *own* observed label.
    - ``own_label_threshold``: ``t_{observed_label, group}`` for this
      member (from :class:`GroupClassThresholds`), i.e. what this member's
      own group/class combination's *typical* confidence looks like.
    """

    group: Any
    counts: np.ndarray
    joint: np.ndarray
    global_indices: np.ndarray
    observed_labels: np.ndarray
    assigned_class: np.ndarray
    assigned_class_joint_prob: np.ndarray
    own_label_prob: np.ndarray
    own_label_threshold: np.ndarray


def build_group_confident_joints(
    labels: Any,
    probs: Any,
    groups: Any,
    *,
    thresholds: GroupClassThresholds | None = None,
) -> list[GroupConfidentJoint]:
    """Build one confident-joint matrix ``Q_g`` per group.

    For each group ``g``, this constructs an unnormalized
    ``[n_classes, n_classes]`` integer count matrix ``C_g`` where
    ``C_g[i, j]`` counts examples in group ``g`` whose observed label is
    ``i`` and whose predicted probability for class ``j`` both (a) exceeds
    that group's own threshold ``t_{j,g}`` (see
    :func:`compute_group_class_thresholds`) and (b) is the ``argmax`` over
    every class whose threshold was exceeded -- the standard confident-joint
    construction rule, scoped to one group's own labels and thresholds
    instead of global ones. An example whose predicted probabilities don't
    exceed *any* class's threshold contributes to no cell of ``C_g`` at all
    (this is standard Confident Learning behavior, not a bug: such an
    example is not confidently associated with any class).

    ``C_g`` is then calibrated against group ``g``'s own observed
    noisy-label marginal (the count of examples in ``g`` whose observed
    label is each ``i``, over *all* of ``g``'s members, not just the ones
    that landed in some cell of ``C_g``) by rescaling each row ``i`` of
    ``C_g`` so it sums to that marginal count, then normalizing the whole
    calibrated matrix to sum to 1.0 -- yielding ``Q_g``, this group's
    estimate of the latent joint distribution over (true label, observed
    label). A row of `C_g` that is entirely zero (no example with that
    observed label was ever confidently assigned to any class) is left as
    zero after calibration rather than dividing by zero.

    Args:
        labels: 1D array-like of observed labels; see
            :func:`_validate_and_encode`.
        probs: ``(n_samples, n_classes)`` out-of-sample predicted
            probabilities.
        groups: 1D array-like of group/demographic attribute values.
        thresholds: precomputed :class:`GroupClassThresholds`; if ``None``
            (the default), computed internally via
            :func:`compute_group_class_thresholds`. Passing a precomputed
            value lets callers reuse thresholds across multiple calls
            without recomputing them.

    Returns:
        A list of one :class:`GroupConfidentJoint` per unique group
        (deterministically ordered by :func:`_sorted_unique` over
        ``groups``).
    """
    labels_idx, groups_arr, n_classes, _class_names = _validate_and_encode(labels, probs, groups)
    probs_arr = np.asarray(probs, dtype=float)
    if thresholds is None:
        thresholds = compute_group_class_thresholds(labels, probs, groups)

    unique_groups = _sorted_unique(groups_arr)
    joints: list[GroupConfidentJoint] = []
    for group in unique_groups:
        member_indices = np.where(groups_arr == group)[0]
        threshold_row = np.array(
            [thresholds.get(group, j) for j in range(n_classes)], dtype=float
        )

        group_probs = probs_arr[member_indices]
        group_labels = labels_idx[member_indices]

        # exceeds[m, j] is False whenever threshold_row[j] is NaN (a class
        # never observed anywhere in the dataset) -- NumPy comparisons
        # against NaN are always False, so that class can never be
        # "exceeded" and is correctly excluded from candidacy below.
        exceeds = group_probs > threshold_row[np.newaxis, :]
        any_exceeds = exceeds.any(axis=1)
        masked_probs = np.where(exceeds, group_probs, -np.inf)
        assigned_class = np.where(any_exceeds, np.argmax(masked_probs, axis=1), -1)

        counts = np.zeros((n_classes, n_classes), dtype=int)
        valid = assigned_class != -1
        if valid.any():
            np.add.at(counts, (group_labels[valid], assigned_class[valid]), 1)

        label_counts = np.array(
            [(group_labels == i).sum() for i in range(n_classes)], dtype=float
        )
        row_sums = counts.sum(axis=1).astype(float)
        calibrated = np.zeros((n_classes, n_classes), dtype=float)
        nonzero_rows = row_sums > 0
        calibrated[nonzero_rows, :] = (
            counts[nonzero_rows, :]
            / row_sums[nonzero_rows, np.newaxis]
            * label_counts[nonzero_rows, np.newaxis]
        )
        total = calibrated.sum()
        joint = calibrated / total if total > 0 else calibrated

        own_label_prob = group_probs[np.arange(group_probs.shape[0]), group_labels]
        own_label_threshold = threshold_row[group_labels]

        assigned_class_joint_prob = np.full(member_indices.shape[0], np.nan, dtype=float)
        if valid.any():
            assigned_class_joint_prob[valid] = joint[
                group_labels[valid], assigned_class[valid]
            ]

        joints.append(
            GroupConfidentJoint(
                group=group,
                counts=counts,
                joint=joint,
                global_indices=member_indices,
                observed_labels=group_labels,
                assigned_class=assigned_class,
                assigned_class_joint_prob=assigned_class_joint_prob,
                own_label_prob=own_label_prob,
                own_label_threshold=own_label_threshold,
            )
        )
    return joints


# ---------------------------------------------------------------------------
# Step 3: pruning strategies
# ---------------------------------------------------------------------------


def prune_by_noise_rate(
    joints: Sequence[GroupConfidentJoint],
    n_samples: int,
    *,
    frac: float = 0.05,
    n: int | None = None,
) -> np.ndarray:
    """Flag examples in the highest-estimated-noise-rate confident-joint cells.

    A candidate example is one with a confident-joint assignment
    (``assigned_class != -1``, see :func:`build_group_confident_joints`)
    that *disagrees* with its own observed label (``assigned_class !=
    observed_label``) -- i.e. the model confidently predicts this example
    belongs to a different class than the one it was annotated with.
    Candidates are pooled across every group in ``joints`` and ranked by
    their group's estimated joint probability for that specific
    ``(observed, assigned)`` cell (``Q_g[observed, assigned]``, i.e.
    ``assigned_class_joint_prob``) -- the cells estimated to contain the
    largest fraction of true label noise are flagged first, which is the
    "prune by noise rate" strategy's namesake.

    The top ``n`` (or ``round(frac * n_samples)`` if ``n`` is not given,
    default ``frac=0.05`` -- flag ~5% of examples, a middle-of-the-road
    assumption for real-world tabular label-noise prevalence) candidates by
    that score are flagged. **If fewer candidates exist than requested,
    every candidate is flagged and no more** -- this is not an error; it
    simply means the confident joints found fewer likely-mislabeled
    examples than the requested budget.

    Args:
        joints: per-group confident joints from
            :func:`build_group_confident_joints`.
        n_samples: total number of examples in the original dataset (used
            to size the returned mask and to resolve ``frac``).
        frac: fraction of ``n_samples`` to flag when ``n`` is not given.
            Must be in ``[0.0, 1.0]``.
        n: exact number of examples to flag; takes precedence over ``frac``
            when given. Must satisfy ``0 <= n <= n_samples``.

    Returns:
        A boolean mask of shape ``(n_samples,)``; ``True`` marks a flagged
        (likely-mislabeled) example.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples!r}.")
    count = _resolve_count(n_samples, frac=frac, n=n)

    index_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    for joint in joints:
        off_diagonal = (joint.assigned_class != -1) & (
            joint.assigned_class != joint.observed_labels
        )
        if not off_diagonal.any():
            continue
        index_chunks.append(joint.global_indices[off_diagonal])
        score_chunks.append(joint.assigned_class_joint_prob[off_diagonal])

    mask = np.zeros(n_samples, dtype=bool)
    if not index_chunks or count == 0:
        return mask

    indices = np.concatenate(index_chunks)
    scores = np.concatenate(score_chunks)
    order = np.argsort(-scores, kind="stable")  # highest estimated noise probability first
    flagged = indices[order[:count]]
    mask[flagged] = True
    return mask


def prune_by_class(
    joints: Sequence[GroupConfidentJoint],
    n_samples: int,
    *,
    frac: float = 0.05,
    n: int | None = None,
) -> np.ndarray:
    """Flag examples whose own-label confidence is most anomalously low for their group/class.

    Every member of every group in ``joints`` (not just those with a
    confident-joint assignment) is scored by
    ``own_label_prob - own_label_threshold``: how far below its own
    ``(group, observed-label)`` typical confidence (the mean, per
    :func:`compute_group_class_thresholds`) this example's predicted
    probability for its *own* observed label falls. The most negative
    scores -- examples the model is anomalously unconfident about, relative
    to what is typical for other examples sharing their group and observed
    label -- are flagged first. This is the group-conditioned analogue of
    "this example doesn't look like it genuinely belongs to its assigned
    class": conditioning the anomaly threshold on group is exactly the
    DeCoLe fix described in the module docstring, applied here instead of
    to a single global per-class average.

    Examples whose ``(group, observed-label)`` threshold is ``NaN`` (that
    class was never observed anywhere in the dataset at all -- see
    :func:`compute_group_class_thresholds`) are excluded from ranking
    entirely: "anomalously low relative to typical behavior" is undefined
    without any typical-behavior baseline to compare against.

    The top ``n`` (or ``round(frac * n_samples)``, default ``frac=0.05``)
    lowest-scoring examples are flagged. As with :func:`prune_by_noise_rate`,
    if fewer scoreable examples exist than requested, every scoreable
    example is flagged and no more.

    Args:
        joints: per-group confident joints from
            :func:`build_group_confident_joints`.
        n_samples: total number of examples in the original dataset.
        frac: fraction of ``n_samples`` to flag when ``n`` is not given.
        n: exact number of examples to flag; takes precedence over ``frac``.

    Returns:
        A boolean mask of shape ``(n_samples,)``; ``True`` marks a flagged
        (likely-mislabeled) example.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples!r}.")
    count = _resolve_count(n_samples, frac=frac, n=n)

    index_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    for joint in joints:
        score = joint.own_label_prob - joint.own_label_threshold
        valid = ~np.isnan(score)
        if not valid.any():
            continue
        index_chunks.append(joint.global_indices[valid])
        score_chunks.append(score[valid])

    mask = np.zeros(n_samples, dtype=bool)
    if not index_chunks or count == 0:
        return mask

    indices = np.concatenate(index_chunks)
    scores = np.concatenate(score_chunks)
    order = np.argsort(scores, kind="stable")  # most anomalously low score first
    flagged = indices[order[:count]]
    mask[flagged] = True
    return mask


def prune_by_both(
    joints: Sequence[GroupConfidentJoint],
    n_samples: int,
    *,
    frac: float = 0.05,
    n: int | None = None,
) -> np.ndarray:
    """Flag examples that both :func:`prune_by_noise_rate` and :func:`prune_by_class` would flag.

    This is the group-conditioned analogue of cleanlab's
    ``find_label_issues(filter_by="both")`` mode: run both pruning
    strategies independently (each with the same ``frac``/``n`` budget) and
    intersect their flagged sets, rather than introducing a new ranking
    criterion of its own. Intersecting two independently-derived signals --
    "confidently mislabeled relative to another class" (noise rate) and
    "anomalously low own-label confidence for this group/class" (class) --
    is a stricter, higher-precision/lower-recall criterion than either
    strategy alone: an example must look suspicious by both measures to be
    flagged here, at the cost of missing examples either strategy alone
    would have caught but the other wouldn't.

    This is intentionally just a two-line composition of the existing
    :func:`prune_by_noise_rate` and :func:`prune_by_class` functions -- no
    new confident-joint or thresholding logic is introduced.

    Args:
        joints: per-group confident joints from
            :func:`build_group_confident_joints`.
        n_samples: total number of examples in the original dataset.
        frac: fraction of ``n_samples`` requested from *each* underlying
            strategy (see :func:`prune_by_noise_rate`/:func:`prune_by_class`)
            before intersecting; the intersection itself may end up smaller
            than ``frac * n_samples``, since it only keeps examples both
            strategies independently selected.
        n: exact number of examples requested from *each* underlying
            strategy; takes precedence over ``frac`` when given, same as in
            :func:`prune_by_noise_rate`/:func:`prune_by_class`.

    Returns:
        A boolean mask of shape ``(n_samples,)``; ``True`` marks an example
        flagged by both underlying strategies.
    """
    noise_rate_mask = prune_by_noise_rate(joints, n_samples, frac=frac, n=n)
    class_mask = prune_by_class(joints, n_samples, frac=frac, n=n)
    return noise_rate_mask & class_mask


# ---------------------------------------------------------------------------
# Step 4: one-call convenience entrypoint
# ---------------------------------------------------------------------------


def detect_label_errors(
    labels: Any,
    probs: Any,
    groups: Any,
    *,
    strategy: str = "noise_rate",
    frac: float = 0.05,
    n: int | None = None,
) -> np.ndarray:
    """DeCoLe end-to-end: compute per-group thresholds and confident joints, then prune.

    Composes :func:`compute_group_class_thresholds` ->
    :func:`build_group_confident_joints` -> (:func:`prune_by_noise_rate`,
    :func:`prune_by_class`, or :func:`prune_by_both`, selected by
    ``strategy``) into one call, matching this package's "one clear
    entrypoint" pattern (see
    ``dscraft.clean.detect_near_duplicate_text`` in ``dedup.py``/
    ``embeddings.py`` for the analogous convenience wrapper).

    Args:
        labels: 1D array-like of observed (possibly noisy) labels; see
            :func:`_validate_and_encode` for accepted forms.
        probs: ``(n_samples, n_classes)`` out-of-sample predicted
            probabilities; each row must be finite and sum to ~1.0.
        groups: 1D array-like of group/demographic attribute values, same
            length as ``labels``/``probs``.
        strategy: ``"noise_rate"`` (default, see :func:`prune_by_noise_rate`),
            ``"class"`` (see :func:`prune_by_class`), or ``"both"`` (see
            :func:`prune_by_both` -- the intersection of the other two,
            mirroring cleanlab's ``find_label_issues(filter_by="both")``).
            Any other value raises ``ValueError``.
        frac: fraction of examples to flag when ``n`` is not given.
        n: exact number of examples to flag; takes precedence over ``frac``.

    Returns:
        A boolean mask of shape ``(n_samples,)``; ``True`` marks a flagged
        (likely-mislabeled) example.
    """
    if strategy not in ("noise_rate", "class", "both"):
        raise ValueError(f"strategy must be 'noise_rate', 'class', or 'both', got {strategy!r}.")

    thresholds = compute_group_class_thresholds(labels, probs, groups)
    joints = build_group_confident_joints(labels, probs, groups, thresholds=thresholds)
    n_samples = np.asarray(probs, dtype=float).shape[0]

    if strategy == "noise_rate":
        return prune_by_noise_rate(joints, n_samples, frac=frac, n=n)
    if strategy == "class":
        return prune_by_class(joints, n_samples, frac=frac, n=n)
    return prune_by_both(joints, n_samples, frac=frac, n=n)
