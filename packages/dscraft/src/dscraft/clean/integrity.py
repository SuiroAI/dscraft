"""Aggregate "Dataset Integrity Score" (architecture doc Part 3, "Module 2: LazyClean").

Combines three independently-computable data-quality signals into a single
weighted scalar:

1. a label-error rate (see :func:`label_error_rate_component`),
2. a train/test contamination rate (see :func:`contamination_rate_component`),
3. a representational-drift measure between a demographic/group attribute's
   distribution in a training split vs. a test split, computed as the
   Jensen-Shannon divergence between the two splits' categorical group
   distributions (see :func:`jensen_shannon_divergence` and
   :func:`demographic_drift_component`).

**Decoupling from sibling modules.** LazyClean's ``label_errors.py`` (a
DeCoLe-style label-error detector) and ``contamination.py`` (a train/test
contamination auditor) are separate, independently-developed modules in this
package. Rather than importing their concrete return types here (which would
create a premature, circular-risk dependency between three sibling files
developed concurrently), this module accepts *plain* inputs -- a boolean
mask, a sequence of per-item states, or a bare float rate -- and reduces
those down to a rate itself. A caller wires ``detect_label_errors(...)``'s
mask or ``audit_contamination(...)``'s per-item states into
:func:`dataset_integrity_score` directly; this module never imports either
sibling.

**Score convention: higher `I_data` is better.** Each of the three
components below is a "badness" rate/divergence in ``[0.0, 1.0]`` (0 =
perfect, 1 = worst possible) -- higher label-error rate, higher
contamination rate, and higher normalized JS divergence all mean *worse*
data quality. The final ``I_data`` inverts the weighted sum of those
badness components (``I_data = 1 - weighted_sum_of_bad_things``), so a
perfectly clean, uncontaminated, zero-drift dataset scores ``I_data = 1.0``
(best) and a maximally bad one scores ``I_data = 0.0`` (worst). This matches
the intuitive reading of a scalar "integrity score" name: higher = more
integrity = better.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Union

import numpy as np

__all__ = [
    "IntegrityReport",
    "jensen_shannon_divergence",
    "label_error_rate_component",
    "contamination_rate_component",
    "demographic_drift_component",
    "dataset_integrity_score",
]

#: A three-state per-test-item contamination classification, matching the
#: sibling `contamination.py` module's documented "clean /
#: validated_contaminated / candidate_unvalidated" result shape. Only the
#: exact string value matters here (see
#: :func:`contamination_rate_component`) -- this module never imports an
#: enum type from `contamination.py`. This MUST exactly match
#: `contamination.py`'s `ContaminationStatus.VALIDATED_CONTAMINATED.value`
#: (underscore, not hyphen) -- see regression test
#: `test_contamination_rate_component_matches_real_contamination_status_value`
#: in `tests/clean/test_integrity.py`.
VALIDATED_CONTAMINATED_STATE = "validated_contaminated"

#: Default equal weighting across the three components -- see
#: :func:`dataset_integrity_score`.
_DEFAULT_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

#: Absolute tolerance for validating that supplied weights sum to 1.0.
_WEIGHT_SUM_TOLERANCE = 1e-6


class IntegrityReport(NamedTuple):
    """Result of :func:`dataset_integrity_score`: the final score plus its per-component breakdown.

    ``score`` is the final ``I_data`` value in ``[0.0, 1.0]`` (higher is
    better -- see module docstring). The three ``*_rate``/``*_divergence``
    fields are each component's raw "badness" value (0 = perfect, 1 =
    worst) *before* inversion and weighting, so a caller can see exactly
    which signal is driving a low score rather than only the final number.
    """

    score: float
    label_error_rate: float
    contamination_rate: float
    drift: float
    weights: tuple[float, float, float]


def _clip_unit_interval(value: float, *, name: str) -> float:
    """Validate ``value`` is a finite real number and clip tiny float noise into ``[0.0, 1.0]``.

    Raises ``ValueError`` for non-finite input or values meaningfully
    outside ``[0.0, 1.0]`` (beyond a small floating-point tolerance) --
    this module treats every component as a rate/probability, and a caller
    passing e.g. a raw count instead of a rate is a usage error, not
    something to silently clamp away.
    """
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    if -1e-9 <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + 1e-9:
        return 1.0
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}.")
    return float(value)


def label_error_rate_component(mask_or_rate: Union[float, int, np.ndarray, Sequence[bool]]) -> float:
    """Reduce a label-error signal down to a scalar rate in ``[0.0, 1.0]``.

    Accepts EITHER:

    - a plain scalar rate (``float``/``int``) already in ``[0.0, 1.0]``, or
    - a boolean mask (e.g. the sibling `label_errors.py` module's
      ``detect_label_errors(...)`` return value -- a per-example boolean
      array flagging suspected label errors), in which case the rate is
      computed as ``flagged_count / total_count``.

    This function never imports `label_errors.py`; wiring its actual mask
    output into this function is done by the caller, decoupling this module
    from that sibling's concrete API.

    Raises:
        ValueError: if given an empty mask (rate is undefined for zero
            items), a mask that isn't boolean-like, or a scalar rate outside
            ``[0.0, 1.0]``.
    """
    if isinstance(mask_or_rate, (float, int)) and not isinstance(mask_or_rate, bool):
        return _clip_unit_interval(float(mask_or_rate), name="label_error rate")

    mask = np.asarray(mask_or_rate)
    if mask.ndim != 1:
        raise ValueError(f"Expected a 1D boolean mask, got shape {mask.shape!r}.")
    if mask.size == 0:
        raise ValueError("label_error mask must not be empty -- rate is undefined for zero items.")
    if mask.dtype != np.bool_:
        try:
            mask = mask.astype(np.bool_)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"label_error mask must be boolean-like, got dtype {mask.dtype!r}."
            ) from exc
    return float(mask.sum()) / float(mask.size)


def contamination_rate_component(
    rate_or_states: Union[float, int, Sequence[object], np.ndarray],
) -> float:
    """Reduce a train/test contamination signal down to a scalar rate in ``[0.0, 1.0]``.

    Accepts EITHER:

    - a plain scalar rate (``float``/``int``) already in ``[0.0, 1.0]``, or
    - a sequence of per-test-item states/labels (e.g. the sibling
      `contamination.py` module's three-state "clean /
      validated-contaminated / candidate-unvalidated" per-item result),
      in which case the rate is computed as the fraction of items whose
      state equals :data:`VALIDATED_CONTAMINATED_STATE` -- i.e. only
      *validated*-contaminated items count as contaminated for this score;
      "candidate-unvalidated" items are treated as not-yet-confirmed and do
      not count, matching the conservative "don't accuse without
      validation" semantics of a three-state result.

    A boolean-like sequence/array is also accepted as a plain "is this item
    contaminated" mask (``True``/``1`` counts, ``False``/``0`` does not) --
    useful when a caller has already reduced the three-state result down to
    a boolean themselves. This is deliberately robust to every reasonable
    boolean-like shape a caller might hand in, not just a genuine
    ``numpy.bool_``-dtype array:

    - a ``numpy.ndarray`` of dtype ``bool``,
    - a plain ``list``/``tuple`` of Python ``bool`` values (NumPy infers a
      native ``bool`` dtype for these automatically -- this function never
      forces an ``object`` dtype, which would otherwise defeat that
      inference and silently fall through to the state-string comparison
      path below, comparing each Python ``bool`` against a string and
      always getting ``False``),
    - a ``numpy.ndarray`` of an integer dtype containing only ``0``/``1``
      values, or a plain ``list``/``tuple`` of ``0``/``1`` ints.

    Any other integer values (e.g. ``2``, ``-1``) are rejected with a clear
    ``ValueError`` rather than silently misinterpreted as booleans or as
    (numeric, never-matching) state labels.

    This function never imports `contamination.py`; wiring its actual
    per-item result into this function is done by the caller.

    Raises:
        ValueError: if given an empty sequence (rate is undefined for zero
            items), a scalar rate outside ``[0.0, 1.0]``, or an integer
            sequence containing a value other than ``0``/``1``.
    """
    if isinstance(rate_or_states, (float, int)) and not isinstance(rate_or_states, bool):
        return _clip_unit_interval(float(rate_or_states), name="contamination rate")

    if isinstance(rate_or_states, np.ndarray):
        states = rate_or_states
    else:
        # Let NumPy infer the natural dtype for a plain sequence -- do NOT
        # force `dtype=object` here. Forcing `object` is what silently
        # broke this function for a plain `list[bool]` (and a plain
        # `list` of 0/1 ints): NumPy would otherwise give `list[bool]` a
        # native `bool` dtype and a 0/1 `list[int]` a native integer
        # dtype, both handled explicitly below; forcing `object` instead
        # produced an array of bare Python `bool`/`int` objects that would
        # then be compared against a string (`VALIDATED_CONTAMINATED_STATE`)
        # and *always* evaluate to `False`, silently returning a
        # wrong-but-plausible-looking zero contamination rate instead of
        # either computing correctly or raising.
        states = np.asarray(list(rate_or_states))

    if states.ndim != 1:
        raise ValueError(f"Expected a 1D sequence of per-item states, got shape {states.shape!r}.")
    if states.size == 0:
        raise ValueError(
            "contamination states must not be empty -- rate is undefined for zero items."
        )

    flagged = _coerce_boolean_like_or_state_mask(states)
    return float(np.count_nonzero(flagged)) / float(states.size)


def _reject_non_boolean_integers(values: Sequence[object]) -> np.ndarray:
    """Validate that every value in ``values`` is exactly ``0``/``1`` (or ``True``/``False``) and return a boolean array.

    Raises:
        ValueError: if any value is an integer other than ``0``/``1``.
    """
    unique_values = sorted({int(value) for value in values})
    if not set(unique_values) <= {0, 1}:
        raise ValueError(
            "Integer contamination input must contain only 0/1 values to be "
            f"treated as a boolean mask, got values {unique_values!r}."
        )
    return np.array([bool(value) for value in values], dtype=np.bool_)


def _coerce_boolean_like_or_state_mask(states: np.ndarray) -> np.ndarray:
    """Reduce a validated, non-empty 1D ``states`` array to a boolean "is contaminated" mask.

    Handles every dtype :func:`contamination_rate_component` can receive
    after its own ``np.asarray`` normalization: a native boolean dtype, a
    native integer dtype (accepted only if every value is ``0``/``1``), an
    ``object`` dtype (e.g. a mixed-type or all-Python-bool list NumPy could
    not give a uniform native dtype to), or anything else (e.g. NumPy's
    native fixed-width unicode dtype for a list of Python strings) -- the
    last two fall back to comparing each element against
    :data:`VALIDATED_CONTAMINATED_STATE`.
    """
    if states.dtype == np.bool_:
        return states
    if np.issubdtype(states.dtype, np.integer):
        return _reject_non_boolean_integers(states.tolist())
    if states.dtype == object:
        elements = states.tolist()
        if all(isinstance(value, (bool, np.bool_)) for value in elements):
            return np.array(elements, dtype=np.bool_)
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in elements):
            return _reject_non_boolean_integers(elements)
        return states == VALIDATED_CONTAMINATED_STATE
    return states == VALIDATED_CONTAMINATED_STATE


def _categorical_distributions(
    train_groups: Sequence[object], test_groups: Sequence[object]
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned categorical frequency distributions over the union of both splits' categories.

    A category present in one split but entirely absent from the other is
    given probability 0 in the missing split, per the module contract --
    both output arrays are aligned to the same category ordering (the
    sorted union of categories seen in either split) so they can be passed
    directly to :func:`jensen_shannon_divergence`.

    Raises:
        ValueError: if either split is empty, or if the union of
            categories across both splits is empty.
    """
    train_arr = np.asarray(list(train_groups), dtype=object)
    test_arr = np.asarray(list(test_groups), dtype=object)
    if train_arr.size == 0:
        raise ValueError("train_groups must not be empty.")
    if test_arr.size == 0:
        raise ValueError("test_groups must not be empty.")

    categories = sorted(set(train_arr.tolist()) | set(test_arr.tolist()), key=repr)
    if not categories:
        raise ValueError("No categories found across train_groups and test_groups.")

    train_counts = np.array([np.count_nonzero(train_arr == category) for category in categories], dtype=np.float64)
    test_counts = np.array([np.count_nonzero(test_arr == category) for category in categories], dtype=np.float64)

    p = train_counts / train_counts.sum()
    q = test_counts / test_counts.sum()
    return p, q


def jensen_shannon_divergence(p: Union[Sequence[float], np.ndarray], q: Union[Sequence[float], np.ndarray]) -> float:
    """Compute the Jensen-Shannon divergence between two categorical distributions ``p`` and ``q``.

    Implemented from scratch with NumPy (no SciPy): ``JS(P, Q) = 0.5 *
    KL(P || M) + 0.5 * KL(Q || M)``, where ``M = 0.5 * (P + Q)`` and
    ``KL(A || B) = sum(a * log(a / b))`` using the natural logarithm.

    **Convention: natural log, range ``[0, ln(2)]``.** With natural-log KL
    divergence, JS divergence is bounded in ``[0, ln(2)]`` (≈0.6931), not
    ``[0, 1]`` -- ``ln(2)`` is attained exactly when ``p`` and ``q`` have
    disjoint support (share no category with nonzero probability in both).
    This is the standard natural-log convention (as opposed to the
    log-base-2 convention some references use, which bounds JS in
    ``[0, 1]``); this module always uses natural log and documents/tests
    against ``ln(2)`` as the known maximum, never rescaling internally.
    Callers that want a maximum of 1.0 elsewhere should note that
    :func:`demographic_drift_component` normalizes by ``ln(2)`` explicitly
    for that reason -- see its docstring.

    ``p`` and ``q`` are validated as proper probability distributions:
    non-negative, finite, and summing to 1 (within a small tolerance) --
    they are NOT renormalized here. Callers with raw, un-normalized
    frequency counts should use :func:`demographic_drift_component`
    instead, which builds normalized distributions from raw group-label
    arrays before calling this function.

    ``0 * log(0 / q)`` terms (a zero-probability bin in ``p`` or ``q``) are
    handled via masking so no ``log(0)`` or ``0/0`` ever occurs -- by
    convention ``0 * log(0 / anything) = 0`` (the standard, measure-
    theoretically correct convention for KL divergence), and since
    ``M = 0.5 * (P + Q)`` is strictly positive everywhere ``P`` or ``Q`` is
    positive, a ``log(p / m)`` term is only ever evaluated where ``p > 0``
    (and likewise for ``q``/``m``), so no division by zero occurs either.

    Raises:
        ValueError: if ``p``/``q`` have mismatched shapes, contain
            negative or non-finite values, or do not each sum to 1 (within
            tolerance).
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError(f"p and q must have the same shape, got {p_arr.shape!r} and {q_arr.shape!r}.")
    if p_arr.ndim != 1:
        raise ValueError(f"p and q must be 1D, got shape {p_arr.shape!r}.")
    for name, arr in (("p", p_arr), ("q", q_arr)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values.")
        if (arr < 0.0).any():
            raise ValueError(f"{name} must be non-negative (a probability distribution).")
        total = arr.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"{name} must sum to 1.0 (a probability distribution), got sum {total!r}.")

    m = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        # 0 * log(0 / b) = 0 by convention; mask out a == 0 terms entirely
        # so log(0) is never evaluated. b (== m here) is guaranteed > 0
        # wherever a > 0, since m = 0.5 * (p + q) >= 0.5 * a.
        safe_ratio = np.where(a > 0.0, a / np.where(b > 0.0, b, 1.0), 1.0)
        terms = np.where(a > 0.0, a * np.log(safe_ratio), 0.0)
        return float(terms.sum())

    return 0.5 * _kl(p_arr, m) + 0.5 * _kl(q_arr, m)


def demographic_drift_component(
    train_groups: Sequence[object], test_groups: Sequence[object]
) -> float:
    """Compute normalized representational drift between a group attribute's train/test distributions.

    Internally converts ``train_groups`` and ``test_groups`` (raw,
    per-example group-attribute values -- e.g. a demographic category per
    row, NOT pre-computed distributions) into aligned categorical frequency
    distributions over the union of categories seen in either split (a
    category present in one split but entirely absent from the other gets
    probability 0 there), then returns their
    :func:`jensen_shannon_divergence`, normalized by ``ln(2)`` so the
    result is in ``[0.0, 1.0]`` (0 = identical group distributions, 1 =
    disjoint support / maximally different) -- matching the ``[0.0, 1.0]``
    "badness" convention the other two components use (see module
    docstring for why :func:`jensen_shannon_divergence` itself is *not*
    normalized this way).

    Args:
        train_groups: per-example group/demographic labels for the
            training split (e.g. a list of category strings). Must be
            non-empty.
        test_groups: per-example group/demographic labels for the test
            split. Must be non-empty.

    Raises:
        ValueError: if either split is empty.
    """
    p, q = _categorical_distributions(train_groups, test_groups)
    divergence = jensen_shannon_divergence(p, q)
    return divergence / np.log(2.0)


def _validate_weights(weights: tuple[float, float, float]) -> None:
    if len(weights) != 3:
        raise ValueError(f"weights must have exactly 3 elements (w1, w2, w3), got {len(weights)}.")
    for weight in weights:
        if not np.isfinite(weight):
            raise ValueError(f"weights must be finite numbers, got {weights!r}.")
    total = sum(weights)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"weights must sum to 1.0 (within tolerance), got {weights!r} summing to {total!r}.")


def dataset_integrity_score(
    label_error_input: Union[float, int, np.ndarray, Sequence[bool]],
    contamination_input: Union[float, int, Sequence[object], np.ndarray],
    train_groups: Sequence[object],
    test_groups: Sequence[object],
    weights: tuple[float, float, float] | None = None,
) -> IntegrityReport:
    """Compute the aggregate Dataset Integrity Score ``I_data`` from three component signals.

    ``I_data = 1 - (w1 * label_error_rate + w2 * contamination_rate + w3 * drift)``,
    where each component is a "badness" rate/divergence in ``[0.0, 1.0]``
    (see :func:`label_error_rate_component`,
    :func:`contamination_rate_component`, and
    :func:`demographic_drift_component`) and ``w1 + w2 + w3 == 1``. Higher
    ``I_data`` is better -- see module docstring for the convention.

    Args:
        label_error_input: scalar rate or boolean mask -- passed straight
            to :func:`label_error_rate_component`.
        contamination_input: scalar rate or per-item state sequence --
            passed straight to :func:`contamination_rate_component`.
        train_groups: per-example group/demographic labels for the
            training split -- passed to :func:`demographic_drift_component`.
        test_groups: per-example group/demographic labels for the test
            split -- passed to :func:`demographic_drift_component`.
        weights: ``(w1, w2, w3)`` weighting the label-error, contamination,
            and drift components respectively. Defaults to equal weighting
            (``1/3`` each) when omitted. Must sum to 1.0 within a small
            tolerance, or ``ValueError`` is raised.

    Returns:
        An :class:`IntegrityReport` with the final ``score`` plus each raw
        component value and the resolved weights, so a caller can see the
        full breakdown, not just the final number.

    Raises:
        ValueError: if ``weights`` does not have exactly 3 elements or does
            not sum to 1.0 (within tolerance), or if any component function
            raises (empty input, malformed mask, etc. -- see their
            docstrings).
    """
    resolved_weights = tuple(weights) if weights is not None else _DEFAULT_WEIGHTS
    _validate_weights(resolved_weights)
    w1, w2, w3 = resolved_weights

    label_error_rate = label_error_rate_component(label_error_input)
    contamination_rate = contamination_rate_component(contamination_input)
    drift = demographic_drift_component(train_groups, test_groups)

    weighted_bad = w1 * label_error_rate + w2 * contamination_rate + w3 * drift
    score = 1.0 - weighted_bad

    return IntegrityReport(
        score=score,
        label_error_rate=label_error_rate,
        contamination_rate=contamination_rate,
        drift=drift,
        weights=resolved_weights,
    )
