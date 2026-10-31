"""Probabilistic sketches for scalable single-pass column profiling.

This module provides two probabilistic sketches: HyperLogLog (HLL) for distinct-count estimation
and KLL for quantile estimation. As part of the first-party automl/eda rebuild (see
``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``), **both sketches are now first-party
implementations** built from scratch on ``numpy`` (plus ``hashlib`` for HLL):
:class:`dscraft.eda._hll.HyperLogLog` behind :func:`estimate_cardinality`, and
:class:`dscraft.eda._kll.KLL` behind :func:`estimate_quantiles`. Neither wraps the Apache
``datasketches`` library anymore (that dependency is removed by the same rebuild). The
``HLLResult``/``KLLResult`` public result types and documented error-bound semantics are unchanged
by the swap, so callers (notably ``LazyEDA``) are unaffected.

**First-party implementations (no third-party sketch library).** The HLL and KLL algorithms live
in :mod:`dscraft.eda._hll` and :mod:`dscraft.eda._kll` respectively, built from scratch on
``numpy`` (plus ``hashlib`` for HLL's stable hashing). This module keeps a thin, validated public
surface over them:

- :func:`estimate_cardinality` -> :class:`dscraft.eda._hll.HyperLogLog`. The caller-facing
  ``log2_k`` parameter must be in ``[7, 21]`` (validated here with a clear message before the
  sketch is constructed). ``int``/``float``/``str`` values are hashed directly; any other item is
  ``str()``-coerced (see :func:`estimate_cardinality` for the narrow collision caveat). The RSE is
  the family-wide ``1.04 / sqrt(2**log2_k)``.
- :func:`estimate_quantiles` -> :class:`dscraft.eda._kll.KLL`. The caller-facing ``k`` parameter
  must be in ``[8, 65535]`` (validated here). Values must be finite, numeric, non-``bool``. The
  reported ``normalized_rank_error`` is KLL's published single-sided bound (``~2.296 / k``). The
  sketch uses a fixed internal seed so repeated calls on the same data are deterministic.

Both sketch *algorithms* were previously wrapped from the Apache ``datasketches`` library; the
first-party automl/eda rebuild (see ``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``)
replaced them, and ``datasketches`` is no longer a dependency of the ``eda`` extra. The public
``HLLResult``/``KLLResult`` result types and their documented error-bound semantics are unchanged.

**Empty-input behavior: raise ``ValueError``, consistently for both
sketches.** A cardinality or quantile estimate over zero values is not a
degenerate-but-meaningful answer (unlike, say, an empty embedding batch
correctly returning a ``(0, dim)`` array in
``dscraft.clean.embeddings.EmbeddingModel.embed``) -- there is no sensible
"estimate" of a distinct count or a quantile position over no data, and
returning a sentinel like ``0`` or ``float("nan")`` here would be silently
indistinguishable from "a real sketch that happens to estimate exactly
zero/NaN". Raising makes the "no data was profiled" case impossible to
mistake for a real (if imprecise) estimate, matching this module's
error-loudly stance on non-numeric/out-of-range input below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from dscraft.eda._hll import HyperLogLog
from dscraft.eda._kll import KLL

__all__ = [
    "HLLResult",
    "KLLResult",
    "estimate_cardinality",
    "estimate_quantiles",
]

#: Valid range for HLL's ``lg_k`` (this module's ``log2_k``) parameter, per
#: the installed ``datasketches.hll_sketch`` constructor's own validation
#: (confirmed by probing ``hll_sketch(3)`` / ``hll_sketch(22)``, which raise
#: ``ValueError: Invalid value of k: <n>``).
_HLL_LOG2_K_MIN = 7
_HLL_LOG2_K_MAX = 21

#: Valid range for KLL's ``k`` parameter, per the installed
#: ``datasketches.kll_floats_sketch`` constructor's own validation
#: (``ValueError: K must be >= 8 and <= 65535: <n>``).
_KLL_K_MIN = 8
_KLL_K_MAX = 65535

#: HyperLogLog's standard relative-standard-error formula, as documented by
#: the DataSketches project for the HLL family: RSE ~= 1.04 / sqrt(2^lg_k).
#: This is the sketch's own family-wide theoretical error bound for a given
#: precision parameter, not an empirical measurement of any particular run
#: -- unlike :meth:`kll_floats_sketch.normalized_rank_error`, the installed
#: ``hll_sketch`` does not expose an equivalent "give me my configured RSE"
#: accessor directly (only ``get_lower_bound``/``get_upper_bound``, which
#: report a bound around *this sketch's current estimate*, not the
#: context-free precision constant), so this module computes it directly
#: from ``log2_k`` per the documented formula.
_HLL_RSE_COEFFICIENT = 1.04


@dataclass(frozen=True)
class HLLResult:
    """Result of a HyperLogLog cardinality estimate.

    Attributes:
        estimate: The sketch's estimated distinct-value count.
        relative_error: The HLL family's theoretical relative standard
            error (RSE) for the given ``log2_k``, i.e.
            ``1.04 / sqrt(2**log2_k)``. This is a property of the precision
            parameter, not of the particular data profiled -- it describes
            the expected spread of estimates across many independent runs
            at this ``log2_k``, not a per-call confidence interval computed
            from this one sketch's internal state.
        log2_k: The precision parameter used (``lg_k`` in the underlying
            ``datasketches.hll_sketch`` constructor). Higher values trade
            more registers (memory) for a tighter ``relative_error``.
        num_values_processed: Count of values fed into the sketch (not the
            distinct count -- this is the raw stream length).
    """

    estimate: float
    relative_error: float
    log2_k: int
    num_values_processed: int


@dataclass(frozen=True)
class KLLResult:
    """Result of a KLL quantile estimate.

    Attributes:
        quantile_estimates: Maps each requested quantile (e.g. ``0.5`` for
            the median) to its estimated value. If ``quantiles`` contained
            duplicate entries, the mapping necessarily collapses them to one
            key -- see :func:`estimate_quantiles`.
        normalized_rank_error: The sketch's own documented single-sided
            rank-error bound for the configured ``k``
            (``kll_floats_sketch.normalized_rank_error(as_pmf=False)``),
            i.e. the guarantee that applies to individual quantile/rank
            queries (as opposed to the "double-sided" PMF bound, which
            would apply to :meth:`kll_floats_sketch.get_pmf` calls this
            module does not make).
        k: The precision parameter used. Higher values trade a larger
            sketch (more retained samples across levels) for a tighter
            ``normalized_rank_error``.
        num_values_processed: Count of values fed into the sketch.
    """

    quantile_estimates: dict[float, float]
    normalized_rank_error: float
    k: int
    num_values_processed: int


def estimate_cardinality(values: Iterable, log2_k: int = 12) -> HLLResult:
    """Estimate the number of distinct values in ``values`` via HyperLogLog.

    Intended for high-cardinality nominal/categorical columns (UUIDs,
    surrogate IDs, free-text keys) where an exact distinct count would
    require an O(n)-memory hash set of every unique value seen. HLL instead
    holds a fixed-size register array of ``2**log2_k`` registers regardless
    of ``n`` or the true cardinality, trading a small, quantifiable amount
    of estimation error (:attr:`HLLResult.relative_error`) for O(1)-class
    space.

    ``values`` may contain any hashable items, not just ``int``/``float``/
    ``str``: the underlying ``datasketches.hll_sketch.update`` method is
    only overloaded for those three types, so any other item (e.g. a
    ``tuple`` or a custom object) is converted via ``str(item)`` before
    being fed to the sketch. This means two distinct non-string/int/float
    values that happen to share a ``str()`` representation (unusual, but
    possible for a poorly-behaved ``__str__``) would be undercounted as one
    -- a documented, narrow edge case, not a silent general-purpose hashing
    scheme. Plain ``int``/``float``/``str`` items (the expected case for
    real ID/categorical columns) are passed through to ``update`` directly
    and are not subject to this caveat.

    Args:
        values: Any iterable of hashable values. Must be non-empty -- see
            the module docstring for why an empty column raises rather than
            returning a sentinel estimate.
        log2_k: HLL's precision parameter (``lg_k`` in the underlying
            library), controlling the ``2**log2_k``-register array size.
            Must be in ``[7, 21]`` inclusive, per the underlying
            ``datasketches.hll_sketch`` constructor's own validation.
            Higher values give a tighter :attr:`HLLResult.relative_error`
            at the cost of more memory (still O(1) in the input size, just
            a larger constant).

    Returns:
        An :class:`HLLResult` with the estimate and its theoretical
        relative error for the given ``log2_k``.

    Raises:
        ValueError: if ``log2_k`` is outside ``[7, 21]``, or if ``values``
            is empty.
    """
    if not (_HLL_LOG2_K_MIN <= log2_k <= _HLL_LOG2_K_MAX):
        raise ValueError(
            f"log2_k must be in [{_HLL_LOG2_K_MIN}, {_HLL_LOG2_K_MAX}], got {log2_k!r}."
        )

    sketch = HyperLogLog(log2_k)
    num_values_processed = 0
    for item in values:
        # The first-party HyperLogLog hashes int/float/str directly and str()-coerces any other
        # item type -- the same narrow str()-collision caveat documented above for non-primitive
        # items applies, now in `dscraft.eda._hll._hash64` rather than the wrapped library.
        sketch.update(item)
        num_values_processed += 1

    if num_values_processed == 0:
        raise ValueError(
            "estimate_cardinality requires a non-empty iterable of values; got zero items. "
            "There is no sensible cardinality estimate over no data -- see the module "
            "docstring for why this raises instead of returning a sentinel like 0."
        )

    relative_error = _HLL_RSE_COEFFICIENT / math.sqrt(2**log2_k)
    return HLLResult(
        estimate=sketch.estimate(),
        relative_error=relative_error,
        log2_k=log2_k,
        num_values_processed=num_values_processed,
    )


def estimate_quantiles(
    values: Iterable[float],
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    k: int = 200,
) -> KLLResult:
    """Estimate quantiles of a numeric column via a KLL sketch.

    Intended for continuous numeric columns where computing exact
    quantiles would require sorting (or otherwise retaining) the full
    dataset. KLL instead maintains a bounded hierarchy of log-sized
    sampling buffers -- when a buffer overflows, its contents are sorted,
    half discarded, and the remainder compacted up a level -- giving a
    quantile estimate within a documented error bound
    (:attr:`KLLResult.normalized_rank_error`) using sub-linear space.

    Args:
        values: Any iterable of numeric (``int`` or ``float``) values. Must
            be non-empty -- see the module docstring for why an empty
            column raises rather than returning a sentinel result. Every
            item must be a finite ``int``/``float`` (not ``bool`` -- see
            below -- and not ``str``, ``NaN``, or infinite); a non-numeric
            item raises ``TypeError``, and a non-finite numeric item raises
            ``ValueError``.
        quantiles: The quantiles to estimate, each in ``[0.0, 1.0]``
            (0.0 = minimum, 0.5 = median, 1.0 = maximum). Must be
            non-empty. Duplicate entries collapse to one key in
            :attr:`KLLResult.quantile_estimates` (a plain ``dict``), since a
            dict cannot hold two values under the same key -- pass distinct
            quantiles if you need every one individually represented.
        k: KLL's precision parameter, controlling the sketch's per-level
            buffer size. Must be in ``[8, 65535]`` inclusive, per the
            underlying ``datasketches.kll_floats_sketch`` constructor's own
            validation. Higher values give a tighter
            :attr:`KLLResult.normalized_rank_error` at the cost of a larger
            (still sub-linear) sketch.

    Returns:
        A :class:`KLLResult` with the per-quantile estimates and the
        sketch's own rank-error bound for the given ``k``.

    Raises:
        TypeError: if any item in ``values`` is not an ``int``/``float``
            (this explicitly includes ``bool``, since silently treating
            ``True``/``False`` as ``1.0``/``0.0`` in a numeric-quantile
            context is far more likely to be an upstream data-typing bug
            than an intentional 0/1 numeric column).
        ValueError: if ``values`` is empty, if ``quantiles`` is empty, if
            any requested quantile is outside ``[0.0, 1.0]``, if ``k`` is
            outside ``[8, 65535]``, or if any item in ``values`` is
            non-finite (``NaN``/``inf``).
    """
    if not (_KLL_K_MIN <= k <= _KLL_K_MAX):
        raise ValueError(f"k must be in [{_KLL_K_MIN}, {_KLL_K_MAX}], got {k!r}.")

    quantiles = list(quantiles)
    if not quantiles:
        raise ValueError("quantiles must be a non-empty sequence.")
    for q in quantiles:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"quantiles values must be in [0.0, 1.0], got {q!r}.")

    sketch = KLL(k=k, random_state=0)
    num_values_processed = 0
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(
                f"estimate_quantiles requires numeric (int/float, non-bool) values, "
                f"got {item!r} of type {type(item).__name__}."
            )
        if not math.isfinite(item):
            raise ValueError(f"estimate_quantiles requires finite values, got {item!r}.")
        sketch.update(float(item))
        num_values_processed += 1

    if num_values_processed == 0:
        raise ValueError(
            "estimate_quantiles requires a non-empty iterable of values; got zero items. "
            "There is no sensible quantile estimate over no data -- see the module "
            "docstring for why this raises instead of returning a sentinel like NaN."
        )

    estimates = sketch.quantiles(quantiles)
    quantile_estimates = {q: float(v) for q, v in zip(quantiles, estimates)}
    return KLLResult(
        quantile_estimates=quantile_estimates,
        normalized_rank_error=sketch.normalized_rank_error(),
        k=k,
        num_values_processed=num_values_processed,
    )
