"""Probabilistic sketches for scalable single-pass column profiling.

This module wraps two data structures from the Apache Software Foundation's
`datasketches <https://datasketches.apache.org/>`_ library (Apache-2.0
licensed Python bindings, package name ``datasketches`` on PyPI) rather than
hand-rolling either from scratch. HyperLogLog (HLL) and KLL quantile sketches
are genuinely subtle to implement correctly -- their value comes entirely
from carefully-derived error-bound guarantees (register-count-vs-accuracy
tradeoffs, compaction-schedule proofs), and DataSketches is a well-tested,
permissively-licensed reference implementation of both. There is no
LazyClean/LazyForecast-style "clean-room reimplementation" rationale here:
unlike PyCaret (source-available, non-compete license, per CLAUDE.md's
licensing policy), DataSketches is Apache-2.0 and safe to depend on directly.

**Real installed API used here** (``pip install datasketches``, verified
against the installed package via ``help(datasketches.hll_sketch)`` /
``help(datasketches.kll_floats_sketch)`` -- the docstring text quoted below
is the library's own):

- :class:`datasketches.hll_sketch` -- constructor is
  ``hll_sketch(lg_k: int, tgt_type: tgt_hll_type = tgt_hll_type.HLL_8, ...)``
  (the parameter is named ``lg_k``, not ``log2_k`` -- this module's public
  ``log2_k`` parameter name matches the task's requested interface and is
  passed through positionally). ``lg_k`` must be in ``[7, 21]``; the sketch
  raises its own ``ValueError`` outside that range, which this module
  re-raises with a clearer message before ever constructing the sketch.
  Values are fed in via ``.update(datum)``, overloaded for ``int``/``float``/
  ``str`` only (not arbitrary hashables) -- see :func:`estimate_cardinality`
  for how this module bridges that gap. The estimate is read back via
  ``.get_estimate() -> float``.
- :class:`datasketches.kll_floats_sketch` -- constructor is
  ``kll_floats_sketch(k: int = 200)``; ``k`` must be in ``[8, 65535]`` (the
  sketch raises its own ``ValueError`` outside that range, likewise
  re-raised here first with a clearer message). Values are fed in via
  ``.update(item: float)``. Quantiles are read back via
  ``.get_quantiles(ranks: Sequence[float]) -> list[float]``, and the
  sketch's own documented rank-error bound via
  ``.normalized_rank_error(as_pmf: bool) -> float`` (the single-sided,
  non-PMF bound -- ``as_pmf=False`` -- is what applies to individual
  ``get_quantile``/``get_quantiles`` calls, which is what this module uses;
  see :class:`KLLResult`).

**Hard dependency, not a further-nested optional extra.** Unlike
``dscraft.automl.compile``'s ``onnx``/``skl2onnx`` stack (imported lazily
inside :func:`dscraft.automl.compile.compile`, guarded by
``ONNXExtraNotInstalledError``, so that plain ``dscraft[automl]`` never
force-installs the ONNX toolchain for callers who only need
``fit``/``predict``), ``datasketches`` is not an optional capability bolted
onto a broader subpackage that has other things to do without it -- sketch-
based cardinality/quantile estimation *is* the scalable-single-pass-
profiling capability the ``eda`` subpackage exists to provide (see the
architecture doc's EDA scope). There is no meaningful "import
``dscraft.eda`` without ``datasketches``" use case the way there is a
meaningful "``fit``/``predict`` a pipeline without ever calling
``.compile()``" use case in AutoML. This module therefore imports
``datasketches`` unconditionally at module level and lets a missing install
fail with Python's normal ``ModuleNotFoundError`` at
``import dscraft.eda.sketches`` time -- no custom wrapper exception, no
lazy/deferred import. The wiring implication (left to the ``eda`` extra's
definition in ``pyproject.toml``, not touched by this file) is that
``datasketches`` belongs in the base ``eda`` extra's dependency list
alongside ``numpy``/``polars``, not a separate ``eda-sketches`` sub-extra.

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

from datasketches import hll_sketch, kll_floats_sketch

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

    sketch = hll_sketch(log2_k)
    num_values_processed = 0
    for item in values:
        if isinstance(item, (int, float, str)):
            sketch.update(item)
        else:
            # hll_sketch.update() is only overloaded for int/float/str --
            # see the docstring above for the narrow str()-collision caveat
            # this introduces for non-primitive item types.
            sketch.update(str(item))
        num_values_processed += 1

    if num_values_processed == 0:
        raise ValueError(
            "estimate_cardinality requires a non-empty iterable of values; got zero items. "
            "There is no sensible cardinality estimate over no data -- see the module "
            "docstring for why this raises instead of returning a sentinel like 0."
        )

    relative_error = _HLL_RSE_COEFFICIENT / math.sqrt(2**log2_k)
    return HLLResult(
        estimate=sketch.get_estimate(),
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

    sketch = kll_floats_sketch(k=k)
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

    estimates = sketch.get_quantiles(quantiles)
    quantile_estimates = {q: float(v) for q, v in zip(quantiles, estimates)}
    return KLLResult(
        quantile_estimates=quantile_estimates,
        normalized_rank_error=sketch.normalized_rank_error(False),
        k=k,
        num_values_processed=num_values_processed,
    )
