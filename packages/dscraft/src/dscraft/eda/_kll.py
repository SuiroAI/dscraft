"""First-party KLL quantile sketch for ``dscraft.eda``.

This internal module implements the KLL quantile sketch from scratch on ``numpy`` + the standard
library -- the first-party replacement for ``datasketches.kll_floats_sketch`` that
``dscraft.eda.sketches`` wraps. It is the algorithm behind
:func:`dscraft.eda.sketches.estimate_quantiles`; the public ``KLLResult``/``estimate_quantiles``
surface and its documented rank-error semantics are preserved, so ``LazyEDA`` and other callers are
unaffected by the swap.

KLL (Karnin-Lang-Liberty, 2016) estimates quantiles of a numeric stream in sub-linear space with a
guaranteed, formula-derived rank-error bound. The structure is a hierarchy of *compactors* (sorted
buffers), one per level:

- Items enter level 0. When a level's buffer reaches its capacity it is *compacted*: the buffer is
  sorted, and every other item (odd or even indices chosen by a fair coin for this compaction) is
  promoted to the next level up, the rest discarded. Each level-``h`` item thus represents
  ``2**h`` stream items.
- Buffer capacities shrink geometrically going up the hierarchy (``capacity ~ k * c**(H-h)`` with
  ``c = 2/3``), which is exactly what yields KLL's space/accuracy optimality; a small minimum
  capacity floors the top levels.

A quantile query sorts all retained ``(value, weight=2**level)`` pairs, forms the cumulative
weighted rank, and returns the value whose cumulative rank first reaches the requested fraction of
the total weight.

The reported ``normalized_rank_error`` is the standard single-sided KLL bound for the configured
``k`` (``~ 2.296 / k``, the published constant for the non-PMF, single-quantile guarantee this
module's queries use), matching the semantics the wrapped library's
``normalized_rank_error(as_pmf=False)`` exposed. Randomized compaction uses an explicit
``numpy.random.Generator`` so a seed gives reproducible sketches.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "KLL",
]

#: Capacity shrink ratio between adjacent levels (the KLL ``c``; 2/3 is the standard choice).
_CAPACITY_RATIO = 2.0 / 3.0
#: Minimum per-level buffer capacity (floors the geometric shrink at the top of the hierarchy).
_MIN_CAPACITY = 8
#: Published single-sided normalized-rank-error constant for KLL at parameter ``k`` (error ~ C/k).
_RANK_ERROR_CONSTANT = 2.296


class KLL:
    """A from-scratch KLL quantile sketch over ``float`` values.

    Args:
        k: precision parameter -- the (level-0) buffer capacity. Larger ``k`` gives a tighter
            :meth:`normalized_rank_error` at the cost of a larger (still sub-linear) sketch.
            Caller-facing ``[8, 65535]`` validation lives in
            :func:`dscraft.eda.sketches.estimate_quantiles`; this class accepts any ``k >= 2`` so it
            stays a pure algorithm component.
        random_state: seed / ``numpy.random.Generator`` for the fair-coin compaction choice.

    Feed values with :meth:`update` / :meth:`update_many`, then query :meth:`quantiles`.
    """

    def __init__(self, k: int = 200, random_state=None) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k!r}.")
        self.k = int(k)
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        # One Python list per level; levels grow as the hierarchy deepens.
        self._levels: list[list[float]] = [[]]
        self.n = 0  # total items seen
        # Exact extremes, tracked separately from the compacted hierarchy: KLL compaction can
        # discard the true min/max, but they are cheap to keep exactly and callers (and the
        # datasketches sketch this replaces) expect rank 0.0/1.0 to return them exactly.
        self._min = float("inf")
        self._max = float("-inf")

    def _capacity(self, level: int) -> int:
        """Buffer capacity for ``level`` (geometric shrink from the top of the hierarchy down)."""
        height = len(self._levels) - 1
        cap = int(np.ceil(self.k * (_CAPACITY_RATIO ** (height - level))))
        return max(_MIN_CAPACITY, cap)

    def update(self, value: float) -> None:
        """Add one numeric value to the sketch."""
        v = float(value)
        self._levels[0].append(v)
        self.n += 1
        if v < self._min:
            self._min = v
        if v > self._max:
            self._max = v
        if len(self._levels[0]) >= self._capacity(0):
            self._compact(0)

    def update_many(self, values) -> int:
        """Add every value in ``values``; return the count added."""
        count = 0
        for v in values:
            self.update(v)
            count += 1
        return count

    def _compact(self, level: int) -> None:
        """Compact ``level`` upward: sort, promote every other item, discard the rest."""
        buffer = self._levels[level]
        buffer.sort()
        # Ensure the destination level exists.
        if level + 1 >= len(self._levels):
            self._levels.append([])
        # Fair coin: promote the even- or odd-indexed half of the sorted buffer.
        offset = int(self._rng.integers(0, 2))
        promoted = buffer[offset::2]
        self._levels[level + 1].extend(promoted)
        self._levels[level] = []
        # Compacting may have overfilled the parent; cascade upward.
        if len(self._levels[level + 1]) >= self._capacity(level + 1):
            self._compact(level + 1)

    def _weighted_items(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all retained ``(values, weights)`` where an item at level ``h`` has weight ``2**h``."""
        values: list[float] = []
        weights: list[int] = []
        for level, buffer in enumerate(self._levels):
            if not buffer:
                continue
            w = 1 << level
            values.extend(buffer)
            weights.extend([w] * len(buffer))
        return np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    def normalized_rank_error(self) -> float:
        """Published single-sided normalized rank-error bound for this ``k`` (``~2.296 / k``)."""
        return _RANK_ERROR_CONSTANT / self.k

    def quantiles(self, ranks: Sequence[float]) -> list[float]:
        """Estimate the values at the given normalized ``ranks`` (each in ``[0, 1]``).

        ``rank=0.0`` returns the minimum, ``0.5`` the median, ``1.0`` the maximum. Uses the
        weighted cumulative-rank construction over all retained items.
        """
        if self.n == 0:
            raise ValueError("Cannot query quantiles of an empty KLL sketch.")
        values, weights = self._weighted_items()
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        total = cumulative[-1]

        out: list[float] = []
        for rank in ranks:
            if rank <= 0.0:
                out.append(self._min)  # exact minimum, tracked separately from the hierarchy
                continue
            if rank >= 1.0:
                out.append(self._max)  # exact maximum
                continue
            target = rank * total
            idx = int(np.searchsorted(cumulative, target, side="left"))
            idx = min(idx, sorted_values.shape[0] - 1)
            out.append(float(sorted_values[idx]))
        return out
