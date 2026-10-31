"""First-party HyperLogLog cardinality sketch for ``dscraft.eda``.

This internal module implements HyperLogLog from scratch on ``numpy`` + the Python standard
library's ``hashlib`` -- the first-party replacement for the ``datasketches.hll_sketch`` the
``dscraft.eda.sketches`` module currently wraps. It is the algorithm behind
:func:`dscraft.eda.sketches.estimate_cardinality`; the public ``HLLResult``/``estimate_cardinality``
surface and its documented error-bound semantics are preserved exactly, so ``LazyEDA`` and any
other caller are unaffected by the swap.

HyperLogLog estimates the number of *distinct* values in a stream using a fixed-size register
array, independent of the true cardinality -- the whole point of a sketch. The implementation here
follows Flajolet et al. (2007) with the standard practical corrections:

1. Hash each item to a 64-bit value (SHA-1 of its bytes, truncated -- a stable, well-distributed
   hash that does not vary across processes the way Python's salted ``hash()`` does).
2. Use the top ``p = log2_k`` bits to pick one of ``m = 2**p`` registers; count the number of
   leading zeros (plus one) in the remaining bits and keep the per-register maximum.
3. Estimate the cardinality from the registers via the bias-corrected harmonic mean
   ``E = alpha_m * m^2 / sum(2**-M[j])``, with:
   - the **small-range correction** (linear counting over empty registers) when ``E`` is small and
     some registers are still zero, and
   - the **large-range correction** for ``E`` approaching the 64-bit hash space (negligible in
     practice at these register counts, included for completeness).

The ``relative_error`` reported is the family-wide theoretical relative standard error
``1.04 / sqrt(m)`` -- identical to the formula the wrapped library's result exposed -- a property of
the precision parameter, not of the particular data.

Register state is kept as a ``numpy`` ``uint8`` array (a register never needs to exceed ~64), so an
``m = 2**21`` sketch is ~2MB at most.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "HyperLogLog",
]

#: 64-bit mask -- the hash space HLL operates over.
_HASH_BITS = 64
_HASH_MASK = (1 << _HASH_BITS) - 1


def _hash64(item) -> int:
    """Hash an int/float/str (or its ``str()``) to a stable 64-bit unsigned integer.

    Uses SHA-1 of the UTF-8/byte encoding rather than Python's built-in ``hash()``, which is
    salted per-process for ``str``/``bytes`` (so it would give non-reproducible estimates across
    runs) and is only 64-bit for ints incidentally. SHA-1 is not used here for any security
    property -- only as a fast, stable, well-distributed hash -- so its cryptographic weaknesses
    are irrelevant to this use.
    """
    if isinstance(item, bool):
        data = b"bool:" + (b"1" if item else b"0")
    elif isinstance(item, int):
        data = b"int:" + repr(item).encode("utf-8")
    elif isinstance(item, float):
        data = b"float:" + repr(item).encode("utf-8")
    elif isinstance(item, str):
        data = b"str:" + item.encode("utf-8")
    else:
        data = b"str:" + str(item).encode("utf-8")
    digest = hashlib.sha1(data).digest()
    return int.from_bytes(digest[:8], "big") & _HASH_MASK


def _alpha(m: int) -> float:
    """Bias-correction constant ``alpha_m`` for register count ``m`` (Flajolet et al.)."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLog:
    """A from-scratch HyperLogLog distinct-count sketch.

    Args:
        log2_k: precision parameter ``p``; the sketch uses ``m = 2**log2_k`` registers. Higher
            values give a tighter :attr:`relative_error` at the cost of more memory (still constant
            in the input size). Validation of the caller-facing ``[7, 21]`` range lives in
            :func:`dscraft.eda.sketches.estimate_cardinality`; this class accepts any ``log2_k >= 1``
            so it stays a pure algorithm component.

    Use :meth:`update` per value (or :meth:`update_many`), then read :meth:`estimate`.
    """

    def __init__(self, log2_k: int = 12) -> None:
        if log2_k < 1:
            raise ValueError(f"log2_k must be >= 1, got {log2_k!r}.")
        self.log2_k = int(log2_k)
        self.m = 1 << self.log2_k
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._register_bits = _HASH_BITS - self.log2_k

    def update(self, item) -> None:
        """Add one item to the sketch."""
        h = _hash64(item)
        register_index = h >> self._register_bits  # top log2_k bits select the register
        remaining = h & ((1 << self._register_bits) - 1)  # low bits for the rank
        rank = self._leading_zero_rank(remaining)
        if rank > self.registers[register_index]:
            self.registers[register_index] = rank

    def update_many(self, items) -> int:
        """Add every item in ``items``; return the count added."""
        count = 0
        for item in items:
            self.update(item)
            count += 1
        return count

    def _leading_zero_rank(self, value: int) -> int:
        """Position of the leftmost 1-bit (1-indexed) within the ``_register_bits``-wide value.

        Equivalently ``1 + number of leading zeros`` in the register-bit field. A value of 0 (all
        zero bits) ranks as ``_register_bits + 1``.
        """
        if value == 0:
            return self._register_bits + 1
        return self._register_bits - value.bit_length() + 1

    def relative_error(self) -> float:
        """Theoretical relative standard error ``1.04 / sqrt(m)`` for this ``log2_k``."""
        return 1.04 / np.sqrt(self.m)

    def estimate(self) -> float:
        """Current distinct-count estimate (bias-corrected, with small/large-range corrections)."""
        m = self.m
        # Raw HyperLogLog estimate via the harmonic mean of 2**-register.
        inverse_sum = float(np.sum(np.power(2.0, -self.registers.astype(np.float64))))
        raw = _alpha(m) * m * m / inverse_sum

        if raw <= 2.5 * m:
            # Small-range correction: linear counting over the empty registers, when any are empty.
            zeros = int(np.count_nonzero(self.registers == 0))
            if zeros != 0:
                return float(m * np.log(m / zeros))
            return raw
        # Large-range correction for estimates approaching the 2^64 hash space.
        two_pow_64 = float(2**_HASH_BITS)
        if raw > two_pow_64 / 30.0:
            return float(-two_pow_64 * np.log(1.0 - raw / two_pow_64))
        return raw
