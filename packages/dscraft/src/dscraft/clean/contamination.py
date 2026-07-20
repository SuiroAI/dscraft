"""Train/test contamination detection (architecture doc Part 3, "Module 2: LazyClean").

Implements the two-stage pipeline the architecture doc calls out for
train/test contamination auditing:

1. **Stage 1 -- LSHBloom candidate screening.** Cheap, scalable, run on every
   train/test pair. Each text row is turned into a MinHash signature (via the
   ``datasketch`` package -- see below), banded into ``b`` bands of ``r`` rows
   each, and every training-set band-bucket is recorded in a per-band Bloom
   filter. A test row is a stage-1 "candidate" if *any* of its bands collides
   with the training set's Bloom filter for that band; if *no* band collides,
   the row is classified clean immediately and stage 2 is never run for it.
   This mirrors LSHBloom's actual innovation over traditional LSH: replacing a
   prefix-tree/hashmap bucket index (which must store every bucket ID it has
   ever seen) with a fixed-size array of independent Bloom filters, trading a
   small, tunable false-positive rate for O(1)-ish, constant-memory-per-band
   bucket membership checks that scale to corpora far larger than a hashmap
   index could hold in memory.
2. **Stage 2 -- Min-K%++ validation.** Expensive (requires precomputed
   per-token log-probabilities from a language model this module never runs
   itself), so it only runs on stage-1 collisions. Implements both the base
   Min-K% score (Shi et al., "Detecting Pretraining Data from Large Language
   Models") and the Min-K%++ normalization on top of it (Zhang et al.,
   "Min-K%++: Improved Baseline for Detecting Pretraining Data from Large
   Language Models").

**How ``datasketch`` is used, precisely** (per this module's task spec, so
the dependency wiring step knows exactly what surface is depended on):
only ``datasketch.MinHash`` is used, and only two of its members:

- ``MinHash(num_perm=...)`` to construct a signature generator, and
  ``.update(shingle_bytes)`` (called once per shingle, UTF-8-encoded) to fold
  each shingle into the signature -- this is ``datasketch``'s own,
  well-tested MinHash implementation, not reimplemented here.
- ``.hashvalues`` -- the raw ``(num_perm,) uint32`` array of per-permutation
  minimum hash values -- read directly off the ``MinHash`` object rather than
  going through ``datasketch.MinHashLSH``, because this module needs custom
  control over the band split (to route each band through its own Bloom
  filter) rather than ``datasketch``'s own hashmap-backed bucket index. That
  custom band-to-Bloom-filter routing (:class:`BloomFilter` and
  :class:`LSHBloomIndex` below) is this module's own code, built directly on
  NumPy -- no additional dependency beyond ``datasketch`` itself is
  introduced for it.

Nothing in this module runs model inference or imports PyTorch/transformers:
stage 2's inputs (per-token log-probabilities, and either per-position
mean/std or a full vocabulary logit matrix) are caller-supplied, precomputed
arrays, and everything downstream of them is plain NumPy math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from datasketch import MinHash

__all__ = [
    "BloomFilter",
    "MinHashSignature",
    "compute_minhash_signature",
    "band_signature",
    "LSHBloomIndex",
    "min_k_percent_plus_plus_score",
    "ContaminationStatus",
    "ContaminationResult",
    "ContaminationReport",
    "ContaminationDetector",
    "detect_contamination",
]


# ---------------------------------------------------------------------------
# Bloom filter -- the custom part. A small, self-contained bit-packed array
# plus k hash functions derived via double hashing from two independent
# 64-bit hashes of the item, sized per the standard optimal-m/optimal-k
# Bloom-filter formulas given an expected item count and a target
# false-positive rate. No dependency beyond NumPy.
# ---------------------------------------------------------------------------


class BloomFilter:
    """A probabilistic set-membership structure: never a false negative, sometimes a false positive.

    Sized via the standard optimal Bloom-filter formulas given the expected
    number of items ``capacity`` and a target false-positive rate ``fp_rate``:

    - optimal bit-array size: ``m = -(n * ln(p)) / (ln(2)^2)``
    - optimal hash-function count: ``k = (m / n) * ln(2)``

    where ``n = capacity`` and ``p = fp_rate``. The bit array is packed into a
    ``numpy.uint8`` array (8 bits per byte) rather than one bool per bit, to
    keep the memory footprint close to the theoretical ``m`` bits rather than
    ``8 * m`` bytes.

    Hash functions are derived via double hashing (Kirsch-Mitzenmacher): two
    independent 64-bit hashes ``h1``, ``h2`` of the raw item bytes are
    combined as ``(h1 + i * h2) mod m`` for ``i in range(k)``, which is a
    well-known, standard technique for simulating ``k`` independent hash
    functions from only two real hash computations without materially
    hurting the false-positive-rate guarantees.
    """

    def __init__(self, capacity: int, *, fp_rate: float = 0.01) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}.")
        if not (0.0 < fp_rate < 1.0):
            raise ValueError(f"fp_rate must be in (0.0, 1.0), got {fp_rate!r}.")

        self.capacity = capacity
        self.fp_rate = fp_rate

        num_bits = -(capacity * math.log(fp_rate)) / (math.log(2) ** 2)
        self.num_bits = max(8, int(math.ceil(num_bits)))
        num_hashes = (self.num_bits / capacity) * math.log(2)
        self.num_hashes = max(1, int(math.ceil(num_hashes)))

        num_bytes = (self.num_bits + 7) // 8
        self._bits = np.zeros(num_bytes, dtype=np.uint8)
        self._count = 0

    def _double_hash(self, item: bytes) -> tuple[int, int]:
        """Two independent 64-bit hashes of ``item``, used to derive ``k`` bit positions."""
        # hashlib.blake2b with distinct salts gives two cheap, independent,
        # stable-across-processes 64-bit hashes (unlike builtin hash(),
        # which is randomized per-process unless PYTHONHASHSEED is fixed).
        import hashlib

        h1 = int.from_bytes(hashlib.blake2b(item, digest_size=8, salt=b"bloomh1\x00").digest(), "big")
        h2 = int.from_bytes(hashlib.blake2b(item, digest_size=8, salt=b"bloomh2\x00").digest(), "big")
        return h1, h2

    def _bit_positions(self, item: bytes) -> list[int]:
        h1, h2 = self._double_hash(item)
        # "Enhanced" double hashing (Dillinger & Manolios): plain
        # (h1 + i*h2) mod m degenerates to far fewer than num_hashes
        # *distinct* positions whenever gcd(h2, m) > 1 for a given item --
        # not a hypothetical edge case, it is occasionally hit in practice
        # for specific items/bit-array sizes. Adding a triangular-number
        # term i*(i+1)/2 breaks that arithmetic-progression degeneracy
        # without needing a third independent hash.
        return [
            (h1 + i * h2 + (i * (i + 1)) // 2) % self.num_bits for i in range(self.num_hashes)
        ]

    def add(self, item: bytes) -> None:
        """Record ``item`` (bytes) as present in the set."""
        for bit_index in self._bit_positions(item):
            byte_index, bit_offset = divmod(bit_index, 8)
            self._bits[byte_index] |= np.uint8(1 << bit_offset)
        self._count += 1

    def might_contain(self, item: bytes) -> bool:
        """Return whether ``item`` *might* be present.

        ``True`` means "possibly present" (may be a false positive).
        ``False`` is an absolute guarantee of "definitely not present" -- a
        Bloom filter never produces a false negative for an item that was
        actually added via :meth:`add`.
        """
        for bit_index in self._bit_positions(item):
            byte_index, bit_offset = divmod(bit_index, 8)
            if not (self._bits[byte_index] & np.uint8(1 << bit_offset)):
                return False
        return True

    @property
    def estimated_fp_rate(self) -> float:
        """Current estimated false-positive rate given items actually added so far.

        Uses the standard formula ``(1 - e^(-k*n/m))^k`` with the *actual*
        item count ``n`` inserted so far (not :attr:`capacity`), so this
        reflects the real current fill level rather than the design target.
        """
        if self._count == 0:
            return 0.0
        exponent = -self.num_hashes * self._count / self.num_bits
        return (1.0 - math.exp(exponent)) ** self.num_hashes


# ---------------------------------------------------------------------------
# MinHash signature generation (via datasketch.MinHash) + shingling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinHashSignature:
    """A MinHash signature for one text row, plus the shingling params it was built with."""

    hashvalues: np.ndarray  # (num_perm,) uint32/uint64, from datasketch.MinHash.hashvalues
    num_perm: int
    shingle_size: int


def _word_shingles(text: str, shingle_size: int) -> set[bytes]:
    """Word-level k-shingles of ``text``, encoded as UTF-8 bytes for ``MinHash.update``.

    Word-level (rather than character-level) shingles are the default here
    because they are more robust to minor spelling/formatting noise while
    still being sensitive to reordering and substitution at the phrase
    level -- a reasonable default for detecting near-duplicate natural-
    language passages between a train and test split. Character-level
    shingling is a reasonable alternative for very short strings (where a
    word-shingle set could be empty or size-1) but is not implemented here;
    callers needing it can pass their own precomputed shingle sets by
    calling :func:`compute_minhash_signature` with a custom ``shingle_size``
    tuned to their text, or by shingling text themselves before calling this
    module's lower-level pieces.
    """
    tokens = text.lower().split()
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        # Too short for a full shingle window -- treat the whole token
        # sequence as a single shingle rather than producing an empty set
        # (an empty shingle set would MinHash to a signature indistinguishable
        # from every other empty/near-empty text, over-collapsing distinct
        # short strings into the same LSH buckets).
        return {" ".join(tokens).encode("utf-8")}
    return {
        " ".join(tokens[i : i + shingle_size]).encode("utf-8")
        for i in range(len(tokens) - shingle_size + 1)
    }


def compute_minhash_signature(
    text: str,
    *,
    num_perm: int = 128,
    shingle_size: int = 4,
) -> MinHashSignature:
    """Compute a MinHash signature for ``text`` via ``datasketch.MinHash``.

    Shingles ``text`` into word-level k-shingles of size ``shingle_size``
    (default 4, within the sensible word-shingle range of 3-5 the LSHBloom
    literature typically uses), folds each shingle into a
    ``datasketch.MinHash(num_perm=num_perm)`` instance via ``.update()``, and
    returns the raw per-permutation minimum hash values
    (``MinHash.hashvalues``) wrapped in a :class:`MinHashSignature`.

    Args:
        text: the row's raw text.
        num_perm: number of independent hash-function permutations in the
            signature (must be evenly divisible by the band size ``r`` used
            later in :func:`band_signature` / :class:`LSHBloomIndex`).
        shingle_size: word-shingle window size (see :func:`_word_shingles`).

    Returns:
        A :class:`MinHashSignature` with a ``(num_perm,)`` array of hash
        values.
    """
    if num_perm <= 0:
        raise ValueError(f"num_perm must be positive, got {num_perm!r}.")
    if shingle_size <= 0:
        raise ValueError(f"shingle_size must be positive, got {shingle_size!r}.")

    minhash = MinHash(num_perm=num_perm)
    shingles = _word_shingles(text, shingle_size)
    for shingle in shingles:
        minhash.update(shingle)
    return MinHashSignature(
        hashvalues=np.asarray(minhash.hashvalues),
        num_perm=num_perm,
        shingle_size=shingle_size,
    )


def band_signature(signature: MinHashSignature, *, num_bands: int) -> list[bytes]:
    """Split a MinHash signature into ``num_bands`` bands and hash each to a bucket ID.

    ``signature.num_perm`` must be evenly divisible by ``num_bands`` (the
    standard LSH-banding constraint: ``num_perm = b * r`` for band count
    ``b`` and rows-per-band ``r``). Each band's ``r`` consecutive hash values
    are combined into one bucket-ID token: rows whose corresponding band's
    ``r`` MinHash values are identical (i.e. their shingle sets are similar
    enough that the same minimum-hash values were drawn for every
    permutation in that band) hash to the same bucket ID.

    Returns a list of ``num_bands`` bucket-ID byte-strings, one per band,
    suitable as the ``item`` argument to :class:`BloomFilter`.
    """
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands!r}.")
    if signature.num_perm % num_bands != 0:
        raise ValueError(
            f"num_perm ({signature.num_perm}) must be evenly divisible by "
            f"num_bands ({num_bands}), got remainder "
            f"{signature.num_perm % num_bands}."
        )
    rows_per_band = signature.num_perm // num_bands
    values = signature.hashvalues
    buckets: list[bytes] = []
    for band_index in range(num_bands):
        start = band_index * rows_per_band
        end = start + rows_per_band
        band_values = values[start:end]
        # Prefix with the band index so identical band contents in
        # *different* bands never collide with each other's Bloom filter --
        # each band gets its own BloomFilter instance in LSHBloomIndex, but
        # the bucket-ID bytes themselves are also disambiguated defensively.
        buckets.append(band_index.to_bytes(4, "big") + band_values.tobytes())
    return buckets


# ---------------------------------------------------------------------------
# LSHBloom index -- one Bloom filter per band, over the training set
# ---------------------------------------------------------------------------


class LSHBloomIndex:
    """LSHBloom candidate index: one :class:`BloomFilter` per LSH band, over a training set.

    Replaces the prefix-tree/hashmap bucket index of traditional MinHashLSH
    with a fixed-size array of ``num_bands`` independent Bloom filters (this
    is LSHBloom's specific contribution over vanilla LSH) -- each training
    item's band-bucket ID is recorded via :meth:`BloomFilter.add` in that
    band's filter at index time. A query item collides ("is a stage-1
    candidate") if *any* of its own bands' bucket IDs is present
    (:meth:`BloomFilter.might_contain`) in the corresponding training-set
    band filter.
    """

    def __init__(
        self,
        *,
        num_perm: int = 128,
        num_bands: int = 16,
        shingle_size: int = 4,
        expected_train_size: int = 1024,
        fp_rate: float = 0.01,
    ) -> None:
        if num_perm % num_bands != 0:
            raise ValueError(
                f"num_perm ({num_perm}) must be evenly divisible by "
                f"num_bands ({num_bands}), got remainder {num_perm % num_bands}."
            )
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.shingle_size = shingle_size
        self._band_filters = [
            BloomFilter(capacity=max(1, expected_train_size), fp_rate=fp_rate)
            for _ in range(num_bands)
        ]

    def index_train_texts(self, train_texts: Iterable[str]) -> None:
        """Add every training text's per-band bucket IDs to this index's Bloom filters."""
        for text in train_texts:
            signature = compute_minhash_signature(
                text, num_perm=self.num_perm, shingle_size=self.shingle_size
            )
            buckets = band_signature(signature, num_bands=self.num_bands)
            for band_index, bucket in enumerate(buckets):
                self._band_filters[band_index].add(bucket)

    def has_candidate_collision(self, text: str) -> bool:
        """Return whether ``text`` collides with the indexed training set in *any* band.

        ``True`` -- a stage-1 candidate, needs stage-2 validation.
        ``False`` -- no band collided anywhere; classified clean immediately,
        stage 2 is never run for this item.
        """
        signature = compute_minhash_signature(
            text, num_perm=self.num_perm, shingle_size=self.shingle_size
        )
        buckets = band_signature(signature, num_bands=self.num_bands)
        return any(
            self._band_filters[band_index].might_contain(bucket)
            for band_index, bucket in enumerate(buckets)
        )


# ---------------------------------------------------------------------------
# Stage 2 -- Min-K%++ validation
# ---------------------------------------------------------------------------


def min_k_percent_plus_plus_score(
    token_log_probs: np.ndarray,
    *,
    position_mean: np.ndarray | None = None,
    position_std: np.ndarray | None = None,
    vocab_logits: np.ndarray | None = None,
    k_percent: float = 20.0,
) -> float:
    """Compute the Min-K%++ contamination score for one sequence.

    Base Min-K% (Shi et al.): select the ``k_percent`` fraction of token
    positions with the LOWEST raw ``token_log_probs`` (the model's least
    confident/most-surprising tokens), and average their values. Genuinely
    unseen text tends to have low log-probability on its rarest tokens;
    memorized (contaminated) text tends to have anomalously higher
    log-probability there.

    Min-K%++ normalization (Zhang et al.): rather than averaging the raw
    log-probabilities of that bottom-K% selection, each selected position's
    log-probability is first normalized against that position's own
    vocabulary-wide logit distribution: ``(log_prob_t - mu_t) / sigma_t``,
    where ``mu_t``/``sigma_t`` are the mean/std of the full per-step
    vocabulary logit vector at position ``t``. This z-score-style transform
    corrects for positions where *every* token (not just the observed one)
    naturally has a high or low log-probability, which the un-normalized
    base Min-K% score cannot distinguish from genuine memorization.

    The bottom-K% *selection* is always made using the raw, un-normalized
    ``token_log_probs`` (matching the base Min-K% criterion) -- only the
    *scoring* of the selected positions is normalized.

    Args:
        token_log_probs: ``(seq_len,)`` array of the sequence's own observed
            per-token log-probabilities.
        position_mean: optional ``(seq_len,)`` array of precomputed
            per-position vocabulary-logit means (``mu_t``). This is the
            primary, preferred interface: computing this from a full
            ``[seq_len, vocab_size]`` logit matrix is cheap for the caller to
            do once (and cache/reuse), whereas passing the full matrix here
            on every call is often impractical at real vocabulary sizes
            (e.g. 128k tokens -- a single sequence's logits can already be
            hundreds of MB). Required together with ``position_std`` unless
            ``vocab_logits`` is supplied instead.
        position_std: optional ``(seq_len,)`` array of precomputed
            per-position vocabulary-logit standard deviations (``sigma_t``).
            Required together with ``position_mean`` unless ``vocab_logits``
            is supplied instead.
        vocab_logits: optional convenience path -- a full
            ``[seq_len, vocab_size]`` logit matrix. If supplied (and
            ``position_mean``/``position_std`` are not), ``mu_t``/``sigma_t``
            are computed internally via ``np.mean``/``np.std`` along the
            vocabulary axis. Prefer precomputing and passing
            ``position_mean``/``position_std`` directly when possible.
        k_percent: percentage (0, 100] of positions (by count, rounded up,
            at least 1) to select as the "least confident" bottom-K%.

    Returns:
        The Min-K%++ score (a float): the mean of ``(log_prob_t - mu_t) /
        sigma_t`` over the bottom-``k_percent`` positions by raw
        ``token_log_probs``. Higher (less negative / more positive) suggests
        contamination.

    Raises:
        ValueError: on shape mismatches, invalid ``k_percent``, a
            non-positive ``sigma_t``, or if neither
            ``(position_mean, position_std)`` nor ``vocab_logits`` is
            supplied.
    """
    token_log_probs = np.asarray(token_log_probs, dtype=np.float64)
    if token_log_probs.ndim != 1:
        raise ValueError(
            f"token_log_probs must be a 1D (seq_len,) array, got shape "
            f"{token_log_probs.shape!r}."
        )
    seq_len = token_log_probs.shape[0]
    if seq_len == 0:
        raise ValueError("token_log_probs must be non-empty.")
    if not (0.0 < k_percent <= 100.0):
        raise ValueError(f"k_percent must be in (0.0, 100.0], got {k_percent!r}.")

    if vocab_logits is not None:
        if position_mean is not None or position_std is not None:
            raise ValueError(
                "Pass either (position_mean, position_std) or vocab_logits, not both."
            )
        vocab_logits = np.asarray(vocab_logits, dtype=np.float64)
        if vocab_logits.ndim != 2:
            raise ValueError(
                f"vocab_logits must be a 2D (seq_len, vocab_size) array, got shape "
                f"{vocab_logits.shape!r}."
            )
        if vocab_logits.shape[0] != seq_len:
            raise ValueError(
                f"vocab_logits.shape[0] ({vocab_logits.shape[0]}) must match "
                f"len(token_log_probs) ({seq_len})."
            )
        position_mean = np.mean(vocab_logits, axis=1)
        position_std = np.std(vocab_logits, axis=1)
    else:
        if position_mean is None or position_std is None:
            raise ValueError(
                "Must supply either vocab_logits, or both position_mean and position_std."
            )
        position_mean = np.asarray(position_mean, dtype=np.float64)
        position_std = np.asarray(position_std, dtype=np.float64)
        if position_mean.ndim != 1 or position_mean.shape[0] != seq_len:
            raise ValueError(
                f"position_mean must be a 1D array of length {seq_len} "
                f"(matching token_log_probs), got shape {position_mean.shape!r}."
            )
        if position_std.ndim != 1 or position_std.shape[0] != seq_len:
            raise ValueError(
                f"position_std must be a 1D array of length {seq_len} "
                f"(matching token_log_probs), got shape {position_std.shape!r}."
            )

    if np.any(position_std <= 0.0):
        raise ValueError(
            "position_std must be strictly positive at every position (a zero or "
            "negative standard deviation makes the z-score normalization undefined)."
        )

    num_selected = max(1, math.ceil(seq_len * (k_percent / 100.0)))
    # Bottom-K% by raw log-probability -- the smallest (most negative) values.
    bottom_indices = np.argsort(token_log_probs)[:num_selected]

    normalized = (token_log_probs[bottom_indices] - position_mean[bottom_indices]) / position_std[
        bottom_indices
    ]
    return float(np.mean(normalized))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class ContaminationStatus(Enum):
    """The three possible per-test-item outcomes of the two-stage pipeline."""

    #: No LSHBloom band collided with the training set -- stage 2 was never run.
    CLEAN = "clean"
    #: A collision occurred and the caller supplied the data stage 2 needed;
    #: the Min-K%++ score exceeded ``contamination_threshold``.
    VALIDATED_CONTAMINATED = "validated_contaminated"
    #: A collision occurred but the caller did not supply the logprob/vocab-
    #: stats data needed to run stage 2 for this item -- neither cleared nor
    #: confirmed.
    CANDIDATE_UNVALIDATED = "candidate_unvalidated"


@dataclass(frozen=True)
class ContaminationResult:
    """The per-test-item outcome of the two-stage contamination pipeline."""

    index: int
    status: ContaminationStatus
    stage1_candidate: bool
    min_k_score: float | None = None


@dataclass
class ContaminationReport:
    """Result of running :class:`ContaminationDetector` / :func:`detect_contamination` over a test set."""

    results: list[ContaminationResult]
    num_train: int
    num_test: int
    contamination_threshold: float

    def clean_indices(self) -> list[int]:
        """Indices of test items classified clean by stage 1 alone (stage 2 never ran)."""
        return [r.index for r in self.results if r.status is ContaminationStatus.CLEAN]

    def validated_contaminated_indices(self) -> list[int]:
        """Indices of test items confirmed contaminated by stage 2."""
        return [
            r.index for r in self.results if r.status is ContaminationStatus.VALIDATED_CONTAMINATED
        ]

    def candidate_unvalidated_indices(self) -> list[int]:
        """Indices of test items that collided in stage 1 but had no stage-2 data supplied."""
        return [
            r.index for r in self.results if r.status is ContaminationStatus.CANDIDATE_UNVALIDATED
        ]


@dataclass
class ContaminationDetector:
    """Two-stage train/test contamination detector: LSHBloom screening + Min-K%++ validation.

    Stage 1 (:class:`LSHBloomIndex`) is cheap and runs on every test item.
    Stage 2 (:func:`min_k_percent_plus_plus_score`) only runs on the subset
    of test items that collided in stage 1 *and* for which the caller
    supplied the per-token log-probability (and mean/std or vocab-logit)
    data it needs -- see :meth:`detect`.

    Construct once per training corpus (indexing happens in ``__post_init__``
    via :meth:`index`, or call :meth:`index` explicitly / re-index a fresh
    training set) and reuse across multiple :meth:`detect` calls against
    different test batches.
    """

    num_perm: int = 128
    num_bands: int = 16
    shingle_size: int = 4
    fp_rate: float = 0.01
    k_percent: float = 20.0
    contamination_threshold: float = 0.0

    _index: LSHBloomIndex | None = field(default=None, init=False, repr=False)
    _num_train: int = field(default=0, init=False, repr=False)

    def index(self, train_texts: Sequence[str]) -> None:
        """Build (or rebuild) the stage-1 LSHBloom index over ``train_texts``."""
        train_texts = list(train_texts)
        self._index = LSHBloomIndex(
            num_perm=self.num_perm,
            num_bands=self.num_bands,
            shingle_size=self.shingle_size,
            expected_train_size=max(1, len(train_texts)),
            fp_rate=self.fp_rate,
        )
        self._index.index_train_texts(train_texts)
        self._num_train = len(train_texts)

    def detect(
        self,
        test_texts: Sequence[str],
        *,
        test_logprobs: Sequence[np.ndarray | None] | None = None,
        test_position_mean: Sequence[np.ndarray | None] | None = None,
        test_position_std: Sequence[np.ndarray | None] | None = None,
        test_vocab_logits: Sequence[np.ndarray | None] | None = None,
    ) -> ContaminationReport:
        """Run the two-stage pipeline over ``test_texts`` against the indexed training set.

        For each test item: run stage 1 (:meth:`LSHBloomIndex.has_candidate_collision`).
        If it does not collide, the item is :attr:`ContaminationStatus.CLEAN`
        and stage 2 is never invoked for it. If it does collide, stage 2 runs
        only if the caller supplied, for that item's index, either
        ``test_logprobs[i]`` plus (``test_position_mean[i]`` and
        ``test_position_std[i]``), or ``test_logprobs[i]`` plus
        ``test_vocab_logits[i]`` -- otherwise the item is reported as
        :attr:`ContaminationStatus.CANDIDATE_UNVALIDATED` rather than being
        silently skipped or raising.

        Args:
            test_texts: the test-set text rows to screen.
            test_logprobs: optional per-item ``(seq_len,)`` log-probability
                arrays (or ``None`` per item), aligned by index with
                ``test_texts``.
            test_position_mean: optional per-item ``(seq_len,)`` precomputed
                position-mean arrays (or ``None`` per item).
            test_position_std: optional per-item ``(seq_len,)`` precomputed
                position-std arrays (or ``None`` per item).
            test_vocab_logits: optional per-item ``(seq_len, vocab_size)``
                full logit matrices (or ``None`` per item) -- the convenience
                path for :func:`min_k_percent_plus_plus_score`; used only
                when ``test_position_mean``/``test_position_std`` are not
                supplied for that item.

        Returns:
            A :class:`ContaminationReport` with one :class:`ContaminationResult`
            per test item, in input order.

        Raises:
            ValueError: if :meth:`index` has not been called yet, or if any
                of the optional per-item sequences is supplied with a length
                that does not match ``len(test_texts)``.
        """
        if self._index is None:
            raise ValueError(
                "No training set indexed yet -- call .index(train_texts) before .detect()."
            )

        num_test = len(test_texts)
        for name, seq in (
            ("test_logprobs", test_logprobs),
            ("test_position_mean", test_position_mean),
            ("test_position_std", test_position_std),
            ("test_vocab_logits", test_vocab_logits),
        ):
            if seq is not None and len(seq) != num_test:
                raise ValueError(
                    f"{name} has length {len(seq)}, expected {num_test} (one entry per "
                    f"test_texts item, use None for items with no data)."
                )

        results: list[ContaminationResult] = []
        for i, text in enumerate(test_texts):
            is_candidate = self._index.has_candidate_collision(text)
            if not is_candidate:
                results.append(
                    ContaminationResult(
                        index=i, status=ContaminationStatus.CLEAN, stage1_candidate=False
                    )
                )
                continue

            logprobs = test_logprobs[i] if test_logprobs is not None else None
            pos_mean = test_position_mean[i] if test_position_mean is not None else None
            pos_std = test_position_std[i] if test_position_std is not None else None
            vocab_logits = test_vocab_logits[i] if test_vocab_logits is not None else None

            has_mean_std = pos_mean is not None and pos_std is not None
            can_validate = logprobs is not None and (has_mean_std or vocab_logits is not None)

            if not can_validate:
                results.append(
                    ContaminationResult(
                        index=i,
                        status=ContaminationStatus.CANDIDATE_UNVALIDATED,
                        stage1_candidate=True,
                    )
                )
                continue

            score = min_k_percent_plus_plus_score(
                logprobs,
                position_mean=pos_mean if has_mean_std else None,
                position_std=pos_std if has_mean_std else None,
                vocab_logits=None if has_mean_std else vocab_logits,
                k_percent=self.k_percent,
            )
            status = (
                ContaminationStatus.VALIDATED_CONTAMINATED
                if score >= self.contamination_threshold
                else ContaminationStatus.CLEAN
            )
            results.append(
                ContaminationResult(
                    index=i,
                    status=status,
                    stage1_candidate=True,
                    min_k_score=score,
                )
            )

        return ContaminationReport(
            results=results,
            num_train=self._num_train,
            num_test=num_test,
            contamination_threshold=self.contamination_threshold,
        )


def detect_contamination(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    *,
    test_logprobs: Sequence[np.ndarray | None] | None = None,
    test_position_mean: Sequence[np.ndarray | None] | None = None,
    test_position_std: Sequence[np.ndarray | None] | None = None,
    test_vocab_logits: Sequence[np.ndarray | None] | None = None,
    num_perm: int = 128,
    num_bands: int = 16,
    shingle_size: int = 4,
    fp_rate: float = 0.01,
    k_percent: float = 20.0,
    contamination_threshold: float = 0.0,
) -> ContaminationReport:
    """One-shot convenience wrapper: index ``train_texts`` then :meth:`ContaminationDetector.detect`.

    Prefer constructing a :class:`ContaminationDetector` directly and reusing
    it via multiple :meth:`~ContaminationDetector.detect` calls when
    screening more than one test batch against the same training set --
    this function rebuilds the stage-1 index from scratch on every call.

    See :meth:`ContaminationDetector.detect` for the full parameter and
    three-state-result documentation.
    """
    detector = ContaminationDetector(
        num_perm=num_perm,
        num_bands=num_bands,
        shingle_size=shingle_size,
        fp_rate=fp_rate,
        k_percent=k_percent,
        contamination_threshold=contamination_threshold,
    )
    detector.index(train_texts)
    return detector.detect(
        test_texts,
        test_logprobs=test_logprobs,
        test_position_mean=test_position_mean,
        test_position_std=test_position_std,
        test_vocab_logits=test_vocab_logits,
    )
