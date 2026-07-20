"""Tests for the two-stage LSHBloom + Min-K%++ train/test contamination pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.clean.contamination import (
    BloomFilter,
    ContaminationDetector,
    ContaminationStatus,
    LSHBloomIndex,
    band_signature,
    compute_minhash_signature,
    detect_contamination,
    min_k_percent_plus_plus_score,
)

TRAIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog in the park",
    "Machine learning models require large amounts of training data",
    "Quantum computing uses qubits to perform parallel computation",
    "The stock market fell sharply after the earnings report",
    "Deep neural networks have many layers of nonlinear transformations",
]


# ---------------------------------------------------------------------------
# MinHash / LSH signature generation
# ---------------------------------------------------------------------------


def test_near_identical_texts_produce_highly_similar_signatures():
    a = "The quick brown fox jumps over the lazy dog in the park"
    b = "the quick brown fox jumps over the lazy dog in the park!!"  # near-identical

    sig_a = compute_minhash_signature(a, num_perm=128, shingle_size=4)
    sig_b = compute_minhash_signature(b, num_perm=128, shingle_size=4)

    agreement = np.mean(sig_a.hashvalues == sig_b.hashvalues)
    assert agreement > 0.7  # most permutation slots agree for near-duplicate text


def test_near_identical_texts_collide_in_at_least_one_band():
    a = "The quick brown fox jumps over the lazy dog in the park"
    b = "the quick brown fox jumps over the lazy dog in the park!!"

    sig_a = compute_minhash_signature(a, num_perm=128, shingle_size=4)
    sig_b = compute_minhash_signature(b, num_perm=128, shingle_size=4)

    bands_a = band_signature(sig_a, num_bands=16)
    bands_b = band_signature(sig_b, num_bands=16)

    assert any(ba == bb for ba, bb in zip(bands_a, bands_b))


def test_very_different_texts_do_not_collide_across_all_bands():
    a = "The quick brown fox jumps over the lazy dog in the park"
    b = "Quantum entanglement enables non-local correlations between distant particles"

    sig_a = compute_minhash_signature(a, num_perm=128, shingle_size=4)
    sig_b = compute_minhash_signature(b, num_perm=128, shingle_size=4)

    bands_a = band_signature(sig_a, num_bands=16)
    bands_b = band_signature(sig_b, num_bands=16)

    # Near-certainly no shared bucket ID across any band for two unrelated
    # texts with essentially disjoint shingle vocabularies.
    assert not any(ba == bb for ba, bb in zip(bands_a, bands_b))


def test_band_signature_rejects_non_divisible_num_bands():
    sig = compute_minhash_signature("hello world", num_perm=128, shingle_size=2)
    with pytest.raises(ValueError, match="evenly divisible"):
        band_signature(sig, num_bands=17)


def test_compute_minhash_signature_rejects_non_positive_params():
    with pytest.raises(ValueError, match="num_perm"):
        compute_minhash_signature("hello", num_perm=0)
    with pytest.raises(ValueError, match="shingle_size"):
        compute_minhash_signature("hello", shingle_size=0)


# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------


def test_bloom_filter_add_and_might_contain_no_false_negatives():
    rng = np.random.default_rng(0)
    items = [f"item-{i}".encode() for i in range(500)]

    bloom = BloomFilter(capacity=500, fp_rate=0.01)
    for item in items:
        bloom.add(item)

    # A Bloom filter must NEVER say "not present" for something actually added.
    for item in items:
        assert bloom.might_contain(item) is True


def test_bloom_filter_absent_items_mostly_report_not_present():
    bloom = BloomFilter(capacity=500, fp_rate=0.01)
    for i in range(500):
        bloom.add(f"item-{i}".encode())

    # Items that were never added, drawn from a disjoint namespace.
    absent = [f"absent-{i}".encode() for i in range(500)]
    false_positives = sum(1 for item in absent if bloom.might_contain(item))
    false_positive_rate = false_positives / len(absent)

    # Generous tolerance around the 1% target -- this is probabilistic.
    assert false_positive_rate < 0.10


def test_bloom_filter_false_positive_rate_rough_sanity_check():
    target_fp_rate = 0.05
    n_items = 2000
    bloom = BloomFilter(capacity=n_items, fp_rate=target_fp_rate)

    for i in range(n_items):
        bloom.add(f"train-{i}".encode())

    n_queries = 2000
    false_positives = sum(
        1 for i in range(n_queries) if bloom.might_contain(f"query-{i}".encode())
    )
    observed_fp_rate = false_positives / n_queries

    # Generous tolerance: allow up to ~4x the target rate given the small
    # sample size and the coarse-grained k/m rounding in the constructor.
    assert observed_fp_rate < target_fp_rate * 4 + 0.02


def test_bloom_filter_rejects_invalid_params():
    with pytest.raises(ValueError, match="capacity"):
        BloomFilter(capacity=0)
    with pytest.raises(ValueError, match="fp_rate"):
        BloomFilter(capacity=10, fp_rate=1.5)


# ---------------------------------------------------------------------------
# Stage-1 screening end-to-end, including the "stage 2 never runs" guarantee
# ---------------------------------------------------------------------------


def test_stage1_flags_near_duplicate_test_item_as_candidate():
    index = LSHBloomIndex(num_perm=128, num_bands=16, shingle_size=4, expected_train_size=len(TRAIN_TEXTS))
    index.index_train_texts(TRAIN_TEXTS)

    near_dup_of_train_0 = "the quick brown fox jumps over the lazy dog in the park!!"
    assert index.has_candidate_collision(near_dup_of_train_0) is True


def test_stage1_clears_distinct_test_item_immediately():
    index = LSHBloomIndex(num_perm=128, num_bands=16, shingle_size=4, expected_train_size=len(TRAIN_TEXTS))
    index.index_train_texts(TRAIN_TEXTS)

    distinct_text = "A recipe for baking sourdough bread requires a live starter culture"
    assert index.has_candidate_collision(distinct_text) is False


def test_orchestration_never_invokes_stage2_for_immediately_clean_items(monkeypatch):
    calls = []
    import dscraft.clean.contamination as contamination_module

    original = contamination_module.min_k_percent_plus_plus_score

    def _instrumented(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(contamination_module, "min_k_percent_plus_plus_score", _instrumented)

    detector = ContaminationDetector(num_perm=128, num_bands=16, shingle_size=4)
    detector.index(TRAIN_TEXTS)

    distinct_text = "A recipe for baking sourdough bread requires a live starter culture"
    report = detector.detect(
        [distinct_text],
        test_logprobs=[np.array([-1.0, -2.0, -3.0])],
        test_position_mean=[np.array([-1.5, -2.5, -3.5])],
        test_position_std=[np.array([0.5, 0.5, 0.5])],
    )

    assert report.results[0].status is ContaminationStatus.CLEAN
    assert report.results[0].stage1_candidate is False
    assert report.results[0].min_k_score is None
    # The whole point of the two-stage funnel: stage 2 must never run for an
    # item that stage 1 already cleared, even if the caller supplied
    # everything stage 2 would have needed.
    assert calls == []


# ---------------------------------------------------------------------------
# Min-K%++ math
# ---------------------------------------------------------------------------


def test_min_k_percent_plus_plus_matches_hand_computation():
    # 10 positions, known log-probs and known per-position mean/std.
    token_log_probs = np.array(
        [-1.0, -5.0, -2.0, -8.0, -3.0, -0.5, -9.0, -1.5, -4.0, -0.1]
    )
    position_mean = np.array(
        [-2.0, -4.0, -3.0, -5.0, -3.5, -1.0, -6.0, -2.0, -3.0, -0.5]
    )
    position_std = np.array(
        [1.0, 2.0, 1.5, 2.0, 1.0, 0.5, 3.0, 1.0, 1.0, 0.5]
    )

    k_percent = 30.0  # -> 3 lowest-log-prob positions: indices 6, 3, 1 (values -9, -8, -5)
    expected_indices = np.argsort(token_log_probs)[:3]
    assert set(expected_indices.tolist()) == {6, 3, 1}

    expected_scores = (
        token_log_probs[expected_indices] - position_mean[expected_indices]
    ) / position_std[expected_indices]
    expected = float(np.mean(expected_scores))

    actual = min_k_percent_plus_plus_score(
        token_log_probs,
        position_mean=position_mean,
        position_std=position_std,
        k_percent=k_percent,
    )
    assert actual == pytest.approx(expected)


def test_min_k_percent_plus_plus_vocab_logits_convenience_path():
    rng = np.random.default_rng(42)
    seq_len, vocab_size = 6, 50
    vocab_logits = rng.normal(size=(seq_len, vocab_size))
    token_log_probs = rng.normal(size=seq_len)

    expected_mean = np.mean(vocab_logits, axis=1)
    expected_std = np.std(vocab_logits, axis=1)

    via_matrix = min_k_percent_plus_plus_score(
        token_log_probs, vocab_logits=vocab_logits, k_percent=50.0
    )
    via_precomputed = min_k_percent_plus_plus_score(
        token_log_probs,
        position_mean=expected_mean,
        position_std=expected_std,
        k_percent=50.0,
    )
    assert via_matrix == pytest.approx(via_precomputed)


def test_min_k_percent_plus_plus_rejects_shape_mismatches_and_bad_params():
    token_log_probs = np.array([-1.0, -2.0, -3.0])

    with pytest.raises(ValueError, match="k_percent"):
        min_k_percent_plus_plus_score(
            token_log_probs,
            position_mean=np.zeros(3),
            position_std=np.ones(3),
            k_percent=0.0,
        )

    with pytest.raises(ValueError, match="position_mean"):
        min_k_percent_plus_plus_score(
            token_log_probs,
            position_mean=np.zeros(2),  # wrong length
            position_std=np.ones(3),
        )

    with pytest.raises(ValueError, match="position_std"):
        min_k_percent_plus_plus_score(
            token_log_probs,
            position_mean=np.zeros(3),
            position_std=np.ones(2),  # wrong length
        )

    with pytest.raises(ValueError, match="either"):
        min_k_percent_plus_plus_score(token_log_probs)  # no mean/std or vocab_logits

    with pytest.raises(ValueError, match="not both"):
        min_k_percent_plus_plus_score(
            token_log_probs,
            position_mean=np.zeros(3),
            position_std=np.ones(3),
            vocab_logits=np.zeros((3, 5)),
        )

    with pytest.raises(ValueError, match="position_std"):
        min_k_percent_plus_plus_score(
            token_log_probs,
            position_mean=np.zeros(3),
            position_std=np.zeros(3),  # non-positive std
        )


# ---------------------------------------------------------------------------
# Orchestration three-state result
# ---------------------------------------------------------------------------


def _make_detector() -> ContaminationDetector:
    detector = ContaminationDetector(
        num_perm=128, num_bands=16, shingle_size=4, contamination_threshold=0.0
    )
    detector.index(TRAIN_TEXTS)
    return detector


def test_orchestration_state_clean():
    detector = _make_detector()
    distinct_text = "A recipe for baking sourdough bread requires a live starter culture"

    report = detector.detect([distinct_text])

    assert report.results[0].status is ContaminationStatus.CLEAN
    assert report.results[0].stage1_candidate is False
    assert report.clean_indices() == [0]


def test_orchestration_state_candidate_unvalidated():
    detector = _make_detector()
    near_dup = "the quick brown fox jumps over the lazy dog in the park!!"

    report = detector.detect([near_dup])  # no logprob data supplied

    assert report.results[0].status is ContaminationStatus.CANDIDATE_UNVALIDATED
    assert report.results[0].stage1_candidate is True
    assert report.candidate_unvalidated_indices() == [0]


def test_orchestration_state_validated_contaminated():
    detector = _make_detector()
    near_dup = "the quick brown fox jumps over the lazy dog in the park!!"

    # Construct logprobs/mean/std such that the Min-K%++ score is clearly
    # above the threshold=0.0: observed log-probs at the low-prob positions
    # are much HIGHER than the position mean (positive z-score).
    token_log_probs = np.array([-0.1, -0.2, -0.15, -0.3, -0.25])
    position_mean = np.array([-5.0, -5.0, -5.0, -5.0, -5.0])
    position_std = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    report = detector.detect(
        [near_dup],
        test_logprobs=[token_log_probs],
        test_position_mean=[position_mean],
        test_position_std=[position_std],
    )

    assert report.results[0].status is ContaminationStatus.VALIDATED_CONTAMINATED
    assert report.results[0].stage1_candidate is True
    assert report.results[0].min_k_score is not None
    assert report.results[0].min_k_score > 0.0
    assert report.validated_contaminated_indices() == [0]


def test_detect_raises_when_not_indexed():
    detector = ContaminationDetector()
    with pytest.raises(ValueError, match="index"):
        detector.detect(["hello"])


def test_detect_rejects_mismatched_optional_sequence_lengths():
    detector = _make_detector()
    with pytest.raises(ValueError, match="test_logprobs"):
        detector.detect(
            ["one", "two"],
            test_logprobs=[np.array([-1.0])],  # length 1, expected 2
        )


def test_detect_contamination_convenience_function_end_to_end():
    near_dup = "the quick brown fox jumps over the lazy dog in the park!!"
    distinct_text = "A recipe for baking sourdough bread requires a live starter culture"

    report = detect_contamination(
        TRAIN_TEXTS,
        [near_dup, distinct_text],
        test_logprobs=[np.array([-0.1, -0.2, -0.15]), None],
        test_position_mean=[np.array([-5.0, -5.0, -5.0]), None],
        test_position_std=[np.array([1.0, 1.0, 1.0]), None],
    )

    assert report.num_train == len(TRAIN_TEXTS)
    assert report.num_test == 2
    assert report.results[0].status is ContaminationStatus.VALIDATED_CONTAMINATED
    assert report.results[1].status is ContaminationStatus.CLEAN
    assert report.results[1].stage1_candidate is False
