"""Tests for cosine-similarity near-duplicate detection over embeddings.

Threshold choice (0.9): with the hashing bag-of-words preprocessor, two
sentences that differ only in punctuation/case tokenize to the *identical*
bag of words, so their embeddings are effectively identical (cosine
similarity ~1.0). A sentence about a completely different topic shares
almost no vocabulary, so its embedding direction is essentially unrelated
(cosine similarity well below 0.9 in practice, typically near/at 0 for a
fixed random linear projection of two disjoint feature-hash supports).
0.9 sits comfortably between those two regimes for this scaffold's fixture,
without being so close to 1.0 that ordinary token-order/whitespace noise
would produce false negatives on true near-duplicates.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dscraft.clean import detect_near_duplicate_text
from dscraft.clean.dedup import (
    DedupReport,
    DuplicatePair,
    cosine_similarity_matrix,
    find_near_duplicates,
)
from dscraft.clean.embeddings import build_synthetic_embedding_model

NEAR_DUP_THRESHOLD = 0.9

ROWS = [
    "The quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog!!",  # near-duplicate of row 0
    "Quantum entanglement enables non-local correlations between particles",  # distinct
]


@pytest.fixture()
def model(tmp_path):
    """Build a small synthetic ONNX embedding model, cached under ``tmp_path``."""
    return build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=128, embedding_dim=32)


def test_cosine_similarity_matrix_shape_and_diagonal():
    """A cosine-similarity matrix is (n, n), has a unit diagonal, and rates
    identical vectors as similarity 1.0 and orthogonal vectors as 0.0."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sims = cosine_similarity_matrix(embeddings)
    assert sims.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sims), 1.0, atol=1e-6)
    assert sims[0, 2] == pytest.approx(1.0, abs=1e-6)  # identical vectors
    assert sims[0, 1] == pytest.approx(0.0, abs=1e-6)  # orthogonal vectors


def test_cosine_similarity_matrix_zero_vectors_are_undefined_not_zero_or_one():
    """A zero embedding means the vectorizer extracted no features, not that
    the source rows are equal or distinct.

    Two earlier bugs bracket the correct behavior here:

    - Originally, a zero-vector row's cosine similarity to itself or to
      another zero vector always read 0.0 (the old implementation clamped
      zero norms to 1.0 before normalizing, leaving the zero vector as the
      zero vector) -- silently missing genuinely-identical zero-feature
      rows (e.g. two empty strings) as duplicates.
    - A subsequent fix over-corrected by making ANY two zero-vector rows
      read similarity 1.0 -- silently flagging unrelated zero-feature rows
      (e.g. "!!!" and "???", which share no tokens at all) as duplicates of
      each other at every valid threshold.

    Neither silent guess is right: a hashing bag-of-words vectorizer that
    extracted zero features genuinely cannot tell whether two zero-feature
    texts are "the same". The correct, honest answer is NaN -- undefined,
    not comparable -- for every pair involving at least one zero-vector
    row, including a zero-vector row against itself and against a
    genuinely non-zero row.
    """
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],  # zero row A ("")
            [0.0, 0.0, 0.0],  # zero row B ("   ") -- identical zero vector
            [1.0, 0.0, 0.0],  # genuinely non-zero row
        ],
        dtype=np.float32,
    )
    sims = cosine_similarity_matrix(embeddings)

    # Self-similarity of a zero vector is undefined, not 1.0 and not 0.0.
    assert np.isnan(sims[0, 0])
    assert np.isnan(sims[1, 1])

    # Two zero vectors against each other are undefined too -- NOT silently
    # 1.0 (over-correction) and NOT silently 0.0 (original bug).
    assert np.isnan(sims[0, 1])
    assert np.isnan(sims[1, 0])

    # A zero vector against a genuinely non-zero vector is also undefined,
    # not silently 0.0 -- the vectorizer still extracted no features from
    # the zero row, so no similarity claim can be made either way.
    assert np.isnan(sims[0, 2])
    assert np.isnan(sims[1, 2])

    # The non-zero row against itself is unaffected -- normal cosine
    # similarity still applies once neither row is a zero vector.
    assert sims[2, 2] == pytest.approx(1.0)


def test_find_near_duplicates_never_flags_zero_vector_rows_as_duplicates():
    """find_near_duplicates() never flags a zero-vector row as a duplicate of
    anything -- not of a genuinely non-zero row, and not even of another
    zero-vector row (the prior over-correction). Zero-vector rows are
    instead surfaced separately via ``zero_vector_row_indices``."""
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],  # zero row A
            [0.0, 0.0, 0.0],  # zero row B -- identical zero vector to A
            [1.0, 0.0, 0.0],  # genuinely non-zero row
        ],
        dtype=np.float32,
    )
    report = find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)

    flagged = report.flagged_indices()
    assert flagged == set()  # no pair is a confirmed duplicate here
    assert not any({pair.index_a, pair.index_b} == {0, 1} for pair in report.pairs)

    # Both zero-vector rows are reported as "not comparable", separately
    # from the duplicate/distinct pair scan.
    assert set(report.zero_vector_row_indices) == {0, 1}
    assert 2 not in report.zero_vector_row_indices


def test_find_near_duplicates_flags_identical_non_empty_text_no_regression():
    """Regression guard: two rows with the exact same real (non-empty)
    content still embed to the same non-zero vector and are still flagged
    as duplicates -- the zero-vector fix must not touch this path at all."""
    embeddings = np.array(
        [
            [0.6, 0.8, 0.0],  # non-zero row
            [0.6, 0.8, 0.0],  # identical non-zero row -- a true duplicate
            [0.0, 0.0, 1.0],  # distinct non-zero row
        ],
        dtype=np.float32,
    )
    report = find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)

    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)
    assert 2 not in flagged
    assert report.zero_vector_row_indices == []


def test_cosine_similarity_matrix_rejects_non_2d_input():
    """A 1-D array (not an (n_rows, dim) matrix) raises ValueError."""
    with pytest.raises(ValueError):
        cosine_similarity_matrix(np.zeros(5, dtype=np.float32))


def test_find_near_duplicates_rejects_bad_threshold():
    """Thresholds outside the valid (0.0, 1.0] range raise ValueError."""
    embeddings = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError):
        find_near_duplicates(embeddings, threshold=0.0)
    with pytest.raises(ValueError):
        find_near_duplicates(embeddings, threshold=1.5)


def test_find_near_duplicates_rejects_non_2d_input_with_clear_error():
    """A 1-D embeddings array raises a clear ValueError from
    find_near_duplicates() itself -- not a confusing error surfaced later
    from inside np.linalg.norm(embeddings, axis=1, ...)."""
    embeddings = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)


def test_find_near_duplicates_rejects_non_finite_values():
    """A row containing NaN or Inf raises ValueError rather than silently
    producing NaN similarities that get skipped by the near-duplicate scan
    while also being absent from zero_vector_row_indices -- which would
    otherwise make a NaN/Inf row indistinguishable from a row that was
    normally, successfully compared."""
    nan_embeddings = np.array(
        [[1.0, 0.0], [np.nan, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="finite"):
        find_near_duplicates(nan_embeddings, threshold=NEAR_DUP_THRESHOLD)

    inf_embeddings = np.array(
        [[1.0, 0.0], [np.inf, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="finite"):
        find_near_duplicates(inf_embeddings, threshold=NEAR_DUP_THRESHOLD)


def test_find_near_duplicates_flags_the_near_duplicate_pair(model):
    """End-to-end via the synthetic model: the near-duplicate ROWS[0]/ROWS[1]
    pair is flagged above threshold, and the unrelated ROWS[2] is not."""
    embeddings = model.embed(ROWS)
    report = find_near_duplicates(embeddings, threshold=NEAR_DUP_THRESHOLD)

    assert isinstance(report, DedupReport)
    assert report.num_rows == 3
    assert report.threshold == NEAR_DUP_THRESHOLD

    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)  # the near-duplicate pair is flagged
    assert 2 not in flagged  # the distinct row is not flagged

    assert any(
        {pair.index_a, pair.index_b} == {0, 1} and pair.similarity >= NEAR_DUP_THRESHOLD
        for pair in report.pairs
    )


def test_dedup_report_pairs_sorted_by_descending_similarity(model):
    """DedupReport.pairs is sorted by descending similarity, most-similar
    pair first."""
    embeddings = model.embed(ROWS + ["the quick brown fox jumps over the lazy dog"])
    report = find_near_duplicates(embeddings, threshold=0.1)
    similarities = [pair.similarity for pair in report.pairs]
    assert similarities == sorted(similarities, reverse=True)


def test_detect_near_duplicate_text_end_to_end(model):
    """detect_near_duplicate_text() embeds plain string rows and returns
    both the embeddings and a report flagging the expected near-duplicate
    pair."""
    embeddings, report = detect_near_duplicate_text(ROWS, model, threshold=NEAR_DUP_THRESHOLD)
    assert embeddings.shape == (3, model.embedding_dim)
    assert {0, 1}.issubset(report.flagged_indices())
    assert 2 not in report.flagged_indices()


def test_detect_near_duplicate_text_accepts_arrow_backed_pandas_series(model):
    """detect_near_duplicate_text() accepts a Tier-1 Arrow-backed pandas
    Series (ArrowDtype) directly, without needing to convert to a plain
    list first."""
    pd = pytest.importorskip("pandas")
    series = pd.Series(ROWS).convert_dtypes(dtype_backend="pyarrow")
    embeddings, report = detect_near_duplicate_text(series, model, threshold=NEAR_DUP_THRESHOLD)
    assert embeddings.shape == (3, model.embedding_dim)
    assert {0, 1}.issubset(report.flagged_indices())
    assert 2 not in report.flagged_indices()


def test_detect_near_duplicate_text_reports_empty_and_whitespace_only_text_as_not_comparable(
    model,
):
    """End-to-end regression test for the original failure scenario, updated
    for the corrected design: two genuinely empty-text rows ("" and "   ")
    tokenize to zero tokens via hashing_bag_of_words_vectorizer, so both
    embed to the identical all-zero vector through the real
    EmbeddingModel/ONNX path (not hand-constructed arrays).

    This used to assert the two empty rows were flagged as duplicate pairs
    (similarity 1.0). That was itself an over-correction of the original
    bug (which silently read their similarity as 0.0, missing them
    entirely) -- a hashing bag-of-words vectorizer that extracted zero
    features from both cannot actually tell whether "" and "   " are "the
    same" text or not, so neither silent guess is honest. The corrected,
    most honest behavior is to report them as "zero-vector, not
    comparable" via ``zero_vector_row_indices`` rather than either silently
    missing them (original bug) or silently calling them duplicates
    (the over-correction).
    """
    rows = ["", "   ", "The quick brown fox jumps over the lazy dog"]
    embeddings, report = detect_near_duplicate_text(rows, model, threshold=NEAR_DUP_THRESHOLD)

    # Sanity check that these two rows really do embed to the same all-zero
    # vector via the real preprocessor + ONNX model, i.e. this test actually
    # exercises the zero-vector scenario rather than some other coincidental
    # match.
    np.testing.assert_array_equal(embeddings[0], np.zeros(model.embedding_dim, dtype=np.float32))
    np.testing.assert_array_equal(embeddings[1], np.zeros(model.embedding_dim, dtype=np.float32))

    # Not flagged as a duplicate pair -- a zero-vector row's similarity to
    # anything, including another zero-vector row, is undefined.
    assert report.flagged_indices() == set()
    # Instead surfaced separately as "could not be compared".
    assert set(report.zero_vector_row_indices) == {0, 1}
    assert 2 not in report.zero_vector_row_indices


def test_detect_near_duplicate_text_distinct_zero_vector_texts_are_not_flagged_as_duplicates(
    model,
):
    """Two DIFFERENT texts that both happen to produce all-zero embeddings
    (because hashing_bag_of_words_vectorizer's ``[a-z0-9]+`` tokenizer
    matches no tokens in either of them) are not actually duplicates of
    each other, and must not be flagged as such at any valid threshold --
    this is exactly the CodeRabbit-flagged over-correction scenario:
    punctuation-only and non-ASCII rows collapsing to the same zero vector
    used to be reported as duplicates."""
    rows = [
        "!!!",  # punctuation only -- no [a-z0-9]+ tokens
        "???",  # different punctuation only -- also no tokens, NOT a duplicate of row 0
        "日本語",  # non-ASCII ("日本語") -- also no [a-z0-9]+ tokens
        "The quick brown fox jumps over the lazy dog",  # genuinely non-zero, distinct
    ]
    embeddings, report = detect_near_duplicate_text(rows, model, threshold=NEAR_DUP_THRESHOLD)

    # Sanity check: rows 0-2 really do all collapse to the identical
    # all-zero vector, i.e. this test exercises the real over-correction
    # scenario and not some other coincidental match.
    zero_vec = np.zeros(model.embedding_dim, dtype=np.float32)
    np.testing.assert_array_equal(embeddings[0], zero_vec)
    np.testing.assert_array_equal(embeddings[1], zero_vec)
    np.testing.assert_array_equal(embeddings[2], zero_vec)

    # None of the zero-vector rows are flagged as duplicates of each other
    # or of anything else.
    assert report.flagged_indices() == set()
    assert set(report.zero_vector_row_indices) == {0, 1, 2}
    assert 3 not in report.zero_vector_row_indices


def test_detect_near_duplicate_text_identical_non_empty_rows_still_flagged_no_regression(
    model,
):
    """No-regression check: two rows with the exact same real, non-empty
    content still embed to the same non-zero vector through the real
    EmbeddingModel/ONNX path and are still flagged as duplicates -- the
    zero-vector fix must not weaken detection of genuine duplicates."""
    rows = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",  # exact duplicate
        "Quantum entanglement enables non-local correlations between particles",
    ]
    embeddings, report = detect_near_duplicate_text(rows, model, threshold=NEAR_DUP_THRESHOLD)

    assert not np.all(embeddings[0] == 0.0)  # sanity: genuinely non-zero
    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)
    assert 2 not in flagged
    assert report.zero_vector_row_indices == []


def test_duplicate_pair_is_frozen_dataclass():
    """DuplicatePair is immutable: assigning to a field after construction
    raises."""
    pair = DuplicatePair(index_a=0, index_b=1, similarity=0.99)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.similarity = 0.5  # type: ignore[misc]
