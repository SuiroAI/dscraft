"""Tests for the composed Sanitizer / SanitizerReport wiring.

These tests exercise ``Sanitizer`` as the glue between three
independently-tested sibling modules (``label_errors.py``,
``contamination.py``, ``integrity.py`` -- see their own test files for
each module's own correctness proofs) rather than re-testing each
module's internal math. Two flavors of test are used:

1. **End-to-end via ``Sanitizer(...).audit(...)``** -- proves the wiring
   itself is correct (right columns feed the right sibling call, the
   returned :class:`SanitizerReport` is populated, and the "no logprobs
   supplied -> CANDIDATE_UNVALIDATED, never a fabricated verdict" contract
   documented in ``Sanitizer.audit``'s docstring actually holds).
2. **Directly constructing a ``SanitizerReport``** -- ``SanitizerReport``
   is a plain dataclass, so tests that need to pin down ``purge()``'s
   demographic-preserving capping math and its contamination-driven
   training-row removal to *exact*, hand-computed outcomes build the
   report directly with fully controlled mask/confidence/contamination
   inputs, instead of depending on ``detect_label_errors``'s own frac-based
   selection (already covered by ``test_label_errors.py``) to happen to
   produce a specific count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dscraft.clean import Sanitizer, SanitizerReport, detect_near_duplicate_text
from dscraft.clean.contamination import (
    ContaminationReport,
    ContaminationResult,
    ContaminationStatus,
)
from dscraft.clean.integrity import IntegrityReport

# ---------------------------------------------------------------------------
# Sanitizer.__init__ validation
# ---------------------------------------------------------------------------


def _make_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"row {i} some text content" for i in range(10)],
            "label": [0] * 5 + [1] * 5,
            "group": ["A"] * 5 + ["B"] * 5,
        }
    )


def test_sanitizer_init_rejects_missing_columns():
    df = _make_train_df()
    with pytest.raises(ValueError, match="target_col"):
        Sanitizer(df, target_col="nope", label_col="label", group_col="group")
    with pytest.raises(ValueError, match="label_col"):
        Sanitizer(df, target_col="text", label_col="nope", group_col="group")
    with pytest.raises(ValueError, match="group_col"):
        Sanitizer(df, target_col="text", label_col="label", group_col="nope")


def test_sanitizer_init_stores_references():
    df = _make_train_df()
    sanitizer = Sanitizer(df, target_col="text", label_col="label", group_col="group")
    assert sanitizer.df is df
    assert sanitizer.target_col == "text"
    assert sanitizer.label_col == "label"
    assert sanitizer.group_col == "group"


# ---------------------------------------------------------------------------
# Sanitizer.audit -- end-to-end composition
# ---------------------------------------------------------------------------


def _make_probs(class0_probs: list[float]) -> np.ndarray:
    class0 = np.array(class0_probs, dtype=float)
    return np.stack([class0, 1.0 - class0], axis=1)


def _audit_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """20 training rows across 2 groups (A: high-confidence-correct, B:
    lower-confidence-correct, mirroring test_label_errors.py's fixture) plus
    a small test split with one near-duplicate of a training row (a
    contamination stage-1 candidate) and one entirely distinct row (clean).
    """
    offsets = [-0.02, -0.01, 0.0, 0.01, 0.02, -0.015, -0.005, 0.005, 0.015, 0.0]
    group_a_probs = [0.95 + o for o in offsets]
    group_b_probs = [0.60 + o for o in offsets]

    labels = [0] * 20
    probs = _make_probs(group_a_probs + group_b_probs)
    groups = ["A"] * 10 + ["B"] * 10
    texts = [f"training passage number {i} about widgets and gadgets" for i in range(20)]
    # A longer sentence for row 0 (matching the near-duplicate construction
    # already proven to reliably collide in at least one LSH band in
    # test_contamination.py's own fixture) so the "near-duplicate of a
    # training row" stage-1-candidate scenario below is not left to chance
    # on a short 8-word sentence.
    texts[0] = "The quick brown fox jumps over the lazy dog in the park"

    train_df = pd.DataFrame({"text": texts, "label": labels, "group": groups})

    test_df = pd.DataFrame(
        {
            "text": [
                # Near-duplicate of training row 0 -- should collide in stage 1.
                texts[0] + "!!",
                "a completely unrelated sentence about sourdough bread baking",
            ],
            "group": ["A", "B"],
        }
    )
    return train_df, test_df, probs


def test_audit_returns_populated_sanitizer_report():
    train_df, test_df, probs = _audit_fixture()
    sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")

    report = sanitizer.audit(test_df, probs)

    assert isinstance(report, SanitizerReport)
    assert report.df is train_df
    assert report.test_df is test_df
    assert report.label_error_mask.dtype == bool
    assert report.label_error_mask.shape == (len(train_df),)
    assert report.own_label_confidence.shape == (len(train_df),)
    assert isinstance(report.contamination_report, ContaminationReport)
    assert report.contamination_report.num_train == len(train_df)
    assert report.contamination_report.num_test == len(test_df)
    assert isinstance(report.integrity_report, IntegrityReport)
    assert 0.0 <= report.integrity_report.score <= 1.0


def test_audit_never_fabricates_a_validated_contamination_verdict_without_logprobs():
    """Sanitizer.audit() never supplies stage-2 logprob data, so any stage-1
    collision must come back CANDIDATE_UNVALIDATED, never
    VALIDATED_CONTAMINATED -- per the documented contract."""
    train_df, test_df, probs = _audit_fixture()
    sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")

    report = sanitizer.audit(test_df, probs)

    statuses = {r.status for r in report.contamination_report.results}
    assert ContaminationStatus.VALIDATED_CONTAMINATED not in statuses
    # The near-duplicate test row (index 0) must be a stage-1 candidate.
    assert report.contamination_report.results[0].status is ContaminationStatus.CANDIDATE_UNVALIDATED
    # The distinct test row (index 1) must clear immediately.
    assert report.contamination_report.results[1].status is ContaminationStatus.CLEAN


def test_audit_rejects_test_df_missing_columns():
    train_df, test_df, probs = _audit_fixture()
    sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")
    bad_test_df = test_df.drop(columns=["group"])
    with pytest.raises(ValueError, match="group_col"):
        sanitizer.audit(bad_test_df, probs)


def test_audit_integrity_report_matches_manual_recomputation():
    """The IntegrityReport embedded in the SanitizerReport must be exactly
    reproducible by calling dataset_integrity_score with the same inputs
    Sanitizer.audit() is documented to construct."""
    from dscraft.clean.integrity import dataset_integrity_score

    train_df, test_df, probs = _audit_fixture()
    sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")
    report = sanitizer.audit(test_df, probs)

    contaminated_mask = np.array(
        [
            r.status is ContaminationStatus.VALIDATED_CONTAMINATED
            for r in report.contamination_report.results
        ],
        dtype=bool,
    )
    expected = dataset_integrity_score(
        label_error_input=report.label_error_mask,
        contamination_input=contaminated_mask,
        train_groups=train_df["group"].tolist(),
        test_groups=test_df["group"].tolist(),
    )
    assert report.integrity_report.score == pytest.approx(expected.score)
    assert report.integrity_report.label_error_rate == pytest.approx(expected.label_error_rate)
    assert report.integrity_report.contamination_rate == pytest.approx(expected.contamination_rate)
    assert report.integrity_report.drift == pytest.approx(expected.drift)


def test_detect_near_duplicate_text_still_exists_unchanged():
    """The lower-level building block is not superseded by Sanitizer."""
    assert callable(detect_near_duplicate_text)


# ---------------------------------------------------------------------------
# SanitizerReport.purge -- basic flagged-row removal
# ---------------------------------------------------------------------------


def _empty_contamination_report(num_train: int, num_test: int) -> ContaminationReport:
    return ContaminationReport(
        results=[
            ContaminationResult(index=i, status=ContaminationStatus.CLEAN, stage1_candidate=False)
            for i in range(num_test)
        ],
        num_train=num_train,
        num_test=num_test,
        contamination_threshold=0.0,
    )


def test_purge_rejects_unknown_strategy():
    df = _make_train_df()
    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(len(df), dtype=bool),
        own_label_confidence=np.ones(len(df)),
        contamination_report=_empty_contamination_report(len(df), len(df)),
        integrity_report=IntegrityReport(
            score=1.0, label_error_rate=0.0, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )
    with pytest.raises(ValueError, match="strategy"):
        report.purge(strategy="bogus")


def test_purge_with_no_flagged_rows_returns_full_copy():
    df = _make_train_df()
    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(len(df), dtype=bool),
        own_label_confidence=np.ones(len(df)),
        contamination_report=_empty_contamination_report(len(df), len(df)),
        integrity_report=IntegrityReport(
            score=1.0, label_error_rate=0.0, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )
    cleaned = report.purge()
    assert len(cleaned) == len(df)
    pd.testing.assert_frame_equal(cleaned, df.reset_index(drop=True))


def test_purge_removes_flagged_label_error_rows_when_under_cap():
    """With flags spread evenly across groups (no group over-represented),
    every flagged row is removed -- the cap never engages."""
    df = _make_train_df()  # 10 rows, group A = idx 0-4, group B = idx 5-9
    label_mask = np.zeros(10, dtype=bool)
    label_mask[[1, 6]] = True  # one flagged row per group -- balanced

    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=label_mask,
        own_label_confidence=np.full(10, 0.9),
        contamination_report=_empty_contamination_report(10, 10),
        integrity_report=IntegrityReport(
            score=0.8, label_error_rate=0.2, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )
    cleaned = report.purge()
    assert len(cleaned) == 8
    assert "row 1 some text content" not in cleaned["text"].tolist()
    assert "row 6 some text content" not in cleaned["text"].tolist()


def test_purge_output_path_writes_parquet_and_csv(tmp_path):
    df = _make_train_df()
    label_mask = np.zeros(10, dtype=bool)
    label_mask[0] = True
    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=label_mask,
        own_label_confidence=np.full(10, 0.9),
        contamination_report=_empty_contamination_report(10, 10),
        integrity_report=IntegrityReport(
            score=0.9, label_error_rate=0.1, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )

    parquet_path = tmp_path / "cleaned.parquet"
    cleaned_parquet = report.purge(output_path=parquet_path)
    assert parquet_path.exists()
    assert len(pd.read_parquet(parquet_path)) == len(cleaned_parquet) == 9

    csv_path = tmp_path / "cleaned.csv"
    cleaned_csv = report.purge(output_path=csv_path)
    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == len(cleaned_csv) == 9


def test_purge_rejects_unsupported_output_extension(tmp_path):
    df = _make_train_df()
    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(10, dtype=bool),
        own_label_confidence=np.full(10, 0.9),
        contamination_report=_empty_contamination_report(10, 10),
        integrity_report=IntegrityReport(
            score=1.0, label_error_rate=0.0, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )
    with pytest.raises(ValueError, match="Unsupported output_path extension"):
        report.purge(output_path=tmp_path / "cleaned.json")


# ---------------------------------------------------------------------------
# SanitizerReport.purge -- demographic-preserving capping (exact math)
# ---------------------------------------------------------------------------


def test_purge_demographic_preserving_caps_over_represented_group():
    """Group 'A' (size 10) has 8 flagged rows (80%); group 'B' (size 40) has
    zero flagged rows. Overall rate = 8/50 = 16%. The cap allows group 'A'
    to lose at most floor(min(1.0, 0.16 * 2) * 10) = floor(0.32 * 10) = 3
    rows -- so only the 3 lowest-own-label-confidence flagged rows in group
    A are actually removed; the other 5 are reprieved.
    """
    n_a, n_b = 10, 40
    n = n_a + n_b
    texts = [f"text-{i}" for i in range(n)]
    groups = ["A"] * n_a + ["B"] * n_b
    labels = [0] * n
    df = pd.DataFrame({"text": texts, "label": labels, "group": groups})

    label_mask = np.zeros(n, dtype=bool)
    flagged_a = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 of group A's 10 rows flagged
    label_mask[flagged_a] = True

    # Distinct confidence per flagged row so the "drop lowest confidence
    # first" order is unambiguous. Rows 0,1,2 are the 3 least confident
    # (lowest own_label_confidence) among the flagged group-A rows, so they
    # are exactly the ones purge() should remove; 3-7 should be reprieved.
    own_label_confidence = np.full(n, 0.99)
    own_label_confidence[0] = 0.10
    own_label_confidence[1] = 0.20
    own_label_confidence[2] = 0.30
    own_label_confidence[3] = 0.40
    own_label_confidence[4] = 0.50
    own_label_confidence[5] = 0.60
    own_label_confidence[6] = 0.70
    own_label_confidence[7] = 0.80

    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=label_mask,
        own_label_confidence=own_label_confidence,
        contamination_report=_empty_contamination_report(n, n),
        integrity_report=IntegrityReport(
            score=0.84, label_error_rate=0.16, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )

    cleaned = report.purge()

    remaining_texts = set(cleaned["text"].tolist())
    # Exactly 3 removed from group A (the 3 least confident flagged rows).
    assert "text-0" not in remaining_texts
    assert "text-1" not in remaining_texts
    assert "text-2" not in remaining_texts
    # The other 5 flagged-but-reprieved group-A rows must survive.
    for i in range(3, 8):
        assert f"text-{i}" in remaining_texts
    # Untouched group-A rows and all of group B must survive.
    assert "text-8" in remaining_texts
    assert "text-9" in remaining_texts
    for i in range(n_a, n):
        assert f"text-{i}" in remaining_texts

    assert len(cleaned) == n - 3


def test_purge_demographic_preserving_does_not_cap_when_groups_are_balanced():
    """If every group's flagged share matches the dataset-wide rate, no
    group is ever capped -- all flagged rows are removed."""
    n_a, n_b = 20, 20
    texts = [f"text-{i}" for i in range(n_a + n_b)]
    groups = ["A"] * n_a + ["B"] * n_b
    df = pd.DataFrame({"text": texts, "label": [0] * (n_a + n_b), "group": groups})

    label_mask = np.zeros(n_a + n_b, dtype=bool)
    # 2 flagged in each group -- identical 10% rate in both groups and
    # overall, well under any cap.
    label_mask[[0, 1, 20, 21]] = True

    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=label_mask,
        own_label_confidence=np.full(n_a + n_b, 0.5),
        contamination_report=_empty_contamination_report(n_a + n_b, n_a + n_b),
        integrity_report=IntegrityReport(
            score=0.9, label_error_rate=0.1, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )

    cleaned = report.purge()
    assert len(cleaned) == n_a + n_b - 4
    remaining = set(cleaned["text"].tolist())
    for removed in ("text-0", "text-1", "text-20", "text-21"):
        assert removed not in remaining


# ---------------------------------------------------------------------------
# SanitizerReport.purge -- contamination-driven training-row removal
# ---------------------------------------------------------------------------


def test_purge_removes_training_rows_matching_validated_contaminated_test_items():
    """A validated-contaminated test item that is a near-duplicate of a
    specific training row causes purge() to remove that training row, even
    though it was never flagged by the label-error mask."""
    train_texts = [
        "The quick brown fox jumps over the lazy dog in the park",
        "Completely unrelated training passage about gardening tools",
        "Another unrelated passage about mountain hiking trails",
    ]
    df = pd.DataFrame(
        {
            "text": train_texts,
            "label": [0, 0, 0],
            "group": ["A", "A", "A"],
        }
    )
    # Near-duplicate of train_texts[0].
    contaminated_test_text = "the quick brown fox jumps over the lazy dog in the park!!"
    test_df = pd.DataFrame({"text": [contaminated_test_text], "group": ["A"]})

    contamination_report = ContaminationReport(
        results=[
            ContaminationResult(
                index=0,
                status=ContaminationStatus.VALIDATED_CONTAMINATED,
                stage1_candidate=True,
                min_k_score=1.23,
            )
        ],
        num_train=3,
        num_test=1,
        contamination_threshold=0.0,
    )

    report = SanitizerReport(
        df=df,
        test_df=test_df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(3, dtype=bool),
        own_label_confidence=np.full(3, 0.9),
        contamination_report=contamination_report,
        integrity_report=IntegrityReport(
            score=0.5, label_error_rate=0.0, contamination_rate=1.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )

    cleaned = report.purge()
    remaining = set(cleaned["text"].tolist())
    assert train_texts[0] not in remaining
    assert train_texts[1] in remaining
    assert train_texts[2] in remaining
    assert len(cleaned) == 2


def test_purge_ignores_candidate_unvalidated_contamination():
    """CANDIDATE_UNVALIDATED items (no stage-2 data) must never cause a
    training row to be removed -- only VALIDATED_CONTAMINATED does."""
    train_texts = [
        "The quick brown fox jumps over the lazy dog in the park",
        "Completely unrelated training passage about gardening tools",
    ]
    df = pd.DataFrame({"text": train_texts, "label": [0, 0], "group": ["A", "A"]})
    test_df = pd.DataFrame(
        {"text": ["the quick brown fox jumps over the lazy dog in the park!!"], "group": ["A"]}
    )

    contamination_report = ContaminationReport(
        results=[
            ContaminationResult(
                index=0, status=ContaminationStatus.CANDIDATE_UNVALIDATED, stage1_candidate=True
            )
        ],
        num_train=2,
        num_test=1,
        contamination_threshold=0.0,
    )

    report = SanitizerReport(
        df=df,
        test_df=test_df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(2, dtype=bool),
        own_label_confidence=np.full(2, 0.9),
        contamination_report=contamination_report,
        integrity_report=IntegrityReport(
            score=1.0, label_error_rate=0.0, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )

    cleaned = report.purge()
    assert len(cleaned) == 2
    assert set(cleaned["text"].tolist()) == set(train_texts)


def test_purge_rejects_mismatched_label_error_mask_length():
    df = _make_train_df()
    report = SanitizerReport(
        df=df,
        test_df=df,
        target_col="text",
        label_col="label",
        group_col="group",
        label_error_mask=np.zeros(3, dtype=bool),  # wrong length (df has 10 rows)
        own_label_confidence=np.full(10, 0.9),
        contamination_report=_empty_contamination_report(10, 10),
        integrity_report=IntegrityReport(
            score=1.0, label_error_rate=0.0, contamination_rate=0.0, drift=0.0, weights=(1 / 3, 1 / 3, 1 / 3)
        ),
    )
    with pytest.raises(ValueError, match="label_error_mask length"):
        report.purge()


# ---------------------------------------------------------------------------
# Sanitizer.audit -- mixed-type/unsortable labels (CodeRabbit regression)
# ---------------------------------------------------------------------------


def test_audit_tolerates_mixed_type_unsortable_labels():
    """`_own_label_confidence`'s categorical-label branch used to do a bare
    ``sorted(set(labels))``, which raises ``TypeError`` for genuinely
    mixed-type labels (e.g. a mix of strings and ints) -- even though the
    sibling ``label_errors.detect_label_errors`` call already tolerates
    exactly this via ``_sorted_unique``'s insertion-order fallback. This
    proves ``Sanitizer.audit()`` no longer crashes on that same label data.
    """
    labels = ["a", 1, "a", 1]
    texts = [f"training passage number {i} about widgets" for i in range(4)]
    groups = ["A", "A", "B", "B"]
    train_df = pd.DataFrame({"text": texts, "label": labels, "group": groups})
    test_df = pd.DataFrame(
        {"text": ["a totally unrelated sentence about sourdough bread"], "group": ["A"]}
    )
    # Two distinct label values ("a", 1) -> a 2-class probability matrix.
    probs = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.3, 0.7],
        ]
    )
    sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")

    report = sanitizer.audit(test_df, probs)

    assert isinstance(report, SanitizerReport)
    assert report.own_label_confidence.shape == (len(train_df),)
    assert np.all(np.isfinite(report.own_label_confidence))
