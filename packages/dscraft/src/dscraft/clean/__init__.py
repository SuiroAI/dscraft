"""dscraft.clean -- LazyClean: data-quality firewall for tabular/text datasets.

Public API surface (architecture doc Part 3, "Module 2: LazyClean"):

- :func:`detect_near_duplicate_text` -- embed a batch of text rows via
  native ONNX Runtime (no PyTorch, no ``transformers``), then flag
  near-duplicate row pairs via cosine-similarity thresholding over those
  embeddings -- a minimal stand-in for the Density-Based Semantic
  Deduplication (D4) idea. See ``dedup.py`` for the explicit naive-O(n^2)
  scope boundary and ``embeddings.py`` for the PyTorch-free embedding path.
  This remains the lower-level building block it always was; it is not
  superseded by anything below.
- :class:`Sanitizer` / :class:`SanitizerReport` -- the primary, composed
  entrypoint for auditing and cleaning a *training* DataFrame against a
  held-out *test* DataFrame. ``Sanitizer.audit()`` wires together three
  independently-implemented sibling modules -- DeCoLe group-conditioned
  label-error detection (``label_errors.py``), two-stage LSHBloom +
  Min-K%++ train/test contamination auditing (``contamination.py``), and
  the aggregate Dataset Integrity Score (``integrity.py``) -- into one
  :class:`SanitizerReport`, and ``SanitizerReport.purge()`` turns that
  audit into an actual cleaned-and-written-out DataFrame. See the
  :class:`Sanitizer` docstring below for the exact composition and the
  ``"demographic-preserving"`` purge strategy's precise, documented
  interpretation.
- :func:`detect_near_duplicate_images` -- the image-modality counterpart of
  ``detect_near_duplicate_text``: embed a batch of already-decoded images
  via native ONNX Runtime (no PyTorch, no CLIP-specific Python package),
  then flag near-duplicate image pairs by reusing ``dedup.py``'s existing,
  modality-agnostic near-duplicate scan. See ``image_dedup.py`` for the
  full rationale (and ``docs/decisions/2026-07-image-dedup-evaluation.md``
  for the evaluation that motivated it).

The IVF-HNSW ANN index and spherical mini-batch k-means scale-out path for
:func:`detect_near_duplicate_text` (see ``dedup.py``'s own module
docstring for that explicit naive-O(n^2) scope boundary) remain out of
scope for this pass -- see README.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Sequence, Union

import numpy as np

from dscraft.core.data import is_arrow_backed_pandas

from .contamination import (
    ContaminationReport,
    ContaminationStatus,
    compute_minhash_signature,
    detect_contamination,
)
from .dedup import DedupReport, DuplicatePair, cosine_similarity_matrix, find_near_duplicates
from .embeddings import (
    MODEL_ALLOWLIST,
    RECOMMENDED_MODEL_NAME,
    EmbeddingModel,
    build_synthetic_embedding_model,
    build_synthetic_embedding_onnx,
    download_recommended_model,
    hashing_bag_of_words_vectorizer,
)
from .image_dedup import (
    RECOMMENDED_IMAGE_MODEL_NAME,
    ImageEmbeddingModel,
    ModelIntegrityError,
    build_synthetic_image_embedding_model,
    build_synthetic_image_embedding_onnx,
    detect_near_duplicate_images,
    download_recommended_clip_vision_model,
    resize_and_normalize,
)
from .integrity import IntegrityReport, dataset_integrity_score
from .label_errors import detect_label_errors

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import pandas as pd
    import polars as pl

__all__ = [
    "EmbeddingModel",
    "MODEL_ALLOWLIST",
    "RECOMMENDED_MODEL_NAME",
    "build_synthetic_embedding_model",
    "build_synthetic_embedding_onnx",
    "download_recommended_model",
    "hashing_bag_of_words_vectorizer",
    "DedupReport",
    "DuplicatePair",
    "cosine_similarity_matrix",
    "find_near_duplicates",
    "detect_near_duplicate_text",
    "RECOMMENDED_IMAGE_MODEL_NAME",
    "ImageEmbeddingModel",
    "ModelIntegrityError",
    "build_synthetic_image_embedding_model",
    "build_synthetic_image_embedding_onnx",
    "download_recommended_clip_vision_model",
    "resize_and_normalize",
    "detect_near_duplicate_images",
    "Sanitizer",
    "SanitizerReport",
]

#: Text rows may be supplied as a plain iterable of strings, or as a
#: Tier-1 Arrow-backed column (a pandas Series or a Polars Series), per
#: dscraft.core.data's §2.1 conventions.
TextRows = "Iterable[str] | pd.Series | pl.Series"


def _coerce_text_rows(rows: object) -> list[str]:
    """Normalize the accepted input shapes down to a plain ``list[str]``.

    Accepts a plain iterable of strings, or a Tier-1 Arrow-backed pandas
    Series / Polars Series (see ``dscraft.core.data``, architecture doc §2.1).
    A pandas Series that is *not* Arrow-backed is still accepted (this
    package does not hard-require Tier-1 storage to function), but emits a
    warning pointing at the convention, using
    ``dscraft.core.data.is_arrow_backed_pandas`` for the check -- reusing
    dscraft.core's Tier-1 helper rather than re-implementing an Arrow-dtype
    check here.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(rows, pd.DataFrame):
        raise TypeError(
            "Expected a single text column (a pandas Series), not a full "
            "DataFrame. Pass e.g. `frame['text_column']`."
        )

    if pd is not None and isinstance(rows, pd.Series):
        if not is_arrow_backed_pandas(rows.to_frame()):
            warnings.warn(
                "Input pandas Series is not Arrow-backed (pandas 2.x "
                "ArrowDtype). dscraft.clean follows dscraft.core's Tier-1 "
                "convention (architecture doc §2.1) for zero-copy interop "
                "across DSCraft modules; consider "
                "`series.convert_dtypes(dtype_backend='pyarrow')`. "
                "Proceeding anyway -- this is not a hard requirement.",
                stacklevel=3,
            )
        return [str(value) for value in rows.tolist()]

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None and isinstance(rows, pl.Series):
        return [str(value) for value in rows.to_list()]

    return [str(value) for value in rows]  # type: ignore[union-attr]


def detect_near_duplicate_text(
    rows: object,
    model: EmbeddingModel,
    *,
    threshold: float = 0.92,
) -> tuple[np.ndarray, DedupReport]:
    """Embed ``rows`` via ONNX Runtime and flag near-duplicate row pairs.

    This is the one canonical entrypoint tying the embedding path
    (``embeddings.py``) to the dedup path (``dedup.py``) together -- prefer
    it over calling ``model.embed`` + ``find_near_duplicates`` separately
    unless you need the embeddings for something else in between.

    Args:
        rows: an iterable of text strings, or a Tier-1 Arrow-backed pandas
            Series / Polars Series (a single text column).
        model: an :class:`EmbeddingModel` (e.g. from
            :func:`build_synthetic_embedding_model` for hermetic use, or a
            real production model wired via
            :meth:`EmbeddingModel.from_onnx_file`).
        threshold: cosine-similarity cutoff in ``(0.0, 1.0]`` -- see
            :func:`find_near_duplicates`.

    Returns:
        ``(embeddings, report)`` -- the ``(n_rows, embedding_dim)`` float32
        embedding array and the :class:`DedupReport` of flagged pairs.
    """
    texts = _coerce_text_rows(rows)
    embeddings = model.embed(texts)
    report = find_near_duplicates(embeddings, threshold=threshold)
    return embeddings, report


# ---------------------------------------------------------------------------
# Sanitizer: the composed Sanitizer/SanitizerReport entrypoint tying
# label_errors.py, contamination.py, and integrity.py together.
# ---------------------------------------------------------------------------


def _own_label_confidence(labels: object, probs: object) -> np.ndarray:
    """Per-row predicted probability for each row's own observed label.

    Mirrors -- without importing -- the label-encoding convention documented
    in ``dscraft.clean.label_errors._validate_and_encode`` (integer values
    used directly as class indices; any other dtype treated as categorical
    and mapped to indices by sorted-unique order) just closely enough to
    extract each row's own-label confidence for
    :meth:`SanitizerReport.purge`'s demographic-preserving ranking. This is
    a small, independent, read-only re-derivation rather than an import of
    ``label_errors``'s private helper, keeping this module's coupling to its
    siblings at "shared public API only," matching the decoupling
    convention already documented in ``integrity.py``'s module docstring.
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels)
    if np.issubdtype(labels_arr.dtype, np.integer) or labels_arr.dtype == np.bool_:
        label_idx = labels_arr.astype(int)
    else:
        items = labels_arr.tolist()
        try:
            unique_values = sorted(set(items))
        except TypeError:
            # Genuinely mixed-type/unsortable labels (e.g. a mix of strings
            # and numbers) -- fall back to first-seen (insertion) order,
            # mirroring label_errors.py's `_sorted_unique` fallback so
            # Sanitizer.audit() tolerates exactly the label data that
            # detect_label_errors() alone would have tolerated.
            seen: dict[Any, None] = {}
            for item in items:
                seen.setdefault(item, None)
            unique_values = list(seen.keys())
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        label_idx = np.array([value_to_index[value] for value in labels_arr.tolist()], dtype=int)
    return probs_arr[np.arange(label_idx.shape[0]), label_idx]


def _find_near_duplicate_train_rows(
    test_texts: Sequence[str],
    train_texts: Sequence[str],
    *,
    similarity_threshold: float = 0.5,
    num_perm: int = 128,
) -> set[int]:
    """Identify training-row indices likely responsible for validated contamination.

    ``contamination.py``'s :class:`~dscraft.clean.contamination.ContaminationReport`
    only tracks per-*test*-item outcomes (its stage-1 LSHBloom index is a
    set-membership screen, not a per-pair index, by design -- see that
    module's docstring), so it never records *which* training row(s) a
    validated-contaminated test item collided with. To let
    :meth:`SanitizerReport.purge` remove the actual contaminating training
    rows (not just flag the test side, which it does not own), this helper
    re-derives an approximate per-pair match directly: it computes a MinHash
    signature (:func:`dscraft.clean.contamination.compute_minhash_signature`)
    for every training text and every given (already-validated-contaminated)
    test text, and flags every training row whose fraction of agreeing
    MinHash permutation slots -- an unbiased estimator of Jaccard
    shingle-set similarity -- meets or exceeds ``similarity_threshold``.

    This is a small, deliberately simple ``O(n_test * n_train)`` direct
    comparison (bypassing the banding/Bloom-filter machinery, which is built
    for *scalable set-membership* screening, not *identifying* a specific
    matching training row) -- acceptable for the same "not a substitute for
    a real ANN index at scale" reason ``dedup.py``'s naive O(n^2) pass is
    acceptable, and only ever run over the (typically small)
    validated-contaminated subset of the test set, not the whole test set.
    """
    if not test_texts or not train_texts:
        return set()
    train_signatures = [
        compute_minhash_signature(text, num_perm=num_perm) for text in train_texts
    ]
    matches: set[int] = set()
    for test_text in test_texts:
        test_signature = compute_minhash_signature(test_text, num_perm=num_perm)
        for idx, train_signature in enumerate(train_signatures):
            agreement = float(np.mean(test_signature.hashvalues == train_signature.hashvalues))
            if agreement >= similarity_threshold:
                matches.add(idx)
    return matches


def _write_dataframe(df: "pd.DataFrame", output_path: Union[str, Path]) -> None:
    """Write ``df`` to ``output_path``, inferring the format from its file extension.

    Supports ``.parquet`` (via ``DataFrame.to_parquet``) and ``.csv`` (via
    ``DataFrame.to_csv(index=False)``). Any other extension raises
    ``ValueError`` naming the unsupported suffix rather than guessing a
    format.
    """
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output_path extension {suffix!r} (from {output_path!r}); "
            "expected '.parquet' or '.csv'."
        )


class Sanitizer:
    """Compose DeCoLe, contamination auditing, and the Dataset Integrity Score
    into one training-data audit workflow.

    This is the primary, documented entrypoint for ``dscraft.clean``'s
    label-error / contamination / integrity capabilities -- see
    :func:`detect_near_duplicate_text` for the separate, lower-level
    near-duplicate-detection entrypoint, which ``Sanitizer`` does not
    replace or call.

    ``Sanitizer`` itself does no computation; it just stores a reference to
    the *training* DataFrame and the three column names its downstream
    calls need. All the actual work happens in :meth:`audit`.

    Args:
        df: the training ``pandas.DataFrame`` to audit (and, later, clean
            via :meth:`SanitizerReport.purge`).
        target_col: name of the text/feature column used for train/test
            contamination screening (passed to
            :func:`dscraft.clean.contamination.detect_contamination`).
        label_col: name of the (possibly noisy) observed-label column,
            passed to :func:`dscraft.clean.label_errors.detect_label_errors`.
        group_col: name of the sensitive/demographic group-attribute column,
            passed both to DeCoLe's per-group thresholding and to the
            Dataset Integrity Score's demographic-drift component.

    Raises:
        ValueError: if any of ``target_col``/``label_col``/``group_col`` is
            not a column of ``df``.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        target_col: str,
        label_col: str,
        group_col: str,
    ) -> None:
        for name, col in (
            ("target_col", target_col),
            ("label_col", label_col),
            ("group_col", group_col),
        ):
            if col not in df.columns:
                raise ValueError(
                    f"{name}={col!r} is not a column of df; available columns: "
                    f"{list(df.columns)!r}."
                )
        self.df = df
        self.target_col = target_col
        self.label_col = label_col
        self.group_col = group_col

    def audit(self, test_df: "pd.DataFrame", out_of_sample_probs: object) -> "SanitizerReport":
        """Run DeCoLe + contamination auditing + the Dataset Integrity Score together.

        Composes, in order:

        1. :func:`dscraft.clean.label_errors.detect_label_errors` over the
           training DataFrame's ``label_col``/``group_col`` and the supplied
           ``out_of_sample_probs`` (an ``(n_train, n_classes)`` array of
           out-of-sample predicted probabilities the caller must have
           produced themselves, e.g. via k-fold cross-validation --
           ``Sanitizer`` never trains a model itself). Produces a per-row
           boolean label-error mask.
        2. :func:`dscraft.clean.contamination.detect_contamination` over the
           training and test DataFrames' ``target_col`` text columns. This
           call deliberately does **not** supply Min-K%++'s stage-2
           per-token log-probability/vocab-logit inputs (they require a
           language model that neither ``Sanitizer`` nor
           ``dscraft.clean.contamination`` runs) -- so every stage-1
           collision this simple audit flow finds legitimately comes back
           as :attr:`~dscraft.clean.contamination.ContaminationStatus.CANDIDATE_UNVALIDATED`,
           not a fabricated validated/clean verdict. A caller with real
           per-token log-probabilities available should call
           :func:`dscraft.clean.contamination.detect_contamination` (or
           :class:`~dscraft.clean.contamination.ContaminationDetector`)
           directly with that data and feed the resulting report's states
           into :func:`dscraft.clean.integrity.dataset_integrity_score`
           itself for a fully-validated integrity score.
        3. :func:`dscraft.clean.integrity.dataset_integrity_score`, combining
           (1)'s label-error mask, (2)'s per-test-item
           *validated*-contaminated boolean mask (candidate-unvalidated
           items are conservatively treated as not-yet-confirmed, per that
           function's own documented three-state semantics), and the
           ``group_col`` distributions of the training vs. test DataFrames.

        Args:
            test_df: the held-out test ``pandas.DataFrame``. Must contain
                ``target_col`` and ``group_col`` (a test-side ``label_col``
                is not required -- this audit never scores test labels).
            out_of_sample_probs: ``(n_train, n_classes)`` array of
                out-of-sample predicted class probabilities for the
                *training* DataFrame's rows, aligned by row order with
                ``self.df``. See :func:`dscraft.clean.label_errors.detect_label_errors`.

        Returns:
            A :class:`SanitizerReport` bundling the label-error mask, the
            full :class:`~dscraft.clean.contamination.ContaminationReport`,
            the :class:`~dscraft.clean.integrity.IntegrityReport`, and
            everything :meth:`SanitizerReport.purge` needs to act on them.

        Raises:
            ValueError: if ``test_df`` is missing ``target_col`` or
                ``group_col``, or if any composed call raises (see their
                own docstrings -- e.g. mismatched lengths, empty inputs).
        """
        for name, col in (("target_col", self.target_col), ("group_col", self.group_col)):
            if col not in test_df.columns:
                raise ValueError(
                    f"{name}={col!r} is not a column of test_df; available columns: "
                    f"{list(test_df.columns)!r}."
                )

        train_labels = self.df[self.label_col].tolist()
        train_groups = self.df[self.group_col].tolist()
        label_error_mask = detect_label_errors(train_labels, out_of_sample_probs, train_groups)

        train_texts = [str(value) for value in self.df[self.target_col].tolist()]
        test_texts = [str(value) for value in test_df[self.target_col].tolist()]
        contamination_report = detect_contamination(train_texts, test_texts)

        # Explicitly constructed as a genuine numpy bool array (rather than
        # left as a plain Python list) before being passed to
        # contamination_rate_component (via dataset_integrity_score).
        # contamination_rate_component now robustly coerces a plain
        # `list[bool]` (or a 0/1 int list) to the same boolean mask itself
        # -- see its docstring -- so this explicit `dtype=bool` construction
        # is no longer required to avoid a miscomputation. It is kept
        # anyway as good, self-documenting defensive practice: it makes the
        # intended "this is a boolean mask, not a sequence of state labels"
        # contract obvious at the call site regardless of what any callee
        # does internally.
        contaminated_test_mask = np.array(
            [
                result.status is ContaminationStatus.VALIDATED_CONTAMINATED
                for result in contamination_report.results
            ],
            dtype=bool,
        )

        integrity_report = dataset_integrity_score(
            label_error_input=label_error_mask,
            contamination_input=contaminated_test_mask,
            train_groups=train_groups,
            test_groups=test_df[self.group_col].tolist(),
        )

        own_label_confidence = _own_label_confidence(train_labels, out_of_sample_probs)

        return SanitizerReport(
            df=self.df,
            test_df=test_df,
            target_col=self.target_col,
            label_col=self.label_col,
            group_col=self.group_col,
            label_error_mask=label_error_mask,
            own_label_confidence=own_label_confidence,
            contamination_report=contamination_report,
            integrity_report=integrity_report,
        )


@dataclass
class SanitizerReport:
    """Bundled result of :meth:`Sanitizer.audit`.

    Attributes:
        df: the training DataFrame the audit was run over (same object
            passed to :class:`Sanitizer`).
        test_df: the test DataFrame passed to :meth:`Sanitizer.audit`.
        target_col: the text/feature column name.
        label_col: the observed-label column name.
        group_col: the demographic/group-attribute column name.
        label_error_mask: boolean ``(len(df),)`` array from
            :func:`dscraft.clean.label_errors.detect_label_errors` --
            ``True`` marks a suspected label error in ``df``.
        own_label_confidence: ``(len(df),)`` float array -- each training
            row's out-of-sample predicted probability for its own observed
            label (used by :meth:`purge`'s demographic-preserving ranking;
            see :func:`_own_label_confidence`).
        contamination_report: the full
            :class:`~dscraft.clean.contamination.ContaminationReport` from
            auditing ``test_df`` against ``df``.
        integrity_report: the aggregate
            :class:`~dscraft.clean.integrity.IntegrityReport`.
    """

    df: "pd.DataFrame"
    test_df: "pd.DataFrame"
    target_col: str
    label_col: str
    group_col: str
    label_error_mask: np.ndarray
    own_label_confidence: np.ndarray
    contamination_report: ContaminationReport
    integrity_report: IntegrityReport

    #: A group's realized removal rate is never allowed to exceed the
    #: dataset-wide removal rate by more than this multiplier -- see
    #: :meth:`purge`'s docstring for the exact, documented interpretation of
    #: "demographic-preserving."
    _MAX_GROUP_REMOVAL_MULTIPLIER: ClassVar[float] = 2.0
    #: MinHash-agreement threshold used by :func:`_find_near_duplicate_train_rows`
    #: to identify contaminating training rows during :meth:`purge`.
    _NEAR_DUPLICATE_SIMILARITY_THRESHOLD: ClassVar[float] = 0.5

    def purge(
        self,
        strategy: str = "demographic-preserving",
        output_path: Union[str, "Path", None] = None,
    ) -> "pd.DataFrame":
        """Return (and optionally write) a cleaned copy of the training DataFrame.

        **Only the ``"demographic-preserving"`` strategy is implemented.**
        Any other value raises ``ValueError``.

        **Exact, documented interpretation of "demographic-preserving"**
        (the source architecture doc does not specify an algorithm for this,
        so this is this implementation's own, deliberate choice):

        1. **Identify every row flagged for removal.** A training row is
           flagged if either (a) :attr:`label_error_mask` marks it as a
           suspected label error, or (b) it is a near-duplicate (MinHash
           agreement >= ``_NEAR_DUPLICATE_SIMILARITY_THRESHOLD``, see
           :func:`_find_near_duplicate_train_rows`) of a test item
           :attr:`contamination_report` validated as contaminated -- i.e.
           the specific training row(s) responsible for that contamination.
        2. **Compute the dataset-wide removal rate** (flagged rows / total
           rows).
        3. **Cap each group's realized removal rate.** For each distinct
           value of ``group_col``, if that group's own flagged-row count
           would remove more than ``_MAX_GROUP_REMOVAL_MULTIPLIER`` times
           the dataset-wide removal rate from *that group alone*, only the
           top ``allowed_count = floor(min(1.0, dataset_rate *
           _MAX_GROUP_REMOVAL_MULTIPLIER) * group_size)`` of that group's
           flagged rows are actually removed; the rest are reprieved
           (kept). This is the "demographic-preserving" behavior: no single
           group can be pruned at more than a bounded multiple of the
           dataset-wide rate, so a group that happens to have
           disproportionately more flagged rows is not disproportionately
           emptied out.
        4. **Within an over-capped group, drop the most-confidently-bad
           rows first** (i.e. keep the borderline ones): contamination-
           matched rows are treated as maximally confident removals
           (dropped before any label-error-only row), and among
           label-error-flagged rows, the ones with the *lowest*
           :attr:`own_label_confidence` (the model's own out-of-sample
           confidence in that row's observed label -- lower means more
           suspect) are dropped first. This means when a group must be
           capped, the rows reprieved back into the cleaned dataset are the
           ones this audit was *least* sure about, not an arbitrary subset.

        Row order is preserved (via ``DataFrame.loc`` boolean indexing) and
        the returned DataFrame's index is reset (``reset_index(drop=True)``).

        Args:
            strategy: purge strategy; only ``"demographic-preserving"`` is
                implemented.
            output_path: if given, the cleaned DataFrame is additionally
                written there, with the format inferred from the file
                extension (``.parquet`` or ``.csv`` -- see
                :func:`_write_dataframe`). The cleaned DataFrame is always
                returned regardless of whether ``output_path`` is given.

        Returns:
            The cleaned training ``pandas.DataFrame``.

        Raises:
            ValueError: for an unknown ``strategy``, a mismatched
                ``label_error_mask`` length, or an unsupported
                ``output_path`` file extension.
        """
        if strategy != "demographic-preserving":
            raise ValueError(
                f"Unknown purge strategy {strategy!r}; only 'demographic-preserving' "
                "is currently implemented."
            )

        df = self.df
        n = len(df)
        label_mask = np.asarray(self.label_error_mask, dtype=bool)
        if label_mask.shape[0] != n:
            raise ValueError(
                f"label_error_mask length ({label_mask.shape[0]}) does not match the "
                f"training df length ({n})."
            )

        contaminating_train_mask = np.zeros(n, dtype=bool)
        contaminated_test_indices = self.contamination_report.validated_contaminated_indices()
        if contaminated_test_indices:
            contaminated_test_texts = [
                str(self.test_df.iloc[i][self.target_col]) for i in contaminated_test_indices
            ]
            train_texts = [str(value) for value in df[self.target_col].tolist()]
            matching_train_indices = _find_near_duplicate_train_rows(
                contaminated_test_texts,
                train_texts,
                similarity_threshold=self._NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
            )
            if matching_train_indices:
                contaminating_train_mask[list(matching_train_indices)] = True

        flagged_mask = label_mask | contaminating_train_mask
        if not flagged_mask.any():
            cleaned_df = df.copy().reset_index(drop=True)
            if output_path is not None:
                _write_dataframe(cleaned_df, output_path)
            return cleaned_df

        # Removal priority: lower means "drop first" when a group must be
        # capped. Contamination-matched rows are always most confidently
        # bad (dropped first); label-error rows are ranked by their own
        # out-of-sample confidence in the observed label (lower = more
        # suspect = dropped first).
        priority = np.where(
            contaminating_train_mask,
            -np.inf,
            np.asarray(self.own_label_confidence, dtype=float),
        )

        groups = np.asarray(df[self.group_col].tolist(), dtype=object)
        overall_rate = float(flagged_mask.sum()) / n

        final_removal_mask = np.zeros(n, dtype=bool)
        for group in sorted(set(groups.tolist()), key=repr):
            group_positions = np.where(groups == group)[0]
            group_flagged = group_positions[flagged_mask[group_positions]]
            if group_flagged.size == 0:
                continue

            group_size = group_positions.size
            allowed_rate = min(1.0, overall_rate * self._MAX_GROUP_REMOVAL_MULTIPLIER)
            allowed_count = int(np.floor(allowed_rate * group_size))

            if group_flagged.size <= allowed_count:
                final_removal_mask[group_flagged] = True
                continue

            # Over the cap: only remove the allowed_count most-confidently-
            # bad rows in this group; the rest are reprieved.
            group_priority = priority[group_flagged]
            order = np.argsort(group_priority, kind="stable")
            rows_to_remove = group_flagged[order[:allowed_count]]
            final_removal_mask[rows_to_remove] = True

        cleaned_df = df.loc[~final_removal_mask].reset_index(drop=True)

        if output_path is not None:
            _write_dataframe(cleaned_df, output_path)

        return cleaned_df
