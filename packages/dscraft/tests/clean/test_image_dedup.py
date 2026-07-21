"""Tests for the PyTorch-free ONNX Runtime image-embedding + near-duplicate path.

Fully hermetic: uses build_synthetic_image_embedding_model() (a hand-built
ONNX graph via the `onnx` package's graph-builder API), never touches the
network, and never bundles a real multi-hundred-MB CLIP checkpoint. Mirrors
test_embeddings.py's/test_dedup.py's structure for the image modality.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

import dscraft.clean.image_dedup as image_dedup_module
from dscraft.clean import (
    MODEL_ALLOWLIST,
    RECOMMENDED_IMAGE_MODEL_NAME,
    ImageEmbeddingModel,
    ModelIntegrityError,
    build_synthetic_image_embedding_model,
    build_synthetic_image_embedding_onnx,
    detect_near_duplicate_images,
    download_recommended_clip_vision_model,
    resize_and_normalize,
)
from dscraft.clean.dedup import DedupReport
from dscraft.core.licensing import ModelTier, RestrictedLicenseNotAcceptedError


def _solid_color_image(color: tuple[int, int, int], size: int = 32) -> np.ndarray:
    """A tiny (size, size, 3) uint8 image filled with a single RGB color."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def test_no_pytorch_or_transformers_imported():
    """Same hard constraint as embeddings.py: dscraft.clean never imports
    torch/transformers, statically -- not just at the call sites this test
    file happens to exercise."""
    forbidden_import_re = re.compile(r"^\s*(import|from)\s+(torch|transformers)\b", re.MULTILINE)
    package_dir = Path(image_dedup_module.__file__).parent
    offending: list[str] = []
    for source_file in package_dir.rglob("*.py"):
        text = source_file.read_text(encoding="utf-8")
        if forbidden_import_re.search(text):
            offending.append(str(source_file))
    assert offending == [], (
        f"Found forbidden torch/transformers import(s) in: {offending!r}. "
        "dscraft.clean is deliberately PyTorch-free."
    )


def test_recommended_image_model_is_registered_tier_2():
    """The recommended CLIP vision checkpoint's licensing evidence is not
    solid enough for Tier 1: OpenAI's own Hugging Face model card carries
    no explicit SPDX license tag, so the MIT classification rests only on
    inheritance from the openai/CLIP *code* repo's license, not a
    first-party declaration on the weights themselves. Per CLAUDE.md's
    LazyIsolate policy this must be Tier 2 (opt-in-gated), not auto-usable
    Tier 1 -- see docs/decisions/2026-07-image-dedup-evaluation.md."""
    with pytest.raises(RestrictedLicenseNotAcceptedError):
        MODEL_ALLOWLIST.check(RECOMMENDED_IMAGE_MODEL_NAME)

    entry = MODEL_ALLOWLIST.check(RECOMMENDED_IMAGE_MODEL_NAME, accept_restricted_licenses=True)
    assert entry.tier is ModelTier.TIER_2


def test_download_recommended_clip_vision_model_requires_explicit_opt_in(tmp_path):
    """download_recommended_clip_vision_model() must refuse to proceed (and
    must not touch the network) unless the caller explicitly passes
    accept_restricted_licenses=True -- same Tier 2 gate as
    MODEL_ALLOWLIST.check() itself, enforced before any download attempt."""
    with pytest.raises(RestrictedLicenseNotAcceptedError):
        download_recommended_clip_vision_model(cache_dir=tmp_path)
    # No file should have been written to the cache dir.
    assert list(tmp_path.iterdir()) == []


def test_download_recommended_clip_vision_model_rejects_hash_mismatch(tmp_path, monkeypatch):
    """A freshly 'downloaded' file that fails SHA-256 verification against
    the pinned expected digest must be rejected: never promoted to the
    cache path, and the temp file cleaned up."""
    bad_bytes = b"not the real onnx checkpoint"

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(bad_bytes)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(ModelIntegrityError):
        download_recommended_clip_vision_model(cache_dir=tmp_path, accept_restricted_licenses=True)

    remaining = list(tmp_path.iterdir())
    assert remaining == [], f"Expected no files left in cache dir, found: {remaining!r}"


def test_download_recommended_clip_vision_model_revalidates_stale_cache(tmp_path, monkeypatch):
    """A previously-cached file that no longer matches the pinned digest
    (corrupted, tampered, or stale) must be revalidated -- not blindly
    reused -- and transparently replaced by a fresh, verified download."""
    dest = tmp_path / "clip-vit-base-patch32-vision_model_int8.onnx"
    dest.write_bytes(b"stale or corrupted cached content")

    good_bytes = b"pretend this is the real onnx checkpoint bytes"
    good_digest = hashlib.sha256(good_bytes).hexdigest()

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(good_bytes)

    monkeypatch.setattr(image_dedup_module, "_RECOMMENDED_IMAGE_MODEL_ONNX_SHA256", good_digest)
    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = download_recommended_clip_vision_model(cache_dir=tmp_path, accept_restricted_licenses=True)

    assert result == dest
    assert dest.read_bytes() == good_bytes


# ---------------------------------------------------------------------------
# resize_and_normalize
# ---------------------------------------------------------------------------


def test_resize_and_normalize_output_shape_and_range():
    image = _solid_color_image((255, 0, 0), size=16)
    vector = resize_and_normalize(image, size=4)
    assert vector.shape == (4 * 4 * 3,)
    assert vector.dtype == np.float32
    assert vector.min() >= 0.0
    assert vector.max() <= 1.0


def test_resize_and_normalize_accepts_grayscale_input():
    grayscale = np.full((16, 16), 128, dtype=np.uint8)
    vector = resize_and_normalize(grayscale, size=4)
    assert vector.shape == (4 * 4 * 3,)


def test_resize_and_normalize_rejects_bad_shapes():
    with pytest.raises(ValueError):
        resize_and_normalize(np.zeros((4, 4, 4), dtype=np.uint8))  # wrong channel count
    with pytest.raises(ValueError):
        resize_and_normalize(np.zeros((4, 4, 4, 3), dtype=np.uint8))  # too many dims


def test_resize_and_normalize_scales_uint16_by_full_scale_not_255():
    """A uint16 image's full-scale value is 65535, not 255 -- dividing by a
    fixed 255 would produce wildly out-of-[0,1]-range output for any
    non-trivial uint16 pixel value."""
    image = np.full((16, 16, 3), 65535, dtype=np.uint16)
    vector = resize_and_normalize(image, size=4)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(vector, 1.0, atol=1e-6)

    half_scale = np.full((16, 16, 3), 32768, dtype=np.uint16)
    half_vector = resize_and_normalize(half_scale, size=4)
    np.testing.assert_allclose(half_vector, 32768 / 65535, atol=1e-6)


def test_resize_and_normalize_rejects_empty_spatial_dimensions():
    """A zero-height or zero-width image array must raise a clear
    ValueError up front, not misbehave (or raise an unclear error) deep
    inside the resize/index logic."""
    with pytest.raises(ValueError, match=r"non-zero height and width"):
        resize_and_normalize(np.zeros((0, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"non-zero height and width"):
        resize_and_normalize(np.zeros((4, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"non-zero height and width"):
        resize_and_normalize(np.zeros((0, 0, 3), dtype=np.uint8))


def test_resize_and_normalize_rejects_non_positive_size():
    """A zero or negative resize-target ``size`` must raise a clear
    ValueError up front, not silently misbehave (or raise an unclear
    error) deep inside the row_idx/col_idx index computation."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"positive resize target size"):
        resize_and_normalize(image, size=0)
    with pytest.raises(ValueError, match=r"positive resize target size"):
        resize_and_normalize(image, size=-1)


def test_resize_and_normalize_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match=r"Unsupported image dtype"):
        resize_and_normalize(np.zeros((4, 4, 3), dtype=np.int32))


def test_resize_and_normalize_rejects_out_of_range_float():
    with pytest.raises(ValueError):
        resize_and_normalize(np.full((4, 4, 3), 300.0, dtype=np.float32))
    with pytest.raises(ValueError):
        resize_and_normalize(np.full((4, 4, 3), -1.0, dtype=np.float32))


def test_resize_and_normalize_accepts_float_already_in_unit_range():
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)
    vector = resize_and_normalize(image, size=4)
    np.testing.assert_allclose(vector, 0.5, atol=1e-6)


def test_resize_and_normalize_solid_color_is_uniform():
    """A solid-color image resized to any smaller grid should still be a
    uniform vector -- every pixel sampled is the same color."""
    image = _solid_color_image((10, 20, 30), size=32)
    vector = resize_and_normalize(image, size=4).reshape(4, 4, 3)
    expected = np.array([10, 20, 30], dtype=np.float32) / 255.0
    np.testing.assert_allclose(vector[0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(vector[-1, -1], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Synthetic ONNX graph / ImageEmbeddingModel
# ---------------------------------------------------------------------------


def test_build_synthetic_image_embedding_onnx_writes_a_valid_model(tmp_path):
    onnx_path = tmp_path / "tiny_image.onnx"
    result = build_synthetic_image_embedding_onnx(onnx_path, feature_dim=48, embedding_dim=16, seed=1)

    assert result == onnx_path
    assert onnx_path.exists()
    assert onnx_path.stat().st_size < 100_000


def test_build_synthetic_image_embedding_model_returns_working_model(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    assert isinstance(model, ImageEmbeddingModel)
    assert model.embedding_dim == 16

    images = [_solid_color_image((255, 0, 0)), _solid_color_image((0, 255, 0))]
    embeddings = model.embed(images)
    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32


def test_image_embed_output_is_l2_normalized(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    embeddings = model.embed([_solid_color_image((10, 20, 30)), _solid_color_image((200, 5, 90))])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_image_embed_is_deterministic_across_calls(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    image = [_solid_color_image((77, 88, 99))]
    first = model.embed(image)
    second = model.embed(image)
    np.testing.assert_array_equal(first, second)


def test_image_embed_empty_input_returns_empty_array(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=4, embedding_dim=16)
    embeddings = model.embed([])
    assert embeddings.shape == (0, 16)


# ---------------------------------------------------------------------------
# detect_near_duplicate_images
# ---------------------------------------------------------------------------


def test_detect_near_duplicate_images_flags_identical_and_near_identical_colors(tmp_path):
    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=8, embedding_dim=32)

    red = _solid_color_image((200, 10, 10))
    near_red = _solid_color_image((202, 12, 9))  # visually near-identical to `red`
    blue = _solid_color_image((10, 10, 200))  # clearly distinct

    embeddings, report = detect_near_duplicate_images(
        [red, near_red, blue], model, threshold=0.999
    )

    assert isinstance(report, DedupReport)
    assert embeddings.shape == (3, 32)
    flagged = report.flagged_indices()
    assert {0, 1}.issubset(flagged)
    assert 2 not in flagged


def test_detect_near_duplicate_images_reuses_dedup_find_near_duplicates(tmp_path):
    """detect_near_duplicate_images must not reimplement near-duplicate
    scanning -- it should produce the exact same report dedup.py's
    find_near_duplicates would produce for the same embeddings."""
    from dscraft.clean.dedup import find_near_duplicates

    model = build_synthetic_image_embedding_model(cache_dir=tmp_path, size=8, embedding_dim=32)
    images = [_solid_color_image((5, 5, 5)), _solid_color_image((250, 250, 250))]

    embeddings, report = detect_near_duplicate_images(images, model, threshold=0.5)
    expected_report = find_near_duplicates(model.embed(images), threshold=0.5)

    # Full-object equality, not just a subset of fields -- covers pair
    # indices (index_a/index_b, not just similarity) and
    # zero_vector_row_indices too, so a future regression that reimplements
    # (and subtly diverges from) find_near_duplicates cannot slip through
    # by only breaking a field this test doesn't check.
    assert report == expected_report
