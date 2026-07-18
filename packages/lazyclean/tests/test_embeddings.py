"""Tests for the PyTorch-free ONNX Runtime embedding path.

Fully hermetic: uses build_synthetic_embedding_model() (a hand-built ONNX
graph via the `onnx` package's graph-builder API), never touches the
network, and never bundles a real multi-hundred-MB model file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import benchcraft_lazyclean.embeddings as embeddings_module
from benchcraft_lazyclean import MODEL_ALLOWLIST, RECOMMENDED_MODEL_NAME
from benchcraft_lazyclean.embeddings import (
    EmbeddingModel,
    build_synthetic_embedding_model,
    build_synthetic_embedding_onnx,
    download_recommended_model,
    hashing_bag_of_words_vectorizer,
)
from lazycore.licensing import ModelTier


def test_no_pytorch_or_transformers_imported():
    """The hard constraint: this package must never pull in torch/transformers.

    Checking ``sys.modules`` alone is a weak proxy: it only catches a
    top-level ``import torch``/``import transformers`` that has *already*
    executed by the time this test runs (e.g. a module-level import in
    ``embeddings.py``). It would silently miss a deferred import stashed
    inside a rarely-called function that this test suite happens not to
    exercise. Back it with a static source scan of this package's own code
    as a second, independent guarantee: no ``.py`` file under
    ``benchcraft_lazyclean`` may contain an ``import torch``/``from torch``
    or ``import transformers``/``from transformers`` statement at all,
    called or not.
    """
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules

    forbidden_import_re = re.compile(r"^\s*(import|from)\s+(torch|transformers)\b", re.MULTILINE)
    package_dir = Path(embeddings_module.__file__).parent
    offending: list[str] = []
    for source_file in package_dir.rglob("*.py"):
        text = source_file.read_text(encoding="utf-8")
        if forbidden_import_re.search(text):
            offending.append(str(source_file))
    assert offending == [], (
        f"Found forbidden torch/transformers import(s) in: {offending!r}. "
        "benchcraft_lazyclean is deliberately PyTorch-free -- see the "
        "package README and embeddings.py's module docstring."
    )


def test_build_synthetic_embedding_onnx_writes_a_valid_model(tmp_path):
    """build_synthetic_embedding_onnx() writes a valid, small (<100KB) .onnx
    file to the requested path."""
    onnx_path = tmp_path / "tiny.onnx"
    result = build_synthetic_embedding_onnx(onnx_path, vocab_dim=64, embedding_dim=16, seed=1)

    assert result == onnx_path
    assert onnx_path.exists()
    # A hand-built graph fixture, not a bundled real model -- expect it to
    # be tiny (a few KB), not multi-hundred-MB.
    assert onnx_path.stat().st_size < 100_000


def test_build_synthetic_embedding_model_returns_working_embedding_model(tmp_path):
    """build_synthetic_embedding_model() returns a ready-to-use EmbeddingModel
    that embeds text rows into float32 vectors of the requested dimension."""
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)

    assert isinstance(model, EmbeddingModel)
    assert model.embedding_dim == 16

    embeddings = model.embed(["hello world", "goodbye world"])
    assert embeddings.shape == (2, 16)
    assert embeddings.dtype == np.float32


def test_embed_output_is_l2_normalized(tmp_path):
    """The synthetic model's ONNX graph L2-normalizes its output, so every
    embedded row has unit norm."""
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    embeddings = model.embed(["some sample text", "another distinct sample"])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_embed_is_deterministic_across_calls(tmp_path):
    """Embedding the same text twice through the same model produces
    identical output arrays (no hidden randomness at inference time)."""
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    first = model.embed(["repeat this exact sentence"])
    second = model.embed(["repeat this exact sentence"])
    np.testing.assert_array_equal(first, second)


def test_embed_empty_input_returns_empty_array(tmp_path):
    """Embedding an empty list of texts returns a (0, embedding_dim) array
    rather than raising or running the ONNX session."""
    model = build_synthetic_embedding_model(cache_dir=tmp_path, vocab_dim=64, embedding_dim=16)
    embeddings = model.embed([])
    assert embeddings.shape == (0, 16)


def test_hashing_bag_of_words_vectorizer_is_deterministic_and_shaped():
    """The hashing vectorizer produces a fixed-length vector and is
    case-insensitive, so casing differences alone don't change the result."""
    vectorize = hashing_bag_of_words_vectorizer(vocab_dim=32)
    vec_a = vectorize("The Quick Brown Fox")
    vec_b = vectorize("the quick brown fox")  # case-insensitive tokenization
    assert vec_a.shape == (32,)
    np.testing.assert_array_equal(vec_a, vec_b)


def test_hashing_bag_of_words_vectorizer_empty_string_is_zero_vector():
    """Whitespace-only text tokenizes to zero tokens, so the vectorizer
    returns an all-zero vector rather than dividing by zero."""
    vectorize = hashing_bag_of_words_vectorizer(vocab_dim=16)
    vec = vectorize("   ")
    np.testing.assert_array_equal(vec, np.zeros(16, dtype=np.float32))


def test_recommended_model_registered_as_tier_1():
    """The recommended production checkpoint is registered in MODEL_ALLOWLIST
    as Tier 1 / Apache-2.0, so it is auto-usable without an opt-in flag."""
    entry = MODEL_ALLOWLIST.get(RECOMMENDED_MODEL_NAME)
    assert entry is not None
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_from_onnx_file_infers_input_output_names(tmp_path):
    """EmbeddingModel.from_onnx_file() infers the input/output tensor names
    from the ONNX graph itself when they aren't passed explicitly."""
    onnx_path = build_synthetic_embedding_onnx(
        tmp_path / "infer_names.onnx", vocab_dim=32, embedding_dim=8
    )
    preprocessor = hashing_bag_of_words_vectorizer(vocab_dim=32)
    model = EmbeddingModel.from_onnx_file(
        onnx_path, preprocessor=preprocessor, embedding_dim=8
    )
    assert model.input_name == "input"
    assert model.output_name == "embedding"


def test_embed_supports_multi_input_preprocessor_dict(tmp_path):
    """EmbeddingModel.embed() must handle a real multi-input ONNX embedding
    model, not just the synthetic single-input fixture's I/O shape.

    ``from_onnx_file`` infers a single ``input_name`` from
    ``session.get_inputs()[0]``, and the synthetic fixture / hashing
    preprocessor only ever exercises that one-input-one-output contract.
    A real sentence-transformer ONNX export (e.g. the recommended
    Xenova/all-MiniLM-L6-v2 checkpoint) instead has multiple required
    named inputs (``input_ids``/``attention_mask``/``token_type_ids``).
    This test builds a small two-input ONNX graph directly (mirroring that
    real shape, without needing network access) and verifies a
    dict-returning preprocessor is routed to the right named inputs rather
    than only ever feeding ``self.input_name``.
    """
    dim = 4
    input_ids_info = helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [None, dim])
    attention_mask_info = helper.make_tensor_value_info(
        "attention_mask", TensorProto.FLOAT, [None, dim]
    )
    output_info = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, dim])
    node = helper.make_node(
        "Add", ["input_ids", "attention_mask"], ["embedding"], name="add_two_inputs"
    )
    graph = helper.make_graph(
        [node],
        "two_input_test_fixture",
        [input_ids_info, attention_mask_info],
        [output_info],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx_path = tmp_path / "two_input.onnx"
    onnx.save(model, str(onnx_path))

    def dict_preprocessor(text: str) -> dict[str, np.ndarray]:
        # Deterministic per-text arrays (based on text length) so the
        # expected output of the Add node is easy to check below.
        input_ids = np.full(dim, float(len(text)), dtype=np.float32)
        attention_mask = np.ones(dim, dtype=np.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    embedding_model = EmbeddingModel.from_onnx_file(
        onnx_path,
        preprocessor=dict_preprocessor,
        embedding_dim=dim,
        output_name="embedding",
    )
    result = embedding_model.embed(["ab", "abcd"])

    assert result.shape == (2, dim)
    np.testing.assert_allclose(result[0], np.full(dim, 2.0 + 1.0), atol=1e-5)
    np.testing.assert_allclose(result[1], np.full(dim, 4.0 + 1.0), atol=1e-5)


def test_recommended_model_url_pins_a_commit_revision_not_a_mutable_branch():
    """The recommended checkpoint's download URL must be pinned to a fixed,
    immutable commit revision, not a mutable branch ref like ``main``.

    ``main`` can be force-pushed to a different model (different weights,
    license, or a broken export) at any time by the upstream repo owner,
    silently changing what a previously-reviewed
    ``Xenova/all-MiniLM-L6-v2`` download resolves to for every caller.
    """
    url = embeddings_module._RECOMMENDED_MODEL_ONNX_URL
    assert "/resolve/main/" not in url
    assert embeddings_module._RECOMMENDED_MODEL_REVISION in url
    # A full 40-character git commit SHA, not e.g. a short 7-char prefix or
    # a mutable tag name.
    assert re.fullmatch(r"[0-9a-f]{40}", embeddings_module._RECOMMENDED_MODEL_REVISION)


def test_download_recommended_model_writes_atomically(tmp_path, monkeypatch):
    """A successful download lands at the final destination path with no
    leftover temporary/partial file in the cache directory."""

    def fake_urlretrieve(url: str, filename) -> None:
        Path(filename).write_bytes(b"fake onnx bytes")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    dest = download_recommended_model(cache_dir=tmp_path)

    assert dest.exists()
    assert dest.read_bytes() == b"fake onnx bytes"
    leftover_partials = list(tmp_path.glob("*.part"))
    assert leftover_partials == []


def test_download_recommended_model_cleans_up_partial_file_on_failure(tmp_path, monkeypatch):
    """If the download is interrupted partway through, no corrupt/truncated
    file is left behind at the final destination path, and no stray
    temporary file lingers in the cache directory either -- a later call
    must not mistake a partial download for a valid cached model just
    because *a* file exists at ``dest``."""

    def failing_urlretrieve(url: str, filename) -> None:
        # Simulate a network failure partway through -- onnxruntime's real
        # urlretrieve would also leave a truncated file at `filename` (the
        # temp path, not `dest`) in this scenario.
        Path(filename).write_bytes(b"only partial bytes")
        raise ConnectionError("simulated network failure mid-download")

    monkeypatch.setattr("urllib.request.urlretrieve", failing_urlretrieve)

    with pytest.raises(ConnectionError):
        download_recommended_model(cache_dir=tmp_path)

    dest = tmp_path / "all-MiniLM-L6-v2.onnx"
    assert not dest.exists()
    leftover_partials = list(tmp_path.glob("*.part"))
    assert leftover_partials == []
