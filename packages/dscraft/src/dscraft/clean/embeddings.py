"""ONNX Runtime text embedding generation (architecture doc Part 3, "Module 2: LazyClean").

The one hard constraint this module exists to enforce: embeddings are
produced by loading a ``.onnx`` model file directly via the ``onnxruntime``
Python package and running our own lightweight, tokenizer-adjacent
preprocessing in plain Python/NumPy -- never via PyTorch or the HuggingFace
``transformers`` package. That keeps this package's runtime dependency
footprint under the architecture doc's ~100MB target and avoids pulling in
the PyTorch/HuggingFace stack, which is LazyClean's specific differentiator
per Part 3 and Appendix A ("LazyClean"). Do not import ``torch`` or
``transformers`` anywhere in this package, including for type hints.

Two embedding-model sources are provided:

1. :func:`build_synthetic_embedding_model` -- hand-builds a tiny ONNX graph
   on the fly via the ``onnx`` package's graph-builder API (a linear
   projection + L2 normalization over a hashed bag-of-words feature vector).
   This is fully hermetic (no network access, no bundled multi-hundred-MB
   model file) and is what the test suite and ``examples/dedup_example.py``
   use. It is a stand-in for a real sentence-embedding model, good enough to
   demonstrate the embed -> dedup pipeline end-to-end, and is **not**
   intended to produce semantically meaningful embeddings for production use.
2. :func:`download_recommended_model` -- documents (and, given network
   access, performs) the production wiring: fetching the Tier-1-allowlisted
   ONNX sentence-embedding checkpoint referenced in :data:`MODEL_ALLOWLIST`
   and caching it locally. This path is optional and lazy -- it is never
   called by tests or the example, and nothing in this package requires
   network access to import or to run its test suite.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from dscraft.core.licensing import Allowlist, ModelTier

__all__ = [
    "MODEL_ALLOWLIST",
    "EmbeddingModel",
    "hashing_bag_of_words_vectorizer",
    "build_synthetic_embedding_onnx",
    "build_synthetic_embedding_model",
    "download_recommended_model",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: A preprocessor may return either a single ``(feature_dim,)`` array fed to
#: the model's one input (:data:`EmbeddingModel.input_name` -- what the
#: synthetic single-input fixture and ``hashing_bag_of_words_vectorizer``
#: use), or a mapping of ONNX input name -> ``(feature_dim,)`` array for a
#: real multi-input sentence-transformer checkpoint (e.g.
#: ``{"input_ids": ..., "attention_mask": ..., "token_type_ids": ...}``).
#: See :meth:`EmbeddingModel.embed`.
PreprocessorOutput = Union[np.ndarray, Mapping[str, np.ndarray]]

# ---------------------------------------------------------------------------
# Model licensing allowlist (architecture doc §2.10) -- LazyClean's own
# instance, per dscraft.core.licensing's documented per-module ownership
# pattern.
# ---------------------------------------------------------------------------

#: LazyClean's model-checkpoint allowlist (architecture doc §2.10). Starts
#: empty per dscraft.core's contract; LazyClean populates it with the one
#: production embedding checkpoint this module documents wiring for. This
#: is a *recommendation*, not a bundled artifact -- see
#: :func:`download_recommended_model`.
MODEL_ALLOWLIST = Allowlist()

RECOMMENDED_MODEL_NAME = "Xenova/all-MiniLM-L6-v2"

MODEL_ALLOWLIST.register(
    name=RECOMMENDED_MODEL_NAME,
    tier=ModelTier.TIER_1,
    license_identifier="Apache-2.0",
    notes=(
        "Recommended production checkpoint for dscraft.clean.embeddings.EmbeddingModel: "
        "an ONNX-exported sentence-transformer (384-dim mean-pooled embeddings, "
        "~90MB fp32 / ~23MB int8-quantized -- well under this module's <100MB "
        "ONNX Runtime footprint target). Apache-2.0 licensed, auto-usable "
        "(Tier 1), no opt-in gate required. Not bundled with this package and "
        "not downloaded by default -- see download_recommended_model() and the "
        "README's 'Wiring in a real production model' section for how to point "
        "EmbeddingModel.from_onnx_file at a local copy of its model.onnx, paired "
        "with a real WordPiece tokenizer (e.g. via the standalone `tokenizers` "
        "library) and mean-pooling in place of this module's default hashing "
        "bag-of-words preprocessor. Loading it never requires torch or "
        "transformers -- only onnxruntime plus a tokenizer/vocab file."
    ),
)

# A specific, pinned commit of the recommended checkpoint's HF repo -- NOT
# "main". "main" is a mutable branch ref: the repo owner can force-push a
# different model (different weights, a different license, or a broken
# export) to it at any time, silently changing what a previously-verified
# "Xenova/all-MiniLM-L6-v2" download resolves to. Pinning to an immutable
# commit SHA means download_recommended_model() always fetches the exact,
# previously-reviewed artifact this module's Tier-1/Apache-2.0
# MODEL_ALLOWLIST entry above was actually reviewed against. Re-verify and
# bump this SHA deliberately (a human decision, not automatic) if the
# upstream repo needs to be updated.
_RECOMMENDED_MODEL_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"

# A direct HTTP source for the recommended checkpoint's ONNX export and its
# tokenizer config. Referenced only by the optional, lazy download path in
# download_recommended_model() -- never touched by import or by tests.
_RECOMMENDED_MODEL_ONNX_URL = (
    f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/"
    f"{_RECOMMENDED_MODEL_REVISION}/onnx/model.onnx"
)


# ---------------------------------------------------------------------------
# Preprocessing ("tokenization-adjacent" -- runs in plain Python/NumPy, not
# inside the ONNX graph, and never touches PyTorch/transformers).
# ---------------------------------------------------------------------------


def hashing_bag_of_words_vectorizer(vocab_dim: int = 128) -> Callable[[str], np.ndarray]:
    """Return a text -> fixed-length float32 vector feature-hashing function.

    This is the default preprocessor for the synthetic test/example model.
    It lower-cases and word-tokenizes with a simple regex, hashes each token
    into one of ``vocab_dim`` buckets via SHA-256 (stable across Python
    processes, unlike the builtin ``hash()``), accumulates token counts per
    bucket, and normalizes by token count so sentence length doesn't dominate
    the resulting vector's magnitude. It has no PyTorch/transformers
    dependency and no learned vocabulary -- it is intentionally simple,
    matching the ~O(n^2)-for-now scaffold depth of this package (see
    dedup.py). A real production model (see README) would instead use a
    proper subword tokenizer (e.g. the standalone `tokenizers` library) and
    feed token IDs/attention masks into a transformer ONNX graph.
    """

    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _vectorize(text: str) -> np.ndarray:
        vector = np.zeros(vocab_dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % vocab_dim
            vector[bucket] += 1.0
        vector /= float(len(tokens))
        return vector

    return _vectorize


# ---------------------------------------------------------------------------
# Synthetic ONNX graph builder (hermetic test/example fixture)
# ---------------------------------------------------------------------------


def build_synthetic_embedding_onnx(
    path: str | Path,
    *,
    vocab_dim: int = 128,
    embedding_dim: int = 32,
    seed: int = 0,
) -> Path:
    """Hand-build a tiny ONNX graph and save it to ``path``.

    This is a **test/example fixture, not a real embedding model**. It is
    built directly via the ``onnx`` package's graph-builder API
    (``onnx.helper``/``onnx.numpy_helper``) so that tests and the example in
    this package are fully self-contained: no network access is required,
    and no multi-hundred-MB model file needs to be checked into the repo.

    Graph shape: ``embedding = L2Normalize(input @ weight + bias)``, where
    ``weight``/``bias`` are fixed (seeded) random initializers baked into
    the graph. This is enough to demonstrate the embed -> cosine-similarity
    dedup pipeline end-to-end -- near-identical input text produces
    near-identical hashed feature vectors (see
    :func:`hashing_bag_of_words_vectorizer`), and a linear map (even a
    random one) preserves that similarity closely enough for near-duplicate
    detection to work in tests. It is not a semantically meaningful sentence
    embedding and must not be used for anything beyond tests/examples.

    Returns the resolved ``Path`` the model was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    weight = rng.normal(scale=1.0 / np.sqrt(vocab_dim), size=(vocab_dim, embedding_dim)).astype(
        np.float32
    )
    bias = np.zeros((embedding_dim,), dtype=np.float32)
    eps = np.array([1e-12], dtype=np.float32)

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, vocab_dim])
    output_info = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [None, embedding_dim]
    )

    weight_init = numpy_helper.from_array(weight, name="weight")
    bias_init = numpy_helper.from_array(bias, name="bias")
    eps_init = numpy_helper.from_array(eps, name="eps")

    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["linear_raw"], name="matmul"),
        helper.make_node("Add", ["linear_raw", "bias"], ["linear_out"], name="add_bias"),
        helper.make_node("Mul", ["linear_out", "linear_out"], ["squared"], name="square"),
        helper.make_node(
            "ReduceSum",
            ["squared"],
            ["sum_squared"],
            name="reduce_sum",
            axes=[1],
            keepdims=1,
        ),
        helper.make_node("Sqrt", ["sum_squared"], ["norm"], name="sqrt"),
        helper.make_node("Add", ["norm", "eps"], ["norm_eps"], name="add_eps"),
        helper.make_node("Div", ["linear_out", "norm_eps"], ["embedding"], name="l2_normalize"),
    ]

    graph = helper.make_graph(
        nodes,
        "lazyclean_synthetic_embedding",
        [input_info],
        [output_info],
        initializer=[weight_init, bias_init, eps_init],
    )
    # Opset 11: ReduceSum's `axes` is still an attribute here (it moved to
    # an optional second input starting at opset 13), which keeps this
    # graph-construction code simple. All other ops used below (MatMul,
    # Add, Mul, Sqrt, Div) are stable well before and after opset 11.
    model = helper.make_model(
        graph,
        producer_name="dscraft-clean",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# EmbeddingModel -- the one canonical embedding path
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingModel:
    """Wraps an ``onnxruntime.InferenceSession`` plus a text preprocessor.

    This is the single canonical way this package turns text into
    embeddings, whether the underlying ``.onnx`` graph is the synthetic
    test fixture from :func:`build_synthetic_embedding_onnx` (single input,
    single output -- e.g. ``hashing_bag_of_words_vectorizer``) or a real
    production sentence-embedding checkpoint (see README), which typically
    has *multiple* named inputs (``input_ids``/``attention_mask``/
    ``token_type_ids``). There is deliberately no second/parallel embedding
    code path -- both shapes go through :attr:`preprocessor` and
    :meth:`embed` below; see :data:`PreprocessorOutput` for how a
    preprocessor selects between them.
    """

    session: ort.InferenceSession
    input_name: str
    output_name: str
    preprocessor: Callable[[str], PreprocessorOutput]
    embedding_dim: int

    @classmethod
    def from_onnx_file(
        cls,
        model_path: str | Path,
        *,
        preprocessor: Callable[[str], PreprocessorOutput],
        embedding_dim: int,
        input_name: str | None = None,
        output_name: str | None = None,
        providers: Sequence[str] | None = None,
    ) -> "EmbeddingModel":
        """Load a ``.onnx`` embedding model via ``onnxruntime`` directly.

        No PyTorch, no `transformers` -- ``onnxruntime.InferenceSession`` is
        the only inference runtime this package ever touches.

        ``input_name`` (like ``session.get_inputs()[0].name`` inferred by
        default) only matters for a **single-input** graph, i.e. when
        ``preprocessor`` returns a plain array per row -- see
        :meth:`embed`. A real multi-input sentence-transformer ONNX
        checkpoint has more than one required input
        (``input_ids``/``attention_mask``/``token_type_ids``, typically);
        for that case, write ``preprocessor`` to return a
        ``{input_name: array}`` mapping instead, and this inferred/passed
        ``input_name`` is simply unused (the mapping's own keys are fed to
        the session directly). This function does not validate that a
        single inferred input name covers every input the graph actually
        requires -- if you load a multi-input model with a preprocessor
        that returns a single array, ``onnxruntime`` will raise a missing-
        input error at ``embed()`` time, not here.
        """
        session = ort.InferenceSession(
            str(model_path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        resolved_input = input_name or session.get_inputs()[0].name
        resolved_output = output_name or session.get_outputs()[0].name
        return cls(
            session=session,
            input_name=resolved_input,
            output_name=resolved_output,
            preprocessor=preprocessor,
            embedding_dim=embedding_dim,
        )

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch of text rows, returning a ``(n, embedding_dim)`` float32 array.

        ``preprocessor`` may return, per row, either:

        - a plain ``(feature_dim,)`` array -- fed as the single named input
          ``self.input_name`` (the synthetic fixture / hashing-vectorizer
          shape), or
        - a ``{input_name: (feature_dim,) array}`` mapping -- each named
          input is stacked across the batch and fed to the session under
          its own name (the real multi-input sentence-transformer shape,
          e.g. ``input_ids``/``attention_mask``/``token_type_ids``).

        Mixing the two shapes across rows in the same call is not
        supported; ``preprocessor``'s return type is assumed to be
        consistent for a given :class:`EmbeddingModel`.
        """
        rows = list(texts)
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        raw_features = [self.preprocessor(text) for text in rows]
        if isinstance(raw_features[0], Mapping):
            # Multi-input ONNX graph: each preprocessed row is a dict of
            # named arrays (e.g. input_ids/attention_mask/token_type_ids)
            # rather than a single array for self.input_name. Stack each
            # named input across the batch and feed the session all of them
            # at once, by name -- not just self.input_name.
            input_names = raw_features[0].keys()
            feed = {
                name: np.stack([row[name] for row in raw_features]) for name in input_names  # type: ignore[index]
            }
            (output,) = self.session.run([self.output_name], feed)
        else:
            features = np.stack(raw_features).astype(np.float32)
            (output,) = self.session.run([self.output_name], {self.input_name: features})
        return np.asarray(output, dtype=np.float32)


def build_synthetic_embedding_model(
    *,
    cache_dir: str | Path | None = None,
    vocab_dim: int = 128,
    embedding_dim: int = 32,
    seed: int = 0,
) -> EmbeddingModel:
    """Build (or reuse a cached) synthetic ONNX embedding model, ready to use.

    Convenience wrapper around :func:`build_synthetic_embedding_onnx` +
    :func:`hashing_bag_of_words_vectorizer` + :meth:`EmbeddingModel.from_onnx_file`
    so tests and the example call one function instead of duplicating this
    three-step setup. Fully hermetic: no network access, writes a small
    (a few KB) ``.onnx`` file to a temp/cache directory.
    """
    cache_dir_path = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
    onnx_path = cache_dir_path / f"lazyclean_synthetic_v{vocab_dim}x{embedding_dim}_seed{seed}.onnx"
    if not onnx_path.exists():
        build_synthetic_embedding_onnx(
            onnx_path, vocab_dim=vocab_dim, embedding_dim=embedding_dim, seed=seed
        )
    preprocessor = hashing_bag_of_words_vectorizer(vocab_dim=vocab_dim)
    return EmbeddingModel.from_onnx_file(
        onnx_path, preprocessor=preprocessor, embedding_dim=embedding_dim
    )


def download_recommended_model(
    *,
    cache_dir: str | Path | None = None,
    accept_restricted_licenses: bool = False,
) -> Path:
    """Lazily download the Tier-1 recommended production checkpoint (optional).

    This is the documented production wiring path -- **never called by
    tests, the example, or any import-time code in this package.** It
    requires network access, which is deliberately not a hard requirement
    for installing or testing this package (per CLAUDE.md's local-only
    constraint: any model-download path must be optional/lazy).

    Looks the checkpoint up in :data:`MODEL_ALLOWLIST` first (demonstrating
    the shared dscraft.core.licensing.Allowlist usage pattern) before touching
    the network -- this call would raise ``RestrictedLicenseNotAcceptedError``
    for a Tier 2 model without the opt-in flag, though the currently
    registered recommended model is Tier 1 and does not require it.

    After downloading, wire the result into :meth:`EmbeddingModel.from_onnx_file`
    together with a real tokenizer (see README) -- this function only
    fetches and caches the ``.onnx`` graph itself.

    The download is written to a temporary file in ``cache_dir`` first and
    atomically renamed into place only once it completes successfully. If
    the download is interrupted (network error, process kill, etc.) the
    temporary file is removed and ``dest`` is left untouched, so a later
    call never mistakes a truncated/corrupt partial download for a valid
    cached model just because a file already exists at ``dest``. The
    fetched URL is pinned to a specific immutable commit of the upstream
    repo (see ``_RECOMMENDED_MODEL_REVISION``), not a mutable branch ref,
    so repeated calls (and calls across machines) always fetch the exact
    same, previously-reviewed artifact.
    """
    MODEL_ALLOWLIST.check(
        RECOMMENDED_MODEL_NAME, accept_restricted_licenses=accept_restricted_licenses
    )
    cache_dir_path = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "dscraft" / "clean"
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    dest = cache_dir_path / "all-MiniLM-L6-v2.onnx"
    if dest.exists():
        return dest

    import urllib.request

    fd, tmp_name = tempfile.mkstemp(dir=cache_dir_path, suffix=".onnx.part")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        urllib.request.urlretrieve(_RECOMMENDED_MODEL_ONNX_URL, tmp_path)  # noqa: S310
        tmp_path.replace(dest)  # atomic on the same filesystem
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest
