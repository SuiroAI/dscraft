"""Adapter-Factory pattern for local LLM fine-tuning (architecture doc Part 3, "Module 6: LazyTune").

This module implements exactly one signature capability from LazyTune's
Adapter-Factory design: a minimal :class:`BaseTrainingAdapter` interface
plus one concrete, in-process :class:`ProgrammaticAdapter` implementation
(the architecture doc's "`ProgrammaticAdapter`s (Unsloth, TRL)" family --
in-process, as opposed to the subprocess-isolated `SubprocessAdapter`
family for torchtune/Axolotl via `torchrun`, which is explicitly out of
scope for this pass -- see README).

``ProgrammaticAdapter`` loads a small causal language model, wraps it with
`peft`'s LoRA config, and runs real forward + backward + optimizer-step
training on a tiny text dataset -- a genuine (if tiny) LoRA fine-tuning
step, not a mock. `torch`/`transformers`/`peft` are intentional, expected
dependencies of this package (unlike LazyClean, which is deliberately
PyTorch-free) -- see README "Scope" for why that constraint does not apply
here.

Two model-construction paths are provided, mirroring the pattern already
used by ``benchcraft_lazyclean.embeddings`` for the same underlying
problem (needing a "real-ish" model for hermetic tests without bundling
something huge or requiring network access at test time):

1. :func:`build_hermetic_causal_lm` -- constructs a tiny GPT-2-architecture
   model **from scratch** via ``transformers.GPT2Config`` +
   ``AutoModelForCausalLM.from_config`` (random weights, no download), paired
   with :class:`TinyTokenizer` (a hand-built, corpus-derived word-level
   tokenizer with zero external vocab/merges files). This is what the test
   suite and ``examples/lora_finetune_example.py`` use. Fully hermetic: no
   network access, no bundled checkpoint file, deterministic given a seed.
2. :data:`MODEL_ALLOWLIST` documents (but does not download or require) a
   real production base-model path -- see README "Wiring in a real base
   model" and the module docstring on :data:`MODEL_ALLOWLIST` below.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedModel

from lazycore.licensing import Allowlist, ModelTier

__all__ = [
    "MODEL_ALLOWLIST",
    "RECOMMENDED_BASE_MODEL_NAME",
    "TrainStepResult",
    "TinyTokenizer",
    "BaseTrainingAdapter",
    "ProgrammaticAdapter",
    "build_hermetic_causal_lm",
    "default_lora_config",
]

# ---------------------------------------------------------------------------
# Model licensing allowlist (architecture doc §2.10) -- LazyTune's own
# instance, per lazycore.licensing's documented per-module ownership pattern.
# ---------------------------------------------------------------------------

#: LazyTune's model-checkpoint allowlist (architecture doc §2.10). Starts
#: empty per lazycore's contract; LazyTune populates it with the one
#: documented real base-model checkpoint this module's ProgrammaticAdapter
#: is designed to also work against. This is a *recommendation* for real
#: usage, not a bundled/downloaded artifact -- the hermetic default used by
#: tests/examples is :func:`build_hermetic_causal_lm`, which needs no
#: allowlist entry at all because it never touches an external checkpoint.
MODEL_ALLOWLIST = Allowlist()

RECOMMENDED_BASE_MODEL_NAME = "openai-community/gpt2"

MODEL_ALLOWLIST.register(
    name=RECOMMENDED_BASE_MODEL_NAME,
    tier=ModelTier.TIER_1,
    license_identifier="MIT",
    notes=(
        "Documented real base-model path for ProgrammaticAdapter: OpenAI's "
        "original GPT-2 (124M, 'small') checkpoint, MIT-licensed, auto-usable "
        "(Tier 1). Not bundled with this package and not downloaded by "
        "default -- tests and examples use build_hermetic_causal_lm() "
        "instead (a from-scratch, randomly-initialized GPT-2-architecture "
        "model requiring no network access), which sidesteps needing any "
        "allowlist entry at all for the hermetic path. To fine-tune this "
        "real checkpoint instead: "
        "`AutoModelForCausalLM.from_pretrained('openai-community/gpt2')` "
        "plus `AutoTokenizer.from_pretrained('openai-community/gpt2')` "
        "(requires network access on first use; results are cached "
        "locally by `transformers` afterwards), then pass "
        "`(model, tokenizer)` as `model_ref` to ProgrammaticAdapter.prepare(). "
        "See README 'Wiring in a real base model'."
    ),
)


# ---------------------------------------------------------------------------
# Hermetic tokenizer + model construction (no network access required)
# ---------------------------------------------------------------------------


class TinyTokenizer:
    """A minimal, corpus-derived word-level tokenizer with zero external files.

    This exists for the same reason ``benchcraft_lazyclean``'s
    ``hashing_bag_of_words_vectorizer`` exists: real subword tokenizers
    (GPT-2's BPE vocab/merges files, a HuggingFace ``AutoTokenizer``) are
    either bundled files or a network fetch on first use, and hermetic
    tests/examples in this package must not require either. Instead,
    :meth:`build_from_corpus` derives a small fixed vocabulary directly from
    the tiny synthetic training corpus used in tests/examples, plus a
    `<pad>`/`<unk>` token. It is intentionally simple (whitespace + lowercase
    splitting, no subword merges) -- matching the scaffold depth of this
    package, not a production tokenizer.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab: dict[str, int]) -> None:
        """Wrap a pre-built vocabulary mapping token strings to integer ids.

        Args:
            vocab: token -> id mapping that must already contain both
                :data:`PAD_TOKEN` and :data:`UNK_TOKEN`. Prefer
                :meth:`build_from_corpus` over calling this directly unless
                you already have a vocabulary in hand.
        """
        self.vocab = vocab
        self.pad_id = vocab[self.PAD_TOKEN]
        self.unk_id = vocab[self.UNK_TOKEN]

    @classmethod
    def build_from_corpus(cls, texts: Sequence[str]) -> "TinyTokenizer":
        """Derive a fixed vocabulary from ``texts`` (whitespace-tokenized, lowercased)."""
        unique_tokens = sorted({token for text in texts for token in text.lower().split()})
        vocab: dict[str, int] = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 1}
        for token in unique_tokens:
            vocab.setdefault(token, len(vocab))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary (including pad/unk tokens)."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Lowercase + whitespace-split ``text`` into a list of vocabulary ids.

        Tokens not present in the vocabulary map to :attr:`unk_id`.
        """
        return [self.vocab.get(token, self.unk_id) for token in text.lower().split()]

    def batch_encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize + pad a batch of strings, returning ``input_ids``/``attention_mask``.

        Also returns ``labels`` with padded positions set to ``-100`` (the
        standard HuggingFace "ignore this position in the loss" sentinel),
        so padding never contributes to the training loss.
        """
        sequences = [self.encode(text) or [self.unk_id] for text in texts]
        max_len = max(len(seq) for seq in sequences)

        input_ids = torch.full((len(sequences), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        labels = torch.full((len(sequences), max_len), -100, dtype=torch.long)

        for row, seq in enumerate(sequences):
            length = len(seq)
            token_ids = torch.tensor(seq, dtype=torch.long)
            input_ids[row, :length] = token_ids
            attention_mask[row, :length] = 1
            labels[row, :length] = token_ids

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_hermetic_causal_lm(
    corpus: Sequence[str],
    *,
    n_embd: int = 32,
    n_layer: int = 2,
    n_head: int = 2,
    n_positions: int = 64,
    seed: int = 0,
) -> tuple[PreTrainedModel, TinyTokenizer]:
    """Build a tiny, randomly-initialized GPT-2-architecture causal LM from scratch.

    Uses ``transformers.GPT2Config`` + ``AutoModelForCausalLM.from_config`` --
    architecture instantiation only, **no checkpoint download, no network
    access, no bundled weights file.** This sidesteps the model-licensing
    question entirely for hermetic tests/examples (there is no external
    checkpoint to license-check at all) -- see README for the reasoning.

    Paired with a :class:`TinyTokenizer` derived from ``corpus`` via
    :meth:`TinyTokenizer.build_from_corpus`, so the returned model's
    vocabulary exactly matches what the tokenizer can produce.

    A few hundred KB to low single-digit MB in memory at these default
    dimensions -- appropriate for hermetic CPU testing, not a production
    model.
    """
    torch.manual_seed(seed)
    tokenizer = TinyTokenizer.build_from_corpus(corpus)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_ctx=n_positions,
        bos_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.pad_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model, tokenizer


def default_lora_config() -> LoraConfig:
    """The default LoRA config used by :class:`ProgrammaticAdapter`.

    Targets GPT-2's attention projection (``c_attn``, a ``Conv1D`` layer
    rather than ``nn.Linear`` -- hence ``fan_in_fan_out=True``, which is
    required for LoRA to wrap GPT-2-style ``Conv1D`` layers correctly).
    A tiny rank (``r=4``) is more than enough to demonstrate a real
    forward/backward/optimizer-step LoRA update on a model this small.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )


# ---------------------------------------------------------------------------
# Adapter-Factory interface (architecture doc Part 3, "Module 6: LazyTune")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainStepResult:
    """Result of one :meth:`BaseTrainingAdapter.train_step` call."""

    loss: float
    step: int


class BaseTrainingAdapter(abc.ABC):
    """Minimal Adapter-Factory training interface (architecture doc Part 3).

    The architecture doc describes an Adapter-Factory pattern unifying
    in-process ``ProgrammaticAdapter``s (Unsloth, TRL) and
    subprocess-isolated ``SubprocessAdapter``s (torchtune, Axolotl via
    `torchrun`) behind one ``BaseTrainingAdapter`` interface. Nothing in
    `lazycore` defines this base class -- per §2.9, formal inter-module
    contracts are explicitly deferred, and this is LazyTune-specific
    training machinery, not shared cross-module infrastructure -- so it is
    defined here as this package's one canonical interface.

    Only :class:`ProgrammaticAdapter` is implemented in this pass; the
    subprocess-isolated family is out of scope (see README).
    """

    @abc.abstractmethod
    def prepare(self, model_ref: object, dataset: Sequence[str]) -> None:
        """Load/attach a base model and get the adapter ready to train.

        Args:
            model_ref: a "bring your own local model handle" reference, per
                architecture doc §2.8 -- for :class:`ProgrammaticAdapter`
                this is either ``None`` (build the hermetic from-scratch
                model via :func:`build_hermetic_causal_lm`, sized off
                ``dataset``) or an explicit ``(model, tokenizer)`` tuple.
            dataset: the training text corpus (a small, in-memory sequence
                of strings at this scaffold's depth -- no streaming/Arrow
                dataset plumbing is implemented here).
        """

    @abc.abstractmethod
    def train_step(self, batch: Sequence[str]) -> TrainStepResult:
        """Run one real forward + backward + optimizer-step update on ``batch``."""

    @abc.abstractmethod
    def save_adapter(self, path: str | Path) -> None:
        """Persist the trained adapter weights (not the full base model) to ``path``."""


class ProgrammaticAdapter(BaseTrainingAdapter):
    """In-process LoRA fine-tuning adapter (the ``ProgrammaticAdapter`` family, §Part 3).

    Loads a small causal LM (hermetic-by-default via
    :func:`build_hermetic_causal_lm`, or a real ``(model, tokenizer)`` pair
    you supply), wraps it with `peft` LoRA, and runs genuine
    forward/backward/optimizer-step training via a plain PyTorch loop --
    the "in-process, no subprocess isolation" half of LazyTune's
    Adapter-Factory pattern. There is deliberately only one
    ``ProgrammaticAdapter`` implementation in this package (per CLAUDE.md's
    "one canonical adapter interface" rule) -- it does not wrap
    Unsloth/TRL specifically at this scaffold depth, only demonstrates the
    shape those wrappers would fill.
    """

    def __init__(
        self,
        *,
        lora_config: LoraConfig | None = None,
        learning_rate: float = 5e-3,
    ) -> None:
        """Construct an untrained adapter; call :meth:`prepare` before use.

        Args:
            lora_config: LoRA configuration to wrap the base model with.
                Defaults to :func:`default_lora_config` (targets GPT-2's
                ``c_attn`` projection with a tiny rank) when not provided.
            learning_rate: learning rate for the ``torch.optim.AdamW``
                optimizer created over the LoRA-only trainable parameters
                once :meth:`prepare` is called.
        """
        self.lora_config = lora_config or default_lora_config()
        self.learning_rate = learning_rate
        self.model: PeftModel | None = None
        self.tokenizer: TinyTokenizer | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._step = 0

    def prepare(self, model_ref: object, dataset: Sequence[str]) -> None:
        """Build/attach the base model, wrap it with LoRA, and set up the optimizer.

        Args:
            model_ref: ``None`` to build the hermetic from-scratch model via
                :func:`build_hermetic_causal_lm` (sized off ``dataset``), or
                an explicit ``(model, tokenizer)`` tuple to fine-tune a real
                base model instead (e.g. the checkpoint documented in
                :data:`MODEL_ALLOWLIST`).
            dataset: training text corpus. Only used to size/derive the
                hermetic tokenizer's vocabulary when ``model_ref`` is
                ``None``; ignored otherwise.

        Raises:
            RuntimeError: if wrapping the base model with
                ``self.lora_config`` via `peft`'s ``get_peft_model``
                produces no trainable parameters (usually a sign that
                ``lora_config.target_modules`` doesn't match any module
                name on the base model).
        """
        if model_ref is None:
            base_model, tokenizer = build_hermetic_causal_lm(dataset)
        else:
            base_model, tokenizer = model_ref  # type: ignore[misc]

        self.tokenizer = tokenizer
        self.model = get_peft_model(base_model, self.lora_config)
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "No trainable LoRA parameters found after get_peft_model() -- "
                "check that lora_config.target_modules matches the base "
                "model's module names."
            )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        self._step = 0

    def _require_prepared(self) -> tuple[PeftModel, TinyTokenizer]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call prepare(model_ref, dataset) before using this adapter")
        return self.model, self.tokenizer

    def compute_loss(self, batch: Sequence[str]) -> float:
        """Compute the current loss on ``batch`` without taking an optimizer step.

        Useful for a before/after comparison around one or more
        :meth:`train_step` calls (see ``examples/lora_finetune_example.py``).
        """
        model, tokenizer = self._require_prepared()
        was_training = model.training
        model.eval()
        try:
            encoded = tokenizer.batch_encode(batch)
            with torch.no_grad():
                outputs = model(**encoded)
            return float(outputs.loss.item())
        finally:
            if was_training:
                model.train()

    def train_step(self, batch: Sequence[str]) -> TrainStepResult:
        """Run one real forward + backward + ``optimizer.step()`` update on ``batch``.

        Tokenizes ``batch`` with the prepared tokenizer, computes the causal
        LM loss, backpropagates through the LoRA parameters only (the base
        model's weights are frozen by `peft`), and applies one AdamW step.

        Args:
            batch: a small in-memory sequence of training text strings.

        Returns:
            A :class:`TrainStepResult` with the scalar loss for this step
            and the running step counter (starts at 1, incremented on every
            call).

        Raises:
            RuntimeError: if called before :meth:`prepare`.
        """
        model, tokenizer = self._require_prepared()
        assert self.optimizer is not None

        encoded = tokenizer.batch_encode(batch)
        outputs = model(**encoded)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._step += 1
        return TrainStepResult(loss=float(loss.detach().item()), step=self._step)

    def save_adapter(self, path: str | Path) -> None:
        """Persist only the LoRA adapter weights (not the frozen base model) to ``path``.

        Creates parent directories as needed and delegates to `peft`'s
        ``PeftModel.save_pretrained``, which writes ``adapter_config.json``
        plus an ``adapter_model.*`` weights file. The saved directory can be
        reloaded onto a matching base model via ``PeftModel.from_pretrained``.

        Args:
            path: destination directory for the adapter files.

        Raises:
            RuntimeError: if called before :meth:`prepare`.
        """
        model, _tokenizer = self._require_prepared()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path))
