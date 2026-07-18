"""Tests for `benchcraft_lazyvision.model`: model construction and the
package's canonical device-resolution helper, `resolve_device`.

`resolve_device` mirrors `benchcraft_lazygraph.gcn.resolve_device`'s exact
MPS -> CUDA -> CPU auto-detect-with-fallback pattern (see that module's
`tests/test_gcn.py` for the matching test pattern this file follows), so
that this package's production-facing default (what a caller gets when
they don't pass a `device`) is MPS-first, not an implicit CPU default --
while this module's own tests still explicitly pin `device="cpu"` for
hermetic, deterministic, portable automated verification.
"""

from __future__ import annotations

import torch

from benchcraft_lazyvision import (
    ModelConfig,
    TinyCNN,
    build_model,
    resolve_device,
    synthetic_classification_batch,
)


def test_resolve_device_default_never_raises() -> None:
    """Calling `resolve_device` with no arguments always returns a valid
    `torch.device`, regardless of what hardware is actually available."""
    device = resolve_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps", "cuda")


def test_resolve_device_falls_back_cleanly_for_bogus_preference() -> None:
    """Passing an unavailable/invalid preferred device string should not
    raise -- `resolve_device` should fall through to auto-detection
    instead."""
    device = resolve_device(preferred="not-a-real-device")
    assert device.type in ("cpu", "mps", "cuda")


def test_resolve_device_prefers_mps_when_available() -> None:
    """When MPS is actually available on this machine, `resolve_device()`
    (called with no explicit preference) must return it -- confirming the
    production-facing default is genuinely MPS-first, not just
    CPU-with-an-MPS-escape-hatch."""
    device = resolve_device()
    if torch.backends.mps.is_available():
        assert device.type == "mps"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_resolve_device_honors_explicit_cpu_preference() -> None:
    """An explicit, valid `preferred` device must be honored even when a
    higher-priority backend (MPS/CUDA) is available."""
    device = resolve_device(preferred="cpu")
    assert device.type == "cpu"


def test_build_model_default_device_matches_resolve_device() -> None:
    """`build_model` with no explicit `device` must place the model on
    exactly whatever `resolve_device()` returns -- i.e. it must not
    silently default to CPU regardless of what's available."""
    expected_device = resolve_device()
    model = build_model(ModelConfig(image_size=8), seed=0)
    assert isinstance(model, TinyCNN)
    actual_device = next(model.parameters()).device
    assert actual_device.type == expected_device.type


def test_build_model_honors_explicit_device_override() -> None:
    """An explicit `device="cpu"` must be honored regardless of what
    `resolve_device()` would otherwise pick -- this is what this package's
    own tests rely on for hermetic, deterministic verification."""
    model = build_model(ModelConfig(image_size=8), seed=0, device="cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_synthetic_classification_batch_default_device_matches_resolve_device() -> None:
    """`synthetic_classification_batch` with no explicit `device` must
    place its output tensors on exactly whatever `resolve_device()`
    returns."""
    expected_device = resolve_device()
    images, labels = synthetic_classification_batch(
        ModelConfig(image_size=8), batch_size=2, seed=0
    )
    assert images.device.type == expected_device.type
    assert labels.device.type == expected_device.type


def test_synthetic_classification_batch_honors_explicit_device_override() -> None:
    """An explicit `device="cpu"` must be honored for both returned
    tensors."""
    images, labels = synthetic_classification_batch(
        ModelConfig(image_size=8), batch_size=2, seed=0, device="cpu"
    )
    assert images.device.type == "cpu"
    assert labels.device.type == "cpu"
