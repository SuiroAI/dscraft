"""Export interface stubs (architecture doc §2.5, export backend 2: local LLM serving formats).

Per §2.5, LazyTune's v1 export scope is **narrowed** to local-only serving
formats -- GGUF (llama.cpp) and MLX-native (Apple Silicon) -- with the
original cloud/datacenter-serving pipeline (vLLM/SGLang/TensorRT-LLM,
AutoAWQ/Marlin quantization) explicitly deferred (Part 3, "Module 6:
LazyTune", v1 rescope note; Part 6 roadmap).

This module documents that narrowed interface **shape** without
implementing real conversion, which would require vendoring/depending on
heavyweight, environment-specific external tooling:

- Real GGUF export requires llama.cpp's own conversion scripts
  (``convert_hf_to_gguf.py`` and friends), which track a fast-moving,
  architecture-specific mapping from HuggingFace model configs to GGUF
  tensor layouts -- not something to reimplement or vendor at this
  scaffold's depth.
- Real MLX export requires Apple's ``mlx-lm`` conversion tooling
  (``mlx_lm.convert``), which is itself a moving target tied to specific
  MLX/Apple Silicon toolchain versions.

Both are therefore explicit, documented ``NotImplementedError`` stubs
rather than partial/fake implementations -- calling them tells you exactly
what real dependency you'd need to add and where, instead of silently
producing an incorrect or empty output file.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["export_gguf_stub", "export_mlx_stub"]


def export_gguf_stub(adapter_path: str | Path, output_path: str | Path) -> None:
    """Stub for exporting a fine-tuned model/adapter to GGUF (llama.cpp).

    **Not implemented.** Real GGUF export requires llama.cpp's own
    conversion tooling (e.g. ``convert_hf_to_gguf.py``, plus
    ``llama-quantize`` for the quantization step), which understands
    llama.cpp's tensor-layout/metadata format for each supported model
    architecture. That is a heavyweight, environment-specific external
    build dependency (a llama.cpp checkout/build, or the standalone
    ``gguf`` Python package plus per-architecture conversion logic) that is
    explicitly out of scope for this scaffold-depth pass -- see this
    package's README "What's deferred and why".

    Args:
        adapter_path: path to a saved adapter/model directory (e.g. from
            :meth:`dscraft.tune.adapter.ProgrammaticAdapter.save_adapter`).
        output_path: intended destination ``.gguf`` file path.

    Raises:
        NotImplementedError: always. This function exists to document the
            export interface's shape (per architecture doc §2.5's export
            backend 2), not to perform real conversion.
    """
    raise NotImplementedError(
        "export_gguf_stub() documents the GGUF export interface shape "
        "(architecture doc §2.5, export backend 2) but does not perform "
        "real conversion. GGUF export requires llama.cpp's external "
        "conversion tooling (convert_hf_to_gguf.py / llama-quantize), "
        "which is out of scope for this scaffold-depth pass. See the "
        "'dscraft.tune' section of packages/dscraft/README.md for what's "
        "deferred and why. "
        f"(requested: {Path(adapter_path)} -> {Path(output_path)})"
    )


def export_mlx_stub(adapter_path: str | Path, output_path: str | Path) -> None:
    """Stub for exporting a fine-tuned model/adapter to MLX-native format.

    **Not implemented.** Real MLX export requires Apple's ``mlx-lm``
    conversion tooling (``mlx_lm.convert``), which depends on a specific
    MLX/Apple Silicon toolchain and its own per-architecture conversion
    mapping. Vendoring or reimplementing that conversion path is
    explicitly out of scope for this scaffold-depth pass -- see this
    package's README "What's deferred and why".

    Args:
        adapter_path: path to a saved adapter/model directory (e.g. from
            :meth:`dscraft.tune.adapter.ProgrammaticAdapter.save_adapter`).
        output_path: intended destination MLX model directory/path.

    Raises:
        NotImplementedError: always. This function exists to document the
            export interface's shape (per architecture doc §2.5's export
            backend 2), not to perform real conversion.
    """
    raise NotImplementedError(
        "export_mlx_stub() documents the MLX export interface shape "
        "(architecture doc §2.5, export backend 2) but does not perform "
        "real conversion. MLX export requires Apple's external mlx-lm "
        "conversion tooling (mlx_lm.convert), which is out of scope for "
        "this scaffold-depth pass. See the 'dscraft.tune' section of "
        "packages/dscraft/README.md for what's deferred and why. "
        f"(requested: {Path(adapter_path)} -> {Path(output_path)})"
    )
