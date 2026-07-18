# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository contains real source code under `packages/` — nine independently-installable Python packages (`lazycore` plus eight modules), each with its own `pyproject.toml`, `src/`, `tests/`, and `examples/`. Install a package with `pip install -e packages/<name>[dev]` (install `lazycore` first, since every other package depends on it) and run its tests with `pytest packages/<name>/tests`. See the module table and dependency graph below for what's built and how packages relate.

## What this document is

`Benchcraft_Unified_Architecture.md` is the locked v1 architecture spec for **Benchcraft**, a unified, MIT-licensed, local-first ML tooling platform (tabular AutoML, data cleaning, time-series forecasting, graph ML, computer vision, LLM fine-tuning, LLM/agent red-teaming, agent/RAG benchmark eval). It was synthesized from nine independent module research docs plus a stakeholder scope-lock Q&A, and its decisions marked "Locked" in Part 4 should be treated as settled — do not silently re-litigate them when writing code or docs. Read the full document before implementing any module; the summary below only orients you.

**Post-v1 direction (not a v1 requirement):** Benchcraft is meant to eventually expose an MCP server and an installable Skill so Hermes and other AI agent tools can consume its modules as callable tools, not just as a Python library. This is a generic library capability, not a bespoke integration for one consumer. See Part 6 of the architecture doc. Don't shape any module's v1 public API around this prematurely — just keep APIs clean enough to wrap later.

## Core scope constraints (apply to any code written in this repo)

- **Local-only, v1.** No cloud deployment path, no remote inference endpoint, no cloud/remote targets for red-teaming or agent-eval modules. Everything runs on the reference machine (Apple Silicon M4 Max class, 128GB unified memory).
- **MPS is the primary backend**, Linux+CUDA is a secondary, non-validated target — don't build CUDA-only code paths without an MPS-compatible fallback story.
- **No LLM router, no multi-provider abstraction.** Each module takes a bare-minimum "bring your own local model handle" approach (a loaded HF/PyTorch/MLX model object or local file path). Do not introduce a shared `BaseTarget`/router interface — this is explicitly and deliberately excluded, not a phase-1 gap.
- **No formal inter-module data contracts yet.** Don't build a shared typed interface (e.g., MLflow "flavors"-style) between modules preemptively; that's deferred until two real modules need to exchange data.
- **MoE over dense** for local LLM guidance at large parameter counts (dense 70B is impractically slow on this hardware; ~70B-total/~3B-active MoE is not).
- **Edge/microcontroller compilation (LazyEdge) is fully deferred** — internal roadmap only, not user-facing, do not build it.

## Shared architecture (LazyCore) — apply these conventions across modules

- **Data/tensor foundation is tiered, not uniform.** Dense tabular/text/time-series → Apache Arrow (pandas 2.x `ArrowDtype`, Polars), zero-copy. Sparse graph tensors → dedicated COO/CSR-CSC adapter (DLPack cannot represent sparsity — real conversion required). Dense image/audio → native FFCV-style decode+augment pipeline with DLPack handoff only at the final dense-tensor stage (these workloads are compute-bound, not copy-bound).
- **Packaging is per-module, not a monorepo.** Independently-versioned packages per module (Hugging Face-style), sharing one thin `lazycore` package for common schemas/interfaces with near-zero dependencies. PyTorch-heavy, PyTorch-free (LazyClean), and Node-adjacent (LazyRed/Promptfoo) modules have genuinely conflicting dependency universes — don't try to unify them under one dependency tree or pip-extras setup.
- **Telemetry uses OpenTelemetry GenAI semantic conventions** as the shared schema across security reports, agent trajectories, and ML leaderboards (spans + custom attributes like `security.severity`, `owasp.mapping`, `ml.metric.accuracy`).
- **Export/compilation has three permanently separate backends** — don't unify under one IR: (1) ONNX/ONNX Runtime (AutoML via skl2onnx, LazyVision via `torch.export`→onnx-graphsurgeon), (2) local-only LLM serving formats GGUF/MLX for LazyTune (no vLLM/TensorRT-LLM/AutoAWQ-Marlin in v1), (3) edge compilation (deferred).
- **Sandbox execution is shared between LazyRed and LazyAgent** (one executor + adapter base class, mode-specific policies on top) — both contain the same kernel-level threat class (arbitrary code execution). LazyRed's semantic-level threats (prompt injection, credential leakage) are handled by a *separate* Guardrail/Firewall layer, not the sandbox.
- **Mac sandboxing uses a split-trust model**: GPU-bound model inference runs unsandboxed (no Mac container runtime exposes Metal/MPS passthrough, and there's no adversarial surface there); only the tool-calling/code-execution layer is sandboxed — `sandbox-exec`/Seatbelt on macOS, gVisor/Firecracker/namespaces on Linux. Do not attempt to run GPU inference inside a hard isolation boundary on Mac.

## Licensing policy (LazyIsolate) — check before adding any dependency or bundled model

- GPL/copyleft code needed at build/link time → exclude entirely via build flags (e.g., SuiteSparse/CHOLMOD).
- Restrictively-licensed optional build deps (e.g., METIS) → never enable; use a permissive alternative (e.g., torch-cluster's Graclus instead of torch-sparse+METIS).
- AGPL/GPL code needed at runtime, purely local/internal → isolate via subprocess/IPC (e.g., Ultralytics YOLO detectors run in a separate subprocess/virtualenv). This is a risk-reduction posture pending counsel review, not a cleared legal position — see Part 5 of the architecture doc.
- AGPL/GPL code that would itself be the network-facing service → full replacement or licensing negotiation, no subprocess workaround applies.
- Source-available non-compete licenses (e.g., PyCaret's FSL-1.1-MIT) → clean-room reimplementation of the API surface only, never copy code.
- Non-commercial-licensed weights/data (e.g., CC BY-NC) → isolate behind an explicit opt-in flag (`accept_restricted_licenses=True`), per the Tier 1/Tier 2 model allowlist below.
- **Platform code target: 100% MIT.** External model checkpoints are classified per module as Tier 1 (permissive, e.g., Apache-2.0/MIT — auto-usable) or Tier 2 (restricted, e.g., CC BY-NC — opt-in-gated). Maintaining and re-verifying these allowlists is an ongoing task, not one-time.

## The nine modules (all "Lazy*" names are internal codenames, not user-facing)

| Module | Status | One-line focus |
|---|---|---|
| AutoML | In v1 | Tabular AutoML; clean-room PyCaret-successor; streaming `partial_fit`, PSI drift detection, `.compile()` → single ONNX graph via skl2onnx |
| LazyClean | In v1 | Data-quality firewall: DeCoLe label-error detection + D4 semantic dedup; ONNX Runtime embeddings (not PyTorch) to stay <100MB |
| LazyForecast | In v1 | Classical (StatsForecast) + tree-based (MLForecast) + zero-shot TSFMs (TimesFM, Chronos-Bolt) under one Polars/Arrow pipeline; conformal prediction (MSCP/EnbPI) |
| LazyGraph | In v1 | MPNNs (GCN/GAT/GraphSAGE) + Graph Transformers across PyG/DGL via a Universal Sparse Tensor layer; SQL-to-graph mapping; oversmoothing/oversquashing monitoring |
| LazyVision | In v1, MPS-targeted | CNNs/ViTs/detectors (YOLO/D-FINE/RT-DETR) under one preprocessing abstraction; Rust/PyO3 data loading; AGPL detectors isolated as subprocess plugins |
| LazyTune | In v1, local models only | LLM fine-tuning adapter factory over Axolotl/LLaMA-Factory/Unsloth/TRL/torchtune; multi-fidelity BOHB micro-tuning; export limited to GGUF/MLX (no cloud-serving quant) |
| LazyRed | In v1, local targets only | Red-teaming adapter over garak/DeepTeam/PyRIT/Promptfoo; offline-first; findings mapped to OWASP LLM/Agentic Top 10 + MITRE ATLAS |
| LazyAgent | In v1, local targets only | Agent/RAG benchmark eval; bring-your-own-agent adapter; Pareto RAG optimization (accuracy/latency/cost); shares sandbox executor with LazyRed |
| LazyEdge | Deferred (roadmap only) | Edge/microcontroller compilation — not user-facing, do not implement |

For full per-module detail (algorithms, failure modes, licensing specifics) read Part 3 and Appendix A of `Benchcraft_Unified_Architecture.md` directly rather than relying on the table above.

## Module dependency graph (what can be built in parallel)

Per §2.9, formal inter-module contracts are explicitly deferred, so modules never call into each other. That means the only real build-order constraint comes from two pieces of *shared* infrastructure:

- **`packages/lazycore/`** (three-tier data conventions §2.1, OTel schema helpers §2.6, license-isolation policy §2.2/§2.10) — must exist before any module work starts, but is intentionally thin/small.
- **Shared sandbox executor + adapter base class** (§2.3), living under `packages/lazycore/sandbox/` — blocks only LazyRed and LazyAgent, nothing else.

Once `lazycore` exists, these have **zero cross-module dependency** and can be built fully in parallel: AutoML, LazyClean, LazyForecast, LazyGraph, LazyVision, LazyTune. LazyRed and LazyAgent additionally need the shared sandbox executor. LazyEdge is not built (deferred). Don't invent inter-module imports or a shared runtime interface to "help" this along — the architecture doc treats that as premature until two real modules need it.

## Working as a library, not a pile of scripts

1. **No net-new scripts.** Every capability lives inside a module's installable package (`packages/<module>/src/<pkg>/...`) behind a real function/class API — not a standalone script at the repo root or inside a module dir. Runnable demos go in that module's `examples/` and import the package; they never reimplement logic inline.
2. **Fix what's there before adding new.** Before writing a new file or function, search the target module (and `lazycore`) for an existing implementation of the same capability and extend/fix it in place. Don't create `foo_v2.py`, `foo_new.py`, or a parallel class hierarchy next to something that already does the job — this is the exact duplicated-ONNX-export failure mode the architecture doc was written to eliminate.
3. **One canonical location per capability per module** (one export path, one preprocessing entrypoint, etc.).
4. **Consistent per-module layout** (HF-style independent packaging, §2.7):
   ```text
   packages/<module>/
     pyproject.toml
     src/<module_pkg>/__init__.py   # public API surface
     src/<module_pkg>/...
     tests/
     README.md                       # what the package does + its signature capability
   ```
5. **`lazycore` stays thin.** Only the shared conventions already locked in Part 2 of the architecture doc belong there (data tiers, OTel schema, license policy, sandbox executor) — never module-specific logic.

## Subagent workflow rules

1. **All implementation work is delegated to subagents** via the Agent tool. The orchestrating session designs prompts, reviews diffs, and sequences work — it does not write module code directly.
2. **Every subagent prompt must be a complete spec**, never a one-liner. Required components: the exact goal/deliverable; the specific architecture-doc subsection to implement against; exact file paths to create/modify; an explicit "search existing code first, extend don't duplicate" instruction; the relevant locked constraints (from the sections above) restated for that task; and concrete acceptance criteria (what to run/verify to confirm it's done).
3. **Parallel code-writing subagents must work in disjoint directories** (each module owns its own `packages/<module>/` tree). Use `isolation: worktree` when running multiple concurrently.
4. **Shared-infrastructure work is reviewed before fan-out.** `lazycore` (and later, the sandbox executor) gets reviewed against its acceptance criteria before module agents that depend on it are launched.
