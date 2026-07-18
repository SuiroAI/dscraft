# lazycore

The thin, shared substrate underneath every Benchcraft module (AutoML,
LazyClean, LazyForecast, LazyGraph, LazyVision, LazyTune, LazyRed,
LazyAgent). It exists so that modules working on largely the same
underlying data, telemetry, and licensing concerns don't each invent their
own conventions -- without forcing premature unification of things that
genuinely differ per module.

## This package is intentionally thin. Here's why.

Benchcraft's packaging model (architecture doc §2.7) is Hugging Face-style:
independently-versioned, separate packages per module, sharing one thin
`lazycore` package for common schemas/interfaces with near-zero
dependencies -- explicitly **not** one monorepo with pip extras. The reason
is concrete, not aesthetic: PyTorch-heavy modules (LazyTune, LazyVision,
LazyGraph), the deliberately PyTorch-free LazyClean (which stays under a
100MB footprint by using ONNX Runtime instead of PyTorch), and the
Node-adjacent tooling implied by LazyRed's Promptfoo integration have
genuinely conflicting dependency universes that pip's extras resolver
cannot cleanly reconcile. If `lazycore` pulled in pandas, polars, or torch
as hard dependencies, every module -- including the ones explicitly
designed to avoid those dependencies -- would be forced to install them
just to get `import lazycore` to work. That defeats the entire point of
per-module packaging.

The architecture doc is equally explicit that formal, typed interface
contracts between modules (in the style of MLflow's model "flavors") are
**deferred** until at least two real modules actually need to exchange
data (§2.9) -- so `lazycore` does not attempt to be a data-contract system.
What it provides instead are lightweight, checked-in *conventions*:
type-hint-only interfaces, small helper functions, policy tables encoded as
data. Nothing here assumes what shape a future formal contract should take.

Concretely, `lazycore`'s runtime dependency footprint is exactly one
non-stdlib package: `opentelemetry-api` (never the SDK). It does not depend
on pandas, polars, torch, or any ML framework -- where those are referenced
for type hints, the imports are guarded behind `typing.TYPE_CHECKING` or
done lazily inside a function body, so importing `lazycore` never forces
any of them to be installed.

## What's in here

Everything in this package maps directly to a locked decision in Part 2 of
`Benchcraft_Unified_Architecture.md`. Nothing module-specific belongs here
(see the repo's `CLAUDE.md`, "lazycore stays thin").

### `lazycore.data` -- three-tier data/tensor conventions (§2.1)

- **Tier 1 (dense tabular/text/time-series):** small conversion/validation
  helpers over Arrow-backed pandas 2.x (`ArrowDtype`) and Polars, since both
  are confirmed near-zero-cost, interchangeable front-ends over the same
  Arrow buffers. `is_arrow_backed_pandas`, `pandas_arrow_dtypes`,
  `to_polars_zero_copy`, `from_polars_zero_copy`.
- **Tier 2 (sparse graph tensors):** `SparseGraphTensorAdapter`, an abstract
  base class describing the COO/CSR-CSC conversion boundary DLPack cannot
  represent. No graph-library dependency; LazyGraph provides concrete PyG-
  and DGL-facing implementations.
- **Tier 3 (dense image/audio):** `DenseMediaPipeline`, an abstract base
  class describing an FFCV-style decode → augment → DLPack-handoff
  pipeline shape (decode/augment are compute-bound, not copy-bound, so
  DLPack only applies at the final dense-tensor stage). LazyVision provides
  concrete implementations.

### `lazycore.telemetry` -- OpenTelemetry GenAI semantic conventions (§2.6)

Shared attribute-name constants (`security.severity`, `owasp.mapping`,
`ml.metric.*`) and a thin wrapper (`genai_span`, `set_security_finding`,
`set_ml_metric`, `add_transcript_event`) over `opentelemetry-api` so
LazyRed's security-audit reports, LazyAgent's execution trajectories, and
the ML leaderboards from AutoML/LazyForecast/LazyGraph/LazyVision all
report through the same schema. Depends only on `opentelemetry-api`; if the
calling application hasn't configured an SDK TracerProvider, spans are
OTel's documented no-ops -- safe to call, nothing exported until a real
exporter is wired up by whichever module needs one.

### `lazycore.licensing` -- license-isolation and model-allowlist policy (§2.2, §2.10)

- `RiskType` / `Mitigation` / `RISK_MITIGATIONS`: the §2.2 License Isolation
  Policy decision table, encoded as data rather than prose, so module
  owners can look up the exact required mitigation for a given dependency
  risk (GPL-at-build-time, restrictive optional build dep,
  AGPL/GPL-at-runtime-internal, AGPL/GPL-network-facing,
  source-available-non-compete, non-commercial weights).
- `ModelTier` / `ModelLicenseEntry` / `Allowlist`: the §2.10 mechanism every
  module uses to register and check model checkpoints against Tier 1
  (permissive, auto-usable) or Tier 2 (restricted, opt-in-gated) --
  including the runtime guard that raises `RestrictedLicenseNotAcceptedError`
  for a Tier 2 checkpoint unless the caller explicitly passes
  `accept_restricted_licenses=True`.

This is a **policy and mechanism**, not a populated list. Every `Allowlist`
starts empty; populating and maintaining the actual per-module allowlists
(which specific model checkpoints go in Tier 1 vs Tier 2) is called out in
the architecture doc (Part 6) as an ongoing, per-module maintenance task,
not something `lazycore` does on a module's behalf.

### `lazycore.sandbox` -- shared sandbox executor + adapter base class (§2.3, §2.3.1)

LazyRed and LazyAgent contain the same kernel-level threat class --
arbitrary code execution by a red-team target or a benchmarked agent -- so
per §2.3, LazyCore provides **one** shared sandbox executor and **one**
generic policy dataclass, with mode-specific policy *values* layered on
top by each module when it's built (LazyRed's "red-team target sandbox",
LazyAgent's "benchmark task sandbox"). Nothing module-specific (e.g. an
OWASP mapping, a benchmark-task allowlist) lives in this package -- only
the generic shape both modes share.

- `SandboxPolicy` -- generic, frozen dataclass config: `allow_network`,
  `allowed_read_paths`, `allowed_write_paths`, `allowed_executables`,
  plus env/timeout/cwd knobs. The exact same dataclass is meant to be
  instantiated with different values for LazyRed vs. LazyAgent.
- `BaseSandboxExecutor` -- the shared ABC (`is_available`, `run_command`,
  `run_callable`, returning a structured `SandboxResult`).
- `SeatbeltSandboxExecutor` -- the real, tested macOS backend. Generates an
  SBPL (Sandbox Profile Language) profile from a `SandboxPolicy` and runs
  the target command under `/usr/bin/sandbox-exec -f <profile> -- ...`.
- `LinuxNamespaceSandboxExecutor` -- a **documented stub**, not a real
  implementation. It satisfies the same ABC and reports availability via
  `shutil.which("bwrap"/"unshare")`, but every actual run method raises
  `SandboxBackendUnavailableError`. This repository's reference/dev
  environment is macOS, so a real gVisor/Firecracker/namespace-based
  backend (the intended Linux implementation per §2.3, since Linux has no
  VM-boundary GPU problem to design around) cannot be meaningfully built or
  verified here. Do not treat this stub as production-ready.
- `get_default_executor()` -- returns `SeatbeltSandboxExecutor` on macOS
  (when `/usr/bin/sandbox-exec` exists) or `LinuxNamespaceSandboxExecutor`
  on Linux; raises `SandboxBackendUnavailableError` on anything else.

**Why GPU/Metal/MPS access is never sandboxed here, on either platform**
(§2.3.1): 2026 research confirmed no Mac container/VM runtime (Docker
Desktop, Podman+libkrun, Apple's `container`/Containerization framework)
exposes Metal/MPS passthrough into an isolation boundary, and -- more
fundamentally -- Seatbelt itself **cannot** gate GPU/Metal/Cocoa access
even in principle, because those are system-level services outside SBPL's
reach. The locked v1 design is a **split-trust architecture**: GPU-bound
model inference always runs unsandboxed (there's no adversarial surface in
local weights/forward-pass compute), and only the CPU-bound tool-calling/
code-execution layer -- shell commands, file I/O, network egress, the
actual adversarial surface in both LazyRed's and LazyAgent's threat models
-- is what this package ever constrains. `SeatbeltSandboxExecutor` does not
implement, and will never implement, any GPU-blocking rule; see its module
docstring for the exact technical reasoning.

Runtime dependencies added by this subpackage: **none** -- it uses only
`subprocess`, `shutil`, `tempfile`, `pickle`, `platform`, `dataclasses`,
and `abc` from the standard library.

## What's deliberately NOT in here

- A *real* Linux sandbox backend -- `LinuxNamespaceSandboxExecutor` is a
  documented stub only; see `lazycore.sandbox` above.
- Any LLM router or `BaseTarget`/multi-provider abstraction (§2.8) --
  explicitly and permanently excluded platform-wide.
- Formal inter-module data contracts / MLflow "flavors"-style manifests
  (§2.9) -- deferred until two real modules need to exchange data.
- pandas, polars, torch, or any ML framework as a hard dependency.

## Installation

```bash
cd packages/lazycore
pip install -e .          # runtime only
pip install -e ".[dev]"   # + pytest for running the test suite
```

## Running tests

```bash
pytest packages/lazycore/tests
```

The Tier 1 data-helper tests additionally require `pandas` and `polars` to
be installed in the test environment (they are not runtime dependencies of
`lazycore` itself, so those tests are skipped via `pytest.importorskip` if
unavailable).

`tests/sandbox/test_seatbelt.py` actually invokes `/usr/bin/sandbox-exec`
against real temp directories to demonstrate enforcement (an allowed write
succeeds, a write outside the policy is denied, network egress is denied
by default) -- it is skipped entirely on non-macOS hosts via
`pytest.mark.skipif`, rather than mocked. `tests/sandbox/test_linux_stub.py`
only verifies the documented-stub behavior of `LinuxNamespaceSandboxExecutor`
on this non-Linux machine (it raises `SandboxBackendUnavailableError`); it
cannot and does not validate real Linux namespace isolation.
