# benchcraft-security

A scaffold-depth implementation of one signature capability from
Benchcraft's LazyRed module (architecture doc Part 3, "Module 7:
LazyRed"): a **real, minimal red-teaming probe/detector loop against a
local target**, built around a `BaseSecurityAdapter` interface, with the
target invocation genuinely run through the shared `lazycore.sandbox`
executor and findings mapped to the OWASP LLM Top 10 and reported via
`lazycore.telemetry`'s OTel GenAI-schema helpers.

## Scope

The architecture doc describes LazyRed as unifying garak, DeepTeam, PyRIT,
and Promptfoo Red Team under one `BaseSecurityAdapter` pattern, with an
Adversarial Target-Mutator Layer ("TopicAttack"), a Guardrail/Firewall
semantic layer, Multi-Model Jury Consensus judging, and a unified JSONL
report on a Vulnerability & Failure Rate Leaderboard. This pass implements
the smallest **real, end-to-end** slice of that: one probe, one detector,
one adapter, one sandbox wiring, one small leaderboard.

In scope for this pass:

1. **`BaseSecurityAdapter`** (`adapter.py`) -- a minimal abstract
   interface: `generate_attempt(probe_input) -> Attempt`,
   `run_target(attempt, executor) -> Attempt`, `detect(attempt) -> Finding`,
   plus a `run()` convenience that chains all three. Mirrors the same
   Adapter-Factory reasoning `benchcraft_lazytune.adapter.BaseTrainingAdapter`
   used for LazyTune -- LazyRed-specific machinery, not shared lazycore
   infrastructure.

   **Deliberately separate from LazyAgent's `AgentAdapter`, not an
   oversight.** LazyAgent's benchmark-eval module has its own,
   differently-shaped adapter interface for wrapping a bring-your-own
   agent under test. `BaseSecurityAdapter` here is shaped around garak's
   Probe/Generator/Detector attempt-transaction pattern (attack payload
   in, leak/failure verdict out), which is a genuinely different contract
   than "run an agent against a benchmark task and score its trajectory."
   The two modules do share the sandbox *executor* (`lazycore.sandbox`,
   §2.3) -- that's the actual shared infrastructure the architecture doc
   calls out -- but per §2.9, formal inter-module data contracts (e.g. a
   single merged adapter base class both modules implement against) are
   explicitly deferred until two real modules need to exchange data
   through one shared shape, not built preemptively because two adapters
   happen to look superficially similar. If a third module later needs
   the same "probe in, verdict out" shape, promoting a common base to
   `lazycore` becomes worth revisiting then; until that's a real,
   demonstrated need, keeping `BaseSecurityAdapter` module-specific here
   is the correct call, not a gap to fix.
2. **`PromptInjectionAdapter`** (`probes.py`) -- the one concrete probe: a
   deliberately naive, deliberately vulnerable local "target" function
   (`naive_vulnerable_target`, a plain Python function standing in for a
   local model) that echoes back its canned system prompt -- including a
   fake protected secret -- whenever the input contains a
   repeat/echo/override trigger phrase. `detect_secret_leak` is a simple
   substring/regex check for whether the fake secret leaked into the
   output.
3. **`Finding`** (`adapter.py`) -- mapped to `"LLM01: Prompt Injection"`,
   hardcoded because that is genuinely and exactly what this probe tests.
   Reported via `lazycore.telemetry.genai_span` /
   `lazycore.telemetry.set_security_finding` (one OTel span per attempt) --
   not a parallel reporting schema.
4. **`LeaderboardReport` / `run_leaderboard`** (`leaderboard.py`) -- a
   small, real, in-memory/printable pass/fail aggregation: run the probe N
   times with slight payload variations (`default_payload_variations`),
   count vulnerable vs. resisted attempts, compute a failure rate. This is
   a stand-in for the architecture doc's full unified-JSONL Vulnerability
   & Failure Rate Leaderboard, not that machinery itself.
5. **Offline-first** -- the "target" is a local Python function, never a
   network call. No dependency in this package's core path makes any
   network request.

## Sandbox wiring (real, not decorative)

Per architecture doc §2.3, LazyRed does **not** define its own sandbox
executor class -- it reuses `lazycore.sandbox`'s shared
`BaseSandboxExecutor` (`SeatbeltSandboxExecutor` on macOS via
`get_default_executor()`), layering its own mode-specific
`SandboxPolicy` on top (`probes.build_probe_sandbox_policy`: no
filesystem writes, no network, a short timeout).

`PromptInjectionAdapter.run_target` binds the attack payload to the
naive target function via `functools.partial` (the target function must
stay a plain, picklable, module-level function -- no closures -- because
the Seatbelt backend marshals the callable into a real subprocess to
actually sandbox it) and calls `executor.run_callable(...)`. On macOS this
launches a genuine `/usr/bin/sandbox-exec`-wrapped Python subprocess; the
result comes back as a `SandboxResult` with a real `exit_code`,
`stdout`, and `policy_blocked` flag, which `run_target` decodes back into
`Attempt.raw_output`. `tests/test_probes.py` asserts against the real
`SeatbeltSandboxExecutor` (skipping, not mocking, on platforms where no
sandbox backend is available) -- see that file's module docstring.

Per §2.3.1's split-trust model, this is treated purely as the CPU-bound
orchestration/tool-execution step worth demonstrating the sandbox wiring
on. There is no real GPU-bound model inference in this scaffold at all --
`naive_vulnerable_target` is a plain Python function standing in for a
local model's generation step, not an actual model forward pass, so no
split-trust GPU-unsandboxed path needs to be built here.

## OWASP mapping used

This probe maps to **`LLM01: Prompt Injection`** exclusively — hardcoded
in `probes.OWASP_PROMPT_INJECTION`, because a system prompt's protected
content leaking into model output via an injected "repeat/reveal"
instruction is exactly and only that failure mode. No generic
probe-to-OWASP-ID inference logic exists in this package; each probe
states its own true mapping.

## What's deferred, and why

Everything below is explicitly out of scope for this pass, tracked as
real future work per the architecture doc, not silently dropped:

- **garak / DeepTeam / PyRIT / Promptfoo integration.** The architecture
  doc's actual plan is to unify these four toolkits behind
  `BaseSecurityAdapter`. This pass adds none of them as dependencies --
  `PromptInjectionAdapter` only demonstrates the *shape* a real adapter
  wrapping one of these tools would fill. Wiring in a real toolkit is
  genuine future integration work (dependency footprint, config surface,
  their own generator/harness abstractions), not something to fake here.
- **Guardrail/Firewall semantic layer (§2.4).** This is explicitly a
  separate LazyRed subsystem from the sandbox executor -- responsible for
  detecting/blocking prompt injection and credential leakage at the
  prompt-response *semantic* level, as a standing policy layer. This
  scaffold's `detect_secret_leak` is a single, one-off detector function
  for one probe's own scoring step; it is not the Guardrail/Firewall layer
  and should not be conflated with it, per the task's explicit instruction
  not to conflate kernel-level sandboxing with semantic policy
  enforcement.
- **Adversarial Target-Mutator Layer ("TopicAttack").** The
  contextual-topic-transition-bridge mutation paradigm for testing RAG
  indirect-prompt-injection resilience is a substantial mutator system in
  its own right, unrelated to this pass's single static-payload-variation
  probe.
- **Multi-Model Jury Consensus.** The elevated architecture-doc paragraph
  describes a heterogeneous panel of judges from non-overlapping model
  lineages voting on each verdict. That requires multiple real local judge
  models and a voting/aggregation protocol across them -- genuinely out of
  scope for a scaffold with no real model targets at all. This pass's
  `detect_secret_leak` is a single deterministic detector, not a judge
  panel, and is not attempting to approximate one.
- **Full unified JSONL report / OWASP Agentic Top 10 / MITRE ATLAS ID
  aggregation across many probes.** `LeaderboardReport` aggregates one
  probe's own findings in-memory and prints a summary; it is not the
  cross-probe, persisted, multi-run reporting system described in Part 3.
- **Appendix A's LazyRed findings** (multi-turn state fragmentation,
  token-cost optimization for search-style attacks, measurement
  instability mitigations like Structured Binary Rubrics/golden-dataset
  calibration) -- informational/deferred per the task, and this scaffold's
  probe is single-turn with a deterministic, non-search-based detector, so
  none of these apply yet.

## Installation

`lazycore` is a local sibling package and is not listed as a formal path
dependency in `pyproject.toml` (hatchling/pip have no portable way to
express that in metadata) -- install it first:

```bash
pip install -e packages/lazycore
pip install -e "packages/lazyred[dev]"
```

## Running the tests

```bash
pytest packages/lazyred/tests
```

`tests/test_probes.py` exercises the real `SeatbeltSandboxExecutor` on
macOS (this project's reference platform) -- it is not mocked.

## Running the example

```bash
python packages/lazyred/examples/prompt_injection_probe_example.py
```

Runs the prompt-injection probe against the naive local target 8 times
(cycling through known injection triggers and benign control payloads),
through the real sandbox executor, and prints a leaderboard summary
showing the target's actual failure rate.
