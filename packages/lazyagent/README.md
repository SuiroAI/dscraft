# benchcraft-agent

LazyAgent's signature capability at this scaffold's depth (architecture
doc `Benchcraft_Unified_Architecture.md`, Part 3, "Module 8: LazyAgent"): a
minimal, real, **bring-your-own-agent task-execution benchmark loop**. A
plain Python callable -- standing in for a real framework-agnostic agent,
per MASEval's `AgentAdapter` interface (smolagents/LangGraph/AutoGen/CAMEL,
per the module survey) -- executes a small file-manipulation tool-use task
*inside the shared `lazycore.sandbox` executor*, and the run is scored for
success/failure with basic accuracy/latency metrics reported via
`lazycore.telemetry`'s OTel GenAI helpers.

## Scope

**In scope for this pass:**

- One `AgentAdapter` interface (`benchcraft_lazyagent.adapter`) with one
  concrete implementation, `SandboxedAgentAdapter`, that always executes
  the agent's chosen action through a caller-supplied
  `lazycore.sandbox.BaseSandboxExecutor`.
- One concrete task family (`benchcraft_lazyagent.tasks`): a
  file-manipulation tool-use task -- "create a file named X with content Y
  in the sandboxed working directory" -- with two fixed variants: a
  pass-designed task (write target inside the allowed sandbox path) and a
  fail-designed sandbox-escape-attempt task (write target deliberately
  outside the allowed sandbox path).
- One reference agent callable, `rule_based_agent`: a plain, deterministic
  Python function (not an LLM, not a real framework integration) that
  reads a task's structured fields and proposes a shell command. It exists
  to exercise the loop end-to-end, not to be a capable agent.
- One tiny multi-task benchmark runner (`benchcraft_lazyagent.benchmark`)
  that runs a small, fixed task suite and reports an aggregate pass rate +
  mean wall-clock latency (measured via `time.perf_counter()`).
- OTel GenAI telemetry via `lazycore.telemetry` (`genai_span`,
  `set_ml_metric` for `ml.metric.accuracy`, `add_transcript_event` for each
  trajectory step) -- no parallel telemetry/reporting schema is built here.

**Explicitly deferred (not in this pass), and why:**

| Deferred | Why |
|---|---|
| Real agent framework integrations (smolagents, LangGraph, AutoGen, CAMEL) | "Bring your own agent" at this scaffold depth means the core loop accepts *any* Python callable matching `AgentFn`'s signature -- wiring in a specific framework's decision function is real future integration work, not core-loop plumbing. Per §2.8, this platform explicitly does not build a router/registry of supported frameworks. |
| Multi-Objective Pareto RAG Optimization loop (accuracy vs. latency vs. cost) | This is LazyAgent's headline capability in Part 3, but it's a full optimization loop over a RAG pipeline's chunking/indexing/reranking/model-choice search space -- a substantially larger scope than "prove the sandboxed benchmark-eval loop works end-to-end," which is this pass's goal. |
| DISCO-style sample condensation (~1% informative task subset) | A data-selection technique for the Pareto RAG optimization loop above; deferred along with it. |
| SWE-bench-style heavyweight task suites | Per the architecture doc's own v1 rescope note for this module: "given ARM64/Apple-Silicon friction with Docker-based SWE-bench-style isolation, the initial benchmark suite should prioritize tasks compatible with the Mac-first sandbox strategy" -- this pass follows that guidance literally by using a Seatbelt-compatible file-manipulation task instead of a Docker-dependent suite. |
| RAG pipeline tuning (chunking/indexing/reranking search space, reranker-latency tradeoffs, prod-vector-DB disconnects -- Appendix A) | Informational/deferred per the task brief; not implemented in this pass. |
| Cloud/remote agent targets | Locked out of v1 scope platform-wide for LazyAgent (architecture doc Part 4, Part 6). |

## Sandbox wiring

This package **reuses `lazycore.sandbox` for all containment** -- it does
not build a second sandbox mechanism (per CLAUDE.md's "fix what's there /
no duplication" rule and architecture doc §2.3). `SandboxedAgentAdapter`
always calls `executor.run_command(...)` on a caller-supplied
`lazycore.sandbox.BaseSandboxExecutor` (typically
`lazycore.sandbox.get_default_executor()`, which resolves to the real
`SeatbeltSandboxExecutor` on macOS).

LazyAgent layers its **own** mode-specific `SandboxPolicy` values on top of
that shared executor, per architecture doc §2.3's description of this
module's sandboxing research as "the platform's most rigorous":

- **`allow_network=False`** (default-deny egress) on every task policy --
  this module's benchmark tasks have no legitimate reason to reach the
  network, and per CLAUDE.md's "local-only, v1" constraint, no network
  calls belong in the core path.
- **`allowed_write_paths` scoped to a single per-task temp workspace
  directory** -- never the whole filesystem, never a shared directory
  across tasks. The pass-designed task's write target is inside this
  directory; the fail-designed task's write target is a sibling directory
  deliberately *excluded* from `allowed_write_paths`, to prove containment
  is real (see "Fail-designed task" below).
- **`allowed_read_paths` left at its default (unrestricted read)** --
  matching `lazycore.sandbox`'s own documented default behavior (the
  write/network surfaces are the actual enforcement points for this task
  family; there is no sensitive read-only data these tasks need to be
  isolated from).
- **`timeout_seconds` set** on every task policy, so a misbehaving agent
  action can't hang the benchmark run indefinitely.

Per architecture doc §2.3.1's split-trust architecture, this package never
attempts to sandbox any GPU/Metal/MPS-bound process -- there is none in
this scaffold; the only thing ever run inside the sandbox is the agent's
proposed shell command.

### Fail-designed task proves containment is real, not decorative

`benchcraft_lazyagent.tasks.make_fail_task` builds a task whose target file
lives in a `forbidden/` directory that is a sibling of (but not included
in) the sandbox policy's `allowed_write_paths`. The rule-based reference
agent still *attempts* the write (it doesn't know the sandbox will block
it) -- the Seatbelt backend's default-deny write policy is what actually
stops it. The test suite (`tests/test_tasks.py`) asserts on the real
filesystem state after the run: not just that the task is scored `False`,
but that the forbidden file (and even its parent directory, since `mkdir
-p` on it is blocked too) was never created. This is the concrete
demonstration that sandbox containment drives the scored benchmark
outcome, not just decorates it.

## Public API

```python
from benchcraft_lazyagent import (
    # adapter.py
    AgentAdapter, SandboxedAgentAdapter, AgentAction, AgentFn,
    AgentTrajectory, TrajectoryStep, TaskSpec, TaskResult,
    # tasks.py
    FileTaskSpec, rule_based_agent, score_file_task,
    make_pass_task, make_fail_task, default_task_suite,
    # benchmark.py
    BenchmarkReport, ScorerFn, run_task, run_benchmark,
)
```

- `AgentFn = Callable[[TaskSpec], AgentAction]` is the bring-your-own-agent
  seam: any Python callable with this signature can stand in for "the
  agent" -- a real framework's step function could be adapted to match
  this signature without touching `SandboxedAgentAdapter` or
  `run_benchmark` at all.
- `SandboxedAgentAdapter(agent_fn).run_task(task, executor)` runs one task
  and returns an `AgentTrajectory` (a 3-step transcript: the task
  description as a `"user"` turn, the agent's chosen action as an
  `"assistant"` turn, and the sandbox's real `SandboxResult` as a `"tool"`
  turn).
- `run_benchmark(tasks, agent_fn)` runs a small task suite end-to-end and
  returns a `BenchmarkReport` with per-task `TaskResult`s plus an aggregate
  `pass_rate` and `mean_latency_seconds`.

## Installation

```bash
pip install -e packages/lazycore
pip install -e "packages/lazyagent[dev]"
```

`lazycore` is declared as a bare (unpinned) dependency in `pyproject.toml`,
matching the convention already established by `packages/lazytune`,
`packages/automl`, and `packages/lazyforecast`. It is a local sibling
package, not published to PyPI, so it still must be installed (or
otherwise made resolvable) first -- a plain `pip install -e
"packages/lazyagent[dev]"` without `lazycore` already installed/resolvable
will fail to resolve the dependency.

This package's own dependency surface is **stdlib + lazycore only** -- no
smolagents/langgraph/autogen/camel, per the "explicitly deferred" table
above.

## Running tests

```bash
pytest packages/lazyagent/tests
```

The suite exercises the **real** `SeatbeltSandboxExecutor` on macOS (this
repo's reference platform) -- it is skipped, not mocked, on non-macOS
hosts via `pytest.mark.skipif`. Tests assert on real filesystem state
(files that should exist do; files that should have been blocked do not)
and on a real, finite, non-zero mean latency computed from
`time.perf_counter()` measurements.

## Running the example

```bash
python packages/lazyagent/examples/agent_benchmark_example.py
```

Runs the two-task default suite (one pass-designed, one fail-designed) and
prints each task's pass/fail status plus the aggregate pass rate and mean
latency.
