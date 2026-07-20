"""A tiny multi-task benchmark runner (architecture doc Part 3, "Module 8: LazyAgent").

This is a deliberately minimal stand-in for the full "leaderboard" style
Multi-Objective Pareto RAG Optimization loop (accuracy vs. latency vs.
cost) described in Part 3 -- that loop, along with DISCO-style sample
condensation, is explicitly out of scope for this pass (see README). What
*is* implemented here: run a small, fixed set of :class:`TaskSpec` variants
through a :class:`~dscraft.agent.adapter.SandboxedAgentAdapter`,
score each one, and aggregate a pass rate + mean wall-clock latency --
enough to prove the sandboxed benchmark-eval loop end-to-end.

Reports each run via `dscraft.core.telemetry`'s OTel GenAI helpers
(`genai_span`, `set_ml_metric`, `add_transcript_event`) rather than a
parallel telemetry/reporting schema, per architecture doc §2.6.

**`add_transcript_event` call sites deliberately use the safe-by-default
metadata-only path** (no ``include_raw_content=True``, no ``sanitizer``).
This is a conscious choice, not an oversight: per "bring your own agent"
(see `dscraft.agent.adapter`), ``agent_fn`` is an arbitrary
caller-supplied callable, and its proposed command's real stdout/stderr
(captured in the "tool" trajectory step) can contain whatever that
command actually printed -- which, for a real agent (not just this
package's synthetic `rule_based_agent` reference), could include secrets,
credentials, or other sensitive tool output. Exporting that verbatim into
an OTel span by default would be exactly the credential/PII-leak risk
`dscraft.core.telemetry.add_transcript_event`'s safe-by-default contract
exists to prevent. A caller who wants full transcript content in their own
exported traces can wrap/re-emit these spans with ``include_raw_content=True``
or a ``sanitizer`` at their own telemetry layer; this package does not
opt in on their behalf.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from dscraft.core.sandbox import BaseSandboxExecutor, get_default_executor
from dscraft.core.telemetry import add_transcript_event, genai_span, set_ml_metric

from dscraft.agent.adapter import (
    AgentFn,
    AgentTrajectory,
    SandboxedAgentAdapter,
    TaskResult,
    TaskSpec,
)
from dscraft.agent.tasks import score_file_task

__all__ = ["BenchmarkReport", "ScorerFn", "run_benchmark", "run_task"]

#: A scorer inspects the task and its recorded trajectory (which includes
#: the real `dscraft.core.sandbox.SandboxResult`) and returns
#: ``(success, human_readable_detail)``. `dscraft.agent.tasks.score_file_task`
#: is the one reference scorer this package ships, matching the one task
#: family it implements.
ScorerFn = Callable[[TaskSpec, AgentTrajectory], tuple[bool, str]]


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate result of running a small task suite through the benchmark loop."""

    results: tuple[TaskResult, ...]
    pass_rate: float
    mean_latency_seconds: float

    @property
    def task_count(self) -> int:
        """Number of tasks included in this report (``len(self.results)``)."""
        return len(self.results)


def run_task(
    task: TaskSpec,
    *,
    agent_fn: AgentFn,
    executor: BaseSandboxExecutor,
    scorer: ScorerFn = score_file_task,
) -> TaskResult:
    """Run one task through a fresh :class:`SandboxedAgentAdapter` and score it.

    Latency is measured via ``time.perf_counter()`` around the adapter's
    ``run_task`` call only (the actual agent-decide + sandboxed-execute
    work), not around scoring/telemetry overhead.

    This is the **one** per-task exception boundary for the whole benchmark
    loop (`run_benchmark` deliberately does not add a second one around its
    call to this function -- see that function's docstring). Any exception
    raised by ``agent_fn`` (invoked indirectly via ``adapter.run_task``),
    by ``adapter.run_task`` itself (e.g. a sandbox executor raising
    `dscraft.core.sandbox.SandboxPolicyViolationError` for fail-fast policy
    violations, or `dscraft.core.sandbox.SandboxBackendUnavailableError`), or
    by ``scorer`` is caught here and converted into a failed
    :class:`TaskResult` rather than propagating out and aborting every
    other task in the suite. ``KeyboardInterrupt``/``SystemExit`` are
    intentionally *not* caught (``except Exception``, not
    ``except BaseException``) -- a user's Ctrl-C or an explicit process
    exit should still stop the run.

    On such a failure, ``latency_seconds`` records wall-clock time up to
    the point of failure (not ``NaN``/``0``) so `run_benchmark`'s mean-
    latency aggregation stays a meaningful number rather than needing
    special-casing, and ``trajectory`` is ``None`` if the failure happened
    before ``adapter.run_task`` returned one (e.g. ``agent_fn`` itself
    raised), or the real trajectory if only ``scorer`` raised afterwards.
    """
    adapter = SandboxedAgentAdapter(agent_fn)

    start = time.perf_counter()
    trajectory: AgentTrajectory | None = None
    try:
        trajectory = adapter.run_task(task, executor)
        latency_seconds = time.perf_counter() - start
        success, detail = scorer(task, trajectory)
    except Exception as exc:  # noqa: BLE001 - deliberate: one task's failure must not crash the suite.
        latency_seconds = time.perf_counter() - start
        success = False
        detail = f"task raised {type(exc).__name__}: {exc}"

        with genai_span(
            f"lazyagent.task.{task.name}",
            attributes={"lazyagent.task.name": task.name},
        ) as span:
            set_ml_metric(span, "accuracy", 0.0)
            span.set_attribute("lazyagent.task.latency_seconds", latency_seconds)
            span.set_attribute("lazyagent.task.detail", detail)
            span.set_attribute("lazyagent.task.error", True)
            if trajectory is not None:
                for step in trajectory.steps:
                    add_transcript_event(span, step.role, step.content)
            add_transcript_event(span, "error", detail)

        return TaskResult(
            task_name=task.name,
            success=False,
            latency_seconds=latency_seconds,
            trajectory=trajectory,
            detail=detail,
        )

    with genai_span(
        f"lazyagent.task.{task.name}",
        attributes={"lazyagent.task.name": task.name},
    ) as span:
        set_ml_metric(span, "accuracy", 1.0 if success else 0.0)
        span.set_attribute("lazyagent.task.latency_seconds", latency_seconds)
        span.set_attribute("lazyagent.task.detail", detail)
        for step in trajectory.steps:
            add_transcript_event(span, step.role, step.content)

    return TaskResult(
        task_name=task.name,
        success=success,
        latency_seconds=latency_seconds,
        trajectory=trajectory,
        detail=detail,
    )


def run_benchmark(
    tasks: Sequence[TaskSpec],
    agent_fn: AgentFn,
    *,
    executor: BaseSandboxExecutor | None = None,
    scorer: ScorerFn = score_file_task,
) -> BenchmarkReport:
    """Run ``tasks`` through ``agent_fn`` and report an aggregate pass rate + mean latency.

    Args:
        tasks: the small, fixed task suite to run (e.g.
            `dscraft.agent.tasks.default_task_suite`).
        agent_fn: the bring-your-own-agent callable (see
            `dscraft.agent.adapter.AgentFn`).
        executor: a `dscraft.core.sandbox.BaseSandboxExecutor`; defaults to
            `dscraft.core.sandbox.get_default_executor()` (Seatbelt on macOS).
        scorer: how to score each task's trajectory; defaults to
            `dscraft.agent.tasks.score_file_task`, matching the one
            task family this package implements.

    A task whose ``agent_fn``/``adapter.run_task``/``scorer`` raises does
    not abort the run: `run_task` is the one per-task exception boundary
    (see its docstring) and converts such a failure into a failed
    :class:`~dscraft.agent.adapter.TaskResult` before it ever
    reaches this loop, so this function never needs a second try/except
    around its call to `run_task`. That failed result's ``success=False``
    is counted in ``pass_rate``'s denominator like any other failure, and
    its ``latency_seconds`` (wall-clock time up to the point of failure)
    is included in ``mean_latency_seconds`` like any other task's --
    deliberately not excluded or treated as ``0``/``NaN``, so the
    aggregate mean latency stays a single well-defined number regardless
    of how many tasks in the suite raised.
    """
    if not tasks:
        raise ValueError("run_benchmark requires at least one task")

    active_executor = executor or get_default_executor()

    results = tuple(
        run_task(task, agent_fn=agent_fn, executor=active_executor, scorer=scorer)
        for task in tasks
    )

    pass_rate = sum(1 for r in results if r.success) / len(results)
    mean_latency_seconds = sum(r.latency_seconds for r in results) / len(results)

    with genai_span("lazyagent.benchmark.run") as span:
        set_ml_metric(span, "accuracy", pass_rate)
        span.set_attribute("lazyagent.benchmark.task_count", len(results))
        span.set_attribute("lazyagent.benchmark.mean_latency_seconds", mean_latency_seconds)

    return BenchmarkReport(
        results=results,
        pass_rate=pass_rate,
        mean_latency_seconds=mean_latency_seconds,
    )
