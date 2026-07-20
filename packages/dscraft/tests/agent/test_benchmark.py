"""Tests for dscraft.agent.benchmark -- the tiny multi-task benchmark runner."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dscraft.core.sandbox import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    get_default_executor,
)
from dscraft.agent.adapter import AgentAction, TaskSpec
from dscraft.agent.benchmark import BenchmarkReport, run_benchmark
from dscraft.agent.tasks import default_task_suite, make_pass_task, rule_based_agent


@pytest.fixture()
def executor() -> BaseSandboxExecutor:
    """The real sandbox executor for this host, or skip if unavailable.

    Only the tests that actually run a task through `run_benchmark` need a
    real backend. `test_run_benchmark_rejects_empty_task_list` below
    returns before executor selection ever happens (see
    `benchmark.run_benchmark`'s empty-task-list guard), so it doesn't take
    this fixture and runs unconditionally on every platform -- it was
    previously, incorrectly, skipped on non-Darwin CI by this module's old
    OS-wide `pytestmark`.
    """
    try:
        ex = get_default_executor()
    except SandboxBackendUnavailableError as exc:
        pytest.skip(f"no real sandbox backend available on this host: {exc}")
    if not ex.is_available():
        pytest.skip("resolved sandbox backend reports itself unavailable on this host")
    return ex


def test_run_benchmark_rejects_empty_task_list():
    """`run_benchmark` raises ValueError instead of dividing by zero when
    given an empty task suite (pass_rate/mean_latency would otherwise
    divide by len(results) == 0). This never touches the sandbox executor
    (the empty-list check happens before executor selection), so it runs
    on any platform regardless of sandbox backend availability."""
    with pytest.raises(ValueError):
        run_benchmark([], rule_based_agent)


def test_run_benchmark_aggregates_pass_rate_and_mean_latency(executor, tmp_path: Path):
    """The default two-task suite (one designed to pass, one to fail)
    yields a 50% pass rate, correctly attributes which named task passed
    vs. failed, and reports real, finite, positive latencies for every
    task and for the aggregate mean."""
    suite = default_task_suite(tmp_path)  # one pass-designed, one fail-designed

    report = run_benchmark(suite, rule_based_agent, executor=executor)

    assert isinstance(report, BenchmarkReport)
    assert report.task_count == 2

    # Exactly one of the two fixed default tasks is designed to succeed.
    assert report.pass_rate == pytest.approx(0.5)

    successes = [r for r in report.results if r.success]
    failures = [r for r in report.results if not r.success]
    assert len(successes) == 1
    assert len(failures) == 1
    assert successes[0].task_name == "create_file_pass"
    assert failures[0].task_name == "create_file_escape_fail"

    # Latency must be real: measured via time.perf_counter(), so it should
    # be a small positive, finite number for both tasks -- not zero, not
    # NaN/inf, and not a mocked/fabricated constant.
    assert math.isfinite(report.mean_latency_seconds)
    assert report.mean_latency_seconds > 0.0
    for result in report.results:
        assert math.isfinite(result.latency_seconds)
        assert result.latency_seconds > 0.0


def test_run_benchmark_with_only_passing_tasks_reports_perfect_pass_rate(executor, tmp_path: Path):
    """A suite made entirely of pass-designed tasks reports a 100% pass
    rate and every individual result marked successful."""
    from dscraft.agent.tasks import make_pass_task

    tasks = [
        make_pass_task(tmp_path, name="pass_a"),
        make_pass_task(tmp_path, name="pass_b"),
    ]

    report = run_benchmark(tasks, rule_based_agent, executor=executor)

    assert report.pass_rate == pytest.approx(1.0)
    assert all(r.success for r in report.results)


def test_run_benchmark_isolates_a_task_whose_agent_fn_raises(executor, tmp_path: Path):
    """Regression test for the per-task exception boundary.

    A 3-task suite where the *middle* task's ``agent_fn`` deliberately
    raises a plain ``RuntimeError`` (standing in for e.g. a caller's custom
    agent throwing on an unexpected task type, or a sandbox executor
    surfacing `dscraft.core.sandbox.SandboxPolicyViolationError`). Before the
    fix, this exception propagated out of `run_benchmark` entirely, so a
    10-task suite with one bad task returned zero results and a crash
    instead of a partial report. This asserts the fixed behavior instead:
    the run completes for the whole suite, the failing task is reported as
    a single failed `TaskResult` with the exception captured, and the
    *other* tasks in the suite still ran and scored correctly -- one bad
    task doesn't skip or corrupt its siblings' results.
    """
    task_a = make_pass_task(tmp_path, name="pass_before")
    task_b = make_pass_task(tmp_path, name="raises_in_the_middle")
    task_c = make_pass_task(tmp_path, name="pass_after")

    def agent_fn_that_raises_for_one_task(task: TaskSpec) -> AgentAction:
        if task.name == "raises_in_the_middle":
            raise RuntimeError("simulated agent_fn crash on an unexpected task")
        return rule_based_agent(task)

    report = run_benchmark(
        [task_a, task_b, task_c], agent_fn_that_raises_for_one_task, executor=executor
    )

    # The run completed for the whole suite -- no exception propagated, and
    # no task was silently dropped.
    assert isinstance(report, BenchmarkReport)
    assert report.task_count == 3

    results_by_name = {r.task_name: r for r in report.results}
    assert set(results_by_name) == {"pass_before", "raises_in_the_middle", "pass_after"}

    # The failing task is recorded as a failed TaskResult with the
    # exception's type/message captured in `detail`, and no trajectory
    # (agent_fn raised before the adapter could produce one).
    failed = results_by_name["raises_in_the_middle"]
    assert failed.success is False
    assert failed.trajectory is None
    assert "RuntimeError" in failed.detail
    assert "simulated agent_fn crash on an unexpected task" in failed.detail
    # Latency is still a real, finite wall-clock measurement up to the
    # point of failure -- not NaN, not a sentinel/fabricated 0.
    assert math.isfinite(failed.latency_seconds)
    assert failed.latency_seconds >= 0.0

    # The sibling tasks ran independently and scored correctly -- their
    # results are untouched by the one bad task.
    for name in ("pass_before", "pass_after"):
        sibling = results_by_name[name]
        assert sibling.success is True
        assert sibling.trajectory is not None
        assert math.isfinite(sibling.latency_seconds)
        assert sibling.latency_seconds > 0.0

    # Aggregation correctly counts the exception-failure in the pass-rate
    # denominator (2 real passes out of 3 total tasks) and folds its
    # latency into the mean like any other task's.
    assert report.pass_rate == pytest.approx(2 / 3)
    assert math.isfinite(report.mean_latency_seconds)
    assert report.mean_latency_seconds > 0.0
