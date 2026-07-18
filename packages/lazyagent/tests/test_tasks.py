"""Tests for benchcraft_lazyagent.tasks -- the file-manipulation task family.

The key acceptance criterion these tests demonstrate: the fail-designed
(sandbox-escape-attempt) task is genuinely blocked by the real sandbox
backend -- not merely scored as a failure by convention. The forbidden
file must not exist on disk after the run.

Only the tests that actually run a task through the sandbox need a real,
available backend -- those take the ``executor`` fixture below, which
skips gracefully if none is available rather than gating the whole module
on ``platform.system()``. Pure validation/logic tests further down (e.g.
`test_default_task_suite_has_one_pass_and_one_fail_designed_task`,
`test_score_file_task_rejects_non_file_task_spec`, and the fresh-workspace
regression tests) never touch the executor and run on any platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lazycore.sandbox import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    get_default_executor,
)
from benchcraft_lazyagent.adapter import SandboxedAgentAdapter
from benchcraft_lazyagent.tasks import (
    default_task_suite,
    make_fail_task,
    make_pass_task,
    rule_based_agent,
    score_file_task,
)


@pytest.fixture()
def executor() -> BaseSandboxExecutor:
    """The real sandbox executor for this host, or skip if unavailable."""
    try:
        ex = get_default_executor()
    except SandboxBackendUnavailableError as exc:
        pytest.skip(f"no real sandbox backend available on this host: {exc}")
    if not ex.is_available():
        pytest.skip("resolved sandbox backend reports itself unavailable on this host")
    return ex


def test_pass_task_creates_expected_file_and_scores_success(executor, tmp_path: Path):
    """Running the pass-designed task end-to-end creates the target file
    with the exact expected content, and `score_file_task` scores it as a
    success."""
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)
    success, detail = score_file_task(task, trajectory)

    assert success is True, detail
    target = Path(task.target_path)
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == task.expected_content


def test_fail_task_is_genuinely_blocked_by_the_sandbox_not_just_scored_false(
    executor, tmp_path: Path
):
    """The escape-attempt task's target file must not exist at all -- proving
    the shared lazycore.sandbox executor's write-path allowlist actually
    prevented the write, rather than the agent simply not trying."""
    task = make_fail_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)
    success, detail = score_file_task(task, trajectory)

    # The sandbox should report the command as blocked/failed.
    assert trajectory.sandbox_result.exit_code != 0

    # The scored outcome must be failure...
    assert success is False, detail

    # ...and the containment must be real: no file was created anywhere,
    # not even the forbidden target path.
    forbidden_target = Path(task.target_path)
    assert not forbidden_target.exists()
    assert not forbidden_target.parent.exists(), (
        "the sandbox should have blocked mkdir -p on the forbidden parent "
        "directory as well, since it is outside allowed_write_paths"
    )


def test_default_task_suite_has_one_pass_and_one_fail_designed_task(tmp_path: Path):
    """`default_task_suite` always returns exactly two tasks: one with
    `expect_success=True` and one with `expect_success=False`."""
    suite = default_task_suite(tmp_path)

    assert len(suite) == 2
    assert sum(1 for t in suite if t.expect_success) == 1
    assert sum(1 for t in suite if not t.expect_success) == 1


def test_score_file_task_rejects_non_file_task_spec():
    """`score_file_task` raises TypeError when given a plain `TaskSpec`
    rather than the `FileTaskSpec` subclass it requires to read
    `target_path`/`expected_content` off of."""
    from benchcraft_lazyagent.adapter import AgentTrajectory, TaskSpec
    from lazycore.sandbox import SandboxPolicy, SandboxResult

    bad_task = TaskSpec(name="x", description="x", sandbox_policy=SandboxPolicy())
    trajectory = AgentTrajectory(
        task_name="x",
        steps=(),
        sandbox_result=SandboxResult(exit_code=0, stdout="", stderr=""),
    )

    with pytest.raises(TypeError):
        score_file_task(bad_task, trajectory)


def test_make_pass_task_does_not_reuse_a_stale_workspace_across_calls(tmp_path: Path):
    """Regression test (CodeRabbit): re-invoking `make_pass_task` with the
    same ``base_dir``/``name`` must not reuse a prior task's workspace
    directory, since `score_file_task` scores by trusting real filesystem
    state -- a stale file left over from an earlier run could otherwise
    make a fresh run falsely "pass" without this run's agent ever having
    created it.

    Pure filesystem/logic test -- never touches the sandbox executor, so it
    runs on any platform.
    """
    task1 = make_pass_task(tmp_path, name="dup")

    # Pre-pollute what a stale prior run might have left behind: write the
    # exact expected content WITHOUT ever running the agent or the sandbox.
    Path(task1.target_path).write_text(task1.expected_content, encoding="utf-8")
    assert Path(task1.target_path).is_file()

    # A second call with the *same* base_dir/name must get its own fresh
    # workspace, not task1's polluted one.
    task2 = make_pass_task(tmp_path, name="dup")

    assert task2.target_path != task1.target_path, (
        "make_pass_task must allocate a fresh, unique workspace per call, "
        "even when base_dir/name are reused"
    )
    assert not Path(task2.target_path).exists(), (
        "a fresh task instance must not inherit a stale file from a "
        "previous run with the same base_dir/name -- this is exactly the "
        "false-pass scenario the fix guards against"
    )


def test_make_fail_task_does_not_reuse_a_stale_workspace_across_calls(tmp_path: Path):
    """Same regression as above, for `make_fail_task`: a stale ``forbidden/``
    directory or file left over from a prior run must not linger into a
    fresh task instance, since that could mask a real containment
    regression (score_file_task would then be checking old evidence of a
    block, not this run's)."""
    task1 = make_fail_task(tmp_path, name="dup_fail")

    # Simulate a stale escape file left behind (e.g. from a run against an
    # unpatched/misconfigured sandbox in the past).
    Path(task1.target_path).parent.mkdir(parents=True, exist_ok=True)
    Path(task1.target_path).write_text(task1.expected_content, encoding="utf-8")
    assert Path(task1.target_path).is_file()

    task2 = make_fail_task(tmp_path, name="dup_fail")

    assert task2.target_path != task1.target_path
    assert not Path(task2.target_path).exists()
    assert not Path(task2.target_path).parent.exists(), (
        "a fresh fail-task instance must not inherit a stale forbidden/ "
        "directory from a previous run with the same base_dir/name"
    )
