"""Tests for dscraft.agent.adapter -- the AgentAdapter/SandboxedAgentAdapter pattern.

These tests exercise the real sandbox backend
(`dscraft.core.sandbox.get_default_executor()`) rather than mocking the
sandbox (see README "Sandbox wiring"). Every test in this module genuinely
needs a real, available sandbox backend to run an agent action, so the
``executor`` fixture below -- not a module-wide OS check -- is what decides
whether to skip: it skips gracefully (`pytest.skip`) if
`dscraft.core.sandbox.SandboxBackendUnavailableError` is raised or the resolved
executor reports itself unavailable, rather than gating on
``platform.system()``. That keeps the skip reason tied to actual backend
capability instead of host OS, and doesn't accidentally skip any
non-sandbox-dependent test that might be added to this module later.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dscraft.core.sandbox import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    get_default_executor,
)
from dscraft.agent.adapter import (
    AgentAction,
    AgentTrajectory,
    SandboxedAgentAdapter,
    TaskSpec,
)
from dscraft.agent.tasks import make_pass_task, rule_based_agent


@pytest.fixture()
def executor() -> BaseSandboxExecutor:
    """The real sandbox executor for this host, or skip if unavailable.

    Resolves via `dscraft.core.sandbox.get_default_executor()` and skips this
    test (rather than failing, and rather than gating the whole module on
    ``platform.system()``) if no real backend is usable here --
    `SandboxBackendUnavailableError` (e.g. non-macOS/non-Linux host, or
    macOS without ``sandbox-exec``) or an executor that resolves but
    reports ``is_available() is False``.
    """
    try:
        ex = get_default_executor()
    except SandboxBackendUnavailableError as exc:
        pytest.skip(f"no real sandbox backend available on this host: {exc}")
    if not ex.is_available():
        pytest.skip("resolved sandbox backend reports itself unavailable on this host")
    return ex


def test_sandboxed_agent_adapter_returns_trajectory_with_three_steps(executor, tmp_path: Path):
    """A run produces exactly one user/assistant/tool step, in that order."""
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)

    assert isinstance(trajectory, AgentTrajectory)
    assert trajectory.task_name == task.name
    assert len(trajectory.steps) == 3
    assert trajectory.steps[0].role == "user"
    assert trajectory.steps[1].role == "assistant"
    assert trajectory.steps[2].role == "tool"


def test_sandboxed_agent_adapter_actually_executes_via_sandbox_result(executor, tmp_path: Path):
    """A pass-designed task's command is really executed by the sandbox: it
    exits 0, is not policy-blocked, and the target file actually exists on
    disk afterward -- not just a stubbed/fabricated result."""
    task = make_pass_task(tmp_path)
    adapter = SandboxedAgentAdapter(rule_based_agent)

    trajectory = adapter.run_task(task, executor)

    # The sandbox result is real: exit_code/stdout/stderr came from an
    # actual subprocess.run() call inside SeatbeltSandboxExecutor, not a
    # stub -- a successful file-creation command in an allowed path exits 0.
    assert trajectory.sandbox_result.exit_code == 0
    assert trajectory.sandbox_result.policy_blocked is False
    assert Path(task.target_path).is_file()


def test_agent_fn_plug_in_point_accepts_any_matching_callable(executor, tmp_path: Path):
    """A caller-supplied callable matching AgentFn's signature works without
    subclassing anything -- this is the "bring your own agent" contract."""
    task = make_pass_task(tmp_path)

    def custom_agent(t: TaskSpec) -> AgentAction:
        # Deliberately different rationale/behavior from rule_based_agent,
        # to prove the adapter doesn't hardcode a specific agent.
        return AgentAction(command=["/bin/sh", "-c", "true"], rationale="no-op stand-in agent")

    adapter = SandboxedAgentAdapter(custom_agent)
    trajectory = adapter.run_task(task, executor)

    assert trajectory.sandbox_result.exit_code == 0
    assert "no-op stand-in agent" in trajectory.steps[1].content
