"""Bring-your-own-agent ``AgentAdapter`` pattern (architecture doc Part 3, "Module 8: LazyAgent").

The architecture doc's LazyAgent survey names MASEval's "framework-agnostic
``AgentAdapter`` interface spanning smolagents/LangGraph/AutoGen/CAMEL" as
the design this module should implement a Bring-Your-Own-Agent version of.
Per §2.8 ("no LLM router, no multi-provider abstraction... each module
takes a bare-minimum 'bring your own local model handle' approach"),
"bring your own agent" here means concretely: :class:`AgentAdapter` accepts
any Python callable matching the documented :data:`AgentFn` signature as
"the agent" -- it is *not* a registry/router of supported agent frameworks.
A caller could later plug in a real framework's decision function (e.g. a
thin wrapper around a smolagents/LangGraph step) without changing this
package's core loop at all.

Nothing here duplicates the shared sandbox executor (`dscraft.core.sandbox`) --
:class:`AgentAdapter` always executes the agent's chosen action *through*
a caller-supplied `dscraft.core.sandbox.BaseSandboxExecutor` instance. This
module only adds the LazyAgent-specific machinery layered on top: the task/
trajectory/result data shapes, and the adapter that wires an arbitrary
agent callable to the shared executor. Per §2.3, LazyAgent does not get its
own executor class -- only its own mode-specific `SandboxPolicy` values
(see `dscraft.agent.tasks`).

:class:`AgentAdapter` itself subclasses the one shared
`dscraft.core.adapter.BaseSandboxedAdapter` base (also per §2.3's "one ...
adapter base class" requirement) rather than defining an independent `abc.ABC`
hierarchy -- see that module's docstring for what is actually shared between
this and `dscraft.security.adapter.BaseSecurityAdapter`.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from dscraft.core.adapter import BaseSandboxedAdapter
from dscraft.core.sandbox import BaseSandboxExecutor, SandboxPolicy, SandboxResult

__all__ = [
    "AgentAction",
    "AgentAdapter",
    "AgentFn",
    "AgentTrajectory",
    "SandboxedAgentAdapter",
    "TaskResult",
    "TaskSpec",
    "TrajectoryStep",
]


@dataclass(frozen=True)
class TaskSpec:
    """A single benchmark task the agent is asked to complete.

    Deliberately minimal at this scaffold's depth (see README "Scope") --
    one concrete task family is implemented in ``dscraft.agent.tasks``
    (file-manipulation tool-use), not a general task-description schema for
    arbitrary tool-use suites (SWE-bench-style task suites are explicitly
    out of scope for this pass).

    Attributes:
        name: Short, unique task identifier (used in reports/telemetry).
        description: Natural-language task description handed to the agent
            callable -- e.g. "create a file named out.txt with content
            'hello' in the sandboxed working directory".
        sandbox_policy: The :class:`~dscraft.core.sandbox.SandboxPolicy` this
            task's action is executed under. LazyAgent's own mode-specific
            policy values (default-deny egress, restrictive write paths)
            are constructed by ``dscraft.agent.tasks``, not by
            `dscraft.core` itself -- see that module and the README for why.
        expect_success: Whether this task variant is *designed* to succeed
            (``True``) or *designed* to fail via sandbox containment
            (``False``) -- used only for documentation/test clarity; the
            actual pass/fail determination always comes from ``scorer``.
    """

    name: str
    description: str
    sandbox_policy: SandboxPolicy
    expect_success: bool = True


@dataclass(frozen=True)
class AgentAction:
    """What the agent callable decided to do for a given :class:`TaskSpec`.

    At this scaffold's depth the only supported action shape is a single
    shell command (argv-style, matching
    :meth:`~dscraft.core.sandbox.BaseSandboxExecutor.run_command`'s contract) --
    a real framework-agnostic adapter would eventually need to support
    multi-step tool-call sequences, but a single action per task is enough
    to exercise the sandbox-wiring and scoring loop that is this pass's
    signature capability.
    """

    command: Sequence[str]
    rationale: str = ""


#: The "bring your own agent" plug-in point. A caller supplies any Python
#: callable matching this signature -- it receives the task and must return
#: the action it wants executed. `dscraft.agent.tasks.rule_based_agent`
#: is the one reference implementation provided by this package (a plain,
#: deterministic, rule-based function -- not a real LLM-backed agent); a
#: real framework's step function could be adapted to this exact signature
#: without touching `AgentAdapter` itself.
AgentFn = Callable[[TaskSpec], AgentAction]


@dataclass(frozen=True)
class TrajectoryStep:
    """One recorded step of an agent's execution trajectory.

    Mirrors the shape `dscraft.core.telemetry.add_transcript_event` expects
    (a ``role`` + ``content`` pair), so trajectories reported by this
    module reuse the exact same OTel transcript-event convention LazyRed
    uses for its own conversational transcripts (architecture doc §2.6).
    """

    role: str
    content: str


@dataclass(frozen=True)
class AgentTrajectory:
    """Everything the agent "did" for one task: its steps plus the sandboxed result."""

    task_name: str
    steps: tuple[TrajectoryStep, ...]
    sandbox_result: SandboxResult


@dataclass(frozen=True)
class TaskResult:
    """The scored outcome of running one :class:`TaskSpec` through an adapter.

    ``trajectory`` is ``None`` exactly when the task never produced one --
    i.e. ``agent_fn``/``adapter.run_task`` raised before returning an
    :class:`AgentTrajectory` at all. `dscraft.agent.benchmark.run_task`
    catches such exceptions at a per-task boundary (so one task's crash
    never aborts the whole benchmark run) and reports them here as a
    failed result: ``success=False``, ``latency_seconds`` set to the
    wall-clock time elapsed up to the point of failure, ``trajectory=None``,
    and the exception's type/message captured in ``detail`` -- the same
    field already used for a scorer's human-readable pass/fail explanation,
    rather than inventing a second, parallel error-reporting field.
    """

    task_name: str
    success: bool
    latency_seconds: float
    trajectory: AgentTrajectory | None
    detail: str = ""


class AgentAdapter(BaseSandboxedAdapter, abc.ABC):
    """Minimal bring-your-own-agent interface.

    ``run_task`` is the one canonical entrypoint (per CLAUDE.md's "one
    canonical adapter interface" rule) -- concrete subclasses decide *how*
    the agent's chosen action actually gets executed. This pass provides
    exactly one concrete implementation, :class:`SandboxedAgentAdapter`,
    which always executes through the shared `dscraft.core.sandbox` executor.

    Subclasses `dscraft.core.adapter.BaseSandboxedAdapter`, the one shared
    adapter base LazyAgent and LazyRed both build on (§2.3) -- only the
    task/trajectory/result data shapes and the ``run_task`` abstract method
    below are LazyAgent-specific. `BaseSandboxedAdapter` itself is no longer
    an `abc.ABC` (it defines no abstract methods of its own), so this class
    adds `abc.ABC` directly alongside it to keep ``run_task``'s
    ``@abc.abstractmethod`` decorator actually enforced (i.e. this class,
    like `BaseSandboxedAdapter`'s previous behavior, still cannot be
    instantiated directly).
    """

    @abc.abstractmethod
    def run_task(self, task: TaskSpec, executor: BaseSandboxExecutor) -> AgentTrajectory:
        """Run ``task`` via this adapter's agent, executing inside ``executor``.

        Args:
            task: the task to attempt.
            executor: a `dscraft.core.sandbox.BaseSandboxExecutor` instance
                (e.g. `dscraft.core.sandbox.get_default_executor()`) that the
                adapter must use for any containment of the agent's chosen
                action -- this package never builds a second sandbox
                mechanism.
        """
        raise NotImplementedError


class SandboxedAgentAdapter(AgentAdapter):
    """The one concrete :class:`AgentAdapter`: wires an arbitrary agent
    callable's chosen action through the shared sandbox executor.

    Construction takes ``agent_fn`` (see :data:`AgentFn`) -- this is the
    "bring your own agent" seam. This package supplies exactly one
    reference ``agent_fn`` (``dscraft.agent.tasks.rule_based_agent``)
    but any callable with the same signature works, including one that
    wraps a real agent framework's decision step.
    """

    def __init__(self, agent_fn: AgentFn) -> None:
        """Wrap ``agent_fn`` -- the "bring your own agent" callable to drive.

        Args:
            agent_fn: a callable matching :data:`AgentFn` (task in, action
                out). Stored as-is; not validated or wrapped further, so any
                exception it raises during :meth:`run_task` propagates
                unchanged to the caller (`dscraft.agent.benchmark.run_task`
                is what converts such exceptions into a failed result).
        """
        self._agent_fn = agent_fn

    def run_task(self, task: TaskSpec, executor: BaseSandboxExecutor) -> AgentTrajectory:
        """Ask ``agent_fn`` for one action, run it in ``executor``, and record the trajectory.

        Builds a three-step :class:`AgentTrajectory` -- a ``user`` step with
        the task description, an ``assistant`` step with the chosen action's
        command/rationale, and a ``tool`` step with the real
        `dscraft.core.sandbox.SandboxResult` (exit code, whether the policy
        blocked it, stdout/stderr) -- regardless of whether the sandboxed
        command actually succeeded. Scoring the outcome is the caller's
        responsibility (see `dscraft.agent.tasks.score_file_task`),
        not this method's.

        Args:
            task: the task to attempt.
            executor: the sandbox executor to run the agent's chosen
                command through, per :meth:`AgentAdapter.run_task`'s
                contract.
        """
        self._require_sandbox_executor(executor)

        steps: list[TrajectoryStep] = [
            TrajectoryStep(role="user", content=task.description)
        ]

        action = self._agent_fn(task)
        steps.append(
            TrajectoryStep(
                role="assistant",
                content=f"command={list(action.command)!r} rationale={action.rationale!r}",
            )
        )

        result = executor.run_command(action.command, policy=task.sandbox_policy)
        steps.append(
            TrajectoryStep(
                role="tool",
                content=(
                    f"exit_code={result.exit_code} policy_blocked={result.policy_blocked} "
                    f"stdout={result.stdout!r} stderr={result.stderr!r}"
                ),
            )
        )

        return AgentTrajectory(
            task_name=task.name,
            steps=tuple(steps),
            sandbox_result=result,
        )
