"""The one concrete benchmark task family: file-manipulation tool-use.

Per README "Scope", this pass implements exactly one task family -- create
a file with expected content in a sandboxed working directory -- rather
than a general task-description schema or a SWE-bench-style suite. Two
task variants are provided:

- :func:`make_pass_task` -- the agent's write target is inside the
  sandbox's ``allowed_write_paths``. The rule-based reference agent
  (:func:`rule_based_agent`) issues a shell command to create the file, the
  shared `lazycore.sandbox` executor allows it, and the task scores as a
  success.
- :func:`make_fail_task` -- the agent (deliberately, to exercise
  containment) attempts to write *outside* ``allowed_write_paths``. The
  shared sandbox executor's default-deny write policy blocks this, the
  file is never created, and the task scores as a failure -- proving the
  sandbox's containment genuinely drives the scored outcome, not just
  decoration around it.

:func:`rule_based_agent` is the one reference "agent" callable this package
ships: a plain, deterministic Python function (not an LLM, not a real
framework) that reads a :class:`~benchcraft_lazyagent.adapter.FileTaskSpec`
and decides on a shell command satisfying (or, for the fail variant,
attempting to satisfy) the task description. Its signature matches
:data:`~benchcraft_lazyagent.adapter.AgentFn` exactly, so a caller could
swap in a real framework's decision function without changing
`benchcraft_lazyagent.adapter` or `benchcraft_lazyagent.benchmark` at all.
"""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lazycore.sandbox import SandboxPolicy

from benchcraft_lazyagent.adapter import AgentAction, AgentTrajectory, TaskSpec

__all__ = [
    "FileTaskSpec",
    "default_task_suite",
    "make_fail_task",
    "make_pass_task",
    "rule_based_agent",
    "score_file_task",
]


@dataclass(frozen=True)
class FileTaskSpec(TaskSpec):
    """A :class:`TaskSpec` for the file-manipulation task family.

    Attributes:
        target_path: Absolute path of the file the agent is asked to
            create. For :func:`make_pass_task` this is inside the sandbox
            policy's ``allowed_write_paths``; for :func:`make_fail_task` it
            is deliberately outside that allowlist.
        expected_content: Exact file content required for the task to
            score as a success.
    """

    target_path: str = ""
    expected_content: str = ""


def rule_based_agent(task: TaskSpec) -> AgentAction:
    """The one reference "bring your own agent" callable this package ships.

    Deliberately simple and deterministic: it does not parse
    ``task.description`` with any NLP -- it reads the structured fields off
    a :class:`FileTaskSpec` directly and always proposes the same kind of
    shell command (``mkdir -p`` the parent directory, then write the exact
    expected content via ``printf``). This is intentional -- the point of
    this scaffold is to exercise the sandbox-wiring and scoring loop, not
    to build a capable agent. Whether the command actually succeeds is
    entirely up to the sandbox policy the task executes under (see
    ``benchcraft_lazyagent.tasks`` module docstring).

    Raises:
        TypeError: if ``task`` is not a :class:`FileTaskSpec` -- this
            reference agent only knows how to handle this one task family.
    """
    if not isinstance(task, FileTaskSpec):
        raise TypeError(
            f"rule_based_agent only supports FileTaskSpec tasks, got {type(task).__name__}"
        )

    quoted_content = shlex.quote(task.expected_content)
    quoted_path = shlex.quote(task.target_path)
    parent_dir = shlex.quote(str(Path(task.target_path).parent))
    shell_script = f"mkdir -p {parent_dir} && printf '%s' {quoted_content} > {quoted_path}"

    return AgentAction(
        command=["/bin/sh", "-c", shell_script],
        rationale=f"write expected content to {task.target_path!r} via a shell one-liner",
    )


def score_file_task(task: TaskSpec, trajectory: AgentTrajectory) -> tuple[bool, str]:
    """Score a :class:`FileTaskSpec` run by checking the real filesystem.

    Success requires the target file to actually exist *and* contain
    exactly ``task.expected_content`` -- deliberately re-checking the real
    filesystem rather than trusting the sandbox's reported exit code, so
    that a sandbox-blocked write (nonzero exit, or a zero exit that still
    didn't actually create the file) is scored as a failure regardless of
    what the agent or shell reported.
    """
    if not isinstance(task, FileTaskSpec):
        raise TypeError(f"score_file_task only supports FileTaskSpec, got {type(task).__name__}")

    path = Path(task.target_path)
    if not path.is_file():
        return False, f"expected file {task.target_path!r} does not exist"

    actual_content = path.read_text(encoding="utf-8")
    if actual_content != task.expected_content:
        return False, (
            f"file {task.target_path!r} exists but content mismatch: "
            f"expected {task.expected_content!r}, got {actual_content!r}"
        )

    return True, f"file {task.target_path!r} created with expected content"


def _fresh_task_root(base_dir: Path, name: str) -> Path:
    """Allocate a genuinely fresh, unique task-scoped directory under ``base_dir``.

    Regression fix (CodeRabbit): the previous implementation derived the
    task root deterministically as ``base_dir / name``. Re-invoking a
    factory (:func:`make_pass_task`/:func:`make_fail_task`) with the same
    ``base_dir``/``name`` pair -- e.g. re-running a benchmark suite against
    a persistent workspace directory instead of a brand-new
    `tempfile.TemporaryDirectory` each time -- would silently reuse
    whatever directory (and any stale files inside it) a prior run left
    behind. Since :func:`score_file_task` scores by trusting real
    filesystem state, a stale target file from a previous run could make a
    task falsely "pass" without this run's agent ever having created it,
    and could also mask a real containment regression in the fail-designed
    task (a stale ``forbidden/`` directory from an old, unpatched sandbox
    run could linger even after containment is fixed).

    ``tempfile.mkdtemp`` guarantees a new, unique directory on every call
    (even for repeated ``base_dir``/``name`` pairs), so each task instance
    always starts from a genuinely empty workspace.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=str(base_dir)))


def make_pass_task(base_dir: Path, *, name: str = "create_file_pass") -> FileTaskSpec:
    """A task the sandbox should allow: write inside ``allowed_write_paths``.

    Args:
        base_dir: A directory this task family creates its own
            task-scoped subdirectory under (the caller typically passes a
            fresh `tempfile.TemporaryDirectory` path). Each call gets a
            fresh, unique subdirectory (see :func:`_fresh_task_root`) even
            if ``base_dir``/``name`` are reused across calls.
    """
    task_root = _fresh_task_root(base_dir, name)
    workspace = task_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    target_path = workspace / "hello.txt"
    expected_content = "hello from benchcraft lazyagent"

    policy = SandboxPolicy(
        allow_network=False,
        allowed_write_paths=(str(workspace),),
        working_directory=str(workspace),
        timeout_seconds=15.0,
    )

    return FileTaskSpec(
        name=name,
        description=(
            f"Create a file named {target_path.name!r} with content "
            f"{expected_content!r} in the sandboxed working directory."
        ),
        sandbox_policy=policy,
        expect_success=True,
        target_path=str(target_path),
        expected_content=expected_content,
    )


def make_fail_task(base_dir: Path, *, name: str = "create_file_escape_fail") -> FileTaskSpec:
    """A task designed to fail: the target path is outside ``allowed_write_paths``.

    This exists specifically to prove sandbox containment is exercised and
    genuinely affects the scored outcome (README acceptance criteria): the
    rule-based agent attempts to write to ``forbidden/escape.txt``, a
    sibling directory of the sandbox's allowed workspace that is *not*
    included in ``allowed_write_paths``. The shared sandbox executor's
    default-deny write policy should block the write, the file should
    never be created, and :func:`score_file_task` should therefore score
    this task as a failure. Like :func:`make_pass_task`, each call gets a
    fresh, unique task root (see :func:`_fresh_task_root`) so a stale
    ``forbidden/`` directory from a previous run can never linger into
    this run's containment check.
    """
    task_root = _fresh_task_root(base_dir, name)
    workspace = task_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    forbidden_dir = task_root / "forbidden"  # deliberately NOT in allowed_write_paths

    target_path = forbidden_dir / "escape.txt"
    expected_content = "this should never be written"

    policy = SandboxPolicy(
        allow_network=False,
        allowed_write_paths=(str(workspace),),  # forbidden_dir is intentionally excluded
        working_directory=str(workspace),
        timeout_seconds=15.0,
    )

    return FileTaskSpec(
        name=name,
        description=(
            f"Create a file at {str(target_path)!r} with content "
            f"{expected_content!r} -- outside the sandboxed working "
            "directory (a sandbox-escape attempt used to prove containment)."
        ),
        sandbox_policy=policy,
        expect_success=False,
        target_path=str(target_path),
        expected_content=expected_content,
    )


def default_task_suite(base_dir: Path) -> list[FileTaskSpec]:
    """The small, fixed set of task variants the benchmark runner exercises by default."""
    return [make_pass_task(base_dir), make_fail_task(base_dir)]
