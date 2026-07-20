"""DSCraft LazyAgent -- bring-your-own-agent task-execution benchmark loop.

Implements exactly one signature capability from the architecture doc's
"Module 8: LazyAgent" (see `DSCraft_Unified_Architecture.md`, Part 3):
a minimal, real, bring-your-own-agent task-execution benchmark loop where a
plain Python callable (standing in for a real framework-agnostic agent, per
MASEval's `AgentAdapter` interface) executes a small file-manipulation
tool-use task inside the shared `dscraft.core.sandbox` executor, and the run is
scored for success/failure with accuracy/latency metrics reported via
`dscraft.core.telemetry`.

See this package's README for full scope, the sandbox policy used, and
everything explicitly deferred (real agent framework integrations, the
Multi-Objective Pareto RAG Optimization loop, DISCO-style sample
condensation, SWE-bench-style task suites).
"""

from __future__ import annotations

from dscraft.agent.adapter import (
    AgentAction,
    AgentAdapter,
    AgentFn,
    AgentTrajectory,
    SandboxedAgentAdapter,
    TaskResult,
    TaskSpec,
    TrajectoryStep,
)
from dscraft.agent.benchmark import (
    BenchmarkReport,
    ScorerFn,
    run_benchmark,
    run_task,
)
from dscraft.agent.tasks import (
    FileTaskSpec,
    default_task_suite,
    make_fail_task,
    make_pass_task,
    rule_based_agent,
    score_file_task,
)

__all__ = [
    # Alphabetically sorted (ruff RUF022) across adapter.py/benchmark.py/
    # tasks.py -- see the import blocks above for which module each name
    # actually comes from.
    "AgentAction",
    "AgentAdapter",
    "AgentFn",
    "AgentTrajectory",
    "BenchmarkReport",
    "FileTaskSpec",
    "SandboxedAgentAdapter",
    "ScorerFn",
    "TaskResult",
    "TaskSpec",
    "TrajectoryStep",
    "default_task_suite",
    "make_fail_task",
    "make_pass_task",
    "rule_based_agent",
    "run_benchmark",
    "run_task",
    "score_file_task",
]
