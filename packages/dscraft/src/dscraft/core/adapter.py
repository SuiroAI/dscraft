"""Shared adapter base for the sandbox-executing adapter pattern (architecture
doc §2.3).

Per §2.3: "Sandbox execution is shared between LazyRed and LazyAgent (one
executor + adapter base class, mode-specific policies on top) -- both
contain the same kernel-level threat class (arbitrary code execution).
LazyRed's semantic-level threats (prompt injection, credential leakage)
are handled by a *separate* Guardrail/Firewall layer, not the sandbox."

Before this module existed, `dscraft.agent.adapter.AgentAdapter` (LazyAgent)
and `dscraft.security.adapter.BaseSecurityAdapter` (LazyRed) each
independently defined their own `abc.ABC` adapter hierarchy -- a deviation
from the locked "one ... adapter base class" decision above. This module is
that one shared base class: :class:`BaseSandboxedAdapter`. Each module's own
adapter base now subclasses it and adds only what is genuinely
module-specific on top:

- `dscraft.agent.adapter.AgentAdapter` adds the ``run_task`` abstract method
  and the task/trajectory/result data shapes.
- `dscraft.security.adapter.BaseSecurityAdapter` adds the
  ``generate_attempt``/``run_target``/``detect`` abstract methods, the
  concrete ``run()`` convenience chain, and the attempt/finding data shapes.

Neither module's own mode-specific `~dscraft.core.sandbox.SandboxPolicy`
construction (`dscraft.agent.tasks`, `dscraft.security.probes`) moves here --
per §2.3, that stays "layered on top" in each module, not centralized.
Likewise, none of LazyRed's semantic-level probe/detector logic (prompt
injection triggers, secret-leak detection) lives here or in this class --
that separation is exactly what §2.3's last sentence, quoted above,
requires this module to preserve.
"""

from __future__ import annotations

from dscraft.core.sandbox import BaseSandboxExecutor

__all__ = ["BaseSandboxedAdapter"]


class BaseSandboxedAdapter:
    """Common ancestor for adapters that execute untrusted work through a
    caller-supplied `dscraft.core.sandbox.BaseSandboxExecutor`.

    Carries no state of its own and defines no abstract methods -- per
    §2.3, `dscraft.core` supplies one *executor* interface
    (`~dscraft.core.sandbox.BaseSandboxExecutor`) and one generic *policy*
    dataclass (`~dscraft.core.sandbox.SandboxPolicy`); it deliberately does
    not also prescribe a single canonical *method signature* here, because
    `AgentAdapter.run_task(task, executor)` and `BaseSecurityAdapter`'s
    ``generate_attempt``/``run_target``/``detect`` pipeline have genuinely
    different, module-specific shapes (task/trajectory/result vs.
    probe/attempt/finding) that this class does not try to force into one
    mold.

    What both modules' adapters do genuinely share, and what this class
    exists to make structurally explicit (one shared base) rather than
    duplicated as two parallel, independently-written `abc.ABC`
    definitions, is:

    1. Being under the same lineage -- `AgentAdapter` and
       `BaseSecurityAdapter` are now both `BaseSandboxedAdapter` subclasses,
       matching the locked "one ... adapter base class" architecture
       decision (§2.3) instead of two independent hierarchies that happen
       to look similar. `BaseSandboxedAdapter` itself is a plain class, not
       an `abc.ABC` -- it defines no abstract methods of its own, so
       `abc.ABC` here would not enforce anything (Ruff B024). Each
       subclass that does have its own abstract methods (both currently
       do) adds `abc.ABC` itself alongside this base.
    2. The contract that any executable action a concrete subclass performs
       must be routed through a `BaseSandboxExecutor` instance supplied by
       the *caller* -- never a sandbox mechanism the adapter constructs
       itself. :meth:`_require_sandbox_executor` is a small, genuinely
       shared defensive helper concrete adapters use to enforce exactly
       that at the one point both module's adapters take an executor
       argument, rather than each module re-implementing the same type
       check (or omitting it) independently.
    """

    @staticmethod
    def _require_sandbox_executor(executor: BaseSandboxExecutor) -> None:
        """Raise ``TypeError`` if ``executor`` is not a `BaseSandboxExecutor`.

        Both `~dscraft.agent.adapter.SandboxedAgentAdapter.run_task` and
        `~dscraft.security.adapter.BaseSecurityAdapter.run` accept an
        executor argument at the exact point where they are about to run
        untrusted, potentially attacker-influenced work, so both benefit
        from failing fast with one explicit, actionable error here rather
        than a confusing `AttributeError` several stack frames deeper
        (e.g. inside a concrete executor's ``run_command``/``run_callable``)
        if a caller passes ``None`` or the wrong type by mistake.

        Args:
            executor: the value a caller supplied where a
                `BaseSandboxExecutor` instance is required.

        Raises:
            TypeError: if ``executor`` is not a `BaseSandboxExecutor`
                instance.
        """
        if not isinstance(executor, BaseSandboxExecutor):
            raise TypeError(
                "expected a dscraft.core.sandbox.BaseSandboxExecutor "
                f"instance, got {executor!r} ({type(executor).__name__})"
            )
