"""Tests for dscraft.core.adapter -- the shared BaseSandboxedAdapter base
class both ``dscraft.agent.adapter.AgentAdapter`` and
``dscraft.security.adapter.BaseSecurityAdapter`` build on (architecture doc
§2.3's "one ... adapter base class" requirement).

``BaseSandboxedAdapter`` itself is deliberately thin (no state, no abstract
methods of its own -- see its docstring), so these tests focus on the two
things that are genuinely worth asserting in isolation: that both module
adapter bases really do inherit from it (proving the "one shared base"
architecture decision is actually satisfied in code, not just in
docstrings), and that its one piece of shared logic --
``_require_sandbox_executor`` -- behaves correctly on both valid and
invalid input, independent of either concrete module.
"""

from __future__ import annotations

import pytest

from dscraft.core.adapter import BaseSandboxedAdapter
from dscraft.core.sandbox import BaseSandboxExecutor, SandboxResult


class _StubExecutor(BaseSandboxExecutor):
    """Minimal real ``BaseSandboxExecutor`` subclass, used only to exercise
    ``_require_sandbox_executor`` against a genuine (if trivial) executor
    instance without depending on any real sandbox backend being available
    on this host."""

    def is_available(self) -> bool:
        return True

    def run_command(self, command, *, policy=None):
        return SandboxResult(exit_code=0, stdout="", stderr="")

    def run_callable(self, func, *, policy=None):
        return SandboxResult(exit_code=0, stdout=repr(func()), stderr="")


def test_base_sandboxed_adapter_is_instantiable_without_abstract_methods():
    """``BaseSandboxedAdapter`` is a plain class (not an ``abc.ABC``) and
    defines no abstract methods of its own, so nothing blocks direct
    instantiation. It exists purely as a shared base for concrete adapter
    hierarchies and is never used as an adapter in its own right in
    practice, but nothing prevents direct instantiation."""
    adapter = BaseSandboxedAdapter()
    assert isinstance(adapter, BaseSandboxedAdapter)


def test_agent_adapter_subclasses_shared_base():
    """``dscraft.agent.adapter.AgentAdapter`` really inherits from the one
    shared base, satisfying §2.3's "one ... adapter base class" decision in
    code rather than just in the two modules' docstrings."""
    from dscraft.agent.adapter import AgentAdapter

    assert issubclass(AgentAdapter, BaseSandboxedAdapter)


def test_security_adapter_subclasses_shared_base():
    """``dscraft.security.adapter.BaseSecurityAdapter`` really inherits
    from the same shared base as ``AgentAdapter``."""
    from dscraft.security.adapter import BaseSecurityAdapter

    assert issubclass(BaseSecurityAdapter, BaseSandboxedAdapter)


def test_require_sandbox_executor_accepts_real_executor():
    """A genuine ``BaseSandboxExecutor`` instance passes validation silently."""
    BaseSandboxedAdapter._require_sandbox_executor(_StubExecutor())


@pytest.mark.parametrize("bad_value", [None, "not-an-executor", object(), 42])
def test_require_sandbox_executor_rejects_non_executor(bad_value):
    """Anything that is not a ``BaseSandboxExecutor`` instance raises
    ``TypeError`` with a message naming the expected type -- this is the
    one piece of logic genuinely shared between ``AgentAdapter`` and
    ``BaseSecurityAdapter`` subclasses, both of which call this helper at
    the point they are handed a caller-supplied executor."""
    with pytest.raises(TypeError, match="BaseSandboxExecutor"):
        BaseSandboxedAdapter._require_sandbox_executor(bad_value)
