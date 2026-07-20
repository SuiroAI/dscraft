"""Tests for dscraft.core.sandbox.base (shared executor interface, §2.3)."""

from __future__ import annotations

import dataclasses

import pytest

from dscraft.core.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxError,
    SandboxPolicy,
    SandboxPolicyViolationError,
    SandboxResult,
)


def test_sandbox_policy_defaults_are_maximally_restrictive():
    """A default-constructed SandboxPolicy denies network, reads, writes, and execs.

    This only checks the dataclass field *values* -- it does not exercise
    real Seatbelt enforcement, since this module is platform-agnostic and
    has no `sandbox-exec` dependency. For a real, effective-profile-behavior
    proof that an empty `allowed_read_paths` genuinely denies reads outside
    a narrow bootstrap set (rather than granting the whole filesystem, per
    the CodeRabbit Finding-1 review), see
    `test_default_deny_reads_and_safe_callable.
    test_empty_allowed_read_paths_blocks_read_of_real_out_of_bootstrap_file`
    -- a real `sandbox-exec` invocation, not a string-inspection test.
    """
    policy = SandboxPolicy()
    assert policy.allow_network is False
    assert policy.allowed_read_paths == ()
    assert policy.allowed_write_paths == ()
    assert policy.allowed_executables == ()
    assert policy.env == {}
    assert policy.inherit_env is False
    assert policy.timeout_seconds is None


def test_sandbox_policy_is_frozen():
    """Mutating a field on a constructed SandboxPolicy raises, per its frozen=True dataclass."""
    policy = SandboxPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.allow_network = True  # type: ignore[misc]


def test_sandbox_policy_env_is_immutable_and_isolated_from_the_original_dict():
    """policy.env is coerced to an immutable mapping at construction time: mutating the original dict passed in afterward does not change the effective policy, and attempting to mutate policy.env directly raises rather than silently succeeding.

    This is the regression test for the CodeRabbit finding that
    SandboxPolicy.env could still be mutated in place after construction
    despite the dataclass being frozen=True -- frozen dataclasses only
    block *field reassignment* (policy.env = {...}), not in-place mutation
    of a mutable object a field happens to point at (policy.env["x"] = "y").
    """
    original = {"FOO": "bar"}
    policy = SandboxPolicy(env=original)

    # Mutating the original dict object passed in must not affect the
    # policy -- proves a defensive copy was taken, not just a reference.
    original["FOO"] = "mutated-after-construction"
    original["NEW_KEY"] = "should-not-appear"
    assert policy.env["FOO"] == "bar"
    assert "NEW_KEY" not in policy.env

    # Attempting to mutate policy.env directly must raise, not silently
    # succeed -- this is the core of the finding: a frozen dataclass whose
    # mutable-typed field can still be mutated in place defeats the
    # frozen=True promise.
    with pytest.raises(TypeError):
        policy.env["FOO"] = "attempted-mutation"  # type: ignore[index]

    # MappingProxyType doesn't even expose .update() -- it isn't a mutable
    # mapping at all, so this raises AttributeError rather than TypeError.
    with pytest.raises(AttributeError):
        policy.env.update({"BAZ": "qux"})  # type: ignore[union-attr]

    # Read access still works exactly like a plain dict.
    assert dict(policy.env) == {"FOO": "bar"}
    assert policy.env["FOO"] == "bar"


def test_sandbox_policy_env_default_is_immutable_too():
    """Even the default (no env passed) SandboxPolicy().env is the same immutable-mapping type, not a plain mutable dict -- the coercion in __post_init__ applies unconditionally, not only when a caller passes one."""
    policy = SandboxPolicy()
    with pytest.raises(TypeError):
        policy.env["X"] = "y"  # type: ignore[index]


def test_sandbox_policy_with_overrides_returns_new_instance():
    """with_overrides() returns a distinct copy with only the given fields changed, leaving the original untouched."""
    base = SandboxPolicy(allow_network=False)
    modified = base.with_overrides(allow_network=True)

    assert base.allow_network is False
    assert modified.allow_network is True
    assert modified is not base


def test_sandbox_policy_generic_enough_for_two_different_mode_configs():
    """The same SandboxPolicy dataclass can express both a LazyRed-style target policy and a differently-valued LazyAgent-style benchmark policy, with no module-specific fields."""
    # Simulates a LazyRed-style "red-team target sandbox" policy and a
    # LazyAgent-style "benchmark task sandbox" policy both being built from
    # the exact same SandboxPolicy dataclass with different values -- per
    # §2.3's "mode-specific policy configs layered on top" framing. Neither
    # module-specific logic nor field is hardcoded into the dataclass.
    lazyred_style = SandboxPolicy(
        allow_network=False,
        allowed_read_paths=("/tmp/redteam-target",),
        allowed_write_paths=(),
        allowed_executables=("/usr/bin/python3",),
    )
    lazyagent_style = SandboxPolicy(
        allow_network=True,
        allowed_read_paths=("/tmp/benchmark-workspace",),
        allowed_write_paths=("/tmp/benchmark-workspace/output",),
        allowed_executables=(),
    )

    assert lazyred_style.allow_network is False
    assert lazyagent_style.allow_network is True
    assert type(lazyred_style) is type(lazyagent_style) is SandboxPolicy


def test_sandbox_result_succeeded_true_only_when_clean_exit():
    """succeeded is True only for exit_code == 0 AND policy_blocked is False; either a nonzero exit or a blocked-but-zero-exit result is unsuccessful."""
    ok = SandboxResult(exit_code=0, stdout="hi", stderr="")
    assert ok.succeeded is True

    bad_exit = SandboxResult(exit_code=1, stdout="", stderr="boom")
    assert bad_exit.succeeded is False

    blocked_but_zero_exit = SandboxResult(
        exit_code=0, stdout="", stderr="", policy_blocked=True
    )
    assert blocked_but_zero_exit.succeeded is False


def test_sandbox_result_is_frozen():
    """Mutating a field on a constructed SandboxResult raises, per its frozen=True dataclass."""
    result = SandboxResult(exit_code=0, stdout="", stderr="")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.exit_code = 1  # type: ignore[misc]


def test_base_sandbox_executor_cannot_be_instantiated_directly():
    """BaseSandboxExecutor is an ABC and cannot be instantiated without implementing its abstract methods."""
    with pytest.raises(TypeError):
        BaseSandboxExecutor()  # type: ignore[abstract]


def test_base_sandbox_executor_subclass_must_implement_full_interface():
    """A subclass implementing only is_available() (not run_command/run_callable) still cannot be instantiated."""
    class Incomplete(BaseSandboxExecutor):
        def is_available(self) -> bool:
            return True

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_minimal_concrete_executor_tracks_configured_policy():
    """A full concrete executor exposes the policy passed at construction via .policy, and .configure() swaps it for a new one."""
    class NoOpExecutor(BaseSandboxExecutor):
        def is_available(self) -> bool:
            return True

        def run_command(self, command, *, policy=None):
            return SandboxResult(exit_code=0, stdout="", stderr="")

        def run_callable(self, func, *, policy=None):
            return SandboxResult(exit_code=0, stdout="", stderr="")

    default_policy = SandboxPolicy()
    executor = NoOpExecutor(default_policy)
    assert executor.policy is default_policy

    new_policy = SandboxPolicy(allow_network=True)
    executor.configure(new_policy)
    assert executor.policy is new_policy


def test_exception_hierarchy():
    """SandboxBackendUnavailableError and SandboxPolicyViolationError are both SandboxError subclasses, which is itself a RuntimeError."""
    assert issubclass(SandboxBackendUnavailableError, SandboxError)
    assert issubclass(SandboxPolicyViolationError, SandboxError)
    assert issubclass(SandboxError, RuntimeError)
