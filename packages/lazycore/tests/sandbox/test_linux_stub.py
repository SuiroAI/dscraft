"""Tests for lazycore.sandbox.linux_stub.LinuxNamespaceSandboxExecutor.

This backend is a documented stub (see its module docstring for why): this
development machine is macOS, so most of these tests confirm the stub
behaves correctly *as a stub* regardless of host platform (either by being
genuinely platform-independent, or by monkeypatching ``platform.system()``
so the "on Linux" branch of the stub's own logic is actually exercised
here) -- they do not, and cannot, validate real Linux namespace isolation.

Two categories of platform-sensitive test, per CodeRabbit's review:

- Tests that assert something true only about *this specific host's* real
  OS (e.g. "is_available() is False on an actual non-Linux machine") are
  marked with ``@pytest.mark.skipif`` so they run only where the assertion
  is actually meaningful, instead of hardcoding "not Linux" as if that were
  a universal truth.
- Tests that are really unit tests of the *stub's own dispatch logic*
  (e.g. "does is_available()/get_default_executor() correctly special-case
  Linux") use ``monkeypatch.setattr(platform, "system", ...)`` so the
  Linux-specific code path is exercised and verified on any host, including
  this macOS development machine.
"""

from __future__ import annotations

import platform

import pytest

from lazycore.sandbox.base import SandboxBackendUnavailableError
from lazycore.sandbox.linux_stub import LinuxNamespaceSandboxExecutor


@pytest.mark.skipif(
    platform.system() == "Linux",
    reason=(
        "This test asserts real non-Linux host behavior (is_available() "
        "is False because platform.system() != 'Linux' on the actual "
        "host); it is redundant, though not wrong, on a real Linux host, "
        "where is_available() is also always False -- this stub is never "
        "usable regardless of platform. See "
        "test_is_available_is_always_false_on_linux_regardless_of_helper_presence "
        "(monkeypatched) for the platform-independent unit-level check that "
        "covers the Linux branch specifically, including both the "
        "helper-absent and helper-present cases."
    ),
)
def test_is_available_is_false_on_a_real_non_linux_machine():
    """is_available() returns False on a real (unpatched) non-Linux host, since the stub only ever reports availability on Linux."""
    executor = LinuxNamespaceSandboxExecutor()
    assert executor.is_available() is False


def test_is_available_is_false_when_platform_is_not_linux(monkeypatch):
    """is_available() returns False whenever platform.system() reports something other than "Linux", verified by monkeypatching platform.system() so this holds regardless of the real host OS."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    executor = LinuxNamespaceSandboxExecutor()
    assert executor.is_available() is False


def test_is_available_is_always_false_on_linux_regardless_of_helper_presence(monkeypatch):
    """When platform.system() reports "Linux", is_available() is always False -- both when no namespace-sandboxing helper (bwrap/unshare) is present on PATH, and when one is -- because this backend is an intentional, documented, unimplemented stub: every actual execution method (run_command/run_callable) always raises SandboxBackendUnavailableError regardless of what is_available() reports, so helper presence alone must never make is_available() report True. Exercised here via monkeypatching both platform.system() and shutil.which() so this passes on any host, including this macOS machine."""
    import lazycore.sandbox.linux_stub as linux_stub_module

    monkeypatch.setattr(platform, "system", lambda: "Linux")

    # No helper present -> not available.
    monkeypatch.setattr(linux_stub_module.shutil, "which", lambda tool: None)
    executor = LinuxNamespaceSandboxExecutor()
    assert executor.is_available() is False

    # A helper present -> still not available: presence of bwrap/unshare on
    # PATH says something about the host, not about this stub, which never
    # execs those helpers and has no real backend behind it.
    monkeypatch.setattr(
        linux_stub_module.shutil,
        "which",
        lambda tool: f"/usr/bin/{tool}" if tool == "bwrap" else None,
    )
    assert executor.is_available() is False


def test_run_command_raises_documented_unavailable_error():
    """run_command() always raises SandboxBackendUnavailableError mentioning it is a documented stub, never fabricating a sandboxed run."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError, match="documented stub"):
        executor.run_command(["echo", "hello"])


def test_run_callable_raises_documented_unavailable_error():
    """run_callable() always raises SandboxBackendUnavailableError mentioning it is a documented stub, never fabricating a sandboxed run."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError, match="documented stub"):
        executor.run_callable(lambda: 1)


def test_stub_can_be_instantiated_without_raising():
    """Instantiating LinuxNamespaceSandboxExecutor itself never raises -- only calling run_command/run_callable does."""
    # Instantiation itself must not raise -- only actual use should.
    executor = LinuxNamespaceSandboxExecutor()
    assert isinstance(executor, LinuxNamespaceSandboxExecutor)


def test_error_message_names_the_intended_real_backend_family():
    """The SandboxBackendUnavailableError message mentions both "namespace" and "linux" so a caller understands what real backend is intended."""
    executor = LinuxNamespaceSandboxExecutor()
    with pytest.raises(SandboxBackendUnavailableError) as excinfo:
        executor.run_command(["true"])
    message = str(excinfo.value)
    assert "namespace" in message.lower()
    assert "linux" in message.lower()


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason=(
        "This test asserts real macOS-host behavior (get_default_executor() "
        "dispatches to SeatbeltSandboxExecutor on the actual host platform); "
        "it is not meaningful, and would legitimately fail, on a real Linux "
        "host, where get_default_executor() correctly dispatches to "
        "LinuxNamespaceSandboxExecutor instead. See "
        "test_get_default_executor_dispatches_to_linux_stub_when_platform_is_linux "
        "(monkeypatched) for the platform-independent unit-level check of "
        "that dispatch logic."
    ),
)
def test_get_default_executor_returns_seatbelt_not_linux_stub_on_this_machine():
    """On a real macOS host, get_default_executor() dispatches to SeatbeltSandboxExecutor, never to the Linux stub."""
    from lazycore.sandbox import SeatbeltSandboxExecutor, get_default_executor

    executor = get_default_executor()
    assert isinstance(executor, SeatbeltSandboxExecutor)
    assert not isinstance(executor, LinuxNamespaceSandboxExecutor)


def test_get_default_executor_dispatches_to_linux_stub_when_platform_is_linux(monkeypatch):
    """get_default_executor() dispatches to LinuxNamespaceSandboxExecutor whenever platform.system() reports "Linux" -- verified by monkeypatching platform.system() (as referenced directly in lazycore.sandbox's own get_default_executor()) so this passes on any host, including this macOS machine."""
    import lazycore.sandbox as sandbox_package
    from lazycore.sandbox import get_default_executor

    monkeypatch.setattr(sandbox_package.platform, "system", lambda: "Linux")

    executor = get_default_executor()
    assert isinstance(executor, LinuxNamespaceSandboxExecutor)
