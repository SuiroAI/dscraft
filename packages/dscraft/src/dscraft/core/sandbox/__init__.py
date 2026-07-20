"""Shared sandbox executor + adapter base class (architecture doc §2.3, §2.3.1).

Both LazyRed and LazyAgent contain the same kernel-level threat class --
arbitrary code execution by a red-team target or a benchmarked agent -- so
per §2.3, dscraft.core provides **one** shared sandbox executor interface and
**one** generic policy dataclass, with mode-specific policy *values*
layered on top by each module (not built here; see :class:`SandboxPolicy`).

Per §2.3.1's locked split-trust architecture, GPU-bound model inference is
never run inside any sandbox provided by this subpackage -- only the
CPU-bound tool-calling/code-execution layer (shell commands, file I/O,
network egress) is ever constrained. See :mod:`dscraft.core.sandbox.seatbelt`'s
module docstring for the specific technical reason this is also not merely
a policy choice but a hard technical limit on macOS.

Public API:

- :class:`SandboxPolicy` -- generic policy configuration.
- :class:`SandboxResult` -- structured result of a sandboxed run.
- :class:`BaseSandboxExecutor` -- the shared executor ABC.
- :class:`SeatbeltSandboxExecutor` -- real macOS backend (`sandbox-exec`).
- :class:`LinuxNamespaceSandboxExecutor` -- documented stub for the real
  Linux backend (namespaces/gVisor/Firecracker); not implemented here.
- :func:`get_default_executor` -- picks the right backend for this host.
- :class:`SandboxError`, :class:`SandboxBackendUnavailableError`,
  :class:`SandboxPolicyViolationError` -- exception types.
"""

from __future__ import annotations

import platform

from dscraft.core.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxError,
    SandboxPolicy,
    SandboxPolicyViolationError,
    SandboxResult,
)
from dscraft.core.sandbox.linux_stub import LinuxNamespaceSandboxExecutor
from dscraft.core.sandbox.seatbelt import SeatbeltSandboxExecutor

__all__ = [
    "BaseSandboxExecutor",
    "SandboxPolicy",
    "SandboxResult",
    "SandboxError",
    "SandboxBackendUnavailableError",
    "SandboxPolicyViolationError",
    "SeatbeltSandboxExecutor",
    "LinuxNamespaceSandboxExecutor",
    "get_default_executor",
]


def get_default_executor(policy: SandboxPolicy | None = None) -> BaseSandboxExecutor:
    """Return the appropriate :class:`BaseSandboxExecutor` for this host.

    - macOS with ``/usr/bin/sandbox-exec`` present ->
      :class:`SeatbeltSandboxExecutor` (real, functional backend).
    - Linux -> :class:`LinuxNamespaceSandboxExecutor` (documented stub --
      instantiable, but every actual run method raises
      :class:`SandboxBackendUnavailableError`; see that class's docstring).
    - Anything else (Windows, unrecognized platforms) -> raises
      :class:`SandboxBackendUnavailableError` immediately, since no backend
      in this subpackage targets those platforms.
    """
    system = platform.system()

    if system == "Darwin":
        executor = SeatbeltSandboxExecutor(policy)
        if executor.is_available():
            return executor
        raise SandboxBackendUnavailableError(
            "Detected macOS but /usr/bin/sandbox-exec is not present or "
            "usable on this host; no other macOS sandbox backend is "
            "implemented in dscraft.core.sandbox."
        )

    if system == "Linux":
        # Returned even though it's a stub, per the documented contract:
        # callers get a real BaseSandboxExecutor instance whose run_*
        # methods raise a clear, actionable error -- not a fabricated
        # sandboxing implementation.
        return LinuxNamespaceSandboxExecutor(policy)

    raise SandboxBackendUnavailableError(
        f"No sandbox backend is available for platform {system!r}. "
        "dscraft.core.sandbox only implements macOS (Seatbelt) and a "
        "documented Linux stub."
    )
