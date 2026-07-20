"""Linux namespace-based sandbox backend -- DOCUMENTED STUB, not a real
implementation (architecture doc §2.3).

Per §2.3, Linux is DSCraft's secondary v1 platform, and the "existing
options from prior research apply unchanged" there: gVisor/runsc,
Firecracker microVMs, Docker/runc, or lightweight namespace-based
approaches (the "SWE-MiniSandbox" pattern using `unshare`/`bwrap`-style
primitives) are all viable, genuinely Linux-native isolation mechanisms,
because -- unlike macOS -- Linux has no VM-boundary GPU passthrough problem
to design around in the first place (§2.3.1).

**This class is intentionally not a real backend.** The reference
development machine for this repository is macOS (Apple Silicon), per
`CLAUDE.md` and the architecture doc's stated reference hardware. A real
namespace/gVisor/Firecracker-based executor cannot be meaningfully built
*or verified* from this machine -- there is no Linux kernel here to test
namespace isolation against, and shipping untested isolation code as if it
were real would be worse than not shipping it. This class exists only to:

1. Satisfy :class:`~dscraft.core.sandbox.base.BaseSandboxExecutor`'s interface,
   so callers can reference it and `get_default_executor()` can dispatch to
   it on Linux without an import error.
2. Honestly report, via :meth:`is_available`, that it is never usable --
   always ``False``, on every platform, regardless of whether a namespace-
   sandboxing helper (`unshare`/`bwrap` or similar) happens to be present.
   Presence of such a helper says something about the *host*, not about
   this *stub*, which never execs those helpers and has no real backend
   behind it to report as available.
3. Raise a clear, documented error on any actual use, rather than silently
   pretending to sandbox anything.

Building the real Linux backend is future work for whoever has an actual
Linux dev/CI environment to validate it against -- do not fill in this
stub with unverified logic just to make it "look done."
"""

from __future__ import annotations

import platform
import shutil
from typing import Callable, Sequence

from dscraft.core.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxPolicy,
    SandboxResult,
)

__all__ = ["LinuxNamespaceSandboxExecutor"]

#: Helper binaries a real namespace-based backend would depend on
#: (bubblewrap's `bwrap`, or raw `unshare` from util-linux). Documented here
#: for whoever implements the real backend -- **not** consulted by
#: :meth:`LinuxNamespaceSandboxExecutor.is_available`, which always returns
#: ``False`` regardless of helper presence, since this stub never execs
#: anything and has no real availability to report. An earlier version of
#: this module used this tuple (via ``shutil.which``) to make
#: ``is_available()`` return ``True`` when a helper was found on ``PATH``;
#: that was itself the bug this stub now avoids -- see
#: :meth:`LinuxNamespaceSandboxExecutor.is_available`'s docstring.
_LINUX_SANDBOX_HELPERS = ("bwrap", "unshare")


class LinuxNamespaceSandboxExecutor(BaseSandboxExecutor):
    """Stub for the Linux-native namespace/gVisor/Firecracker backend.

    Every method that would need to actually sandbox something raises
    :class:`~dscraft.core.sandbox.base.SandboxBackendUnavailableError` -- this
    class never fabricates sandboxing behavior. See the module docstring
    for why this is a deliberate stub rather than a real implementation.
    """

    def is_available(self) -> bool:
        """Always False -- this backend is a documented stub, not a real one.

        Per :meth:`~dscraft.core.sandbox.base.BaseSandboxExecutor.is_available`'s
        contract, ``True`` is supposed to mean "this backend can actually
        run on this host". That is never true here: every actual execution
        method (:meth:`run_command`, :meth:`run_callable`) unconditionally
        raises :class:`SandboxBackendUnavailableError` regardless of what
        this method reports, because no real namespace/gVisor/Firecracker
        backend has been implemented (see the module docstring for why).
        Presence of a helper binary such as ``bwrap``/``unshare`` on
        ``PATH`` says something about whether the *host* could plausibly
        run a real backend some day -- it says nothing about whether *this
        stub* can, since this stub never execs those helpers at all. An
        earlier version of this method returned ``True`` when such a helper
        was found, which broke the ``is_available()`` API contract: a
        caller that checks ``is_available()`` before deciding whether to
        run something would get a false "yes" on a Linux host with
        ``bwrap`` installed, then crash on the very next call with
        :class:`SandboxBackendUnavailableError`. Always returning ``False``
        here makes this class honest about being unusable, on every
        platform, until a real backend is implemented.
        """
        return False

    def _unavailable(self) -> SandboxBackendUnavailableError:
        return SandboxBackendUnavailableError(
            "LinuxNamespaceSandboxExecutor is a documented stub, not a "
            "real sandbox backend. Per architecture doc §2.3, the intended "
            "real Linux backend is a namespace-based isolation mechanism "
            "(gVisor/Firecracker/unshare+bwrap-style, per prior sandbox "
            "research) -- it was deliberately not implemented here because "
            "this repository's reference/dev environment is macOS "
            f"(detected platform: {platform.system()!r}), and namespace "
            "isolation cannot be meaningfully built or verified without a "
            "real Linux kernel to test against. Implement and validate "
            "this backend from an actual Linux environment before relying "
            "on it."
        )

    def run_command(
        self,
        command: Sequence[str],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Always raise :class:`SandboxBackendUnavailableError`.

        This stub never executes ``command`` under any isolation -- see the
        module docstring for why a real namespace-based implementation
        isn't provided here.
        """
        raise self._unavailable()

    def run_callable(
        self,
        func: Callable[..., object],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Always raise :class:`SandboxBackendUnavailableError`.

        This stub never executes ``func`` under any isolation -- see the
        module docstring for why a real namespace-based implementation
        isn't provided here.
        """
        raise self._unavailable()
