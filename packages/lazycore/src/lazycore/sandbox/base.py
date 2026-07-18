"""Shared sandbox executor interface (architecture doc §2.3, §2.3.1).

This module defines the *shared* abstraction both LazyRed and LazyAgent
build mode-specific policies on top of. Per §2.3, the two modules contain
the same kernel-level threat class -- arbitrary code execution by an agent
or red-team target -- so LazyCore provides one executor interface and one
generic policy dataclass, not two parallel implementations. Mode-specific
*policy values* (what a "LazyRed target sandbox" or a "LazyAgent benchmark
task sandbox" actually allows) are each module's job when it is built;
this module only defines the generic shape those policies share.

**The GPU/split-trust boundary (§2.3.1) is load-bearing here.** Research
found that no Mac container/VM boundary exposes Metal/MPS passthrough, and
that Seatbelt (`sandbox-exec`) *cannot* mediate GPU/Metal/Cocoa access even
in principle -- "GPU and display passthrough flags have no effect on macOS
because Metal and Cocoa are system-level and cannot be blocked via SBPL."
The locked v1 recommendation is therefore a **split-trust architecture**:
GPU-bound model inference always runs unsandboxed (there is no adversarial
surface there -- it's local weights and forward-pass compute), and only the
CPU-bound tool-calling/code-execution layer is sandboxed. Every executor in
this package -- including this base class -- exists to constrain *that*
layer only. Do not add, or ask an implementation to add, any logic that
attempts to gate GPU/Metal/Cocoa access; per §2.3.1 this is both technically
impossible via Seatbelt and out of scope by design.
"""

from __future__ import annotations

import abc
import dataclasses
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


__all__ = [
    "SandboxPolicy",
    "SandboxResult",
    "BaseSandboxExecutor",
    "SandboxError",
    "SandboxBackendUnavailableError",
    "SandboxPolicyViolationError",
]


class SandboxError(RuntimeError):
    """Base class for all sandbox-executor errors."""


class SandboxBackendUnavailableError(SandboxError):
    """Raised when a requested sandbox backend cannot actually be used on
    this host (e.g. the Linux-namespace backend invoked on macOS, or a
    required helper binary such as ``bwrap``/``unshare`` is missing).

    This is distinct from a policy violation: it means the *executor
    itself* cannot run here, not that a command was blocked by policy.
    """


class SandboxPolicyViolationError(SandboxError):
    """Raised by executors that choose to surface a policy violation as an
    exception rather than (or in addition to) a non-zero
    :class:`SandboxResult`. Most callers should prefer inspecting
    :attr:`SandboxResult.policy_blocked` instead of relying on this being
    raised -- it exists for callers that want fail-fast semantics.
    """


@dataclass(frozen=True)
class SandboxPolicy:
    """Generic, mode-agnostic sandbox policy configuration.

    This dataclass is the "mode-specific policy configs layered on top [of
    the shared executor]" described in architecture doc §2.3. It is
    deliberately generic: both a LazyRed-style "red-team target sandbox"
    policy and a LazyAgent-style "benchmark task sandbox" policy are
    expected to be constructed from *this same dataclass* with different
    field values -- lazycore does not encode any module-specific policy
    logic (e.g. no OWASP-mapping-aware defaults) here.

    Only the tool-calling/code-execution layer is ever governed by this
    policy (§2.3.1's split-trust model) -- there is intentionally no field
    here for GPU/Metal/MPS access, because that layer is never sandboxed.

    **Path fields must be absolute.** ``allowed_read_paths``,
    ``allowed_write_paths``, and ``working_directory`` are validated eagerly
    in ``__post_init__`` and a relative entry raises ``ValueError``
    immediately at construction time, even though this dataclass is
    ``frozen=True``. This is not an arbitrary restriction: a relative path
    would otherwise be silently re-resolved against whatever the process's
    current working directory happens to be at *``run_command()`` call
    time* (inside :func:`~lazycore.sandbox.seatbelt.build_sbpl_profile`,
    which re-derives the profile fresh on every call), which can differ
    from the directory in effect when the policy was constructed --
    especially since this executor is explicitly shared in-process across
    modules (LazyRed, LazyAgent) that may call ``os.chdir()`` for unrelated
    reasons. Requiring absolute paths up front makes a constructed
    ``SandboxPolicy``'s effective meaning actually match its ``frozen=True``
    promise: it cannot change out from under a caller later. Callers should
    pass paths from sources that are already absolute, e.g.
    ``tempfile.mkdtemp()``'s return value or ``Path(...).resolve()``.
    ``allowed_executables`` is intentionally exempt from this requirement
    (see below) since bare command names resolved via ``PATH`` are a
    documented, legitimate value for that field.

    Attributes:
        allow_network: Whether the sandboxed process may open any network
            connections at all. When ``False``, executors should apply a
            default-deny egress policy.
        allowed_read_paths: Filesystem paths (files or directories) the
            sandboxed process may read, in addition to each backend's own
            small, fixed, non-configurable set of bootstrap paths needed
            for a process to start at all (see that backend's docstring,
            e.g. :func:`~lazycore.sandbox.seatbelt.build_sbpl_profile`).
            Directories are allowed recursively. **An empty tuple (the
            default) means NO user-data read access** -- home directory,
            arbitrary ``/tmp`` contents, project files, secrets, etc. are
            all unreadable unless explicitly listed here. This is a
            default-deny contract, not "unrestricted reads with an opt-in
            restriction" -- if your command needs to read a file, you must
            list it (or its containing directory) in this field. Must be
            absolute (see above); a relative entry raises ``ValueError`` at
            construction time.
        allowed_write_paths: Filesystem paths (files or directories) the
            sandboxed process may write to. Must be a subset of what's
            readable in spirit, but is tracked separately since "can read"
            and "can write" are different risk levels. Must be absolute
            (see above); a relative entry raises ``ValueError`` at
            construction time.
        allowed_executables: Absolute paths (or bare names resolved via
            ``PATH`` by the executor) of executables the sandboxed process
            is allowed to exec. An empty tuple means "no restriction is
            applied by this field" -- executors document their own default
            behavior when this is empty. Unlike the read/write path fields,
            bare (non-absolute) names are accepted here by design.
        env: Extra environment variables to set for the sandboxed process,
            on top of (or instead of, depending on ``inherit_env``) the
            calling process's environment. Coerced to an immutable
            ``types.MappingProxyType`` snapshot in ``__post_init__``: since
            this dataclass is ``frozen=True``, a plain ``dict`` here would
            otherwise still be mutable in place (``policy.env["X"] = "y"``
            does not touch the frozen *field*, only the dict object it
            points at), silently changing the effective sandbox environment
            without ever replacing the policy. The snapshot is taken from a
            *copy* of whatever mapping was passed in, so later mutating the
            original object the caller passed also has no effect. Pass a
            plain ``dict`` (or any ``Mapping``) at construction time; read
            access (``policy.env["X"]``, iteration, ``len()``, etc.) works
            exactly like a read-only ``dict``.
        inherit_env: Whether the sandboxed process inherits the calling
            process's full environment (with ``env`` applied as overrides)
            or starts from an empty environment plus only ``env``.
        timeout_seconds: Optional wall-clock timeout applied to the
            sandboxed command. ``None`` means no timeout.
        working_directory: Optional working directory for the sandboxed
            process. If set, it should also be included in
            ``allowed_read_paths``/``allowed_write_paths`` as appropriate --
            executors do not implicitly grant it access. Must be absolute
            (see above) if set; a relative value raises ``ValueError`` at
            construction time.
    """

    allow_network: bool = False
    allowed_read_paths: tuple[str, ...] = field(default_factory=tuple)
    allowed_write_paths: tuple[str, ...] = field(default_factory=tuple)
    allowed_executables: tuple[str, ...] = field(default_factory=tuple)
    env: Mapping[str, str] = field(default_factory=dict)
    inherit_env: bool = False
    timeout_seconds: float | None = None
    working_directory: str | None = None

    def __post_init__(self) -> None:
        """Eagerly reject relative paths (see class docstring).

        Deliberately does *not* attempt to coerce/normalize a relative path
        into an absolute one (e.g. via ``Path(p).resolve()``) -- doing so
        here would just move the "resolves against ambient CWD" problem
        from run time to construction time instead of eliminating it, and
        would silently paper over what is almost always a caller bug. This
        raises instead.

        Also coerces ``env`` into an immutable ``types.MappingProxyType``
        snapshot of a *copy* of whatever was passed in (see the ``env``
        field's docstring above for why this is needed even though the
        dataclass itself is frozen). ``object.__setattr__`` is required
        here, same as everywhere else in this method, because normal
        attribute assignment is disabled by ``frozen=True``.
        """
        object.__setattr__(self, "env", types.MappingProxyType(dict(self.env)))

        for field_name in ("allowed_read_paths", "allowed_write_paths"):
            for entry in getattr(self, field_name):
                if not Path(entry).is_absolute():
                    raise ValueError(
                        f"SandboxPolicy.{field_name} entries must be "
                        f"absolute paths; got {entry!r}. Relative paths "
                        "resolve against whatever the process's current "
                        "working directory happens to be at run_command() "
                        "call time, not at policy-construction time -- see "
                        "the SandboxPolicy class docstring. Use an "
                        "absolute path, e.g. from tempfile.mkdtemp() or "
                        "Path(...).resolve()."
                    )
        if self.working_directory is not None and not Path(self.working_directory).is_absolute():
            raise ValueError(
                "SandboxPolicy.working_directory must be an absolute path; "
                f"got {self.working_directory!r}. See the SandboxPolicy "
                "class docstring for why relative paths are rejected."
            )

    def with_overrides(self, **changes: object) -> "SandboxPolicy":
        """Return a copy of this policy with the given fields replaced.

        Convenience for building mode-specific policies off a shared base,
        e.g. ``base_policy.with_overrides(allow_network=True)``. Goes
        through the normal dataclass construction path, so the same
        absolute-path validation in ``__post_init__`` applies to the
        result.
        """
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SandboxResult:
    """Structured result of running a command/callable under a sandbox policy."""

    #: Process exit code. Backends that run a Python callable in-process
    #: (rather than as a subprocess) use ``0`` for a normal return and a
    #: nonzero sentinel (e.g. ``1``) if the callable raised.
    exit_code: int

    #: Captured stdout, decoded as UTF-8 (errors replaced).
    stdout: str

    #: Captured stderr, decoded as UTF-8 (errors replaced).
    stderr: str

    #: Best-effort flag: True if the executor has reason to believe the
    #: policy itself blocked something (e.g. a denied filesystem/network
    #: operation), as opposed to the command simply failing on its own
    #: terms. Backends set this heuristically -- e.g. the Seatbelt backend
    #: matches known denial phrases in captured stdout/stderr (such as
    #: `Operation not permitted`) and then cross-references the specific
    #: command and the policy that was actually in effect for this call to
    #: rule out the common false-positive case of an ordinary POSIX/DAC
    #: permission error unrelated to Seatbelt (see
    #: :mod:`lazycore.sandbox.seatbelt` for the exact mechanism and its
    #: documented limits). This materially reduces, but does not
    #: eliminate, false positives/negatives -- callers that need a
    #: guarantee should treat any nonzero ``exit_code`` combined with an
    #: empty/absent expected output as suspect regardless of this flag.
    policy_blocked: bool = False

    @property
    def succeeded(self) -> bool:
        """True if the command exited zero and no policy block was detected.

        Convenience combining :attr:`exit_code` and :attr:`policy_blocked`
        into a single pass/fail check for the common case where a caller
        doesn't need to distinguish *why* a run failed.
        """
        return self.exit_code == 0 and not self.policy_blocked


class BaseSandboxExecutor(abc.ABC):
    """Shared executor interface for the tool-calling/code-execution layer.

    Concrete backends (:class:`~lazycore.sandbox.seatbelt.SeatbeltSandboxExecutor`
    on macOS, :class:`~lazycore.sandbox.linux_stub.LinuxNamespaceSandboxExecutor`
    on Linux) implement this ABC. Per §2.3.1, no backend implementing this
    interface is ever responsible for constraining GPU/Metal/MPS access --
    that is deliberately outside the isolation boundary on every platform.
    """

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        """Create an executor with the given active policy.

        Args:
            policy: Initial :class:`SandboxPolicy` to apply. Defaults to a
                freshly-constructed, maximally-restrictive ``SandboxPolicy()``
                (no network, no read/write paths, no allowed executables)
                when omitted.
        """
        self._policy: SandboxPolicy = policy or SandboxPolicy()

    @property
    def policy(self) -> SandboxPolicy:
        """The policy this executor currently applies."""
        return self._policy

    def configure(self, policy: SandboxPolicy) -> None:
        """Replace this executor's active policy."""
        self._policy = policy

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can actually run on this host.

        Should be a cheap, side-effect-free check (e.g. platform check plus
        ``shutil.which`` for a required helper binary) -- it must not
        itself require sandbox privileges to answer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_command(
        self,
        command: Sequence[str],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Run a shell command under the given (or this executor's active)
        policy and return a structured result.

        Args:
            command: An argv-style sequence, e.g. ``["cat", "/etc/hosts"]``.
                Backends must not invoke a shell unless the caller has
                explicitly included a shell (e.g. ``["/bin/sh", "-c", ...]``)
                in ``command`` themselves -- this avoids surprise shell
                metacharacter expansion.
            policy: If given, overrides this executor's configured policy
                for this call only (does not persist).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_callable(
        self,
        func: Callable[..., object],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Run a Python callable under the given (or this executor's active)
        policy and return a structured result.

        Backends that cannot practically sandbox in-process Python (e.g.
        because true isolation requires a subprocess boundary) should
        implement this by marshalling ``func`` into a subprocess-executed
        script rather than silently running it unsandboxed.

        **Backends must never pickle (or otherwise deserialize-on-the-way-in)
        an arbitrary, possibly-untrusted callable in the trusted host
        process before the sandbox boundary exists** -- doing so would
        invoke that object's ``__reduce__``/``__reduce_ex__`` unsandboxed,
        which is itself an arbitrary-code-execution vector. Backends that
        need to hand a callable to a subprocess should instead require it
        to be resolvable by a stable, re-importable name (e.g. a module-
        level function, or a ``functools.partial`` wrapping one, with
        JSON-serializable bound arguments), pass that name plus JSON-safe
        arguments to the child process, and perform the actual import and
        argument decoding *inside* the sandboxed child -- see
        :class:`~lazycore.sandbox.seatbelt.SeatbeltSandboxExecutor`'s
        ``run_callable`` for the reference implementation of this pattern.
        """
        raise NotImplementedError
