"""macOS Seatbelt (`sandbox-exec`) backend (architecture doc §2.3.1).

This is the v1-default backend on macOS for the shared sandbox executor.
It generates a Seatbelt Scheme-based SBPL (Sandbox Profile Language)
profile from a :class:`~lazycore.sandbox.base.SandboxPolicy` and invokes
``/usr/bin/sandbox-exec -f <profile> -- <command>`` via :mod:`subprocess`.

**Read this before touching GPU/Metal anything.** Per §2.3.1's research
findings, Seatbelt is confirmed to be a real, production-used isolation
primitive for constraining untrusted *tool-calling* processes on macOS
(used by Gemini CLI and other "agent-in-a-sandbox" projects) -- but it is
also confirmed that **Seatbelt cannot block or gate Metal/GPU/Cocoa access
even if you wanted it to**: "GPU and display passthrough flags have no
effect on macOS because Metal and Cocoa are system-level and cannot be
blocked via SBPL." Seatbelt is a process-level filesystem/network/syscall
allowlist, not a VM/kernel boundary, and it has no mechanism to mediate the
GPU driver stack at all. This module therefore does not attempt to add any
GPU-blocking rule to the generated profile -- there is no such rule to add.
Per the locked split-trust architecture, GPU-bound model inference is
expected to run *outside* this sandbox entirely; this executor only ever
constrains the CPU-bound tool-calling/code-execution layer (shell commands,
file I/O, network egress).
"""

from __future__ import annotations

import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from typing import Callable, Sequence

from lazycore.sandbox.base import (
    BaseSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxPolicy,
    SandboxResult,
)

__all__ = ["SeatbeltSandboxExecutor", "build_sbpl_profile"]

_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"

#: Substrings observed in stderr/output when Seatbelt denies an operation.
#: Used only as a best-effort heuristic for :attr:`SandboxResult.policy_blocked`
#: -- most Unix tools surface a Seatbelt EPERM/EACCES denial as one of these
#: standard POSIX ``strerror`` messages, but callers should not treat this
#: as a hard guarantee (see :class:`~lazycore.sandbox.base.SandboxResult`
#: docstring and :meth:`SeatbeltSandboxExecutor._classify_denial` below for
#: how false positives are reduced).
#:
#: This previously also included ``"Sandbox: "`` and ``"deny("``, which are
#: the format of macOS's Seatbelt *unified log* denial line (e.g.
#: ``Sandbox: cat(1234) deny(1) file-read-data ...``). That line is written
#: to the system unified log via ``os_log``/ASL, not to the sandboxed
#: process's own stdout/stderr -- ``subprocess.run(..., capture_output=True)``
#: never sees it, so these two markers could never actually match anything
#: this backend captures and were dead code. Capturing that log line for
#: real would require running a concurrent ``log stream``/``log show``
#: collector alongside every sandboxed invocation and correlating its
#: output back to this specific subprocess by PID and timestamp -- doable
#: in principle, but it needs extra process-management complexity, a
#: predicate stable across macOS versions, and (on some macOS versions)
#: elevated log-reading privileges, for a benefit that's marginal given the
#: precision improvements below. That's a reasonable future enhancement,
#: not something this fix takes on; the markers were simply removed rather
#: than left in as non-functional decoration.
_DENIAL_MARKERS = (
    "Operation not permitted",
    "Permission denied",
)

#: Conservative allowlist of command basenames that are read-only in every
#: normal invocation (no flag combination makes them write to the
#: filesystem). Used only to *rule out* a policy-attributed denial, never
#: to assert one -- see :meth:`SeatbeltSandboxExecutor._classify_denial`.
#: Deliberately small: an incomplete allowlist just means some false
#: positives aren't caught (the pre-existing, documented behavior), not
#: that anything is misclassified in the other direction.
_READ_ONLY_COMMAND_BASENAMES = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "file",
        "stat",
        "md5",
        "md5sum",
        "shasum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "readlink",
        "realpath",
        "od",
        "xxd",
        "hexdump",
        "strings",
    }
)

#: Filesystem locations that a resolved ``allowed_read_paths``/
#: ``allowed_write_paths`` entry must never land on exactly -- see
#: :func:`_reject_overbroad_allowed_path` (Finding 2). This is an
#: exact-match set, not a prefix/depth check: it exists to catch the
#: concrete "resolves to a filesystem root or another suspiciously broad
#: system directory" case without also rejecting legitimate, much more
#: common deep paths that happen to live a couple of components below root
#: (e.g. ``/opt/homebrew`` or ``/private/tmp``).
_FORBIDDEN_BROAD_PATHS = frozenset(
    str(Path(p))
    for p in (
        "/",
        "/Users",
        "/home",
        "/System",
        "/etc",
        "/Library",
        "/private",
        "/private/etc",
        "/private/var",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/opt",
        "/root",
        "/Applications",
    )
)

try:
    _HOME_DIR: Path | None = Path.home()
except Exception:  # pragma: no cover -- exotic envs with no resolvable home
    _HOME_DIR = None


#: Fixed, hardcoded, non-user-configurable set of read paths granted to
#: *every* sandboxed run regardless of ``policy.allowed_read_paths`` (Finding
#: 1 fix). These are backend-owned bootstrap paths needed for basically any
#: process (including a fresh Python interpreter for ``run_callable``) to
#: start up at all on macOS -- dynamic linker/shared libraries, coreutils,
#: system frameworks, and locale/timezone data. This is deliberately a
#: narrow allowlist, not "the whole system": it grants zero access to any
#: user-data location (home directory, arbitrary ``/tmp`` entries, project
#: directories, etc.) -- those must be explicitly listed by the caller via
#: ``allowed_read_paths`` if a command needs to read them.
#:
#: Unlike caller-supplied ``allowed_read_paths``/``allowed_write_paths``
#: entries, these are never passed through
#: :func:`_reject_overbroad_allowed_path` -- that check exists specifically
#: to catch *caller* input silently widening via a symlink; this list is a
#: fixed, reviewed constant, not caller input, and several of its entries
#: (``/bin``, ``/private/etc``) are intentionally on
#: :data:`_FORBIDDEN_BROAD_PATHS` *for caller-supplied paths* precisely
#: because they are broad -- but are still the correct, minimal, real-world
#: bootstrap requirement here.
_STATIC_BOOTSTRAP_READ_PATHS: tuple[str, ...] = (
    "/usr/lib",
    "/usr/bin",
    "/bin",
    "/System/Library/Frameworks",
    "/System/Library/PrivateFrameworks",
    "/private/etc",
    "/private/var/db/timezone",
)


def _python_bootstrap_read_paths() -> tuple[str, ...]:
    """The running Python interpreter's own installation prefixes.

    Needed so ``run_callable``'s child-process runner script (invoked as
    ``sys.executable <runner> ...``) can actually read its own interpreter
    binary, stdlib, and site-packages -- without this, *any* Python
    subprocess (not just user commands) would fail to start under the new
    default-deny-reads profile. Includes ``sys.base_prefix``/
    ``sys.base_exec_prefix`` in addition to ``sys.prefix``/
    ``sys.exec_prefix`` so this also works correctly when the calling
    process is itself running inside a virtualenv/venv (``sys.prefix``
    alone would miss the base interpreter's stdlib in that case).
    """
    prefixes = {
        sys.prefix,
        sys.exec_prefix,
        getattr(sys, "base_prefix", sys.prefix),
        getattr(sys, "base_exec_prefix", sys.exec_prefix),
    }
    return tuple(sorted(p for p in prefixes if p))


def _bootstrap_read_paths() -> tuple[str, ...]:
    """Full set of hardcoded bootstrap read paths (static + Python-derived),
    each resolved to its canonical, symlink-free form via :func:`_canonical`.
    """
    return tuple(
        _canonical(p) for p in (*_STATIC_BOOTSTRAP_READ_PATHS, *_python_bootstrap_read_paths())
    )


def _escape_sbpl_string(value: str) -> str:
    """Escape a path for embedding in an SBPL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _canonical(path: str) -> str:
    """Resolve a path to its canonical, symlink-free form.

    macOS resolves several common temp-directory prefixes through symlinks
    (e.g. ``/tmp`` -> ``/private/tmp``), and Seatbelt matches against the
    canonical filesystem path, not the symlinked alias. Failing to resolve
    this is a classic source of "my allowed path doesn't work" bugs in
    hand-written Seatbelt profiles.

    This function performs *no* safety validation of its own, and -- unlike
    an earlier version of this module -- it is **not** used for bare
    (non-path) ``allowed_executables`` entries such as ``"python3"``.
    ``Path("python3").resolve()`` resolves a bare name against the
    *current working directory*, producing something like
    ``<cwd>/python3`` -- a path that (almost) never exists and will never
    match where the real ``python3`` executable actually lives on ``PATH``,
    silently making the documented "bare names are resolved via PATH"
    contract for ``allowed_executables`` a no-op that rejects the real
    executable. Use :func:`_resolve_allowed_executable` for
    ``allowed_executables`` entries instead. This function remains in use
    for ``allowed_read_paths``/``allowed_write_paths`` (via
    :func:`_reject_overbroad_allowed_path`, which wraps it with the
    Finding-2 anti-symlink-widening check) and for the fixed, always-a-real-
    path bootstrap read paths, where entries are never bare names.
    """
    return str(Path(path).resolve())


def _resolve_allowed_executable(name: str) -> str:
    """Resolve one ``allowed_executables`` entry to an absolute, canonical
    path suitable for an SBPL ``(literal ...)`` clause.

    ``allowed_executables`` documents bare command names (no path
    separator, e.g. ``"python3"``) as a legitimate value, meant to be
    resolved against ``PATH`` -- exactly like a shell would when exec'ing
    an unqualified command name. Passing such a bare name through
    :func:`_canonical` (plain ``Path.resolve()``) does *not* do that: it
    resolves the name against the current working directory instead,
    producing e.g. ``<cwd>/python3``, a path that almost never exists and
    will never match the real executable's location -- silently turning
    the documented allowlist entry into a rule that can never be satisfied
    by the process the caller actually intended to allow.

    This function instead:

    - For an entry that already looks like a path (contains ``os.sep``, or
      ``os.altsep`` on platforms that have one), resolves it exactly as
      before via :func:`_canonical` (``Path.resolve()``) -- full/relative
      paths are left to that existing, correct behavior.
    - For a bare name (no path separator), resolves it via
      :func:`shutil.which`, mirroring how the shell/``exec`` family would
      actually locate it on this host's current ``PATH``, then canonicalizes
      the result (resolving any symlinks, e.g. a Homebrew shim) via
      :func:`_canonical`.

    Raises:
        ValueError: If a bare name cannot be resolved via ``PATH`` at
            profile-build time -- silently emitting a bogus, never-matching
            ``(literal ...)`` clause would be strictly worse than failing
            loudly here, since the resulting policy would look like it
            allows the executable while actually allowing nothing.
    """
    looks_like_path = os.sep in name or (os.altsep is not None and os.altsep in name)
    if looks_like_path:
        return _canonical(name)

    resolved = shutil.which(name)
    if resolved is None:
        raise ValueError(
            f"SandboxPolicy.allowed_executables entry {name!r} could not be "
            "resolved via PATH (shutil.which() found nothing). Bare "
            "executable names in allowed_executables are resolved against "
            "PATH at profile-build time, exactly like a shell would -- if "
            f"{name!r} is not on this process's PATH, the resulting policy "
            "would silently never match the real executable. Either ensure "
            f"{name!r} is installed and on PATH, or pass its full absolute "
            "path instead."
        )
    return _canonical(resolved)


def _reject_overbroad_allowed_path(path: str, *, kind: str) -> str:
    """Resolve an ``allowed_read_paths``/``allowed_write_paths`` entry, and
    refuse to build a profile if doing so would silently grant far broader
    access than the path string suggests.

    ``Path.resolve()`` correctly (and desirably) follows well-known, benign
    OS-provided symlinks such as ``/tmp`` -> ``/private/tmp`` -- that case
    must keep working exactly as before. The problem this guards against is
    different: if a *caller-supplied* allowed-path entry is itself a
    symlink (or resolves through one) to something the path string gives no
    hint of -- e.g. an entry that looks like a scoped project directory but
    is actually a symlink to ``/`` -- ``resolve()`` would silently
    substitute that broader target into the generated SBPL profile, and the
    resulting policy would grant access far wider than anyone reviewing the
    ``SandboxPolicy`` construction call could tell from the path string
    alone.

    This is caught here by refusing to build the profile at all if the
    *resolved* path lands exactly on a filesystem root or another
    well-known, suspiciously broad system directory (``/``, ``/Users``,
    ``/home``, ``/System``, ``/etc``, and similar -- see
    ``_FORBIDDEN_BROAD_PATHS``), or exactly on the user's home directory.

    **Honest limits.** This is a targeted check, not a general anti-symlink
    or anti-traversal mechanism:

    - It only catches resolution landing on one of a fixed set of
      well-known broad locations -- a symlink pointing at some other
      specific-but-still-wrong directory (e.g. a different user's home
      directory, or an unrelated project's data dir) will not be flagged.
      Getting this fully general would require a subjective notion of
      "how much broader is too much broader" relative to the *original*
      path string, which this function deliberately does not attempt.
    - It is a construction-time/profile-build-time check on the
      *configured* allowed-path entries. It cannot and does not attempt to
      prevent a sandboxed process from creating its own symlink at runtime
      that points from an allowed directory to somewhere outside it -- that
      is an OS-level property of how Seatbelt evaluates ``subpath`` rules
      against live symlinks, a separate, already-noted test gap, and out of
      scope for this fix.
    """
    resolved = Path(path).resolve()
    resolved_str = str(resolved)
    is_root = resolved_str == resolved.anchor
    is_forbidden_broad = resolved_str in _FORBIDDEN_BROAD_PATHS
    is_home = _HOME_DIR is not None and resolved == _HOME_DIR
    if is_root or is_forbidden_broad or is_home:
        raise ValueError(
            f"Refusing to build sandbox profile: {kind} entry {path!r} "
            f"resolves to {resolved_str!r}, which is a filesystem root or "
            "another suspiciously broad location (e.g. '/', '/Users', "
            "'/etc', '/System', or the user's home directory itself). If "
            f"{path!r} is a symlink, Path.resolve() has silently "
            "substituted its target here -- that would grant access far "
            "broader than the path string suggests. Pass a more specific, "
            "non-symlinked subdirectory instead."
        )
    return resolved_str


def _safe_type_name(value: object) -> str:
    """Return ``type(value).__name__`` without ever risking a hostile
    metaclass's ``__getattribute__`` hook (Finding 1 fix, second half).

    ``type(value)`` itself is safe -- it is the C-level ``Py_TYPE()``
    lookup, and (unlike ``value.__class__``, which is an overridable
    property/descriptor an attacker-crafted instance could hijack) cannot
    be fooled into returning anything other than ``value``'s real,
    underlying type. But the *next* step, a plain ``.__name__`` attribute
    access on that type object, is a normal Python attribute lookup --
    which means if ``type(value)`` itself was created with a sufficiently
    exotic, attacker-controlled **metaclass** that overrides
    ``__getattribute__``, reading ``.__name__`` off it would invoke that
    metaclass's ``__getattribute__`` hook. That hook runs in this trusted
    host process, before/while this function is merely trying to *report*
    that ``value`` is invalid -- the same "even rejection isn't safe"
    problem already solved for ``repr()``/``str()`` elsewhere in this
    module, just one level up the type hierarchy.

    ``type.__getattribute__(type(value), "__name__")`` sidesteps this: it
    explicitly invokes the base :class:`type` class's own, non-overridden
    attribute-lookup machinery on the type object, rather than whatever
    ``__getattribute__`` the type's own (possibly malicious) metaclass
    defines. This has been empirically verified to still return the
    correct, ordinary name for normal types (e.g.
    ``type.__getattribute__(type(5), "__name__") == "int"``) while
    bypassing a metaclass ``__getattribute__`` override that would
    otherwise fire for a crafted type.
    """
    return type.__getattribute__(type(value), "__name__")


def _validate_json_safe_value(value: object, *, path: str) -> None:
    """Recursively validate that ``value`` contains only genuinely
    JSON-native types, for use before ``run_callable()`` writes bound
    ``functools.partial`` arguments to the payload file the sandboxed
    child reads (Finding 3 fix).

    ``json.dumps()`` on its own is too permissive for this security
    boundary: it *coerces* rather than rejects several Python types on
    encode, so "it round-tripped through ``json.dumps``/``json.load``
    without an exception" is not the same guarantee as "the sandboxed
    child sees exactly the value the caller passed". Concretely:

    - A ``tuple`` serializes to a JSON array **indistinguishable** from a
      ``list`` -- the child's ``json.load()`` always reconstructs a
      ``list``, so a caller who bound a tuple argument via
      ``functools.partial`` would silently have it replaced with a list
      of the same elements inside the sandbox: a different runtime
      type/value than what was actually requested, for what may be a
      security- or correctness-sensitive call.
    - A non-``str`` dict key (e.g. an ``int`` key) is silently stringified
      by ``json.dumps`` (``{1: "a"}`` becomes ``{"1": "a"}`` on the wire),
      again changing what the child actually receives relative to what
      the caller specified.

    This function walks ``value`` and only accepts ``str``, ``int``,
    ``float``, ``bool``, ``None``, ``list`` (recursing into elements), and
    ``dict`` with string-only keys (recursing into values) -- explicitly
    rejecting ``tuple`` (even though ``json.dumps`` happily accepts it),
    any non-``str`` dict key, and any other type (``set``, custom class
    instances, etc.).

    NaN/Infinity/-Infinity floats are deliberately *not* rejected here --
    plain ``float`` is a legitimate JSON-native scalar type, and
    ``json.dumps(..., allow_nan=False)`` (used at the actual serialization
    call sites, see :func:`_validate_run_callable_arguments`) already
    raises ``ValueError`` for those specific values with a clear message.
    Duplicating that check here would just be two ways to catch the same
    thing.

    Args:
        value: The (sub)value to validate.
        path: A human-readable description of ``value``'s location within
            the overall ``args``/``kwargs`` structure, used to build a
            precise error message (e.g. ``"args[0]"``, ``"kwargs['x'][2]"``).

    **Rejection messages never call ``repr()``/``str()`` on the untrusted,
    already-invalid ``value`` (or a non-``str`` dict key).** This is the
    same vulnerability class already fixed twice elsewhere in this module
    (attacker-controlled dunder methods -- there, ``__eq__``/``__ne__``/
    ``__contains__``; here, ``__repr__``/``__str__``): calling ``repr()``
    on an arbitrary object invokes ``value.__repr__()``, which a malicious
    class could override to run arbitrary code the moment this function
    tries to *report* that the value is invalid -- i.e. even the "we're
    rejecting this" path is not safe unless it avoids touching the object
    itself. Error messages below therefore only ever use
    :func:`_safe_type_name` (never a bare ``type(value).__name__``) and the
    caller-constructed, already-known-safe ``path`` string, never a
    ``repr()``/``str()``/f-string ``!r``/``!s`` of ``value`` itself. The one
    exception is a dict key already confirmed to be exactly ``str`` (not a
    subclass) by the ``type(key) is not str`` check immediately above each
    such use -- plain ``str.__repr__`` cannot be overridden without
    subclassing, so that specific ``repr()`` call is safe.

    **Why not a bare ``type(value).__name__``?** ``type(value)`` correctly
    bypasses an overridable ``value.__class__`` property and returns
    ``value``'s true underlying type -- but a plain ``.__name__`` attribute
    read on *that* type object is still a normal attribute lookup, which a
    sufficiently exotic custom **metaclass** (one that overrides
    ``__getattribute__``) could hook to run code the moment this validator
    merely tries to name the rejected type. :func:`_safe_type_name` avoids
    this by explicitly using ``type``'s own attribute-lookup machinery
    (``type.__getattribute__(type(value), "__name__")``) instead of
    whatever ``__getattribute__`` the value's actual type's metaclass might
    define. The same reasoning is why the tuple check just below uses
    ``type(value) is tuple`` rather than ``isinstance(value, tuple)``:
    ``isinstance()`` can, on some code paths, consult the overridable
    ``value.__class__`` rather than purely the C-level true type, whereas
    ``type(value) is tuple`` is a direct ``Py_TYPE()`` identity comparison
    that a crafted ``__class__`` property cannot influence. (A genuine
    ``tuple`` *subclass* falls through to the generic "not a JSON-native
    type" rejection below instead of the tuple-specific message -- it is
    still rejected either way, just via the more general error message,
    exactly like every other non-JSON-native type.)

    Raises:
        ValueError: If ``value`` (or anything nested inside it) is a
            ``tuple``, a ``dict`` with a non-``str`` key, or any other
            non-JSON-native type.
    """
    if value is None or type(value) in (str, int, float, bool):
        return

    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_safe_value(item, path=f"{path}[{index}]")
        return

    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(
                    f"{path} has a non-string key of type "
                    f"{_safe_type_name(key)!r}. run_callable() rejects this "
                    "instead of letting json.dumps() silently stringify it "
                    '(e.g. turning {1: ...} into {"1": ...} on the wire) -- '
                    "that would change what key the sandboxed child "
                    "actually sees relative to what was passed. Use only "
                    "string keys. (The key's value is deliberately omitted "
                    "from this message -- repr()/str() on an untrusted, "
                    "already-invalid object could invoke an "
                    "attacker-controlled __repr__/__str__ in this trusted "
                    "host process.)"
                )
            # Safe to repr() here: the type(key) is not str check above
            # already raised for anything else, so key is guaranteed to be
            # a plain str (not a subclass) at this point, and plain str's
            # __repr__ cannot be attacker-overridden.
            _validate_json_safe_value(item, path=f"{path}[{key!r}]")
        return

    if type(value) is tuple:
        raise ValueError(
            f"{path} is a tuple. run_callable() rejects tuple "
            "arguments instead of letting json.dumps() silently coerce "
            "them into a JSON array indistinguishable from a list -- the "
            "sandboxed child's json.load() always reconstructs a list, so "
            "it would receive a different runtime type than the one "
            "actually passed. Convert it to a list explicitly if a list is "
            "what you intend the child to receive. (The tuple's contents "
            "are deliberately omitted from this message -- repr() on an "
            "untrusted value could invoke an attacker-controlled __repr__ "
            "in this trusted host process.)"
        )

    raise ValueError(
        f"{path} is of type {_safe_type_name(value)!r}, which is "
        "not a JSON-native type. run_callable() only supports str, int, "
        "float, bool, None, list, and dict (with string keys) for "
        "functools.partial bound arguments. (The value itself is "
        "deliberately omitted from this message -- repr()/str() on an "
        "untrusted, already-invalid object could invoke an "
        "attacker-controlled __repr__/__str__ in this trusted host "
        "process.)"
    )


def _validate_run_callable_arguments(
    args: Sequence[object],
    kwargs: dict[str, object],
    module_name: str,
    qualname: str,
) -> None:
    """Validate that ``args``/``kwargs`` are safe to write as the JSON
    payload :meth:`SeatbeltSandboxExecutor.run_callable` hands to its
    sandboxed child process.

    Two checks, both required (Finding 3 fix):

    1. :func:`_validate_json_safe_value`, recursively, over every element
       of ``args`` and every value in ``kwargs`` -- rejects ``tuple``
       values and non-``str`` dict keys that plain ``json.dumps()`` would
       otherwise silently coerce (see that function's docstring).
    2. ``json.dumps(..., allow_nan=False)`` over the whole structure --
       rejects ``float("nan")``/``float("inf")``/``float("-inf")``, which
       standard ``json.dumps()`` accepts by default as a non-standard
       extension that many strict JSON parsers would reject. Raising here,
       at serialization time, keeps the "this payload is boring, portable,
       standard-compliant JSON" guarantee this whole design leans on.

    Raises:
        ValueError: If any argument fails either check.
    """
    for index, value in enumerate(args):
        _validate_json_safe_value(value, path=f"args[{index}]")
    for key, value in kwargs.items():
        if type(key) is not str:
            raise ValueError(
                f"a kwargs key of type {_safe_type_name(key)!r} is not a "
                "string. run_callable() keyword arguments must have string "
                "keys. (The key's value is deliberately omitted from this "
                "message -- repr()/str() on an untrusted, already-invalid "
                "object could invoke an attacker-controlled "
                "__repr__/__str__ in this trusted host process.)"
            )
        _validate_json_safe_value(value, path=f"kwargs[{key!r}]")

    try:
        json.dumps(list(args), allow_nan=False)
        json.dumps(kwargs, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "run_callable() only supports module-level functions or "
            "functools.partial wrapping one, with JSON-serializable "
            f"arguments -- the bound arguments for {module_name}."
            f"{qualname} are not JSON-serializable ({exc!r}). Pickle is "
            "not used for arguments (or the callable itself) because it "
            "is unsafe to deserialize untrusted data; pass only "
            "JSON-safe types (str, int, float, bool, None, list, dict "
            "with string keys), and avoid NaN/Infinity/-Infinity floats."
        ) from exc


def build_sbpl_profile(policy: SandboxPolicy) -> str:
    """Generate an SBPL (Sandbox Profile Language) profile string for ``policy``.

    Profile structure:

    - ``(version 1)`` / ``(deny default)`` -- default-deny baseline.
    - A small fixed set of ``allow`` rules needed for *any* process to
      start and exit cleanly (fork, signal-to-self, sysctl-read,
      file-read-metadata) -- these are not policy-configurable because
      they are required for basic process lifecycle, not for accessing
      user data.
    - **Reads: default-deny.** ``policy.allowed_read_paths`` being empty
      (the default) means the sandboxed process has **no user-data read
      access at all** -- not "unrestricted reads". Reads are always
      restricted to exactly the union of two sets: (1) a small, fixed,
      non-configurable set of backend-owned bootstrap paths needed for
      *any* process to start at all on macOS (dynamic linker/shared
      libraries, coreutils, system frameworks, locale/timezone data, and
      the running Python interpreter's own install prefix -- see
      :func:`_bootstrap_read_paths`), and (2) whatever
      ``policy.allowed_read_paths`` explicitly lists. **If your command
      needs to read a file, you must list it (or its containing directory)
      in ``allowed_read_paths`` -- an empty tuple grants zero access to the
      home directory, arbitrary ``/tmp`` contents, project files, secrets,
      or any other user data.**
    - **Writes:** always restricted to ``policy.allowed_write_paths``
      (subpath-based). An empty tuple (the default) means no writes are
      allowed anywhere -- this is the primary enforcement demonstrated by
      this backend's tests.
    - **Network:** ``(allow network*)`` only if ``policy.allow_network`` is
      True; otherwise omitted, so the default-deny baseline blocks all
      network egress.
    - **Executables:** if ``policy.allowed_executables`` is non-empty,
      ``process-exec`` is restricted to those literal paths; otherwise
      ``process-exec`` is left unrestricted (needed for ordinary shell
      commands to exec `/bin/sh`, coreutils, `python3`, etc.). Each entry
      is resolved via :func:`_resolve_allowed_executable`, not plain
      ``Path.resolve()``: a bare name with no path separator (e.g.
      ``"python3"``) is resolved against ``PATH`` via :func:`shutil.which`
      -- resolving it with ``Path.resolve()`` instead would silently
      produce a bogus path relative to the current working directory
      (``<cwd>/python3``) that the real executable never matches. An entry
      that already looks like a path is resolved exactly as before.

    **Anti-widening check on read/write paths.** Each entry in
    ``allowed_read_paths``/``allowed_write_paths`` is resolved via
    :func:`_reject_overbroad_allowed_path`, not plain ``Path.resolve()``:
    this raises ``ValueError`` (refusing to build the profile at all)
    rather than silently emitting a ``subpath`` rule if an entry resolves
    to a filesystem root or another suspiciously broad system directory
    (see that function's docstring for exactly what is and is not caught --
    it is a targeted check, not a general anti-symlink/anti-traversal
    mechanism).

    Never adds, and will never add, any rule attempting to gate GPU/Metal/
    Cocoa access -- see the module docstring.

    Raises:
        ValueError: If an ``allowed_read_paths``/``allowed_write_paths``
            entry resolves to a suspiciously broad location -- see
            :func:`_reject_overbroad_allowed_path` -- or if a bare
            ``allowed_executables`` entry cannot be resolved via ``PATH``
            -- see :func:`_resolve_allowed_executable`.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        "",
        "; --- baseline process lifecycle (not user-configurable) ---",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        "(allow mach-lookup)",
        "(allow iokit-open)",
        "(allow file-ioctl)",
        # Reading the root directory entry itself ("/", not its contents --
        # subpath rules below govern those) is needed by dyld/libSystem
        # during ordinary process startup (observed empirically: without
        # this, /bin/cat and other simple dynamically-linked binaries abort
        # with SIGABRT before even reaching main(), regardless of which
        # subpaths are otherwise allowed). This is a `literal` match on "/"
        # itself, not a `subpath` -- it does not grant listing/reading of
        # any file *under* root.
        '(allow file-read-data (literal "/"))',
    ]

    lines.append("")
    lines.append(
        "; --- filesystem reads: bootstrap paths (always) + allowed_read_paths ---"
    )
    read_paths = list(_bootstrap_read_paths())
    for p in policy.allowed_read_paths:
        read_paths.append(_reject_overbroad_allowed_path(p, kind="allowed_read_paths"))
    read_clauses = " ".join(
        f'(subpath "{_escape_sbpl_string(p)}")' for p in read_paths
    )
    lines.append(f"(allow file-read* {read_clauses})")

    lines.append("")
    lines.append("; --- filesystem writes ---")
    if policy.allowed_write_paths:
        write_clauses = " ".join(
            f'(subpath "{_escape_sbpl_string(_reject_overbroad_allowed_path(p, kind="allowed_write_paths"))}")'
            for p in policy.allowed_write_paths
        )
        lines.append(f"(allow file-write* {write_clauses})")
    else:
        lines.append("; no allowed_write_paths configured -- all writes denied")

    lines.append("")
    lines.append("; --- network egress ---")
    if policy.allow_network:
        lines.append("(allow network*)")
    else:
        lines.append("; allow_network=False -- all network egress denied")

    lines.append("")
    lines.append("; --- process execution ---")
    if policy.allowed_executables:
        exec_clauses = " ".join(
            f'(literal "{_escape_sbpl_string(_resolve_allowed_executable(p))}")'
            for p in policy.allowed_executables
        )
        lines.append(f"(allow process-exec {exec_clauses})")
    else:
        lines.append("(allow process-exec)")

    return "\n".join(lines) + "\n"


class SeatbeltSandboxExecutor(BaseSandboxExecutor):
    """Sandbox executor backed by macOS's `sandbox-exec`/Seatbelt.

    Restricts filesystem reads/writes, network egress, and (optionally)
    which executables may be exec'd by the tool-calling/code-execution
    layer. Never attempts, and cannot in principle (per §2.3.1), gate GPU/
    Metal/Cocoa access -- inference processes must be run outside this
    sandbox entirely, per the platform's split-trust architecture.

    **Stricter contract than the original implementation:**

    - **Reads are default-deny.** An empty ``allowed_read_paths`` means the
      sandboxed process can read *only* a small, fixed set of
      backend-owned bootstrap paths needed to execute a process at all
      (dynamic linker/shared libraries, coreutils, system frameworks,
      locale/timezone data, and the running Python interpreter's own
      install prefix) -- it grants **no** access to the home directory,
      arbitrary ``/tmp`` contents, project files, or any other user data.
      If your command needs to read a file, you must list it (or its
      containing directory) in ``allowed_read_paths`` explicitly. See
      :func:`build_sbpl_profile` and :func:`_bootstrap_read_paths` for the
      exact bootstrap set.
    - ``run_callable()`` never pickles the caller-supplied callable in this
      (trusted, host) process -- only a plain module-level function, or a
      ``functools.partial`` wrapping one with JSON-serializable bound
      arguments, is accepted. Its identity is validated in this process
      purely via ``func.__globals__`` introspection (never by importing
      ``func.__module__``, which is a freely rewritable string an attacker
      could point anywhere); the function is only ever imported by
      ``(module, qualname)`` and called *inside* the sandboxed child
      process. See :meth:`run_callable`'s and
      :meth:`_resolve_module_level_function`'s docstrings.
    - ``SandboxPolicy.allowed_read_paths``/``allowed_write_paths``/
      ``working_directory`` must be absolute paths, enforced eagerly by
      :class:`~lazycore.sandbox.base.SandboxPolicy`'s own
      ``__post_init__`` -- a relative entry raises ``ValueError`` at
      policy-construction time, not when ``run_command()`` happens to be
      called from a different working directory later.
    - ``allowed_read_paths``/``allowed_write_paths`` entries are rejected
      with ``ValueError`` (profile build refused entirely) if they resolve
      to a filesystem root or another suspiciously broad system directory
      -- see :func:`_reject_overbroad_allowed_path`. This is a targeted
      anti-widening check, not a general anti-symlink mechanism; benign,
      well-known OS symlinks (e.g. ``/tmp`` -> ``/private/tmp``) are
      unaffected.
    - ``SandboxResult.policy_blocked`` is still a best-effort heuristic,
      never a guarantee, but is more precise than plain substring matching
      against stdout/stderr: it cross-references the specific command and
      the policy actually in effect for that call to rule out the common
      false-positive case of an ordinary POSIX/DAC permission error (e.g.
      a chmod 000 file) that has nothing to do with Seatbelt -- see
      :meth:`_classify_denial`. It still cannot see Seatbelt's own denial
      logging (written to the unified log, not to the sandboxed process's
      captured stdout/stderr), and it still cannot classify commands
      outside its small internal allowlist with the same precision.
    """

    def is_available(self) -> bool:
        """True if running on macOS with ``/usr/bin/sandbox-exec`` present.

        Cheap, side-effect-free check per the
        :meth:`~lazycore.sandbox.base.BaseSandboxExecutor.is_available`
        contract -- it does not itself invoke ``sandbox-exec``.
        """
        return platform.system() == "Darwin" and Path(_SANDBOX_EXEC_PATH).exists()

    def _require_available(self) -> None:
        if not self.is_available():
            raise SandboxBackendUnavailableError(
                "SeatbeltSandboxExecutor requires macOS with "
                f"{_SANDBOX_EXEC_PATH} present; this host does not "
                "satisfy that (platform="
                f"{platform.system()!r})."
            )

    def _resolve_policy(self, policy: SandboxPolicy | None) -> SandboxPolicy:
        return policy if policy is not None else self._policy

    def _build_env(self, policy: SandboxPolicy) -> dict[str, str]:
        import os

        env: dict[str, str] = dict(os.environ) if policy.inherit_env else {}
        env.update(policy.env)
        return env

    def _classify_denial(
        self,
        policy: SandboxPolicy,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> bool:
        """Best-effort classification of whether Seatbelt (as opposed to an
        ordinary POSIX/DAC permission error) caused this failure.

        This is necessarily still a heuristic, not a guarantee -- see the
        :class:`~lazycore.sandbox.base.SandboxResult.policy_blocked`
        docstring.

        **Historical note (retired precision case).** An earlier version of
        this method downgraded a "Permission denied"/"Operation not
        permitted" report to ``policy_blocked=False`` whenever the failing
        command was a member of ``_READ_ONLY_COMMAND_BASENAMES`` *and*
        ``policy.allowed_read_paths`` was empty -- reasoning that
        :func:`build_sbpl_profile` at the time emitted an unconditional,
        completely unrestricted ``(allow file-read*)`` for an empty
        ``allowed_read_paths``, so Seatbelt could not structurally have
        caused the denial. That invariant no longer holds: per the Finding-1
        fix, :func:`build_sbpl_profile` now *always* restricts reads to a
        fixed bootstrap set plus ``allowed_read_paths`` (never fully
        unrestricted, regardless of whether ``allowed_read_paths`` is
        empty), so a read-only command failing under *any* policy could
        genuinely be a Seatbelt denial (e.g. the file being read is outside
        both the bootstrap set and ``allowed_read_paths``) rather than only
        ever an ordinary DAC error. Downgrading based on command basename
        alone is therefore no longer sound, and this method no longer does
        it -- ``_READ_ONLY_COMMAND_BASENAMES`` is retained only for
        documentation/potential future use (e.g. if this method is later
        extended to inspect which specific path a command tried to read and
        compare it against the granted set), not used by this method's
        current logic.
        """
        if exit_code == 0:
            return False
        combined = stdout + stderr
        return any(marker in combined for marker in _DENIAL_MARKERS)

    def run_command(
        self,
        command: Sequence[str],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Run ``command`` under a freshly-generated Seatbelt profile.

        Builds an SBPL profile from the active (or overriding) policy via
        :func:`build_sbpl_profile`, writes it to a temporary ``.sb`` file,
        and invokes it through ``sandbox-exec -f <profile> -- <command>``.
        The temporary profile file is always removed afterward, even on
        timeout or other failure. A :exc:`subprocess.TimeoutExpired` (per
        ``policy.timeout_seconds``) is translated into a ``SandboxResult``
        with ``exit_code=124`` rather than propagating as an exception.

        Raises:
            SandboxBackendUnavailableError: If :meth:`is_available` is False
                on this host.
        """
        self._require_available()
        active_policy = self._resolve_policy(policy)
        profile = build_sbpl_profile(active_policy)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sb", delete=False
        ) as profile_file:
            profile_file.write(profile)
            profile_path = profile_file.name

        try:
            argv = [_SANDBOX_EXEC_PATH, "-f", profile_path, "--", *command]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=active_policy.timeout_seconds,
                    env=self._build_env(active_policy),
                    cwd=active_policy.working_directory,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    exit_code=124,
                    stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                    stderr=(exc.stderr or "" if isinstance(exc.stderr, str) else "")
                    + "\n[lazycore.sandbox] command timed out",
                    policy_blocked=False,
                )

            policy_blocked = self._classify_denial(
                active_policy, command, completed.returncode, completed.stdout, completed.stderr
            )
            return SandboxResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                policy_blocked=policy_blocked,
            )
        finally:
            Path(profile_path).unlink(missing_ok=True)

    def _resolve_module_level_function(self, func: object) -> tuple[str, str]:
        """Validate that ``func`` is a plain, re-importable module-level
        function and return its ``(module_name, qualname)``.

        Rejects anything that is not re-importable by name: lambdas, local
        closures, bound/unbound methods, callable class instances, and
        (crucially) any object whose ``__reduce__``/``__reduce_ex__`` could
        run attacker-controlled code if it were ever pickled -- this
        function never pickles ``func``.

        **This validation never imports anything.** An earlier version of
        this method verified ``func.__module__``/``__qualname__`` by calling
        ``importlib.import_module(module_name)`` directly in this (trusted,
        host) process and attribute-chasing ``qualname`` on the result. That
        was itself a vulnerability of the same shape as the original
        pickle-based Finding 2: ``__module__`` is a plain, freely writable
        string attribute on any function object (``func.__module__ =
        "attacker/controlled/module"`` is valid Python with no special
        privileges required), so a caller -- or anything upstream that
        constructs "a function" and hands it to ``run_callable`` -- could
        make this host process import and execute arbitrary module-level
        code *before* the identity check even ran, let alone before the
        sandbox existed.

        Instead, this method validates identity purely via introspection of
        ``func.__globals__`` -- the dict object that *is* the namespace
        ``func`` was actually defined in. Unlike ``__module__``/
        ``__qualname__``, ``__globals__`` is not a re-assignable string that
        can be pointed somewhere else after the fact by simple attribute
        assignment; it is fixed to the defining module's namespace at
        function-creation time. Three checks, none of which ever import
        anything:

        1. ``qualname`` must be a flat, undotted name -- i.e. ``func`` must
           be bound directly at module level, not nested inside a class or
           another function. (Bound/nested names would require attribute-
           chasing through ``__globals__``, which a dict lookup by full
           qualname string cannot do; since this method's whole contract is
           "module-level functions", rejecting dotted names is a
           tightening, not a regression.)
        2. ``func.__globals__["__name__"]`` (the *actual* module ``func`` was
           defined in) must equal ``func.__module__`` (the caller-visible,
           possibly-tampered claim). If these disagree, ``__module__`` has
           been reassigned since definition and is untrustworthy -- reject
           without ever importing ``module_name`` to find out.
        3. ``func.__globals__[qualname]`` must be ``func`` itself -- i.e.
           the name still resolves, inside its own defining namespace, to
           the exact same function object (not a reassigned/shadowed name).

        The actual ``importlib.import_module(module_name)`` call for this
        ``(module_name, qualname)`` pair happens later, but only ever inside
        the already-sandboxed child process (see the runner script built by
        :meth:`run_callable`) -- never here in the host.

        **Plain-``str`` type check happens before anything else.**
        ``__module__`` and ``__qualname__`` are ordinary, freely-writable
        attributes on a function object -- nothing stops
        ``func.__module__ = some_object`` where ``some_object`` is not even
        a string. If that were allowed through, the very next line of
        defense (``"<lambda>" in qualname``, or
        ``globals_dict.get("__name__") != module_name``) would perform a
        containment check or rich comparison *against attacker-supplied
        input*, which for a custom object means invoking that object's
        ``__contains__``/``__eq__``/``__ne__`` dunder methods -- in this
        trusted host process, before the sandbox exists. That is the exact
        same class of vulnerability as the already-fixed
        ``importlib.import_module(func.__module__)`` bug this method's
        docstring describes above: it doesn't matter whether the untrusted
        value is *imported* or merely *compared*, either way attacker code
        runs unsandboxed. So every value later used in a comparison,
        containment check, or f-string built from untrusted metadata --
        ``qualname``, ``module_name``, and ``globals_dict.get("__name__")``
        -- is required to be an exact ``str`` (``type(x) is str``, not
        ``isinstance(x, str)``, since a ``str`` subclass could itself
        override ``__eq__``/``__contains__``) *before* any such operation is
        attempted on it, and this check is the very first thing done after
        confirming ``func`` is a plain function.

        Raises:
            TypeError: If ``func`` is not a plain function object.
            ValueError: If ``func``'s ``__qualname__``/``__module__`` (or its
                own ``__globals__["__name__"]``) is not a plain ``str``; if
                ``qualname`` is not a flat module-level name; if
                ``__module__``/``__globals__`` do not agree on where it
                lives; or if it cannot be resolved back to the exact same
                function object from within its own ``__globals__``.
        """
        # `type(x) is types.FunctionType` (not `isinstance`) deliberately:
        # isinstance()'s slow path consults the target's overridable
        # `__class__` property when the fast type check fails, so a hostile
        # object with a `__class__` property could execute arbitrary code in
        # this trusted host process merely by being passed to isinstance(),
        # before any of the checks below even run.
        if type(func) is not types.FunctionType:
            raise TypeError(
                "run_callable() only supports module-level functions or "
                "functools.partial wrapping one -- got a non-function "
                f"object of type {_safe_type_name(func)!r}."
            )

        qualname = getattr(func, "__qualname__", "")
        module_name = getattr(func, "__module__", None)

        # Type check FIRST, before any comparison/containment check touches
        # these values -- see the docstring section above. `type(x) is str`
        # (not `isinstance`) deliberately rejects `str` subclasses too, since
        # a subclass could itself override __eq__/__contains__/__hash__.
        if type(qualname) is not str or type(module_name) is not str:
            raise ValueError(
                "run_callable() requires plain string module metadata -- "
                f"got __qualname__ of type {_safe_type_name(qualname)!r} and "
                f"__module__ of type {_safe_type_name(module_name)!r} for "
                f"{func!r}. Both must be exactly `str` (not a subclass): "
                "comparing or containment-testing a non-str value here "
                "would invoke attacker-controlled __eq__/__ne__/__contains__ "
                "methods in this trusted host process, before the sandbox "
                "even exists -- refusing before any such operation is "
                "attempted."
            )

        if "<lambda>" in qualname or "<locals>" in qualname or not module_name:
            raise ValueError(
                "run_callable() only supports module-level functions or "
                "functools.partial wrapping one, with JSON-serializable "
                f"arguments -- got {func!r} (qualname={qualname!r}), which "
                "looks like a lambda or a local/nested closure. Define the "
                "target as a plain module-level function instead."
            )

        if "." in qualname:
            raise ValueError(
                "run_callable() only supports module-level functions or "
                "functools.partial wrapping one -- "
                f"{module_name}.{qualname} is not a flat module-level name "
                "(it looks like a method, nested class attribute, or "
                "otherwise non-top-level binding)."
            )

        globals_dict = func.__globals__
        actual_module_name = globals_dict.get("__name__")

        # Same type check applied to the (normally trustworthy) __globals__
        # namespace's own "__name__" entry -- defense in depth in case
        # something upstream ever manages to influence it.
        if type(actual_module_name) is not str:
            raise ValueError(
                "run_callable() requires plain string module metadata -- "
                f"{func!r}'s own __globals__['__name__'] is of type "
                f"{_safe_type_name(actual_module_name)!r}, not `str`. "
                "Refusing to compare it against __module__ before any such "
                "comparison is attempted."
            )

        if actual_module_name != module_name:
            raise ValueError(
                "run_callable() only supports module-level functions or "
                "functools.partial wrapping one, with JSON-serializable "
                f"arguments -- {func!r}'s __module__ ({module_name!r}) does "
                "not match the __name__ of its own __globals__ "
                f"({actual_module_name!r}). This function refuses "
                "to import module_name to check this (that would execute "
                "attacker-influenced module code in the trusted host "
                "process before validation completes) -- __globals__ is "
                "the trustworthy source of truth for where this function "
                "actually lives."
            )

        if globals_dict.get(qualname) is not func:
            raise ValueError(
                "run_callable() only supports module-level functions or "
                "functools.partial wrapping one, with JSON-serializable "
                f"arguments -- {module_name}.{qualname} does not resolve "
                "back to the exact same function object inside its own "
                "__globals__ (it may have been reassigned since "
                "definition)."
            )

        return module_name, qualname

    def _decompose_callable(
        self, func: Callable[..., object]
    ) -> tuple[str, str, tuple[object, ...], dict[str, object]]:
        """Break ``func`` down into ``(module_name, qualname, args, kwargs)``
        without ever pickling it (Finding 2 fix).

        Supports exactly two shapes:

        - A plain module-level function, called with no arguments.
        - A :func:`functools.partial` wrapping a plain module-level
          function, whose bound ``.args``/``.keywords`` become the call
          arguments.

        Anything else (a lambda, a local closure, a bound method, a
        callable object with a custom ``__call__``, a ``functools.partial``
        wrapping something that is not itself a plain module-level
        function, or one whose bound arguments are not JSON-serializable)
        raises a clear error *before* any subprocess, pickling, or
        importing machinery is touched. Unlike the previous
        ``pickle.dumps(func)``-based implementation, no pickling of ``func``
        itself ever happens in this process. And unlike an earlier
        iteration of *this* fix, resolving ``func``'s identity also never
        calls :func:`importlib.import_module` in this (host) process either
        -- :meth:`_resolve_module_level_function` validates purely via
        ``func.__globals__`` introspection, so neither attacker-controlled
        ``__reduce__``/``__reduce_ex__`` logic nor attacker-influenced
        top-level module code can ever run here. The only import of
        ``module_name`` happens later, inside the already-sandboxed child
        process.

        Raises:
            TypeError: If ``func`` (or a partial's ``.func``) is not a
                plain function object.
            ValueError: If ``func`` cannot be resolved back to itself by
                module/qualname, or its bound arguments are not
                JSON-serializable.
        """
        # `type(func) is functools.partial`, not `isinstance(func,
        # functools.partial)`: a *subclass* of functools.partial can shadow
        # the base class's `.func`/`.args`/`.keywords` getset-descriptors
        # with its own instance property (verified empirically -- a
        # subclass defining `func` as a `@property` runs its getter body,
        # unsandboxed, the moment `func.func` below is merely read). That is
        # the exact same "even introspecting this untrusted object runs
        # attacker code in the trusted host process" vulnerability class as
        # the metaclass-`__getattribute__`-on-`.__name__` issue fixed via
        # :func:`_safe_type_name` elsewhere in this module, just triggered
        # through instance-attribute override instead of a metaclass hook.
        # An exact-type identity check accepts only the real, unsubclassed
        # `functools.partial`, whose C-level descriptors cannot be
        # overridden per-instance -- a genuine partial subclass instead
        # falls through to the plain `_resolve_module_level_function(func)`
        # branch below, which itself uses `type(func) is types.FunctionType`
        # (not `isinstance`, for the same exact-type-identity reason) and so
        # correctly (and safely) rejects it.
        if type(func) is functools.partial:
            module_name, qualname = self._resolve_module_level_function(func.func)
            args: tuple[object, ...] = func.args
            kwargs: dict[str, object] = dict(func.keywords or {})
        else:
            module_name, qualname = self._resolve_module_level_function(func)
            args = ()
            kwargs = {}

        _validate_run_callable_arguments(args, kwargs, module_name, qualname)
        # The actual values (not the JSON strings validated above) are
        # written to the payload file by run_callable() so the child
        # process can json.load() them directly.

        return module_name, qualname, args, kwargs

    def run_callable(
        self,
        func: Callable[..., object],
        *,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Run ``func`` in a sandboxed subprocess by re-importing it by name.

        **Never pickles ``func`` in this (trusted, host) process.** Since
        Seatbelt sandboxes an OS process, not an in-process Python call,
        this needs some way to hand ``func`` to a fresh child interpreter --
        but calling :func:`pickle.dumps` on an arbitrary, possibly
        attacker-influenced callable *before* the sandbox exists would
        invoke that object's ``__reduce__``/``__reduce_ex__`` unsandboxed,
        which is itself an arbitrary-code-execution vector (this was
        Finding 2). Instead:

        1. ``func`` must be a plain module-level function, or a
           :func:`functools.partial` wrapping one (see
           :meth:`_decompose_callable`). Its identity is validated purely
           via ``func.__globals__`` introspection (see
           :meth:`_resolve_module_level_function`) -- this host process
           never calls :func:`importlib.import_module` on the caller's
           (freely rewritable) ``__module__`` string to do this, since that
           would import and execute attacker-influenced module code here,
           before the sandbox even exists.
        2. Any bound arguments (from the ``functools.partial``, if used)
           must be genuinely JSON-*native* -- validated via
           :func:`_validate_run_callable_arguments` (Finding 3 fix), never
           :mod:`pickle`, since pickle is unsafe for untrusted data on the
           way out of the sandbox too. This is stricter than "passes
           ``json.dumps()`` without raising": a bare ``json.dumps()`` call
           would silently *coerce* a ``tuple`` argument into a
           JSON-array-that-looks-like-a-list (so the child would
           reconstruct a ``list`` where a ``tuple`` was actually bound) and
           silently stringify non-``str`` dict keys, either of which means
           the sandboxed child could end up executing a subtly different
           call than the one requested. ``allow_nan=False`` is also passed
           at every actual serialization call site so
           NaN/Infinity/-Infinity floats raise ``ValueError`` instead of
           being written as non-standard, potentially-ambiguous JSON.
        3. ``(module_name, qualname, args, kwargs)`` is written as plain
           JSON to a small file in a temporary directory alongside a
           runner script, and :meth:`run_command` executes
           ``python <runner> <payload.json>`` under the sandbox. The
           *child* (already-sandboxed) process is the one that calls
           ``importlib.import_module(module_name)`` and resolves/calls the
           function -- exactly the "perform importing inside Seatbelt"
           structure the finding asked for.

        The runner captures the callable's return value via ``repr()`` on
        stdout, or a traceback plus a nonzero exit code if it raises. The
        temporary directory (and the harness's own runner/payload files
        within it) is always granted read access regardless of the
        caller's policy, since it is internal plumbing rather than user
        data.

        Raises:
            TypeError: If ``func`` (or a ``functools.partial``'s ``.func``)
                is not a plain function object.
            ValueError: If ``func`` cannot be resolved back to itself by
                module/qualname (e.g. a lambda, local closure, or bound
                method), or its bound arguments are not JSON-serializable.
            SandboxBackendUnavailableError: If :meth:`is_available` is False
                on this host.
        """
        self._require_available()
        active_policy = self._resolve_policy(policy)

        module_name, qualname, args, kwargs = self._decompose_callable(func)

        with tempfile.TemporaryDirectory(prefix="lazycore-sandbox-") as tmp_dir:
            payload_path = Path(tmp_dir) / "call.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "module": module_name,
                        "qualname": qualname,
                        "args": list(args),
                        "kwargs": kwargs,
                    },
                    allow_nan=False,
                )
            )

            runner_path = Path(tmp_dir) / "runner.py"
            runner_path.write_text(
                textwrap.dedent(
                    """\
                    import importlib
                    import json
                    import sys

                    with open(sys.argv[1], "r", encoding="utf-8") as f:
                        call = json.load(f)

                    module = importlib.import_module(call["module"])
                    func = module
                    for part in call["qualname"].split("."):
                        func = getattr(func, part)

                    try:
                        result = func(*call["args"], **call["kwargs"])
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        sys.exit(1)
                    else:
                        sys.stdout.write("" if result is None else repr(result))
                        sys.exit(0)
                    """
                )
            )

            # The harness's own temp files (script + JSON payload) must be
            # readable regardless of the caller's policy -- this is
            # internal plumbing, not user data, and mirrors how
            # run_command's own generated .sb profile file needs no
            # explicit allow-read rule (profile files are read by
            # sandbox-exec itself, before the sandbox is even active).
            # Unlike the pre-Finding-1 default (unrestricted reads), the
            # new default-deny-reads profile means this must always be
            # added explicitly -- it is not implied by an empty
            # allowed_read_paths anymore.
            #
            # Similarly, if the caller's policy sets a non-empty
            # `allowed_executables` (meaning "process-exec is restricted to
            # only these literal paths" -- see build_sbpl_profile), this
            # method's own internal `sys.executable <runner> <payload>`
            # invocation below must itself be added to that allowlist, or
            # this method would unconditionally fail under any caller policy
            # that restricts allowed_executables at all, regardless of what
            # the caller actually intended to allow: `run_command` re-derives
            # the SBPL profile from `effective_policy` and would deny
            # `sys.executable` itself before the runner script ever gets a
            # chance to import and call the target function. An *empty*
            # `allowed_executables` (the default, meaning "unrestricted")
            # must stay empty here -- appending `sys.executable` to an empty
            # tuple would flip "unrestricted" into "restricted to only this
            # one executable", which is not this fix's job to do.
            effective_policy = active_policy.with_overrides(
                allowed_read_paths=(*active_policy.allowed_read_paths, tmp_dir),
                allowed_executables=(
                    (*active_policy.allowed_executables, sys.executable)
                    if active_policy.allowed_executables
                    and sys.executable not in active_policy.allowed_executables
                    else active_policy.allowed_executables
                ),
            )

            return self.run_command(
                [sys.executable, str(runner_path), str(payload_path)],
                policy=effective_policy,
            )
