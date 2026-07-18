"""Regression tests for two CRITICAL CodeRabbit findings on
`lazycore.sandbox` (macOS Seatbelt backend), distinct from (and layered on
top of) the earlier three findings covered by `test_hardening.py`:

1. An empty ``allowed_read_paths`` used to generate an unconditional,
   unrestricted ``(allow file-read*)`` SBPL rule -- effectively granting the
   sandboxed process read access to the entire host filesystem (SSH keys,
   tokens, project secrets, ...) rather than denying it. Fixed in
   ``build_sbpl_profile`` by always restricting reads to a small, fixed
   bootstrap set plus whatever ``allowed_read_paths`` explicitly lists.
2. ``run_callable()`` used to call ``pickle.dumps(func)`` on the
   caller-supplied callable directly in the trusted host process, before
   the sandbox boundary existed -- invoking attacker-controlled
   ``__reduce__``/``__reduce_ex__`` unsandboxed. Fixed by requiring a
   module-level function (or a ``functools.partial`` wrapping one with
   JSON-serializable bound arguments), resolved and called by name *inside*
   the sandboxed child process, with the callable object itself never
   pickled at all.

All tests here perform a real ``/usr/bin/sandbox-exec`` invocation (no
mocks) and are skipped on non-macOS hosts, matching this package's existing
convention in `test_seatbelt.py`/`test_hardening.py`.
"""

from __future__ import annotations

import functools
import platform
import tempfile
from pathlib import Path

import pytest

from lazycore.sandbox.base import SandboxPolicy
from lazycore.sandbox.seatbelt import SeatbeltSandboxExecutor, build_sbpl_profile

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="Requires macOS with /usr/bin/sandbox-exec present",
)

_FIXTURES_DIR = str(Path(__file__).parent)


# ---------------------------------------------------------------------------
# Finding 1: empty allowed_read_paths must DENY reads outside a narrow
# bootstrap set, not grant unrestricted host filesystem access.
# ---------------------------------------------------------------------------


def test_empty_allowed_read_paths_blocks_read_of_real_out_of_bootstrap_file():
    """The core Finding-1 regression test: with the default (empty)
    allowed_read_paths, reading a real file in a temp directory that is
    neither in the hardcoded bootstrap set nor explicitly granted is
    genuinely denied by Seatbelt -- proving the fix closed the
    "empty allowlist grants the entire host filesystem" hole, not just that
    the generated profile text looks different.
    """
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as tmp_dir:
        secret_file = Path(tmp_dir) / "totally-not-an-ssh-key.txt"
        secret_file.write_text("super-secret-value-should-not-leak")

        # Default SandboxPolicy() -- allowed_read_paths=() -- must NOT be
        # able to read this file, since tmp_dir is not part of the fixed
        # bootstrap set.
        result = executor.run_command(["/bin/cat", str(secret_file)])

        assert result.exit_code != 0
        assert "super-secret-value-should-not-leak" not in result.stdout
        assert result.policy_blocked is True


def test_read_explicitly_granted_via_allowed_read_paths_still_succeeds():
    """The positive counterpart: a file the caller explicitly lists via
    allowed_read_paths (even though it's outside the bootstrap set) IS
    readable -- the fix denies by default, it does not deny everything
    unconditionally.
    """
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as tmp_dir:
        granted_file = Path(tmp_dir) / "explicitly-granted.txt"
        granted_file.write_text("this should be readable")

        policy = SandboxPolicy(allowed_read_paths=(tmp_dir,))
        result = executor.run_command(["/bin/cat", str(granted_file)], policy=policy)

        assert result.exit_code == 0, result.stderr
        assert result.policy_blocked is False
        assert result.stdout.strip() == "this should be readable"


def test_build_sbpl_profile_never_emits_bare_unrestricted_file_read_allow():
    """The generated profile must never contain a bare, unqualified
    ``(allow file-read*)`` with no subpath scoping -- that exact string was
    the literal manifestation of Finding 1. The read rule must always carry
    at least one ``subpath`` clause (the bootstrap set, at minimum).
    """
    profile = build_sbpl_profile(SandboxPolicy())
    for line in profile.splitlines():
        stripped = line.strip()
        assert stripped != "(allow file-read*)", (
            "build_sbpl_profile emitted an unconditional, unrestricted "
            "file-read* rule with no subpath scoping -- this is exactly "
            "the Finding-1 vulnerability."
        )
    assert "(allow file-read*" in profile
    assert "(subpath " in profile


def test_bootstrap_reads_never_grant_the_whole_home_directory():
    """The hardcoded bootstrap set must never grant the user's *entire*
    home directory -- only backend-owned system paths needed to start a
    process (which, on some Python installs, e.g. pyenv/uv-managed
    interpreters, may legitimately be a narrow, specific subdirectory
    *under* the home directory, such as the interpreter's own install
    prefix -- that is fine; a bare, recursive grant of the whole home
    directory is not). This directly guards the "no user-data read access"
    part of the fix's contract.
    """
    profile = build_sbpl_profile(SandboxPolicy())
    home = str(Path.home())
    assert f'(subpath "{home}")' not in profile


def test_bootstrap_reads_do_not_include_an_arbitrary_temp_directory():
    """A generic, unrelated temp directory (not the running interpreter's
    own install prefix) must never appear in the generated profile's read
    section -- confirming the bootstrap set is a narrow, fixed list, not
    something that accidentally widens to cover arbitrary filesystem
    locations.
    """
    profile = build_sbpl_profile(SandboxPolicy())
    with tempfile.TemporaryDirectory() as unrelated_dir:
        assert str(Path(unrelated_dir).resolve()) not in profile


# ---------------------------------------------------------------------------
# Finding 2: run_callable() must never pickle the caller-supplied callable
# in the host process; only module-level functions (optionally wrapped in
# functools.partial with JSON-safe args) are supported.
# ---------------------------------------------------------------------------


def test_run_callable_module_level_function_with_json_safe_partial_args_works():
    """End-to-end: a functools.partial wrapping a module-level function with
    simple JSON-serializable positional args executes successfully through
    the real sandbox -- the new (module, qualname, args, kwargs) calling
    convention works end-to-end.
    """
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )

    bound = functools.partial(_callable_fixtures.add_numbers, 19, b=23)
    result = executor.run_callable(bound, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_run_callable_rejects_lambda_without_ever_pickling_it():
    """A lambda cannot satisfy the module/qualname-resolution contract and
    must raise a clear error -- never silently fall back to pickling it.
    """
    executor = SeatbeltSandboxExecutor()
    with pytest.raises((ValueError, TypeError)):
        executor.run_callable(lambda: 1)


def test_run_callable_rejects_partial_with_non_json_serializable_args():
    """A functools.partial wrapping a valid module-level function, but with
    a bound argument that is not JSON-serializable (an arbitrary object),
    must raise a clear error rather than silently pickling it.
    """
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()

    class NotJsonSerializable:
        pass

    bound = functools.partial(_callable_fixtures.add_numbers, NotJsonSerializable(), b=1)
    with pytest.raises((ValueError, TypeError)):
        executor.run_callable(bound)


def test_run_callable_rejects_partial_wrapping_a_lambda():
    """A functools.partial wrapping a lambda (rather than a module-level
    function) must also be rejected -- the module-level-function
    requirement applies to the partial's underlying .func, not just to a
    bare callable passed directly.
    """
    executor = SeatbeltSandboxExecutor()
    bound = functools.partial(lambda x: x, 1)
    with pytest.raises((ValueError, TypeError)):
        executor.run_callable(bound)


def test_run_callable_never_calls_pickle_dumps_on_the_callable(monkeypatch):
    """Direct proof the fix removed pickling of the caller's callable: patch
    ``pickle.dumps`` to raise if invoked at all during a successful
    run_callable() call, then confirm the call still succeeds -- if
    anything in the run_callable() path still pickled the callable, this
    would fail loudly instead of silently passing.
    """
    import pickle

    import _callable_fixtures  # type: ignore[import-not-found]

    def _boom(*args, **kwargs):
        raise AssertionError(
            "pickle.dumps() must never be called on the caller-supplied "
            "callable by run_callable() (Finding 2)"
        )

    monkeypatch.setattr(pickle, "dumps", _boom)

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )
    result = executor.run_callable(_callable_fixtures.compute_answer, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"
