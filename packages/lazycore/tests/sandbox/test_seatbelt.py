"""Real tests for lazycore.sandbox.seatbelt.SeatbeltSandboxExecutor.

These tests actually invoke `/usr/bin/sandbox-exec` on the machine running
the test suite -- they are skipped (not mocked) on non-macOS hosts, since
the whole point is demonstrating real Seatbelt enforcement, not plumbing.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import pytest

from lazycore.sandbox.base import SandboxPolicy
from lazycore.sandbox.seatbelt import SeatbeltSandboxExecutor, build_sbpl_profile

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="SeatbeltSandboxExecutor only runs on macOS with /usr/bin/sandbox-exec present",
)

_FIXTURES_DIR = str(Path(__file__).parent)


def test_is_available_on_this_macos_machine():
    """is_available() returns True on this macOS host where /usr/bin/sandbox-exec is present."""
    executor = SeatbeltSandboxExecutor()
    assert executor.is_available() is True


def test_build_sbpl_profile_contains_deny_default_baseline():
    """The generated SBPL profile for a default policy has the deny-default baseline and no network-allow rule."""
    profile = build_sbpl_profile(SandboxPolicy())
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    # No allow_network -> no network* allow rule anywhere in the profile.
    assert "(allow network*)" not in profile


def test_build_sbpl_profile_never_mentions_gpu_metal_or_cocoa():
    """The generated SBPL profile never references GPU/Metal/Cocoa/MPS terms, per §2.3.1's rule that Seatbelt cannot and must not attempt to gate GPU access."""
    # Per §2.3.1: Seatbelt cannot gate GPU/Metal/Cocoa access, and this
    # backend must never pretend otherwise by emitting rules that reference
    # them.
    profile = build_sbpl_profile(
        SandboxPolicy(allow_network=True, allowed_write_paths=("/tmp",))
    )
    lowered = profile.lower()
    for forbidden_term in ("metal", "cocoa", "gpu", "mps"):
        assert forbidden_term not in lowered


def test_command_touching_allowed_write_path_succeeds():
    """A command writing inside an allowed_write_paths directory succeeds, with a clean exit and no policy_blocked flag."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir:
        target_file = str(Path(allowed_dir) / "written-by-sandbox.txt")
        policy = SandboxPolicy(allowed_write_paths=(allowed_dir,))

        result = executor.run_command(
            ["/usr/bin/touch", target_file], policy=policy
        )

        assert result.exit_code == 0, result.stderr
        assert result.policy_blocked is False
        assert Path(target_file).exists()


def test_command_writing_outside_allowed_path_is_blocked():
    """A command attempting to write to a directory outside allowed_write_paths is genuinely denied by Seatbelt: nonzero exit, file never created, policy_blocked=True."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as other_dir:
        # allowed_dir is granted write access; other_dir is a completely
        # separate real temp directory that is NOT in allowed_write_paths.
        forbidden_target = str(Path(other_dir) / "should-not-be-created.txt")
        policy = SandboxPolicy(allowed_write_paths=(allowed_dir,))

        result = executor.run_command(
            ["/usr/bin/touch", forbidden_target], policy=policy
        )

        # This is the core enforcement assertion: sandbox-exec must have
        # actually denied the write. `touch` on a Seatbelt-denied path
        # exits non-zero and reports "Operation not permitted".
        assert result.exit_code != 0
        assert not Path(forbidden_target).exists()
        assert result.policy_blocked is True
        assert "Operation not permitted" in result.stderr


def test_default_policy_denies_all_writes_anywhere():
    """With the default (empty) allowed_write_paths, a write is denied even to a directory that would otherwise seem harmless."""
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as some_dir:
        target_file = str(Path(some_dir) / "no-write-paths-configured.txt")
        # Default SandboxPolicy() has allowed_write_paths=() -- nothing is
        # writable, not even a directory that would otherwise seem "safe".
        result = executor.run_command(["/usr/bin/touch", target_file])

        assert result.exit_code != 0
        assert not Path(target_file).exists()


def test_read_only_command_succeeds_by_default():
    """Reading a system file (/etc/hosts) succeeds under the default policy, since reads are left broadly allowed unless allowed_read_paths is set."""
    executor = SeatbeltSandboxExecutor()
    # Default policy leaves reads broadly allowed (documented in
    # build_sbpl_profile's docstring) -- only writes/network are
    # restricted by default.
    result = executor.run_command(["/bin/cat", "/etc/hosts"])
    assert result.exit_code == 0
    assert result.policy_blocked is False
    assert len(result.stdout) > 0


def test_network_egress_denied_by_default():
    """A curl request to an external host fails when allow_network=False, since the profile omits the network-allow rule."""
    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(allow_network=False, timeout_seconds=10)
    result = executor.run_command(
        ["/usr/bin/curl", "-s", "--max-time", "5", "https://example.com"],
        policy=policy,
    )
    assert result.exit_code != 0


def test_run_callable_executes_picklable_function_and_returns_output():
    """run_callable() marshals a module-level picklable function into a sandboxed subprocess and returns its repr()'d output via stdout."""
    # tests/sandbox has no __init__.py (matching this package's existing
    # flat tests/ convention), so pytest's rootdir-based import mode has
    # already put this directory on sys.path to import test_seatbelt.py
    # itself as a top-level module -- `import _callable_fixtures` here
    # resolves the same way, and its __module__ (used by pickle) matches
    # what the sandboxed subprocess will resolve via PYTHONPATH below.
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    # Per the Finding-1 default-read-deny fix, the sandboxed child process
    # can no longer read _callable_fixtures.py's containing directory just
    # because PYTHONPATH points at it -- it must also be explicitly listed
    # in allowed_read_paths (bootstrap paths alone do not cover arbitrary
    # user-supplied module directories).
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )

    result = executor.run_callable(_callable_fixtures.compute_answer, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_run_callable_surfaces_exception_as_nonzero_exit():
    """A callable that raises inside the sandboxed subprocess exits nonzero and the traceback (including the exception type/message) appears in stderr."""
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    # Per the Finding-1 default-read-deny fix, the sandboxed child process
    # can no longer read _callable_fixtures.py's containing directory just
    # because PYTHONPATH points at it -- it must also be explicitly listed
    # in allowed_read_paths (bootstrap paths alone do not cover arbitrary
    # user-supplied module directories).
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )

    result = executor.run_callable(_callable_fixtures.raise_value_error, policy=policy)

    assert result.exit_code == 1
    assert "ValueError" in result.stderr
    assert "boom from sandboxed callable" in result.stderr


def test_run_callable_rejects_unpicklable_lambda():
    """run_callable() raises ValueError immediately for a lambda, since lambdas cannot be pickled for the subprocess handoff."""
    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError):
        executor.run_callable(lambda: 1)


def _install_no_subprocess_no_tempdir_tripwires(monkeypatch, *, reason: str):
    """Monkeypatch both ``subprocess.run`` and ``tempfile.TemporaryDirectory``
    (as seen from ``lazycore.sandbox.seatbelt``) to raise ``AssertionError``
    if called.

    Used by the rejection regression tests below to prove that argument
    validation happens before *any* side effect of ``run_callable()`` --
    not just before the sandboxed subprocess is spawned (the original
    guard), but also before the temporary directory/payload/runner files
    that ``run_callable()`` would otherwise create on disk (CodeRabbit
    finding: the subprocess-only tripwire proved rejection happens before
    execution, but not before filesystem side effects -- a rejected call
    should have zero footprint on disk, not just zero execution).
    """

    def _fail_if_subprocess_run_called(*args, **kwargs):
        raise AssertionError(
            f"subprocess.run() must not be invoked once argument validation "
            f"has rejected {reason}"
        )

    def _fail_if_tempdir_created(*args, **kwargs):
        raise AssertionError(
            f"tempfile.TemporaryDirectory() must not be invoked once "
            f"argument validation has rejected {reason} -- rejection must "
            "happen before any temp file/directory is created, not just "
            "before the subprocess is spawned"
        )

    monkeypatch.setattr(
        "lazycore.sandbox.seatbelt.subprocess.run", _fail_if_subprocess_run_called
    )
    monkeypatch.setattr(
        "lazycore.sandbox.seatbelt.tempfile.TemporaryDirectory",
        _fail_if_tempdir_created,
    )


def test_run_callable_rejects_tuple_bound_argument(monkeypatch):
    """A functools.partial whose bound args include a tuple raises a clear ValueError before anything is written to disk or executed -- json.dumps() would otherwise silently coerce the tuple into a JSON array indistinguishable from a list, so the sandboxed child would reconstruct a list where a tuple was actually passed (Finding 3 fix)."""
    import functools

    import _callable_fixtures  # type: ignore[import-not-found]

    # If _decompose_callable's validation were bypassed, run_callable()
    # would go on to create a temp directory and/or run_command() would
    # invoke subprocess.run() (via sandbox-exec) -- fail loudly if either
    # ever happens, so this test actually proves nothing was written to
    # disk or executed, not just that *a* ValueError was raised somewhere.
    _install_no_subprocess_no_tempdir_tripwires(
        monkeypatch, reason="a tuple bound argument"
    )

    executor = SeatbeltSandboxExecutor()
    partial_call = functools.partial(_callable_fixtures.add_numbers, (1, 2, 3))

    with pytest.raises(ValueError, match="tuple"):
        executor.run_callable(partial_call)


def test_run_callable_rejects_dict_with_non_string_key(monkeypatch):
    """A dict argument with a non-string key (e.g. {1: "a"}) is rejected the same way tuples are -- json.dumps() would otherwise silently stringify the key, changing what the sandboxed child actually sees relative to what was passed (Finding 3 fix)."""
    import functools

    import _callable_fixtures  # type: ignore[import-not-found]

    _install_no_subprocess_no_tempdir_tripwires(
        monkeypatch, reason="a dict with a non-string key"
    )

    executor = SeatbeltSandboxExecutor()
    partial_call = functools.partial(_callable_fixtures.add_numbers, {1: "a"})

    with pytest.raises(ValueError, match="non-string key"):
        executor.run_callable(partial_call)


def test_run_callable_rejects_nan_and_infinite_float_arguments(monkeypatch):
    """A float("nan") or float("inf") argument is rejected with ValueError instead of being silently written as non-standard JSON (allow_nan=False fix, Finding 3)."""
    import functools

    import _callable_fixtures  # type: ignore[import-not-found]

    _install_no_subprocess_no_tempdir_tripwires(
        monkeypatch, reason="a NaN/Infinity float"
    )

    executor = SeatbeltSandboxExecutor()

    for bad_value in (float("nan"), float("inf"), float("-inf")):
        partial_call = functools.partial(_callable_fixtures.add_numbers, bad_value)
        with pytest.raises(ValueError):
            executor.run_callable(partial_call)


def test_run_callable_still_accepts_json_native_args_end_to_end():
    """A functools.partial bound with a legitimate mix of JSON-native argument types (dict with string keys, list, str, int, float, bool, None) still executes correctly end-to-end through the real sandbox, exactly as before the stricter Finding-3 validation was added."""
    import functools

    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )

    data = {"key": "value", "nested": {"inner": 1}}
    tags = ["a", "b", 3]
    partial_call = functools.partial(
        _callable_fixtures.describe_payload,
        data,
        tags,
        "hello",
        7,
        3.5,
        True,
        None,
    )

    result = executor.run_callable(partial_call, policy=policy)

    # The runner script writes repr(result) to stdout (see run_callable()'s
    # docstring) -- describe_payload() itself returns a str, so the
    # expected stdout is repr() of *that* string, i.e. a double repr.
    expected_return_value = repr((data, tags, "hello", 7, 3.5, True, None))
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == repr(expected_return_value)


def test_run_callable_with_restrictive_allowed_executables_succeeds():
    """Regression test: run_callable() must still succeed when the caller's
    policy sets a non-empty allowed_executables that does NOT itself list
    sys.executable (e.g. a policy meant to restrict the *target* command to
    just "cat"). Before the fix, run_callable() built effective_policy by
    extending only allowed_read_paths, leaving allowed_executables exactly
    as the caller set it -- so its own internal
    `sys.executable <runner> <payload>` invocation was denied by the very
    restrictive process-exec rule the caller's policy requested, and
    run_callable() failed unconditionally for *any* caller who set a
    non-empty allowed_executables, regardless of intent."""
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
        # Deliberately does NOT include sys.executable -- if run_callable()
        # failed to extend this internally, sandbox-exec would deny the
        # runner's own `python <runner.py> <payload.json>` exec attempt.
        allowed_executables=("cat",),
    )

    result = executor.run_callable(_callable_fixtures.compute_answer, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_run_callable_with_empty_allowed_executables_is_unaffected():
    """No-regression check: the default, empty allowed_executables (meaning
    "process-exec is unrestricted") must remain unrestricted after the fix
    -- run_callable() must not add sys.executable to an empty tuple, which
    would silently turn "no restriction" into "restricted to only this one
    executable" and could affect any *other* process-exec the sandboxed
    child might legitimately need to perform."""
    import _callable_fixtures  # type: ignore[import-not-found]

    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(
        inherit_env=True,
        env={"PYTHONPATH": _FIXTURES_DIR},
        allowed_read_paths=(_FIXTURES_DIR,),
    )
    assert policy.allowed_executables == ()

    result = executor.run_callable(_callable_fixtures.compute_answer, policy=policy)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "42"


def test_bare_executable_name_resolves_via_path_and_is_actually_usable():
    """A bare (no-path) allowed_executables entry like "touch" is resolved via PATH (not against CWD) and the resulting profile actually permits running the real executable -- regression test for the finding that _canonical("python3")-style resolution silently produced a bogus <cwd>/touch path that could never match the real binary."""
    import shutil

    resolved_touch = shutil.which("touch")
    assert resolved_touch is not None, "test requires `touch` to be on PATH"

    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir:
        target_file = str(Path(allowed_dir) / "touched-via-bare-name.txt")
        policy = SandboxPolicy(
            allowed_write_paths=(allowed_dir,),
            allowed_executables=("touch",),
        )

        # Sanity check: the generated profile actually contains the real,
        # canonical, resolved-via-PATH executable path, not a CWD-relative
        # bogus one.
        profile = build_sbpl_profile(policy)
        assert str(Path(resolved_touch).resolve()) in profile
        assert f'(literal "{Path.cwd()}/touch")' not in profile

        result = executor.run_command(["touch", target_file], policy=policy)

        assert result.exit_code == 0, result.stderr
        assert Path(target_file).exists()


def test_bare_unresolvable_executable_name_raises_clear_value_error():
    """An allowed_executables entry that cannot be resolved via PATH raises ValueError at profile-build time rather than silently emitting a bogus, never-matching CWD-relative path."""
    policy = SandboxPolicy(
        allowed_executables=("definitely-not-a-real-executable-xyz123",)
    )
    with pytest.raises(ValueError, match="could not be resolved via PATH"):
        build_sbpl_profile(policy)


def test_full_path_executable_entry_is_unaffected_by_bare_name_fix():
    """An allowed_executables entry that is already a full path (contains a path separator) continues to be resolved via Path.resolve(), exactly as before -- the bare-name PATH-resolution fix only changes behavior for bare names."""
    policy = SandboxPolicy(allowed_executables=("/usr/bin/touch",))
    profile = build_sbpl_profile(policy)
    assert str(Path("/usr/bin/touch").resolve()) in profile


def test_get_default_executor_returns_seatbelt_on_this_machine():
    """get_default_executor() returns an available SeatbeltSandboxExecutor on this macOS host."""
    from lazycore.sandbox import get_default_executor

    executor = get_default_executor()
    assert isinstance(executor, SeatbeltSandboxExecutor)
    assert executor.is_available() is True
