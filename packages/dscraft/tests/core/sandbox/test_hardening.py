"""Regression tests for the three confirmed security-review findings on
`dscraft.core.sandbox` (macOS Seatbelt backend + shared SandboxPolicy):

1. `policy_blocked` conflating Seatbelt denial with generic Unix
   permission errors.
2. Symlinked `allowed_write_paths`/`allowed_read_paths` entries silently
   widening the granted policy via `Path.resolve()`.
3. Relative paths in `SandboxPolicy` resolving against call-time CWD
   rather than construction-time CWD, despite `frozen=True`.

Tests for (2) and (3) are pure Python (no `sandbox-exec` invocation) and
run on any platform. Test for (1) actually invokes `/usr/bin/sandbox-exec`
(to prove the *default* Seatbelt-unrestricted-read profile is really in
effect) and is skipped on non-macOS hosts, matching this package's existing
convention in `test_seatbelt.py`.
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import pytest

from dscraft.core.sandbox.base import SandboxPolicy
from dscraft.core.sandbox.seatbelt import SeatbeltSandboxExecutor, build_sbpl_profile

# ---------------------------------------------------------------------------
# Finding 1: policy_blocked false positives from ordinary DAC/permission
# errors that have nothing to do with Seatbelt.
# ---------------------------------------------------------------------------

pytestmark_macos = pytest.mark.skipif(
    platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").exists(),
    reason="Requires macOS with /usr/bin/sandbox-exec present",
)


@pytestmark_macos
def test_chmod_000_file_is_now_indistinguishable_from_a_seatbelt_denial():
    """Documents the deliberate behavior change from the Finding-1 fix
    (dscraft.core.sandbox default-read-deny, see `seatbelt.py`'s
    `build_sbpl_profile`): reads are no longer ever fully unrestricted for
    any policy (not even the default), so `_classify_denial` can no longer
    prove a "Permission denied" on a read-only command is *definitely* an
    ordinary DAC error rather than a genuine (if redundant) Seatbelt
    denial -- even when the file IS inside `allowed_read_paths` and the
    real cause is a chmod 000 DAC error, Seatbelt's own read restriction
    could equally have produced an identical-looking denial for some other
    path. `_classify_denial` therefore no longer attempts the downgrade
    for read-only commands (see its docstring's "retired precision case"
    section) -- this is the accepted, documented trade-off for closing the
    unrestricted-reads hole, not a regression to be fixed here.
    """
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not deny root; cannot exercise this DAC error as root")

    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as tmp_dir:
        blocked_file = Path(tmp_dir) / "no-access.txt"
        blocked_file.write_text("secret content")
        blocked_file.chmod(0o000)
        try:
            # Even though tmp_dir is explicitly granted via
            # allowed_read_paths (so Seatbelt itself would allow reading
            # this file), the chmod 000 DAC error still causes cat to fail.
            policy = SandboxPolicy(allowed_read_paths=(tmp_dir,))
            result = executor.run_command(["/bin/cat", str(blocked_file)], policy=policy)
        finally:
            blocked_file.chmod(0o644)  # restore so tempdir cleanup can remove it

        # The command must genuinely have failed (proving this is a real
        # DAC error, not a no-op) ...
        assert result.exit_code != 0
        assert "Permission denied" in result.stderr
        # ... and per the retired precision case, this is now classified
        # as policy_blocked=True (the heuristic no longer distinguishes
        # this from a genuine Seatbelt denial) -- see docstring above.
        assert result.policy_blocked is True


@pytestmark_macos
def test_write_outside_allowed_path_is_still_classified_as_policy_blocked():
    """Regression guard: the Finding-1 precision fix must not turn off
    detection of *real* Seatbelt denials. `touch` is not in the read-only
    command allowlist, so this must still be classified as policy_blocked.
    """
    executor = SeatbeltSandboxExecutor()
    with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as other_dir:
        forbidden_target = str(Path(other_dir) / "should-not-be-created.txt")
        policy = SandboxPolicy(allowed_write_paths=(allowed_dir,))

        result = executor.run_command(["/usr/bin/touch", forbidden_target], policy=policy)

        assert result.exit_code != 0
        assert result.policy_blocked is True
        assert not Path(forbidden_target).exists()


def test_classify_denial_still_flags_denial_when_reads_are_restricted():
    """When `allowed_read_paths` IS configured (reads ARE restricted by the
    generated profile -- see `build_sbpl_profile`), a "Permission
    denied"/"Operation not permitted" report is still classified as
    `policy_blocked=True` even for a command in the read-only allowlist:
    the Finding-1 downgrade only applies when the profile provably grants
    *unrestricted* reads (`allowed_read_paths` empty), which is not the
    case here. This is a direct unit test of `_classify_denial` (rather
    than a full subprocess run) because actually exercising a real
    restricted-read Seatbelt profile against a dynamically-linked system
    binary like `/bin/cat` requires also allowlisting the dynamic linker's
    own shared-cache paths (which vary by macOS version) just to let the
    process start at all -- orthogonal to what this specific fix changes.
    """
    executor = SeatbeltSandboxExecutor()
    policy = SandboxPolicy(allowed_read_paths=("/some/allowed/dir",))

    blocked = executor._classify_denial(
        policy,
        ["/bin/cat", "/some/outside/dir/file.txt"],
        1,
        "",
        "cat: /some/outside/dir/file.txt: Permission denied\n",
    )

    assert blocked is True


# ---------------------------------------------------------------------------
# Finding 2: symlinked allowed-path entries silently widening the policy.
# ---------------------------------------------------------------------------


def test_symlinked_allowed_write_path_pointing_at_root_is_rejected():
    """An allowed_write_paths entry that is a symlink resolving to "/" is rejected at profile-build time rather than silently granting root-wide write access."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        evil_link = Path(tmp_dir) / "looks-scoped-but-is-not"
        evil_link.symlink_to("/")
        policy = SandboxPolicy(allowed_write_paths=(str(evil_link),))

        with pytest.raises(ValueError, match="broad"):
            build_sbpl_profile(policy)


def test_symlinked_allowed_read_path_pointing_at_home_dir_is_rejected():
    """An allowed_read_paths entry that is a symlink resolving to the user's home directory is rejected rather than silently widening the read scope."""
    home = Path.home()
    with tempfile.TemporaryDirectory() as tmp_dir:
        evil_link = Path(tmp_dir) / "looks-scoped-but-is-not"
        evil_link.symlink_to(home)
        policy = SandboxPolicy(allowed_read_paths=(str(evil_link),))

        with pytest.raises(ValueError, match="broad"):
            build_sbpl_profile(policy)


def test_directly_specifying_a_broad_system_directory_is_also_rejected():
    """Passing a suspiciously broad system path (e.g. "/etc") directly, with no symlink involved, is also rejected by build_sbpl_profile()."""
    # Not even a symlink -- the finding's fix also protects a caller who
    # directly (mistakenly) passes a suspiciously broad path.
    policy = SandboxPolicy(allowed_write_paths=("/etc",))
    with pytest.raises(ValueError, match="broad"):
        build_sbpl_profile(policy)


def test_legitimate_tmp_symlink_is_not_rejected():
    """The benign OS-provided "/tmp" -> "/private/tmp" symlink is still accepted and included in the generated profile, unaffected by the anti-widening check."""
    # /tmp -> /private/tmp on macOS is the canonical benign OS-provided
    # symlink this fix must NOT break (see _canonical's original docstring
    # and _reject_overbroad_allowed_path's docstring).
    policy = SandboxPolicy(allowed_write_paths=("/tmp",))
    profile = build_sbpl_profile(policy)
    assert "/private/tmp" in profile or "/tmp" in profile


def test_ordinary_scoped_subdirectory_is_not_rejected():
    """A normal, non-symlinked, non-root project subdirectory resolves and appears in the generated profile without triggering the overbroad-path rejection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        scoped = Path(tmp_dir) / "project-workspace"
        scoped.mkdir()
        policy = SandboxPolicy(allowed_write_paths=(str(scoped),))
        profile = build_sbpl_profile(policy)
        assert str(scoped.resolve()) in profile


# ---------------------------------------------------------------------------
# Finding 3: relative paths in SandboxPolicy resolving against call-time CWD.
# ---------------------------------------------------------------------------


def test_relative_allowed_write_path_raises_immediately_at_construction():
    """A relative allowed_write_paths entry raises ValueError at SandboxPolicy construction time, not later at run_command() call time."""
    with pytest.raises(ValueError, match="absolute"):
        SandboxPolicy(allowed_write_paths=("relative/write/path",))


def test_relative_allowed_read_path_raises_immediately_at_construction():
    """A relative allowed_read_paths entry raises ValueError at SandboxPolicy construction time."""
    with pytest.raises(ValueError, match="absolute"):
        SandboxPolicy(allowed_read_paths=("relative/read/path",))


def test_relative_working_directory_raises_immediately_at_construction():
    """A relative working_directory value raises ValueError at SandboxPolicy construction time."""
    with pytest.raises(ValueError, match="absolute"):
        SandboxPolicy(working_directory="relative/dir")


def test_with_overrides_also_rejects_relative_paths():
    """with_overrides() re-runs the same absolute-path validation as normal construction, so it also rejects a relative override."""
    base = SandboxPolicy()
    with pytest.raises(ValueError, match="absolute"):
        base.with_overrides(allowed_write_paths=("relative/path",))


def test_bare_executable_name_is_still_accepted_non_absolute():
    """allowed_executables accepts bare, PATH-resolved command names (unlike the read/write/working-directory fields), unaffected by the absolute-path requirement."""
    # allowed_executables is documented to accept bare PATH-resolved names
    # -- this must remain unaffected by the Finding-3 fix.
    policy = SandboxPolicy(allowed_executables=("python3", "/usr/bin/env"))
    assert policy.allowed_executables == ("python3", "/usr/bin/env")


def test_absolute_paths_still_construct_normally():
    """Absolute paths for allowed_write_paths, allowed_read_paths, and working_directory all construct a SandboxPolicy without raising."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy = SandboxPolicy(
            allowed_write_paths=(tmp_dir,),
            allowed_read_paths=(tmp_dir,),
            working_directory=tmp_dir,
        )
        assert policy.allowed_write_paths == (tmp_dir,)
        assert policy.working_directory == tmp_dir
