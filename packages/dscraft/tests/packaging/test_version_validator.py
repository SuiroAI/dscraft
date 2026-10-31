"""Tests for dscraft's known-good version set and runtime validator.

These exercise the "one dependency, conflict-free versions" contract's
runtime half -- the pure-stdlib ``dscraft._versions`` windows and
``dscraft.validate_environment`` checker. They need NO heavy dependency
installed (they drive the checker with injected specs and monkeypatched
lookups), so they run on a bare ``dscraft[dev]`` install, matching this
repo's lean-core testing posture. No ``sys.modules`` inspection is used
(see tests/clean/test_embeddings.py for why that pattern was abandoned
here).
"""

from __future__ import annotations

import pytest

import dscraft
from dscraft import _versions
from dscraft._versions import KNOWN_GOOD, VersionRange, _parse, _within

# --- pure version-string parsing / window logic ---------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("2.1.3", (2, 1, 3)),
        ("1", (1,)),
        ("1.2.0rc1", (1, 2, 0)),      # pre-release suffix truncated
        ("2.0.0.post1", (2, 0, 0)),   # post-release suffix truncated
        ("1.5+local", (1, 5)),        # local version segment truncated
        ("", (0,)),                   # degenerate input -> (0,)
        ("x.y.z", (0,)),              # non-numeric -> (0,)
    ],
)
def test_parse_release_tuple(text, expected):
    assert _parse(text) == expected


@pytest.mark.parametrize(
    "installed, low, high, inside",
    [
        ("2.1.3", "1.26", "2.2", True),
        ("1.26.0", "1.26", "2.2", True),   # lower bound inclusive
        ("1.25.9", "1.26", "2.2", False),  # just below
        ("2.2.0", "1.26", "2.2", False),   # upper bound exclusive
        ("2.1.99", "1.26", "2.2", True),
        ("3", "2.0", "3", False),          # upper exclusive on bare major
        ("2.0", "2.0", "3", True),
    ],
)
def test_within_window(installed, low, high, inside):
    assert _within(installed, low, high) is inside


# --- known-good set shape --------------------------------------------------

def test_known_good_groups_present():
    assert set(KNOWN_GOOD) == {"tabular", "forecast", "eda"}
    assert _versions.ANALYTICS_GROUPS == ("tabular", "forecast", "eda")


def test_shared_distributions_use_identical_windows_across_groups():
    """numpy/pandas/scipy recur across groups; the windows must match so any
    combination of groups co-resolves (and de-dup in the validator is safe)."""
    windows: dict[str, set[tuple[str, str]]] = {}
    for specs in KNOWN_GOOD.values():
        for spec in specs:
            windows.setdefault(spec.distribution, set()).add((spec.minimum, spec.below))
    for dist in ("numpy", "pandas", "scipy"):
        if dist in windows:
            assert len(windows[dist]) == 1, f"{dist} has divergent windows: {windows[dist]}"


# --- validate_environment behavior ----------------------------------------

def test_unknown_group_raises_value_error():
    with pytest.raises(ValueError, match="Unknown dscraft capability group"):
        dscraft.validate_environment("graph")  # out of scope this round


def test_missing_distribution_required_vs_optional(monkeypatch):
    """A distribution absent from the environment is a problem by default,
    but skipped when require_installed=False."""
    monkeypatch.setitem(
        KNOWN_GOOD,
        "tabular",
        (VersionRange("definitely-not-installed-xyz", "1.0", "2.0"),),
    )
    with pytest.raises(dscraft.EnvironmentValidationError) as exc:
        dscraft.validate_environment("tabular")
    assert "definitely-not-installed-xyz" in str(exc.value)
    # require_installed=False: missing dist is ignored, nothing in range -> empty ok
    assert dscraft.validate_environment("tabular", require_installed=False) == {}


def test_out_of_range_version_reported(monkeypatch):
    """A present-but-too-old version is reported with installed vs. window."""
    monkeypatch.setattr(_versions, "KNOWN_GOOD", dict(KNOWN_GOOD), raising=True)
    monkeypatch.setitem(
        _versions.KNOWN_GOOD,
        "tabular",
        (VersionRange("pretend-pkg", "2.0", "3.0"),),
    )
    # Also patch the module-level reference the validator closed over.
    monkeypatch.setattr(dscraft, "KNOWN_GOOD", _versions.KNOWN_GOOD, raising=True)
    monkeypatch.setattr(dscraft, "_installed_version", lambda d: "1.9.0")
    with pytest.raises(dscraft.EnvironmentValidationError) as exc:
        dscraft.validate_environment("tabular")
    msg = str(exc.value)
    assert "pretend-pkg" in msg and "1.9.0" in msg and ">=2.0,<3.0" in msg


def test_in_range_version_returned(monkeypatch):
    monkeypatch.setattr(dscraft, "KNOWN_GOOD", {"tabular": (VersionRange("pretend-pkg", "2.0", "3.0"),)})
    monkeypatch.setattr(dscraft, "_installed_version", lambda d: "2.5.1")
    assert dscraft.validate_environment("tabular") == {"pretend-pkg": "2.5.1"}


def test_shared_distribution_checked_once(monkeypatch):
    """numpy appears in all three groups; validating the union must not
    double-report it."""
    calls: list[str] = []

    def fake_installed(dist: str):
        calls.append(dist)
        return "1.0.0"  # out of every real window -> forces a problem per dist

    monkeypatch.setattr(dscraft, "_installed_version", fake_installed)
    with pytest.raises(dscraft.EnvironmentValidationError):
        dscraft.validate_environment(None)  # analytics = all three groups
    assert calls.count("numpy") == 1
