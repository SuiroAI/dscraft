"""Single source of truth for dscraft's known-good dependency version set.

DSCraft's packaging goal is "one dependency, conflict-free versions": a
consumer depends only on ``dscraft`` (plus the capability extras it needs)
and never has to hand-assemble a mutually-compatible data-science stack
itself. The ranges declared per capability group in ``pyproject.toml`` are
the installer-honored expression of that guarantee; this module is the
machine-readable companion used by :func:`dscraft.validate_environment` to
check, at runtime, that the *actually installed* versions fall inside the
tested, co-resolvable window for whatever capabilities the caller uses.

Reconciled (first-party rebuild): this merges the capability-group /
``validate_environment`` machinery introduced on the unified-backend branch
(PR #1) with the first-party AutoML/EDA rebuild (PR #2). The third-party
gradient-boosting backends (xgboost / lightgbm / catboost) and
``imbalanced-learn`` are REMOVED from the ``tabular`` window: their
capability is now first-party (``dscraft.automl`` DS* backends and
``dscraft.automl._resampling``), so no boosting or resampling library is a
dependency of the tabular group anymore. scikit-learn stays (estimator
protocol/plumbing plus HDBSCAN until its first-party replacement lands).

Scope (this round): the TABULAR, FORECAST and EDA capability groups only.
The torch-based groups (graph/vision/tune) and the automl-onnx export path
are tracked separately (Linear DSC-61 / DSC-62) and are intentionally not
represented here yet.

This module imports only the standard library, so importing it (and the
top-level ``dscraft`` package that exposes it) never pulls in any heavy
dependency -- the lean-core import contract is preserved.
"""

from __future__ import annotations

from typing import NamedTuple


class VersionRange(NamedTuple):
    """A tested, co-resolvable version window for one distribution.

    ``minimum`` is inclusive; ``below`` is the first *excluded* version
    (i.e. an exclusive upper bound), mirroring the ``>=min,<below`` form
    used in ``pyproject.toml`` so the two stay trivially comparable.
    """

    distribution: str
    minimum: str
    below: str


#: Capability group -> distributions whose installed version must fall
#: inside the known-good window. Keys match the ``pyproject.toml`` extra
#: names for the three groups shipped this round. A distribution that
#: legitimately appears in more than one group (numpy, pandas, scipy) is
#: listed once per group with the SAME window, so any single group can be
#: validated in isolation.
#:
#: NOTE (first-party reconciliation): the ``tabular`` group no longer lists
#: xgboost / lightgbm / catboost / imbalanced-learn -- those third-party
#: backends were removed by the first-party rebuild. Boosting is now the
#: first-party ``DSBoost`` backend and resampling is first-party, so neither
#: is an installed dependency to validate.
KNOWN_GOOD: dict[str, tuple[VersionRange, ...]] = {
    "tabular": (
        VersionRange("numpy", "1.26", "2.2"),            # resolved: 2.1.3
        VersionRange("pandas", "2.1", "2.3"),            # resolved: 2.2.3
        VersionRange("scikit-learn", "1.4", "1.6"),      # resolved: 1.5.2
        VersionRange("scipy", "1.11", "1.14"),           # resolved: 1.13.1
    ),
    "forecast": (
        VersionRange("numpy", "1.26", "2.2"),            # resolved: 2.1.3
        VersionRange("pandas", "2.1", "2.3"),            # resolved: 2.2.3
        VersionRange("scipy", "1.11", "1.14"),           # resolved: 1.13.1
        VersionRange("statsforecast", "1.7", "2.1"),     # resolved: 2.0.3
        VersionRange("statsmodels", "0.14", "0.15"),     # resolved: 0.14.6
        VersionRange("utilsforecast", "0.2", "0.3"),     # resolved: 0.2.17
    ),
    "eda": (
        VersionRange("numpy", "1.26", "2.2"),            # resolved: 2.1.3
        VersionRange("scipy", "1.11", "1.14"),           # resolved: 1.13.1
        VersionRange("polars", "0.20", "1.1"),           # resolved: 1.0.0
        VersionRange("datasketches", "5.0", "6"),        # resolved: 5.2.0
    ),
}

#: Convenience alias: the full set of groups shipped this round, i.e. what
#: ``dscraft[analytics]`` installs. Kept in sync with ``pyproject.toml``'s
#: ``analytics`` extra.
ANALYTICS_GROUPS: tuple[str, ...] = ("tabular", "forecast", "eda")


def _parse(version: str) -> tuple[int, ...]:
    """Parse a dotted release string into a comparable integer tuple.

    Only the leading numeric release segment is considered; any pre/post
    /dev/local suffix (``1.2.0rc1``, ``2.0.0.post1``, ``1.5+local``) is
    truncated at the first non-numeric component. This is deliberately a
    small, dependency-free comparison adequate for the ``>=min,<below``
    window check -- it is not a full PEP 440 implementation, and it never
    imports ``packaging`` (which would not be guaranteed present on a bare
    base install).
    """
    release: list[int] = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        release.append(int(digits))
    return tuple(release) or (0,)


def _within(installed: str, low: str, high: str) -> bool:
    """Return whether ``low <= installed < high`` on parsed release tuples."""
    return _parse(low) <= _parse(installed) < _parse(high)
