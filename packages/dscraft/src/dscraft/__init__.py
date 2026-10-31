"""DSCraft: a unified, Apache-2.0-licensed, local-first ML tooling platform.

This is the top-level ``dscraft`` package. Its subpackages (``dscraft.core``,
and eventually ``dscraft.automl``, ``dscraft.clean``, ``dscraft.forecast``,
``dscraft.graph``, ``dscraft.vision``, ``dscraft.tune``, ``dscraft.security``,
``dscraft.agent``) are the real implementations, not re-export shims --
``dscraft`` is one real, installable package rather than a pile of
independently-versioned packages.

``dscraft.core`` is the only subpackage with a hard, always-installed
dependency (``opentelemetry-api``); every other subpackage is expected to be
optional-extras-gated once it lands, so this top-level module deliberately
does not eagerly import any subpackage here.

Unified-backend contract (tabular/forecast/eda groups, this round): a
consumer depends only on ``dscraft`` plus the capability extras it needs and
gets a tested, mutually-compatible data-science stack -- it never
hand-assembles numpy/pandas/scikit-learn/statsforecast/etc. versions itself.
The known-good windows are declared in ``pyproject.toml`` (install time) and
mirrored in ``dscraft._versions`` (run time). :func:`validate_environment`
checks the active environment against them and raises a clear, actionable
error on a mismatch, so a subtly incompatible env fails loudly up front
instead of deep inside a subpackage with a cryptic traceback.

Like ``dscraft._versions``, this module imports only the standard library,
so ``import dscraft`` and ``dscraft.validate_environment(...)`` never pull in
any heavy dependency.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from dscraft._versions import ANALYTICS_GROUPS, KNOWN_GOOD, _within

__version__ = version("dscraft")

__all__ = [
    "ANALYTICS_GROUPS",
    "EnvironmentValidationError",
    "__version__",
    "validate_environment",
]


class EnvironmentValidationError(RuntimeError):
    """The installed dependency versions are outside dscraft's known-good set.

    Raised by :func:`validate_environment`. The message lists every offending
    distribution with what is installed versus the tested window, plus a
    reinstall hint, so the problem is actionable without reading source.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        joined = "\n  - ".join(problems)
        super().__init__(
            "dscraft found dependency versions outside its tested, "
            "conflict-free set:\n  - "
            + joined
            + "\n\nReinstall the matched stack with, e.g., "
            '`pip install "dscraft[analytics]"` (or the specific capability '
            "extra you need) so versions are resolved from dscraft's declared "
            "ranges. See dscraft._versions.KNOWN_GOOD for the exact windows."
        )


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def validate_environment(
    groups: str | list[str] | tuple[str, ...] | None = None,
    *,
    require_installed: bool = True,
) -> dict[str, str]:
    """Check installed versions against dscraft's known-good set.

    Args:
        groups: which capability group(s) to validate -- a single group
            name (``"tabular"``, ``"forecast"``, ``"eda"``), an iterable of
            them, or ``None`` to validate the full ``analytics`` set (all
            three shipped this round). Unknown group names raise
            ``ValueError``.
        require_installed: when ``True`` (default), a distribution that is
            not installed at all counts as a problem -- use this when the
            caller depends on the group and expects it present. When
            ``False``, a missing distribution is skipped and only
            *present-but-out-of-range* versions are reported, which is
            useful for validating a partial environment.

    Returns:
        A mapping of ``distribution -> installed version`` for every
        checked distribution that was present and in range. (Primarily
        informational; the important signal is whether it raises.)

    Raises:
        ValueError: an unknown capability group name was passed.
        EnvironmentValidationError: at least one checked distribution is
            installed with a version outside the known-good window, or
            (when ``require_installed``) is missing entirely.
    """
    if groups is None:
        selected: tuple[str, ...] = ANALYTICS_GROUPS
    elif isinstance(groups, str):
        selected = (groups,)
    else:
        selected = tuple(groups)

    unknown = [g for g in selected if g not in KNOWN_GOOD]
    if unknown:
        raise ValueError(
            f"Unknown dscraft capability group(s): {sorted(unknown)!r}. "
            f"Known groups: {sorted(KNOWN_GOOD)!r}."
        )

    problems: list[str] = []
    ok: dict[str, str] = {}
    # De-duplicate across groups (numpy/pandas/scipy recur) so a shared
    # distribution is reported at most once.
    seen: set[str] = set()
    for group in selected:
        for spec in KNOWN_GOOD[group]:
            if spec.distribution in seen:
                continue
            seen.add(spec.distribution)
            installed = _installed_version(spec.distribution)
            if installed is None:
                if require_installed:
                    problems.append(
                        f"{spec.distribution}: not installed "
                        f"(need >={spec.minimum},<{spec.below})"
                    )
                continue
            if _within(installed, spec.minimum, spec.below):
                ok[spec.distribution] = installed
            else:
                problems.append(
                    f"{spec.distribution}: installed {installed}, "
                    f"outside tested range >={spec.minimum},<{spec.below}"
                )

    if problems:
        raise EnvironmentValidationError(problems)
    return ok
