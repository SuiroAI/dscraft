"""License-isolation policy and model-licensing allowlist (§2.2, §2.10).

This module encodes two related, but distinct, policies from the
architecture doc as structured, checked-in artifacts rather than prose:

1. **License Isolation Policy ("LazyIsolate", §2.2)** -- a decision table
   mapping a *dependency/code* risk type to its required engineering
   mitigation. This is a deliberately-designed internal policy, not an
   adopted external standard (no such standard exists for this exact
   problem). :data:`RISK_MITIGATIONS` is the checked-in representation of
   that table. It is documentation encoded as data -- dscraft.core does not
   (and cannot) automatically detect which risk type applies to a given
   dependency; a module's maintainer looks up the right :class:`RiskType`
   and follows its :class:`Mitigation`.

2. **Model Licensing Allowlist Policy (§2.10)** -- each module maintains its
   own per-module allowlist of pre-approved model checkpoints, split into
   :class:`ModelTier.TIER_1` (permissive, auto-usable) and
   :class:`ModelTier.TIER_2` (restricted, opt-in-gated). :class:`Allowlist`
   is the shared mechanism every module uses to register and check
   checkpoints against those tiers, including the runtime guard that
   refuses to hand back a Tier 2 entry unless the caller explicitly passes
   ``accept_restricted_licenses=True``.

Per the architecture doc, populating the *actual* allowlists (which real
model checkpoints go in Tier 1 vs Tier 2, per module) is an ongoing,
per-module maintenance task -- not something dscraft.core does. Every
``Allowlist`` instance starts empty.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

__all__ = [
    "RiskType",
    "Mitigation",
    "RISK_MITIGATIONS",
    "ModelTier",
    "ModelLicenseEntry",
    "Allowlist",
    "RestrictedLicenseNotAcceptedError",
]


class RiskType(enum.Enum):
    """Risk categories from the §2.2 License Isolation Policy decision table."""

    #: GPL/copyleft code needed at build/link time (e.g. SuiteSparse's
    #: CHOLMOD, GPLv2+).
    GPL_BUILD_TIME_LINK = "gpl_build_time_link"

    #: Restrictively-licensed optional build dependency (e.g. METIS,
    #: University of Minnesota license).
    RESTRICTIVE_OPTIONAL_BUILD_DEP = "restrictive_optional_build_dep"

    #: AGPL/GPL code needed at runtime, for internal-only local processing
    #: (never exposed as its own network-facing service to third parties).
    AGPL_GPL_RUNTIME_INTERNAL = "agpl_gpl_runtime_internal"

    #: AGPL/GPL code that would itself be the network-facing/exposed
    #: service.
    AGPL_GPL_NETWORK_FACING = "agpl_gpl_network_facing"

    #: Source-available non-compete license (e.g. FSL-1.1-MIT, BUSL)
    #: covering code you want equivalent functionality of.
    SOURCE_AVAILABLE_NON_COMPETE = "source_available_non_compete"

    #: Non-commercial-licensed weights/data (e.g. CC BY-NC 4.0).
    NON_COMMERCIAL_WEIGHTS = "non_commercial_weights"


@dataclass(frozen=True)
class Mitigation:
    """Required mitigation for a given :class:`RiskType`, per §2.2."""

    risk_type: RiskType
    required_mitigation: str
    notes: str
    flag_for_counsel: bool = False


#: The checked-in §2.2 decision table. Keyed by :class:`RiskType` so
#: consumers do a structured lookup rather than string-matching prose.
RISK_MITIGATIONS: dict[RiskType, Mitigation] = {
    RiskType.GPL_BUILD_TIME_LINK: Mitigation(
        risk_type=RiskType.GPL_BUILD_TIME_LINK,
        required_mitigation=(
            "Exclude via build flags/config; never compile/link against it "
            "(e.g. CHOLMOD_CONFIG=-DNPARTITION)."
        ),
        notes="Solid, uncontested.",
        flag_for_counsel=False,
    ),
    RiskType.RESTRICTIVE_OPTIONAL_BUILD_DEP: Mitigation(
        risk_type=RiskType.RESTRICTIVE_OPTIONAL_BUILD_DEP,
        required_mitigation=(
            "Never enable the optional flag; route equivalent functionality "
            "through a permissively-licensed alternative (e.g. "
            "torch-cluster's Graclus algorithm instead of "
            "torch-sparse+METIS)."
        ),
        notes="Solid.",
        flag_for_counsel=False,
    ),
    RiskType.AGPL_GPL_RUNTIME_INTERNAL: Mitigation(
        risk_type=RiskType.AGPL_GPL_RUNTIME_INTERNAL,
        required_mitigation=(
            "Isolate via subprocess/IPC with a loosely-coupled, documented "
            "interface (CLI/pipe/socket)."
        ),
        notes=(
            "Supported by the FSF's own 'mere aggregation' framing and "
            "AGPL §13's focus on the covered program itself being "
            "network-exposed -- but this is not a bright-line legal "
            "guarantee. Flag for counsel."
        ),
        flag_for_counsel=True,
    ),
    RiskType.AGPL_GPL_NETWORK_FACING: Mitigation(
        risk_type=RiskType.AGPL_GPL_NETWORK_FACING,
        required_mitigation=(
            "Full replacement with a permissive alternative, or explicit "
            "dual-licensing negotiation."
        ),
        notes=(
            "Subprocess-isolation theory does not apply here; high risk."
        ),
        flag_for_counsel=True,
    ),
    RiskType.SOURCE_AVAILABLE_NON_COMPETE: Mitigation(
        risk_type=RiskType.SOURCE_AVAILABLE_NON_COMPETE,
        required_mitigation=(
            "Clean-room reimplementation, rigorously documented (no "
            "engineer with prior access to the licensed source "
            "participates; spec-only reference)."
        ),
        notes=(
            "Legally sound because you are never a licensee of the covered "
            "code -- the Competing Use clause restricts use of the "
            "licensed software itself, not independent reimplementation. "
            "Flag for counsel to formalize the clean-room protocol."
        ),
        flag_for_counsel=True,
    ),
    RiskType.NON_COMMERCIAL_WEIGHTS: Mitigation(
        risk_type=RiskType.NON_COMMERCIAL_WEIGHTS,
        required_mitigation=(
            "Isolate into a separate, opt-in-gated plugin package; load "
            "only if the user explicitly sets an acceptance flag (e.g. "
            "accept_restricted_licenses=True)."
        ),
        notes=(
            "Standard approach; corresponds to Tier 2 of the Model "
            "Licensing Allowlist Policy (§2.10)."
        ),
        flag_for_counsel=False,
    ),
}


class ModelTier(enum.Enum):
    """Per §2.10: every allowlisted model checkpoint is Tier 1 or Tier 2."""

    #: Permissive license (e.g. Apache-2.0, MIT): auto-usable by default.
    TIER_1 = "tier_1"

    #: Restricted license (e.g. CC BY-NC): isolated behind an explicit
    #: opt-in acceptance flag.
    TIER_2 = "tier_2"


@dataclass(frozen=True)
class ModelLicenseEntry:
    """One allowlisted model checkpoint and its licensing classification."""

    name: str
    tier: ModelTier
    license_identifier: str
    notes: str = ""


class RestrictedLicenseNotAcceptedError(RuntimeError):
    """Raised when a Tier 2 checkpoint is requested without explicitly
    setting ``accept_restricted_licenses=True``."""


@dataclass
class Allowlist:
    """Per-module registry of allowlisted model checkpoints (§2.10).

    Each dscraft subpackage owns its own ``Allowlist`` instance (e.g.
    LazyForecast's allowlist listing TimesFM/Chronos-Bolt as Tier 1 and
    MOIRAI as Tier 2). ``dscraft.core`` provides the mechanism only; every
    instance starts empty and modules populate it themselves. Populating
    and maintaining the actual list is called out in the architecture doc
    as an ongoing per-module task, not a one-time design exercise.
    """

    _entries: dict[str, ModelLicenseEntry] = field(default_factory=dict)

    def register(
        self,
        name: str,
        tier: ModelTier,
        license_identifier: str,
        notes: str = "",
    ) -> ModelLicenseEntry:
        """Add (or replace) an allowlist entry for a model checkpoint.

        Raises:
            TypeError: If ``tier`` is not actually a :class:`ModelTier`
                member (e.g. a raw string like ``"tier_2"``). Python's type
                annotations are not enforced at runtime, so without this
                check a caller could pass ``tier="tier_2"``, which would be
                stored as-is and then silently fail the ``entry.tier is
                ModelTier.TIER_2`` identity check in :meth:`check` (a plain
                string is never identical to, or even equal to, the enum
                member) -- making a restricted checkpoint pass through
                :meth:`check` as if it were unrestricted, with no error at
                any point. This is deliberately not silently coerced (e.g.
                via ``ModelTier(tier)``) -- requiring the actual enum member
                makes the caller's intent explicit and catches this exact
                mistake at registration time instead of letting a
                restricted checkpoint slip through ungated.
        """
        if not isinstance(tier, ModelTier):
            raise TypeError(
                f"Allowlist.register() requires tier to be a ModelTier "
                f"member, got {tier!r} ({type(tier).__name__}). Passing a "
                "raw string (even one that matches a ModelTier value, e.g. "
                "\"tier_2\") is rejected rather than silently coerced: "
                "entry.tier is later compared to ModelTier.TIER_2 with `is`, "
                "so a string value would silently behave as an unrestricted "
                "entry instead of raising. Pass ModelTier.TIER_1 or "
                "ModelTier.TIER_2 explicitly."
            )
        entry = ModelLicenseEntry(
            name=name, tier=tier, license_identifier=license_identifier, notes=notes
        )
        self._entries[name] = entry
        return entry

    def __contains__(self, name: str) -> bool:
        """True if ``name`` has been registered in this allowlist (either tier)."""
        return name in self._entries

    def get(self, name: str) -> ModelLicenseEntry | None:
        """Look up an entry without the Tier 2 runtime guard. Prefer
        :meth:`check` for actual usage decisions."""
        return self._entries.get(name)

    def list_tier(self, tier: ModelTier) -> list[ModelLicenseEntry]:
        """Return every registered entry classified under ``tier``.

        Does not apply the Tier 2 runtime guard (unlike :meth:`check`) --
        this is for enumeration/reporting, not for gating actual usage.
        """
        return [entry for entry in self._entries.values() if entry.tier is tier]

    def check(
        self, name: str, *, accept_restricted_licenses: bool = False
    ) -> ModelLicenseEntry:
        """Return the allowlist entry for ``name``, enforcing the Tier 2 gate.

        Raises:
            KeyError: ``name`` is not registered in this allowlist at all.
            RestrictedLicenseNotAcceptedError: ``name`` is registered as
                Tier 2 and ``accept_restricted_licenses`` was not set to
                ``True``.
        """
        try:
            entry = self._entries[name]
        except KeyError as exc:
            raise KeyError(
                f"Model checkpoint {name!r} is not in this allowlist. "
                "Register it first, or use an approved checkpoint."
            ) from exc

        if entry.tier is ModelTier.TIER_2 and not accept_restricted_licenses:
            raise RestrictedLicenseNotAcceptedError(
                f"Model checkpoint {name!r} is licensed under "
                f"{entry.license_identifier!r} and classified as Tier 2 "
                "(restricted). Pass accept_restricted_licenses=True to use "
                "it. See architecture doc §2.2/§2.10."
            )
        return entry
