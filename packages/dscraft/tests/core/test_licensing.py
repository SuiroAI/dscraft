"""Tests for dscraft.core.licensing (license-isolation + allowlist policy, §2.2/§2.10)."""

from __future__ import annotations

import dataclasses

import pytest

from dscraft.core.licensing import (
    Allowlist,
    Mitigation,
    ModelLicenseEntry,
    ModelTier,
    RestrictedLicenseNotAcceptedError,
    RISK_MITIGATIONS,
    RiskType,
)


# ---------------------------------------------------------------------------
# §2.2 License Isolation Policy decision table
# ---------------------------------------------------------------------------


def test_every_risk_type_has_a_mitigation_entry():
    """Every RiskType enum member has a corresponding, non-empty Mitigation entry in RISK_MITIGATIONS."""
    for risk_type in RiskType:
        assert risk_type in RISK_MITIGATIONS
        mitigation = RISK_MITIGATIONS[risk_type]
        assert isinstance(mitigation, Mitigation)
        assert mitigation.risk_type is risk_type
        assert mitigation.required_mitigation  # non-empty
        assert mitigation.notes  # non-empty


def test_agpl_gpl_runtime_internal_is_flagged_for_counsel():
    """The AGPL_GPL_RUNTIME_INTERNAL mitigation is flagged for counsel review and its required mitigation text mentions subprocess isolation."""
    mitigation = RISK_MITIGATIONS[RiskType.AGPL_GPL_RUNTIME_INTERNAL]
    assert mitigation.flag_for_counsel is True
    assert "subprocess" in mitigation.required_mitigation.lower()


def test_agpl_gpl_network_facing_is_flagged_for_counsel():
    """The AGPL_GPL_NETWORK_FACING mitigation is flagged for counsel review, since subprocess-isolation theory does not apply to it."""
    mitigation = RISK_MITIGATIONS[RiskType.AGPL_GPL_NETWORK_FACING]
    assert mitigation.flag_for_counsel is True


def test_gpl_build_time_link_is_not_flagged_for_counsel():
    """The GPL_BUILD_TIME_LINK mitigation is NOT flagged for counsel, since simple build-flag exclusion is solid/uncontested."""
    mitigation = RISK_MITIGATIONS[RiskType.GPL_BUILD_TIME_LINK]
    assert mitigation.flag_for_counsel is False


def test_non_commercial_weights_mitigation_references_opt_in_flag():
    """The NON_COMMERCIAL_WEIGHTS mitigation's required-mitigation text names the accept_restricted_licenses opt-in flag."""
    mitigation = RISK_MITIGATIONS[RiskType.NON_COMMERCIAL_WEIGHTS]
    assert "accept_restricted_licenses" in mitigation.required_mitigation


def test_mitigation_dataclass_is_immutable():
    """Mitigation is a frozen dataclass; assigning to a field after construction raises."""
    mitigation = RISK_MITIGATIONS[RiskType.GPL_BUILD_TIME_LINK]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mitigation.required_mitigation = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §2.10 Model Licensing Allowlist Policy
# ---------------------------------------------------------------------------


def test_new_allowlist_starts_empty():
    """A freshly-constructed Allowlist has no Tier 1 or Tier 2 entries and reports any name as not contained."""
    allowlist = Allowlist()
    assert allowlist.list_tier(ModelTier.TIER_1) == []
    assert allowlist.list_tier(ModelTier.TIER_2) == []
    assert "anything" not in allowlist


def test_register_and_check_tier1_entry_succeeds_without_opt_in():
    """check() returns a registered Tier 1 entry without requiring accept_restricted_licenses."""
    allowlist = Allowlist()
    allowlist.register("timesfm-2.5", ModelTier.TIER_1, "Apache-2.0")

    entry = allowlist.check("timesfm-2.5")
    assert isinstance(entry, ModelLicenseEntry)
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_check_unknown_entry_raises_key_error():
    """check() raises KeyError for a model name that was never registered."""
    allowlist = Allowlist()
    with pytest.raises(KeyError):
        allowlist.check("not-registered")


def test_tier2_entry_raises_without_opt_in_flag():
    """check() raises RestrictedLicenseNotAcceptedError for a Tier 2 entry both when accept_restricted_licenses is omitted and when explicitly set to False."""
    allowlist = Allowlist()
    allowlist.register(
        "moirai-1.0",
        ModelTier.TIER_2,
        "CC-BY-NC-4.0",
        notes="Salesforce MOIRAI weights are non-commercial.",
    )

    with pytest.raises(RestrictedLicenseNotAcceptedError):
        allowlist.check("moirai-1.0")

    # Explicitly-false is treated the same as omitted.
    with pytest.raises(RestrictedLicenseNotAcceptedError):
        allowlist.check("moirai-1.0", accept_restricted_licenses=False)


def test_tier2_entry_succeeds_with_explicit_opt_in():
    """check() returns a Tier 2 entry when the caller explicitly passes accept_restricted_licenses=True."""
    allowlist = Allowlist()
    allowlist.register("moirai-1.0", ModelTier.TIER_2, "CC-BY-NC-4.0")

    entry = allowlist.check("moirai-1.0", accept_restricted_licenses=True)
    assert entry.tier is ModelTier.TIER_2
    assert entry.name == "moirai-1.0"


def test_list_tier_partitions_entries_correctly():
    """list_tier() returns exactly the entries registered under the requested tier, correctly separating Tier 1 from Tier 2 registrations."""
    allowlist = Allowlist()
    allowlist.register("timesfm-2.5", ModelTier.TIER_1, "Apache-2.0")
    allowlist.register("chronos-bolt", ModelTier.TIER_1, "Apache-2.0")
    allowlist.register("moirai-1.0", ModelTier.TIER_2, "CC-BY-NC-4.0")

    tier1_names = {entry.name for entry in allowlist.list_tier(ModelTier.TIER_1)}
    tier2_names = {entry.name for entry in allowlist.list_tier(ModelTier.TIER_2)}

    assert tier1_names == {"timesfm-2.5", "chronos-bolt"}
    assert tier2_names == {"moirai-1.0"}


def test_register_overwrites_existing_entry():
    """Re-registering the same model name replaces its previous entry entirely, including moving it from Tier 2 to Tier 1."""
    allowlist = Allowlist()
    allowlist.register("some-model", ModelTier.TIER_2, "CC-BY-NC-4.0")
    allowlist.register("some-model", ModelTier.TIER_1, "Apache-2.0")

    entry = allowlist.check("some-model")  # no opt-in needed now
    assert entry.tier is ModelTier.TIER_1
    assert entry.license_identifier == "Apache-2.0"


def test_get_returns_none_for_missing_entry_without_raising():
    """get() returns None (not a raised exception) for a name that was never registered, unlike check()."""
    allowlist = Allowlist()
    assert allowlist.get("missing") is None


def test_register_rejects_raw_string_tier_instead_of_silently_registering_unrestricted():
    """register() raises TypeError when tier is a raw string (e.g. "tier_2") instead of an actual ModelTier member.

    Regression test for the finding that Python does not enforce the
    ModelTier type annotation at runtime: passing tier="tier_2" would
    previously be stored as-is, and then entry.tier is ModelTier.TIER_2 in
    check() would be False for a plain string (identity comparison never
    matches across types), silently treating a restricted entry as
    unrestricted with no error anywhere in the flow.
    """
    allowlist = Allowlist()

    with pytest.raises(TypeError):
        allowlist.register("sneaky-model", "tier_2", "CC-BY-NC-4.0")  # type: ignore[arg-type]

    # The bad call must not have partially registered anything either.
    assert "sneaky-model" not in allowlist


def test_register_still_works_with_the_correct_enum_member():
    """register() succeeds, and the entry is correctly gated, when tier is passed as the actual ModelTier.TIER_2 enum member (not a string)."""
    allowlist = Allowlist()
    allowlist.register("real-restricted-model", ModelTier.TIER_2, "CC-BY-NC-4.0")

    with pytest.raises(RestrictedLicenseNotAcceptedError):
        allowlist.check("real-restricted-model")

    entry = allowlist.check("real-restricted-model", accept_restricted_licenses=True)
    assert entry.tier is ModelTier.TIER_2
