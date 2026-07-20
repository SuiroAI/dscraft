"""Platform-independent unit tests for
``dscraft.core.sandbox.seatbelt._validate_json_safe_value`` (and its
``_safe_type_name`` helper).

This validator is pure Python logic with **no** macOS/Seatbelt/subprocess
dependency at all -- unlike the bulk of `test_seatbelt.py` (which carries a
module-level ``pytestmark = pytest.mark.skipif(...)`` because most of that
file's tests genuinely need to invoke the real ``/usr/bin/sandbox-exec``),
these tests must run on every platform, including Linux CI. This file
deliberately carries no skip marker, matching the same platform-neutral
convention already established by
``test_module_tampering_regression.py`` for other pure host-side validation
logic in this module.

``test_validate_json_safe_value_rejects_malicious_repr_directly`` originally
lived in ``test_seatbelt.py`` and was moved here verbatim (as
``test_rejects_malicious_repr_value`` /
``test_rejects_non_string_dict_key_with_malicious_repr`` /
``test_rejects_tuple_with_malicious_element`` below) since it never touched
the sandbox at all.
"""

from __future__ import annotations

import pytest

from dscraft.core.sandbox.seatbelt import _validate_json_safe_value


class _ExplodingRepr:
    """An object of a non-JSON-native type whose __repr__/__str__ raise if
    ever invoked -- if _validate_json_safe_value's rejection path called
    repr()/str() on the untrusted value, this would blow up with
    AssertionError instead of a clean ValueError."""

    def __repr__(self):
        raise AssertionError("__repr__ must never be called on an untrusted value")

    def __str__(self):
        raise AssertionError("__str__ must never be called on an untrusted value")


class _ExplodingKey:
    """A dict key whose __repr__/__str__ raise if ever invoked -- used to
    confirm the non-string-dict-key rejection path never reprs the key."""

    def __repr__(self):
        raise AssertionError("__repr__ must never be called on an untrusted key")

    def __str__(self):
        raise AssertionError("__str__ must never be called on an untrusted key")

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return self is other


class _NameHookTripwireMeta(type):
    """A metaclass whose ``__getattribute__`` raises ``AssertionError`` if
    ever asked for ``"__name__"`` on a class using it.

    Regression fixture for the CodeRabbit finding that a plain
    ``type(value).__name__`` -- while it correctly bypasses an overridable
    ``value.__class__`` property by going through ``type()`` -- could still
    invoke a *metaclass's* ``__getattribute__`` hook when ``.__name__`` is
    read off the resulting type object, if that type was itself created
    with a sufficiently exotic/malicious custom metaclass.
    ``_validate_json_safe_value`` (via its ``_safe_type_name`` helper) must
    reject an instance of a class using this metaclass without ever
    triggering the hook.
    """

    def __getattribute__(cls, name):
        if name == "__name__":
            raise AssertionError(
                "metaclass __getattribute__ must never be invoked for "
                "'__name__' while building a validation rejection message"
            )
        return super().__getattribute__(name)


class _HookedType(metaclass=_NameHookTripwireMeta):
    """An instance of this class is a non-JSON-native value whose *true
    type* uses a metaclass that tripwires on ``.__name__`` access."""


def test_rejects_malicious_repr_value():
    """A bad value with a malicious __repr__/__str__ is rejected with
    ValueError, and the malicious dunder methods are never invoked."""
    with pytest.raises(ValueError, match="not a JSON-native type"):
        _validate_json_safe_value(_ExplodingRepr(), path="args[0]")


def test_rejects_non_string_dict_key_with_malicious_repr():
    """A dict with a non-string key whose __repr__/__str__ raise is
    rejected with ValueError without ever invoking those dunders."""
    with pytest.raises(ValueError, match="non-string key"):
        _validate_json_safe_value({_ExplodingKey(): "value"}, path="args[0]")


def test_rejects_tuple_with_malicious_element():
    """A tuple containing a malicious element is rejected as a tuple,
    without ever repr()'ing the tuple or its contents."""
    with pytest.raises(ValueError, match="tuple"):
        _validate_json_safe_value((_ExplodingRepr(),), path="args[0]")


def test_rejects_tuple_via_exact_type_check_not_isinstance():
    """A plain tuple is rejected via the ``type(value) is tuple`` exact-type
    check (not ``isinstance``) -- see the CodeRabbit finding that
    ``isinstance()`` can, on some code paths, consult the overridable
    ``value.__class__`` rather than purely the C-level true type."""
    with pytest.raises(ValueError, match="tuple"):
        _validate_json_safe_value((1, 2, 3), path="args[0]")


def test_tuple_subclass_still_rejected_via_generic_path():
    """A ``tuple`` subclass no longer matches the exact-type ``tuple``
    check, so it falls through to the generic "not a JSON-native type"
    rejection instead of the tuple-specific message -- it is still
    rejected either way, which is what actually matters for this security
    boundary (a subclass masquerading as JSON-safe must never be silently
    accepted)."""

    class MaliciousTuple(tuple):
        pass

    with pytest.raises(ValueError, match="not a JSON-native type"):
        _validate_json_safe_value(MaliciousTuple((1, 2)), path="args[0]")


def test_rejects_value_whose_type_has_a_name_hooked_metaclass():
    """A value whose true type was created with a metaclass that raises if
    its __getattribute__ is ever asked for "__name__" is still rejected
    with a clean ValueError -- and the metaclass hook is never triggered.

    Regression test for the metaclass-hook attack surface identified
    alongside the exact-type-check fix for ``isinstance(value, tuple)``:
    naively reading ``type(value).__name__`` to build the rejection message
    would invoke this metaclass's ``__getattribute__`` hook (which raises
    ``AssertionError`` for ``"__name__"``), so a passing test proves both
    (a) the value is still rejected, and (b) the hook was never invoked.
    """
    with pytest.raises(ValueError, match="not a JSON-native type"):
        _validate_json_safe_value(_HookedType(), path="args[0]")


def test_rejects_dict_key_whose_type_has_a_name_hooked_metaclass():
    """Same metaclass-hook regression, but via the non-string-dict-key
    rejection path, which also reports the offending type's name."""
    with pytest.raises(ValueError, match="non-string key"):
        _validate_json_safe_value({_HookedType(): "value"}, path="args[0]")


def test_rejects_malicious_repr_value_nested_inside_a_list():
    """A malicious-__repr__/__str__ value nested inside a list (not at the
    top level) is still rejected without ever invoking those dunders --
    proving the recursive descent into list elements is just as safe as
    the top-level dispatch checked by ``test_rejects_malicious_repr_value``
    above.

    Moved here (and de-duplicated) from ``test_seatbelt.py``, which
    exercised the same validator logic indirectly through
    ``SeatbeltSandboxExecutor.run_callable`` -- a macOS-only integration
    path that (via the module's skip marker) never ran this
    platform-independent validator regression on Linux CI.
    """
    with pytest.raises(ValueError, match="not a JSON-native type"):
        _validate_json_safe_value([1, _ExplodingRepr(), 3], path="args[0]")
