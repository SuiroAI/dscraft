"""Regression test for a CRITICAL follow-up finding on top of Finding 2
(`dscraft.core.sandbox.seatbelt`'s "never pickle the caller's callable" fix).

The original Finding-2 fix replaced ``pickle.dumps(func)`` with a
verification step that called ``importlib.import_module(func.__module__)``
directly in this (trusted, host) process to confirm ``func`` re-resolves to
itself. That verification step was itself vulnerable to the same class of
bug it was meant to close: ``func.__module__`` is a plain, freely-writable
string attribute on any function object -- nothing stops a caller from
doing ``some_func.__module__ = "some/malicious/module/path"`` before
calling ``run_callable``, at which point ``importlib.import_module`` would
import and execute that module's top-level code *in the trusted host
process*, before the sandbox even starts and before the identity check has
a chance to reject anything.

The fix replaces that import-based verification with a check against
``func.__globals__`` -- the dict object that *is* the function's actual
defining namespace, and which (unlike ``__module__``/``__qualname__``)
cannot be reassigned after the fact by simple attribute assignment. No
import of the (possibly-tampered) module name ever happens on the host
side; the module is only ever imported later, inside the sandboxed child
process's runner script.

This test file deliberately does NOT invoke ``/usr/bin/sandbox-exec`` (no
macOS skip marker) -- the vulnerability and its fix both live entirely in
host-side validation logic that runs identically on every platform, before
any subprocess is spawned.
"""

from __future__ import annotations

import importlib

import pytest

from dscraft.core.sandbox.seatbelt import SeatbeltSandboxExecutor

# A real, legitimately-importable-but-irrelevant stdlib module. Used as the
# "malicious" __module__ target below to prove the vulnerability is closed
# even when the tampered name points at something real and harmless (not
# just a nonexistent name, which would be a weaker proof -- a nonexistent
# name failing is not by itself evidence that import was never attempted).
_IRRELEVANT_BUT_REAL_MODULE = "colorsys"


def _victim_module_level_function() -> int:
    """A real, otherwise-ordinary module-level function (flat qualname, no
    ``<locals>``/``<lambda>``) used as the tampering target below. It must
    be defined at module level (not nested inside another function) so
    that only its ``__module__`` is tampered -- its ``__qualname__`` stays
    flat, which is what lets execution reach the new ``__globals__``-based
    check instead of being rejected earlier by the lambda/closure check.
    """
    return 1


def _make_tampered_function(monkeypatch: pytest.MonkeyPatch, tampered_module_name: str):
    """Return :func:`_victim_module_level_function` with its ``__module__``
    overwritten (via ``monkeypatch``, so it is automatically restored at
    the end of the test regardless of outcome) to point somewhere else,
    simulating an attacker (or attacker-influenced code upstream of
    ``run_callable``) tampering with the attribute after the function
    object already exists.

    This mutation is the crux of the vulnerability: ``__module__`` is just
    a plain writable string attribute, not an intrinsic, tamper-proof
    property of the function object.
    """
    monkeypatch.setattr(
        _victim_module_level_function, "__module__", tampered_module_name, raising=True
    )
    return _victim_module_level_function


def test_tampered_module_attribute_is_rejected_with_clear_value_error(monkeypatch):
    """A function whose __module__ has been reassigned since definition
    must be rejected with a clear ValueError -- __globals__["__name__"]
    (this test module's real name) will not match the tampered
    __module__ (`colorsys`), so validation fails closed.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _IRRELEVANT_BUT_REAL_MODULE)

    with pytest.raises(ValueError, match="__globals__"):
        executor._resolve_module_level_function(tampered)


def test_tampering_never_actually_imports_the_tampered_module_name(monkeypatch):
    """The critical proof: validating a tampered callable must NEVER import
    the tampered module name as a side effect. Monkeypatches
    importlib.import_module to raise AssertionError if invoked at all
    during validation -- combined with the ``pytest.raises(ValueError)``
    assertion below, this fully proves no import ever happens. (A
    ``sys.modules``-based check was deliberately not added here: it would
    depend on process-global import state that unrelated tests, plugins, or
    pytest's own collection machinery could perturb, causing a false
    failure unrelated to any real regression -- the ``import_module`` guard
    below is the robust proof.)
    """

    def _boom(name, *args, **kwargs):
        raise AssertionError(
            "_resolve_module_level_function() must NEVER call "
            f"importlib.import_module (attempted with name={name!r}) -- "
            "host-side validation must reject a tampered __module__ purely "
            "via __globals__ introspection, without ever importing "
            "anything, real or fake."
        )

    monkeypatch.setattr(importlib, "import_module", _boom)

    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _IRRELEVANT_BUT_REAL_MODULE)

    with pytest.raises(ValueError):
        executor._resolve_module_level_function(tampered)


def test_tampered_module_pointing_at_nonexistent_module_is_also_rejected(monkeypatch):
    """Same tampering scenario, but pointing __module__ at a module name
    that does not exist at all -- confirms the fix does not depend on the
    tampered name happening to resolve to something importable; it never
    attempts to import it either way.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(
        monkeypatch, "some.totally.nonexistent.module.path"
    )

    with pytest.raises(ValueError, match="__globals__"):
        executor._resolve_module_level_function(tampered)


def test_untampered_module_level_function_still_resolves_correctly():
    """Sanity/non-regression check: an ordinary, non-tampered module-level
    function (this test module's own top-level function, defined below)
    still resolves successfully via the new __globals__-based check.
    """
    executor = SeatbeltSandboxExecutor()
    module_name, qualname = executor._resolve_module_level_function(
        _legitimate_module_level_function
    )
    assert module_name == __name__
    assert qualname == "_legitimate_module_level_function"


def _legitimate_module_level_function() -> int:
    """A real, untampered module-level function used by
    ``test_untampered_module_level_function_still_resolves_correctly``."""
    return 42


class _DunderTripwire:
    """An object whose comparison/containment/truthiness dunder methods
    raise loudly if ever invoked -- used below to prove that
    ``_resolve_module_level_function`` never performs a rich comparison
    (``==``/``!=``), containment check (``in``), or truthiness test
    (``bool(...)``/``not ...``) against an attacker-controlled,
    non-``str`` ``__module__``/``__qualname__`` value.

    This is the next layer of the same vulnerability class as the
    import-based and ``__globals__``-mismatch bugs fixed elsewhere in this
    file: ``func.__module__``/``__qualname__`` are plain, freely-writable
    attributes -- nothing requires them to be strings at all. If validation
    ever did ``globals_dict.get("__name__") != module_name`` or
    ``"<lambda>" in qualname`` with one of these as the right/left-hand
    operand, Python would invoke this object's ``__eq__``/``__ne__``/
    ``__contains__`` in the trusted host process, before the sandbox exists
    -- exactly the "attacker code runs unsandboxed" outcome this function
    exists to prevent, even though no ``importlib.import_module`` call is
    involved this time.
    """

    def __eq__(self, other: object) -> bool:
        raise AssertionError(
            "dunder method invoked -- SECURITY REGRESSION: "
            "_DunderTripwire.__eq__ was called, meaning "
            "_resolve_module_level_function compared an attacker-controlled "
            "non-str value before type-checking it."
        )

    def __ne__(self, other: object) -> bool:
        raise AssertionError(
            "dunder method invoked -- SECURITY REGRESSION: "
            "_DunderTripwire.__ne__ was called, meaning "
            "_resolve_module_level_function compared an attacker-controlled "
            "non-str value before type-checking it."
        )

    def __contains__(self, item: object) -> bool:
        raise AssertionError(
            "dunder method invoked -- SECURITY REGRESSION: "
            "_DunderTripwire.__contains__ was called, meaning "
            "_resolve_module_level_function did a containment check "
            "(`x in qualname`-style) against an attacker-controlled non-str "
            "value before type-checking it."
        )

    def __bool__(self) -> bool:
        raise AssertionError(
            "dunder method invoked -- SECURITY REGRESSION: "
            "_DunderTripwire.__bool__ was called, meaning "
            "_resolve_module_level_function evaluated the truthiness of an "
            "attacker-controlled non-str value before type-checking it."
        )

    def __hash__(self) -> int:  # pragma: no cover -- not expected to be hit
        raise AssertionError(
            "dunder method invoked -- SECURITY REGRESSION: "
            "_DunderTripwire.__hash__ was called."
        )

    def __repr__(self) -> str:
        # Deliberately safe: error messages built by the function under
        # test use %r / !r on func/module_name/qualname, and a raising
        # __repr__ would make failures unreadable and mask the real assert.
        return "<_DunderTripwire>"


def test_non_string_module_attribute_is_rejected_without_invoking_dunders(monkeypatch):
    """CRITICAL regression test: if ``__module__`` is tampered to a
    non-``str`` object whose ``__eq__``/``__ne__``/``__contains__``/
    ``__bool__`` raise ``AssertionError`` when invoked, validation must
    reject it with a clear ``ValueError`` *without ever invoking any of
    those dunder methods*. This proves the type check
    (``type(x) is str``) happens strictly before any comparison,
    containment check, or truthiness test involving the untrusted value --
    closing the "comparison itself can run attacker code" variant of the
    module-tampering vulnerability class documented at the top of this
    file.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _DunderTripwire())

    with pytest.raises(ValueError, match="plain string"):
        executor._resolve_module_level_function(tampered)


def test_non_string_qualname_attribute_is_rejected_without_invoking_dunders(monkeypatch):
    """Same proof as above, but for a tampered ``__qualname__`` instead of
    ``__module__`` -- ``qualname`` is used in ``"<lambda>" in qualname``
    and ``"<locals>" in qualname`` containment checks and
    ``globals_dict.get(qualname)`` dict lookups, any of which could invoke
    a malicious object's dunder methods if it were not type-checked first.

    CPython's C-level slot for ``__qualname__`` enforces
    ``isinstance(value, str)`` at assignment time (unlike ``__module__``,
    which accepts literally any object), so a wholly-unrelated
    ``_DunderTripwire`` object cannot be assigned here at all -- attempting
    it raises ``TypeError: __qualname__ must be set to a string object``
    before this test even reaches ``_resolve_module_level_function``. A
    ``str`` *subclass* satisfies that ``isinstance`` check, though, and is
    exactly the case the ``type(x) is str`` (not ``isinstance``) design
    choice defends against -- so this test tampers with a subclass instance
    whose dunder methods raise instead.
    """

    class _EvilQualname(str):
        def __contains__(self, item: object) -> bool:
            raise AssertionError(
                "dunder method invoked -- SECURITY REGRESSION: "
                "_EvilQualname.__contains__ was called, meaning "
                "_resolve_module_level_function did a containment check "
                "against a str-subclass qualname before type-checking it "
                "with `type(x) is str`."
            )

        def __eq__(self, other: object) -> bool:
            raise AssertionError(
                "dunder method invoked -- SECURITY REGRESSION: "
                "_EvilQualname.__eq__ was called on the qualname before "
                "type-checking it."
            )

        __hash__ = str.__hash__

    monkeypatch.setattr(
        _victim_module_level_function,
        "__qualname__",
        _EvilQualname(_victim_module_level_function.__qualname__),
        raising=True,
    )

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError, match="plain string"):
        executor._resolve_module_level_function(_victim_module_level_function)


def test_str_subclass_module_attribute_is_also_rejected(monkeypatch):
    """``type(x) is str`` (not ``isinstance(x, str)``) is used deliberately:
    a ``str`` subclass could itself override ``__eq__``/``__contains__``/
    ``__hash__`` to run attacker code on comparison despite satisfying
    ``isinstance(x, str)``. Confirms such a subclass instance is rejected
    the same way a wholly-unrelated object would be.
    """

    class _EvilStr(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError(
                "dunder method invoked -- SECURITY REGRESSION: _EvilStr "
                "(a str subclass) had __eq__ invoked despite the "
                "type(x) is str check, which must reject subclasses too."
            )

        __hash__ = str.__hash__

    tampered = _make_tampered_function(monkeypatch, _EvilStr(_IRRELEVANT_BUT_REAL_MODULE))

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError, match="plain string"):
        executor._resolve_module_level_function(tampered)


class _NameHookTripwireMeta(type):
    """A metaclass whose ``__getattribute__`` raises ``AssertionError`` if
    ever asked for ``"__name__"`` on a class using it.

    Same fixture shape as ``test_json_safe_value.py``'s
    ``_NameHookTripwireMeta`` (kept local to this file rather than shared,
    matching this file's existing self-contained-fixture convention): a
    plain ``type(value).__name__`` correctly bypasses an overridable
    ``value.__class__`` property by going through ``type()``, but a bare
    ``.__name__`` attribute read off *that* type object is still a normal
    attribute lookup -- one a sufficiently exotic custom metaclass can hook.

    Used here to prove that ``_resolve_module_level_function`` -- which
    builds error messages mentioning ``type(func).__name__`` and
    ``type(qualname).__name__``/``type(module_name).__name__`` -- now goes
    through ``_safe_type_name()`` everywhere and therefore never triggers
    this hook, even when rejecting exactly the values whose type name it is
    trying to report.
    """

    def __getattribute__(cls, name):
        if name == "__name__":
            raise AssertionError(
                "metaclass __getattribute__ must never be invoked for "
                "'__name__' while building a validation rejection message"
            )
        return super().__getattribute__(name)


class _HookedType(metaclass=_NameHookTripwireMeta):
    """An instance of this class has a true type using a metaclass that
    tripwires on ``.__name__`` access."""


class _HookedStrLike(str, metaclass=_NameHookTripwireMeta):
    """A ``str`` subclass whose metaclass tripwires on ``.__name__`` access.

    Used to simulate ``func.__module__``/``func.__qualname__`` being
    tampered to an instance of a class that satisfies neither the
    ``type(x) is str`` check (since it's a subclass, so it's still rejected
    for that reason) *and* whose type itself is hostile -- proving the
    subsequent error-message construction reporting the rejected value's
    type name does not trigger the metaclass hook either.
    """


def test_non_function_with_name_hooked_metaclass_is_rejected_without_triggering_hook():
    """A ``func`` argument that is not a ``types.FunctionType`` at all, and
    whose own true type uses a ``__name__``-hooking metaclass, is rejected
    via the intended ``TypeError`` -- and the metaclass hook never fires
    while ``_resolve_module_level_function`` builds the rejection message
    (which reports ``type(func).__name__`` via ``_safe_type_name``).
    """
    executor = SeatbeltSandboxExecutor()
    hostile_func = _HookedType()

    with pytest.raises(TypeError, match="non-function"):
        executor._resolve_module_level_function(hostile_func)


def test_tampered_module_attribute_with_name_hooked_metaclass_is_rejected_without_triggering_hook(
    monkeypatch,
):
    """A real module-level function whose ``__module__`` has been tampered
    to an instance of a class using a ``__name__``-hooking metaclass is
    rejected via the intended ``ValueError`` (failing the
    ``type(module_name) is str`` check) -- and the metaclass hook never
    fires while the rejection message (which reports
    ``type(module_name).__name__`` via ``_safe_type_name``) is built.
    """
    executor = SeatbeltSandboxExecutor()
    tampered = _make_tampered_function(monkeypatch, _HookedType())

    with pytest.raises(ValueError, match="plain string"):
        executor._resolve_module_level_function(tampered)


def test_tampered_qualname_attribute_with_name_hooked_metaclass_is_rejected_without_triggering_hook(
    monkeypatch,
):
    """Same regression as above, but for a tampered ``__qualname__``
    instead of ``__module__``. ``__qualname__``'s C-level slot enforces
    ``isinstance(value, str)`` at assignment time, so the tampered value
    must itself be a ``str`` subclass (``_HookedStrLike``, which also uses
    the ``__name__``-hooking metaclass) rather than an unrelated object --
    exactly like ``test_non_string_qualname_attribute_is_rejected_without_invoking_dunders``
    above needs a ``str`` subclass for the same structural reason.
    """
    monkeypatch.setattr(
        _victim_module_level_function,
        "__qualname__",
        _HookedStrLike(_victim_module_level_function.__qualname__),
        raising=True,
    )

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError, match="plain string"):
        executor._resolve_module_level_function(_victim_module_level_function)


class _EvilPartialFunc:
    """Property getter that raises if ever accessed -- used below as the
    shadowed ``.func`` value on a malicious ``functools.partial`` subclass.
    """

    def __get__(self, obj, objtype=None):
        raise AssertionError(
            "SECURITY REGRESSION: a functools.partial *subclass*'s "
            "overridden `.func` property getter was invoked -- "
            "_decompose_callable must only ever treat an exact, "
            "unsubclassed `functools.partial` instance as a partial "
            "(via `type(func) is functools.partial`), never a subclass, "
            "since a subclass can shadow `.func`/`.args`/`.keywords` with "
            "its own instance property that runs arbitrary code the "
            "moment it is merely read."
        )


def test_functools_partial_subclass_overriding_func_property_is_rejected_via_type_function_path():
    """A ``functools.partial`` *subclass* that shadows ``.func`` with a
    property whose getter raises if invoked must be rejected via the plain
    ``_resolve_module_level_function(func)`` path (which correctly raises
    ``TypeError`` since a partial subclass is not a ``types.FunctionType``)
    -- without ever calling the hostile subclass's ``.func`` getter.

    This proves ``_decompose_callable`` uses ``type(func) is
    functools.partial`` (exact-type identity), not
    ``isinstance(func, functools.partial)``: the latter would let this
    subclass through the "it's a partial" branch and immediately trigger
    the hostile ``.func`` property access below.
    """
    import functools

    class EvilPartial(functools.partial):
        func = _EvilPartialFunc()

    hostile = EvilPartial(_legitimate_module_level_function)
    assert isinstance(hostile, functools.partial)  # sanity: it IS a partial subclass

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(TypeError, match="non-function"):
        executor._decompose_callable(hostile)


class _HostileClassProperty:
    """An object whose ``__class__`` property getter raises if ever
    accessed -- used to prove ``_resolve_module_level_function``'s very
    first check (rejecting non-function ``func`` arguments) uses
    ``type(func) is types.FunctionType`` and not ``isinstance(func,
    types.FunctionType)``.

    ``isinstance()``'s slow path consults the target's ``__class__``
    attribute when the fast ``Py_TYPE()`` check fails, and ``__class__`` is
    an ordinary, overridable property like any other. A hostile object
    could make its getter return ``types.FunctionType`` (spoofing the
    isinstance check into passing) or do anything else -- either way, the
    getter body already ran, unsandboxed, in the trusted host process,
    before any of this function's other checks get a chance to run. This is
    the same "even introspecting this untrusted object runs attacker code"
    vulnerability class as the ``functools.partial``-subclass and
    metaclass-``__getattribute__`` findings above, just triggered through
    ``__class__`` instead of ``.func`` or ``.__name__``.
    """

    @property
    def __class__(self):  # type: ignore[override]
        raise AssertionError(
            "SECURITY REGRESSION: a hostile object's __class__ property "
            "getter was invoked -- _resolve_module_level_function must use "
            "`type(func) is types.FunctionType` (exact-type identity), "
            "never `isinstance(func, types.FunctionType)`, since "
            "isinstance()'s slow path reads the overridable __class__ "
            "attribute and would run this getter in the trusted host "
            "process before any other check runs."
        )


def test_hostile_class_property_is_rejected_without_triggering_its_getter():
    """The very first check in ``_resolve_module_level_function`` must
    reject a non-function ``func`` via plain ``TypeError`` without ever
    invoking a hostile ``__class__`` property getter on it -- proving the
    exact-type check (``type(func) is types.FunctionType``) is used instead
    of ``isinstance``.
    """
    executor = SeatbeltSandboxExecutor()
    hostile = _HostileClassProperty()

    with pytest.raises(TypeError, match="non-function"):
        executor._resolve_module_level_function(hostile)


def test_dotted_qualname_is_rejected_as_unsupported():
    """A function whose __qualname__ contains a dot (e.g. a method or a
    function nested under a class) is rejected outright -- the
    __globals__-based check can only verify flat module-level names via a
    direct dict lookup, unlike the old attribute-chasing approach which
    could (insecurely) walk a dotted path. This is a deliberate tightening:
    run_callable()'s documented contract is "module-level functions or
    functools.partial wrapping one" -- methods were never actually
    supported in a safe way.
    """

    class Container:
        def method(self) -> int:
            return 1

    executor = SeatbeltSandboxExecutor()
    with pytest.raises(ValueError, match="module-level"):
        executor._resolve_module_level_function(Container.method)
