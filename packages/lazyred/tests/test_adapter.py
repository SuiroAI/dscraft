"""Tests for the BaseSecurityAdapter interface shape (adapter.py)."""

from __future__ import annotations

import pytest
from lazycore.sandbox import BaseSandboxExecutor, SandboxResult

from benchcraft_lazyred.adapter import Attempt, BaseSecurityAdapter, Finding
from lazycore.telemetry import SecuritySeverity


class _StubExecutor(BaseSandboxExecutor):
    """A minimal stub executor used only to exercise the ABC's ``run()``
    chaining logic in isolation from any real sandbox backend -- the real
    Seatbelt backend is exercised separately in test_probes.py."""

    def is_available(self) -> bool:
        return True

    def run_command(self, command, *, policy=None):
        return SandboxResult(exit_code=0, stdout="", stderr="")

    def run_callable(self, func, *, policy=None):
        return SandboxResult(exit_code=0, stdout=repr(func()), stderr="")


class _EchoAdapter(BaseSecurityAdapter):
    """Trivial concrete adapter: echoes the payload as output, and flags
    "vulnerable" whenever the payload contains the word "bad"."""

    probe_id = "echo_test"

    def generate_attempt(self, probe_input: str) -> Attempt:
        return Attempt(probe_id=self.probe_id, payload=probe_input, prompt=probe_input)

    def run_target(self, attempt: Attempt, executor: BaseSandboxExecutor) -> Attempt:
        result = executor.run_command(["true"])
        attempt.sandbox_result = result
        attempt.raw_output = attempt.prompt
        return attempt

    def detect(self, attempt: Attempt) -> Finding:
        vulnerable = "bad" in (attempt.raw_output or "")
        return Finding(
            probe_id=attempt.probe_id,
            vulnerable=vulnerable,
            severity=SecuritySeverity.HIGH if vulnerable else SecuritySeverity.INFO,
            owasp_mapping=("LLM01: Prompt Injection",),
            detail="stub",
            attempt=attempt,
        )


def test_base_security_adapter_cannot_be_instantiated_directly():
    """``BaseSecurityAdapter`` is an ABC with abstract methods, so
    instantiating it directly (without a concrete subclass implementing
    ``generate_attempt``/``run_target``/``detect``) must raise ``TypeError``."""
    with pytest.raises(TypeError):
        BaseSecurityAdapter()  # type: ignore[abstract]


def test_attempt_defaults_are_none_until_populated():
    """A freshly-constructed ``Attempt`` has ``prompt``/``raw_output``/
    ``sandbox_result`` still unset (``None``) and an empty ``metadata``
    dict, since those fields are only filled in as the attempt moves
    through ``generate_attempt`` -> ``run_target`` -> ``detect``."""
    attempt = Attempt(probe_id="p", payload="x")
    assert attempt.prompt is None
    assert attempt.raw_output is None
    assert attempt.sandbox_result is None
    assert attempt.metadata == {}


def test_run_chains_generate_run_detect():
    """``BaseSecurityAdapter.run()`` chains all three steps end to end: for
    a payload containing "bad", the stub adapter's ``run_target`` records
    the echoed output and sandbox result, and ``detect`` correctly flags
    the resulting ``Finding`` as vulnerable."""
    adapter = _EchoAdapter()
    executor = _StubExecutor()

    finding = adapter.run("this is a bad payload", executor)

    assert isinstance(finding, Finding)
    assert finding.vulnerable is True
    assert finding.probe_id == "echo_test"
    assert finding.attempt.raw_output == "this is a bad payload"
    assert finding.attempt.sandbox_result is not None
    assert finding.attempt.sandbox_result.succeeded


def test_run_reports_not_vulnerable_for_benign_payload():
    """``run()`` reports ``vulnerable=False`` with ``INFO`` severity for a
    payload that does not contain the stub adapter's "bad" trigger word."""
    adapter = _EchoAdapter()
    executor = _StubExecutor()

    finding = adapter.run("a perfectly nice payload", executor)

    assert finding.vulnerable is False
    assert finding.severity == SecuritySeverity.INFO
