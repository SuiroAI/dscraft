"""Tests for the leaderboard pass/fail aggregation (leaderboard.py).

Uses a trivial stub adapter/executor (rather than the real sandbox) so
this suite is purely about aggregation-logic correctness; the real-sandbox
version of this same aggregation is covered end-to-end by
``test_probes.py::test_leaderboard_via_probe_mixed_payloads_matches_manual_count``.
"""

from __future__ import annotations

from lazycore.sandbox import BaseSandboxExecutor, SandboxResult
from lazycore.telemetry import SecuritySeverity

from benchcraft_lazyred.adapter import Attempt, BaseSecurityAdapter, Finding
from benchcraft_lazyred.leaderboard import LeaderboardReport, run_leaderboard


class _NoOpExecutor(BaseSandboxExecutor):
    def is_available(self) -> bool:
        return True

    def run_command(self, command, *, policy=None):
        return SandboxResult(exit_code=0, stdout="", stderr="")

    def run_callable(self, func, *, policy=None):
        return SandboxResult(exit_code=0, stdout=repr(func()), stderr="")


class _ContainsBadAdapter(BaseSecurityAdapter):
    """Flags "vulnerable" whenever the payload contains the substring "bad"."""

    probe_id = "contains_bad"

    def generate_attempt(self, probe_input: str) -> Attempt:
        return Attempt(probe_id=self.probe_id, payload=probe_input, prompt=probe_input)

    def run_target(self, attempt: Attempt, executor: BaseSandboxExecutor) -> Attempt:
        attempt.sandbox_result = executor.run_command(["true"])
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


class _NeverExecutedAdapter(BaseSecurityAdapter):
    """Simulates an attempt that was never actually executed against the
    target at all: ``run_target`` deliberately leaves
    ``attempt.sandbox_result`` as ``None`` (e.g. standing in for a real
    adapter that hit an exception before a ``SandboxResult`` could even be
    constructed), rather than returning a failed/crashed
    ``SandboxResult``. Used to verify the aggregation logic correctly
    excludes this ``None``-triggered inconclusive case from
    ``resisted_count``/``pass_rate``, the same way it already does for the
    ``sandbox_result.succeeded=False`` case -- without assuming that
    coverage generalizes without a dedicated test."""

    probe_id = "never_executed"

    def generate_attempt(self, probe_input: str) -> Attempt:
        return Attempt(probe_id=self.probe_id, payload=probe_input, prompt=probe_input)

    def run_target(self, attempt: Attempt, executor: BaseSandboxExecutor) -> Attempt:
        # Deliberately does not call the executor and leaves
        # sandbox_result as None -- the attempt was never executed.
        return attempt

    def detect(self, attempt: Attempt) -> Finding:
        if attempt.sandbox_result is None:
            return Finding(
                probe_id=attempt.probe_id,
                vulnerable=False,
                severity=SecuritySeverity.INFO,
                owasp_mapping=("LLM01: Prompt Injection",),
                detail="Attempt was never executed (no sandbox_result recorded).",
                attempt=attempt,
                inconclusive=True,
            )
        raise AssertionError("sandbox_result should always be None in this stub")


def test_leaderboard_excludes_none_sandbox_result_inconclusive_from_resisted():
    """Regression coverage for the narrower ``sandbox_result is None`` case
    (never executed at all, as opposed to executed-but-failed): the
    leaderboard must exclude these from ``resisted_count``/``pass_rate``
    exactly like the existing ``succeeded=False`` inconclusive case."""
    adapter = _NeverExecutedAdapter()
    executor = _NoOpExecutor()

    report = run_leaderboard(adapter, executor, ["payload one", "payload two", "payload three"])

    assert report.total_attempts == 3
    assert report.vulnerable_count == 0
    assert report.inconclusive_count == 3
    assert report.resisted_count == 0
    assert report.pass_rate == 0.0
    assert report.failure_rate == 0.0
    assert "INCONCLUSIVE" in report.format_summary()


def test_run_leaderboard_aggregates_mixed_pass_fail():
    """With 2 of 4 payloads containing "bad", ``run_leaderboard`` produces a
    report with the correct total/vulnerable/resisted counts and a 50%
    failure rate / 50% pass rate."""
    adapter = _ContainsBadAdapter()
    executor = _NoOpExecutor()
    payloads = ["good one", "a bad payload", "another good one", "so bad"]

    report = run_leaderboard(adapter, executor, payloads)

    assert isinstance(report, LeaderboardReport)
    assert report.probe_id == "contains_bad"
    assert report.total_attempts == 4
    assert report.vulnerable_count == 2
    assert report.resisted_count == 2
    assert report.failure_rate == 0.5
    assert report.pass_rate == 0.5


def test_run_leaderboard_all_pass():
    """When no payload contains the "bad" trigger, the report shows zero
    vulnerable attempts, 0% failure rate, and 100% pass rate."""
    adapter = _ContainsBadAdapter()
    executor = _NoOpExecutor()
    payloads = ["fine", "also fine", "still fine"]

    report = run_leaderboard(adapter, executor, payloads)

    assert report.vulnerable_count == 0
    assert report.failure_rate == 0.0
    assert report.pass_rate == 1.0


def test_run_leaderboard_all_fail():
    """When every payload contains the "bad" trigger, the report shows all
    attempts vulnerable, 100% failure rate, and 0% pass rate."""
    adapter = _ContainsBadAdapter()
    executor = _NoOpExecutor()
    payloads = ["bad", "so bad", "extremely bad"]

    report = run_leaderboard(adapter, executor, payloads)

    assert report.vulnerable_count == 3
    assert report.failure_rate == 1.0
    assert report.pass_rate == 0.0


def test_run_leaderboard_empty_payloads_does_not_error():
    """Running the leaderboard with an empty payload list does not raise a
    division error; it returns a valid, empty report with 0% failure rate
    and 100% pass rate (per ``LeaderboardReport``'s documented convention
    for the zero-attempts case) while still reporting the adapter's
    ``probe_id``."""
    adapter = _ContainsBadAdapter()
    executor = _NoOpExecutor()

    report = run_leaderboard(adapter, executor, [])

    assert report.total_attempts == 0
    assert report.vulnerable_count == 0
    assert report.failure_rate == 0.0
    assert report.pass_rate == 1.0
    assert report.probe_id == "contains_bad"


def test_format_summary_contains_key_fields():
    """``format_summary()`` renders a human-readable report that includes
    the probe id, the total attempt count, per-attempt VULNERABLE/resisted
    verdicts, and the OWASP mapping ("LLM01")."""
    adapter = _ContainsBadAdapter()
    executor = _NoOpExecutor()
    report = run_leaderboard(adapter, executor, ["bad", "good"])

    summary = report.format_summary()

    assert "contains_bad" in summary
    assert "total attempts   : 2" in summary
    assert "VULNERABLE" in summary
    assert "resisted" in summary
    assert "LLM01" in summary
