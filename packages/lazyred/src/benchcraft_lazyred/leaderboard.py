"""A tiny "Vulnerability & Failure Rate Leaderboard" stand-in (architecture
doc Part 3, "Module 7: LazyRed").

The architecture doc describes findings compiling into "a unified JSONL
report ... on a Vulnerability & Failure Rate Leaderboard." That full
report machinery (unified JSONL schema, OWASP/MITRE ATLAS ID aggregation
across an arbitrary number of probes, multi-run historical comparison) is
explicitly out of scope for this scaffold pass -- see README. What's
implemented here is the small, real piece that machinery would sit on top
of: running one probe N times with slight payload variations and
aggregating the resulting :class:`~benchcraft_lazyred.adapter.Finding`
objects into a printable, in-memory pass/fail summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from lazycore.sandbox import BaseSandboxExecutor

from benchcraft_lazyred.adapter import BaseSecurityAdapter, Finding

__all__ = ["LeaderboardReport", "run_leaderboard"]


@dataclass(frozen=True)
class LeaderboardReport:
    """Aggregated pass/fail result of running one adapter over N payloads."""

    probe_id: str
    findings: tuple[Finding, ...]

    @property
    def total_attempts(self) -> int:
        """Number of findings aggregated into this report (one per payload run)."""
        return len(self.findings)

    @property
    def vulnerable_count(self) -> int:
        """Number of attempts where the target was genuinely vulnerable
        (i.e. :attr:`Finding.vulnerable` is ``True``)."""
        return sum(1 for finding in self.findings if finding.vulnerable)

    @property
    def inconclusive_count(self) -> int:
        """Attempts that could not be evaluated at all (target crashed,
        timed out, or was blocked by the sandbox for reasons unrelated to
        the probe's own intent) -- see :attr:`Finding.inconclusive`. These
        are neither a genuine pass nor a genuine fail and must never be
        silently folded into :attr:`resisted_count`."""
        return sum(1 for finding in self.findings if finding.inconclusive)

    @property
    def resisted_count(self) -> int:
        """Attempts the target genuinely, verifiably resisted -- i.e. it
        ran to completion and did not leak/fail. Excludes both vulnerable
        and inconclusive attempts, so an errored/blocked run is never
        counted as a "pass"."""
        return self.total_attempts - self.vulnerable_count - self.inconclusive_count

    @property
    def failure_rate(self) -> float:
        """Fraction of attempts where the target was vulnerable (i.e. the
        target *failed* to resist the attack). ``0.0`` if there were no
        attempts, rather than raising a division error -- an empty
        leaderboard run is a valid (if useless) state, not an error."""
        if self.total_attempts == 0:
            return 0.0
        return self.vulnerable_count / self.total_attempts

    @property
    def inconclusive_rate(self) -> float:
        """Fraction of attempts that could not be evaluated at all."""
        if self.total_attempts == 0:
            return 0.0
        return self.inconclusive_count / self.total_attempts

    @property
    def pass_rate(self) -> float:
        """Fraction of attempts the target *genuinely, verifiably*
        resisted. Computed from :attr:`resisted_count` (not as
        ``1.0 - failure_rate``) so that inconclusive attempts -- which are
        neither vulnerable nor a confirmed resist -- are never silently
        counted as passes."""
        if self.total_attempts == 0:
            return 1.0
        return self.resisted_count / self.total_attempts

    def format_summary(self) -> str:
        """A small, human-printable summary table -- not the full unified
        JSONL report described in the architecture doc, just enough to
        demonstrate the aggregation at this scaffold's depth."""
        lines = [
            f"LazyRed Vulnerability & Failure Rate Leaderboard -- probe={self.probe_id!r}",
            f"  total attempts   : {self.total_attempts}",
            f"  vulnerable (fail): {self.vulnerable_count}",
            f"  resisted (pass)  : {self.resisted_count}",
            f"  inconclusive     : {self.inconclusive_count}",
            f"  failure rate     : {self.failure_rate:.1%}",
            f"  pass rate        : {self.pass_rate:.1%}",
            "",
            "  per-attempt detail:",
        ]
        for index, finding in enumerate(self.findings, start=1):
            if finding.inconclusive:
                verdict = "INCONCLUSIVE"
            elif finding.vulnerable:
                verdict = "VULNERABLE"
            else:
                verdict = "resisted"
            payload = finding.attempt.payload
            lines.append(
                f"    [{index:02d}] {verdict:<10} "
                f"severity={finding.severity.value:<8} "
                f"owasp={','.join(finding.owasp_mapping):<28} "
                f"payload={payload!r}"
            )
        return "\n".join(lines)


def run_leaderboard(
    adapter: BaseSecurityAdapter,
    executor: BaseSandboxExecutor,
    payloads: Sequence[str],
) -> LeaderboardReport:
    """Run ``adapter`` once per payload in ``payloads`` and aggregate results.

    Each payload runs through the adapter's full
    ``generate_attempt`` -> ``run_target`` -> ``detect`` chain (via
    :meth:`~benchcraft_lazyred.adapter.BaseSecurityAdapter.run`), so every
    attempt genuinely goes through the shared sandbox executor -- this
    function does not itself touch the sandbox, it only sequences and
    aggregates.
    """
    findings = tuple(adapter.run(payload, executor) for payload in payloads)
    probe_id = findings[0].probe_id if findings else getattr(adapter, "probe_id", "unknown")
    return LeaderboardReport(probe_id=probe_id, findings=findings)
