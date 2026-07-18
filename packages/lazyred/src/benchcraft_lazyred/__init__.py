"""Benchcraft LazyRed: a scaffold-depth ``BaseSecurityAdapter`` red-teaming
loop against local targets (architecture doc Part 3, "Module 7: LazyRed").

Public API surface -- see each module's docstring for detail:

- :class:`~benchcraft_lazyred.adapter.Attempt`,
  :class:`~benchcraft_lazyred.adapter.Finding`,
  :class:`~benchcraft_lazyred.adapter.BaseSecurityAdapter` --
  the canonical adapter interface (``adapter.py``).
- :class:`~benchcraft_lazyred.probes.PromptInjectionAdapter`,
  :func:`~benchcraft_lazyred.probes.naive_vulnerable_target`,
  :func:`~benchcraft_lazyred.probes.detect_secret_leak`,
  :func:`~benchcraft_lazyred.probes.build_probe_sandbox_policy`,
  :func:`~benchcraft_lazyred.probes.default_payload_variations` --
  the one concrete probe implementation (``probes.py``).
- :class:`~benchcraft_lazyred.leaderboard.LeaderboardReport`,
  :func:`~benchcraft_lazyred.leaderboard.run_leaderboard` --
  the pass/fail aggregation report (``leaderboard.py``).

See README.md for scope, sandbox wiring, the OWASP mapping used, and what
is explicitly deferred (Guardrail/Firewall layer, garak/DeepTeam/PyRIT/
Promptfoo integration, TopicAttack mutator, Multi-Model Jury Consensus).
"""

from __future__ import annotations

from benchcraft_lazyred.adapter import Attempt, BaseSecurityAdapter, Finding
from benchcraft_lazyred.leaderboard import LeaderboardReport, run_leaderboard
from benchcraft_lazyred.probes import (
    BENIGN_PAYLOADS,
    DEFAULT_SECRET,
    OWASP_PROMPT_INJECTION,
    PROMPT_INJECTION_TRIGGERS,
    PromptInjectionAdapter,
    build_probe_sandbox_policy,
    default_payload_variations,
    detect_secret_leak,
    naive_vulnerable_target,
)

__all__ = [
    "Attempt",
    "Finding",
    "BaseSecurityAdapter",
    "PromptInjectionAdapter",
    "naive_vulnerable_target",
    "build_probe_sandbox_policy",
    "detect_secret_leak",
    "default_payload_variations",
    "DEFAULT_SECRET",
    "OWASP_PROMPT_INJECTION",
    "PROMPT_INJECTION_TRIGGERS",
    "BENIGN_PAYLOADS",
    "LeaderboardReport",
    "run_leaderboard",
]

__version__ = "0.1.0"
