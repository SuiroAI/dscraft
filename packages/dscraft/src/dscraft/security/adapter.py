"""``BaseSecurityAdapter`` pattern for local-target red-teaming (architecture
doc Part 3, "Module 7: LazyRed").

The architecture doc describes LazyRed as unifying garak's Probes/
Detectors/Generators/Evaluators/Harnesses/Buffs around an ``attempt``
transaction object, plus DeepTeam/PyRIT/Promptfoo, under a
``BaseSecurityAdapter`` pattern. Nothing in `dscraft.core` defines this base
class -- per §2.9, formal inter-module contracts are explicitly deferred,
and this is LazyRed-specific red-teaming machinery, not shared
cross-module infrastructure -- so it is defined here as this package's one
canonical adapter interface, mirroring the same reasoning
``dscraft.tune.adapter.BaseTrainingAdapter`` used for LazyTune's
Adapter-Factory pattern.

Only one concrete adapter is implemented against this interface in this
pass: :class:`dscraft.security.probes.PromptInjectionAdapter` (see that
module). This file only defines the shared shape:

- :class:`Attempt` -- garak's "attempt" transaction object, scaled down to
  this scaffold's depth: the payload sent to the target, the full prompt
  built from it, the target's raw output, and the structured
  :class:`~dscraft.core.sandbox.SandboxResult` from actually invoking the
  target through the shared sandbox executor.
- :class:`Finding` -- one probe attempt's verdict, mapped to an OWASP
  LLM/Agentic Top 10 ID per architecture doc §2.6/Part 3, carrying enough
  detail to be reported via :mod:`dscraft.core.telemetry` and aggregated by
  :mod:`dscraft.security.leaderboard`.
- :class:`BaseSecurityAdapter` -- the three-method interface
  (``generate_attempt`` / ``run_target`` / ``detect``) plus a convenience
  ``run()`` that chains all three, matching the task's required shape.

Per §2.3, LazyRed does **not** define its own sandbox executor class here
-- ``run_target`` is required to take a
:class:`~dscraft.core.sandbox.BaseSandboxExecutor` (the shared executor from
`dscraft.core.sandbox`) and is responsible only for supplying a LazyRed-shaped
:class:`~dscraft.core.sandbox.SandboxPolicy` on top of it. See
``probes.py``'s ``build_probe_sandbox_policy`` for LazyRed's mode-specific
policy values.

:class:`BaseSecurityAdapter` itself subclasses the one shared
`dscraft.core.adapter.BaseSandboxedAdapter` base (also per §2.3's "one ...
adapter base class" requirement) rather than defining an independent
`abc.ABC` hierarchy -- see that module's docstring for what is actually
shared between this and `dscraft.agent.adapter.AgentAdapter`. This does
**not** weaken the separate-Guardrail/Firewall-layer boundary from §2.3's
last sentence: LazyRed's semantic-level probe/detector logic (prompt
injection triggers, secret-leak detection in ``probes.py``) stays entirely
in `dscraft.security`; only the generic sandbox-wiring adapter substrate is
shared.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from dscraft.core.adapter import BaseSandboxedAdapter
from dscraft.core.sandbox import BaseSandboxExecutor, SandboxResult
from dscraft.core.telemetry import SecuritySeverity

__all__ = [
    "Attempt",
    "Finding",
    "BaseSecurityAdapter",
]


@dataclass
class Attempt(object):
    """One probe "attempt" transaction (garak's term, per Part 3).

    Bundles everything about a single red-team attempt against a local
    target: the attack payload, the fully-built prompt sent to the target,
    the target's raw output, and the raw
    :class:`~dscraft.core.sandbox.SandboxResult` from running the target
    invocation through the shared sandbox executor (§2.3). Mutated in
    place as it moves through :meth:`BaseSecurityAdapter.generate_attempt`
    -> :meth:`~BaseSecurityAdapter.run_target` ->
    :meth:`~BaseSecurityAdapter.detect`, mirroring garak's own
    single-object-threaded-through-the-pipeline attempt design.
    """

    #: Stable identifier for the probe that produced this attempt (e.g.
    #: ``"prompt_injection"``).
    probe_id: str

    #: The raw attack payload (before being embedded in a full prompt) --
    #: e.g. one variation of an injection trigger phrase.
    payload: str

    #: The full prompt/input actually sent to the target, after
    #: :meth:`BaseSecurityAdapter.generate_attempt` has built it from
    #: ``payload``. ``None`` until that step has run.
    prompt: str | None = None

    #: The target's raw output, decoded from the sandboxed run. ``None``
    #: until :meth:`BaseSecurityAdapter.run_target` has run.
    raw_output: str | None = None

    #: The structured result of the sandboxed target invocation. ``None``
    #: until :meth:`BaseSecurityAdapter.run_target` has run. Kept around
    #: (rather than discarded once ``raw_output`` is extracted) so callers
    #: can inspect ``policy_blocked``/``exit_code``/``stderr`` for
    #: debugging or for a future Guardrail/Firewall layer (§2.4, out of
    #: scope for this pass) to consume.
    sandbox_result: SandboxResult | None = None

    #: Free-form extra context (e.g. which payload-variation index this
    #: attempt is, for the leaderboard). Never read by this module's own
    #: logic -- purely a carry-along bag for callers.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding:
    """The verdict for one :class:`Attempt`, mapped to OWASP LLM Top 10.

    Reported via :mod:`dscraft.core.telemetry`'s GenAI-schema OTel helpers
    (§2.6) rather than a parallel reporting schema -- see
    :func:`dscraft.security.probes.PromptInjectionAdapter.detect` for
    where the actual span is emitted using this dataclass's fields.
    """

    probe_id: str

    #: True if the probe payload actually achieved its attack goal against
    #: the target (e.g. the protected secret leaked into the output) --
    #: i.e. the target is *vulnerable* to this attempt. Named to match
    #: garak/OWASP framing directly (a red-team "finding" is normally
    #: reported when something goes wrong), not "passed"/"failed" from the
    #: target's perspective, since that phrasing is ambiguous about which
    #: side "passing" refers to.
    vulnerable: bool

    severity: SecuritySeverity

    #: OWASP LLM Top 10 / OWASP Agentic Top 10 / MITRE ATLAS ID(s) this
    #: finding maps to (architecture doc §2.6, Part 3). A tuple because a
    #: single finding can legitimately map to more than one framework ID,
    #: even though this scaffold's one probe only ever emits one.
    owasp_mapping: tuple[str, ...]

    #: Short, human-readable explanation of the verdict (e.g. what
    #: substring/pattern was or wasn't found in the target's output).
    detail: str

    #: The attempt this finding was derived from, kept for traceability
    #: (e.g. the leaderboard or a future report can look up the exact
    #: payload/output pair behind a given verdict).
    attempt: Attempt

    #: True if the target could not actually be evaluated for this attempt
    #: -- it crashed, timed out, or was blocked by the sandbox for reasons
    #: unrelated to the probe's own semantic intent (see
    #: ``SandboxResult.succeeded`` / ``policy_blocked`` /
    #: ``exit_code``) -- as opposed to a genuine pass/fail verdict about
    #: whether the target resisted the attack. Defaults to ``False`` so
    #: existing call sites that only ever produce genuine verdicts are
    #: unaffected. When ``True``, ``vulnerable`` is always ``False`` (an
    #: inconclusive attempt is never reported as "the target resisted the
    #: attack" -- callers/aggregators must check this flag rather than
    #: inferring "resisted" from ``vulnerable is False`` alone).
    inconclusive: bool = False


class BaseSecurityAdapter(BaseSandboxedAdapter, abc.ABC):
    """Minimal ``BaseSecurityAdapter`` interface (architecture doc Part 3).

    Three required steps, matching garak's Probe -> Generator -> Detector
    pipeline scaled down to this scaffold's depth:

    1. :meth:`generate_attempt` -- turn a raw probe input into a full
       :class:`Attempt` (the prompt/payload to send to the target).
    2. :meth:`run_target` -- actually invoke the target, through the
       shared :mod:`dscraft.core.sandbox` executor, and record its raw output
       on the same :class:`Attempt`.
    3. :meth:`detect` -- score the attempt's output against a detector
       rule and produce a :class:`Finding`.

    There is deliberately only one concrete adapter built against this
    interface in this pass (per CLAUDE.md's "one canonical adapter
    interface" rule): :class:`dscraft.security.probes.PromptInjectionAdapter`.

    Subclasses `dscraft.core.adapter.BaseSandboxedAdapter`, the one shared
    adapter base LazyRed and LazyAgent both build on (§2.3) -- only the
    attempt/finding data shapes and the three abstract methods plus ``run``
    below are LazyRed-specific. `BaseSandboxedAdapter` itself is no longer
    an `abc.ABC` (it defines no abstract methods of its own), so this class
    adds `abc.ABC` directly alongside it to keep ``generate_attempt``/
    ``run_target``/``detect``'s ``@abc.abstractmethod`` decorators actually
    enforced (i.e. this class, like `BaseSandboxedAdapter`'s previous
    behavior, still cannot be instantiated directly).
    """

    @abc.abstractmethod
    def generate_attempt(self, probe_input: str) -> Attempt:
        """Build an :class:`Attempt` (prompt/payload) from ``probe_input``.

        Args:
            probe_input: the raw attack payload/variation for this attempt
                (e.g. one phrasing of an injection trigger).
        """

    @abc.abstractmethod
    def run_target(self, attempt: Attempt, executor: BaseSandboxExecutor) -> Attempt:
        """Invoke the target for ``attempt`` via the shared sandbox ``executor``.

        Must route the actual target invocation through ``executor``'s
        ``run_callable``/``run_command`` (per §2.3 -- LazyRed never builds
        its own sandbox mechanism) and populate
        ``attempt.raw_output``/``attempt.sandbox_result`` before returning
        it.
        """

    @abc.abstractmethod
    def detect(self, attempt: Attempt) -> Finding:
        """Score ``attempt.raw_output`` and produce a :class:`Finding`."""

    def run(self, probe_input: str, executor: BaseSandboxExecutor) -> Finding:
        """Convenience: chain ``generate_attempt`` -> ``run_target`` -> ``detect``."""
        self._require_sandbox_executor(executor)
        attempt = self.generate_attempt(probe_input)
        attempt = self.run_target(attempt, executor)
        return self.detect(attempt)
