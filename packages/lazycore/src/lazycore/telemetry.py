"""OpenTelemetry GenAI semantic-convention helpers (architecture doc §2.6).

LazyCore adopts OpenTelemetry GenAI semantic conventions as the shared
trace/result schema across LazyRed's security-audit reports, LazyAgent's
execution trajectories, and the leaderboard-style results emitted by the
AutoML module, LazyForecast, LazyGraph, and LazyVision. Data is represented
as OTel spans plus a small set of custom attributes (e.g.
``security.severity``, ``owasp.mapping``, ``ml.metric.accuracy``) and span
events for conversational transcripts.

This module depends only on ``opentelemetry-api`` (never the SDK). Without
an SDK/exporter configured by the *application*, the tracer returned here is
a documented OTel no-op tracer -- spans are created and attributes/events
are accepted, but nothing is exported anywhere. That's intentional: it keeps
``lazycore`` import-safe and dependency-light for every module, while still
letting a module that *does* wire up the SDK (e.g. an OTLP exporter, or a
console exporter for local debugging) get consistently-shaped spans for
free.
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

__all__ = [
    "ATTR_SECURITY_SEVERITY",
    "ATTR_OWASP_MAPPING",
    "ATTR_ML_METRIC_PREFIX",
    "ATTR_ML_METRIC_ACCURACY",
    "ATTR_GENAI_EVENT_ROLE",
    "ATTR_GENAI_EVENT_CONTENT",
    "ATTR_GENAI_EVENT_CONTENT_LENGTH",
    "ATTR_GENAI_EVENT_CONTENT_SHA256",
    "SecuritySeverity",
    "ml_metric_attribute",
    "get_tracer",
    "genai_span",
    "set_security_finding",
    "set_ml_metric",
    "add_transcript_event",
]

# Default tracer/instrumentation-scope name used when a caller doesn't
# supply their own module name.
_DEFAULT_TRACER_NAME = "benchcraft.lazycore"

# --- Shared attribute names (§2.6) -----------------------------------------

#: Severity of a security/red-team finding, one of the values in
#: :class:`SecuritySeverity`. Used by LazyRed's security-audit reports.
ATTR_SECURITY_SEVERITY = "security.severity"

#: OWASP LLM Top 10 / OWASP Agentic Top 10 / MITRE ATLAS mapping ID(s) for a
#: given finding. Used by LazyRed.
ATTR_OWASP_MAPPING = "owasp.mapping"

#: Namespace prefix for ML leaderboard metrics (accuracy, F1, PICP, MPIW,
#: etc.) reported by the AutoML module, LazyForecast, LazyGraph, and
#: LazyVision.
ATTR_ML_METRIC_PREFIX = "ml.metric."

#: The specific example attribute named in the architecture doc.
ATTR_ML_METRIC_ACCURACY = f"{ATTR_ML_METRIC_PREFIX}accuracy"

#: Span-event attribute keys for conversational transcripts (agent
#: trajectories, red-team conversations).
ATTR_GENAI_EVENT_ROLE = "gen_ai.event.role"
ATTR_GENAI_EVENT_CONTENT = "gen_ai.event.content"

#: Metadata-only attribute key used by :func:`add_transcript_event`'s
#: safe-by-default path (no raw content attached): the content's length,
#: so a caller/operator can still get a coarse signal about a transcript
#: turn across an exported trace without the raw text (prompts, tool
#: output, credentials, PII, etc.) ever leaving the process.
ATTR_GENAI_EVENT_CONTENT_LENGTH = "gen_ai.event.content.length"

#: Reserved for a possible future *opt-in, keyed* content-correlation
#: attribute (e.g. an HMAC). :func:`add_transcript_event` never sets this
#: itself -- an earlier version of this module attached an *unsalted*
#: SHA-256 hash of ``content`` here by default, which was a real leak
#: vector (see that function's docstring) and has been removed. The name
#: stays defined and reserved (so ``extra_attributes`` still can't spoof
#: it) purely for forward/backward attribute-key stability.
ATTR_GENAI_EVENT_CONTENT_SHA256 = "gen_ai.event.content.sha256"


class SecuritySeverity(str, enum.Enum):
    """Allowed values for :data:`ATTR_SECURITY_SEVERITY`.

    Modeled on common vulnerability-severity scales used by LazyRed's
    OWASP/MITRE ATLAS-mapped findings.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def ml_metric_attribute(metric_name: str) -> str:
    """Build a namespaced ``ml.metric.<metric_name>`` attribute key.

    Example: ``ml_metric_attribute("accuracy") == "ml.metric.accuracy"``.
    """
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    return f"{ATTR_ML_METRIC_PREFIX}{metric_name}"


def get_tracer(instrumenting_module_name: str | None = None) -> Tracer:
    """Return an OTel tracer scoped to ``instrumenting_module_name``.

    Thin wrapper over ``opentelemetry.trace.get_tracer`` so every module
    goes through one lazycore entrypoint rather than importing
    ``opentelemetry.trace`` directly. If the calling application has not
    configured a TracerProvider (via the SDK), this returns OTel's
    documented no-op tracer -- calls are safe, they simply don't export.
    """
    return trace.get_tracer(instrumenting_module_name or _DEFAULT_TRACER_NAME)


@contextmanager
def genai_span(
    name: str,
    *,
    tracer: Tracer | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Start a span using the shared GenAI schema conventions.

    Usage::

        with genai_span("lazyred.probe.run", attributes={...}) as span:
            set_security_finding(span, severity=SecuritySeverity.HIGH,
                                  owasp_mapping=["LLM01"])
            ...

    ``tracer`` defaults to :func:`get_tracer` with the shared lazycore
    instrumentation-scope name; pass your own (e.g.
    ``get_tracer(__name__)``) if you want spans attributed to your module.
    """
    active_tracer = tracer or get_tracer()
    with active_tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def set_security_finding(
    span: Span,
    *,
    severity: SecuritySeverity | str,
    owasp_mapping: str | Sequence[str] | None = None,
) -> None:
    """Attach a security-audit finding's severity/OWASP mapping to ``span``.

    ``severity`` may be a :class:`SecuritySeverity` member or a raw string
    matching one of its values; anything else raises ``ValueError`` so
    inconsistent severity strings don't silently leak into telemetry.
    """
    if isinstance(severity, SecuritySeverity):
        severity_value = severity.value
    else:
        try:
            severity_value = SecuritySeverity(severity).value
        except ValueError as exc:
            valid = ", ".join(s.value for s in SecuritySeverity)
            raise ValueError(
                f"Invalid security severity {severity!r}; expected one of: {valid}"
            ) from exc

    span.set_attribute(ATTR_SECURITY_SEVERITY, severity_value)
    if owasp_mapping is not None:
        mapping_value = (
            [owasp_mapping] if isinstance(owasp_mapping, str) else list(owasp_mapping)
        )
        span.set_attribute(ATTR_OWASP_MAPPING, mapping_value)


def set_ml_metric(span: Span, metric_name: str, value: float) -> None:
    """Attach a leaderboard metric (e.g. ``accuracy``, ``picp``) to ``span``
    under the shared ``ml.metric.*`` namespace."""
    span.set_attribute(ml_metric_attribute(metric_name), value)


def add_transcript_event(
    span: Span,
    role: str,
    content: str,
    *,
    include_raw_content: bool = False,
    sanitizer: Callable[[str], str] | None = None,
    **extra_attributes: Any,
) -> None:
    """Record one turn of a conversational transcript as a span event.

    Used for agent trajectories (LazyAgent) and red-team conversations
    (LazyRed) so both modules represent multi-turn transcripts the same
    way: one span event per turn, tagged with the speaker ``role`` (e.g.
    ``"user"``, ``"assistant"``, ``"tool"``) and its content.

    **Security rationale / safe-by-default contract.** ``content`` here is
    exactly the kind of data a red-team run (LazyRed) or agent benchmark
    (LazyAgent) is *expected* to surface: arbitrary model prompts, tool
    output, and adversarial payloads, which can and do contain credentials,
    secrets, or PII discovered mid-run. An OTel span is exportable
    telemetry -- once an SDK/exporter is configured by the application, its
    attributes/events can leave the process (to a collector, a file, a
    dashboard). Recording ``content`` verbatim into an exportable span by
    default would silently turn every red-team/agent run's telemetry
    pipeline into a potential credential/PII leak, with no signal to the
    caller that this is happening. This function therefore does **not**
    attach raw ``content`` to the span unless the caller explicitly opts
    in, via one of:

    - ``sanitizer``: a callable applied to ``content`` before it is
      attached (e.g. a redaction function that masks anything
      credential/PII-shaped). Its return value -- not the original
      ``content`` -- is what gets attached under
      :data:`ATTR_GENAI_EVENT_CONTENT`. Takes precedence over
      ``include_raw_content`` if both are given.
    - ``include_raw_content=True``: attach ``content`` completely
      unmodified. Only appropriate when the caller has independently
      verified it's safe to export as-is (e.g. a controlled test fixture,
      or a pipeline stage the caller has already sanitized upstream).

    When neither is given (the default), no raw or sanitized text is
    attached at all -- only :data:`ATTR_GENAI_EVENT_CONTENT_LENGTH`
    (``len(content)``). No content hash of any kind is attached by default.

    **Why no default hash.** An earlier version of this function attached
    an *unsalted* SHA-256 hash of ``content`` (:data:`ATTR_GENAI_EVENT_CONTENT_SHA256`)
    alongside the length, reasoning that a hash alone couldn't leak the
    original text. That reasoning was wrong on two counts, and this
    function no longer does it:

    1. **Correlation without plaintext is still a leak.** Two spans/traces
       carrying the same unsalted hash are provably carrying the same
       underlying content -- an observer can conclude "this run's prompt
       is identical to a known-sensitive one" without ever seeing either
       plaintext.
    2. **Unsalted/un-keyed SHA-256 is reversible for guessable input.**
       Prompts, tool output, and credential-shaped strings are frequently
       low-entropy or drawn from a small/known space (common phrases,
       known secret formats, short strings). SHA-256 is fast to compute
       and offers no protection against a dictionary/brute-force attack
       recovering the plaintext from the hash when the input space is
       guessable -- it needs a secret salt/key to resist that, which a
       bare ``hashlib.sha256(content)`` call does not have.

    A correlation-safe version of this feature would require an *opt-in*,
    caller-supplied secret key (HMAC-SHA256 rather than a bare hash), so
    only someone holding that key could link two hashes to the same
    content. This module deliberately does not add that parameter yet:
    lazycore is still scaffold-depth, no module currently needs
    cross-trace content correlation, and an unused opt-in knob is exactly
    the kind of speculative surface area this repo's conventions ask
    contributors to avoid. If a real need for keyed correlation shows up
    in LazyRed/LazyAgent, add a ``correlation_key`` parameter that computes
    ``hmac.new(key, content.encode(...), hashlib.sha256)`` and attaches it
    under :data:`ATTR_GENAI_EVENT_CONTENT_SHA256` only when the key is
    explicitly supplied -- not as a default-on hash.

    This is a behavior change from an earlier version of this function,
    which always attached ``content`` verbatim, and from a more recent
    version that attached an unsalted SHA-256 hash by default. Every
    caller in *this* package's own tests has been updated to pass
    ``include_raw_content=True`` where a test genuinely needs to assert on
    the recorded content; callers in other Benchcraft packages (LazyRed,
    LazyAgent) must make the same explicit choice deliberately once they
    adopt this signature -- this function does not, and should not, guess
    on their behalf.

    **Reserved attribute keys.** ``extra_attributes`` may not set any of the
    attribute keys this function itself manages (:data:`ATTR_GENAI_EVENT_ROLE`,
    :data:`ATTR_GENAI_EVENT_CONTENT`, :data:`ATTR_GENAI_EVENT_CONTENT_LENGTH`,
    :data:`ATTR_GENAI_EVENT_CONTENT_SHA256`) -- doing so raises ``ValueError``.
    Without this check, a caller-supplied ``extra_attributes`` entry could
    silently overwrite (e.g.) ``gen_ai.event.content`` with an arbitrary
    value once merged into the final attributes dict, completely bypassing
    the safe-by-default redaction contract documented above.
    """
    reserved_attributes = {
        ATTR_GENAI_EVENT_ROLE,
        ATTR_GENAI_EVENT_CONTENT,
        ATTR_GENAI_EVENT_CONTENT_LENGTH,
        ATTR_GENAI_EVENT_CONTENT_SHA256,
    }
    colliding_keys = reserved_attributes.intersection(extra_attributes)
    if colliding_keys:
        raise ValueError(
            "extra_attributes may not set reserved attribute key(s): "
            f"{sorted(colliding_keys)}"
        )

    attributes: dict[str, Any] = dict(extra_attributes)
    attributes[ATTR_GENAI_EVENT_ROLE] = role

    if sanitizer is not None:
        attributes[ATTR_GENAI_EVENT_CONTENT] = sanitizer(content)
    elif include_raw_content:
        attributes[ATTR_GENAI_EVENT_CONTENT] = content
    else:
        attributes[ATTR_GENAI_EVENT_CONTENT_LENGTH] = len(content)

    span.add_event(name="gen_ai.content", attributes=attributes)
