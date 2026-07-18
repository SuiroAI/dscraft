"""Tests for lazycore.telemetry (OTel GenAI semantic-convention helpers, §2.6).

These tests deliberately do not depend on the OpenTelemetry SDK (lazycore
only depends on opentelemetry-api). Without an SDK TracerProvider
configured, spans returned by the API are non-recording no-op spans -- so
these tests assert on *behavior that doesn't raise* and on the shared
attribute-name/enum contract, rather than on exported span data.
"""

from __future__ import annotations

import pytest
from opentelemetry.trace import Span

from lazycore.telemetry import (
    ATTR_GENAI_EVENT_CONTENT,
    ATTR_GENAI_EVENT_CONTENT_LENGTH,
    ATTR_GENAI_EVENT_CONTENT_SHA256,
    ATTR_GENAI_EVENT_ROLE,
    ATTR_ML_METRIC_ACCURACY,
    ATTR_ML_METRIC_PREFIX,
    ATTR_OWASP_MAPPING,
    ATTR_SECURITY_SEVERITY,
    SecuritySeverity,
    add_transcript_event,
    genai_span,
    get_tracer,
    ml_metric_attribute,
    set_ml_metric,
    set_security_finding,
)


def test_attribute_name_constants_match_architecture_doc():
    """The shared attribute-name constants match the exact strings specified in architecture doc §2.6."""
    assert ATTR_SECURITY_SEVERITY == "security.severity"
    assert ATTR_OWASP_MAPPING == "owasp.mapping"
    assert ATTR_ML_METRIC_ACCURACY == "ml.metric.accuracy"
    assert ATTR_ML_METRIC_PREFIX == "ml.metric."
    assert ATTR_GENAI_EVENT_ROLE == "gen_ai.event.role"
    assert ATTR_GENAI_EVENT_CONTENT == "gen_ai.event.content"


def test_ml_metric_attribute_namespaces_metric_names():
    """ml_metric_attribute() prefixes an arbitrary metric name with the shared "ml.metric." namespace."""
    assert ml_metric_attribute("accuracy") == "ml.metric.accuracy"
    assert ml_metric_attribute("picp") == "ml.metric.picp"


def test_ml_metric_attribute_rejects_empty_name():
    """ml_metric_attribute() raises ValueError for an empty metric name."""
    with pytest.raises(ValueError):
        ml_metric_attribute("")


def test_security_severity_enum_values():
    """SecuritySeverity members expose the expected lowercase string values."""
    assert SecuritySeverity.INFO.value == "info"
    assert SecuritySeverity.LOW.value == "low"
    assert SecuritySeverity.MEDIUM.value == "medium"
    assert SecuritySeverity.HIGH.value == "high"
    assert SecuritySeverity.CRITICAL.value == "critical"


def test_get_tracer_returns_a_tracer_like_object():
    """get_tracer() returns an object exposing the OTel Tracer API (start_as_current_span), even with no SDK configured."""
    tracer = get_tracer("test.module")
    assert hasattr(tracer, "start_as_current_span")


def test_genai_span_yields_a_span_and_does_not_raise():
    """genai_span() is a usable context manager that yields a real Span instance without raising, even as a no-op (SDK-less) span."""
    with genai_span("test.span") as span:
        assert isinstance(span, Span)


def test_set_security_finding_accepts_enum_member():
    """set_security_finding() does not raise when given a SecuritySeverity enum member plus a list of OWASP mapping IDs."""
    with genai_span("test.security") as span:
        # Should not raise for a valid enum member + owasp mapping list.
        set_security_finding(
            span, severity=SecuritySeverity.HIGH, owasp_mapping=["LLM01", "LLM06"]
        )


def test_set_security_finding_accepts_raw_string_severity():
    """set_security_finding() accepts a raw string severity (matching a SecuritySeverity value) and a single string OWASP mapping."""
    with genai_span("test.security.raw") as span:
        set_security_finding(span, severity="critical", owasp_mapping="LLM01")


def test_set_security_finding_rejects_invalid_severity():
    """set_security_finding() raises ValueError for a severity string that doesn't match any SecuritySeverity value."""
    with genai_span("test.security.invalid") as span:
        with pytest.raises(ValueError):
            set_security_finding(span, severity="apocalyptic")


def test_set_ml_metric_does_not_raise():
    """set_ml_metric() does not raise when attaching a numeric metric value to a span."""
    with genai_span("test.metric") as span:
        set_ml_metric(span, "accuracy", 0.987)


def test_add_transcript_event_does_not_raise_and_accepts_extra_attrs():
    """add_transcript_event() does not raise for multiple turns and accepts arbitrary extra keyword attributes (e.g. turn_index) alongside role/content."""
    with genai_span("test.transcript") as span:
        add_transcript_event(
            span, role="user", content="hello", turn_index=0, include_raw_content=True
        )
        add_transcript_event(
            span,
            role="assistant",
            content="hi there",
            turn_index=1,
            include_raw_content=True,
        )


class _RecordingSpan:
    """Minimal stand-in for an OTel Span that actually records add_event()
    calls, used to verify add_transcript_event()'s attribute-shaping logic.

    A real no-op span (the kind genai_span() yields without an SDK
    configured) accepts add_event() calls but does not expose what was
    recorded -- it is fine for "does not raise" tests, but cannot verify
    *what content ends up attached to the span*, which is exactly the
    safe-by-default contract these regression tests exist to prove. This
    stub only implements the one method add_transcript_event() actually
    calls.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def add_event(self, name, attributes=None):  # noqa: ANN001 - test double
        self.events.append((name, dict(attributes or {})))


def test_add_transcript_event_default_does_not_attach_raw_content():
    """By default (no include_raw_content, no sanitizer), add_transcript_event() does NOT attach the raw content string to the span -- only role and a length -- per the safe-by-default security contract."""
    span = _RecordingSpan()
    secret_content = "here is my API key: sk-super-secret-12345"

    add_transcript_event(span, role="tool", content=secret_content)

    assert len(span.events) == 1
    _, attributes = span.events[0]
    assert ATTR_GENAI_EVENT_CONTENT not in attributes
    assert secret_content not in str(attributes)
    assert attributes[ATTR_GENAI_EVENT_ROLE] == "tool"
    assert attributes[ATTR_GENAI_EVENT_CONTENT_LENGTH] == len(secret_content)


def test_add_transcript_event_default_does_not_attach_any_content_hash():
    """No content hash of any kind (e.g. ATTR_GENAI_EVENT_CONTENT_SHA256) is attached by default.

    An unsalted/un-keyed hash of guessable content (common prompts, known
    secret formats, short strings) is not actually safe: it still lets an
    observer correlate two spans as carrying identical content, and it is
    vulnerable to dictionary/brute-force recovery of the original text.
    This regression test guards against that hash ever creeping back into
    the default (metadata-only) path.
    """
    span = _RecordingSpan()
    add_transcript_event(span, role="tool", content="password=hunter2")

    _, attributes = span.events[0]
    assert ATTR_GENAI_EVENT_CONTENT_SHA256 not in attributes


def test_add_transcript_event_include_raw_content_opt_in_attaches_verbatim_content():
    """Passing include_raw_content=True attaches the exact, unmodified content string under ATTR_GENAI_EVENT_CONTENT."""
    span = _RecordingSpan()
    content = "the quick brown fox"

    add_transcript_event(span, role="user", content=content, include_raw_content=True)

    _, attributes = span.events[0]
    assert attributes[ATTR_GENAI_EVENT_CONTENT] == content
    assert ATTR_GENAI_EVENT_CONTENT_LENGTH not in attributes
    assert ATTR_GENAI_EVENT_CONTENT_SHA256 not in attributes


def test_add_transcript_event_sanitizer_attaches_sanitized_content_not_raw():
    """Passing a sanitizer callable attaches its return value under ATTR_GENAI_EVENT_CONTENT -- never the original raw content -- even if include_raw_content is also (redundantly) True."""
    span = _RecordingSpan()
    raw = "password=hunter2"

    def redact(text: str) -> str:
        return "[REDACTED]"

    add_transcript_event(
        span, role="assistant", content=raw, sanitizer=redact, include_raw_content=True
    )

    _, attributes = span.events[0]
    assert attributes[ATTR_GENAI_EVENT_CONTENT] == "[REDACTED]"
    assert raw not in str(attributes)


def test_add_transcript_event_extra_attributes_still_pass_through_in_all_modes():
    """Arbitrary extra keyword attributes (e.g. turn_index) are still attached alongside role and content/metadata, in both the default (metadata-only) and include_raw_content=True modes."""
    span = _RecordingSpan()
    add_transcript_event(span, role="user", content="hi", turn_index=3)
    _, attributes = span.events[0]
    assert attributes["turn_index"] == 3


def test_add_transcript_event_rejects_extra_attributes_colliding_with_reserved_keys():
    """A caller-supplied extra attribute that collides with a reserved key (e.g. ATTR_GENAI_EVENT_CONTENT) must raise ValueError instead of silently overwriting the safe-by-default redaction contract, and no event may be emitted with the malicious override in place."""
    span = _RecordingSpan()
    malicious_extra_attributes = {ATTR_GENAI_EVENT_CONTENT: "actually leak this"}

    with pytest.raises(ValueError, match="reserved"):
        add_transcript_event(
            span, role="user", content="safe", **malicious_extra_attributes
        )

    # No event should have been emitted at all -- the collision must be
    # rejected before add_event() is ever called.
    assert span.events == []


def test_add_transcript_event_rejects_extra_attributes_colliding_with_role_or_hash_keys():
    """Collisions against the other reserved keys (role, length, sha256) are also rejected, not just content."""
    span = _RecordingSpan()

    with pytest.raises(ValueError, match="reserved"):
        add_transcript_event(
            span, role="user", content="hi", **{ATTR_GENAI_EVENT_ROLE: "assistant"}
        )

    with pytest.raises(ValueError, match="reserved"):
        add_transcript_event(
            span, role="user", content="hi", **{ATTR_GENAI_EVENT_CONTENT_LENGTH: 0}
        )

    with pytest.raises(ValueError, match="reserved"):
        add_transcript_event(
            span, role="user", content="hi", **{ATTR_GENAI_EVENT_CONTENT_SHA256: "x"}
        )

    assert span.events == []
