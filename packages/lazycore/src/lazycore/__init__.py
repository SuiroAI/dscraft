"""lazycore: the thin, shared substrate underneath every Benchcraft module.

See the package README for scope and the "why thin" rationale. This module
re-exports the small public surface of ``lazycore.data``,
``lazycore.telemetry``, ``lazycore.licensing``, and ``lazycore.sandbox`` so
most consumers only need ``import lazycore``.
"""

from lazycore.data import (
    ArrowBackedFrame,
    DenseMediaPipeline,
    SparseFormat,
    SparseGraphTensorAdapter,
    from_polars_zero_copy,
    is_arrow_backed_pandas,
    pandas_arrow_dtypes,
    to_polars_zero_copy,
)
from lazycore.licensing import (
    Allowlist,
    Mitigation,
    ModelLicenseEntry,
    ModelTier,
    RestrictedLicenseNotAcceptedError,
    RISK_MITIGATIONS,
    RiskType,
)
from lazycore.sandbox import (
    BaseSandboxExecutor,
    LinuxNamespaceSandboxExecutor,
    SandboxBackendUnavailableError,
    SandboxError,
    SandboxPolicy,
    SandboxPolicyViolationError,
    SandboxResult,
    SeatbeltSandboxExecutor,
    get_default_executor,
)
from lazycore.telemetry import (
    ATTR_GENAI_EVENT_CONTENT,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data.py
    "ArrowBackedFrame",
    "DenseMediaPipeline",
    "SparseFormat",
    "SparseGraphTensorAdapter",
    "from_polars_zero_copy",
    "is_arrow_backed_pandas",
    "pandas_arrow_dtypes",
    "to_polars_zero_copy",
    # licensing.py
    "Allowlist",
    "Mitigation",
    "ModelLicenseEntry",
    "ModelTier",
    "RestrictedLicenseNotAcceptedError",
    "RISK_MITIGATIONS",
    "RiskType",
    # sandbox/__init__.py
    "BaseSandboxExecutor",
    "LinuxNamespaceSandboxExecutor",
    "SandboxBackendUnavailableError",
    "SandboxError",
    "SandboxPolicy",
    "SandboxPolicyViolationError",
    "SandboxResult",
    "SeatbeltSandboxExecutor",
    "get_default_executor",
    # telemetry.py
    "ATTR_GENAI_EVENT_CONTENT",
    "ATTR_GENAI_EVENT_ROLE",
    "ATTR_ML_METRIC_ACCURACY",
    "ATTR_ML_METRIC_PREFIX",
    "ATTR_OWASP_MAPPING",
    "ATTR_SECURITY_SEVERITY",
    "SecuritySeverity",
    "add_transcript_event",
    "genai_span",
    "get_tracer",
    "ml_metric_attribute",
    "set_ml_metric",
    "set_security_finding",
]
