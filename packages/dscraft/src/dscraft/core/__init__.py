"""dscraft.core: the thin, shared substrate underneath every dscraft subpackage.

See the package README for scope and the "why thin" rationale. This module
re-exports the small public surface of ``dscraft.core.adapter``,
``dscraft.core.data``, ``dscraft.core.telemetry``, ``dscraft.core.licensing``,
and ``dscraft.core.sandbox`` so most consumers only need ``import dscraft.core``.
"""

from dscraft.core.adapter import BaseSandboxedAdapter
from dscraft.core.data import (
    ArrowBackedFrame,
    DenseMediaPipeline,
    SparseFormat,
    SparseGraphTensorAdapter,
    from_polars_zero_copy,
    is_arrow_backed_pandas,
    pandas_arrow_dtypes,
    to_polars_zero_copy,
)
from dscraft.core.licensing import (
    Allowlist,
    Mitigation,
    ModelLicenseEntry,
    ModelTier,
    RestrictedLicenseNotAcceptedError,
    RISK_MITIGATIONS,
    RiskType,
)
from dscraft.core.sandbox import (
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
from dscraft.core.telemetry import (
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

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # adapter.py
    "BaseSandboxedAdapter",
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
