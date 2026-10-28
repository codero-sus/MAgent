"""MAgent exception hierarchy.

Every error MAgent raises derives from :class:`MAgentError`, so callers can
catch the whole framework with a single ``except`` clause. Each subclass carries
a stable machine readable :attr:`~MAgentError.code` (safe to log, safe to show a
user, safe to branch on) plus optional structured ``details``.

Design rules enforced here:

* No exception message ever contains a secret. :func:`scrub` is applied to
  ``details`` at construction time.
* Errors are retry-aware: :attr:`~MAgentError.retryable` tells the retry policy
  whether another attempt can possibly succeed.
* Errors are cheap: no I/O, no logging, no side effects in ``__init__``.
"""

from __future__ import annotations

import re
from typing import Any

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|authorization|secret|password)\b\s*[:=]\s*\S+"),
    re.compile(r"\b(sk|pk|ghp|gho|github_pat|xox[baprs])-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\"?\s*[:=]\s*\"[^\"]+\""),
)
"""Regexes used to strip credentials out of error payloads."""

REDACTED = "[REDACTED]"
"""Replacement text for anything that looked like a credential."""


def scrub(value: Any) -> Any:
    """Recursively redact anything that looks like a credential.

    Strings are matched against :data:`_SECRET_PATTERNS`; mappings are walked
    and any key whose name suggests a secret has its value replaced wholesale.
    """
    if isinstance(value, str):
        out = value
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(REDACTED, out)
        return out
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key).lower()
            if any(bad in name for bad in ("key", "secret", "token", "password", "auth", "cookie")):
                cleaned[key] = REDACTED
            else:
                cleaned[key] = scrub(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        kind = type(value)
        return kind(scrub(item) for item in value)
    return value


class MAgentError(Exception):
    """Base class for every error raised by MAgent."""

    code = "magent_error"
    """Stable machine readable error identifier."""

    retryable = False
    """Whether an automatic retry could plausibly succeed."""

    exit_code = 1
    """Process exit code the CLI uses when this error escapes."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = scrub(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error into a JSON friendly mapping."""
        return {
            "error": {
                "type": type(self).__name__,
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ConfigError(MAgentError):
    """Raised when configuration is missing, malformed, or contradictory."""

    code = "config_error"
    exit_code = 2


class ConfigFileNotFound(ConfigError):
    """Raised when an explicitly requested configuration file does not exist."""

    code = "config_file_not_found"


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails validation."""

    code = "config_validation_error"


class SecretResolutionError(ConfigError):
    """Raised when a ``env:``/``file:`` secret reference cannot be resolved."""

    code = "secret_resolution_error"


class HTTPError(MAgentError):
    """Base class for transport level failures."""

    code = "http_error"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        url: str | None = None,
        body: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        if status is not None:
            payload["status"] = status
        if url is not None:
            payload["url"] = url
        if body is not None:
            payload["body"] = body[:4096]
        super().__init__(message, details=payload)
        self.status = status
        self.url = url
        self.body = body


class ConnectionError_(HTTPError):
    """Raised when the backend cannot be reached at all."""

    code = "connection_error"
    retryable = True


class TimeoutError_(HTTPError):
    """Raised when a request exceeds its deadline."""

    code = "timeout_error"
    retryable = True


class RateLimitedError(HTTPError):
    """Raised on HTTP 429; carries ``retry_after`` when the server sent one."""

    code = "rate_limited"
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class ServerError(HTTPError):
    """Raised on HTTP 5xx responses from the backend."""

    code = "server_error"
    retryable = True


class ClientError(HTTPError):
    """Raised on HTTP 4xx responses that a retry cannot fix."""

    code = "client_error"
    retryable = False


class AuthenticationError(ClientError):
    """Raised on HTTP 401/403; almost always a missing or wrong API key."""

    code = "authentication_error"

    hint = "Set MAGENT_CORTEX_API_KEY (or CORTEX_API_KEY) to your Cortex bearer key."


class NotFoundError(ClientError):
    """Raised on HTTP 404, including unknown model ids."""

    code = "not_found"


class PayloadTooLargeError(ClientError):
    """Raised when a request or upload exceeds a configured byte limit."""

    code = "payload_too_large"
    retryable = False


class BackendError(MAgentError):
    """Raised when the LLM backend returns a well formed but unusable reply."""

    code = "backend_error"
    retryable = True


class BackendUnavailableError(BackendError):
    """Raised when no configured backend is currently reachable."""

    code = "backend_unavailable"


class ModelNotFoundError(BackendError):
    """Raised when the requested model is not configured on the hoster."""

    code = "model_not_found"
    retryable = False


class ModelNotRunningError(BackendError):
    """Raised when the model exists but its local runtime is stopped."""

    code = "model_not_running"
    retryable = True

    hint = "Start it with: magent cortex start <model-id>"


class CapabilityNotSupportedError(BackendError):
    """Raised when a modality is requested that the model does not declare."""

    code = "capability_not_supported"
    retryable = False


class StreamingError(BackendError):
    """Raised when a streamed response is truncated or malformed."""

    code = "streaming_error"
    retryable = True


class ToolError(MAgentError):
    """Base class for tool execution failures."""

    code = "tool_error"


class ToolNotFoundError(ToolError):
    """Raised when the model asks for a tool that is not registered."""

    code = "tool_not_found"
    retryable = False


class ToolValidationError(ToolError):
    """Raised when tool arguments fail JSON-schema validation."""

    code = "tool_validation_error"
    retryable = False


class ToolExecutionError(ToolError):
    """Raised when a tool body throws."""

    code = "tool_execution_error"
    retryable = True


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds its per-call timeout."""

    code = "tool_timeout"
    retryable = True


class PermissionDeniedError(ToolError):
    """Raised when the sandbox or approval policy blocks an action."""

    code = "permission_denied"
    retryable = False


class ApprovalRequiredError(ToolError):
    """Raised when a dangerous action needs explicit human approval."""

    code = "approval_required"
    retryable = False


class SandboxViolationError(PermissionDeniedError):
    """Raised when a tool tries to escape its workspace jail."""

    code = "sandbox_violation"


class GuardrailError(MAgentError):
    """Base class for guardrail rejections."""

    code = "guardrail_error"
    retryable = False


class PromptInjectionError(GuardrailError):
    """Raised when input trips the prompt-injection detector."""

    code = "prompt_injection_detected"


class PolicyViolationError(GuardrailError):
    """Raised when a policy rule denies an action or content."""

    code = "policy_violation"


class BudgetExceededError(GuardrailError):
    """Raised when a token, cost, step, or wall-clock budget is exhausted."""

    code = "budget_exceeded"


class RateLimitExceededError(GuardrailError):
    """Raised when a local rate limiter refuses a call."""

    code = "local_rate_limit_exceeded"
    retryable = True


class ContentBlockedError(GuardrailError):
    """Raised when output or input is blocked by a content rule."""

    code = "content_blocked"


class MemoryError_(MAgentError):
    """Base class for memory subsystem failures."""

    code = "memory_error"


class StoreError(MemoryError_):
    """Raised when the persistence layer cannot read or write."""

    code = "store_error"


class MigrationError(StoreError):
    """Raised when a stored schema is newer than the code understands."""

    code = "store_migration_error"
    retryable = False


class EmbeddingError(MemoryError_):
    """Raised when embedding generation fails."""

    code = "embedding_error"
    retryable = True


class IndexError_(MemoryError_):
    """Raised when a vector or text index is corrupt or missing."""

    code = "index_error"


class SessionError(MAgentError):
    """Base class for session lifecycle failures."""

    code = "session_error"


class SessionNotFoundError(SessionError):
    """Raised when a referenced session id does not exist."""

    code = "session_not_found"
    retryable = False


class SessionCorruptError(SessionError):
    """Raised when a session transcript cannot be parsed."""

    code = "session_corrupt"
    retryable = False


class AgentError(MAgentError):
    """Base class for agent loop failures."""

    code = "agent_error"


class MaxIterationsError(AgentError):
    """Raised when the ReAct loop hits its iteration ceiling."""

    code = "max_iterations"
    retryable = False


class PlanError(AgentError):
    """Raised when a generated plan cannot be parsed or is invalid."""

    code = "plan_error"
    retryable = True


class StructuredOutputError(AgentError):
    """Raised when the model fails to produce schema-conforming JSON."""

    code = "structured_output_error"
    retryable = True


class WorkflowError(AgentError):
    """Base class for workflow/graph execution failures."""

    code = "workflow_error"


class WorkflowCycleError(WorkflowError):
    """Raised when a workflow graph contains an unbreakable cycle."""

    code = "workflow_cycle"
    retryable = False


class WorkflowStepError(WorkflowError):
    """Raised when a single workflow step fails."""

    code = "workflow_step_failed"


class PluginError(MAgentError):
    """Base class for plugin loading failures."""

    code = "plugin_error"


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not installed."""

    code = "plugin_not_found"
    retryable = False


class PluginManifestError(PluginError):
    """Raised when a plugin manifest is invalid."""

    code = "plugin_manifest_invalid"
    retryable = False


class PluginSecurityError(PluginError):
    """Raised when a plugin requests permissions outside its allowlist."""

    code = "plugin_security_error"
    retryable = False


class SkillError(MAgentError):
    """Base class for skill loading and selection failures."""

    code = "skill_error"


class SkillNotFoundError(SkillError):
    """Raised when a requested skill is not in the library."""

    code = "skill_not_found"
    retryable = False


class SkillFormatError(SkillError):
    """Raised when a skill file cannot be parsed."""

    code = "skill_format_invalid"
    retryable = False


class EvalError(MAgentError):
    """Base class for evaluation failures."""

    code = "eval_error"


class DatasetError(EvalError):
    """Raised when an evaluation dataset cannot be loaded."""

    code = "eval_dataset_invalid"
    retryable = False


class MetricError(EvalError):
    """Raised when a metric is unknown or misconfigured."""

    code = "eval_metric_invalid"
    retryable = False


class EvalThresholdError(EvalError):
    """Raised when an evaluation run scores below the required threshold."""

    code = "eval_threshold_not_met"
    retryable = False


class ServerError_(MAgentError):
    """Base class for the embedded MAgent HTTP server."""

    code = "server_runtime_error"


class CancelledError(MAgentError):
    """Raised when an operation is cancelled by the user or a deadline."""

    code = "cancelled"
    retryable = False


class UserAbortError(CancelledError):
    """Raised when the operator interrupts MAgent (Ctrl-C / deny / stop)."""

    code = "user_abort"


ERROR_REGISTRY: dict[str, type[MAgentError]] = {
    cls.code: cls
    for cls in (
        MAgentError,
        ConfigError,
        ConfigFileNotFound,
        ConfigValidationError,
        SecretResolutionError,
        HTTPError,
        ConnectionError_,
        TimeoutError_,
        RateLimitedError,
        ServerError,
        ClientError,
        AuthenticationError,
        NotFoundError,
        PayloadTooLargeError,
        BackendError,
        BackendUnavailableError,
        ModelNotFoundError,
        ModelNotRunningError,
        CapabilityNotSupportedError,
        StreamingError,
        ToolError,
        ToolNotFoundError,
        ToolValidationError,
        ToolExecutionError,
        ToolTimeoutError,
        PermissionDeniedError,
        ApprovalRequiredError,
        SandboxViolationError,
        GuardrailError,
        PromptInjectionError,
        PolicyViolationError,
        BudgetExceededError,
        RateLimitExceededError,
        ContentBlockedError,
        MemoryError_,
        StoreError,
        MigrationError,
        EmbeddingError,
        IndexError_,
        SessionError,
        SessionNotFoundError,
        SessionCorruptError,
        AgentError,
        MaxIterationsError,
        PlanError,
        StructuredOutputError,
        WorkflowError,
        WorkflowCycleError,
        WorkflowStepError,
        PluginError,
        PluginNotFoundError,
        PluginManifestError,
        PluginSecurityError,
        SkillError,
        SkillNotFoundError,
        SkillFormatError,
        EvalError,
        DatasetError,
        MetricError,
        EvalThresholdError,
        ServerError_,
        CancelledError,
        UserAbortError,
    )
}
"""Lookup table from stable ``code`` string to exception class."""


def error_from_code(code: str, message: str, **details: Any) -> MAgentError:
    """Reconstruct an exception instance from a persisted error code."""
    cls = ERROR_REGISTRY.get(code, MAgentError)
    return cls(message, details=details)


def classify_status(status: int, message: str, **kwargs: Any) -> HTTPError:
    """Map an HTTP status code onto the most specific MAgent error class."""
    if status == 401 or status == 403:
        return AuthenticationError(message, status=status, **kwargs)
    if status == 404:
        return NotFoundError(message, status=status, **kwargs)
    if status == 413:
        return PayloadTooLargeError(message, status=status, **kwargs)
    if status == 429:
        return RateLimitedError(message, status=status, **kwargs)
    if 400 <= status < 500:
        return ClientError(message, status=status, **kwargs)
    if status >= 500:
        return ServerError(message, status=status, **kwargs)
    return HTTPError(message, status=status, **kwargs)


def is_retryable(exc: BaseException) -> bool:
    """Decide whether ``exc`` should be retried by the retry policy."""
    if isinstance(exc, MAgentError):
        return exc.retryable
    return isinstance(exc, (OSError, TimeoutError))


__all__ = [
    "ERROR_REGISTRY",
    "REDACTED",
    "AgentError",
    "ApprovalRequiredError",
    "AuthenticationError",
    "BackendError",
    "BackendUnavailableError",
    "BudgetExceededError",
    "CancelledError",
    "CapabilityNotSupportedError",
    "ClientError",
    "ConfigError",
    "ConfigFileNotFound",
    "ConfigValidationError",
    "ConnectionError_",
    "ContentBlockedError",
    "DatasetError",
    "EmbeddingError",
    "EvalError",
    "EvalThresholdError",
    "GuardrailError",
    "HTTPError",
    "IndexError_",
    "MAgentError",
    "MaxIterationsError",
    "MemoryError_",
    "MetricError",
    "MigrationError",
    "ModelNotFoundError",
    "ModelNotRunningError",
    "NotFoundError",
    "PayloadTooLargeError",
    "PermissionDeniedError",
    "PlanError",
    "PluginError",
    "PluginManifestError",
    "PluginNotFoundError",
    "PluginSecurityError",
    "PolicyViolationError",
    "PromptInjectionError",
    "RateLimitExceededError",
    "RateLimitedError",
    "SandboxViolationError",
    "SecretResolutionError",
    "ServerError",
    "ServerError_",
    "SessionCorruptError",
    "SessionError",
    "SessionNotFoundError",
    "SkillError",
    "SkillFormatError",
    "SkillNotFoundError",
    "StoreError",
    "StreamingError",
    "StructuredOutputError",
    "TimeoutError_",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "ToolValidationError",
    "UserAbortError",
    "WorkflowCycleError",
    "WorkflowError",
    "WorkflowStepError",
    "classify_status",
    "error_from_code",
    "is_retryable",
    "scrub",
]
