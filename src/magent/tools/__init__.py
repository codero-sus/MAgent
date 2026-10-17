"""MAgent's tool layer: schema, registry, sandbox, approvals, and built-ins.

The public surface is intentionally small:

- :func:`default_registry` gives you every built-in tool.
- :func:`tool` turns any function into a validated, documented tool.
- :class:`Sandbox` and :class:`ApprovalManager` are the two enforcement points.
- :class:`ToolContext` is what a tool receives at call time.
"""

from __future__ import annotations

from .approval import (
    MODES,
    RISK_REQUIRING_APPROVAL,
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalRequest,
    ConsoleApprover,
    approve_reads,
    auto_approve,
    build_request,
    deny_all,
    prompter_for_mode,
)
from .base import (
    RISK_LEVELS,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    default_registry,
    parse_docstring,
    schema_from_function,
    tool,
    tool_spec_from_function,
)
from .sandbox import (
    ALWAYS_DENIED_COMMANDS,
    DANGEROUS_PATH_PATTERNS,
    SAFE_DEFAULT_ALLOWLIST,
    Sandbox,
    SandboxDecision,
    SandboxPolicy,
    build_sandbox,
    permissive_sandbox,
    strict_sandbox,
)
from .schema import (
    FORMATS,
    TYPE_CHECKS,
    SchemaValidator,
    ValidationError,
    ValidationResult,
    array_schema,
    boolean_schema,
    enum_schema,
    integer_schema,
    number_schema,
    object_schema,
    string_schema,
)

__all__ = [
    "ALWAYS_DENIED_COMMANDS",
    "DANGEROUS_PATH_PATTERNS",
    "FORMATS",
    "MODES",
    "RISK_LEVELS",
    "RISK_REQUIRING_APPROVAL",
    "SAFE_DEFAULT_ALLOWLIST",
    "TYPE_CHECKS",
    "ApprovalManager",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ApprovalRequest",
    "ConsoleApprover",
    "Sandbox",
    "SandboxDecision",
    "SandboxPolicy",
    "SchemaValidator",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "ValidationError",
    "ValidationResult",
    "approve_reads",
    "array_schema",
    "auto_approve",
    "boolean_schema",
    "build_request",
    "build_sandbox",
    "default_registry",
    "deny_all",
    "enum_schema",
    "integer_schema",
    "number_schema",
    "object_schema",
    "parse_docstring",
    "permissive_sandbox",
    "prompter_for_mode",
    "schema_from_function",
    "strict_sandbox",
    "string_schema",
    "tool",
    "tool_spec_from_function",
]
