"""Safety checks applied before, during, and after model calls.

Four pieces, one façade:

:mod:`magent.guardrails.verdict`
    ``Verdict`` and ``Finding`` - what every check returns.
:mod:`magent.guardrails.injection`
    Heuristic prompt injection detection across nine pattern families.
:mod:`magent.guardrails.pii`
    PII detection with Luhn/IBAN validation and span-based redaction.
:mod:`magent.guardrails.policy`
    Declared rules: length limits, blocked patterns, URL policy, denied tools,
    and a sliding-window rate limiter.
:mod:`magent.guardrails.audit`
    Append-only JSONL audit trail with rotation and secret scrubbing.

:class:`Guardrails` wires them together and exposes the two hooks the agent
loop calls, plus tool-call, tool-result, URL, and budget checks.
"""

from __future__ import annotations

from .audit import DEFAULT_KEEP_ROTATED, DEFAULT_MAX_BYTES, AuditLog
from .injection import (
    BASE64_RUN,
    HOMOGLYPHS,
    PATTERN_FAMILIES,
    ZERO_WIDTH,
    InjectionReport,
    InjectionScanner,
    count_zero_width,
    find_encoded_payloads,
    normalize_for_scan,
    scan_text,
    severity_for,
    try_decode_base64,
)
from .manager import MAX_SCANNED_CHARS, STAGES, Guardrails, build_guardrails
from .pii import (
    DETECTORS,
    PIIRules,
    PIIScanner,
    iban_valid,
    is_sensitive_ip,
    luhn_valid,
    mask_email,
    mask_phone,
    mask_value,
    redact_text,
)
from .policy import RULE_KINDS, PolicyEngine, RateLimiter, Rule, deny_tool_rule, rules_from_config
from .verdict import ACTIONS, SEVERITIES, Finding, Verdict, worst

__all__ = [
    "ACTIONS",
    "AuditLog",
    "BASE64_RUN",
    "DEFAULT_KEEP_ROTATED",
    "DEFAULT_MAX_BYTES",
    "DETECTORS",
    "Finding",
    "Guardrails",
    "HOMOGLYPHS",
    "InjectionReport",
    "InjectionScanner",
    "MAX_SCANNED_CHARS",
    "PATTERN_FAMILIES",
    "PIIRules",
    "PIIScanner",
    "PolicyEngine",
    "RULE_KINDS",
    "RateLimiter",
    "Rule",
    "SEVERITIES",
    "STAGES",
    "Verdict",
    "ZERO_WIDTH",
    "build_guardrails",
    "count_zero_width",
    "deny_tool_rule",
    "find_encoded_payloads",
    "iban_valid",
    "is_sensitive_ip",
    "luhn_valid",
    "mask_email",
    "mask_phone",
    "mask_value",
    "normalize_for_scan",
    "redact_text",
    "rules_from_config",
    "scan_text",
    "severity_for",
    "try_decode_base64",
    "worst",
]
