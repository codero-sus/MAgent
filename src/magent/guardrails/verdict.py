"""Guardrail verdicts and findings.

Every guardrail returns a :class:`Verdict` rather than raising, so a caller can
decide what to do: warn, strip, redact, or block. The agent reads ``blocked``
and ``text``; the CLI prints ``reason``; the audit log stores ``to_dict()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..types import new_id

SEVERITIES: tuple[str, ...] = ("info", "low", "medium", "high", "critical")
"""Ordered severity names; higher index means worse."""

ACTIONS: tuple[str, ...] = ("allow", "warn", "strip", "redact", "block")
"""What a guardrail decided to do about a finding."""


@dataclass(slots=True)
class Finding:
    """One problem detected in a piece of text or a tool call."""

    rule: str
    severity: str = "medium"
    reason: str = ""
    score: float = 0.0
    matched: str = ""
    start: int = -1
    end: int = -1
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize severity into the known ladder."""
        if self.severity not in SEVERITIES:
            self.severity = "medium"

    @property
    def severity_rank(self) -> int:
        """Return the numeric rank of the severity."""
        return SEVERITIES.index(self.severity)

    def to_dict(self, *, include_match: bool = True) -> dict[str, Any]:
        """Serialize the finding for audit and event payloads."""
        data: dict[str, Any] = {
            "rule": self.rule,
            "severity": self.severity,
            "reason": self.reason,
            "score": round(float(self.score), 4),
            "category": self.category,
            "start": self.start,
            "end": self.end,
        }
        if include_match and self.matched:
            data["matched"] = self.matched[:120]
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def __str__(self) -> str:
        """Render the finding as one line."""
        return f"[{self.severity}] {self.rule}: {self.reason}"


@dataclass(slots=True)
class Verdict:
    """The outcome of running one or more guardrails over an input."""

    ok: bool = True
    blocked: bool = False
    action: str = "allow"
    rule: str = ""
    reason: str = ""
    severity: str = "info"
    score: float = 0.0
    text: str = ""
    findings: list[Finding] = field(default_factory=list)
    stage: str = "prompt"
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: new_id("gd"))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """Report whether the guardrail rewrote the text."""
        return bool(self.metadata.get("rewritten", False))

    def severity_rank(self) -> int:
        """Return the numeric rank of the worst finding."""
        return max((finding.severity_rank for finding in self.findings), default=0)

    def rules(self) -> list[str]:
        """Return the distinct rules that fired."""
        return list(dict.fromkeys(finding.rule for finding in self.findings))

    def summary(self, *, limit: int = 3) -> str:
        """Return a short human-readable summary."""
        if self.ok and not self.findings:
            return "ok"
        head = "; ".join(str(finding) for finding in self.findings[:limit])
        if len(self.findings) > limit:
            head += f"; +{len(self.findings) - limit} more"
        return head or self.reason

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize the verdict for the audit log and the HTTP API."""
        data: dict[str, Any] = {
            "id": self.id,
            "ok": self.ok,
            "blocked": self.blocked,
            "action": self.action,
            "rule": self.rule,
            "reason": self.reason,
            "severity": self.severity,
            "score": round(float(self.score), 4),
            "stage": self.stage,
            "ts": self.ts,
            "findings": [finding.to_dict() for finding in self.findings],
            "changed": self.changed,
        }
        if include_text:
            data["text"] = self.text
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def allow(cls, text: str = "", *, stage: str = "prompt") -> Verdict:
        """Return a clean verdict."""
        return cls(ok=True, action="allow", text=text, stage=stage)

    @classmethod
    def block(cls, reason: str, *, rule: str = "", severity: str = "high", stage: str = "prompt",
              findings: list[Finding] | None = None, text: str = "") -> Verdict:
        """Return a verdict that stops the request."""
        return cls(ok=False, blocked=True, action="block", rule=rule, reason=reason,
                   severity=severity, stage=stage, findings=list(findings or []), text=text)

    @classmethod
    def rewrite(cls, text: str, *, action: str = "redact", reason: str = "", rule: str = "",
                findings: list[Finding] | None = None, stage: str = "prompt",
                metadata: dict[str, Any] | None = None) -> Verdict:
        """Return a verdict that modified the text but allowed it through.

        ``rewritten`` is always set, and caller metadata is merged in rather
        than replacing it, so :attr:`changed` stays accurate.
        """
        merged = {"rewritten": True}
        merged.update(metadata or {})
        return cls(ok=True, blocked=False, action=action, rule=rule, reason=reason,
                   severity="medium", text=text, findings=list(findings or []), stage=stage,
                   metadata=merged)

    def merge(self, other: Verdict) -> Verdict:
        """Combine two verdicts, keeping the stricter outcome."""
        merged = Verdict(
            ok=self.ok and other.ok,
            blocked=self.blocked or other.blocked,
            action=_stricter_action(self.action, other.action),
            rule=self.rule or other.rule,
            reason=self.reason or other.reason,
            severity=max((self.severity, other.severity), key=SEVERITIES.index),
            score=max(self.score, other.score),
            text=other.text if other.changed else self.text,
            findings=[*self.findings, *other.findings],
            stage=self.stage,
            metadata={**self.metadata, **other.metadata},
        )
        if merged.blocked:
            merged.action = "block"
            merged.ok = False
        if not merged.reason and merged.findings:
            merged.reason = str(merged.findings[0])
            merged.rule = merged.findings[0].rule
        return merged


def _stricter_action(left: str, right: str) -> str:
    """Return whichever of two actions is more restrictive."""
    return max((left, right), key=ACTIONS.index) if left in ACTIONS and right in ACTIONS else (
        right or left
    )


def worst(findings: list[Finding]) -> Finding | None:
    """Return the highest-severity finding, or ``None``."""
    if not findings:
        return None
    return max(findings, key=lambda item: (item.severity_rank, item.score))


__all__ = ["ACTIONS", "Finding", "SEVERITIES", "Verdict", "worst"]
