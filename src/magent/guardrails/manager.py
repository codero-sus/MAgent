"""The guardrails façade the agent calls.

:class:`Guardrails` owns the three detectors (policy rules, prompt injection,
PII) plus the audit log, and exposes the two hooks the agent loop uses:

``inspect_prompt(text) -> Verdict``
    Run before a request is sent. May rewrite the text (redaction, stripping) or
    block the run.
``inspect_output(text) -> Verdict``
    Run on model output before it is stored or shown.

It also guards the third boundary that matters for a tool-using agent:
``inspect_tool_result(text)`` scans what a tool *returned*, because a file, a web
page, or a command's stdout can carry an injection just as easily as a user's
message can. Everything is audited.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from ..config import GuardrailsConfig
from ..errors import GuardrailError
from ..logging_ import get_logger
from ..types import EventKind
from .audit import AuditLog
from .injection import InjectionScanner
from .pii import PIIRules, PIIScanner
from .policy import PolicyEngine, Rule
from .verdict import Finding, Verdict, worst

LOGGER = get_logger("guardrails")
"""Logger for guardrail activity."""

MAX_SCANNED_CHARS = 200_000
"""Ceiling on how much text one scan will examine."""

STAGES: tuple[str, ...] = ("prompt", "output", "tool", "tool_result", "url", "budget")
"""Stages a guardrail can be applied at."""


def _redact_credentials(text: str, findings: Sequence[Finding]) -> tuple[str, int]:
    """Replace credential-shaped spans with a marker.

    Uses the scanner's own offsets, applied back-to-front so earlier spans keep
    their positions, and only touches findings the scanner classified as a
    credential probe.
    """
    targets = [
        item
        for item in findings
        if item.category == "credential_probe" and item.start >= 0 and item.end > item.start
    ]
    if not targets:
        return text, 0
    out = text
    for item in sorted(targets, key=lambda entry: entry.start, reverse=True):
        label = item.metadata.get("kind") or "CREDENTIAL"
        out = out[: item.start] + f"[REDACTED:{str(label).upper()}]" + out[item.end :]
    return out, len(targets)


class Guardrails:
    """Policy, injection, and PII checks with a shared audit trail."""

    def __init__(
        self,
        *,
        config: GuardrailsConfig | Any = None,
        injection: InjectionScanner | None = None,
        pii: PIIScanner | None = None,
        policy: PolicyEngine | None = None,
        audit: AuditLog | None = None,
        events: Any = None,
        enabled: bool = True,
        scan_tool_results: bool = True,
    ) -> None:
        """Assemble the guardrails from their parts.

        Args:
            config: a :class:`~magent.config.GuardrailsConfig`
            injection: injection scanner override
            pii: PII scanner override
            policy: policy engine override
            audit: audit log override
            events: optional event bus for ``guard.blocked`` notifications
            enabled: master switch
            scan_tool_results: also scan what tools return
        """
        self.config = config if config is not None else GuardrailsConfig()
        self.enabled = bool(enabled and getattr(self.config, "enabled", True))
        self.scan_tool_results = bool(
            scan_tool_results and getattr(self.config, "injection_detection", True)
        )
        self.events = events
        self.policy = policy if policy is not None else PolicyEngine(config=self.config)
        self.injection = (
            injection
            if injection is not None
            else InjectionScanner(
                threshold=float(getattr(self.config, "injection_threshold", 0.55)),
                action=str(getattr(self.config, "injection_action", "warn"))
                if getattr(self.config, "injection_detection", True)
                else "off",
            )
        )
        self.pii = (
            pii
            if pii is not None
            else PIIScanner(
                PIIRules(
                    enabled=bool(getattr(self.config, "pii_redaction", True)),
                    action=str(getattr(self.config, "pii_action", "redact"))
                    if getattr(self.config, "pii_redaction", True)
                    else "off",
                )
            )
        )
        self.audit = (
            audit
            if audit is not None
            else AuditLog(
                str(getattr(self.config, "audit_log", "") or "") or None,
                enabled=bool(getattr(self.config, "audit_enabled", True)),
                redact=True,
            )
        )
        self._lock = threading.RLock()
        self.counts: dict[str, int] = dict.fromkeys(STAGES, 0)
        self.blocked = 0
        self.rewritten = 0
        self.warned = 0

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Any, *, events: Any = None, data_dir: Any = None, audit_path: str = ""
    ) -> Guardrails:
        """Build guardrails from a full MAgent configuration.

        Args:
            config: a :class:`~magent.config.Config` or a ``GuardrailsConfig``
            events: optional event bus
            data_dir: directory used for the default audit log location
            audit_path: explicit audit log path, overriding both
        """
        guardrail_config = getattr(config, "guardrails", config)
        path = audit_path or str(getattr(guardrail_config, "audit_log", "") or "")
        if not path and data_dir:
            from pathlib import Path

            path = str(Path(data_dir).expanduser() / "audit" / "guardrails.jsonl")
        audit = AuditLog(
            path or None, enabled=bool(getattr(guardrail_config, "audit_enabled", True))
        )
        guardrails = cls(config=guardrail_config, audit=audit, events=events)
        tools = getattr(config, "tools", None)
        if tools is not None and getattr(tools, "deny_commands", None):
            guardrails.policy.add_rule(
                Rule(
                    name="deny_configured_commands",
                    kind="tool",
                    action="block",
                    severity="high",
                    pattern="run_command *",
                    reason="run_command is denied by [tools] deny_commands",
                )
            )
        return guardrails

    # -- the two agent hooks -------------------------------------------------

    def inspect_prompt(
        self, text: str, *, session_id: str = "", run_id: str = "", agent: str = "main"
    ) -> Verdict:
        """Check and possibly rewrite a prompt before it is sent."""
        return self.inspect(text, stage="prompt", session_id=session_id, run_id=run_id, agent=agent)

    def inspect_output(
        self, text: str, *, session_id: str = "", run_id: str = "", agent: str = "main"
    ) -> Verdict:
        """Check and possibly rewrite model output before it is stored or shown."""
        return self.inspect(text, stage="output", session_id=session_id, run_id=run_id, agent=agent)

    def inspect_tool_result(
        self, text: str, *, tool: str = "", run_id: str = "", agent: str = "main"
    ) -> Verdict:
        """Scan data returned by a tool for embedded instructions.

        Tool output is untrusted by definition: a README on disk or a page on
        the web can tell the agent to do something. Only the injection scanner
        runs here, because redacting PII out of a file the user asked to read
        would corrupt legitimate work.
        """
        if not self.enabled or not self.scan_tool_results:
            return Verdict.allow(text or "", stage="tool_result")
        self._count("tool_result")
        report = self.injection.scan((text or "")[:MAX_SCANNED_CHARS])
        threshold = float(getattr(self.config, "injection_threshold", 0.55))
        action = str(getattr(self.config, "injection_action", "warn"))
        if report.score < threshold or not report.findings:
            return Verdict.allow(text or "", stage="tool_result")
        top = worst(report.findings)
        rule = top.rule if top else "injection"
        reason = (
            f"tool {tool or 'result'} returned text that looks like an injected "
            f"instruction (score {report.score:.2f})"
        )
        if action == "warn":
            # A credential the scanner recognised is redacted even in warn mode:
            # handing a live provider key back to the model is the leak this
            # check exists to prevent, and the surrounding data stays intact.
            redacted, removed = _redact_credentials(text or "", report.findings)
            if removed:
                reason += f" ({removed} credential(s) redacted)"
            verdict = Verdict.rewrite(
                redacted,
                action="warn",
                reason=reason,
                rule=rule,
                findings=report.findings,
                stage="tool_result",
                metadata={"redacted": removed},
            )
            verdict.severity = report.worst_severity
            verdict.score = report.score
        elif action == "strip":
            verdict = Verdict.rewrite(
                self.injection.sanitize(text or "", report=report),
                action="strip",
                reason=reason,
                rule=rule,
                findings=report.findings,
                stage="tool_result",
                metadata={"injection": report.to_dict()},
            )
            verdict.severity = report.worst_severity
            verdict.score = report.score
        else:
            verdict = Verdict.block(
                reason,
                rule=rule,
                severity=report.worst_severity,
                stage="tool_result",
                findings=report.findings,
                text=text or "",
            )
        return self._audit(verdict, run_id=run_id, agent=agent, tool=tool)

    def inspect_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        run_id: str = "",
        agent: str = "main",
    ) -> Verdict:
        """Check a tool call against the policy engine before dispatch."""
        if not self.enabled:
            return Verdict.allow(name, stage="tool")
        self._count("tool")
        verdict = self.policy.check_tool_call(name, arguments or {})
        return self._audit(verdict, run_id=run_id, agent=agent, tool=name)

    def inspect_url(self, url: str, *, run_id: str = "", agent: str = "main") -> Verdict:
        """Check an outbound URL against the policy engine."""
        if not self.enabled:
            return Verdict.allow(url, stage="url")
        self._count("url")
        verdict = self.policy.check_url(url)
        return self._audit(verdict, run_id=run_id, agent=agent)

    def check_budget(self, tracker: Any, *, run_id: str = "", agent: str = "main") -> Verdict:
        """Turn a usage tracker's violations into a verdict."""
        self._count("budget")
        exceeded = tracker.exceeded() if hasattr(tracker, "exceeded") else []
        if not exceeded:
            return Verdict.allow("", stage="budget")
        reason = "; ".join(item.message for item in exceeded)
        verdict = Verdict.block(
            reason,
            rule="budget:exceeded",
            severity="high",
            stage="budget",
            metadata={"violations": [item.to_dict() for item in exceeded]},
        )
        return self._audit(verdict, run_id=run_id, agent=agent)

    # -- the shared pipeline -------------------------------------------------

    def inspect(
        self,
        text: str,
        *,
        stage: str = "prompt",
        session_id: str = "",
        run_id: str = "",
        agent: str = "main",
        rate_key: str = "",
    ) -> Verdict:
        """Run every applicable guardrail over one string and merge the results.

        Order matters: length and pattern rules first (cheap), then injection,
        then PII rewriting. Each stage sees the text the previous stage produced,
        so redaction cannot hide an injection from the scanner and stripping
        cannot leave PII behind.
        """
        source = text or ""
        if not self.enabled:
            return Verdict.allow(source, stage=stage)
        self._count(stage)
        working = source[:MAX_SCANNED_CHARS]
        merged = self.policy.check_text(working, stage=stage)
        merged.stage = stage
        if merged.blocked:
            return self._audit(merged, run_id=run_id, session_id=session_id, agent=agent)

        injection = self.injection.verdict(working, stage=stage)
        merged = merged.merge(injection)
        if merged.blocked:
            return self._audit(merged, run_id=run_id, session_id=session_id, agent=agent)
        if injection.changed:
            working = injection.text

        if self.pii.rules.enabled and self.pii.rules.action != "off":
            pii = self.pii.verdict(working, stage=stage)
            merged = merged.merge(pii)
            if merged.blocked:
                return self._audit(merged, run_id=run_id, session_id=session_id, agent=agent)
            if pii.changed:
                working = pii.text

        if rate_key:
            rate = self.policy.check_rate(rate_key)
            merged = merged.merge(rate)
            if merged.blocked:
                return self._audit(merged, run_id=run_id, session_id=session_id, agent=agent)

        merged.text = working
        merged.ok = True
        merged.blocked = False
        if merged.action == "block":
            merged.action = "warn"
        elif merged.changed and merged.action == "allow":
            merged.action = "redact"
        return self._audit(merged, run_id=run_id, session_id=session_id, agent=agent)

    def enforce(self, text: str, *, stage: str = "prompt", **kwargs: Any) -> str:
        """Inspect and return safe text, raising :class:`GuardrailError` on a block.

        This is the convenience form for callers that only want the text.
        """
        verdict = self.inspect(text, stage=stage, **kwargs)
        if verdict.blocked:
            raise GuardrailError(
                verdict.reason or f"blocked by guardrail at {stage}",
                details={"stage": stage, "rules": verdict.rules()},
            )
        return verdict.text if verdict.changed else text

    # -- bookkeeping ---------------------------------------------------------

    def _audit(
        self,
        verdict: Verdict,
        *,
        run_id: str = "",
        session_id: str = "",
        agent: str = "main",
        tool: str = "",
    ) -> Verdict:
        """Record a verdict, emit an event when blocked, and update counters."""
        with self._lock:
            if verdict.blocked:
                self.blocked += 1
            elif verdict.changed:
                self.rewritten += 1
            elif verdict.findings:
                self.warned += 1
        severity = verdict.severity if verdict.findings or verdict.blocked else "info"
        self.audit.record(
            f"guardrail.{verdict.stage}",
            {
                "action": verdict.action,
                "blocked": verdict.blocked,
                "reason": verdict.reason,
                "rule": verdict.rule,
                "score": verdict.score,
                "findings": [finding.to_dict() for finding in verdict.findings[:20]],
                "session_id": session_id,
                "tool": tool,
                "changed": verdict.changed,
                "chars": len(verdict.text or ""),
            },
            agent=agent,
            run_id=run_id,
            severity=severity,
        )
        if verdict.blocked and self.events is not None:
            try:
                self.events.emit(
                    EventKind.GUARDRAIL_BLOCKED,
                    run_id=run_id,
                    agent=agent,
                    stage=verdict.stage,
                    reason=verdict.reason,
                    rule=verdict.rule,
                    findings=[finding.to_dict() for finding in verdict.findings[:5]],
                )
            except Exception:
                LOGGER.debug("could not emit guard.blocked")
        return verdict

    def _count(self, stage: str) -> None:
        """Increment the per-stage counter."""
        with self._lock:
            self.counts[stage] = self.counts.get(stage, 0) + 1

    def enable(self) -> None:
        """Turn the guardrails on."""
        self.enabled = True

    def disable(self, *, reason: str = "") -> None:
        """Turn the guardrails off, recording why in the audit log."""
        self.enabled = False
        self.audit.record("guardrail.disabled", {"reason": reason}, severity="high")
        LOGGER.warning("guardrails disabled%s", f": {reason}" if reason else "")

    def stats(self) -> dict[str, Any]:
        """Return telemetry from every component."""
        return {
            "enabled": self.enabled,
            "counts": dict(self.counts),
            "blocked": self.blocked,
            "rewritten": self.rewritten,
            "warned": self.warned,
            "injection": self.injection.stats(),
            "pii": self.pii.stats(),
            "policy": self.policy.stats(),
            "audit": self.audit.stats(),
            "scan_tool_results": self.scan_tool_results,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration and rules for ``magent guardrails show``."""
        return {
            "enabled": self.enabled,
            "injection": {
                "threshold": self.injection.threshold,
                "action": self.injection.action,
                "families": sorted(self.injection.families),
            },
            "pii": {"action": self.pii.rules.action, "kinds": self.pii.stats()["enabled_kinds"]},
            "policy": self.policy.to_dict(),
            "audit": {
                "path": str(self.audit.path) if self.audit.path else None,
                "enabled": self.audit.enabled,
            },
            "stats": self.stats(),
        }

    def describe(self) -> str:
        """Return a one-line human description of the active configuration."""
        return (
            f"guardrails={'on' if self.enabled else 'off'} "
            f"injection={self.injection.action}@{self.injection.threshold:.2f} "
            f"pii={self.pii.rules.action} rules={len(self.policy.rules())} "
            f"audit={'yes' if self.audit.enabled else 'no'}"
        )

    def close(self) -> None:
        """Flush and release audit resources."""
        self.audit.record("guardrail.closed", {}, severity="info")

    def __repr__(self) -> str:
        """Show the enable flag and detector actions."""
        return (
            f"<Guardrails enabled={self.enabled} injection={self.injection.action!r} "
            f"pii={self.pii.rules.action!r} rules={len(self.policy.rules())}>"
        )


def build_guardrails(
    config: Any = None, *, events: Any = None, audit_path: str = "", data_dir: Any = None
) -> Guardrails:
    """Build guardrails from configuration, or a permissive default set."""
    if config is None:
        return Guardrails(
            config=GuardrailsConfig(),
            events=events,
            audit=AuditLog(audit_path or None, enabled=bool(audit_path)),
        )
    return Guardrails.from_config(config, events=events, data_dir=data_dir, audit_path=audit_path)


__all__ = [
    "MAX_SCANNED_CHARS",
    "STAGES",
    "Guardrails",
    "build_guardrails",
]
