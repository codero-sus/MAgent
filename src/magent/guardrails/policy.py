"""Policy rules: length limits, pattern blocks, URL rules, and rate limiting.

The injection and PII scanners answer "is this text suspicious?". The policy
engine answers "is this action permitted at all?", using rules an operator can
read: blocked patterns, a maximum prompt length, an optional URL allowlist, a
denied-tool list, and a requests-per-minute ceiling.

Rules are data, so they can come from the config file, from a plugin, or from
the CLI, and they are all reported the same way through a :class:`Verdict`.
"""

from __future__ import annotations

import fnmatch
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

from ..utils.urls import is_local_url, is_private_host, redact_url
from .verdict import Finding, Verdict

RULE_KINDS: tuple[str, ...] = ("block_pattern", "max_chars", "url", "tool", "rate", "require_pattern")
"""Kinds of policy rule the engine understands."""


@dataclass(slots=True)
class Rule:
    """One declarative policy rule."""

    name: str
    kind: str = "block_pattern"
    pattern: str = ""
    action: str = "block"
    severity: str = "high"
    reason: str = ""
    limit: float = 0.0
    applies_to: str = "both"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    compiled: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the rule shape and compile a regex when needed."""
        if self.kind not in RULE_KINDS:
            raise ValueError(f"Unknown policy rule kind {self.kind!r}")
        if self.action not in {"block", "warn", "allow"}:
            raise ValueError(f"Policy rule action must be block, warn, or allow, not {self.action!r}")
        if self.pattern and self.kind in {"block_pattern", "require_pattern"}:
            self.compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> re.Match[str] | None:
        """Return the first match of this rule's pattern in ``text``."""
        if self.compiled is None:
            return None
        return self.compiled.search(text or "")

    def applies(self, stage: str) -> bool:
        """Report whether the rule covers ``prompt``, ``output``, or both."""
        return self.applies_to in {"both", stage}

    def finding(self, text: str, match: re.Match[str] | None = None) -> Finding:
        """Build the finding this rule produces."""
        return Finding(
            rule=f"policy:{self.name}",
            severity=self.severity,
            reason=self.reason or f"matched blocked pattern {self.pattern!r}",
            score=0.9 if self.action == "block" else 0.4,
            matched=(match.group(0) if match else "")[:120],
            start=match.start() if match else -1,
            end=match.end() if match else -1,
            category="policy",
            metadata={"kind": self.kind, "action": self.action},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rule."""
        return {
            "name": self.name, "kind": self.kind, "pattern": self.pattern,
            "action": self.action, "severity": self.severity, "reason": self.reason,
            "limit": self.limit, "applies_to": self.applies_to, "enabled": self.enabled,
        }


class RateLimiter:
    """A sliding-window rate limiter keyed by caller."""

    def __init__(self, *, per_minute: int = 0, burst: int = 0) -> None:
        """Configure the ceiling.

        Args:
            per_minute: maximum calls per rolling minute, 0 to disable
            burst: extra calls allowed above the ceiling before blocking
        """
        self.per_minute = max(0, int(per_minute))
        self.burst = max(0, int(burst))
        self.window = 60.0
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.RLock()
        self.allowed = 0
        self.throttled = 0

    def allow(self, key: str = "default", *, now: float | None = None) -> bool:
        """Report whether a call from ``key`` may proceed right now."""
        if self.per_minute <= 0:
            self.allowed += 1
            return True
        moment = now if now is not None else time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and moment - events[0] > self.window:
                events.popleft()
            if len(events) >= self.per_minute + self.burst:
                self.throttled += 1
                return False
            events.append(moment)
            self.allowed += 1
            return True

    def retry_after(self, key: str = "default", *, now: float | None = None) -> float:
        """Return how long ``key`` must wait for a slot, in seconds."""
        if self.per_minute <= 0:
            return 0.0
        moment = now if now is not None else time.monotonic()
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 0.0
            oldest = events[0]
            return max(0.0, round(self.window - (moment - oldest), 3))

    def usage(self, key: str = "default", *, now: float | None = None) -> int:
        """Return how many calls ``key`` made inside the current window."""
        moment = now if now is not None else time.monotonic()
        with self._lock:
            events = self._events.get(key)
            if not events:
                return 0
            return sum(1 for stamp in events if moment - stamp <= self.window)

    def reset(self, key: str | None = None) -> None:
        """Forget recorded calls for one key, or all of them."""
        with self._lock:
            if key is None:
                self._events.clear()
            else:
                self._events.pop(key, None)

    def stats(self) -> dict[str, Any]:
        """Return limiter telemetry."""
        with self._lock:
            keys = len(self._events)
        return {"per_minute": self.per_minute, "burst": self.burst, "keys": keys,
                "allowed": self.allowed, "throttled": self.throttled}


class PolicyEngine:
    """Evaluate declared rules against text, tools, and URLs."""

    def __init__(self, rules: Iterable[Rule] = (), *, config: Any = None,
                 rate_limiter: RateLimiter | None = None) -> None:
        """Build an engine from explicit rules and/or a configuration object.

        Args:
            rules: rules to register immediately
            config: a :class:`~magent.config.GuardrailsConfig` for the built-ins
            rate_limiter: limiter override, built from config when omitted
        """
        self.config = config
        self._rules: dict[str, Rule] = {}
        self._lock = threading.RLock()
        self.checks = 0
        self.blocks = 0
        self.warnings = 0
        if config is not None:
            for rule in rules_from_config(config):
                self.add_rule(rule)
        for rule in rules:
            self.add_rule(rule)
        self.rate_limiter = rate_limiter or RateLimiter(
            per_minute=int(getattr(config, "rate_limit_per_minute", 0) or 0)
        )
        self.url_allowlist_only = bool(getattr(config, "url_allowlist_only", False))
        self.allowed_domains = [
            str(item).lower() for item in (getattr(config, "allow_domains", []) or [])
        ]
        self.denied_domains = [str(item).lower() for item in (getattr(config, "deny_domains", []) or [])]

    # -- rule management -----------------------------------------------------

    def add_rule(self, rule: Rule, *, replace: bool = True) -> Rule:
        """Register one rule."""
        with self._lock:
            if rule.name in self._rules and not replace:
                raise KeyError(f"Policy rule {rule.name!r} already exists")
            self._rules[rule.name] = rule
        return rule

    def remove_rule(self, name: str) -> bool:
        """Drop one rule; report whether it existed."""
        with self._lock:
            return self._rules.pop(name, None) is not None

    def rules(self, *, kind: str = "", enabled_only: bool = True) -> list[Rule]:
        """Return the registered rules."""
        with self._lock:
            items = list(self._rules.values())
        return [
            rule for rule in items
            if (not kind or rule.kind == kind) and (rule.enabled or not enabled_only)
        ]

    def get(self, name: str) -> Rule | None:
        """Return one rule by name."""
        with self._lock:
            return self._rules.get(name)

    # -- checks --------------------------------------------------------------

    def check_text(self, text: str, *, stage: str = "prompt") -> Verdict:
        """Apply pattern and length rules to one string."""
        self.checks += 1
        source = text or ""
        findings: list[Finding] = []
        action = "allow"
        for rule in self.rules():
            if not rule.enabled or not rule.applies(stage):
                continue
            if rule.kind == "max_chars" and rule.limit > 0 and len(source) > rule.limit:
                findings.append(Finding(
                    rule=f"policy:{rule.name}", severity=rule.severity, category="policy",
                    reason=rule.reason or f"text is {len(source)} chars, above the {int(rule.limit)} limit",
                    score=0.7, matched="",
                    metadata={"kind": "max_chars", "chars": len(source), "action": rule.action},
                ))
            elif rule.kind == "block_pattern":
                match = rule.matches(source)
                if match:
                    finding = rule.finding(source, match)
                    finding.metadata["action"] = rule.action
                    findings.append(finding)
            elif rule.kind == "require_pattern":
                if not rule.matches(source):
                    findings.append(Finding(
                        rule=f"policy:{rule.name}", severity=rule.severity, category="policy",
                        reason=rule.reason or f"text is missing the required pattern {rule.pattern!r}",
                        score=0.6, metadata={"kind": "require_pattern", "action": rule.action},
                    ))
        blocked = any(finding.metadata.get("action") == "block" for finding in findings)
        if findings and not blocked:
            action = "warn"
            self.warnings += 1
        if blocked:
            action = "block"
            self.blocks += 1
            top = max(findings, key=lambda item: item.severity_rank)
            return Verdict.block(top.reason, rule=top.rule, severity=top.severity, stage=stage,
                                 findings=findings, text=source)
        verdict = Verdict.allow(source, stage=stage)
        verdict.findings = findings
        verdict.action = action
        if findings:
            verdict.reason = findings[0].reason
            verdict.rule = findings[0].rule
            verdict.severity = max(findings, key=lambda item: item.severity_rank).severity
        return verdict

    def check_tool_call(self, name: str, arguments: dict[str, Any] | None = None) -> Verdict:
        """Apply tool rules to a call about to be dispatched."""
        self.checks += 1
        findings: list[Finding] = []
        rendered = f"{name} {arguments or {}}"
        for rule in self.rules(kind="tool"):
            if not rule.enabled:
                continue
            if fnmatch.fnmatch(name, rule.pattern) or fnmatch.fnmatch(rendered, rule.pattern):
                findings.append(Finding(
                    rule=f"policy:{rule.name}", severity=rule.severity, category="tool",
                    reason=rule.reason or f"tool call matches denied pattern {rule.pattern!r}",
                    score=0.9, matched=redact_url(name),
                    metadata={"kind": "tool", "action": rule.action, "tool": name},
                ))
        blocked = any(finding.metadata.get("action") == "block" for finding in findings)
        if blocked:
            self.blocks += 1
            return Verdict.block(findings[0].reason, rule=findings[0].rule,
                                 severity=findings[0].severity, stage="tool", findings=findings)
        verdict = Verdict.allow(name, stage="tool")
        verdict.findings = findings
        if findings:
            self.warnings += 1
            verdict.action = "warn"
            verdict.reason = findings[0].reason
        return verdict

    def check_url(self, url: str) -> Verdict:
        """Apply the URL policy to an outbound request."""
        self.checks += 1
        findings: list[Finding] = []
        parts = urlsplit(url or "")
        host = (parts.hostname or "").lower()
        if self.url_allowlist_only and not self._domain_allowed(host):
            findings.append(Finding(
                rule="policy:url_allowlist", severity="high", category="url",
                reason=f"host {host or '(none)'} is not on the allowlist",
                score=0.9, matched=redact_url(url), metadata={"host": host},
            ))
        if host and any(fnmatch.fnmatch(host, pattern) for pattern in self.denied_domains):
            findings.append(Finding(
                rule="policy:url_denylist", severity="high", category="url",
                reason=f"host {host} is explicitly denied", score=0.95,
                matched=redact_url(url), metadata={"host": host},
            ))
        if parts.scheme and parts.scheme not in {"http", "https"}:
            findings.append(Finding(
                rule="policy:url_scheme", severity="medium", category="url",
                reason=f"unsupported URL scheme {parts.scheme!r}", score=0.6,
                matched=redact_url(url), metadata={"scheme": parts.scheme},
            ))
        if findings:
            self.blocks += 1
            return Verdict.block(findings[0].reason, rule=findings[0].rule,
                                 severity=findings[0].severity, stage="url", findings=findings,
                                 text=redact_url(url))
        return Verdict.allow(redact_url(url), stage="url")

    def _domain_allowed(self, host: str) -> bool:
        """Report whether a host satisfies the allowlist policy."""
        if not host:
            return False
        if not self.allowed_domains:
            return is_local_url(f"http://{host}") or is_private_host(host)
        return any(host == item or host.endswith("." + item.lstrip(".")) or
                   fnmatch.fnmatch(host, item) for item in self.allowed_domains)

    def check_rate(self, key: str = "default") -> Verdict:
        """Apply the rate limit to one caller."""
        self.checks += 1
        if self.rate_limiter.allow(key):
            return Verdict.allow(key, stage="rate")
        wait = self.rate_limiter.retry_after(key)
        self.blocks += 1
        return Verdict.block(
            f"rate limit exceeded for {key!r}; retry in {wait:.1f}s",
            rule="policy:rate_limit", severity="medium", stage="rate",
            findings=[Finding(rule="policy:rate_limit", severity="medium", category="rate",
                              reason="requests per minute exceeded", score=0.8,
                              metadata={"key": key, "retry_after": wait})],
            metadata={"retry_after": wait},
        )

    # -- telemetry -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return engine telemetry."""
        with self._lock:
            rules = list(self._rules.values())
        return {
            "rules": len(rules),
            "enabled": sum(1 for rule in rules if rule.enabled),
            "by_kind": {kind: sum(1 for rule in rules if rule.kind == kind) for kind in RULE_KINDS},
            "checks": self.checks,
            "blocks": self.blocks,
            "warnings": self.warnings,
            "url_allowlist_only": self.url_allowlist_only,
            "rate": self.rate_limiter.stats(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine's rules for the config export and the API."""
        return {"rules": [rule.to_dict() for rule in self.rules(enabled_only=False)],
                "stats": self.stats()}


def rules_from_config(config: Any) -> list[Rule]:
    """Derive the standard rules from a :class:`GuardrailsConfig`."""
    rules: list[Rule] = []
    max_prompt = int(getattr(config, "max_prompt_chars", 0) or 0)
    if max_prompt > 0:
        rules.append(Rule(name="max_prompt_chars", kind="max_chars", limit=max_prompt,
                          applies_to="prompt", action="block", severity="medium",
                          reason=f"prompt exceeds {max_prompt} characters"))
    max_output = int(getattr(config, "max_output_chars", 0) or 0)
    if max_output > 0:
        rules.append(Rule(name="max_output_chars", kind="max_chars", limit=max_output,
                          applies_to="output", action="block", severity="medium",
                          reason=f"output exceeds {max_output} characters"))
    for index, pattern in enumerate(getattr(config, "block_patterns", []) or [], start=1):
        if str(pattern).strip():
            rules.append(Rule(name=f"block_pattern_{index}", kind="block_pattern",
                              pattern=str(pattern), action="block", severity="high",
                              reason=f"matches configured block pattern {index}"))
    for index, pattern in enumerate(getattr(config, "allow_patterns", []) or [], start=1):
        if str(pattern).strip():
            rules.append(Rule(name=f"require_pattern_{index}", kind="require_pattern",
                              pattern=str(pattern), action="warn", severity="low",
                              applies_to="prompt",
                              reason=f"does not match configured allow pattern {index}"))
    return rules


def deny_tool_rule(pattern: str, *, reason: str = "") -> Rule:
    """Build a rule denying tool calls whose name matches a glob."""
    return Rule(name=f"deny_tool:{pattern}", kind="tool", pattern=pattern, action="block",
                severity="high", reason=reason or f"tool {pattern!r} is denied by policy")


__all__ = [
    "RULE_KINDS",
    "PolicyEngine",
    "RateLimiter",
    "Rule",
    "deny_tool_rule",
    "rules_from_config",
]
