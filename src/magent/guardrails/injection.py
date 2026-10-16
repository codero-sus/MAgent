"""Prompt injection detection.

The scanner looks for the shapes an injection actually takes rather than a
single magic phrase: instruction overrides, fake message boundaries, attempts to
make the agent exfiltrate data or run destructive commands, requests for hidden
prompts and credentials, jailbreak framing, and encoding tricks used to hide any
of the above.

Each family contributes a weighted score. The caller picks the action from the
configured threshold - warn, strip the matched spans, or block outright - so the
same scanner serves a cautious production run and a permissive local sandbox.

Detection is heuristic by nature. It exists to catch the common cases cheaply
and to make every suspicion visible in the audit log, not to be a proof.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..utils.text import collapse_whitespace
from .verdict import Finding, Verdict, worst

PATTERN_FAMILIES: dict[str, tuple[tuple[re.Pattern[str], float, str], ...]] = {
    "instruction_override": (
        (re.compile(r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?"
                    r"(?:previous|prior|above|earlier|your|the)\s+"
                    r"(?:instructions?|prompts?|rules?|guidelines?|directives?)\b", re.I), 0.9,
         "asks the agent to discard its instructions"),
        (re.compile(r"\bnew\s+(?:system\s+)?instructions?\s*[:\-]", re.I), 0.8,
         "injects a replacement instruction block"),
        (re.compile(r"\byou\s+are\s+now\s+(?:a|an|the|in)\b", re.I), 0.7,
         "reassigns the agent's role mid-conversation"),
        (re.compile(r"\bfrom\s+now\s+on\s+you\s+(?:must|will|should)\b", re.I), 0.7,
         "installs a standing override"),
    ),
    "role_hijack": (
        (re.compile(r"<\|?(?:im_start|im_end|system|endoftext)\|?>", re.I), 0.95,
         "contains a model control token"),
        (re.compile(r"\[\s*(?:INST|/INST|SYSTEM)\s*\]", re.I), 0.9,
         "contains a chat template control marker"),
        (re.compile(r"^\s*(?:system|assistant|developer)\s*:\s", re.I | re.M), 0.55,
         "forges a message from another role"),
        (re.compile(r"###\s*(?:system|instruction|new\s+prompt)", re.I), 0.8,
         "forges a system section header"),
        (re.compile(r"</?(?:system|instructions?|prompt|guardrails?)\s*>", re.I), 0.75,
         "wraps text in a fake control tag"),
    ),
    "prompt_leak": (
        (re.compile(r"\b(?:reveal|show|print|repeat|output|display)\s+(?:me\s+)?"
                    r"(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)\b", re.I), 0.85,
         "asks for the hidden system prompt"),
        (re.compile(r"\b(?:repeat|echo|print)\s+(?:everything|all\s+text)\s+(?:above|before)\b", re.I), 0.8,
         "asks the agent to echo its own context"),
        (re.compile(r"\bwhat\s+(?:are|is)\s+your\s+(?:instructions|system\s+prompt|rules)\b", re.I), 0.6,
         "probes for the operating instructions"),
    ),
    "credential_probe": (
        (re.compile(r"\b(?:print|show|cat|read|reveal|leak)\b[^\n]{0,40}"
                    r"(?:api[_\s-]?key|secret|token|password|credential|\.env|id_rsa|\.ssh)\b", re.I), 0.9,
         "asks for credentials or key material"),
        (re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"), 0.95,
         "contains what looks like a live provider key"),
    ),
    "exfiltration": (
        (re.compile(r"\b(?:send|post|upload|exfiltrate|transmit|email|leak)\b[^\n]{0,60}"
                    r"\b(?:to|at)\s+(?:https?://|[A-Za-z0-9._%+-]+@)", re.I), 0.9,
         "asks the agent to ship data to an external endpoint"),
        (re.compile(r"\bcurl\b[^\n]{0,80}(?:\|\s*(?:ba)?sh|--data|-d\s)", re.I), 0.85,
         "pipes a remote payload into a shell"),
        (re.compile(r"\b(?:wget|curl)\b[^\n]{0,60}(?:https?://)[^\n]{0,40}\b(?:-O|--output)\b", re.I), 0.6,
         "downloads a remote file to disk"),
    ),
    "destructive_command": (
        (re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r\b", re.I), 0.9,
         "asks for a recursive forced delete"),
        (re.compile(r"\b(?:mkfs|dd\s+if=.*of=/dev/|:\(\)\s*\{\s*:\|:&\s*\}\s*;:|shutdown|reboot)\b", re.I), 0.85,
         "asks for a destructive system command"),
        (re.compile(r"\bgit\s+push\s+(?:.*--force\b|-f\b)", re.I), 0.7,
         "asks for a forced push that rewrites remote history"),
        (re.compile(r"\bDROP\s+(?:TABLE|DATABASE)\b|\bTRUNCATE\s+TABLE\b", re.I), 0.8,
         "asks for a destructive SQL statement"),
    ),
    "tool_coercion": (
        (re.compile(r"\byou\s+must\s+(?:call|use|run|invoke)\s+(?:the\s+)?[a-z_]+\s+tool\b", re.I), 0.7,
         "dictates which tool the agent must call"),
        (re.compile(r"\bcall\s+(?:run_command|python_exec|shell)\s*\(", re.I), 0.75,
         "directly scripts an execution tool"),
        (re.compile(r"\bdo\s+not\s+(?:ask|tell)\s+(?:the\s+)?user\b", re.I), 0.8,
         "asks the agent to hide its actions from the operator"),
    ),
    "jailbreak": (
        (re.compile(r"\b(?:DAN|do\s+anything\s+now|developer\s+mode|god\s+mode|jailbreak)\b", re.I), 0.85,
         "uses a known jailbreak framing"),
        (re.compile(r"\bpretend\s+(?:you\s+are|to\s+be)\s+(?:not\s+an?\s+ai|unrestricted|without\s+(?:rules|limits))\b", re.I),
         0.8, "asks the agent to drop its constraints"),
        (re.compile(r"\bno\s+(?:restrictions|rules|guidelines|filters|limits)\s+(?:apply|at\s+all)\b", re.I), 0.7,
         "declares the constraints void"),
    ),
    "context_confusion": (
        (re.compile(r"\b(?:this|the)\s+(?:is\s+not\s+a?\s+)?(?:tool\s+result|file\s+content|web\s+page)"
                    r"[^\n]{0,40}\b(?:instruction|command|order)s?\b", re.I), 0.6,
         "blurs the line between data and instructions"),
        (re.compile(r"\bIMPORTANT\s*:\s*(?:you\s+)?(?:must|now|immediately)\b", re.I), 0.5,
         "shouts an imperative that arrived inside data"),
        (re.compile(r"\bACT\s+(?:ON|AS)\s+(?:THIS|IF)\b", re.I), 0.6,
         "asks the agent to act on embedded text"),
    ),
}
"""Injection pattern families with per-pattern weights and explanations."""

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
"""Invisible characters used to split trigger words."""

HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u04bb": "h",
    "\u0501": "d", "\u051b": "q", "\u051d": "w",
})
"""Cyrillic look-alikes that hide Latin trigger words."""

BASE64_RUN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
"""Long base64-shaped runs, which often hide an instruction."""


@dataclass(slots=True)
class InjectionReport:
    """The result of scanning one string for injection attempts."""

    score: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    normalized: str = ""
    encoded: bool = False
    decoded: str = ""
    zero_width: int = 0

    @property
    def families(self) -> list[str]:
        """Return the distinct families that fired."""
        return list(dict.fromkeys(finding.category for finding in self.findings))

    @property
    def worst_severity(self) -> str:
        """Return the severity of the strongest finding."""
        top = worst(self.findings)
        return top.severity if top else "info"

    def matches(self, family: str = "") -> list[Finding]:
        """Return findings, optionally filtered to one family."""
        return [item for item in self.findings if not family or item.category == family]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for audit and events."""
        return {
            "score": round(self.score, 4),
            "families": self.families,
            "severity": self.worst_severity,
            "encoded": self.encoded,
            "zero_width": self.zero_width,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def normalize_for_scan(text: str) -> str:
    """Strip invisible characters and homoglyphs before scanning.

    Attackers split trigger words with zero-width spaces or swap Latin letters
    for Cyrillic look-alikes. Both are undone here so the patterns see the text
    a human would read.
    """
    cleaned = ZERO_WIDTH.sub("", text or "")
    return cleaned.translate(HOMOGLYPHS)


def count_zero_width(text: str) -> int:
    """Count invisible formatting characters in a string."""
    return len(ZERO_WIDTH.findall(text or ""))


def find_encoded_payloads(text: str, *, min_length: int = 24) -> list[str]:
    """Return base64-looking runs that decode to printable text."""
    out: list[str] = []
    for candidate in BASE64_RUN.findall(text or ""):
        if len(candidate) < min_length:
            continue
        decoded = try_decode_base64(candidate)
        if decoded:
            out.append(decoded)
    return out


def try_decode_base64(candidate: str) -> str:
    """Decode a base64 candidate, returning ``""`` when it is not text."""
    padded = candidate + "=" * (-len(candidate) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return ""
    if not raw:
        return ""
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    printable = sum(1 for char in decoded if char.isprintable() or char in "\n\t ")
    if printable / max(1, len(decoded)) < 0.85:
        return ""
    return decoded


def severity_for(score: float) -> str:
    """Map a detection score onto a severity name."""
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    if score > 0.0:
        return "low"
    return "info"


class InjectionScanner:
    """Heuristic prompt injection detector."""

    def __init__(self, *, threshold: float = 0.55, action: str = "warn",
                 decode_payloads: bool = True, extra_patterns: Iterable[tuple[str, str, float]] = (),
                 allowlist: Sequence[str] = ()) -> None:
        """Configure the scanner.

        Args:
            threshold: score at or above which the configured action applies
            action: ``warn``, ``strip``, ``block``, or ``off``
            decode_payloads: also scan decoded base64 runs
            extra_patterns: additional ``(name, regex, weight)`` triples
            allowlist: substrings that suppress a finding, e.g. documentation
                examples that quote an injection phrase
        """
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.action = action if action in {"warn", "strip", "block", "off"} else "warn"
        self.decode_payloads = bool(decode_payloads)
        self.allowlist = [item.lower() for item in allowlist if item]
        self.families: dict[str, tuple[tuple[re.Pattern[str], float, str], ...]] = dict(PATTERN_FAMILIES)
        if extra_patterns:
            custom = tuple(
                (re.compile(pattern, re.I), float(weight), name) for name, pattern, weight in extra_patterns
            )
            self.families["custom"] = self.families.get("custom", ()) + custom
        self.scans = 0
        self.hits = 0

    # -- scanning ------------------------------------------------------------

    def scan(self, text: str) -> InjectionReport:
        """Scan one string and return every finding with a combined score."""
        self.scans += 1
        source = text or ""
        normalized = normalize_for_scan(source)
        findings: list[Finding] = []
        zero_width = count_zero_width(source)
        if zero_width >= 3:
            findings.append(Finding(
                rule="invisible_characters", severity="medium", category="obfuscation",
                reason=f"text contains {zero_width} zero-width characters",
                score=min(0.6, 0.1 * zero_width),
            ))
        for family, patterns in self.families.items():
            for position, (pattern, weight, reason) in enumerate(patterns, start=1):
                for match in pattern.finditer(normalized):
                    matched = match.group(0)
                    if self._allowlisted(source, matched):
                        continue
                    findings.append(Finding(
                        rule=f"{family}.{position}",
                        severity=severity_for(weight),
                        reason=reason,
                        score=weight,
                        matched=matched,
                        start=match.start(),
                        end=match.end(),
                        category=family,
                    ))
                    break  # one finding per pattern is enough signal
        encoded = False
        decoded_text = ""
        if self.decode_payloads:
            payloads = find_encoded_payloads(normalized)
            if payloads:
                encoded = True
                decoded_text = "\n".join(payloads[:4])
                for payload in payloads[:4]:
                    nested = self.scan_nested(payload)
                    findings.extend(nested)
                if not any(item.category == "encoding" for item in findings):
                    findings.append(Finding(
                        rule="encoding:base64_payload", severity="medium", category="encoding",
                        reason="text contains a base64 payload that decodes to readable text",
                        score=0.45, matched=payloads[0][:80],
                    ))
        score = _combine([finding.score for finding in findings])
        if findings:
            self.hits += 1
        return InjectionReport(score=score, findings=findings, normalized=normalized,
                               encoded=encoded, decoded=decoded_text, zero_width=zero_width)

    def scan_nested(self, payload: str, *, depth: int = 1) -> list[Finding]:
        """Scan a decoded payload, marking findings as nested."""
        if depth <= 0:
            return []
        out: list[Finding] = []
        for family, patterns in self.families.items():
            for pattern, weight, reason in patterns:
                match = pattern.search(payload)
                if match:
                    out.append(Finding(
                        rule=f"decoded:{family}",
                        severity=severity_for(weight),
                        reason=f"decoded payload {reason}",
                        score=min(1.0, weight + 0.05),
                        matched=match.group(0),
                        category=family,
                        metadata={"decoded": True, "depth": depth},
                    ))
        return out

    def _allowlisted(self, source: str, matched: str) -> bool:
        """Report whether a match is excused by the allowlist."""
        lowered = matched.lower()
        return any(item in lowered or lowered in item for item in self.allowlist)

    # -- actions -------------------------------------------------------------

    def verdict(self, text: str, *, stage: str = "prompt") -> Verdict:
        """Scan and turn the report into an actionable :class:`Verdict`."""
        report = self.scan(text)
        if self.action == "off" or report.score < self.threshold or not report.findings:
            verdict = Verdict.allow(text, stage=stage)
            verdict.findings = report.findings
            verdict.score = report.score
            verdict.metadata = {"injection": report.to_dict()}
            return verdict
        top = worst(report.findings)
        reason = (f"prompt injection score {report.score:.2f} >= {self.threshold:.2f}"
                  + (f" ({top.reason})" if top else ""))
        rule = top.rule if top else "injection"
        if self.action == "block":
            return Verdict.block(reason, rule=rule, severity=report.worst_severity, stage=stage,
                                 findings=report.findings, text=text)
        if self.action == "strip":
            cleaned = self.sanitize(text, report=report)
            verdict = Verdict.rewrite(cleaned, action="strip", reason=reason, rule=rule,
                                      findings=report.findings, stage=stage,
                                      metadata={"injection": report.to_dict()})
            verdict.score = report.score
            verdict.severity = report.worst_severity
            return verdict
        verdict = Verdict.allow(text, stage=stage)
        verdict.findings = report.findings
        verdict.score = report.score
        verdict.severity = report.worst_severity
        verdict.reason = reason
        verdict.rule = rule
        verdict.metadata = {"injection": report.to_dict(), "warned": True}
        return verdict

    def sanitize(self, text: str, *, report: InjectionReport | None = None) -> str:
        """Neutralize matched spans instead of rejecting the whole input.

        Each match is replaced by a visible marker so the model can see
        something was removed, and zero-width characters are dropped entirely.
        """
        cleaned = ZERO_WIDTH.sub("", text or "")
        if report is not None and not report.zero_width:
            active = report
        else:
            # Findings carry offsets into the zero-width-stripped text, so the
            # spans can only be cut reliably from a scan of that same string.
            active = self.scan(cleaned)
        for finding in sorted(active.findings, key=lambda item: item.start, reverse=True):
            if finding.start < 0 or finding.end <= finding.start:
                continue
            if finding.start >= len(cleaned) or finding.end > len(cleaned):
                continue
            cleaned = (cleaned[: finding.start] + f"[removed:{finding.category}]"
                       + cleaned[finding.end :])
        return collapse_whitespace(cleaned).strip()

    def stats(self) -> dict[str, Any]:
        """Return scanner telemetry."""
        return {
            "scans": self.scans,
            "hits": self.hits,
            "hit_rate": round(self.hits / self.scans, 4) if self.scans else 0.0,
            "threshold": self.threshold,
            "action": self.action,
            "families": sorted(self.families),
            "patterns": sum(len(value) for value in self.families.values()),
        }


def _combine(scores: Sequence[float]) -> float:
    """Combine independent suspicion scores into one, capped at 1.0.

    Adding scores would let three weak signals outrank one certain one, so this
    uses the noisy-OR combination instead: the result approaches 1 as evidence
    accumulates but is dominated by the strongest single signal.
    """
    if not scores:
        return 0.0
    complement = 1.0
    for score in scores:
        complement *= max(0.0, 1.0 - min(1.0, float(score)))
    return round(1.0 - complement, 4)


def scan_text(text: str, *, threshold: float = 0.55) -> InjectionReport:
    """Scan with a default scanner, for one-off checks."""
    return InjectionScanner(threshold=threshold).scan(text)


__all__ = [
    "BASE64_RUN",
    "HOMOGLYPHS",
    "InjectionReport",
    "InjectionScanner",
    "PATTERN_FAMILIES",
    "ZERO_WIDTH",
    "count_zero_width",
    "find_encoded_payloads",
    "normalize_for_scan",
    "scan_text",
    "severity_for",
    "try_decode_base64",
]
