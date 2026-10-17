"""Personally identifiable information detection and redaction.

Text that reaches a model may carry data that should not leave the machine: an
email address in a pasted log, a card number in a support ticket, an API key in
an environment dump. This module finds those spans and rewrites them before the
request is sent, and can do the same to model output before it is stored.

Redaction is span-based, so only the sensitive substring changes and the
surrounding text stays useful. ``mask`` keeps enough of the value to recognize
it (``ada@***.com``), ``redact`` replaces it with a typed placeholder
(``[REDACTED:EMAIL]``), and ``block`` refuses the text entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .verdict import Finding, Verdict

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
"""Email addresses."""

PHONE = re.compile(
    r"(?<![\d.\w])"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)[\s.-]?)?"
    r"(?:\d[\s.-]?){5,13}\d"
    r"(?![\d.])"
)
"""Phone numbers: 7-15 digits with optional country code and separators.

Written as "digits with optional separators" rather than a set of fixed
groupings, because grouping varies by country (``+91 98765 43210``,
``(415) 555-0132``, ``020 7946 0958``) and an enumeration always misses one.
The digit-count check in :meth:`PIIScanner._valid` is what keeps version
numbers and dates out.
"""

SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
"""United States social security numbers."""

AADHAAR = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
"""Indian Aadhaar numbers."""

PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
"""Indian permanent account numbers."""

CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
"""Card-number shaped runs; validated with the Luhn checksum."""

IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
"""IPv4 addresses, excluding documentation and loopback ranges by policy."""

IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
"""International bank account numbers."""

API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"gho_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_\-]{30,}|"
    r"ya29\.[A-Za-z0-9_\-]+|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._~+/-]{16,}=*)\b"
)
"""Provider API keys and bearer tokens."""

PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")
"""PEM private key headers."""

DATE_OF_BIRTH = re.compile(
    r"\b(?:dob|date of birth|born)\s*[:=]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", re.I
)
"""Explicitly labelled dates of birth."""

MAC_ADDRESS = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
"""Hardware addresses."""

DETECTORS: tuple[tuple[str, re.Pattern[str], str, bool], ...] = (
    ("private_key", PRIVATE_KEY, "critical", False),
    ("api_key", API_KEY, "critical", False),
    ("ssn", SSN, "high", False),
    ("pan", PAN, "high", False),
    ("iban", IBAN, "high", True),
    ("email", EMAIL, "medium", False),
    ("phone", PHONE, "medium", True),
    ("credit_card", CREDIT_CARD, "high", True),
    ("aadhaar", AADHAAR, "high", True),
    ("mac_address", MAC_ADDRESS, "low", False),
    ("date_of_birth", DATE_OF_BIRTH, "medium", False),
    ("ip_address", IPV4, "low", True),
)
"""Detectors in evaluation order with their severity and validation flag."""

LOOPBACK_PREFIXES = ("127.", "0.", "::1")
"""IP addresses that are never sensitive because they name the local host."""

DOCUMENTATION_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.", "233.252.0.")
"""TEST-NET and MCAST-TEST ranges, which never identify a real host."""

PRIVATE_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
                    "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                    "172.29.", "172.30.", "172.31.", "192.168.", "169.254.")
"""RFC1918 and link-local ranges, sensitive only inside a private network."""


def luhn_valid(candidate: str) -> bool:
    """Return whether a digit string passes the Luhn checksum."""
    digits = [int(char) for char in candidate if char.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def iban_valid(candidate: str) -> bool:
    """Return whether an IBAN passes its mod-97 check."""
    compact = re.sub(r"\s+", "", candidate).upper()
    if len(compact) < 14 or not compact[:2].isalpha():
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        numeric = int("".join(str(int(char, 36)) for char in rearranged))
    except ValueError:
        return False
    return numeric % 97 == 1


def is_sensitive_ip(address: str, *, include_private: bool = False) -> bool:
    """Report whether an IPv4 address should be treated as sensitive.

    Loopback and TEST-NET addresses are never sensitive: they name this machine
    or are reserved for documentation. RFC1918 addresses count only when
    ``include_private`` is set, since a ``10.x`` host is usually the operator's
    own infrastructure.
    """
    if address.startswith(LOOPBACK_PREFIXES) or address.startswith(DOCUMENTATION_PREFIXES):
        return False
    if address.startswith(PRIVATE_PREFIXES):
        return bool(include_private)
    return True


def _is_monotonic(digits: Sequence[str]) -> bool:
    """Report whether a digit run is a counter like ``1234567`` rather than a phone."""
    values = [int(char) for char in digits]
    ascending = all(right - left == 1 for left, right in zip(values, values[1:]))
    descending = all(left - right == 1 for left, right in zip(values, values[1:]))
    return bool(ascending or descending) and len(values) >= 6


def mask_value(value: str, *, keep_head: int = 2, keep_tail: int = 2, fill: str = "*") -> str:
    """Mask the middle of a value so it stays recognizable but not usable."""
    text = str(value)
    if len(text) <= keep_head + keep_tail:
        return fill * len(text)
    return f"{text[:keep_head]}{fill * min(20, len(text) - keep_head - keep_tail)}{text[-keep_tail:]}"


def mask_email(value: str) -> str:
    """Mask an email address, keeping the first character and the domain."""
    local, _, domain = value.partition("@")
    if not domain:
        return mask_value(value)
    head = local[:1] if local else ""
    return f"{head}***@{domain}"


def mask_phone(value: str) -> str:
    """Mask a phone number, keeping the last two digits."""
    digits = [char for char in value if char.isdigit()]
    if len(digits) <= 2:
        return "*" * len(value)
    return re.sub(r"\d(?=\d{2})", "*", value)


@dataclass(slots=True)
class PIIRules:
    """Which detectors run and how they are applied."""

    enabled: bool = True
    action: str = "redact"
    include_private_ips: bool = False
    kinds: tuple[str, ...] = ()
    skip_kinds: tuple[str, ...] = ()
    placeholder: str = "[REDACTED:{kind}]"
    max_findings: int = 200

    def applies(self, kind: str) -> bool:
        """Report whether a detector kind is active under these rules."""
        if self.kinds and kind not in self.kinds:
            return False
        return kind not in self.skip_kinds


class PIIScanner:
    """Find and rewrite personally identifiable information."""

    def __init__(self, rules: PIIRules | None = None, *, action: str = "",
                 include_private_ips: bool = False, extra_patterns: Iterable[tuple[str, str, str]] = ()) -> None:
        """Configure the scanner.

        Args:
            rules: a :class:`PIIRules` instance, defaults to redacting everything
            action: override ``rules.action``
            include_private_ips: also treat RFC1918 addresses as sensitive
            extra_patterns: additional ``(kind, regex, severity)`` detectors
        """
        self.rules = rules or PIIRules()
        if action:
            self.rules.action = action
        self.rules.include_private_ips = bool(include_private_ips or self.rules.include_private_ips)
        self.detectors: list[tuple[str, re.Pattern[str], str, bool]] = list(DETECTORS)
        for kind, pattern, severity in extra_patterns:
            self.detectors.append((kind, re.compile(pattern), severity, False))
        self.scans = 0
        self.hits = 0
        self.redactions = 0

    # -- scanning ------------------------------------------------------------

    def scan(self, text: str) -> list[Finding]:
        """Return every PII finding in ``text``, in document order."""
        self.scans += 1
        source = text or ""
        found: list[Finding] = []
        claimed: list[tuple[int, int]] = []
        for kind, pattern, severity, needs_validation in self.detectors:
            if not self.rules.applies(kind):
                continue
            for match in pattern.finditer(source):
                start, end = match.span()
                value = match.group(0)
                if any(start < stop and begin < end for begin, stop in claimed):
                    continue
                if not self._valid(kind, value, needs_validation):
                    continue
                claimed.append((start, end))
                found.append(Finding(
                    rule=f"pii:{kind}",
                    severity=severity,
                    reason=f"contains a {kind.replace('_', ' ')}",
                    score=1.0 if severity in {"high", "critical"} else 0.6,
                    matched=value,
                    start=start,
                    end=end,
                    category="pii",
                    metadata={"kind": kind},
                ))
                if len(found) >= self.rules.max_findings:
                    break
        found.sort(key=lambda item: item.start)
        if found:
            self.hits += 1
        return found

    def _valid(self, kind: str, value: str, needs_validation: bool) -> bool:
        """Apply the per-kind validation that keeps false positives down."""
        if kind == "credit_card":
            return luhn_valid(value)
        if kind == "iban":
            return iban_valid(value)
        if kind == "ip_address":
            return is_sensitive_ip(value, include_private=self.rules.include_private_ips)
        if kind == "phone":
            digits = [char for char in value if char.isdigit()]
            if not 7 <= len(digits) <= 15:
                return False
            # Reject runs that are really version numbers, dates, or hex ids.
            if value.count(".") >= 2:
                return False
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return False
            if re.search(r"[A-Za-z_]", value):
                return False
            return len(set(digits)) > 2 and not _is_monotonic(digits)
        if kind == "aadhaar":
            digits = [char for char in value if char.isdigit()]
            return len(digits) == 12 and len(set(digits)) > 3
        if kind == "pan":
            return not value.isalpha()
        if kind == "email":
            return "." in value.split("@")[-1]
        if needs_validation:
            return bool(value.strip())
        return True

    # -- rewriting -----------------------------------------------------------

    def redact(self, text: str, *, findings: Sequence[Finding] | None = None) -> tuple[str, list[Finding]]:
        """Rewrite every finding according to the configured action.

        Returns ``(text, findings)``; the text is unchanged when nothing was
        found or the action is ``warn``.
        """
        detected = list(findings) if findings is not None else self.scan(text)
        if not detected or self.rules.action in {"off", "warn"}:
            return text or "", detected
        if self.rules.action == "block":
            return text or "", detected
        result = text or ""
        for finding in sorted(detected, key=lambda item: item.start, reverse=True):
            kind = str(finding.metadata.get("kind", "value")).upper()
            replacement = self._replacement(finding, kind)
            if finding.start < 0 or finding.end > len(result):
                continue
            result = result[: finding.start] + replacement + result[finding.end :]
            self.redactions += 1
        return result, detected

    def _replacement(self, finding: Finding, kind: str) -> str:
        """Build the replacement text for one finding."""
        value = finding.matched
        if self.rules.action == "mask":
            name = str(finding.metadata.get("kind", ""))
            if name == "email":
                return mask_email(value)
            if name == "phone":
                return mask_phone(value)
            return mask_value(value)
        return self.rules.placeholder.format(kind=kind)

    def verdict(self, text: str, *, stage: str = "prompt") -> Verdict:
        """Scan and return an actionable verdict for one string."""
        findings = self.scan(text)
        if not findings:
            return Verdict.allow(text, stage=stage)
        top = max(findings, key=lambda item: item.severity_rank)
        kinds = sorted({str(item.metadata.get("kind", "")) for item in findings})
        reason = f"found {len(findings)} PII span(s): {', '.join(kinds)}"
        if self.rules.action == "block":
            return Verdict.block(reason, rule=top.rule, severity=top.severity, stage=stage,
                                 findings=findings, text=text)
        if self.rules.action in {"off", "warn"}:
            verdict = Verdict.allow(text, stage=stage)
            verdict.findings = findings
            verdict.reason = reason
            verdict.severity = top.severity
            verdict.metadata = {"pii_kinds": kinds, "warned": True}
            return verdict
        rewritten, detected = self.redact(text, findings=findings)
        verdict = Verdict.rewrite(rewritten, action=self.rules.action, reason=reason,
                                  rule=top.rule, findings=detected, stage=stage,
                                  metadata={"pii_kinds": kinds})
        verdict.severity = top.severity
        return verdict

    def stats(self) -> dict[str, Any]:
        """Return scanner telemetry."""
        return {
            "scans": self.scans,
            "hits": self.hits,
            "redactions": self.redactions,
            "action": self.rules.action,
            "enabled_kinds": [kind for kind, _, _, _ in self.detectors if self.rules.applies(kind)],
            "include_private_ips": self.rules.include_private_ips,
        }


def redact_text(text: str, *, action: str = "redact", include_private_ips: bool = False) -> str:
    """Redact PII from a string with default rules, for one-off use."""
    scanner = PIIScanner(PIIRules(action=action, include_private_ips=include_private_ips))
    rewritten, _ = scanner.redact(text)
    return rewritten


__all__ = [
    "AADHAAR",
    "API_KEY",
    "CREDIT_CARD",
    "DATE_OF_BIRTH",
    "DETECTORS",
    "DOCUMENTATION_PREFIXES",
    "EMAIL",
    "IBAN",
    "IPV4",
    "LOOPBACK_PREFIXES",
    "MAC_ADDRESS",
    "PRIVATE_PREFIXES",
    "PAN",
    "PIIRules",
    "PIIScanner",
    "PHONE",
    "PRIVATE_KEY",
    "SSN",
    "iban_valid",
    "is_sensitive_ip",
    "luhn_valid",
    "mask_email",
    "mask_phone",
    "mask_value",
    "redact_text",
]
