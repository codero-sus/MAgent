"""Human-in-the-loop approval policies for tool execution.

Four modes cover the whole spectrum:

``never``
    run everything without asking (CI, batch jobs, fully trusted sandbox)
``dangerous``
    ask only for writes, code execution, network, and destructive actions
``ask``
    ask for anything that is not a pure read
``always``
    ask for every single call, including reads

Approvals are remembered per session (and optionally per exact argument
signature) so an operator is not asked the same question twenty times. Every
decision is emitted on the event bus and written to the audit log.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ApprovalRequiredError
from ..logging_ import get_logger
from ..types import ApprovalDecision, ApprovalMode
from .sandbox import SandboxDecision

LOGGER = get_logger("approval")

MODES: tuple[str, ...] = ("never", "ask", "always", "dangerous")
"""Valid approval modes."""

RISK_REQUIRING_APPROVAL: frozenset[str] = frozenset({"write", "execute", "destructive", "network"})
"""Risk levels that always need approval in ``dangerous`` mode."""

PromptFn = Callable[["ApprovalRequest"], ApprovalDecision | str]
"""Signature of the interactive approval callback."""


@dataclass(slots=True)
class ApprovalRequest:
    """Everything an approver needs to decide."""

    tool: str
    arguments: dict[str, Any]
    risk: str = "safe"
    reason: str = ""
    category: str = "general"
    destructive: bool = False
    network: bool = False
    session_id: str = ""
    run_id: str = ""
    sandbox_decision: dict[str, Any] | None = None
    signature: str = ""
    created_at: float = field(default_factory=time.time)
    timeout: float = 0.0

    def summary(self, *, width: int = 100) -> str:
        """Return a one-line human summary of the request."""
        from ..utils.text import ellipsize

        args = ", ".join(f"{key}={_render(value)}" for key, value in sorted(self.arguments.items()))
        return ellipsize(f"{self.tool}({args})", width)

    def risk_banner(self) -> str:
        """Return a warning banner appropriate to the risk level."""
        if self.destructive:
            return "DESTRUCTIVE ACTION"
        if self.risk == "execute":
            return "CODE EXECUTION"
        if self.risk == "network":
            return "NETWORK ACCESS"
        if self.risk == "write":
            return "FILE WRITE"
        return "TOOL CALL"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for events and audit entries."""
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "risk": self.risk,
            "reason": self.reason,
            "category": self.category,
            "destructive": self.destructive,
            "network": self.network,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "signature": self.signature,
            "summary": self.summary(),
            "sandbox_decision": self.sandbox_decision,
        }


def _render(value: Any) -> str:
    """Render one argument value compactly for display."""
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


@dataclass(slots=True)
class ApprovalPolicy:
    """Declarative approval configuration."""

    mode: ApprovalMode = "dangerous"
    auto_approve_tools: list[str] = field(default_factory=list)
    always_ask_tools: list[str] = field(default_factory=list)
    auto_approve_patterns: list[str] = field(default_factory=list)
    remember_session: bool = True
    remember_exact: bool = True
    deny_on_timeout: bool = True
    timeout: float = 0.0
    max_remembered: int = 512
    require_approval_for_reads: bool = False

    def validate(self) -> None:
        """Reject unknown modes."""
        if self.mode not in MODES:
            raise ValueError(f"approval mode must be one of {MODES}")


@dataclass(slots=True)
class ApprovalRecord:
    """One recorded decision, kept for audit and replay."""

    request: ApprovalRequest
    decision: ApprovalDecision
    by: str = "auto"
    reason: str = ""
    ts: float = field(default_factory=time.time)
    remembered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return {
            "tool": self.request.tool,
            "signature": self.request.signature,
            "risk": self.request.risk,
            "decision": self.decision.value,
            "by": self.by,
            "reason": self.reason,
            "ts": self.ts,
            "remembered": self.remembered,
            "summary": self.request.summary(),
        }


class ApprovalManager:
    """Decides whether a tool call may proceed, and asks a human when needed."""

    def __init__(
        self,
        policy: ApprovalPolicy | None = None,
        *,
        prompt: PromptFn | None = None,
        events: Any = None,
        audit: Any = None,
    ) -> None:
        self.policy = policy or ApprovalPolicy()
        self.policy.validate()
        self.prompt = prompt
        self.events = events
        self.audit = audit
        self._lock = threading.RLock()
        self._session_approved: set[str] = set()
        self._exact_approved: set[str] = set()
        self._denied: set[str] = set()
        self.records: list[ApprovalRecord] = []
        self.counts: Counter[str] = Counter()
        self.auto_approved = 0
        self.asked = 0
        self.denied_count = 0

    # -- decision logic ------------------------------------------------------

    def needs_approval(self, request: ApprovalRequest) -> tuple[bool, str]:
        """Return ``(needs_approval, reason)`` without prompting anyone."""
        policy = self.policy
        if policy.mode == "never":
            return False, "approval mode is 'never'"
        if self._matches_any(request.tool, policy.auto_approve_tools) or self._matches_any(
            request.summary(), policy.auto_approve_patterns
        ):
            return False, "tool is on the auto-approve list"
        if self._matches_any(request.tool, policy.always_ask_tools):
            return True, "tool is on the always-ask list"
        if policy.mode == "always":
            return True, "approval mode is 'always'"
        if policy.mode == "ask":
            if request.risk == "read" or request.risk == "safe":
                if policy.require_approval_for_reads:
                    return True, "approval mode is 'ask' and reads require approval"
                return False, "read-only tool under 'ask' mode"
            return True, "approval mode is 'ask'"
        if request.risk in RISK_REQUIRING_APPROVAL or request.destructive or request.network:
            return True, f"risk level {request.risk!r} requires approval"
        return False, "risk level does not require approval"

    def _matches_any(self, value: str, patterns: list[str]) -> bool:
        """Test a value against a list of exact names or globs."""
        for pattern in patterns:
            if value == pattern or fnmatch.fnmatch(value, pattern):
                return True
        return False

    def _signature(self, request: ApprovalRequest) -> str:
        """Return the stable identity of an exact call."""
        if request.signature:
            return request.signature
        import json

        return f"{request.tool}:{json.dumps(request.arguments, sort_keys=True, default=str)}"

    # -- execution -----------------------------------------------------------

    def evaluate(self, request: ApprovalRequest) -> ApprovalRecord:
        """Decide one request, prompting the operator when required."""
        signature = self._signature(request)
        request.signature = signature
        with self._lock:
            if signature in self._denied:
                self.denied_count += 1
                return self._record(
                    request,
                    ApprovalDecision.DENY,
                    by="memory",
                    reason="previously denied in this session",
                )
            if self.policy.remember_exact and signature in self._exact_approved:
                self.auto_approved += 1
                return self._record(
                    request,
                    ApprovalDecision.APPROVE,
                    by="memory",
                    reason="identical call approved earlier",
                    remembered=True,
                )
            if self.policy.remember_session and request.tool in self._session_approved:
                self.auto_approved += 1
                return self._record(
                    request,
                    ApprovalDecision.APPROVE,
                    by="memory",
                    reason="tool approved for this session",
                    remembered=True,
                )

        needed, reason = self.needs_approval(request)
        if not needed:
            self.auto_approved += 1
            self.counts["auto"] += 1
            return self._record(request, ApprovalDecision.APPROVE, by="policy", reason=reason)

        self._emit("guard.approval_requested", request=request.to_dict(), reason=reason)
        self.asked += 1
        decision, decided_by, note = self._ask(request, reason)
        self.counts[decision.value] += 1

        if decision is ApprovalDecision.APPROVE or decision is ApprovalDecision.APPROVE_SESSION:
            with self._lock:
                if decision is ApprovalDecision.APPROVE_SESSION:
                    self._session_approved.add(request.tool)
                if self.policy.remember_exact:
                    self._exact_approved.add(signature)
                    self._trim_memory()
        else:
            with self._lock:
                self._denied.add(signature)
            self.denied_count += 1
        record = self._record(request, decision, by=decided_by, reason=note or reason)
        self._emit("guard.approval_decided", **record.to_dict())
        if decision is ApprovalDecision.DENY:
            raise ApprovalRequiredError(
                f"Approval denied for {request.tool}: {note or reason}",
                details={"tool": request.tool, "arguments": request.arguments},
            )
        return record

    def _ask(self, request: ApprovalRequest, reason: str) -> tuple[ApprovalDecision, str, str]:
        """Invoke the configured prompter, defaulting to deny."""
        if self.prompt is None:
            if self.policy.deny_on_timeout:
                return ApprovalDecision.DENY, "policy", "no approver configured; denying by default"
            return ApprovalDecision.APPROVE, "policy", "no approver configured; auto-approving"
        try:
            raw = self.prompt(request)
        except Exception as exc:
            LOGGER.warning("approval prompt raised: %s", exc)
            return ApprovalDecision.DENY, "error", f"prompt failed: {exc}"
        if isinstance(raw, ApprovalDecision):
            return raw, "operator", reason
        text = str(raw).strip().lower()
        mapping = {
            "y": ApprovalDecision.APPROVE,
            "yes": ApprovalDecision.APPROVE,
            "a": ApprovalDecision.APPROVE,
            "approve": ApprovalDecision.APPROVE,
            "s": ApprovalDecision.APPROVE_SESSION,
            "session": ApprovalDecision.APPROVE_SESSION,
            "always": ApprovalDecision.APPROVE_SESSION,
            "n": ApprovalDecision.DENY,
            "no": ApprovalDecision.DENY,
            "d": ApprovalDecision.DENY,
            "deny": ApprovalDecision.DENY,
            "edit": ApprovalDecision.EDIT,
        }
        return mapping.get(text, ApprovalDecision.DENY), "operator", text

    def _record(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        *,
        by: str,
        reason: str,
        remembered: bool = False,
    ) -> ApprovalRecord:
        """Store a decision and write it to the audit log."""
        record = ApprovalRecord(
            request=request, decision=decision, by=by, reason=reason, remembered=remembered
        )
        with self._lock:
            self.records.append(record)
            if len(self.records) > 1000:
                del self.records[: len(self.records) - 1000]
        if self.audit is not None:
            try:
                self.audit.record("approval", record.to_dict())
            except Exception:
                pass
        return record

    def _trim_memory(self) -> None:
        """Keep remembered signatures bounded."""
        if len(self._exact_approved) > self.policy.max_remembered:
            excess = len(self._exact_approved) - self.policy.max_remembered
            for signature in list(self._exact_approved)[:excess]:
                self._exact_approved.discard(signature)

    def _emit(self, kind: str, **data: Any) -> None:
        """Publish an approval event when a bus is attached."""
        if self.events is None:
            return
        try:
            self.events.emit(kind, **data)
        except Exception:
            pass

    # -- management ----------------------------------------------------------

    def approve_tool(self, tool_name: str) -> None:
        """Pre-approve a tool for the whole session."""
        with self._lock:
            self._session_approved.add(tool_name)

    def deny_tool(self, tool_name: str) -> None:
        """Pre-deny every call to a tool."""
        with self._lock:
            self._denied.add(tool_name)

    def reset(self) -> None:
        """Forget every remembered decision."""
        with self._lock:
            self._session_approved.clear()
            self._exact_approved.clear()
            self._denied.clear()

    def stats(self) -> dict[str, Any]:
        """Return approval telemetry."""
        with self._lock:
            return {
                "mode": self.policy.mode,
                "records": len(self.records),
                "auto_approved": self.auto_approved,
                "asked": self.asked,
                "denied": self.denied_count,
                "counts": dict(self.counts),
                "remembered_tools": sorted(self._session_approved),
                "remembered_signatures": len(self._exact_approved),
                "denied_signatures": len(self._denied),
            }

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent approval decisions."""
        with self._lock:
            return [record.to_dict() for record in self.records[-max(1, limit) :]]


def build_request(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    risk: str = "safe",
    category: str = "general",
    destructive: bool = False,
    network: bool = False,
    reason: str = "",
    session_id: str = "",
    run_id: str = "",
    sandbox_decision: SandboxDecision | None = None,
) -> ApprovalRequest:
    """Assemble an :class:`ApprovalRequest` from tool metadata."""
    return ApprovalRequest(
        tool=tool_name,
        arguments=dict(arguments or {}),
        risk=risk,
        category=category,
        destructive=destructive,
        network=network,
        reason=reason,
        session_id=session_id,
        run_id=run_id,
        sandbox_decision=sandbox_decision.to_dict() if sandbox_decision else None,
    )


def auto_approve(request: ApprovalRequest) -> ApprovalDecision:
    """A prompter that approves everything, for fully trusted batch runs."""
    return ApprovalDecision.APPROVE


def deny_all(request: ApprovalRequest) -> ApprovalDecision:
    """A prompter that refuses everything, for read-only runs."""
    return ApprovalDecision.DENY


def approve_reads(request: ApprovalRequest) -> ApprovalDecision:
    """A prompter that approves reads and refuses writes."""
    if request.risk in {"safe", "read"} and not request.destructive and not request.network:
        return ApprovalDecision.APPROVE
    return ApprovalDecision.DENY


class ConsoleApprover:
    """Interactive terminal approver used by the CLI and REPL."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], Any] = print,
        default: ApprovalDecision = ApprovalDecision.DENY,
    ) -> None:
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.default = default
        self.decisions: list[ApprovalDecision] = []

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        """Render the request and read the operator's answer."""
        from ..utils.colors import ANSI, colorize
        from ..utils.tables import render_box

        banner = colorize(request.risk_banner(), ANSI.BOLD + ANSI.YELLOW)
        body = (
            f"{banner}\n"
            f"Tool   : {request.tool}\n"
            f"Risk   : {request.risk}  Category: {request.category}\n"
            f"Reason : {request.reason or 'policy requires approval'}\n"
            f"Call   : {request.summary(width=200)}"
        )
        self.output_fn(render_box(body, title="Approval required"))
        self.output_fn("  [y]es  [s]ession  [n]o (default: no)")
        try:
            answer = self.input_fn("Approve? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if not answer:
            decision = self.default
        else:
            decision = {
                "y": ApprovalDecision.APPROVE,
                "yes": ApprovalDecision.APPROVE,
                "s": ApprovalDecision.APPROVE_SESSION,
                "session": ApprovalDecision.APPROVE_SESSION,
                "n": ApprovalDecision.DENY,
                "no": ApprovalDecision.DENY,
            }.get(answer, self.default)
        self.decisions.append(decision)
        return decision


def prompter_for_mode(mode: str, *, interactive: bool = True) -> PromptFn:
    """Return the approver appropriate for a mode and terminal capability."""
    if mode == "never":
        return auto_approve
    if not interactive:
        return approve_reads
    return ConsoleApprover()


__all__ = [
    "MODES",
    "RISK_REQUIRING_APPROVAL",
    "ApprovalManager",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ApprovalRequest",
    "ConsoleApprover",
    "PromptFn",
    "approve_reads",
    "auto_approve",
    "build_request",
    "deny_all",
    "prompter_for_mode",
]
