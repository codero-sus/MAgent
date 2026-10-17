"""Handoffs: the note one agent leaves for the next.

Multi-agent runs fail quietly when the second agent has to guess what the first
one did.  A :class:`Handoff` makes that explicit - task, what was done, what is
still open, which files matter, and any blockers - and it is produced either
from a finished :class:`~magent.types.RunResult` or by asking the agent for one
in JSON (:data:`HANDOFF_PROMPT`).  :func:`handoff_block` renders it back into a
prompt, so the receiving agent starts with the state instead of re-deriving it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import RunResult
from ..utils.json_utils import extract_json
from ..utils.text import ellipsize

HANDOFF_PROMPT = """Write a handoff note for the next agent as a single JSON object.

Keys:
- "task": the goal in one sentence
- "summary": what you actually did, with concrete results
- "next_steps": ordered list of what remains
- "files": paths that matter, most important first
- "blockers": anything that stopped you (empty list if none)
- "confidence": a number 0..1 for how complete the work is

Return only the JSON object. No prose before or after it.
"""
"""Prompt used to ask an agent for a structured handoff."""

MAX_SUMMARY_CHARS = 4_000
"""Ceiling on the handoff summary."""


@dataclass(slots=True)
class Handoff:
    """A structured transfer of work between agents."""

    task: str = ""
    summary: str = ""
    next_steps: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    from_agent: str = ""
    to_agent: str = ""
    run_id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clamp numeric fields and normalise the lists."""
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.summary = ellipsize(self.summary or "", MAX_SUMMARY_CHARS)
        self.next_steps = [str(item).strip() for item in self.next_steps if str(item).strip()]
        self.files = [str(item).strip() for item in self.files if str(item).strip()]
        self.blockers = [str(item).strip() for item in self.blockers if str(item).strip()]

    @property
    def complete(self) -> bool:
        """True when the handoff reports no blockers and full confidence."""
        return not self.blockers and self.confidence >= 0.99

    @property
    def blocked(self) -> bool:
        """True when something stopped the handing-off agent."""
        return bool(self.blockers)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the handoff."""
        return {
            "task": self.task,
            "summary": self.summary,
            "next_steps": list(self.next_steps),
            "files": list(self.files),
            "blockers": list(self.blockers),
            "confidence": round(self.confidence, 3),
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], **extra: Any) -> Handoff:
        """Build a handoff from parsed JSON, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        values = {key: value for key, value in raw.items() if key in known}
        values.update({key: value for key, value in extra.items() if key in known})
        steps = values.get("next_steps")
        if isinstance(steps, str):
            values["next_steps"] = [
                line.strip(" -\t") for line in steps.split("\n") if line.strip()
            ]
        for key in ("files", "blockers"):
            item = values.get(key)
            if isinstance(item, str):
                values[key] = [part.strip() for part in item.split(",") if part.strip()]
        return cls(**values)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise as JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def render(self, *, max_chars: int = 2_000) -> str:
        """Render the handoff as prompt text for the receiving agent."""
        return handoff_block(self, max_chars=max_chars)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (
            f"Handoff(task={ellipsize(self.task, 40)!r}, from={self.from_agent!r}, "
            f"to={self.to_agent!r}, steps={len(self.next_steps)}, "
            f"confidence={self.confidence:.2f})"
        )


def build_handoff(
    result: RunResult | None = None,
    *,
    task: str = "",
    from_agent: str = "",
    to_agent: str = "",
    summary: str = "",
    next_steps: Sequence[str] = (),
    files: Sequence[str] = (),
    blockers: Sequence[str] = (),
    confidence: float | None = None,
    **metadata: Any,
) -> Handoff:
    """Build a handoff, deriving what it can from a finished run.

    Args:
        result: the run to summarise; may be ``None``
        task: goal text, taken from the run metadata when omitted
        from_agent: name of the agent handing off
        to_agent: name of the agent receiving the work
        summary: explicit summary, overriding the run output
        next_steps: remaining work
        files: files touched or referenced
        blockers: what stopped the run
        confidence: 0..1 completeness, inferred from the stop reason when omitted
        **metadata: extra keys stored on the handoff
    """
    text = summary
    derived_task = task
    run_id = ""
    touched: list[str] = list(files)
    if result is not None:
        # Duck typing on purpose: a single-agent RunResult and a team result both
        # describe finished work, and a handoff is useful after either.
        run_id = str(getattr(result, "run_id", "") or "")
        if not text:
            text = str(getattr(result, "output", "") or "")
        metadata = getattr(result, "metadata", None) or {}
        if not derived_task:
            derived_task = str(metadata.get("prompt", "") or "") or str(
                getattr(result, "task", "") or ""
            )
        for step in getattr(result, "steps", None) or []:
            for item in getattr(step, "tool_results", None) or []:
                path = str((getattr(item, "metadata", None) or {}).get("path", "") or "")
                if path and path not in touched:
                    touched.append(path)
        error = str(getattr(result, "error", "") or "")
        if not blockers and error:
            blockers = [error]
        if confidence is None:
            confidence = 1.0 if getattr(result, "succeeded", False) else 0.4
    if confidence is None:
        confidence = 0.8 if text and not blockers else 0.3
    return Handoff(
        task=derived_task,
        summary=text,
        next_steps=list(next_steps),
        files=touched,
        blockers=list(blockers),
        confidence=confidence,
        from_agent=from_agent,
        to_agent=to_agent,
        run_id=run_id,
        metadata=dict(metadata),
    )


def parse_handoff(text: str, *, from_agent: str = "", to_agent: str = "") -> Handoff:
    """Parse a handoff from model output, tolerating prose around the JSON.

    When no JSON object is present the whole text becomes the summary, so a
    model that ignores the format still produces a usable handoff.
    """
    payload: Any = None
    if text:
        candidate = extract_json(text)
        if isinstance(candidate, str):
            try:
                payload = json.loads(candidate)
            except ValueError:
                payload = None
        else:
            payload = candidate
    if isinstance(payload, dict):
        return Handoff.from_dict(payload, from_agent=from_agent, to_agent=to_agent)
    return Handoff(
        task="",
        summary=(text or "").strip(),
        from_agent=from_agent,
        to_agent=to_agent,
        confidence=0.5,
        metadata={"parsed": False, "reason": "no JSON object found"},
    )


def handoff_block(handoff: Handoff, *, max_chars: int = 2_000) -> str:
    """Render a handoff as the context block another agent receives."""
    lines: list[str] = []
    if handoff.from_agent or handoff.to_agent:
        lines.append(f"Handoff: {handoff.from_agent or '?'} -> {handoff.to_agent or '?'}")
    if handoff.task:
        lines.append(f"Task: {handoff.task}")
    if handoff.summary:
        lines.append(f"Done so far: {handoff.summary.strip()}")
    if handoff.next_steps:
        lines.append("Remaining:")
        lines.extend(
            f"  {index}. {step}" for index, step in enumerate(handoff.next_steps[:10], start=1)
        )
    if handoff.files:
        lines.append("Files: " + ", ".join(handoff.files[:10]))
    if handoff.blockers:
        lines.append("Blockers: " + "; ".join(handoff.blockers[:5]))
    lines.append(f"Confidence: {handoff.confidence:.0%}")
    return ellipsize("\n".join(lines), max_chars)


def chain_handoffs(handoffs: Sequence[Handoff]) -> dict[str, Any]:
    """Summarise a chain of handoffs, for team reports."""
    if not handoffs:
        return {"hops": 0}
    return {
        "hops": len(handoffs),
        "agents": [item.from_agent or "?" for item in handoffs] + [handoffs[-1].to_agent or "?"],
        "confidence": round(sum(item.confidence for item in handoffs) / len(handoffs), 3),
        "blockers": [item for handoff in handoffs for item in handoff.blockers],
        "open_steps": sum(len(handoff.next_steps) for handoff in handoffs),
        "files": sorted({path for handoff in handoffs for path in handoff.files}),
    }


__all__ = [
    "HANDOFF_PROMPT",
    "MAX_SUMMARY_CHARS",
    "Handoff",
    "build_handoff",
    "chain_handoffs",
    "handoff_block",
    "parse_handoff",
]
