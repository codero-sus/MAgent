"""Transcripts: an ordered record of what a conversation did.

A transcript is more than the message list. It carries the tool calls, their
results, timings, token usage, and the model that produced each turn, because
that is what makes a past run reviewable. Storing only the messages would
answer "what was said" but not "why".

Three renderings are provided: ``jsonl`` for machines and resume, ``markdown``
for humans and issue reports, and ``text`` for terminals. Redaction happens
once, at write time, so a secret that appeared in a tool result never reaches
disk in any format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..secrets import redact_mapping, redact_text
from ..types import Message, now_ts
from ..utils.timeutil import format_duration, format_timestamp

FORMATS: tuple[str, ...] = ("jsonl", "markdown", "text", "json")
"""Transcript renderings this module can produce."""

ROLE_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "assistant": "MAgent",
    "tool": "Tool",
}
"""Human-readable names for each message role."""


@dataclass(slots=True)
class TranscriptEntry:
    """One turn, with everything needed to replay or audit it."""

    role: str
    content: str = ""
    timestamp: float = field(default_factory=now_ts)
    model: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_name: str = ""
    tool_call_id: str = ""
    tokens: int = 0
    duration_ms: float = 0.0
    step: int = 0
    run_id: str = ""
    ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: Message, *, step: int = 0, run_id: str = "",
                     model: str = "", duration_ms: float = 0.0,
                     timestamp: float | None = None) -> TranscriptEntry:
        """Build an entry from a :class:`~magent.types.Message`.

        Args:
            message: the turn to record
            step: agent loop iteration this turn belongs to
            run_id: identifier of the run that produced the turn
            model: model that generated an assistant turn
            duration_ms: generation time for an assistant turn
            timestamp: explicit time, defaulting to now
        """
        calls = [call.to_dict() if hasattr(call, "to_dict") else dict(call)
                 for call in (message.tool_calls or [])]
        return cls(
            role=message.role,
            content=message.content or "",
            timestamp=timestamp if timestamp is not None else now_ts(),
            model=model if message.role == "assistant" else "",
            tool_calls=calls,
            tool_name=getattr(message, "name", "") or "",
            tool_call_id=getattr(message, "tool_call_id", "") or "",
            duration_ms=duration_ms,
            step=step,
            run_id=run_id,
        )

    @classmethod
    def from_tool_result(cls, result: Any, *, step: int = 0, run_id: str = "") -> TranscriptEntry:
        """Build an entry from a :class:`~magent.types.ToolResult`."""
        content = getattr(result, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, default=str, ensure_ascii=False)
        return cls(
            role="tool",
            content=content,
            tool_name=getattr(result, "name", "") or "",
            tool_call_id=getattr(result, "call_id", "") or "",
            duration_ms=float(getattr(result, "duration_ms", 0.0) or 0.0),
            ok=bool(getattr(result, "ok", True)),
            step=step,
            run_id=run_id,
            metadata={
                "error_code": getattr(result, "error_code", "") or "",
                "truncated": bool(getattr(result, "truncated", False)),
            },
        )

    def redacted(self, *, extra_secrets: Sequence[str] = ()) -> TranscriptEntry:
        """Return a copy with secrets removed from content and metadata."""
        secrets = list(extra_secrets)
        return TranscriptEntry(
            role=self.role,
            content=redact_text(self.content, extra_secrets=secrets),
            timestamp=self.timestamp,
            model=self.model,
            tool_calls=[redact_mapping(call, extra_secrets=secrets) for call in self.tool_calls],
            tool_name=self.tool_name,
            tool_call_id=self.tool_call_id,
            tokens=self.tokens,
            duration_ms=self.duration_ms,
            step=self.step,
            run_id=self.run_id,
            ok=self.ok,
            metadata=redact_mapping(self.metadata, extra_secrets=secrets),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form written to disk."""
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": round(self.timestamp, 3),
        }
        if self.model:
            payload["model"] = self.model
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_name:
            payload["tool"] = self.tool_name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tokens:
            payload["tokens"] = self.tokens
        if self.duration_ms:
            payload["duration_ms"] = round(self.duration_ms, 1)
        if self.step:
            payload["step"] = self.step
        if self.run_id:
            payload["run_id"] = self.run_id
        if not self.ok:
            payload["ok"] = False
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptEntry:
        """Rebuild an entry from its stored dict."""
        metadata = dict(data.get("metadata") or {})
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "") or ""),
            timestamp=float(data.get("timestamp", 0.0) or now_ts()),
            model=str(data.get("model", "") or ""),
            tool_calls=list(data.get("tool_calls") or []),
            tool_name=str(data.get("tool", "") or data.get("tool_name", "") or ""),
            tool_call_id=str(data.get("tool_call_id", "") or ""),
            tokens=int(data.get("tokens", 0) or 0),
            duration_ms=float(data.get("duration_ms", 0.0) or 0.0),
            step=int(data.get("step", 0) or 0),
            run_id=str(data.get("run_id", "") or ""),
            ok=bool(data.get("ok", True)),
            metadata=metadata,
        )

    def to_message(self) -> Message:
        """Return the entry as a :class:`~magent.types.Message` for replay."""
        return Message(role=self.role, content=self.content, name=self.tool_name or None,
                       tool_call_id=self.tool_call_id or None)

    def label(self) -> str:
        """Return the display label for this entry's role."""
        return ROLE_LABELS.get(self.role, self.role.title())

    @property
    def chars(self) -> int:
        """Length of the recorded content."""
        return len(self.content)


@dataclass(slots=True)
class Transcript:
    """An ordered list of entries plus run-level metadata."""

    entries: list[TranscriptEntry] = field(default_factory=list)
    title: str = ""
    started_at: float = field(default_factory=now_ts)
    finished_at: float = 0.0
    model: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, entry: TranscriptEntry) -> TranscriptEntry:
        """Append one entry and return it."""
        self.entries.append(entry)
        return entry

    def add_message(self, message: Message, **kwargs: Any) -> TranscriptEntry:
        """Append a message, converting it to an entry."""
        return self.add(TranscriptEntry.from_message(message, **kwargs))

    def extend(self, messages: Iterable[Message], **kwargs: Any) -> int:
        """Append many messages, returning how many were added."""
        count = 0
        for message in messages:
            self.add_message(message, **kwargs)
            count += 1
        return count

    def close(self, *, finished_at: float | None = None) -> Transcript:
        """Mark the transcript finished."""
        self.finished_at = finished_at or now_ts()
        return self

    def redacted(self, *, extra_secrets: Sequence[str] = ()) -> Transcript:
        """Return a copy with every entry redacted."""
        return Transcript(
            entries=[entry.redacted(extra_secrets=extra_secrets) for entry in self.entries],
            title=redact_text(self.title, extra_secrets=list(extra_secrets)),
            started_at=self.started_at,
            finished_at=self.finished_at,
            model=self.model,
            run_id=self.run_id,
            metadata=redact_mapping(self.metadata, extra_secrets=list(extra_secrets)),
        )

    def messages(self, *, include_tools: bool = True) -> list[Message]:
        """Return the transcript as replayable messages."""
        return [entry.to_message() for entry in self.entries
                if include_tools or entry.role != "tool"]

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration, or the sum of turn durations if unfinished."""
        if self.finished_at:
            return max(0.0, (self.finished_at - self.started_at) * 1000.0)
        return sum(entry.duration_ms for entry in self.entries)

    @property
    def tokens(self) -> int:
        """Total tokens recorded across entries."""
        return sum(entry.tokens for entry in self.entries)

    def tool_counts(self) -> dict[str, int]:
        """Count tool invocations by name."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.role == "tool" and entry.tool_name:
                counts[entry.tool_name] = counts.get(entry.tool_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def failures(self) -> list[TranscriptEntry]:
        """Return entries that recorded an error."""
        return [entry for entry in self.entries if not entry.ok]

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics for display and indexing."""
        roles: dict[str, int] = {}
        for entry in self.entries:
            roles[entry.role] = roles.get(entry.role, 0) + 1
        return {
            "entries": len(self.entries),
            "roles": roles,
            "chars": sum(entry.chars for entry in self.entries),
            "tokens": self.tokens,
            "duration_ms": round(self.duration_ms, 1),
            "tools": self.tool_counts(),
            "failures": len(self.failures()),
            "model": self.model,
            "run_id": self.run_id,
            "title": self.title,
            "started_at": self.started_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full JSON form."""
        return {
            "title": self.title,
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "model": self.model,
            "run_id": self.run_id,
            "metadata": self.metadata,
            "summary": self.summary(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Rebuild a transcript from :meth:`to_dict` output."""
        return cls(
            entries=[TranscriptEntry.from_dict(item) for item in (data.get("entries") or [])],
            title=str(data.get("title", "") or ""),
            started_at=float(data.get("started_at", 0.0) or now_ts()),
            finished_at=float(data.get("finished_at") or 0.0),
            model=str(data.get("model", "") or ""),
            run_id=str(data.get("run_id", "") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def render(self, fmt: str = "markdown", *, width: int = 100,
               include_metadata: bool = False) -> str:
        """Render the transcript in one of :data:`FORMATS`.

        Args:
            fmt: ``jsonl``, ``json``, ``markdown``, or ``text``
            width: wrap column for the text and markdown renderings
            include_metadata: show per-entry metadata dictionaries
        """
        active = fmt if fmt in FORMATS else "markdown"
        if active == "jsonl":
            return render_jsonl(self)
        if active == "json":
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if active == "text":
            return render_text(self, width=width, include_metadata=include_metadata)
        return render_markdown(self, width=width, include_metadata=include_metadata)

    def __len__(self) -> int:
        """Number of entries."""
        return len(self.entries)

    def __iter__(self) -> Any:
        """Iterate entries."""
        return iter(self.entries)


def render_jsonl(transcript: Transcript) -> str:
    """Render one JSON object per line, the on-disk resume format."""
    header = {"type": "meta", "title": transcript.title, "model": transcript.model,
              "run_id": transcript.run_id, "started_at": round(transcript.started_at, 3)}
    lines = [json.dumps(header, ensure_ascii=False)]
    for entry in transcript.entries:
        payload = entry.to_dict()
        payload["type"] = "entry"
        lines.append(json.dumps(payload, ensure_ascii=False))
    if transcript.finished_at:
        lines.append(json.dumps({"type": "end", "summary": transcript.summary()},
                                ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n"


def parse_jsonl(text: str) -> Transcript:
    """Parse :func:`render_jsonl` output back into a transcript.

    Malformed lines are skipped rather than fatal: a partially written file
    from an interrupted run should still resume.
    """
    transcript = Transcript()
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = payload.get("type")
        if kind == "meta":
            transcript.title = str(payload.get("title", "") or "")
            transcript.model = str(payload.get("model", "") or "")
            transcript.run_id = str(payload.get("run_id", "") or "")
            transcript.started_at = float(payload.get("started_at", 0.0) or now_ts())
        elif kind == "end":
            summary = payload.get("summary") or {}
            transcript.finished_at = float(summary.get("started_at", 0.0) or now_ts())
            transcript.metadata["summary"] = summary
        else:
            transcript.entries.append(TranscriptEntry.from_dict(payload))
    return transcript


def render_markdown(transcript: Transcript, *, width: int = 100,
                    include_metadata: bool = False) -> str:
    """Render a transcript as readable Markdown."""
    lines: list[str] = []
    title = transcript.title or "MAgent session"
    lines.append(f"# {title}")
    lines.append("")
    stats = transcript.summary()
    lines.append(f"- Started: {format_timestamp(transcript.started_at)}")
    if transcript.finished_at:
        lines.append(f"- Duration: {format_duration(transcript.duration_ms / 1000.0)}")
    if transcript.model:
        lines.append(f"- Model: `{transcript.model}`")
    lines.append(f"- Turns: {stats['entries']} | tools: "
                 f"{', '.join(f'{name} x{count}' for name, count in stats['tools'].items()) or 'none'}")
    if stats["failures"]:
        lines.append(f"- Failures: {stats['failures']}")
    lines.append("")
    for entry in transcript.entries:
        stamp = format_timestamp(entry.timestamp, fmt="%H:%M:%S")
        suffix = f" · `{entry.tool_name}`" if entry.tool_name else ""
        timing = f" · {format_duration(entry.duration_ms / 1000.0)}" if entry.duration_ms else ""
        marker = "" if entry.ok else " · **failed**"
        lines.append(f"### {entry.label()}{suffix} — {stamp}{timing}{marker}")
        lines.append("")
        body = _wrap(entry.content.strip(), width) if entry.content.strip() else "_no content_"
        lines.append(body)
        for call in entry.tool_calls:
            name = call.get("function", {}).get("name") if isinstance(call, dict) else ""
            arguments = call.get("function", {}).get("arguments") if isinstance(call, dict) else ""
            lines.append("")
            lines.append(f"- calls `{name or 'tool'}` with `{_short(str(arguments), 200)}`")
        if include_metadata and entry.metadata:
            lines.append("")
            lines.append(f"- metadata: `{json.dumps(entry.metadata, ensure_ascii=False, default=str)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_text(transcript: Transcript, *, width: int = 100,
                include_metadata: bool = False) -> str:
    """Render a transcript as plain text for a terminal."""
    lines: list[str] = []
    rule = "-" * min(width, 72)
    if transcript.title:
        lines.append(transcript.title)
        lines.append(rule)
    for entry in transcript.entries:
        stamp = format_timestamp(entry.timestamp, fmt="%H:%M:%S")
        label = entry.label() + (f" ({entry.tool_name})" if entry.tool_name else "")
        flag = "" if entry.ok else " [failed]"
        lines.append(f"[{stamp}] {label}{flag}")
        if entry.content.strip():
            lines.append(_indent(_wrap(entry.content.strip(), width - 2), "  "))
        for call in entry.tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            lines.append(f"  -> {function.get('name', 'tool')} {_short(str(function.get('arguments', '')), 160)}")
        if include_metadata and entry.metadata:
            lines.append(f"  .. {json.dumps(entry.metadata, ensure_ascii=False, default=str)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _wrap(text: str, width: int) -> str:
    """Wrap text to a column width, preserving existing line breaks."""
    limit = max(20, int(width))
    out: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            out.append("")
            continue
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= limit:
                line = f"{line} {word}".strip()
            else:
                out.append(line)
                line = word
        out.append(line)
    return "\n".join(out)


def _indent(text: str, prefix: str) -> str:
    """Prefix every line of ``text``."""
    return "\n".join(prefix + line for line in text.split("\n"))


def _short(text: str, limit: int) -> str:
    """Truncate to one line no longer than ``limit``."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: max(0, limit - 1)] + "…"


__all__ = [
    "FORMATS",
    "ROLE_LABELS",
    "Transcript",
    "TranscriptEntry",
    "parse_jsonl",
    "render_jsonl",
    "render_markdown",
    "render_text",
]
