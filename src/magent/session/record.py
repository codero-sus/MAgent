"""Session records: the durable form of a conversation.

A :class:`SessionRecord` is what ``magent sessions`` lists, what ``--resume``
loads and what the web UI replays.  It stores the message history plus the
bookkeeping that makes a session useful later - title, model, workspace, token
and cost totals, tags, and the rolling summary written by short-term memory.
Records are JSON on disk, so they can be diffed, grepped and shared.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..types import Message, Usage, messages_from_dicts
from ..utils.ids import session_id as new_session_id
from ..utils.text import ellipsize, slugify

RECORD_VERSION = 1
"""Schema version written into every record."""

TITLE_MAX_WORDS = 8
"""Words kept when a title is derived from the first user message."""


@dataclass(slots=True)
class Turn:
    """One exchange: a user message and the assistant's reply."""

    index: int
    prompt: str
    answer: str = ""
    started_at: float = 0.0
    seconds: float = 0.0
    tools: list[str] = field(default_factory=list)
    tokens: int = 0
    ok: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the turn."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "answer": self.answer,
            "started_at": self.started_at,
            "seconds": round(self.seconds, 4),
            "tools": list(self.tools),
            "tokens": self.tokens,
            "ok": self.ok,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Turn:
        """Rebuild a turn from JSON."""
        return cls(
            index=int(raw.get("index", 0) or 0),
            prompt=str(raw.get("prompt", "") or ""),
            answer=str(raw.get("answer", "") or ""),
            started_at=float(raw.get("started_at", 0.0) or 0.0),
            seconds=float(raw.get("seconds", 0.0) or 0.0),
            tools=[str(item) for item in raw.get("tools", []) or []],
            tokens=int(raw.get("tokens", 0) or 0),
            ok=bool(raw.get("ok", True)),
            error=str(raw.get("error", "") or ""),
            metadata=dict(raw.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class SessionRecord:
    """A persisted conversation with its metadata."""

    id: str = ""
    title: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    messages: list[Message] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    model: str = ""
    workspace: str = ""
    tags: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    version: int = RECORD_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in the identity and timestamps when the caller omitted them."""
        if not self.id:
            self.id = new_session_id()
        now = time.time()
        self.created_at = self.created_at or now
        self.updated_at = self.updated_at or now

    # -- editing -------------------------------------------------------------

    def touch(self) -> None:
        """Mark the record as just modified."""
        self.updated_at = time.time()

    def add_message(self, message: Message) -> Message:
        """Append one message and refresh the timestamps."""
        self.messages.append(message)
        self.touch()
        if not self.title and message.role == "user":
            self.title = derive_title(message.text())
        return message

    def add_messages(self, messages: Iterable[Message]) -> int:
        """Append many messages."""
        added = 0
        for message in messages:
            self.add_message(message)
            added += 1
        return added

    def add_turn(self, prompt: str, answer: str, **kwargs: Any) -> Turn:
        """Record one exchange."""
        turn = Turn(index=len(self.turns), prompt=prompt, answer=answer,
                    started_at=kwargs.pop("started_at", time.time()), **kwargs)
        self.turns.append(turn)
        self.touch()
        return turn

    def set_title(self, title: str) -> str:
        """Set an explicit title."""
        self.title = (title or "").strip()[:160]
        self.touch()
        return self.title

    def tag(self, *tags: str) -> list[str]:
        """Add tags, ignoring duplicates."""
        for item in tags:
            name = str(item).strip()
            if name and name not in self.tags:
                self.tags.append(name)
        self.touch()
        return list(self.tags)

    def add_usage(self, usage: Usage | None, cost: float = 0.0) -> Usage:
        """Fold one run's usage into the session totals."""
        if usage is not None:
            self.usage = self.usage.merge(usage)
        self.cost = round(self.cost + float(cost or 0.0), 6)
        self.touch()
        return self.usage

    # -- views ---------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        """Number of recorded exchanges."""
        return len(self.turns)

    @property
    def message_count(self) -> int:
        """Number of stored messages."""
        return len(self.messages)

    @property
    def tokens(self) -> int:
        """Total tokens charged to this session."""
        return self.usage.total_tokens

    @property
    def age_seconds(self) -> float:
        """Seconds since the session was last updated."""
        return max(0.0, time.time() - self.updated_at)

    @property
    def slug(self) -> str:
        """Filesystem-safe form of the title."""
        return slugify(self.title or self.id)[:60] or self.id

    def last_prompt(self) -> str:
        """Most recent user message, or an empty string."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.text()
        return ""

    def preview(self, limit: int = 100) -> str:
        """One-line preview used by ``magent sessions list``."""
        text = self.last_prompt() or self.summary or (self.turns[-1].answer if self.turns else "")
        return ellipsize(text.replace("\n", " "), limit)

    def tool_usage(self) -> dict[str, int]:
        """How often each tool was called across the session."""
        counts: dict[str, int] = {}
        for turn in self.turns:
            for name in turn.tools:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def errors(self) -> list[str]:
        """Errors recorded by failed turns."""
        return [turn.error for turn in self.turns if turn.error]

    # -- (de)serialisation ---------------------------------------------------

    def to_dict(self, *, include_messages: bool = True) -> dict[str, Any]:
        """Serialise the record to JSON-compatible data."""
        payload: dict[str, Any] = {
            "version": self.version,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "model": self.model,
            "workspace": self.workspace,
            "tags": list(self.tags),
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "turns": [turn.to_dict() for turn in self.turns],
            "message_count": len(self.messages),
            "metadata": dict(self.metadata),
        }
        if include_messages:
            payload["messages"] = [message.to_dict() for message in self.messages]
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SessionRecord:
        """Rebuild a record from JSON, tolerating missing keys."""
        record = cls(
            id=str(raw.get("id", "") or ""),
            title=str(raw.get("title", "") or ""),
            created_at=float(raw.get("created_at", 0.0) or 0.0),
            updated_at=float(raw.get("updated_at", 0.0) or 0.0),
            summary=str(raw.get("summary", "") or ""),
            model=str(raw.get("model", "") or ""),
            workspace=str(raw.get("workspace", "") or ""),
            tags=[str(item) for item in raw.get("tags", []) or []],
            cost=float(raw.get("cost", 0.0) or 0.0),
            version=int(raw.get("version", RECORD_VERSION) or RECORD_VERSION),
            metadata=dict(raw.get("metadata", {}) or {}),
        )
        usage = raw.get("usage")
        if isinstance(usage, dict):
            record.usage = Usage.from_dict(usage)
        record.messages = messages_from_dicts(list(raw.get("messages", []) or []))
        record.turns = [Turn.from_dict(item) for item in raw.get("turns", []) or []
                        if isinstance(item, dict)]
        return record

    def to_json(self, *, indent: int = 2, include_messages: bool = True) -> str:
        """Serialise the record as a JSON document."""
        return json.dumps(self.to_dict(include_messages=include_messages), indent=indent,
                          default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> SessionRecord:
        """Parse a JSON document into a record."""
        return cls.from_dict(json.loads(text))

    def filename(self, *, extension: str = ".json") -> str:
        """File name used when storing the record."""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.updated_at or time.time()))
        return f"{stamp}-{self.slug or self.id}{extension}"

    def path_in(self, directory: str | Path) -> Path:
        """Full path of the record inside a session directory."""
        return Path(directory).expanduser() / f"{self.id}.json"

    def __len__(self) -> int:
        """Number of messages."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (f"SessionRecord(id={self.id!r}, title={self.title!r}, "
                f"messages={len(self.messages)}, turns={len(self.turns)})")


def derive_title(text: str, *, max_words: int = TITLE_MAX_WORDS) -> str:
    """Derive a short session title from the first user message."""
    cleaned = " ".join((text or "").replace("\n", " ").split())
    if not cleaned:
        return "Untitled session"
    words = cleaned.split()[:max_words]
    title = " ".join(words)
    if len(words) == max_words and len(cleaned.split()) > max_words:
        title += "…"
    return title[:80]


def merge_records(records: Sequence[SessionRecord]) -> SessionRecord:
    """Merge several records into one, keeping the newest metadata."""
    if not records:
        return SessionRecord()
    base = SessionRecord.from_dict(records[0].to_dict())
    for other in records[1:]:
        base.add_messages(other.messages)
        base.turns.extend(other.turns)
        base.usage = base.usage.merge(other.usage)
        base.cost = round(base.cost + other.cost, 6)
        base.tag(*other.tags)
        base.updated_at = max(base.updated_at, other.updated_at)
        base.created_at = min(base.created_at, other.created_at)
        if not base.summary and other.summary:
            base.summary = other.summary
    return base


__all__ = [
    "RECORD_VERSION",
    "TITLE_MAX_WORDS",
    "SessionRecord",
    "Turn",
    "derive_title",
    "merge_records",
]
