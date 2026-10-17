"""One durable memory entry.

Entries are plain data with a stable identifier, so they can be appended to a
JSONL file, reloaded, re-embedded, and ranked without any database. ``score``
is transient: it is filled in by the retrieval pass and never persisted.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..types import new_id
from ..utils.hashing import content_hash
from ..utils.text import collapse_whitespace

DEFAULT_KIND = "note"
"""Entry kind used when a caller does not specify one."""

KINDS: tuple[str, ...] = (
    "note",
    "fact",
    "decision",
    "preference",
    "task",
    "outcome",
    "entity",
    "episode",
    "skill",
    "error",
)
"""Well-known entry kinds; arbitrary strings are still accepted."""


@dataclass
class MemoryEntry:
    """A single stored memory with provenance and retrieval metadata."""

    content: str
    kind: str = DEFAULT_KIND
    id: str = field(default_factory=lambda: new_id("mem"))
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = 0.0
    access_count: int = 0
    embedding: list[float] = field(default_factory=list)
    score: float = 0.0

    def __post_init__(self) -> None:
        """Normalize the content, tags, and importance into canonical form."""
        self.content = collapse_whitespace(str(self.content or "")).strip()
        self.kind = str(self.kind or DEFAULT_KIND).strip().lower()
        self.tags = [str(tag).strip().lower() for tag in self.tags if str(tag).strip()]
        self.tags = list(dict.fromkeys(self.tags))
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.embedding = [float(value) for value in (self.embedding or [])]

    # -- identity ------------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Return a content hash used for deduplication."""
        return content_hash("memory", self.kind, self.content)

    def age_seconds(self, *, now: float | None = None) -> float:
        """Return how old the entry is, in seconds."""
        return max(0.0, (now if now is not None else time.time()) - self.created_at)

    def idle_seconds(self, *, now: float | None = None) -> float:
        """Return how long it has been since the entry was last used."""
        reference = self.last_accessed or self.created_at
        return max(0.0, (now if now is not None else time.time()) - reference)

    def has_tag(self, tag: str) -> bool:
        """Report whether the entry carries a tag."""
        return str(tag).strip().lower() in self.tags

    def matches_tags(self, tags: Iterable[str]) -> bool:
        """Report whether the entry carries at least one of ``tags``."""
        wanted = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        return not wanted or bool(wanted & set(self.tags))

    # -- mutation ------------------------------------------------------------

    def touch(self, *, now: float | None = None) -> None:
        """Record an access, which feeds the recency term of ranking."""
        self.access_count += 1
        self.last_accessed = now if now is not None else time.time()

    def update(
        self,
        content: str | None = None,
        *,
        importance: float | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the mutable fields and stamp ``updated_at``."""
        if content is not None:
            self.content = collapse_whitespace(str(content)).strip()
            self.embedding = []
        if importance is not None:
            self.importance = max(0.0, min(1.0, float(importance)))
        if tags is not None:
            self.tags = list(
                dict.fromkeys(str(tag).strip().lower() for tag in tags if str(tag).strip())
            )
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = time.time()

    # -- serialization -------------------------------------------------------

    def to_dict(self, *, include_embedding: bool = True) -> dict[str, Any]:
        """Serialize the entry for JSONL storage."""
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "tags": list(self.tags),
            "importance": self.importance,
            "source": self.source,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "score": round(self.score, 6),
            "fingerprint": self.fingerprint,
        }
        if include_embedding and self.embedding:
            data["embedding"] = [round(value, 6) for value in self.embedding]
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MemoryEntry:
        """Rebuild an entry from a persisted dictionary."""
        return cls(
            id=str(raw.get("id") or new_id("mem")),
            content=str(raw.get("content", "")),
            kind=str(raw.get("kind", DEFAULT_KIND)),
            tags=list(raw.get("tags") or []),
            importance=float(raw.get("importance", 0.5)),
            source=str(raw.get("source", "")),
            metadata=dict(raw.get("metadata") or {}),
            created_at=float(raw.get("created_at") or time.time()),
            updated_at=float(raw.get("updated_at") or time.time()),
            last_accessed=float(raw.get("last_accessed") or 0.0),
            access_count=int(raw.get("access_count") or 0),
            embedding=[float(value) for value in (raw.get("embedding") or [])],
        )

    def to_json(self, *, include_embedding: bool = True) -> str:
        """Serialize to a single JSON line."""
        import json

        return json.dumps(
            self.to_dict(include_embedding=include_embedding), ensure_ascii=False, default=str
        )

    def summary_line(self, *, limit: int = 160) -> str:
        """Render the entry as one line for prompts and CLI output."""
        from ..utils.text import ellipsize

        tags = f" #{' #'.join(self.tags)}" if self.tags else ""
        return f"[{self.kind}] {ellipsize(self.content, limit)}{tags}"

    def __repr__(self) -> str:
        """Show the identity, kind, and a content preview."""
        return f"<MemoryEntry {self.id} kind={self.kind!r} content={self.content[:40]!r}>"


__all__ = ["DEFAULT_KIND", "KINDS", "MemoryEntry"]
