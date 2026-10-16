"""Session storage: JSON records on disk, with search, prune and export.

Sessions live in one directory (``$XDG_DATA_HOME/magent/sessions`` by default),
one JSON file per session, so they are inspectable with any editor and survive
upgrades.  The store never keeps every record in memory: listing reads the small
metadata header of each file, and full histories are loaded only when a session
is resumed.  Secrets can be redacted on write, which is what makes transcripts
safe to share.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import SessionError
from ..logging_ import get_logger
from ..secrets import redact_text
from ..utils.files import atomic_write_text, ensure_dir, human_size
from ..utils.text import ellipsize
from ..utils.timeutil import format_relative
from .record import SessionRecord

LOGGER = get_logger("session.store")
"""Logger for session persistence problems."""

DEFAULT_DIRNAME = "sessions"
"""Directory name inside the MAgent data directory."""

MAX_TITLE_CHARS = 80
"""Titles are clipped to this length on save."""


@dataclass(slots=True)
class SessionInfo:
    """Lightweight summary of a stored session, read without the history."""

    id: str
    title: str = ""
    updated_at: float = 0.0
    created_at: float = 0.0
    messages: int = 0
    turns: int = 0
    tokens: int = 0
    cost: float = 0.0
    model: str = ""
    tags: list[str] = field(default_factory=list)
    preview: str = ""
    size: int = 0
    path: str = ""
    error: str = ""

    @property
    def age(self) -> str:
        """Human-readable age, e.g. ``3h ago``."""
        return format_relative(self.updated_at) if self.updated_at else "unknown"

    @classmethod
    def from_record(cls, record: SessionRecord, *, path: str | Path = "",
                    size: int = 0) -> SessionInfo:
        """Build the summary from a full record."""
        return cls(
            id=record.id, title=record.title, updated_at=record.updated_at,
            created_at=record.created_at, messages=len(record.messages),
            turns=len(record.turns), tokens=record.usage.total_tokens, cost=record.cost,
            model=record.model, tags=list(record.tags), preview=record.preview(80),
            size=size or len(record.to_json()), path=str(path),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SessionInfo:
        """Read just enough of a file to summarise it."""
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.debug("unreadable session %s: %s", target, exc)
            return cls(id=target.stem, title="(unreadable)", path=str(target),
                       size=target.stat().st_size if target.exists() else 0, error=str(exc))
        info = cls(
            id=str(raw.get("id", target.stem)),
            title=str(raw.get("title", "") or "")[:MAX_TITLE_CHARS],
            updated_at=float(raw.get("updated_at", 0.0) or 0.0),
            created_at=float(raw.get("created_at", 0.0) or 0.0),
            messages=int(raw.get("message_count", 0) or 0),
            turns=len(raw.get("turns", []) or []),
            tokens=int((raw.get("usage") or {}).get("total_tokens", 0) or 0),
            cost=float(raw.get("cost", 0.0) or 0.0),
            model=str(raw.get("model", "") or ""),
            tags=[str(item) for item in raw.get("tags", []) or []],
            size=target.stat().st_size,
            path=str(target),
        )
        turns = raw.get("turns") or []
        if turns and isinstance(turns[-1], dict):
            info.preview = ellipsize(str(turns[-1].get("prompt", "")).replace("\n", " "), 80)
        return info

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary for tables and JSON output."""
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at,
            "age": self.age,
            "messages": self.messages,
            "turns": self.turns,
            "tokens": self.tokens,
            "cost": self.cost,
            "model": self.model,
            "tags": list(self.tags),
            "preview": self.preview,
            "size": self.size,
            "path": self.path,
            "error": self.error,
        }

    def line(self) -> str:
        """One-line description for ``magent sessions list``."""
        return f"{self.age:>10}  {self.id[:16]:<16}  {self.messages:>4} msgs  {self.title[:40]}"


class SessionStore:
    """Persist and query session records."""

    def __init__(self, directory: str | Path, *, redact: bool = True, max_bytes: int = 0,
                 keep_last: int = 0, indent: int = 2) -> None:
        """Open a store.

        Args:
            directory: where session files live; created on first write
            redact: scrub secret-looking strings before writing
            max_bytes: refuse to write records larger than this, 0 for unlimited
            keep_last: prune older sessions on save, 0 to keep everything
            indent: JSON indentation
        """
        self.directory = Path(directory).expanduser()
        self.redact = bool(redact)
        self.max_bytes = max(0, int(max_bytes))
        self.keep_last = max(0, int(keep_last))
        self.indent = indent
        self.writes = 0
        self.reads = 0
        self.errors = 0

    @classmethod
    def from_config(cls, config: Any = None, *, directory: str | Path | None = None,
                    data_dir: str | Path | None = None) -> SessionStore:
        """Build a store from configuration.

        Args:
            config: a :class:`~magent.config.Config` or a session config
            directory: explicit override
            data_dir: base data directory when no session directory is set
        """
        session = getattr(config, "session", config)
        root = directory or getattr(session, "directory", "") or ""
        if not root:
            base = data_dir or getattr(config, "data_dir", "")
            root = Path(base or Path.home() / ".local/share/magent") / DEFAULT_DIRNAME
        return cls(root, redact=bool(getattr(session, "redact_secrets", True)),
                   max_bytes=int(getattr(session, "max_session_bytes", 0) or 0),
                   keep_last=int(getattr(session, "keep_last", 0) or 0))

    # -- paths ---------------------------------------------------------------

    def path_for(self, session_id: str) -> Path:
        """File path of one session."""
        safe = "".join(char for char in str(session_id) if char.isalnum() or char in "-_.")[:120]
        return self.directory / f"{safe or 'session'}.json"

    def ensure_directory(self) -> Path:
        """Create the session directory if needed."""
        ensure_dir(self.directory)
        return self.directory

    def files(self) -> list[Path]:
        """All session files, newest first."""
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("*.json"),
                      key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)

    # -- writing -------------------------------------------------------------

    def save(self, record: SessionRecord, *, redact: bool | None = None) -> Path:
        """Write one record, pruning old sessions when configured.

        Raises:
            SessionError: when the record exceeds ``max_bytes`` or writing fails
        """
        should_redact = self.redact if redact is None else redact
        payload = record.to_dict()
        if should_redact:
            payload = _redact_payload(payload)
        text = json.dumps(payload, indent=self.indent, default=str, ensure_ascii=False)
        if self.max_bytes and len(text.encode("utf-8")) > self.max_bytes:
            raise SessionError(
                f"session {record.id} is {human_size(len(text))}, over the "
                f"{human_size(self.max_bytes)} limit",
                details={"session_id": record.id, "bytes": len(text), "limit": self.max_bytes},
            )
        target = self.path_for(record.id)
        try:
            ensure_dir(target.parent)
            atomic_write_text(target, text)
        except OSError as exc:
            self.errors += 1
            raise SessionError(f"cannot write session {record.id}: {exc}",
                               details={"path": str(target)}) from exc
        self.writes += 1
        if self.keep_last:
            self.prune(self.keep_last)
        return target

    def save_many(self, records: Iterable[SessionRecord]) -> list[Path]:
        """Write several records."""
        return [self.save(record) for record in records]

    def delete(self, session_id: str) -> bool:
        """Delete one session file; returns whether anything was removed."""
        target = self.path_for(session_id)
        if not target.exists():
            return False
        try:
            target.unlink()
        except OSError as exc:
            self.errors += 1
            raise SessionError(f"cannot delete session {session_id}: {exc}",
                               details={"path": str(target)}) from exc
        return True

    def prune(self, keep_last: int) -> int:
        """Keep only the newest ``keep_last`` sessions; returns how many went.

        Ordering uses each record's own ``updated_at`` rather than the file's
        mtime: several sessions written inside the same second would otherwise
        tie and the wrong one could be deleted.
        """
        if keep_last <= 0:
            return 0
        infos = self.list(limit=100_000)
        if len(infos) <= keep_last:
            return 0
        removed = 0
        for info in infos[keep_last:]:
            target = Path(info.path) if info.path else self.path_for(info.id)
            try:
                target.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.debug("prune failed for %s: %s", target, exc)
        return removed

    def clear(self) -> int:
        """Delete every stored session."""
        removed = 0
        for path in self.files():
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    # -- reading -------------------------------------------------------------

    def load(self, session_id: str) -> SessionRecord | None:
        """Load one session, or ``None`` when it is unknown."""
        target = self.path_for(session_id)
        if not target.exists():
            match = self._fuzzy_path(session_id)
            if match is None:
                return None
            target = match
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.errors += 1
            raise SessionError(f"cannot read session {session_id}: {exc}",
                               details={"path": str(target)}) from exc
        self.reads += 1
        record = SessionRecord.from_dict(raw)
        if not record.id:
            record.id = target.stem
        return record

    def require(self, session_id: str) -> SessionRecord:
        """Load one session, raising when it does not exist."""
        record = self.load(session_id)
        if record is None:
            raise SessionError(f"unknown session {session_id!r}",
                               details={"known": [info.id for info in self.list(limit=10)]})
        return record

    def _fuzzy_path(self, session_id: str) -> Path | None:
        """Resolve a session id prefix to a file."""
        needle = str(session_id).strip()
        if len(needle) < 4:
            return None
        matches = [path for path in self.files() if path.stem.startswith(needle)]
        if len(matches) == 1:
            return matches[0]
        return None

    def list(self, *, limit: int = 25, tag: str = "", query: str = "") -> list[SessionInfo]:
        """Summarise stored sessions, newest first.

        Args:
            limit: maximum entries returned
            tag: only sessions carrying this tag
            query: substring matched against id, title and preview
        """
        infos = [SessionInfo.from_file(path) for path in self.files()]
        if tag:
            infos = [info for info in infos if tag in info.tags]
        if query:
            needle = query.lower()
            infos = [info for info in infos
                     if needle in info.id.lower() or needle in info.title.lower()
                     or needle in info.preview.lower()]
        infos.sort(key=lambda info: -(info.updated_at or 0.0))
        return infos[: max(1, limit)]

    def latest(self) -> SessionRecord | None:
        """Load the most recently updated session."""
        infos = self.list(limit=1)
        if not infos:
            return None
        return self.load(infos[0].id)

    def search(self, query: str, *, limit: int = 10) -> list[tuple[SessionInfo, list[str]]]:
        """Full-text search across stored sessions.

        Returns pairs of ``(info, matching lines)`` so the CLI can show context.
        """
        needle = query.lower().strip()
        if not needle:
            return []
        results: list[tuple[SessionInfo, list[str]]] = []
        for path in self.files():
            info = SessionInfo.from_file(path)
            record = self.load(info.id)
            if record is None:
                continue
            hits: list[str] = []
            for message in record.messages:
                text = message.text()
                if needle in text.lower():
                    hits.append(ellipsize(text.replace("\n", " "), 120))
            if needle in record.summary.lower():
                hits.append(ellipsize(record.summary.replace("\n", " "), 120))
            if hits:
                results.append((info, hits[:5]))
            if len(results) >= limit:
                break
        return results

    def export(self, session_id: str, path: str | Path, *, fmt: str = "json",
               redact: bool | None = None) -> Path:
        """Export one session to a file in ``json`` or ``markdown``."""
        from .transcript import render_transcript

        record = self.require(session_id)
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        if fmt == "json":
            payload = record.to_dict()
            if self.redact if redact is None else redact:
                payload = _redact_payload(payload)
            text = json.dumps(payload, indent=self.indent, default=str, ensure_ascii=False)
        else:
            text = render_transcript(record, fmt=fmt, redact=bool(self.redact if redact is None else redact))
        return atomic_write_text(target, text)

    def import_file(self, path: str | Path, *, session_id: str = "") -> SessionRecord:
        """Import a session from a JSON file."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise SessionError(f"no such session file: {source}", details={"path": str(source)})
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SessionError(f"invalid session JSON in {source}: {exc}",
                               details={"path": str(source)}) from exc
        record = SessionRecord.from_dict(raw)
        if session_id:
            record.id = session_id
        if not record.id:
            record.id = source.stem
        self.save(record)
        return record

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics across stored sessions."""
        infos = self.list(limit=10_000)
        return {
            "directory": str(self.directory),
            "sessions": len(infos),
            "messages": sum(info.messages for info in infos),
            "turns": sum(info.turns for info in infos),
            "tokens": sum(info.tokens for info in infos),
            "cost": round(sum(info.cost for info in infos), 6),
            "bytes": sum(info.size for info in infos),
            "size": human_size(sum(info.size for info in infos)),
            "newest": infos[0].updated_at if infos else 0.0,
            "writes": self.writes,
            "reads": self.reads,
            "errors": self.errors,
            "redact": self.redact,
            "keep_last": self.keep_last,
        }

    def describe(self) -> str:
        """Human-readable one-paragraph description of the store."""
        info = self.stats()
        return (f"{info['sessions']} sessions in {info['directory']} "
                f"({info['size']}, {info['messages']} messages, {info['tokens']} tokens)")

    def __len__(self) -> int:
        """Number of stored sessions."""
        return len(self.files())

    def __contains__(self, session_id: object) -> bool:
        """True when a session id exists."""
        return self.path_for(str(session_id)).exists()

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"SessionStore(directory={str(self.directory)!r}, redact={self.redact})"


def _redact_payload(payload: Any) -> Any:
    """Recursively redact secret-looking strings in a serialised record."""
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, dict):
        return {key: _redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


def redact_record(record: SessionRecord) -> SessionRecord:
    """Return a copy of a record with secrets redacted."""
    return SessionRecord.from_dict(_redact_payload(record.to_dict()))


def sorted_by_recency(records: Sequence[SessionRecord]) -> list[SessionRecord]:
    """Sort records newest-first."""
    return sorted(records, key=lambda item: -(item.updated_at or item.created_at))


def session_dir(config: Any = None) -> Path:
    """Resolve the session directory from configuration."""
    return SessionStore.from_config(config).directory


def touched_within(directory: str | Path, *, hours: float = 24.0) -> list[SessionInfo]:
    """Sessions updated within the last ``hours``."""
    cutoff = time.time() - hours * 3_600
    store = SessionStore(directory)
    return [info for info in store.list(limit=10_000) if info.updated_at >= cutoff]


__all__ = [
    "DEFAULT_DIRNAME",
    "MAX_TITLE_CHARS",
    "SessionInfo",
    "SessionStore",
    "redact_record",
    "session_dir",
    "sorted_by_recency",
    "touched_within",
]
