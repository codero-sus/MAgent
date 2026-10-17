"""Sessions: transcripts on disk, resumable, pruned, and size-capped.

A session is one conversation's durable record. The store appends rather than
rewrites, because an agent run can be killed mid-turn and a half-written file
must still be readable - JSONL gives that for free, and it means saving turn
1000 costs one line instead of re-serializing 999.

Three protections matter in practice and are all enforced here:

* secrets are redacted before they reach disk,
* ``keep_last`` prunes old sessions so a long-lived install does not grow
  without bound,
* ``max_session_bytes`` stops a single runaway session (a tool looping on a
  huge file, say) from filling the disk.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..errors import SessionError
from ..logging_ import get_logger
from ..secrets import redact_text
from ..types import Message, now_ts
from ..utils.files import atomic_write_text, ensure_dir, human_size, read_text_safe, safe_filename
from ..utils.ids import session_id as new_session_id
from ..utils.timeutil import format_timestamp
from .transcript import Transcript, TranscriptEntry, parse_jsonl, render_jsonl

LOGGER = get_logger("session")
"""Logger for session persistence."""

DEFAULT_DIRECTORY = "sessions"
"""Directory name under the MAgent data dir when none is configured."""

SESSION_SUFFIX = ".jsonl"
"""On-disk extension for a session transcript."""

INDEX_FILE = "index.json"
"""Lightweight index of known sessions, rebuilt from disk when stale."""

MIN_SESSION_BYTES = 320
"""Smallest transcript that still resumes: a header plus one entry and marker.

A size cap below this cannot be met without writing a session that holds
nothing, so the cap is raised to the floor and the adjustment is reported in
:meth:`Session.stats` rather than silently overshooting.
"""

CAP_MARKER = "\n[truncated: session size cap]"
"""Appended to the newest entry when the size cap forced content truncation."""

DEFAULT_KEEP_LAST = 50
"""Sessions retained by pruning when the config does not say."""


@dataclass(slots=True)
class SessionInfo:
    """Metadata about a stored session, without loading its transcript."""

    session_id: str
    title: str = ""
    path: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    entries: int = 0
    bytes: int = 0
    model: str = ""
    run_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Seconds since the session was last written."""
        return max(0.0, now_ts() - (self.updated_at or self.created_at or now_ts()))

    def to_dict(self) -> dict[str, Any]:
        """Return the index-record form."""
        return {
            "id": self.session_id,
            "title": self.title,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "entries": self.entries,
            "bytes": self.bytes,
            "model": self.model,
            "run_id": self.run_id,
            "tags": list(self.tags),
        }

    def describe(self) -> str:
        """Return a one-line summary for listings."""
        when = format_timestamp(self.updated_at or self.created_at, fmt="%Y-%m-%d %H:%M")
        title = self.title or "(untitled)"
        return (f"{self.session_id[:12]:<12} {when}  {self.entries:>4} turns  "
                f"{human_size(self.bytes):>8}  {title[:48]}")

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: str = "") -> SessionInfo:
        """Rebuild metadata from an index record."""
        return cls(
            session_id=str(data.get("id") or data.get("session_id") or ""),
            title=str(data.get("title", "") or ""),
            path=path or str(data.get("path", "") or ""),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
            entries=int(data.get("entries", 0) or 0),
            bytes=int(data.get("bytes", 0) or 0),
            model=str(data.get("model", "") or ""),
            run_id=str(data.get("run_id", "") or ""),
            tags=[str(tag) for tag in (data.get("tags") or [])],
        )


class Session:
    """One conversation, its transcript, and its place on disk."""

    def __init__(self, session_id: str = "", *, title: str = "", directory: str | Path = "",
                 model: str = "", redact_secrets: bool = True,
                 max_bytes: int = 64 * 1024 * 1024, tags: Sequence[str] = ()) -> None:
        """Create or reopen a session.

        Args:
            session_id: stable identifier; generated when empty
            title: human-readable label, derived from the first message if unset
            directory: folder holding the transcript file
            model: model recorded in the header
            redact_secrets: scrub secrets before writing
            max_bytes: ceiling on the transcript file size
            tags: searchable labels
        """
        self.id = session_id or new_session_id()
        self.title = title
        self.directory = Path(directory).expanduser() if directory else Path.cwd()
        self.model = model
        self.redact_secrets = bool(redact_secrets)
        self.max_bytes = max(0, int(max_bytes))
        self.tags = [str(tag) for tag in tags]
        self.transcript = Transcript(title=self.title, model=model, run_id="")
        self.created_at = now_ts()
        self.updated_at = self.created_at
        self.appended = 0
        self.bytes_written = 0
        self.truncated_by_size = False
        self.entries_dropped_by_cap = 0
        self.cap_floor_applied = 0
        self._cap_warned = False
        self._lock = threading.RLock()
        self._dirty = False

    # -- paths ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        """File this session persists to."""
        return self.directory / f"{safe_filename(self.id)}{SESSION_SUFFIX}"

    @property
    def exists(self) -> bool:
        """Whether the transcript file is already on disk."""
        return self.path.exists()

    @property
    def size_bytes(self) -> int:
        """Current on-disk size, or 0 when not yet written."""
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    # -- recording -----------------------------------------------------------

    def record(self, message: Message, *, step: int = 0, run_id: str = "",
               model: str = "", duration_ms: float = 0.0,
               tokens: int = 0) -> TranscriptEntry:
        """Add one message to the transcript.

        The session title is derived from the first user message, which is what
        makes a session list readable without opening every file.

        Args:
            message: the turn to record
            step: agent loop iteration
            run_id: run that produced the turn
            model: model that generated an assistant turn
            duration_ms: generation time
            tokens: tokens used by the turn
        """
        with self._lock:
            entry = TranscriptEntry.from_message(message, step=step, run_id=run_id,
                                                 model=model or self.model,
                                                 duration_ms=duration_ms)
            entry.tokens = int(tokens)
            if entry.role == "assistant" and model:
                self.model = model
            if run_id and not self.transcript.run_id:
                self.transcript.run_id = run_id
            if not self.title and entry.role == "user" and entry.content.strip():
                self.title = _derive_title(entry.content)
                self.transcript.title = self.title
            self.transcript.add(self._scrub(entry))
            self.updated_at = now_ts()
            self._dirty = True
            return entry

    def record_tool(self, result: Any, *, step: int = 0, run_id: str = "") -> TranscriptEntry:
        """Record a tool result as its own entry."""
        with self._lock:
            entry = TranscriptEntry.from_tool_result(result, step=step, run_id=run_id)
            self.transcript.add(self._scrub(entry))
            self.updated_at = now_ts()
            self._dirty = True
            return entry

    def record_many(self, messages: Iterable[Message], **kwargs: Any) -> int:
        """Record several messages, returning the count."""
        count = 0
        for message in messages:
            self.record(message, **kwargs)
            count += 1
        return count

    def note(self, text: str, *, kind: str = "note", **metadata: Any) -> TranscriptEntry:
        """Attach a non-message annotation (a decision, an error, a metric)."""
        entry = TranscriptEntry(role="system", content=text, metadata={"kind": kind, **metadata})
        with self._lock:
            self.transcript.add(self._scrub(entry))
            self.updated_at = now_ts()
            self._dirty = True
        return entry

    def _scrub(self, entry: TranscriptEntry) -> TranscriptEntry:
        """Apply secret redaction when the session is configured for it."""
        return entry.redacted() if self.redact_secrets else entry

    # -- persistence ---------------------------------------------------------

    def save(self, *, rewrite: bool = False) -> Path:
        """Write the transcript, appending new entries when possible.

        Appending is the normal path and costs one line per turn. A full rewrite
        happens on first save, after a redaction change, or when the caller asks
        for it. Either way the size cap is enforced before writing.

        Args:
            rewrite: force a full rewrite instead of an append

        Returns:
            The transcript path.

        Raises:
            SessionError: the directory cannot be created
        """
        with self._lock:
            try:
                ensure_dir(self.directory)
            except OSError as exc:
                raise SessionError(f"cannot create session directory {self.directory}",
                                   details={"path": str(self.directory)}) from exc
            path = self.path
            ceiling = self.size_ceiling()
            if rewrite or not path.exists() or self.appended == 0:
                text = render_jsonl(self.transcript)
                encoded = len(text.encode("utf-8"))
                if ceiling and encoded > ceiling:
                    # A single huge turn can overshoot the ceiling on the very
                    # first write, so the cap has to apply to the payload and
                    # not only to appends. Drop the oldest entries until it fits.
                    dropped = self._shrink_to(ceiling)
                    text = render_jsonl(self.transcript)
                    encoded = len(text.encode("utf-8"))
                    self.truncated_by_size = True
                    self.entries_dropped_by_cap += dropped
                    log = LOGGER.debug if self._cap_warned else LOGGER.warning
                    log("session %s exceeded %s; dropped %d oldest entries",
                        self.id, human_size(self.max_bytes), dropped)
                    self._cap_warned = True
                atomic_write_text(path, text)
                self.bytes_written = encoded
                self.appended = len(self.transcript.entries)
            else:
                pending = self.transcript.entries[self.appended:]
                if pending:
                    lines = "".join(
                        json.dumps({**entry.to_dict(), "type": "entry"}, ensure_ascii=False) + "\n"
                        for entry in pending
                    )
                    encoded = len(lines.encode("utf-8"))
                    current = self.size_bytes
                    if ceiling and current + encoded > ceiling:
                        self.truncated_by_size = True
                        room = max(0, ceiling - current)
                        kept_lines = _prefix_within(lines, room)
                        if kept_lines:
                            with path.open("a", encoding="utf-8") as handle:
                                handle.write(kept_lines)
                            self.bytes_written += len(kept_lines.encode("utf-8"))
                            self.appended += kept_lines.count("\n")
                        # A capped session is flushed after every turn, so warn
                        # once and drop to debug afterwards; repeating the same
                        # line hundreds of times buries the rest of the log.
                        log = LOGGER.debug if self._cap_warned else LOGGER.warning
                        log("session %s reached %s; wrote %s of %s pending bytes",
                            self.id, human_size(ceiling),
                            human_size(len(kept_lines.encode("utf-8"))), human_size(encoded))
                        self._cap_warned = True
                    else:
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(lines)
                        self.bytes_written += encoded
                        self.appended += len(pending)
            self._dirty = False
            self.updated_at = now_ts()
            return path

    def size_ceiling(self) -> int:
        """Return the enforced byte ceiling, raised to the usable floor.

        A configured cap smaller than one resumable transcript is honoured in
        spirit - growth is still bounded - and the adjustment is recorded so it
        is visible rather than mysterious.
        """
        if not self.max_bytes:
            return 0
        if self.max_bytes < MIN_SESSION_BYTES:
            self.cap_floor_applied = MIN_SESSION_BYTES
            return MIN_SESSION_BYTES
        return self.max_bytes

    def _shrink_to(self, max_bytes: int) -> int:
        """Drop the oldest entries until the rendered transcript fits.

        The newest turn always survives: a transcript reduced to zero entries
        cannot be resumed or reviewed, which defeats the point of keeping a
        session at all. When even one entry does not fit, its content is
        truncated to the room available rather than discarded.

        Args:
            max_bytes: target ceiling for the serialized transcript

        Returns:
            How many entries were removed.
        """
        entries = self.transcript.entries
        if not entries:
            return 0

        def size() -> int:
            return len(render_jsonl(self.transcript).encode("utf-8"))

        removed = 0
        while len(entries) > 1 and size() > max_bytes:
            entries.pop(0)
            removed += 1
        if size() > max_bytes:
            newest = entries[-1]
            original = newest.content or ""
            keep = len(original.encode("utf-8"))
            # Shrink by measurement, not by arithmetic: JSON escaping (the
            # marker's newline, quotes, non-ASCII) means the serialized length
            # cannot be derived from the raw content length, and guessing
            # overshoots the very ceiling this is enforcing.
            for _ in range(64):
                if size() <= max_bytes or keep == 0:
                    break
                overshoot = size() - max_bytes
                keep = max(0, keep - max(8, overshoot))
                newest.content = original.encode("utf-8")[:keep].decode("utf-8", errors="ignore") \
                    + CAP_MARKER
            newest.metadata["size_capped"] = True
            if size() > max_bytes:
                LOGGER.debug("session %s cannot fit %s even with an empty newest entry",
                             self.id, human_size(max_bytes))
        return removed

    def flush(self) -> Path | None:
        """Save only when something changed since the last write."""
        with self._lock:
            if not self._dirty and self.appended:
                return None
            return self.save()

    def load(self) -> int:
        """Read an existing transcript from disk, returning the entry count.

        Raises:
            SessionError: the file is missing or cannot be parsed at all
        """
        path = self.path
        if not path.exists():
            raise SessionError(f"no such session: {self.id}", details={"path": str(path)})
        text = read_text_safe(path)
        transcript = parse_jsonl(text)
        if not transcript.entries and text.strip():
            raise SessionError(f"session {self.id} is unreadable", details={"path": str(path)})
        with self._lock:
            self.transcript = transcript
            if transcript.title:
                self.title = transcript.title
            if transcript.model:
                self.model = transcript.model
            self.appended = len(transcript.entries)
            self._dirty = False
            try:
                stat = path.stat()
                self.created_at = stat.st_ctime
                self.updated_at = stat.st_mtime
            except OSError:
                pass
        return len(transcript.entries)

    def delete(self) -> bool:
        """Remove the transcript file."""
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            LOGGER.warning("could not delete session %s: %s", self.id, exc)
            return False

    # -- access --------------------------------------------------------------

    @property
    def entries(self) -> list[TranscriptEntry]:
        """All recorded entries."""
        return list(self.transcript.entries)

    def messages(self, *, limit: int = 0, include_tools: bool = True) -> list[Message]:
        """Return replayable messages, most recent last.

        Args:
            limit: keep only this many trailing messages, 0 for all
            include_tools: keep tool-result turns
        """
        items = self.transcript.messages(include_tools=include_tools)
        return items[-limit:] if limit and limit < len(items) else items

    def last_messages(self, count: int = 10) -> list[Message]:
        """Return the trailing ``count`` messages for a resumed conversation."""
        return self.messages(limit=max(1, int(count)))

    def info(self) -> SessionInfo:
        """Return index metadata for this session."""
        return SessionInfo(
            session_id=self.id,
            title=self.title,
            path=str(self.path),
            created_at=self.created_at,
            updated_at=self.updated_at,
            entries=len(self.transcript.entries),
            bytes=self.size_bytes or self.bytes_written,
            model=self.model,
            run_id=self.transcript.run_id,
            tags=list(self.tags),
        )

    def render(self, fmt: str = "markdown", **kwargs: Any) -> str:
        """Render the transcript in one of the supported formats."""
        return self.transcript.render(fmt, **kwargs)

    def stats(self) -> dict[str, Any]:
        """Return transcript statistics plus persistence state."""
        payload = self.transcript.summary()
        payload.update({
            "id": self.id,
            "path": str(self.path),
            "bytes": self.size_bytes,
            "appended": self.appended,
            "dirty": self._dirty,
            "truncated_by_size": self.truncated_by_size,
            "entries_dropped_by_cap": self.entries_dropped_by_cap,
            "size_ceiling": self.size_ceiling(),
            "cap_floor_applied": self.cap_floor_applied,
            "tags": list(self.tags),
            "redact_secrets": self.redact_secrets,
        })
        return payload

    def __len__(self) -> int:
        """Number of entries."""
        return len(self.transcript.entries)

    def __repr__(self) -> str:
        """Debug representation."""
        return (f"Session(id={self.id!r}, title={self.title!r}, "
                f"entries={len(self.transcript.entries)})")


class SessionStore:
    """A directory of sessions with create, resume, list, and prune."""

    def __init__(self, directory: str | Path = "", *, redact_secrets: bool = True,
                 keep_last: int = DEFAULT_KEEP_LAST, autosave: bool = True,
                 max_session_bytes: int = 64 * 1024 * 1024,
                 transcript_format: str = "jsonl") -> None:
        """Configure the store.

        Args:
            directory: folder holding transcripts; defaults to the data dir
            redact_secrets: scrub secrets before writing
            keep_last: sessions retained by :meth:`prune`
            autosave: save after every recorded turn
            max_session_bytes: per-session size ceiling
            transcript_format: default rendering for exports
        """
        self.directory = Path(directory).expanduser() if directory else _default_directory()
        self.redact_secrets = bool(redact_secrets)
        self.keep_last = max(1, int(keep_last))
        self.autosave = bool(autosave)
        self.max_session_bytes = max(0, int(max_session_bytes))
        self.transcript_format = transcript_format or "jsonl"
        self._lock = threading.RLock()
        self._open: dict[str, Session] = {}

    @classmethod
    def from_config(cls, config: Any, *, root: str | Path | None = None) -> SessionStore:
        """Build a store from a resolved :class:`~magent.config.Config`.

        Args:
            config: configuration carrying a ``session`` section
            root: base data directory, overriding the configured path
        """
        section = getattr(config, "session", None)
        directory = str(getattr(section, "directory", "") or "")
        if not directory:
            base = Path(root) if root else Path(getattr(config, "data_dir", "") or _data_root())
            directory = base / DEFAULT_DIRECTORY
        return cls(
            directory=directory,
            redact_secrets=bool(getattr(section, "redact_secrets", True)),
            keep_last=int(getattr(section, "keep_last", DEFAULT_KEEP_LAST) or DEFAULT_KEEP_LAST),
            autosave=bool(getattr(section, "autosave", True)),
            max_session_bytes=int(getattr(section, "max_session_bytes", 0) or 64 * 1024 * 1024),
            transcript_format=str(getattr(section, "transcript_format", "jsonl") or "jsonl"),
        )

    @property
    def enabled(self) -> bool:
        """Whether persistence is usable (a directory is set)."""
        return bool(str(self.directory))

    # -- lifecycle -----------------------------------------------------------

    def create(self, *, title: str = "", session_id: str = "", model: str = "",
               tags: Sequence[str] = ()) -> Session:
        """Start a new session and write its header.

        Args:
            title: label; derived from the first message when empty
            session_id: explicit identifier, generated when empty
            model: model recorded in the header
            tags: searchable labels
        """
        session = Session(session_id, title=title, directory=self.directory, model=model,
                          redact_secrets=self.redact_secrets,
                          max_bytes=self.max_session_bytes, tags=tags)
        with self._lock:
            self._open[session.id] = session
        return session

    def open(self, session_id: str, *, load: bool = True) -> Session:
        """Reopen an existing session by identifier.

        Args:
            session_id: full identifier or a unique prefix
            load: read the transcript from disk immediately

        Raises:
            SessionError: no session matches the identifier
        """
        with self._lock:
            cached = self._open.get(session_id)
        if cached is not None:
            if load and not cached.transcript.entries and cached.exists:
                cached.load()
            return cached
        resolved = self.resolve(session_id)
        session = Session(resolved, directory=self.directory,
                          redact_secrets=self.redact_secrets,
                          max_bytes=self.max_session_bytes)
        if load:
            session.load()
        with self._lock:
            self._open[session.id] = session
        return session

    def resolve(self, session_id: str) -> str:
        """Expand a possibly abbreviated identifier to a full one.

        Raises:
            SessionError: the prefix is unknown or ambiguous
        """
        wanted = str(session_id or "").strip()
        if not wanted:
            raise SessionError("a session identifier is required")
        if (self.directory / f"{safe_filename(wanted)}{SESSION_SUFFIX}").exists():
            return wanted
        matches = [info.session_id for info in self.list()
                   if info.session_id.startswith(wanted)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SessionError(f"no session matches {wanted!r}",
                               details={"prefix": wanted, "known": len(self.list())})
        raise SessionError(f"session prefix {wanted!r} is ambiguous",
                           details={"matches": matches[:10]})

    def latest(self) -> Session | None:
        """Return the most recently updated session, for ``resume_last``."""
        infos = self.list()
        if not infos:
            return None
        newest = max(infos, key=lambda info: info.updated_at or info.created_at)
        try:
            return self.open(newest.session_id)
        except SessionError as exc:
            LOGGER.warning("could not resume %s: %s", newest.session_id, exc)
            return None

    def delete(self, session_id: str) -> bool:
        """Delete one session and forget it."""
        resolved = self.resolve(session_id)
        with self._lock:
            self._open.pop(resolved, None)
        try:
            (self.directory / f"{safe_filename(resolved)}{SESSION_SUFFIX}").unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            LOGGER.warning("delete failed for %s: %s", resolved, exc)
            return False

    def clear(self) -> int:
        """Delete every session, returning how many were removed."""
        removed = 0
        for info in self.list():
            if self.delete(info.session_id):
                removed += 1
        with self._lock:
            self._open.clear()
        return removed

    # -- querying ------------------------------------------------------------

    def list(self, *, limit: int = 0, tag: str = "", query: str = "") -> list[SessionInfo]:
        """List stored sessions, newest first.

        The listing is built from the directory rather than the index file, so
        it stays correct when transcripts are added or removed by hand.

        Args:
            limit: return at most this many
            tag: keep only sessions carrying this tag
            query: case-insensitive substring matched against the title
        """
        if not self.directory.exists():
            return []
        infos: list[SessionInfo] = []
        for path in sorted(self.directory.glob(f"*{SESSION_SUFFIX}")):
            info = _inspect(path)
            if info is None:
                continue
            if tag and tag not in info.tags:
                continue
            if query and query.lower() not in (info.title or "").lower():
                continue
            infos.append(info)
        # Filesystem mtimes can tie within a second, and a stable sort would
        # then leave sessions in directory order. Identifiers are time-ordered,
        # so they make a deterministic tiebreaker.
        infos.sort(key=lambda item: (item.updated_at or item.created_at, item.session_id),
                   reverse=True)
        return infos[:limit] if limit else infos

    def count(self) -> int:
        """Number of stored sessions."""
        return len(self.list())

    def total_bytes(self) -> int:
        """Combined size of every transcript."""
        return sum(info.bytes for info in self.list())

    def search(self, query: str, *, limit: int = 5, max_entries: int = 2000) -> list[dict[str, Any]]:
        """Full-text search across stored transcripts.

        Args:
            query: substring to look for, case-insensitive
            limit: maximum matches to return
            max_entries: per-session entry scan ceiling

        Returns:
            Match records with the session, entry index, and a snippet.
        """
        needle = str(query or "").strip().lower()
        if not needle:
            return []
        found: list[dict[str, Any]] = []
        for info in self.list():
            try:
                session = self.open(info.session_id)
            except SessionError:
                continue
            for index, entry in enumerate(session.entries[:max_entries]):
                if needle in (entry.content or "").lower():
                    found.append({
                        "session_id": info.session_id,
                        "title": info.title,
                        "index": index,
                        "role": entry.role,
                        "timestamp": entry.timestamp,
                        "snippet": _snippet(entry.content, needle),
                    })
                    if len(found) >= limit:
                        return found
        return found

    def prune(self, *, keep_last: int = 0) -> dict[str, Any]:
        """Delete the oldest sessions beyond ``keep_last``.

        Returns:
            A report with the number removed and bytes reclaimed.
        """
        keep = max(1, int(keep_last or self.keep_last))
        infos = self.list()
        if len(infos) <= keep:
            return {"removed": 0, "kept": len(infos), "bytes_freed": 0}
        victims = infos[keep:]
        freed = 0
        removed = 0
        for info in victims:
            freed += info.bytes
            if self.delete(info.session_id):
                removed += 1
        LOGGER.info("pruned %d sessions, reclaimed %s", removed, human_size(freed))
        return {"removed": removed, "kept": len(infos) - removed, "bytes_freed": freed}

    def export(self, session_id: str, path: str | Path, *, fmt: str = "") -> Path:
        """Export one session to a file in a chosen format.

        Args:
            session_id: session to export, prefix allowed
            path: destination file
            fmt: ``markdown``, ``text``, ``json``, or ``jsonl``
        """
        session = self.open(session_id)
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        atomic_write_text(target, session.render(fmt or self.transcript_format))
        return target

    def save_all(self) -> int:
        """Flush every open session, returning how many were written."""
        written = 0
        with self._lock:
            sessions = list(self._open.values())
        for session in sessions:
            if session.flush() is not None:
                written += 1
        return written

    def stats(self) -> dict[str, Any]:
        """Return store-level statistics."""
        infos = self.list()
        return {
            "directory": str(self.directory),
            "sessions": len(infos),
            "bytes": sum(info.bytes for info in infos),
            "entries": sum(info.entries for info in infos),
            "open": len(self._open),
            "keep_last": self.keep_last,
            "autosave": self.autosave,
            "redact_secrets": self.redact_secrets,
            "max_session_bytes": self.max_session_bytes,
            "newest": infos[0].session_id if infos else "",
        }

    def describe(self) -> str:
        """Return a one-line description for the CLI doctor command."""
        stats = self.stats()
        return (f"{stats['sessions']} sessions, {human_size(stats['bytes'])} in "
                f"{self.directory} (keep {self.keep_last})")

    def __iter__(self) -> Iterator[SessionInfo]:
        """Iterate session metadata, newest first."""
        return iter(self.list())

    def __len__(self) -> int:
        """Number of stored sessions."""
        return self.count()


def _prefix_within(lines: str, limit: int) -> str:
    """Return the leading whole lines of ``lines`` that fit in ``limit`` bytes."""
    if limit <= 0:
        return ""
    kept: list[str] = []
    total = 0
    for line in lines.splitlines(keepends=True):
        cost = len(line.encode("utf-8"))
        if total + cost > limit:
            break
        kept.append(line)
        total += cost
    return "".join(kept)


def _default_directory() -> Path:
    """Return the session directory implied by the environment."""
    return Path(_data_root()) / DEFAULT_DIRECTORY


def _data_root() -> Path:
    """Return MAgent's data directory, honouring ``MAGENT_DATA_DIR``."""
    from ..config import user_data_dir

    override = os.environ.get("MAGENT_DATA_DIR", "").strip()
    return Path(override).expanduser() if override else user_data_dir()


def _inspect(path: Path) -> SessionInfo | None:
    """Read a transcript's header and size without parsing every entry."""
    try:
        stat = path.stat()
    except OSError:
        return None
    info = SessionInfo(session_id=path.stem, path=str(path),
                       created_at=stat.st_ctime, updated_at=stat.st_mtime,
                       bytes=stat.st_size)
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
        if first.strip():
            header = json.loads(first)
            if header.get("type") == "meta":
                info.title = str(header.get("title", "") or "")
                info.model = str(header.get("model", "") or "")
                info.run_id = str(header.get("run_id", "") or "")
                if header.get("started_at"):
                    info.created_at = float(header["started_at"])
        entries = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if '"type": "entry"' in line or '"type":"entry"' in line:
                    entries += 1
        info.entries = entries
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOGGER.debug("could not inspect %s: %s", path, exc)
    return info


def _derive_title(content: str, *, limit: int = 60) -> str:
    """Turn a first user message into a short session title."""
    flat = " ".join(redact_text(content or "").split())
    if not flat:
        return ""
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _snippet(content: str, needle: str, *, width: int = 90) -> str:
    """Return a short excerpt centred on the match."""
    text = " ".join((content or "").split())
    position = text.lower().find(needle)
    if position < 0:
        return text[:width]
    start = max(0, position - width // 3)
    excerpt = text[start:start + width]
    prefix = "…" if start else ""
    suffix = "…" if start + width < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"


def build_store(config: Any = None, **kwargs: Any) -> SessionStore:
    """Build a :class:`SessionStore` from config, with keyword overrides."""
    store = SessionStore.from_config(config) if config is not None else SessionStore(**kwargs)
    for key, value in kwargs.items():
        if hasattr(store, key):
            setattr(store, key, value)
    return store


def cleanup(directory: str | Path = "", *, older_than_seconds: float = 0.0,
            dry_run: bool = False) -> dict[str, Any]:
    """Remove stale transcript files, optionally by age.

    Args:
        directory: session directory, defaulting to the configured one
        older_than_seconds: delete only sessions idle longer than this
        dry_run: report what would be deleted without deleting

    Returns:
        A report of removed and retained sessions.
    """
    root = Path(directory).expanduser() if directory else _default_directory()
    if not root.exists():
        return {"removed": 0, "kept": 0, "bytes_freed": 0, "dry_run": dry_run}
    removed = 0
    kept = 0
    freed = 0
    for path in sorted(root.glob(f"*{SESSION_SUFFIX}")):
        try:
            stat = path.stat()
        except OSError:
            continue
        age = max(0.0, now_ts() - stat.st_mtime)
        if older_than_seconds and age <= older_than_seconds:
            kept += 1
            continue
        freed += stat.st_size
        if not dry_run:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.warning("cleanup could not remove %s: %s", path, exc)
                kept += 1
        else:
            removed += 1
    return {"removed": removed, "kept": kept, "bytes_freed": freed, "dry_run": dry_run,
            "human_freed": human_size(freed)}


def format_session_list(infos: Sequence[SessionInfo], *, header: bool = True) -> str:
    """Render a session listing as aligned text."""
    if not infos:
        return "no sessions stored"
    lines = [f"{'ID':<12} {'UPDATED':<16} {'TURNS':>5} {'SIZE':>8}  TITLE"] if header else []
    lines.extend(info.describe() for info in infos)
    return "\n".join(lines)


__all__ = [
    "CAP_MARKER",
    "DEFAULT_DIRECTORY",
    "MIN_SESSION_BYTES",
    "DEFAULT_KEEP_LAST",
    "INDEX_FILE",
    "SESSION_SUFFIX",
    "Session",
    "SessionInfo",
    "SessionStore",
    "build_store",
    "cleanup",
    "format_session_list",
]
