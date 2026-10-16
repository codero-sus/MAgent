"""Session management: start, resume, autosave and hand off conversations.

The manager owns the *current* :class:`~magent.session.record.SessionRecord` and
keeps it in sync with short-term memory.  Every completed run is folded in - the
messages, the tools used, the tokens and cost - and written to disk when
autosave is on.  ``resume_last`` makes ``magent chat`` pick up where the user
left off, and :meth:`SessionManager.handoff_block` turns a session into the
context another agent needs to continue the work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..logging_ import get_logger
from ..types import Message, RunResult, ToolResult
from ..utils.ids import session_id as new_session_id
from ..utils.text import ellipsize
from .record import SessionRecord, Turn, derive_title
from .store import SessionInfo, SessionStore
from .transcript import messages_to_transcript, render_transcript

LOGGER = get_logger("session.manager")
"""Logger for session lifecycle events."""

HANDOFF_MAX_CHARS = 2_400
"""Ceiling on the handoff block injected into another agent's prompt."""


@dataclass(slots=True)
class SessionStats:
    """Counters describing session activity."""

    started: int = 0
    resumed: int = 0
    saved: int = 0
    failed_saves: int = 0
    turns: int = 0
    messages: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the counters."""
        return {
            "started": self.started,
            "resumed": self.resumed,
            "saved": self.saved,
            "failed_saves": self.failed_saves,
            "turns": self.turns,
            "messages": self.messages,
            "seconds": round(self.seconds, 3),
        }


class SessionManager:
    """Own the current session and its persistence."""

    def __init__(self, *, store: SessionStore | None = None, config: Any = None,
                 short_term: Any = None, autosave: bool = True, redact_secrets: bool = True,
                 resume_last: bool = False, enabled: bool = True,
                 transcript_format: str = "markdown", directory: str | Path | None = None) -> None:
        """Configure the manager.

        Args:
            store: session store; built from config when omitted
            config: a :class:`~magent.config.Config` or session config
            short_term: :class:`~magent.memory.short_term.ShortTermMemory` to keep in sync
            autosave: write the record after every run
            redact_secrets: scrub secrets before writing
            resume_last: load the newest session on :meth:`ensure`
            enabled: when false the manager keeps state in memory only
            transcript_format: default transcript rendering
            directory: storage directory override
        """
        session = getattr(config, "session", config)
        self.config = config
        self.enabled = bool(enabled if config is None else getattr(session, "enabled", enabled))
        self.autosave = bool(autosave if config is None else getattr(session, "autosave", autosave))
        self.redact_secrets = bool(redact_secrets if config is None
                                   else getattr(session, "redact_secrets", redact_secrets))
        self.resume_last = bool(resume_last if config is None
                                else getattr(session, "resume_last", resume_last))
        self.keep_last = int(getattr(session, "keep_last", 0) or 0)
        self.transcript_format = str(getattr(session, "transcript_format", transcript_format)
                                     or transcript_format)
        # ``SessionStore`` defines __len__, so an empty store is falsy: test for None.
        self.store = store if store is not None else SessionStore.from_config(config,
                                                                              directory=directory)
        self.short_term = short_term
        self.current: SessionRecord | None = None
        self.stats = SessionStats()
        self._started_at = time.time()

    @classmethod
    def from_config(cls, config: Any = None, *, store: SessionStore | None = None,
                    short_term: Any = None, directory: str | Path | None = None) -> SessionManager:
        """Build a manager from configuration."""
        session = getattr(config, "session", None)
        return cls(config=config, store=store, short_term=short_term, directory=directory,
                   autosave=bool(getattr(session, "autosave", True)),
                   redact_secrets=bool(getattr(session, "redact_secrets", True)),
                   resume_last=bool(getattr(session, "resume_last", False)),
                   enabled=bool(getattr(session, "enabled", True)),
                   transcript_format=str(getattr(session, "transcript_format", "markdown")))

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, session_id: str = "", title: str = "", workspace: str | Path = "",
              model: str = "", tags: Sequence[str] = ()) -> SessionRecord:
        """Begin a new session, making it current."""
        record = SessionRecord(id=session_id or new_session_id(), title=title,
                               workspace=str(workspace or ""), model=model)
        record.tag(*tags)
        self.current = record
        self.stats.started += 1
        self._sync_short_term(reset=True)
        self.save()
        LOGGER.debug("started session %s", record.id)
        return record

    def resume(self, session_id: str = "") -> SessionRecord | None:
        """Load a session and make it current; empty id resumes the newest."""
        record = self.store.latest() if not session_id else self.store.load(session_id)
        if record is None:
            LOGGER.debug("no session to resume (id=%r)", session_id)
            return None
        self.current = record
        self.stats.resumed += 1
        self._sync_short_term(reset=True)
        LOGGER.debug("resumed session %s (%d messages)", record.id, len(record.messages))
        return record

    def ensure(self, *, workspace: str | Path = "", model: str = "") -> SessionRecord:
        """Return the current session, resuming or starting one as configured."""
        if self.current is not None:
            return self.current
        if not self.enabled:
            self.current = SessionRecord(workspace=str(workspace or ""), model=model)
            return self.current
        if self.resume_last:
            resumed = self.resume()
            if resumed is not None:
                return resumed
        return self.start(workspace=workspace, model=model)

    def switch(self, session_id: str) -> SessionRecord:
        """Switch to another stored session."""
        record = self.store.require(session_id)
        self.current = record
        self._sync_short_term(reset=True)
        return record

    def save(self) -> Path | None:
        """Persist the current session; returns the path written."""
        if self.current is None or not self.enabled:
            return None
        try:
            path = self.store.save(self.current, redact=self.redact_secrets)
        except Exception as exc:  # noqa: BLE001 - a failed save must not abort a run
            self.stats.failed_saves += 1
            LOGGER.warning("session save failed: %s", exc)
            return None
        self.stats.saved += 1
        return path

    def close(self, *, save: bool = True) -> Path | None:
        """Finish the current session."""
        path = self.save() if save else None
        if self.current is not None:
            self.stats.seconds += time.time() - self._started_at
            self.stats.messages += len(self.current.messages)
            self.stats.turns += len(self.current.turns)
        self.current = None
        self._started_at = time.time()
        return path

    # -- recording -----------------------------------------------------------

    def record_message(self, message: Message) -> Message:
        """Add one message to the current session."""
        record = self.ensure()
        record.add_message(message)
        if self.short_term is not None:
            try:
                self.short_term.add(message, session_id=record.id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("short-term add failed: %s", exc)
        return message

    def record_messages(self, messages: Iterable[Message]) -> int:
        """Add several messages."""
        return sum(1 for _ in (self.record_message(message) for message in messages))

    def record_run(self, prompt: str, result: RunResult) -> Turn:
        """Fold a completed run into the session.

        Args:
            prompt: the user's text
            result: the run outcome
        """
        record = self.ensure()
        tools = sorted({call.name for step in result.steps for call in step.tool_calls})
        turn = record.add_turn(
            prompt, result.output,
            started_at=result.started_at,
            seconds=round(result.duration_ms / 1000, 4),
            tools=tools,
            tokens=result.usage.total_tokens,
            ok=bool(result.succeeded),
            error=result.error or "",
            metadata={"run_id": result.run_id, "stop_reason": str(result.stop_reason),
                      "model": result.model},
        )
        record.add_messages(result.messages or [])
        if not record.messages:
            record.add_message(Message.user(prompt))
            if result.output:
                record.add_message(Message.assistant(result.output))
        record.add_usage(result.usage, cost=getattr(result.cost, "total", 0.0) or 0.0)
        if result.model:
            record.model = result.model
        if not record.title:
            record.title = derive_title(prompt)
        if self.short_term is not None:
            self._sync_short_term()
        if self.autosave:
            self.save()
        self.stats.turns += 1
        return turn

    def record_tool_results(self, results: Sequence[ToolResult]) -> None:
        """Note tool outcomes on the latest turn."""
        if not self.current or not self.current.turns:
            return
        turn = self.current.turns[-1]
        for item in results:
            if item.name not in turn.tools:
                turn.tools.append(item.name)
            if not item.ok:
                turn.metadata.setdefault("tool_errors", []).append(
                    f"{item.name}: {item.error or item.error_code or 'failed'}")

    def set_title(self, title: str) -> str:
        """Rename the current session."""
        record = self.ensure()
        value = record.set_title(title)
        self.save()
        return value

    def tag(self, *tags: str) -> list[str]:
        """Tag the current session."""
        record = self.ensure()
        values = record.tag(*tags)
        self.save()
        return values

    def summarize(self, text: str) -> str:
        """Store a rolling summary produced by short-term memory."""
        record = self.ensure()
        record.summary = ellipsize(text, 4_000)
        record.touch()
        return record.summary

    # -- views ---------------------------------------------------------------

    def history(self, *, limit: int = 20, include_summary: bool = True) -> list[Message]:
        """Messages of the current session, oldest first, capped at ``limit``."""
        record = self.current
        if record is None:
            return []
        if self.short_term is not None:
            try:
                return self.short_term.messages(record.id, include_summary=include_summary)[-limit:]
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("short-term history failed: %s", exc)
        return record.messages[-limit:]

    def transcript(self, *, fmt: str = "", redact: bool | None = None) -> str:
        """Render the current session as a transcript."""
        record = self.current
        if record is None:
            return ""
        return render_transcript(record, fmt=fmt or self.transcript_format,
                                 redact=self.redact_secrets if redact is None else redact)

    def handoff_block(self, *, max_chars: int = HANDOFF_MAX_CHARS, turns: int = 3) -> str:
        """Build the context another agent needs to continue this session."""
        record = self.current
        if record is None:
            return ""
        parts: list[str] = [f"Session {record.id}: {record.title or '(untitled)'}"]
        if record.summary:
            parts.append(f"Summary: {record.summary}")
        dialogue = [message for message in record.messages
                    if str(getattr(message.role, "value", message.role)) != "system"]
        recent = dialogue[-turns * 2:]
        if recent:
            parts.append("Recent exchange:\n" + messages_to_transcript(recent))
        tools = record.tool_usage()
        if tools:
            parts.append("Tools used: " + ", ".join(f"{name}×{count}"
                                                    for name, count in list(tools.items())[:8]))
        return ellipsize("\n\n".join(parts), max_chars)

    def list(self, *, limit: int = 25, tag: str = "", query: str = "") -> list[SessionInfo]:
        """List stored sessions."""
        return self.store.list(limit=limit, tag=tag, query=query)

    def load(self, session_id: str) -> SessionRecord | None:
        """Load a session without making it current."""
        return self.store.load(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a stored session."""
        removed = self.store.delete(session_id)
        if removed and self.current is not None and self.current.id == session_id:
            self.current = None
        return removed

    def search(self, query: str, *, limit: int = 10) -> list[tuple[SessionInfo, list[str]]]:
        """Full-text search stored sessions."""
        return self.store.search(query, limit=limit)

    def describe(self) -> dict[str, Any]:
        """Describe manager state, for ``magent doctor`` and ``sessions info``."""
        record = self.current
        return {
            "enabled": self.enabled,
            "autosave": self.autosave,
            "resume_last": self.resume_last,
            "redact_secrets": self.redact_secrets,
            "keep_last": self.keep_last,
            "directory": str(self.store.directory),
            "transcript_format": self.transcript_format,
            "current": record.id if record else None,
            "current_title": record.title if record else "",
            "current_messages": len(record.messages) if record else 0,
            "current_turns": len(record.turns) if record else 0,
            "stored": len(self.store),
            "stats": self.stats.to_dict(),
        }

    def _sync_short_term(self, *, reset: bool = False) -> None:
        """Push the current session's history into short-term memory."""
        if self.short_term is None or self.current is None:
            return
        record = self.current
        try:
            if reset:
                self.short_term.clear(record.id)
                if record.messages:
                    self.short_term.extend(record.messages, session_id=record.id)
                if record.summary:
                    self.short_term.window(record.id).set_summary(record.summary)
            else:
                window = self.short_term.window(record.id)
                if window.summary and not record.summary:
                    record.summary = window.summary
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("short-term sync failed: %s", exc)

    def __repr__(self) -> str:
        """Compact developer representation."""
        current = self.current.id if self.current else None
        return f"SessionManager(current={current!r}, dir={str(self.store.directory)!r})"


def build_session_manager(config: Any = None, *, short_term: Any = None,
                          store: SessionStore | None = None) -> SessionManager:
    """Construct a manager from configuration in one call."""
    return SessionManager.from_config(config, store=store, short_term=short_term)


__all__ = [
    "HANDOFF_MAX_CHARS",
    "SessionManager",
    "SessionStats",
    "build_session_manager",
]
