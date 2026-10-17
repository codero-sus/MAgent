"""Durable conversations: transcripts on disk, resumable and redacted.

Quick start::

    from magent.session import SessionStore

    store = SessionStore.from_config(settings)
    session = store.create(title="port question")
    session.record(Message.user("What port does Cortex use?"))
    session.save()

    later = store.latest()              # resume the newest session
    history = later.last_messages(20)   # replayable messages

The store appends JSONL rather than rewriting, so an interrupted run still
leaves a readable transcript, and it enforces three limits: secret redaction
before disk, ``keep_last`` pruning, and a per-session byte ceiling.
"""

from __future__ import annotations

from .store import (
    CAP_MARKER,
    DEFAULT_DIRECTORY,
    DEFAULT_KEEP_LAST,
    INDEX_FILE,
    MIN_SESSION_BYTES,
    SESSION_SUFFIX,
    Session,
    SessionInfo,
    SessionStore,
    build_store,
    cleanup,
    format_session_list,
)
from .transcript import (
    FORMATS,
    ROLE_LABELS,
    Transcript,
    TranscriptEntry,
    parse_jsonl,
    render_jsonl,
    render_markdown,
    render_text,
)

__all__ = [
    "CAP_MARKER",
    "DEFAULT_DIRECTORY",
    "DEFAULT_KEEP_LAST",
    "FORMATS",
    "INDEX_FILE",
    "MIN_SESSION_BYTES",
    "ROLE_LABELS",
    "SESSION_SUFFIX",
    "Session",
    "SessionInfo",
    "SessionStore",
    "Transcript",
    "TranscriptEntry",
    "build_store",
    "cleanup",
    "format_session_list",
    "parse_jsonl",
    "render_jsonl",
    "render_markdown",
    "render_text",
]
