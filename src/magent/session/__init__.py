"""Session persistence: records, storage, transcripts and lifecycle.

Public API:

:class:`SessionRecord` / :class:`Turn`
    the durable form of a conversation, JSON serialisable;
:class:`SessionStore` / :class:`SessionInfo`
    one JSON file per session, with search, prune, export and import;
:class:`SessionManager`
    owns the current session, autosaves after each run, resumes the last one;
:func:`render_transcript` / :func:`parse_transcript`
    markdown/text/json/html rendering and the inverse parse.

>>> from magent.session import SessionRecord, derive_title
>>> record = SessionRecord(title=derive_title("please refactor the parser module"))
>>> record.title
'please refactor the parser module'
>>> record.id.startswith("ses") or len(record.id) > 8
True
"""

from __future__ import annotations

from .manager import HANDOFF_MAX_CHARS, SessionManager, SessionStats, build_session_manager
from .record import RECORD_VERSION, TITLE_MAX_WORDS, SessionRecord, Turn, derive_title, merge_records
from .store import (DEFAULT_DIRNAME, MAX_TITLE_CHARS, SessionInfo, SessionStore, redact_record,
                    session_dir, sorted_by_recency, touched_within)
from .transcript import (MAX_ANSWER_CHARS, SPEAKER_RE, TOOL_LINE_RE, TRANSCRIPT_FORMATS,
                         diff_transcripts, export_transcript, messages_to_transcript,
                         parse_transcript, render_transcript, transcript_html, transcript_markdown,
                         transcript_stats, transcript_text)

__all__ = [
    "DEFAULT_DIRNAME",
    "HANDOFF_MAX_CHARS",
    "MAX_ANSWER_CHARS",
    "MAX_TITLE_CHARS",
    "RECORD_VERSION",
    "SPEAKER_RE",
    "TITLE_MAX_WORDS",
    "TOOL_LINE_RE",
    "TRANSCRIPT_FORMATS",
    "SessionInfo",
    "SessionManager",
    "SessionRecord",
    "SessionStats",
    "SessionStore",
    "Turn",
    "build_session_manager",
    "derive_title",
    "diff_transcripts",
    "export_transcript",
    "merge_records",
    "messages_to_transcript",
    "parse_transcript",
    "redact_record",
    "render_transcript",
    "session_dir",
    "sorted_by_recency",
    "touched_within",
    "transcript_html",
    "transcript_markdown",
    "transcript_stats",
    "transcript_text",
]
