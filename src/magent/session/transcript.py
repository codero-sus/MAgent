"""Transcripts: read a session as text, and turn text back into messages.

``magent sessions show`` and the web UI both need a readable rendering of a
stored conversation, and ``magent sessions import`` needs the inverse - a plain
``user:`` / ``assistant:`` transcript parsed back into
:class:`~magent.types.Message` objects.  Keeping both directions here means the
round trip is tested in one place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..secrets import redact_text
from ..types import Message
from ..utils.files import atomic_write_text, ensure_dir
from ..utils.text import ellipsize, normalize_newlines
from ..utils.timeutil import format_timestamp
from .record import SessionRecord

TRANSCRIPT_FORMATS: tuple[str, ...] = ("markdown", "text", "json", "jsonl", "html")
"""Transcript renderings this module produces."""

SPEAKER_RE = re.compile(r"^\s*(user|assistant|system|tool)\s*[:>]?\s*(.*)$", re.IGNORECASE)
"""Line that starts a new speaker turn in a plain transcript."""

TOOL_LINE_RE = re.compile(r"^\s*(?:->|→)\s*(\w+)\s*\((.*)\)\s*$")
"""Tool-call line emitted in text transcripts."""

MAX_ANSWER_CHARS = 100_000
"""Safety ceiling for one rendered answer."""


def render_transcript(record: SessionRecord, *, fmt: str = "markdown", redact: bool = False,
                      include_metadata: bool = True, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Render a session record in one of :data:`TRANSCRIPT_FORMATS`.

    Args:
        record: the session
        fmt: ``markdown``, ``text``, ``json``, ``jsonl`` or ``html``
        redact: scrub secret-looking strings
        include_metadata: add the header block with model, tokens and tags
        max_chars: clip each message body
    """
    active = (fmt or "markdown").strip().lower()
    if active == "json":
        payload = record.to_dict()
        text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    elif active == "jsonl":
        text = "\n".join(json.dumps(message.to_dict(), default=str, ensure_ascii=False)
                         for message in record.messages)
    elif active == "html":
        text = transcript_html(record, include_metadata=include_metadata, max_chars=max_chars)
    elif active == "text":
        text = transcript_text(record, include_metadata=include_metadata, max_chars=max_chars)
    else:
        text = transcript_markdown(record, include_metadata=include_metadata, max_chars=max_chars)
    return redact_text(text) if redact else text


def _header_lines(record: SessionRecord) -> list[str]:
    """Metadata lines shared by the markdown and text renderings."""
    lines = [
        f"Session: {record.id}",
        f"Title: {record.title or '(untitled)'}",
        f"Created: {format_timestamp(record.created_at)}",
        f"Updated: {format_timestamp(record.updated_at)}",
    ]
    if record.model:
        lines.append(f"Model: {record.model}")
    if record.workspace:
        lines.append(f"Workspace: {record.workspace}")
    if record.tags:
        lines.append(f"Tags: {', '.join(record.tags)}")
    lines.append(f"Messages: {len(record.messages)}  Turns: {len(record.turns)}  "
                 f"Tokens: {record.usage.total_tokens}  Cost: {record.cost:.4f}")
    return lines


def transcript_markdown(record: SessionRecord, *, include_metadata: bool = True,
                        max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Render a session as a Markdown document."""
    parts: list[str] = [f"# {record.title or 'Session'}"]
    if include_metadata:
        parts.append("\n".join(f"- {line}" for line in _header_lines(record)))
    if record.summary:
        parts.append("## Summary\n\n" + record.summary)
    if not record.messages and record.turns:
        # Records built from turns alone still deserve a readable transcript.
        exchange: list[str] = ["## Conversation"]
        for turn in record.turns:
            exchange.append(f"**USER**\n\n{turn.prompt}")
            if turn.answer:
                exchange.append(f"**ASSISTANT**\n\n{ellipsize(turn.answer, max_chars)}")
        parts.append("\n\n".join(exchange))
    if record.messages:
        body: list[str] = ["## Conversation"]
        for message in record.messages:
            role = str(getattr(message.role, "value", message.role)).upper()
            text = ellipsize(normalize_newlines(message.text()), max_chars)
            body.append(f"**{role}**\n\n{text}")
            for call in message.tool_calls or []:
                body.append(f"- tool call: `{call.name}` "
                            f"`{json.dumps(call.arguments, default=str)[:200]}`")
        parts.append("\n\n".join(body))
    if record.turns:
        rows = ["## Turns", "", "| # | Prompt | Seconds | Tools | Tokens | OK |",
                "| --- | --- | --- | --- | --- | --- |"]
        for turn in record.turns:
            rows.append(f"| {turn.index + 1} | {_cell(turn.prompt)} | {turn.seconds:.2f} | "
                        f"{', '.join(turn.tools) or '-'} | {turn.tokens} | "
                        f"{'yes' if turn.ok else 'no'} |")
        parts.append("\n".join(rows))
    return "\n\n".join(parts).strip() + "\n"


def _cell(text: str, limit: int = 60) -> str:
    """Escape one Markdown table cell."""
    return ellipsize((text or "").replace("\n", " ").replace("|", "\\|"), limit)


def transcript_text(record: SessionRecord, *, include_metadata: bool = True,
                    max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Render a session as plain text suitable for a terminal or a log."""
    lines: list[str] = []
    if include_metadata:
        lines.extend(_header_lines(record))
        lines.append("")
    if record.summary:
        lines.append(f"summary: {record.summary}")
        lines.append("")
    for message in record.messages:
        role = str(getattr(message.role, "value", message.role))
        lines.append(f"{role}: {ellipsize(normalize_newlines(message.text()), max_chars)}")
        for call in message.tool_calls or []:
            args = json.dumps(call.arguments, default=str, ensure_ascii=False)
            lines.append(f"  -> {call.name}({ellipsize(args, 160)})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def transcript_html(record: SessionRecord, *, include_metadata: bool = True,
                    max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Render a session as a standalone HTML page."""
    from html import escape

    styles = (
        "body{font:14px/1.55 ui-sans-serif,system-ui,sans-serif;margin:0;padding:24px;"
        "background:#0f1115;color:#e6e6e6}"
        "h1{font-size:20px;margin:0 0 8px}"
        ".meta{color:#8b93a7;font-size:12px;margin-bottom:16px}"
        ".msg{margin:12px 0;padding:10px 14px;border-radius:10px;background:#171a21}"
        ".role{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#7fd1ff;"
        "margin-bottom:4px}"
        ".msg.assistant .role{color:#9be29b}"
        ".msg.system .role{color:#8b93a7}"
        "pre{white-space:pre-wrap;margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,monospace}"
        ".tool{color:#c9a0ff;font-size:12px;margin-top:6px}"
    )
    parts: list[str] = ["<!doctype html>", "<html lang='en'><head><meta charset='utf-8'>",
                        f"<title>{escape(record.title or record.id)}</title>",
                        f"<style>{styles}</style></head><body>"]
    parts.append(f"<h1>{escape(record.title or 'Session')}</h1>")
    if include_metadata:
        parts.append("<div class='meta'>" + "<br>".join(escape(line) for line in
                                                        _header_lines(record)) + "</div>")
    if record.summary:
        parts.append(f"<div class='msg system'><div class='role'>summary</div>"
                     f"<pre>{escape(record.summary)}</pre></div>")
    for message in record.messages:
        role = str(getattr(message.role, "value", message.role))
        text = ellipsize(normalize_newlines(message.text()), max_chars)
        block = [f"<div class='msg {escape(role)}'><div class='role'>{escape(role)}</div>",
                 f"<pre>{escape(text)}</pre>"]
        for call in message.tool_calls or []:
            args = json.dumps(call.arguments, default=str, ensure_ascii=False)
            block.append(f"<div class='tool'>→ {escape(call.name)}({escape(args[:200])})</div>")
        block.append("</div>")
        parts.append("".join(block))
    parts.append("</body></html>")
    return "\n".join(parts)


def export_transcript(record: SessionRecord, path: str | Path, *, fmt: str = "markdown",
                      redact: bool = False) -> Path:
    """Write a transcript to disk, deriving the format from the extension."""
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    active = fmt
    if target.suffix.lower() in {".md", ".markdown"}:
        active = "markdown"
    elif target.suffix.lower() in {".html", ".htm"}:
        active = "html"
    elif target.suffix.lower() == ".jsonl":
        active = "jsonl"
    elif target.suffix.lower() == ".json":
        active = "json"
    elif target.suffix.lower() == ".txt":
        active = "text"
    return atomic_write_text(target, render_transcript(record, fmt=active, redact=redact))


def parse_transcript(text: str, *, default_role: str = "user") -> list[Message]:
    """Parse a ``user:`` / ``assistant:`` transcript back into messages.

    Lines that do not start with a known speaker continue the previous message,
    so multi-paragraph answers survive the round trip.  Tool-call lines are
    dropped: they are rendered output, not conversation state.
    """
    messages: list[Message] = []
    current_role = default_role
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            messages.append(Message(role=current_role, content=body))

    for raw_line in normalize_newlines(text or "").split("\n"):
        if TOOL_LINE_RE.match(raw_line):
            continue
        match = SPEAKER_RE.match(raw_line)
        if match and match.group(1).lower() in {"user", "assistant", "system", "tool"}:
            flush()
            current_role = match.group(1).lower()
            rest = match.group(2)
            if rest:
                buffer.append(rest)
            continue
        buffer.append(raw_line)
    flush()
    return messages


def messages_to_transcript(messages: Iterable[Message], *, include_roles: bool = True) -> str:
    """Render raw messages as a plain transcript."""
    lines: list[str] = []
    for message in messages:
        role = str(getattr(message.role, "value", message.role))
        text = normalize_newlines(message.text())
        lines.append(f"{role}: {text}" if include_roles else text)
    return "\n".join(lines)


def transcript_stats(record: SessionRecord) -> dict[str, Any]:
    """Character and turn statistics for a transcript."""
    chars = sum(len(message.text()) for message in record.messages)
    roles: dict[str, int] = {}
    for message in record.messages:
        key = str(getattr(message.role, "value", message.role))
        roles[key] = roles.get(key, 0) + 1
    return {
        "messages": len(record.messages),
        "turns": len(record.turns),
        "chars": chars,
        "words": sum(len(message.text().split()) for message in record.messages),
        "roles": roles,
        "tokens": record.usage.total_tokens,
        "seconds": round(sum(turn.seconds for turn in record.turns), 3),
        "tools": record.tool_usage(),
    }


def diff_transcripts(before: Sequence[Message], after: Sequence[Message]) -> dict[str, Any]:
    """Compare two message lists, for resume/compaction diagnostics."""
    before_ids = {message.id for message in before}
    after_ids = {message.id for message in after}
    return {
        "before": len(before),
        "after": len(after),
        "added": len(after_ids - before_ids),
        "removed": len(before_ids - after_ids),
        "kept": len(before_ids & after_ids),
    }


__all__ = [
    "MAX_ANSWER_CHARS",
    "SPEAKER_RE",
    "TOOL_LINE_RE",
    "TRANSCRIPT_FORMATS",
    "diff_transcripts",
    "export_transcript",
    "messages_to_transcript",
    "parse_transcript",
    "render_transcript",
    "transcript_html",
    "transcript_markdown",
    "transcript_stats",
    "transcript_text",
]
