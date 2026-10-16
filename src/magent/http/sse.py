"""Server-Sent Events parsing for streamed LLM responses."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

DONE_SENTINEL = "[DONE]"
"""The terminator OpenAI-compatible servers send at the end of a stream."""


@dataclass(slots=True)
class SSEEvent:
    """One parsed ``event:``/``data:`` record from an SSE stream."""

    data: str = ""
    event: str = "message"
    id: str = ""
    retry: int | None = None
    comments: list[str] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        """Report whether this event is the OpenAI ``[DONE]`` terminator."""
        return self.data.strip() == DONE_SENTINEL

    def json(self, default: Any = None) -> Any:
        """Parse ``data`` as JSON, returning ``default`` on failure."""
        if not self.data.strip():
            return default
        try:
            return json.loads(self.data)
        except json.JSONDecodeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event envelope."""
        return {
            "event": self.event,
            "id": self.id,
            "retry": self.retry,
            "data": self.data,
        }


class SSEParser:
    """Incremental, spec-conformant SSE parser.

    Feed it arbitrary byte/str chunks with :meth:`feed`; it yields complete
    :class:`SSEEvent` objects as soon as a blank line terminates a record.
    Multi-line ``data:`` fields are joined with newlines exactly as the WHATWG
    EventSource specification requires.
    """

    def __init__(self, *, max_event_bytes: int = 16 * 1024 * 1024) -> None:
        self._buffer = ""
        self._data_lines: list[str] = []
        self._event = "message"
        self._id = ""
        self._retry: int | None = None
        self._comments: list[str] = []
        self._last_id = ""
        self.max_event_bytes = max_event_bytes
        self.events_parsed = 0
        self.bytes_seen = 0
        self.malformed_lines = 0

    @property
    def last_event_id(self) -> str:
        """Return the most recent ``id:`` seen, for resume support."""
        return self._last_id

    def feed(self, chunk: str | bytes) -> Iterator[SSEEvent]:
        """Consume a chunk and yield every event it completes."""
        if isinstance(chunk, bytes):
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                text = chunk.decode("utf-8", errors="replace")
        else:
            text = chunk
        self.bytes_seen += len(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            event = self._handle_line(line.rstrip("\r"))
            if event is not None:
                yield event

    def _handle_line(self, line: str) -> SSEEvent | None:
        """Process one SSE line; return an event when a record completes."""
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            self._comments.append(line[1:].strip())
            return None
        name, sep, value = line.partition(":")
        if not sep:
            self.malformed_lines += 1
            return None
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data_lines.append(value)
        elif name == "event":
            self._event = value or "message"
        elif name == "id":
            if "\x00" not in value:
                self._id = value
                self._last_id = value
        elif name == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                self.malformed_lines += 1
        return None

    def _dispatch(self) -> SSEEvent | None:
        """Emit the buffered record and reset per-event state."""
        if not self._data_lines and not self._comments:
            self._event = "message"
            return None
        joined = "\n".join(self._data_lines)
        if len(joined) > self.max_event_bytes:
            joined = joined[: self.max_event_bytes]
        event = SSEEvent(
            data=joined,
            event=self._event or "message",
            id=self._id,
            retry=self._retry,
            comments=list(self._comments),
        )
        self._data_lines.clear()
        self._comments.clear()
        self._event = "message"
        self._id = ""
        self._retry = None
        self.events_parsed += 1
        return event

    def close(self) -> SSEEvent | None:
        """Flush any trailing record that was not newline terminated."""
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle_line(line)
        return self._dispatch()

    def stats(self) -> dict[str, Any]:
        """Return parser counters for observability."""
        return {
            "events_parsed": self.events_parsed,
            "bytes_seen": self.bytes_seen,
            "malformed_lines": self.malformed_lines,
            "last_event_id": self._last_id,
        }


def parse_sse_line(line: str) -> tuple[str, str] | None:
    """Parse a single ``field: value`` SSE line into its components."""
    stripped = line.rstrip("\r\n")
    if not stripped or stripped.startswith(":"):
        return None
    name, sep, value = stripped.partition(":")
    if not sep:
        return None
    return name, value[1:] if value.startswith(" ") else value


def iter_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse an iterable of raw lines into :class:`SSEEvent` objects."""
    parser = SSEParser()
    for line in lines:
        for event in parser.feed(line + "\n"):
            yield event
    final = parser.close()
    if final is not None:
        yield final


def extract_delta_text(payload: Any) -> str:
    """Pull the text delta out of an OpenAI-style streaming chunk.

    Handles ``choices[0].delta.content``, the legacy ``choices[0].text``
    completion shape, and the Responses API ``output_text.delta`` event.
    """
    if not isinstance(payload, dict):
        return ""
    event_type = str(payload.get("type") or "")
    if event_type.endswith("output_text.delta"):
        return str(payload.get("delta") or "")
    if event_type in {"response.output_text.delta", "response.text.delta"}:
        return str(payload.get("delta") or "")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            return ""
        text = choice.get("text")
        return str(text) if isinstance(text, str) else ""
    if isinstance(payload.get("delta"), str):
        return str(payload["delta"])
    return ""


def extract_delta_tool_calls(payload: Any) -> list[dict[str, Any]]:
    """Pull streamed tool-call fragments out of a chunk, if present."""
    if not isinstance(payload, dict):
        return []
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        return []
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return []
    calls = delta.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def merge_tool_call_deltas(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassemble streamed tool-call fragments into complete call objects.

    Providers split a single tool call across many chunks, indexing them by
    position and sending the arguments JSON as a string stream.
    """
    merged: dict[int, dict[str, Any]] = {}
    for fragment in fragments:
        index = int(fragment.get("index") or 0)
        target = merged.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if fragment.get("id"):
            target["id"] = str(fragment["id"])
        function = fragment.get("function") or {}
        if isinstance(function, dict):
            if function.get("name"):
                target["function"]["name"] += str(function["name"])
            if function.get("arguments"):
                target["function"]["arguments"] += str(function["arguments"])
    return [merged[key] for key in sorted(merged)]


__all__ = [
    "DONE_SENTINEL",
    "SSEEvent",
    "SSEParser",
    "extract_delta_text",
    "extract_delta_tool_calls",
    "iter_sse",
    "merge_tool_call_deltas",
    "parse_sse_line",
]
