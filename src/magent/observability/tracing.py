"""Tracing: spans, traces and an ASCII flame graph.

A run is a tree - the run itself, each reasoning step, each tool call, each LLM
request, each guardrail check.  :class:`Tracer` builds that tree with
:class:`Span` objects that can be used as context managers, exports it as JSONL
(one span per line, so a long session can be appended to forever), and renders
it as a flame graph for the terminal:

    run  1240ms
    ├─ step.0  1180ms
    │  ├─ llm.chat  900ms
    │  └─ tool.calculate  4ms

Spans are thread-safe and carry attributes plus timestamped events, which is
what makes a slow run diagnosable after the fact rather than only while watching
it happen.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from ..utils.files import ensure_dir
from ..utils.ids import span_id as new_span_id
from ..utils.ids import trace_id as new_trace_id

LOGGER = get_logger("observability.trace")
"""Logger for tracing problems."""

SPAN_STATUS: tuple[str, ...] = ("running", "ok", "error", "cancelled")
"""Lifecycle states a span can be in."""

MAX_SPANS_PER_TRACE = 2_000
"""Ceiling on spans kept for one trace, so a runaway loop cannot eat memory."""

MAX_EVENTS_PER_SPAN = 100
"""Ceiling on events recorded inside one span."""


@dataclass(slots=True)
class SpanEvent:
    """A timestamped note inside a span."""

    name: str
    ts: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event."""
        return {"name": self.name, "ts": self.ts, "attributes": dict(self.attributes)}


@dataclass(slots=True)
class Span:
    """One timed unit of work inside a trace."""

    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_id: str = ""
    kind: str = "internal"
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    status: str = "running"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: str = ""
    children: list[Span] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Assign identifiers when the caller did not."""
        if not self.span_id:
            self.span_id = new_span_id()
        if not self.trace_id:
            self.trace_id = new_trace_id()

    @property
    def duration_ms(self) -> float:
        """Elapsed milliseconds, live when the span is still running."""
        end = self.ended_at or time.time()
        return max(0.0, (end - self.started_at) * 1000.0)

    @property
    def finished(self) -> bool:
        """True once the span has ended."""
        return self.ended_at > 0

    def set_attribute(self, key: str, value: Any) -> Span:
        """Record one attribute; returns self for chaining."""
        self.attributes[key] = value
        return self

    def set_attributes(self, values: dict[str, Any]) -> Span:
        """Record several attributes."""
        self.attributes.update(values)
        return self

    def add_event(self, event_name: str, **attributes: Any) -> Span:
        """Append a timestamped event to the span.

        The event name is positional and separately named so callers can pass an
        attribute called ``name`` - ``span.add_event("tool", name="calculate")`` -
        without colliding with the parameter.
        """
        attributes.pop("event_name", None)
        if len(self.events) < MAX_EVENTS_PER_SPAN:
            self.events.append(SpanEvent(name=event_name, attributes=dict(attributes)))
        return self

    def fail(self, error: str) -> Span:
        """Mark the span as failed and end it."""
        self.error = str(error)
        return self.finish("error")

    def finish(self, status: str = "ok") -> Span:
        """End the span."""
        if not self.finished:
            self.ended_at = time.time()
        self.status = status if status in SPAN_STATUS else "ok"
        return self

    def child(self, name: str, *, kind: str = "internal", **attributes: Any) -> Span:
        """Create a nested span sharing this span's trace."""
        span = Span(
            name=name,
            trace_id=self.trace_id,
            parent_id=self.span_id,
            kind=kind,
            attributes=dict(attributes),
        )
        if len(self.children) < MAX_SPANS_PER_TRACE:
            self.children.append(span)
        return span

    def walk(self) -> Iterator[Span]:
        """Yield this span and every descendant, depth first."""
        yield self
        for item in self.children:
            yield from item.walk()

    def depth(self) -> int:
        """Nesting depth of the deepest descendant."""
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def to_dict(self, *, include_children: bool = False) -> dict[str, Any]:
        """Serialise the span, optionally with its subtree."""
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": dict(self.attributes),
            "events": [event.to_dict() for event in self.events],
            "error": self.error,
        }
        if include_children:
            payload["children"] = [child.to_dict(include_children=True) for child in self.children]
        return payload

    def flame(self, *, indent: int = 0, width: int = 60, unicode: bool = True) -> str:
        """Render the subtree as an ASCII flame graph."""
        branch, last, pipe = ("├─", "└─", "│") if unicode else ("|-", "`-", "|")
        bar = _bar(self.duration_ms, width)
        lines = [
            f"{'  ' * indent}{self.name}  {self.duration_ms:.0f}ms {bar}"
            f"{' ERR' if self.status == 'error' else ''}"
        ]
        for index, child in enumerate(self.children):
            connector = last if index == len(self.children) - 1 else branch
            child_lines = child.flame(indent=0, width=width, unicode=unicode).split("\n")
            lines.append(f"{'  ' * indent}{connector} {child_lines[0]}")
            continuation = f"{pipe} " if connector == branch else "   "
            for extra in child_lines[1:]:
                lines.append(f"{'  ' * indent}{continuation}{extra}")
        return "\n".join(lines)

    def __enter__(self) -> Span:
        """Start using the span as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Finish the span, recording an exception when one propagated."""
        if exc is not None:
            self.fail(f"{type(exc).__name__}: {exc}")
        else:
            self.finish()


def _bar(duration_ms: float, width: int) -> str:
    """Render a duration as a small bar, relative to a 5 second scale."""
    scale = max(1.0, 5_000.0)
    filled = int(min(width, max(0, round(duration_ms / scale * width))))
    return "[" + "#" * filled + "." * (width - filled) + "]"


@dataclass(slots=True)
class Trace:
    """A tree of spans sharing one trace id."""

    name: str = "trace"
    trace_id: str = ""
    root: Span | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Create the trace id and root span when they are missing."""
        if not self.trace_id:
            self.trace_id = new_trace_id()
        if self.root is None:
            self.root = Span(
                name=self.name, trace_id=self.trace_id, kind="run", started_at=self.started_at
            )

    @property
    def duration_ms(self) -> float:
        """Elapsed milliseconds for the whole trace."""
        end = self.ended_at or time.time()
        return max(0.0, (end - self.started_at) * 1000.0)

    def spans(self) -> list[Span]:
        """Every span in the trace, depth first."""
        return list(self.root.walk()) if self.root else []

    def find(self, name: str) -> Span | None:
        """First span with this name."""
        for span in self.spans():
            if span.name == name:
                return span
        return None

    def errors(self) -> list[Span]:
        """Spans that finished in an error state."""
        return [span for span in self.spans() if span.status == "error"]

    def slowest(self, limit: int = 5) -> list[Span]:
        """Slowest spans, excluding the root."""
        spans = [span for span in self.spans() if span is not self.root]
        spans.sort(key=lambda span: -span.duration_ms)
        return spans[: max(1, limit)]

    def finish(self, status: str = "ok") -> Trace:
        """End the trace and any span still running."""
        for span in self.spans():
            if not span.finished:
                span.finish(status if span is self.root else "ok")
        self.ended_at = time.time()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole trace."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 3),
            "spans": len(self.spans()),
            "errors": len(self.errors()),
            "attributes": dict(self.attributes),
            "root": self.root.to_dict(include_children=True) if self.root else None,
        }

    def flame(self, *, width: int = 60, unicode: bool = True) -> str:
        """Render the trace as an ASCII flame graph."""
        return self.root.flame(width=width, unicode=unicode) if self.root else ""

    def jsonl(self) -> str:
        """Serialise every span as one JSON line."""
        import json

        return "\n".join(
            json.dumps(span.to_dict(), default=str, ensure_ascii=False) for span in self.spans()
        )


class Tracer:
    """Create traces and track the span currently in flight."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        export_path: str | Path | None = None,
        sample_rate: float = 1.0,
        service: str = "magent",
    ) -> None:
        """Configure the tracer.

        Args:
            enabled: when false every call is a no-op returning a dummy span
            export_path: file to append finished traces to as JSONL
            sample_rate: fraction of traces to keep, 0..1
            service: service name recorded on every span
        """
        self.enabled = bool(enabled)
        self.export_path = Path(export_path).expanduser() if export_path else None
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.service = service
        self._local = threading.local()
        self._lock = threading.Lock()
        self._traces: list[Trace] = []
        self.spans_created = 0
        self.spans_exported = 0
        self.dropped = 0

    @property
    def current_trace(self) -> Trace | None:
        """Trace active on this thread."""
        return getattr(self._local, "trace", None)

    @property
    def current_span(self) -> Span | None:
        """Span active on this thread."""
        return getattr(self._local, "span", None)

    def start_trace(self, name: str = "run", **attributes: Any) -> Trace:
        """Begin a new trace on this thread."""
        trace = Trace(name=name, attributes={"service": self.service, **attributes})
        if not self._sampled():
            self.dropped += 1
            trace.attributes["sampled_out"] = True
        self._local.trace = trace
        self._local.span = trace.root
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > 64:
                self._traces = self._traces[-64:]
        return trace

    def _sampled(self) -> bool:
        """Decide whether the current trace is kept."""
        if self.sample_rate >= 1.0:
            return True
        import random

        return random.random() < self.sample_rate

    @contextmanager
    def trace(self, name: str = "run", **attributes: Any) -> Iterator[Trace]:
        """Context manager that starts, finishes and exports a trace."""
        active = self.start_trace(name, **attributes)
        try:
            yield active
        finally:
            self.end_trace()

    def start_span(
        self, name: str, *, kind: str = "internal", parent: Span | None = None, **attributes: Any
    ) -> Span:
        """Start a span inside the current trace."""
        if not self.enabled:
            return Span(name=name, kind=kind, status="ok", ended_at=time.time())
        trace = self.current_trace
        if trace is None or trace.root is None:
            trace = self.start_trace(name)
        owner = parent or self.current_span or trace.root
        assert owner is not None
        span = owner.child(name, kind=kind, service=self.service, **attributes)
        self.spans_created += 1
        self._local.span = span
        return span

    @contextmanager
    def span(self, name: str, *, kind: str = "internal", **attributes: Any) -> Iterator[Span]:
        """Context manager that starts and finishes one span."""
        previous = self.current_span
        active = self.start_span(name, kind=kind, **attributes)
        try:
            yield active
        except Exception as exc:
            active.fail(f"{type(exc).__name__}: {exc}")
            raise
        else:
            active.finish()
        finally:
            self._local.span = previous

    def end_trace(self, *, status: str = "ok", export: bool = True) -> Trace | None:
        """Finish the current trace, export it and clear thread state."""
        trace = self.current_trace
        if trace is None:
            return None
        trace.finish(status)
        self._local.trace = None
        self._local.span = None
        if export and self.enabled and not trace.attributes.get("sampled_out"):
            self.export(trace)
        return trace

    def export(self, trace: Trace) -> Path | None:
        """Append a trace's spans to the export file."""
        if self.export_path is None:
            return None
        try:
            ensure_dir(self.export_path.parent)
            with self.export_path.open("a", encoding="utf-8") as handle:
                handle.write(trace.jsonl() + "\n")
        except OSError as exc:
            LOGGER.warning("trace export failed: %s", exc)
            return None
        self.spans_exported += len(trace.spans())
        return self.export_path

    def record(self, trace: Trace) -> Trace:
        """Adopt a trace built elsewhere so it shows up in :meth:`traces`."""
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > 64:
                self._traces = self._traces[-64:]
        return trace

    def traces(self, *, limit: int = 10) -> list[Trace]:
        """Recent traces, newest last."""
        with self._lock:
            return list(self._traces[-max(1, limit) :])

    def last_trace(self) -> Trace | None:
        """Most recent trace."""
        with self._lock:
            return self._traces[-1] if self._traces else None

    def stats(self) -> dict[str, Any]:
        """Tracer counters."""
        with self._lock:
            traces = list(self._traces)
        return {
            "enabled": self.enabled,
            "service": self.service,
            "sample_rate": self.sample_rate,
            "export_path": str(self.export_path) if self.export_path else "",
            "traces": len(traces),
            "spans": sum(len(trace.spans()) for trace in traces),
            "spans_created": self.spans_created,
            "spans_exported": self.spans_exported,
            "dropped": self.dropped,
            "errors": sum(len(trace.errors()) for trace in traces),
        }

    def reset(self) -> None:
        """Forget every trace."""
        with self._lock:
            self._traces.clear()
        self._local.trace = None
        self._local.span = None


def trace_from_events(events: Sequence[Any], *, name: str = "run") -> Trace:
    """Rebuild a trace from agent events, for after-the-fact analysis.

    Args:
        events: :class:`~magent.types.AgentEvent` objects from a run
        name: trace name
    """
    trace = Trace(name=name)
    if trace.root is None:
        return trace
    if events:
        trace.started_at = min(getattr(event, "ts", trace.started_at) for event in events)
    stack: list[Span] = [trace.root]
    open_steps: dict[Any, Span] = {}
    open_tools: dict[Any, Span] = {}
    for event in events:
        kind = str(getattr(getattr(event, "kind", ""), "value", getattr(event, "kind", "")))
        data = getattr(event, "data", {}) or {}
        ts = float(getattr(event, "ts", time.time()) or time.time())
        if kind == "run.started":
            trace.root.set_attributes(
                {"run_id": data.get("run_id", ""), "model": data.get("model", "")}
            )
        elif kind == "step.started":
            span = stack[0].child(f"step.{data.get('index', 0)}", kind="step")
            span.started_at = ts
            open_steps[data.get("index", len(open_steps))] = span
        elif kind == "step.finished":
            span = open_steps.get(data.get("index"))
            if span is not None:
                span.ended_at = ts
                span.set_attribute("tool_calls", data.get("tool_calls", 0))
                span.finish()
        elif kind == "tool.call":
            parent = open_steps.get(data.get("step"), stack[0]) or stack[0]
            span = parent.child(f"tool.{data.get('name', 'tool')}", kind="tool")
            span.started_at = ts
            span.set_attribute("call_id", data.get("call_id", ""))
            open_tools[data.get("call_id", span.span_id)] = span
        elif kind == "tool.result":
            span = open_tools.pop(data.get("call_id", ""), None)
            if span is not None:
                span.ended_at = ts
                span.set_attribute("ok", data.get("ok", True))
                span.finish("ok" if data.get("ok", True) else "error")
        elif kind in {"llm.request", "llm.response"}:
            span = stack[0].child(kind, kind="llm")
            span.started_at = ts
            span.ended_at = ts
            span.set_attributes(
                {
                    key: value
                    for key, value in data.items()
                    if isinstance(value, (str, int, float, bool))
                }
            )
            span.finish()
        elif kind == "run.finished":
            trace.root.set_attribute("stop_reason", data.get("stop_reason", ""))
            trace.root.ended_at = ts
    trace.finish()
    return trace


__all__ = [
    "MAX_EVENTS_PER_SPAN",
    "MAX_SPANS_PER_TRACE",
    "SPAN_STATUS",
    "Span",
    "SpanEvent",
    "Trace",
    "Tracer",
    "trace_from_events",
]
