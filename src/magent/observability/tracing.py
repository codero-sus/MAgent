"""Tracing: nested spans that show where a run actually spent its time.

An agent run is a tree - a run contains steps, a step contains an LLM call and
several tool executions, and a tool may call the model again. A flat log cannot
show that shape, so timing questions ("was it the model or the filesystem?")
stay unanswered.

Spans are propagated through :mod:`contextvars`, which means they cross
``ThreadPoolExecutor`` boundaries correctly when the pool copies the context,
and a nested ``with tracer.span(...)`` automatically becomes a child without
any threading of arguments through the call stack.

Export is JSONL, one span per line, so a trace can be appended incrementally
and read back without loading a whole run into memory.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..logging_ import get_logger
from ..utils.files import ensure_dir
from ..utils.ids import span_id as new_span_id
from ..utils.ids import trace_id as new_trace_id
from ..utils.timeutil import format_duration

LOGGER = get_logger("observability.trace")
"""Logger for tracing faults."""

STATUS_OK = "ok"
"""Span finished without an error."""

STATUS_ERROR = "error"
"""Span finished with an exception recorded on it."""

STATUS_UNSET = "unset"
"""Span was never explicitly finished."""

STATUSES: tuple[str, ...] = (STATUS_UNSET, STATUS_OK, STATUS_ERROR)
"""Every status a span may carry."""

MAX_EVENTS_PER_SPAN = 128
"""Cap on annotations per span, so a looping tool cannot grow one without bound."""

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "magent_current_span", default=None
)
"""Context variable carrying the active span."""


@dataclass(slots=True)
class SpanEvent:
    """A timestamped annotation attached to a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        payload: dict[str, Any] = {"name": self.name, "timestamp": round(self.timestamp, 6)}
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload


class Span:
    """One timed unit of work, optionally nested inside a parent."""

    def __init__(self, name: str, *, trace_id: str = "", parent_id: str = "",
                 span_id: str = "", kind: str = "internal",
                 attributes: Mapping[str, Any] | None = None,
                 tracer: Tracer | None = None) -> None:
        """Start a span; the clock begins here.

        Args:
            name: what the span measures, e.g. ``llm.complete``
            trace_id: identifier shared by every span in one run
            parent_id: identifier of the enclosing span, empty for a root
            span_id: this span's identifier, generated when empty
            kind: ``internal``, ``llm``, ``tool``, ``retrieval``, or ``http``
            attributes: initial dimensions recorded on the span
            tracer: owning tracer, used to publish the span when it ends
        """
        self.name = str(name)
        self.trace_id = trace_id or new_trace_id()
        self.span_id = span_id or new_span_id()
        self.parent_id = parent_id
        self.kind = kind
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.events: list[SpanEvent] = []
        self.status = STATUS_UNSET
        self.error: str = ""
        self.start_time = time.perf_counter()
        self.started_at = time.time()
        self.end_time = 0.0
        self.duration_ms = 0.0
        self._ended = False
        self._tracer = tracer

    # -- recording -----------------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> Span:
        """Record one attribute, returning the span for chaining."""
        self.attributes[str(key)] = value
        return self

    def set_attributes(self, values: Mapping[str, Any]) -> Span:
        """Record many attributes at once."""
        for key, value in dict(values).items():
            self.attributes[str(key)] = value
        return self

    def add_event(self, name: str, **attributes: Any) -> Span:
        """Attach a timestamped annotation.

        Events past :data:`MAX_EVENTS_PER_SPAN` are dropped rather than
        appended, because a tool looping in a tight cycle would otherwise grow
        a single span until it dominated the trace file.
        """
        if len(self.events) < MAX_EVENTS_PER_SPAN:
            self.events.append(SpanEvent(name=str(name), attributes=dict(attributes)))
        elif len(self.events) == MAX_EVENTS_PER_SPAN:
            self.events.append(SpanEvent(name="events_truncated",
                                         attributes={"limit": MAX_EVENTS_PER_SPAN}))
        return self

    def set_status(self, status: str, *, error: str = "") -> Span:
        """Mark the span's outcome."""
        self.status = status if status in STATUSES else STATUS_UNSET
        if error:
            self.error = str(error)
            self.status = STATUS_ERROR
        return self

    @property
    def ended(self) -> bool:
        """Whether :meth:`end` has been called."""
        return self._ended

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since the span started, live if still running."""
        if self._ended:
            return self.duration_ms
        return max(0.0, (time.perf_counter() - self.start_time) * 1000.0)

    @property
    def is_root(self) -> bool:
        """Whether this span has no parent."""
        return not self.parent_id

    def end(self, *, status: str = STATUS_OK, error: str = "") -> Span:
        """Stop the clock and hand the span to the tracer.

        Ending twice is a no-op, so a span can be closed in a ``finally`` block
        without racing the context manager.
        """
        if self._ended:
            return self
        self._ended = True
        self.end_time = time.perf_counter()
        self.duration_ms = max(0.0, (self.end_time - self.start_time) * 1000.0)
        # A status recorded during the span wins: a tool that called
        # set_status("error") and then returned normally must not be rewritten
        # to "ok" by the context manager exiting without an exception.
        if error or self.status == STATUS_UNSET:
            self.set_status(status, error=error)
        if self._tracer is not None:
            self._tracer._publish(self)
        return self

    # -- export --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized span, one JSONL line."""
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": round(self.started_at, 6),
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.parent_id:
            payload["parent_id"] = self.parent_id
        if self.attributes:
            payload["attributes"] = _jsonable(self.attributes)
        if self.events:
            payload["events"] = [event.to_dict() for event in self.events]
        if self.error:
            payload["error"] = self.error
        return payload

    def to_json(self) -> str:
        """Return the span as a single JSON line."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def children(self, spans: Sequence[Span]) -> list[Span]:
        """Return the spans whose parent is this one."""
        return [span for span in spans if span.parent_id == self.span_id]

    def __enter__(self) -> Span:
        """Activate the span for the duration of a ``with`` block."""
        if self._tracer is not None:
            self._tracer._activate(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, _tb: Any) -> None:
        """End the span, recording an exception when one propagated."""
        if exc_type is not None:
            self.end(status=STATUS_ERROR, error=f"{exc_type.__name__}: {exc}")
        else:
            self.end(status=STATUS_OK)
        if self._tracer is not None:
            self._tracer._deactivate(self)

    def __repr__(self) -> str:
        """Debug representation."""
        return (f"Span(name={self.name!r}, kind={self.kind!r}, status={self.status!r}, "
                f"duration_ms={self.duration_ms:.2f})")


class Tracer:
    """Creates spans, tracks the active one, and collects what finished."""

    def __init__(self, *, name: str = "magent", enabled: bool = True,
                 export_path: str | Path | None = None, keep: int = 2000,
                 default_attributes: Mapping[str, Any] | None = None) -> None:
        """Configure a tracer.

        Args:
            name: tracer name recorded on the root span
            enabled: when false, spans are inert and nothing is stored
            export_path: JSONL file that finished spans are appended to
            keep: maximum spans retained in memory
            default_attributes: attributes copied onto every new span
        """
        self.name = name
        self.enabled = bool(enabled)
        self.export_path = Path(export_path).expanduser() if export_path else None
        self.keep = max(16, int(keep))
        self.default_attributes = dict(default_attributes or {})
        self._spans: list[Span] = []
        self._lock = threading.RLock()
        self._token_stack: list[Any] = []
        self.created_spans = 0
        self.exported = 0
        self.export_failures = 0

    # -- creation ------------------------------------------------------------

    def span(self, name: str, *, kind: str = "internal", trace_id: str = "",
             parent: Span | None = None, **attributes: Any) -> Span:
        """Create a span parented to the currently active one.

        Args:
            name: what the span measures
            kind: span category
            trace_id: force a trace identifier; inherited when empty
            parent: explicit parent, defaulting to the active span
            attributes: dimensions recorded on the span
        """
        if not self.enabled:
            return _InertSpan(name, tracer=self)
        active = parent if parent is not None else self.current()
        merged = dict(self.default_attributes)
        merged.update(attributes)
        span = Span(
            name,
            trace_id=trace_id or (active.trace_id if active else ""),
            parent_id=active.span_id if active else "",
            kind=kind,
            attributes=merged,
            tracer=self,
        )
        with self._lock:
            self.created_spans += 1
        return span

    def start(self, name: str, **kwargs: Any) -> Span:
        """Create and immediately activate a span."""
        span = self.span(name, **kwargs)
        span.__enter__()
        return span

    def current(self) -> Span | None:
        """Return the active span for this context, or ``None``."""
        return _current_span.get()

    def current_trace_id(self) -> str:
        """Return the active trace identifier, or an empty string."""
        active = self.current()
        return active.trace_id if active else ""

    def trace(self, name: str = "run", **attributes: Any) -> Span:
        """Start a root span, beginning a new trace."""
        return self.span(name, trace_id=new_trace_id(), parent=None, **attributes)

    # -- context management --------------------------------------------------

    def _activate(self, span: Span) -> None:
        """Make ``span`` the active one, remembering how to undo it."""
        token = _current_span.set(span)
        with self._lock:
            self._token_stack.append(token)

    def _deactivate(self, span: Span) -> None:
        """Restore the previous active span."""
        with self._lock:
            token = self._token_stack.pop() if self._token_stack else None
        if token is not None:
            try:
                _current_span.reset(token)
            except ValueError:
                # The token belongs to another context (a worker thread); the
                # variable there will be garbage-collected with that context.
                LOGGER.debug("span context token from another context for %s", span.name)

    # -- collection ----------------------------------------------------------

    def _publish(self, span: Span) -> None:
        """Record a finished span and export it when configured."""
        with self._lock:
            self._spans.append(span)
            overflow = len(self._spans) - self.keep
            if overflow > 0:
                del self._spans[:overflow]
        if self.export_path is not None:
            self._export(span)

    def _export(self, span: Span) -> None:
        """Append one span to the JSONL export file."""
        try:
            ensure_dir(self.export_path.parent)
            with self.export_path.open("a", encoding="utf-8") as handle:
                handle.write(span.to_json() + "\n")
            self.exported += 1
        except OSError as exc:
            self.export_failures += 1
            LOGGER.debug("could not export span %s: %s", span.name, exc)

    def spans(self, *, trace_id: str = "", kind: str = "", name: str = "") -> list[Span]:
        """Return collected spans, optionally filtered."""
        with self._lock:
            items = list(self._spans)
        if trace_id:
            items = [span for span in items if span.trace_id == trace_id]
        if kind:
            items = [span for span in items if span.kind == kind]
        if name:
            items = [span for span in items if name in span.name]
        return items

    def clear(self) -> int:
        """Forget collected spans, returning how many were dropped."""
        with self._lock:
            count = len(self._spans)
            self._spans.clear()
        return count

    # -- analysis ------------------------------------------------------------

    def tree(self, trace_id: str = "") -> list[dict[str, Any]]:
        """Return the span tree for one trace as nested dicts."""
        items = self.spans(trace_id=trace_id)
        if not items:
            return []
        target = trace_id or items[0].trace_id
        items = [span for span in items if span.trace_id == target]
        by_parent: dict[str, list[Span]] = {}
        for span in items:
            by_parent.setdefault(span.parent_id, []).append(span)
        for children in by_parent.values():
            children.sort(key=lambda span: span.started_at)

        def build(span: Span) -> dict[str, Any]:
            return {
                "name": span.name,
                "kind": span.kind,
                "duration_ms": round(span.duration_ms, 3),
                "status": span.status,
                "children": [build(child) for child in by_parent.get(span.span_id, [])],
            }

        return [build(span) for span in by_parent.get("", [])]

    def render_tree(self, trace_id: str = "", *, show_attributes: bool = False) -> str:
        """Render one trace as an indented tree, for the CLI and bug reports."""
        items = self.spans(trace_id=trace_id)
        if not items:
            return "(no spans)"
        target = trace_id or items[0].trace_id
        items = [span for span in items if span.trace_id == target]
        by_parent: dict[str, list[Span]] = {}
        for span in items:
            by_parent.setdefault(span.parent_id, []).append(span)
        for children in by_parent.values():
            children.sort(key=lambda span: span.started_at)
        lines: list[str] = [f"trace {target}"]

        def describe(span: Span) -> str:
            marker = "!" if span.status == STATUS_ERROR else " "
            detail = f" [{span.kind}]" if span.kind != "internal" else ""
            attrs = ""
            if show_attributes and span.attributes:
                attrs = " " + " ".join(f"{key}={value}"
                                       for key, value in sorted(span.attributes.items()))
            return (f"{marker}{span.name}{detail} "
                    f"{format_duration(span.duration_ms / 1000.0)}{attrs}")

        def walk(span: Span, prefix: str, connector: str, child_prefix: str) -> None:
            lines.append(f"{prefix}{connector}{describe(span)}")
            children = by_parent.get(span.span_id, [])
            for index, child in enumerate(children):
                last = index == len(children) - 1
                walk(child, child_prefix, "└─ " if last else "├─ ",
                     child_prefix + ("   " if last else "│  "))

        for root in by_parent.get("", []):
            walk(root, "  ", "", "  ")
        return "\n".join(lines)

    def slowest(self, limit: int = 5, *, trace_id: str = "") -> list[Span]:
        """Return the slowest spans, which is usually where the time went."""
        items = self.spans(trace_id=trace_id)
        items.sort(key=lambda span: span.duration_ms, reverse=True)
        return items[: max(1, limit)]

    def errors(self, *, trace_id: str = "") -> list[Span]:
        """Return spans that ended in an error."""
        return [span for span in self.spans(trace_id=trace_id) if span.status == STATUS_ERROR]

    def stats(self) -> dict[str, Any]:
        """Return tracer counters and per-kind timing summaries."""
        with self._lock:
            items = list(self._spans)
        by_kind: dict[str, dict[str, Any]] = {}
        for span in items:
            record = by_kind.setdefault(span.kind, {"count": 0, "total_ms": 0.0, "errors": 0})
            record["count"] += 1
            record["total_ms"] += span.duration_ms
            if span.status == STATUS_ERROR:
                record["errors"] += 1
        for record in by_kind.values():
            record["total_ms"] = round(record["total_ms"], 3)
            record["mean_ms"] = round(record["total_ms"] / record["count"], 3) if record["count"] else 0.0
        return {
            "name": self.name,
            "enabled": self.enabled,
            "collected": len(items),
            "created": self.created_spans,
            "exported": self.exported,
            "export_failures": self.export_failures,
            "export_path": str(self.export_path) if self.export_path else "",
            "traces": len({span.trace_id for span in items}),
            "errors": sum(1 for span in items if span.status == STATUS_ERROR),
            "by_kind": by_kind,
        }

    def describe(self) -> str:
        """Return a one-line tracer summary."""
        info = self.stats()
        return (f"{info['collected']} spans across {info['traces']} traces, "
                f"{info['errors']} errors, {info['exported']} exported")

    # -- helpers -------------------------------------------------------------

    def instrument(self, name: str = "", *, kind: str = "internal") -> Callable[[Any], Any]:
        """Return a decorator that wraps a callable in a span."""
        import functools

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            label = name or f"{fn.__module__}.{fn.__qualname__}"

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(label, kind=kind):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def close(self) -> None:
        """End any spans still open and clear the context stack."""
        with self._lock:
            pending = [span for span in self._spans if not span.ended]
            self._token_stack.clear()
        for span in pending:
            span.end(status=STATUS_OK)
        active = self.current()
        if active is not None and not active.ended:
            active.end(status=STATUS_OK)


class _InertSpan(Span):
    """A span that records nothing, used when tracing is disabled."""

    def __init__(self, name: str, *, tracer: Tracer | None = None) -> None:
        """Create an inert span with a fixed identity."""
        super().__init__(name, tracer=None)
        self._tracer = None
        self._ended = True
        self.duration_ms = 0.0
        self.status = STATUS_UNSET

    def set_attribute(self, key: str, value: Any) -> Span:
        """No-op."""
        return self

    def set_attributes(self, values: Mapping[str, Any]) -> Span:
        """No-op."""
        return self

    def add_event(self, name: str, **attributes: Any) -> Span:
        """No-op."""
        return self

    def end(self, *, status: str = STATUS_OK, error: str = "") -> Span:
        """No-op."""
        return self

    def __enter__(self) -> Span:
        """No-op activation."""
        return self

    def __exit__(self, *_exc: Any) -> None:
        """No-op deactivation."""
        return None


def load_spans(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL trace export, skipping lines that will not parse."""
    target = Path(path).expanduser()
    if not target.exists():
        return []
    spans: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return spans


def tracer_from_config(config: Any, *, enabled: bool | None = None) -> Tracer:
    """Build a :class:`Tracer` from a resolved config's ``observability`` section."""
    section = getattr(config, "observability", None)
    active = bool(getattr(section, "trace_enabled", True)) if enabled is None else enabled
    path = str(getattr(section, "trace_file", "") or "") or None
    return Tracer(enabled=active, export_path=path,
                  name=str(getattr(config, "profile", "") or "magent") or "magent")


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce attribute values into JSON-safe primitives."""
    out: dict[str, Any] = {}
    for key, item in dict(value).items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
        elif isinstance(item, Mapping):
            out[str(key)] = _jsonable(item)
        elif isinstance(item, (list, tuple, set)):
            out[str(key)] = [str(entry) for entry in item]
        else:
            out[str(key)] = str(item)
    return out


def current_span() -> Span | None:
    """Return the active span without needing a tracer reference."""
    return _current_span.get()


def iter_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a trace export line by line."""
    yield from load_spans(path)


__all__ = [
    "MAX_EVENTS_PER_SPAN",
    "STATUSES",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNSET",
    "Span",
    "SpanEvent",
    "Tracer",
    "current_span",
    "iter_trace",
    "load_spans",
    "tracer_from_config",
]
