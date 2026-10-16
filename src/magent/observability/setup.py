"""Observability wiring: logging, tracing and metrics from one config section.

:class:`Observability` is the object a CLI command or server builds once per
process.  It configures logging from :class:`~magent.config.ObservabilityConfig`,
owns a :class:`~magent.observability.tracing.Tracer` and a
:class:`~magent.observability.metrics.MetricsRegistry`, and attaches to an agent's
event bus so that every run produces a trace and updates the counters without
the agent knowing observability exists.

Secrets are redacted in logs by :mod:`magent.logging_`, and trace attributes go
through the same redaction so an exported JSONL file never carries an API key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..events import EventBus
from ..logging_ import get_logger, setup_logging
from ..secrets import redact_text
from ..types import AgentEvent, EventKind
from .metrics import MetricsRegistry, record_llm, record_run, record_tool
from .tracing import Span, Trace, Tracer, trace_from_events

LOGGER = get_logger("observability")
"""Logger for observability itself."""

TRACE_ATTRIBUTES: tuple[str, ...] = ("run_id", "model", "mode", "stop_reason", "tool", "name",
                                     "step", "index", "attempt", "error", "tokens")
"""Event data keys copied onto spans as attributes."""

SKIP_KINDS: frozenset[str] = frozenset({EventKind.TEXT_DELTA.value, EventKind.THINKING.value})
"""High-volume events that never become spans."""


@dataclass(slots=True)
class ObservabilityReport:
    """Snapshot of everything observability collected."""

    uptime_seconds: float = 0.0
    runs: int = 0
    traces: int = 0
    spans: int = 0
    errors: int = 0
    metrics: int = 0
    log_file: str = ""
    trace_file: str = ""
    counters: dict[str, Any] = field(default_factory=dict)
    slowest: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report."""
        return {
            "uptime_seconds": round(self.uptime_seconds, 3),
            "runs": self.runs,
            "traces": self.traces,
            "spans": self.spans,
            "errors": self.errors,
            "metrics": self.metrics,
            "log_file": self.log_file,
            "trace_file": self.trace_file,
            "counters": dict(self.counters),
            "slowest": list(self.slowest),
        }

    def summary(self) -> str:
        """One-line summary for ``magent doctor``."""
        return (f"{self.runs} runs, {self.spans} spans, {self.errors} errors, "
                f"{self.metrics} metrics in {self.uptime_seconds:.1f}s")


class Observability:
    """Logging, tracing and metrics for one process."""

    def __init__(self, config: Any = None, *, tracer: Tracer | None = None,
                 metrics: MetricsRegistry | None = None, logger: Any = None) -> None:
        """Build the façade.

        Args:
            config: a full :class:`~magent.config.Config` or an observability config
            tracer: tracer override
            metrics: registry override
            logger: logger override
        """
        section = getattr(config, "observability", config)
        self.config = config
        self.section = section
        self.tracer = tracer if tracer is not None else Tracer(
            enabled=bool(getattr(section, "trace_enabled", True)),
            export_path=getattr(section, "trace_file", "") or None,
        )
        self.metrics = metrics if metrics is not None else MetricsRegistry()
        self.logger = logger or LOGGER
        self.subscriptions: list[tuple[Any, Any]] = []
        self.started_at = time.time()
        self.runs = 0
        self.errors = 0
        self._spans: dict[str, Span] = {}
        self._run_traces: dict[str, Trace] = {}

    @classmethod
    def from_config(cls, config: Any = None, **kwargs: Any) -> Observability:
        """Build from configuration, then apply the logging settings."""
        instance = cls(config, **kwargs)
        instance.configure_logging()
        return instance

    # -- logging -------------------------------------------------------------

    def configure_logging(self, *, force: bool = False) -> Any:
        """Apply the configured logging setup; returns the root MAgent logger."""
        section = self.section
        level = str(getattr(section, "log_level", "INFO") or "INFO").upper()
        fmt = str(getattr(section, "log_format", "text") or "text")
        log_file = getattr(section, "log_file", "") or None
        return setup_logging(
            level=level,
            fmt=fmt,
            log_file=log_file,
            force=force,
            quiet=bool(getattr(getattr(self.config, "output", None), "quiet", False)),
            max_bytes=int(getattr(section, "max_log_bytes", 0) or 0) or 33_554_432,
            backup_count=int(getattr(section, "keep_rotated", 5) or 5),
        )

    # -- attaching -----------------------------------------------------------

    def attach(self, agent: Any = None, *, bus: EventBus | None = None) -> Observability:
        """Subscribe to an agent's (or a bare bus's) events.

        Args:
            agent: agent whose ``events`` bus should be observed
            bus: explicit bus, used instead of the agent's
        """
        target = bus if bus is not None else getattr(agent, "events", None)
        if target is None:
            self.logger.debug("no event bus to observe")
            return self
        subscription = target.subscribe(self.record_event, name="observability", priority=200)
        # The bus owns removal, so keep it alongside the subscription: a
        # Subscription has no cancel method of its own.
        self.subscriptions.append((target, subscription))
        self.metrics.inc("observers_attached")
        return self

    def detach(self) -> int:
        """Unsubscribe from every bus; returns how many were removed."""
        removed = 0
        for bus, subscription in list(self.subscriptions):
            try:
                if hasattr(bus, "unsubscribe"):
                    bus.unsubscribe(subscription)
                removed += 1
            except Exception as exc:  # noqa: BLE001 - detaching must not raise
                self.logger.debug("detach failed: %s", exc)
        self.subscriptions.clear()
        return removed

    # -- event handling ------------------------------------------------------

    def record_event(self, event: AgentEvent) -> None:
        """Turn one agent event into span and metric updates."""
        kind_value = str(getattr(event.kind, "value", event.kind))
        if kind_value in SKIP_KINDS:
            return
        data = event.data or {}
        run_id = str(event.run_id or data.get("run_id", "") or "default")
        try:
            if event.kind == EventKind.RUN_STARTED:
                self._start_run(run_id, data)
            elif event.kind in {EventKind.RUN_FINISHED, EventKind.RUN_FAILED}:
                self._finish_run(run_id, event, data)
            elif event.kind == EventKind.STEP_STARTED:
                self._start_step(run_id, data)
            elif event.kind == EventKind.STEP_FINISHED:
                self._finish_step(run_id, data)
            elif event.kind == EventKind.TOOL_CALL:
                self._tool_call(run_id, data)
            elif event.kind == EventKind.TOOL_RESULT:
                self._tool_result(run_id, data)
            elif event.kind == EventKind.TOOL_ERROR:
                self._tool_error(run_id, data)
            elif event.kind == EventKind.LLM_RESPONSE:
                self._llm_response(run_id, data)
            elif event.kind == EventKind.LLM_RETRY:
                self.metrics.inc("llm_retries_total")
            elif event.kind == EventKind.LLM_CACHE_HIT:
                self.metrics.inc("llm_cache_hits_total")
            elif event.kind == EventKind.GUARDRAIL_BLOCKED:
                self.metrics.inc("guardrail_blocks_total",
                                 labels={"stage": str(data.get("stage", "unknown"))})
            elif event.kind == EventKind.APPROVAL_REQUESTED:
                self.metrics.inc("approvals_requested_total")
            elif event.kind == EventKind.APPROVAL_DECIDED:
                decision = str(data.get("decision", data.get("approved", "")))
                self.metrics.inc("approvals_decided_total", labels={"decision": decision})
            elif event.kind == EventKind.BUDGET_WARNING:
                self.metrics.inc("budget_warnings_total")
            elif event.kind == EventKind.MODEL_SWITCHED:
                self.metrics.inc("model_switches_total",
                                 labels={"model": str(data.get("model", "unknown"))})
            elif event.kind == EventKind.MEMORY_WRITE:
                self.metrics.inc("memory_writes_total", labels={"kind": str(data.get("kind", ""))})
            self.metrics.inc("events_total", labels={"kind": kind_value})
        except Exception as exc:  # noqa: BLE001 - observability must never break a run
            self.logger.debug("observability handler failed for %s: %s", kind_value, exc)

    def _trace_for(self, run_id: str) -> Trace:
        """Return the trace for a run, creating one when needed."""
        trace = self._run_traces.get(run_id)
        if trace is None:
            trace = Trace(name=f"run:{run_id}", attributes={"service": "magent"})
            self._run_traces[run_id] = trace
            if len(self._run_traces) > 64:
                oldest = next(iter(self._run_traces))
                self._run_traces.pop(oldest, None)
        return trace

    def _attributes(self, data: dict[str, Any]) -> dict[str, Any]:
        """Copy the interesting event data onto a span, redacting secrets."""
        out: dict[str, Any] = {}
        for key in TRACE_ATTRIBUTES:
            if key in data and isinstance(data[key], (str, int, float, bool)):
                value = data[key]
                out[key] = redact_text(str(value)) if isinstance(value, str) else value
        return out

    def _start_run(self, run_id: str, data: dict[str, Any]) -> None:
        """Open the root span for a run."""
        self.runs += 1
        trace = self._trace_for(run_id)
        if trace.root is not None:
            trace.root.set_attributes(self._attributes(data))
        self.metrics.inc("runs_started_total", labels={"model": str(data.get("model", "unknown"))})
        self.metrics.gauge("active_runs").inc()

    def _finish_run(self, run_id: str, event: AgentEvent, data: dict[str, Any]) -> None:
        """Close the run's trace and fold the result into metrics."""
        trace = self._run_traces.get(run_id)
        if trace is not None and trace.root is not None:
            trace.root.set_attributes(self._attributes(data))
            status = "error" if event.kind == EventKind.RUN_FAILED else "ok"
            if data.get("error"):
                trace.root.error = redact_text(str(data["error"]))[:400]
                self.errors += 1
            trace.finish(status)
            self.tracer.record(trace)
            self.tracer.export(trace)
            self._run_traces.pop(run_id, None)
        result = data.get("result")
        if result is not None:
            record_run(self.metrics, result)
        else:
            self.metrics.inc("runs_finished_total")
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)):
                self.metrics.observe("run_duration", float(duration))
        self.metrics.gauge("active_runs").dec()

    def _start_step(self, run_id: str, data: dict[str, Any]) -> None:
        """Open a span for one reasoning step."""
        trace = self._trace_for(run_id)
        if trace.root is None:
            return
        index = data.get("index", len(trace.root.children))
        span = trace.root.child(f"step.{index}", kind="step", **self._attributes(data))
        self._spans[f"{run_id}:step:{index}"] = span

    def _finish_step(self, run_id: str, data: dict[str, Any]) -> None:
        """Close the step span and record its duration."""
        index = data.get("index")
        key = f"{run_id}:step:{index}"
        span = self._spans.pop(key, None)
        if span is None:
            return
        span.set_attributes(self._attributes(data))
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)):
            span.ended_at = span.started_at + float(duration) / 1000.0
        span.finish()
        self.metrics.observe("step_duration", float(duration or span.duration_ms))

    def _tool_call(self, run_id: str, data: dict[str, Any]) -> None:
        """Open a span for a tool call."""
        trace = self._trace_for(run_id)
        name = str(data.get("name", "tool"))
        parent = trace.root.children[-1] if trace.root and trace.root.children else trace.root
        if parent is None:
            return
        span = parent.child(f"tool.{name}", kind="tool", **self._attributes(data))
        call_id = str(data.get("call_id", "") or span.span_id)
        self._spans[f"{run_id}:tool:{call_id}"] = span

    def _tool_result(self, run_id: str, data: dict[str, Any]) -> None:
        """Close the tool span and record tool metrics."""
        call_id = str(data.get("call_id", ""))
        span = self._spans.pop(f"{run_id}:tool:{call_id}", None)
        if span is None and call_id:
            for key, candidate in list(self._spans.items()):
                if key.startswith(f"{run_id}:tool:") and candidate.name.endswith(
                        str(data.get("name", ""))):
                    span = self._spans.pop(key)
                    break
        if span is not None:
            span.set_attributes(self._attributes(data))
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)):
                span.ended_at = span.started_at + float(duration) / 1000.0
            ok = bool(data.get("ok", True))
            span.finish("ok" if ok else "error")
            if not ok:
                self.errors += 1
        result = data.get("result")
        if result is not None and hasattr(result, "name"):
            record_tool(self.metrics, result)
        else:
            labels = {"tool": str(data.get("name", "unknown"))}
            self.metrics.inc("tool_calls_total", labels=labels)
            if not data.get("ok", True):
                self.metrics.inc("tool_errors_total", labels=labels)
            duration = data.get("duration_ms")
            if isinstance(duration, (int, float)):
                self.metrics.observe("tool_duration", float(duration), labels=labels)

    def _tool_error(self, run_id: str, data: dict[str, Any]) -> None:
        """Record a tool error event."""
        self.errors += 1
        self.metrics.inc("tool_errors_total", labels={"tool": str(data.get("name", "unknown"))})
        self._tool_result(run_id, {**data, "ok": False})

    def _llm_response(self, run_id: str, data: dict[str, Any]) -> None:
        """Record one LLM completion."""
        trace = self._run_traces.get(run_id)
        if trace is not None and trace.root is not None:
            span = trace.root.child("llm.response", kind="llm", **self._attributes(data))
            latency = data.get("latency_ms")
            if isinstance(latency, (int, float)):
                span.ended_at = span.started_at + float(latency) / 1000.0
            span.finish()
        response = data.get("response")
        if response is not None and hasattr(response, "usage"):
            record_llm(self.metrics, response)
        else:
            self.metrics.inc("llm_requests_total",
                             labels={"model": str(data.get("model", "unknown"))})
            latency = data.get("latency_ms")
            if isinstance(latency, (int, float)):
                self.metrics.observe("llm_latency", float(latency))

    # -- views ---------------------------------------------------------------

    def trace_for_events(self, events: Any, *, name: str = "run") -> Trace:
        """Rebuild a trace from an event list (used by ``magent trace``)."""
        return trace_from_events(list(events), name=name)

    def report(self) -> ObservabilityReport:
        """Collect a snapshot of everything observed so far."""
        traces = self.tracer.traces(limit=64)
        slowest: list[dict[str, Any]] = []
        for trace in traces[-3:]:
            for span in trace.slowest(3):
                slowest.append({"trace": trace.name, "span": span.name,
                                "duration_ms": round(span.duration_ms, 2)})
        slowest.sort(key=lambda item: -item["duration_ms"])
        return ObservabilityReport(
            uptime_seconds=time.time() - self.started_at,
            runs=self.runs,
            traces=len(traces),
            spans=sum(len(trace.spans()) for trace in traces),
            errors=self.errors,
            metrics=len(self.metrics),
            log_file=str(getattr(self.section, "log_file", "") or ""),
            trace_file=str(getattr(self.section, "trace_file", "") or ""),
            counters=self.metrics.flat(),
            slowest=slowest[:10],
        )

    def prometheus(self) -> str:
        """Render metrics for scraping."""
        return self.metrics.prometheus()

    def flame(self, run_id: str = "", *, width: int = 60) -> str:
        """Render the flame graph of one run, or the most recent."""
        trace = self._run_traces.get(run_id) if run_id else self.tracer.last_trace()
        if trace is None and not run_id:
            traces = self.tracer.traces(limit=1)
            trace = traces[-1] if traces else None
        return trace.flame(width=width) if trace is not None else ""

    def export_path(self) -> Path | None:
        """Where traces are written, when export is configured."""
        return self.tracer.export_path

    def close(self) -> None:
        """Detach subscriptions and finish any open trace."""
        self.detach()
        for trace in list(self._run_traces.values()):
            trace.finish("cancelled")
            self.tracer.export(trace)
        self._run_traces.clear()
        self._spans.clear()

    def describe(self) -> dict[str, Any]:
        """Configuration description for ``magent doctor``."""
        section = self.section
        return {
            "log_level": str(getattr(section, "log_level", "INFO")),
            "log_format": str(getattr(section, "log_format", "text")),
            "log_file": str(getattr(section, "log_file", "") or ""),
            "trace_enabled": bool(getattr(section, "trace_enabled", True)),
            "trace_file": str(getattr(section, "trace_file", "") or ""),
            "metrics_enabled": bool(getattr(section, "metrics_enabled", True)),
            "redact_secrets": bool(getattr(section, "redact_secrets", True)),
            "subscriptions": len(self.subscriptions),
            "tracer": self.tracer.stats(),
            "metrics": len(self.metrics),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (f"Observability(runs={self.runs}, metrics={len(self.metrics)}, "
                f"subscriptions={len(self.subscriptions)})")


def build_observability(config: Any = None, *, attach_to: Any = None) -> Observability:
    """Build observability from config and optionally attach it to an agent."""
    instance = Observability.from_config(config)
    if attach_to is not None:
        instance.attach(attach_to)
    return instance


__all__ = [
    "SKIP_KINDS",
    "TRACE_ATTRIBUTES",
    "Observability",
    "ObservabilityReport",
    "build_observability",
]
