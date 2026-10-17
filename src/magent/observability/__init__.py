"""Tracing, metrics and logging wiring.

Public API:

:class:`Tracer` / :class:`Trace` / :class:`Span`
    span trees with an ASCII flame graph and JSONL export;
:class:`MetricsRegistry` / :class:`Counter` / :class:`Gauge` / :class:`Histogram`
    in-process metrics with a Prometheus text export;
:class:`Observability`
    applies ``ObservabilityConfig``, attaches to an agent's event bus, and turns
    events into spans and counters.

>>> from magent.observability import MetricsRegistry
>>> registry = MetricsRegistry()
>>> registry.inc("runs_total", labels={"model": "echo"})
1.0
>>> registry.counter("runs_total", labels={"model": "echo"}).value()
1.0
"""

from __future__ import annotations

from .metrics import (DEFAULT_BUCKETS, METRIC_KINDS, Counter, Gauge, Histogram, Metric,
                      MetricsRegistry, iter_metrics, record_llm, record_run, record_tool)
from .setup import (SKIP_KINDS, TRACE_ATTRIBUTES, Observability, ObservabilityReport,
                    build_observability)
from .tracing import (MAX_EVENTS_PER_SPAN, MAX_SPANS_PER_TRACE, SPAN_STATUS, Span, SpanEvent,
                      Trace, Tracer, trace_from_events)

__all__ = [
    "DEFAULT_BUCKETS",
    "MAX_EVENTS_PER_SPAN",
    "MAX_SPANS_PER_TRACE",
    "METRIC_KINDS",
    "SKIP_KINDS",
    "SPAN_STATUS",
    "TRACE_ATTRIBUTES",
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "MetricsRegistry",
    "Observability",
    "ObservabilityReport",
    "Span",
    "SpanEvent",
    "Trace",
    "Tracer",
    "build_observability",
    "iter_metrics",
    "record_llm",
    "record_run",
    "record_tool",
    "trace_from_events",
]
