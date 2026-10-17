"""Observability: metrics, traces, and structured logging in one place.

Three views of the same run, each answering a different question:

- :mod:`magent.observability.metrics` answers "is it getting worse?" - counters
  and latency histograms, exportable as Prometheus text.
- :mod:`magent.observability.tracing` answers "where did the time go?" - nested
  spans forming a tree per run.
- :mod:`magent.logging_` answers "what happened?" - level-filtered, secret
  redacting log records, as text or JSON.

All three are off by default in the sense that they cost nothing when unused:
an ``enabled=False`` tracer hands out inert spans, and a disabled registry
turns every increment into a no-op.
"""

from __future__ import annotations

from ..logging_ import (
    JsonFormatter,
    LogCapture,
    RedactingFilter,
    TextFormatter,
    env_log_level,
    get_logger,
    log_file_size,
    setup_logging,
    tail_log,
)
from .metrics import (
    DEFAULT_BUCKETS,
    METRIC_TYPES,
    Counter,
    Gauge,
    Histogram,
    MetricMeta,
    MetricsRegistry,
    Timer,
    standard_registry,
)
from .tracing import (
    MAX_EVENTS_PER_SPAN,
    STATUSES,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNSET,
    Span,
    SpanEvent,
    Tracer,
    current_span,
    iter_trace,
    load_spans,
    tracer_from_config,
)


def build_observability(config: object | None = None, *,
                        metrics: MetricsRegistry | None = None,
                        tracer: Tracer | None = None) -> tuple[MetricsRegistry, Tracer]:
    """Build the metrics registry and tracer implied by a configuration.

    Args:
        config: a resolved :class:`~magent.config.Config`, may be ``None``
        metrics: registry to use instead of building one
        tracer: tracer to use instead of building one

    Returns:
        The ``(registry, tracer)`` pair the rest of the app should share.
    """
    section = getattr(config, "observability", None)
    active_metrics = bool(getattr(section, "metrics_enabled", True))
    registry = metrics or standard_registry(enabled=active_metrics)
    active_tracer = tracer or tracer_from_config(config)
    return registry, active_tracer


__all__ = [
    "DEFAULT_BUCKETS",
    "MAX_EVENTS_PER_SPAN",
    "METRIC_TYPES",
    "STATUSES",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_UNSET",
    "Counter",
    "Gauge",
    "Histogram",
    "JsonFormatter",
    "LogCapture",
    "MetricMeta",
    "MetricsRegistry",
    "RedactingFilter",
    "Span",
    "SpanEvent",
    "TextFormatter",
    "Timer",
    "Tracer",
    "build_observability",
    "current_span",
    "env_log_level",
    "get_logger",
    "iter_trace",
    "load_spans",
    "log_file_size",
    "setup_logging",
    "standard_registry",
    "tail_log",
    "tracer_from_config",
]
