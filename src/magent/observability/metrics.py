"""Metrics: counters, gauges and histograms with a Prometheus text export.

The agent already emits events; metrics turn that stream into numbers a person
can compare between runs - requests per model, tool success rate, p95 latency,
tokens per second, cost per run.  Everything is in-process and dependency-free,
and :meth:`MetricsRegistry.prometheus` renders the standard text format so the
same numbers can be scraped when MAgent is served over HTTP.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..utils.numbers import mean, percentile

DEFAULT_BUCKETS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)
"""Default histogram buckets, in milliseconds."""

METRIC_KINDS: tuple[str, ...] = ("counter", "gauge", "histogram")
"""Metric types this module implements."""


@dataclass(slots=True)
class Metric:
    """Base class for a named metric."""

    name: str
    help: str = ""
    unit: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        """Registry key: name plus sorted labels."""
        if not self.labels:
            return self.name
        joined = ",".join(f"{key}={self.labels[key]}" for key in sorted(self.labels))
        return f"{self.name}{{{joined}}}"

    @property
    def kind(self) -> str:
        """Metric type name."""
        return "metric"

    @property
    def prometheus_name(self) -> str:
        """Sample family name, including the unit suffix.

        Prometheus treats ``# TYPE`` and the samples as one family, so a metric
        recorded with ``unit="ms"`` must be announced as ``name_ms`` - otherwise
        a scraper rejects the exposition as malformed.
        """
        return f"{self.name}_{self.unit}" if self.unit else self.name

    def value(self) -> Any:
        """Current value."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metric."""
        return {
            "name": self.name,
            "kind": self.kind,
            "help": self.help,
            "unit": self.unit,
            "labels": dict(self.labels),
            "key": self.key,
            "value": self.value(),
        }

    def prometheus_lines(self) -> list[str]:
        """Lines in the Prometheus text exposition format."""
        return [f"{self.prometheus_name}{_label_text(self.labels)} {self._prometheus_value()}"]

    def _prometheus_value(self) -> str:
        """Value formatted for Prometheus."""
        return repr(float(self.value() or 0.0))


class Counter(Metric):
    """A value that only ever goes up."""

    def __init__(
        self, name: str, help: str = "", *, unit: str = "", labels: Mapping[str, str] | None = None
    ) -> None:
        """Create a counter at zero."""
        super().__init__(name=name, help=help, unit=unit, labels=dict(labels or {}))
        self._value = 0.0
        self._lock = threading.Lock()

    @property
    def kind(self) -> str:
        """Metric type name."""
        return "counter"

    def inc(self, amount: float = 1.0) -> float:
        """Increase the counter; negative amounts are ignored."""
        if amount < 0:
            return self._value
        with self._lock:
            self._value += float(amount)
            return self._value

    def value(self) -> float:
        """Current total."""
        return self._value

    def reset(self) -> None:
        """Return the counter to zero."""
        with self._lock:
            self._value = 0.0


class Gauge(Metric):
    """A value that goes up and down."""

    def __init__(
        self, name: str, help: str = "", *, unit: str = "", labels: Mapping[str, str] | None = None
    ) -> None:
        """Create a gauge at zero."""
        super().__init__(name=name, help=help, unit=unit, labels=dict(labels or {}))
        self._value = 0.0
        self._minimum = float("inf")
        self._maximum = float("-inf")
        self._samples = 0
        self._lock = threading.Lock()

    @property
    def kind(self) -> str:
        """Metric type name."""
        return "gauge"

    def set(self, amount: float) -> float:
        """Set the gauge to a value."""
        with self._lock:
            self._value = float(amount)
            self._samples += 1
            self._minimum = min(self._minimum, self._value)
            self._maximum = max(self._maximum, self._value)
            return self._value

    def inc(self, amount: float = 1.0) -> float:
        """Increase the gauge."""
        return self.set(self._value + amount)

    def dec(self, amount: float = 1.0) -> float:
        """Decrease the gauge."""
        return self.set(self._value - amount)

    def value(self) -> float:
        """Current value."""
        return self._value

    @property
    def minimum(self) -> float:
        """Lowest value seen."""
        return 0.0 if self._minimum == float("inf") else self._minimum

    @property
    def maximum(self) -> float:
        """Highest value seen."""
        return 0.0 if self._maximum == float("-inf") else self._maximum

    def stats(self) -> dict[str, Any]:
        """Value plus min/max/sample count."""
        return {
            "value": self._value,
            "min": self.minimum,
            "max": self.maximum,
            "samples": self._samples,
        }

    def reset(self) -> None:
        """Return the gauge to zero and forget extremes."""
        with self._lock:
            self._value = 0.0
            self._minimum = float("inf")
            self._maximum = float("-inf")
            self._samples = 0


class Histogram(Metric):
    """Latency-style observations with buckets and percentiles."""

    def __init__(
        self,
        name: str,
        help: str = "",
        *,
        unit: str = "ms",
        buckets: Sequence[float] = DEFAULT_BUCKETS,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Create an empty histogram."""
        super().__init__(name=name, help=help, unit=unit, labels=dict(labels or {}))
        self.buckets = tuple(sorted(float(item) for item in buckets))
        self._counts = [0] * (len(self.buckets) + 1)
        self._samples: list[float] = []
        self._sum = 0.0
        self._max_samples = 2_000
        self._lock = threading.Lock()

    @property
    def kind(self) -> str:
        """Metric type name."""
        return "histogram"

    def observe(self, amount: float) -> None:
        """Record one observation."""
        value = float(amount)
        with self._lock:
            self._sum += value
            placed = False
            for index, edge in enumerate(self.buckets):
                if value <= edge:
                    self._counts[index] += 1
                    placed = True
                    break
            if not placed:
                self._counts[-1] += 1
            if len(self._samples) < self._max_samples:
                self._samples.append(value)

    @property
    def count(self) -> int:
        """Number of observations."""
        return sum(self._counts)

    @property
    def total(self) -> float:
        """Sum of all observations."""
        return self._sum

    def value(self) -> float:
        """Mean observation; Prometheus exposes buckets separately."""
        return round(self._sum / self.count, 4) if self.count else 0.0

    def percentile(self, which: float) -> float:
        """Percentile of the retained samples."""
        if not self._samples:
            return 0.0
        return round(percentile(self._samples, which), 4)

    @property
    def p50(self) -> float:
        """Median observation."""
        return self.percentile(50)

    @property
    def p95(self) -> float:
        """95th percentile observation."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """99th percentile observation."""
        return self.percentile(99)

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "count": self.count,
            "sum": round(self._sum, 4),
            "mean": round(mean(self._samples), 4) if self._samples else 0.0,
            "min": round(min(self._samples), 4) if self._samples else 0.0,
            "max": round(max(self._samples), 4) if self._samples else 0.0,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "unit": self.unit,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the histogram with its bucket counts."""
        payload = super().to_dict()
        payload["stats"] = self.stats()
        payload["buckets"] = {
            str(edge): count for edge, count in zip(self.buckets, self._counts[:-1], strict=False)
        }
        payload["overflow"] = self._counts[-1]
        return payload

    def prometheus_lines(self) -> list[str]:
        """Emit ``_bucket``, ``_sum`` and ``_count`` lines."""
        base = self.prometheus_name
        plain = _label_text(self.labels)
        lines: list[str] = []
        cumulative = 0
        for edge, count in zip(self.buckets, self._counts[:-1], strict=False):
            cumulative += count
            lines.append(
                f"{base}_bucket{_label_text({**self.labels, 'le': str(edge)})} {cumulative}"
            )
        cumulative += self._counts[-1]
        lines.append(f"{base}_bucket{_label_text({**self.labels, 'le': '+Inf'})} {cumulative}")
        lines.append(f"{base}_sum{plain} {self._sum}")
        lines.append(f"{base}_count{plain} {self.count}")
        return lines

    def reset(self) -> None:
        """Forget every observation."""
        with self._lock:
            self._counts = [0] * (len(self.buckets) + 1)
            self._samples.clear()
            self._sum = 0.0


def _label_text(labels: Mapping[str, str]) -> str:
    """Render a label set as Prometheus ``{k="v"}`` text, empty when unlabelled."""
    if not labels:
        return ""
    body = ",".join(f'{key}="{labels[key]}"' for key in sorted(labels))
    return "{" + body + "}"


class MetricsRegistry:
    """A thread-safe collection of metrics."""

    def __init__(self, *, prefix: str = "magent") -> None:
        """Create an empty registry.

        Args:
            prefix: namespace prepended to every metric name
        """
        self.prefix = prefix
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    def _key(self, name: str, labels: Mapping[str, str] | None) -> str:
        """Registry key for a metric plus labels."""
        full = f"{self.prefix}_{name}" if self.prefix and not name.startswith(self.prefix) else name
        if not labels:
            return full
        joined = ",".join(f"{key}={labels[key]}" for key in sorted(labels))
        return f"{full}{{{joined}}}"

    def _full(self, name: str) -> str:
        """Namespaced metric name."""
        return f"{self.prefix}_{name}" if self.prefix and not name.startswith(self.prefix) else name

    def counter(
        self, name: str, help: str = "", *, unit: str = "", labels: Mapping[str, str] | None = None
    ) -> Counter:
        """Return an existing counter or create one."""
        key = self._key(name, labels)
        with self._lock:
            metric = self._metrics.get(key)
            if isinstance(metric, Counter):
                return metric
            created = Counter(self._full(name), help, unit=unit, labels=labels)
            self._metrics[key] = created
            return created

    def gauge(
        self, name: str, help: str = "", *, unit: str = "", labels: Mapping[str, str] | None = None
    ) -> Gauge:
        """Return an existing gauge or create one."""
        key = self._key(name, labels)
        with self._lock:
            metric = self._metrics.get(key)
            if isinstance(metric, Gauge):
                return metric
            created = Gauge(self._full(name), help, unit=unit, labels=labels)
            self._metrics[key] = created
            return created

    def histogram(
        self,
        name: str,
        help: str = "",
        *,
        unit: str = "ms",
        buckets: Sequence[float] = DEFAULT_BUCKETS,
        labels: Mapping[str, str] | None = None,
    ) -> Histogram:
        """Return an existing histogram or create one."""
        key = self._key(name, labels)
        with self._lock:
            metric = self._metrics.get(key)
            if isinstance(metric, Histogram):
                return metric
            created = Histogram(self._full(name), help, unit=unit, buckets=buckets, labels=labels)
            self._metrics[key] = created
            return created

    def inc(
        self, name: str, amount: float = 1.0, *, labels: Mapping[str, str] | None = None
    ) -> float:
        """Increment a counter in one call."""
        return self.counter(name, labels=labels).inc(amount)

    def observe(
        self, name: str, amount: float, *, unit: str = "ms", labels: Mapping[str, str] | None = None
    ) -> None:
        """Record a histogram observation in one call."""
        self.histogram(name, unit=unit, labels=labels).observe(amount)

    def set(self, name: str, amount: float, *, labels: Mapping[str, str] | None = None) -> float:
        """Set a gauge in one call."""
        return self.gauge(name, labels=labels).set(amount)

    def get(self, name: str, labels: Mapping[str, str] | None = None) -> Metric | None:
        """Look one metric up."""
        with self._lock:
            return self._metrics.get(self._key(name, labels))

    def metrics(self) -> list[Metric]:
        """Every registered metric, sorted by key."""
        with self._lock:
            return [self._metrics[key] for key in sorted(self._metrics)]

    def names(self) -> list[str]:
        """Distinct metric names."""
        with self._lock:
            return sorted({metric.name for metric in self._metrics.values()})

    def snapshot(self) -> dict[str, Any]:
        """Serialise every metric."""
        metrics = self.metrics()
        return {
            "prefix": self.prefix,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "metrics": len(metrics),
            "values": {metric.key: metric.to_dict() for metric in metrics},
        }

    def flat(self) -> dict[str, Any]:
        """Flat ``key -> value`` mapping, handy for logs and JSON output."""
        return {metric.key: metric.value() for metric in self.metrics()}

    def prometheus(self) -> str:
        """Render all metrics in the Prometheus text exposition format."""
        lines: list[str] = []
        seen_help: set[str] = set()
        for metric in self.metrics():
            family = metric.prometheus_name
            if family not in seen_help:
                if metric.help:
                    lines.append(f"# HELP {family} {metric.help}")
                lines.append(f"# TYPE {family} {metric.kind}")
                seen_help.add(family)
            lines.extend(metric.prometheus_lines())
        lines.append(f"{self.prefix}_uptime_seconds {time.time() - self.started_at:.3f}")
        return "\n".join(lines) + "\n"

    def render_table(self, *, limit: int = 40) -> list[tuple[str, str, str]]:
        """Rows of ``(key, kind, value)`` for CLI tables."""
        rows: list[tuple[str, str, str]] = []
        for metric in self.metrics()[:limit]:
            if isinstance(metric, Histogram):
                value = f"n={metric.count} mean={metric.value()}{metric.unit} p95={metric.p95}"
            elif isinstance(metric, Gauge):
                value = f"{metric.value():g} (min {metric.minimum:g}, max {metric.maximum:g})"
            else:
                value = f"{metric.value():g}"
            rows.append((metric.key, metric.kind, value))
        return rows

    def reset(self) -> None:
        """Reset every metric."""
        for metric in self.metrics():
            metric.reset()
        self.started_at = time.time()

    def merge(self, other: MetricsRegistry) -> int:
        """Merge another registry's counters, gauges and histograms into this one."""
        merged = 0
        for metric in other.metrics():
            if isinstance(metric, Counter):
                self.counter(metric.name, metric.help, unit=metric.unit, labels=metric.labels).inc(
                    metric.value()
                )
            elif isinstance(metric, Gauge):
                self.gauge(metric.name, metric.help, unit=metric.unit, labels=metric.labels).set(
                    metric.value()
                )
            elif isinstance(metric, Histogram):
                target = self.histogram(
                    metric.name,
                    metric.help,
                    unit=metric.unit,
                    buckets=metric.buckets,
                    labels=metric.labels,
                )
                for sample in metric._samples:
                    target.observe(sample)
            merged += 1
        return merged

    def __len__(self) -> int:
        """Number of registered metrics."""
        with self._lock:
            return len(self._metrics)

    def __contains__(self, name: object) -> bool:
        """True when a metric name is registered."""
        return self.get(str(name)) is not None

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"MetricsRegistry(prefix={self.prefix!r}, metrics={len(self)})"


def record_run(registry: MetricsRegistry, result: Any) -> MetricsRegistry:
    """Fold a :class:`~magent.types.RunResult` into a registry.

    Args:
        registry: metrics to update
        result: the finished run
    """
    labels = {"model": str(getattr(result, "model", "") or "unknown")}
    registry.inc("runs_total", labels=labels)
    stop = str(
        getattr(getattr(result, "stop_reason", ""), "value", getattr(result, "stop_reason", ""))
        or "unknown"
    )
    registry.inc("runs_by_stop_reason", labels={**labels, "reason": stop})
    if getattr(result, "succeeded", False):
        registry.inc("runs_succeeded_total", labels=labels)
    else:
        registry.inc("runs_failed_total", labels=labels)
    registry.observe("run_duration", getattr(result, "duration_ms", 0.0), labels=labels)
    usage = getattr(result, "usage", None)
    if usage is not None:
        registry.inc("prompt_tokens_total", getattr(usage, "prompt_tokens", 0), labels=labels)
        registry.inc(
            "completion_tokens_total", getattr(usage, "completion_tokens", 0), labels=labels
        )
        registry.inc("tokens_total", getattr(usage, "total_tokens", 0), labels=labels)
        if getattr(usage, "cached_tokens", 0):
            registry.inc("cached_tokens_total", usage.cached_tokens, labels=labels)
    cost = getattr(result, "cost", None)
    total = getattr(cost, "total", 0.0) or 0.0
    if total:
        registry.inc(
            "cost_total", total, labels={**labels, "unit": getattr(cost, "currency", "USD")}
        )
    steps = getattr(result, "steps", []) or []
    registry.gauge("last_run_steps", labels=labels).set(len(steps))
    for step in steps:
        for item in getattr(step, "tool_results", []) or []:
            record_tool(registry, item)
    return registry


def record_tool(registry: MetricsRegistry, result: Any) -> MetricsRegistry:
    """Fold a :class:`~magent.types.ToolResult` into a registry."""
    name = str(getattr(result, "name", "") or "unknown")
    labels = {"tool": name}
    registry.inc("tool_calls_total", labels=labels)
    if getattr(result, "ok", False):
        registry.inc("tool_success_total", labels=labels)
    else:
        code = str(getattr(result, "error_code", "") or "error")
        registry.inc("tool_errors_total", labels={**labels, "code": code})
    registry.observe("tool_duration", getattr(result, "duration_ms", 0.0), labels=labels)
    if getattr(result, "truncated", False):
        registry.inc("tool_truncated_total", labels=labels)
    return registry


def record_llm(registry: MetricsRegistry, response: Any) -> MetricsRegistry:
    """Fold a :class:`~magent.types.ChatResponse` into a registry."""
    labels = {"model": str(getattr(response, "model", "") or "unknown")}
    registry.inc("llm_requests_total", labels=labels)
    registry.observe("llm_latency", getattr(response, "latency_ms", 0.0), labels=labels)
    if getattr(response, "cache_hit", False):
        registry.inc("llm_cache_hits_total", labels=labels)
    return registry


def iter_metrics(payload: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """Yield ``(key, value)`` pairs from a snapshot, flattening nested dicts."""
    for key, item in payload.items():
        if isinstance(item, Mapping):
            for sub_key, value in iter_metrics(item):
                yield f"{key}.{sub_key}", value
        else:
            yield key, item


__all__ = [
    "DEFAULT_BUCKETS",
    "METRIC_KINDS",
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "MetricsRegistry",
    "iter_metrics",
    "record_llm",
    "record_run",
    "record_tool",
]
