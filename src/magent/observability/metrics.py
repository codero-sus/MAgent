"""Metrics: counters, gauges, and histograms with a Prometheus text export.

The point of collecting metrics in-process is that an agent run is expensive
and non-deterministic, so "it felt slow" is not a diagnosis. Latency
histograms expose percentiles, counters expose retry storms, and gauges expose
a queue that is growing rather than draining.

Export is plain text in the Prometheus exposition format, which means a run can
be scraped, diffed between two versions, or pasted into an issue without any
metrics infrastructure existing at all.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..logging_ import get_logger
from ..utils.files import atomic_write_text, ensure_dir
from ..utils.timeutil import format_duration

LOGGER = get_logger("observability.metrics")
"""Logger for metrics collection faults."""

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)
"""Histogram bucket boundaries in seconds, tuned for request latency."""

METRIC_TYPES: tuple[str, ...] = ("counter", "gauge", "histogram")
"""Metric kinds this registry understands."""


@dataclass(slots=True)
class MetricMeta:
    """Name, type, and help text for one metric."""

    name: str
    kind: str = "counter"
    help: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        """Normalize the metric kind."""
        if self.kind not in METRIC_TYPES:
            self.kind = "counter"

    @property
    def prometheus_type(self) -> str:
        """The type name used in the exposition format."""
        return self.kind


class Counter:
    """A monotonically increasing value."""

    def __init__(self, name: str, *, help: str = "", unit: str = "") -> None:
        """Create a counter.

        Args:
            name: metric name, ``snake_case``
            help: one-line description used in exports
            unit: unit label, e.g. ``tokens`` or ``requests``
        """
        self.meta = MetricMeta(name=name, kind="counter", help=help, unit=unit)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Metric name."""
        return self.meta.name

    def inc(self, amount: float = 1.0, **labels: str) -> float:
        """Add to the counter, returning the new value.

        Args:
            amount: increment; negative values are ignored, since a counter
                that can go down is a gauge
            labels: dimension values, e.g. ``model="qwen"``
        """
        if amount < 0:
            LOGGER.debug("counter %s ignoring negative increment %s", self.name, amount)
            return self.value(**labels)
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)
            return self._values[key]

    def value(self, **labels: str) -> float:
        """Return the counter's current value for a label set."""
        with self._lock:
            return self._values.get(_label_key(labels), 0.0)

    def total(self) -> float:
        """Return the sum across every label set."""
        with self._lock:
            return sum(self._values.values())

    def reset(self) -> None:
        """Clear every label set."""
        with self._lock:
            self._values.clear()

    def series(self) -> list[dict[str, Any]]:
        """Return one record per label set."""
        with self._lock:
            return [{"labels": dict(key), "value": value} for key, value in self._values.items()]

    def __iter__(self) -> Iterator[tuple[dict[str, str], float]]:
        """Iterate ``(labels, value)`` pairs."""
        return iter([(dict(key), value) for key, value in self.series_dict().items()])

    def series_dict(self) -> dict[tuple[tuple[str, str], ...], float]:
        """Return the raw label-keyed values."""
        with self._lock:
            return dict(self._values)


class Gauge:
    """A value that can rise and fall."""

    def __init__(self, name: str, *, help: str = "", unit: str = "") -> None:
        """Create a gauge.

        Args:
            name: metric name
            help: one-line description used in exports
            unit: unit label
        """
        self.meta = MetricMeta(name=name, kind="gauge", help=help, unit=unit)
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Metric name."""
        return self.meta.name

    def set(self, value: float, **labels: str) -> float:
        """Set the gauge to an absolute value."""
        key = _label_key(labels)
        with self._lock:
            self._values[key] = float(value)
            return float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> float:
        """Increase the gauge."""
        return self._adjust(amount, labels)

    def dec(self, amount: float = 1.0, **labels: str) -> float:
        """Decrease the gauge."""
        return self._adjust(-amount, labels)

    def _adjust(self, amount: float, labels: Mapping[str, str]) -> float:
        """Apply a delta and return the new value."""
        key = _label_key(dict(labels))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)
            return self._values[key]

    def value(self, **labels: str) -> float:
        """Return the current value for a label set."""
        with self._lock:
            return self._values.get(_label_key(labels), 0.0)

    def reset(self) -> None:
        """Clear every label set."""
        with self._lock:
            self._values.clear()

    def series(self) -> list[dict[str, Any]]:
        """Return one record per label set."""
        with self._lock:
            return [{"labels": dict(key), "value": value} for key, value in self._values.items()]


class Histogram:
    """A latency or size distribution with fixed buckets and percentiles."""

    def __init__(self, name: str, *, help: str = "", unit: str = "seconds",
                 buckets: Sequence[float] = DEFAULT_BUCKETS) -> None:
        """Create a histogram.

        Args:
            name: metric name
            help: one-line description used in exports
            unit: unit label
            buckets: ascending upper bounds, per label set
        """
        self.meta = MetricMeta(name=name, kind="histogram", help=help, unit=unit)
        self.buckets = tuple(sorted({float(value) for value in buckets})) or DEFAULT_BUCKETS
        self._observations: dict[tuple[tuple[str, str], ...], list[float]] = {}
        self._counts: dict[tuple[tuple[str, str], ...], int] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Metric name."""
        return self.meta.name

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation."""
        key = _label_key(labels)
        number = float(value)
        with self._lock:
            self._observations.setdefault(key, []).append(number)
            self._counts[key] = self._counts.get(key, 0) + 1
            self._sums[key] = self._sums.get(key, 0.0) + number

    def count(self, **labels: str) -> int:
        """Return how many observations were recorded."""
        with self._lock:
            return self._counts.get(_label_key(labels), 0)

    def sum(self, **labels: str) -> float:
        """Return the sum of observations."""
        with self._lock:
            return self._sums.get(_label_key(labels), 0.0)

    def mean(self, **labels: str) -> float:
        """Return the arithmetic mean, or 0 when empty."""
        total = self.count(**labels)
        return self.sum(**labels) / total if total else 0.0

    def percentile(self, fraction: float, **labels: str) -> float:
        """Return one percentile of the observations.

        Args:
            fraction: percentile in ``0..1``, e.g. 0.95 for p95
        """
        key = _label_key(labels)
        with self._lock:
            values = sorted(self._observations.get(key, []))
        if not values:
            return 0.0
        ratio = min(1.0, max(0.0, float(fraction)))
        index = min(len(values) - 1, max(0, int(round(ratio * (len(values) - 1)))))
        return values[index]

    def summary(self, **labels: str) -> dict[str, float]:
        """Return count, sum, mean, min, max, and common percentiles."""
        key = _label_key(labels)
        with self._lock:
            values = sorted(self._observations.get(key, []))
        if not values:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0,
                    "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "count": len(values),
            "sum": round(sum(values), 6),
            "mean": round(sum(values) / len(values), 6),
            "min": values[0],
            "max": values[-1],
            "p50": self.percentile(0.50, **labels),
            "p90": self.percentile(0.90, **labels),
            "p95": self.percentile(0.95, **labels),
            "p99": self.percentile(0.99, **labels),
        }

    def bucket_counts(self, **labels: str) -> list[tuple[float, int]]:
        """Return cumulative ``(upper_bound, count)`` pairs for the exposition format."""
        key = _label_key(labels)
        with self._lock:
            values = self._observations.get(key, [])
        cumulative: list[tuple[float, int]] = []
        for bound in self.buckets:
            cumulative.append((bound, sum(1 for value in values if value <= bound)))
        cumulative.append((float("inf"), len(values)))
        return cumulative

    def reset(self) -> None:
        """Clear every observation."""
        with self._lock:
            self._observations.clear()
            self._counts.clear()
            self._sums.clear()

    def series(self) -> list[dict[str, Any]]:
        """Return one summary record per label set."""
        with self._lock:
            keys = list(self._observations)
        return [{"labels": dict(key), **self.summary(**dict(key))} for key in keys]


class Timer:
    """Context manager and decorator that observes elapsed seconds."""

    def __init__(self, histogram: Histogram | None = None, *, name: str = "",
                 registry: MetricsRegistry | None = None, **labels: str) -> None:
        """Create a timer.

        Args:
            histogram: target histogram; looked up by ``name`` when omitted
            name: histogram name to resolve from ``registry``
            registry: registry providing the histogram
            labels: dimension values recorded with the observation
        """
        self.histogram = histogram
        self.name = name
        self.registry = registry
        self.labels = dict(labels)
        self.elapsed = 0.0
        self._started = 0.0

    def _target(self) -> Histogram | None:
        """Resolve the histogram this timer writes to."""
        if self.histogram is not None:
            return self.histogram
        if self.registry is not None and self.name:
            metric = self.registry.get(self.name)
            return metric if isinstance(metric, Histogram) else None
        return None

    def start(self) -> Timer:
        """Begin timing."""
        self._started = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop timing, record the observation, and return the seconds."""
        self.elapsed = max(0.0, time.perf_counter() - self._started) if self._started else 0.0
        target = self._target()
        if target is not None:
            target.observe(self.elapsed, **self.labels)
        self._started = 0.0
        return self.elapsed

    def __enter__(self) -> Timer:
        """Begin timing."""
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        """Record the elapsed time even when the block raised."""
        self.stop()

    def __call__(self, fn: Any) -> Any:
        """Wrap a callable so every invocation is timed."""
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(self._target(), registry=self.registry, name=self.name, **self.labels):
                return fn(*args, **kwargs)

        return wrapper


class MetricsRegistry:
    """A thread-safe collection of named metrics with export and reset."""

    def __init__(self, *, enabled: bool = True, prefix: str = "magent") -> None:
        """Create a registry.

        Args:
            enabled: when false, every operation is a cheap no-op
            prefix: prepended to metric names in exports
        """
        self.enabled = bool(enabled)
        self.prefix = str(prefix or "").strip()
        self._metrics: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.created_at = time.time()

    # -- registration --------------------------------------------------------

    def counter(self, name: str, *, help: str = "", unit: str = "") -> Counter:
        """Return an existing counter or create one."""
        return self._get_or_create(name, Counter, help=help, unit=unit)

    def gauge(self, name: str, *, help: str = "", unit: str = "") -> Gauge:
        """Return an existing gauge or create one."""
        return self._get_or_create(name, Gauge, help=help, unit=unit)

    def histogram(self, name: str, *, help: str = "", unit: str = "seconds",
                  buckets: Sequence[float] = DEFAULT_BUCKETS) -> Histogram:
        """Return an existing histogram or create one."""
        return self._get_or_create(name, Histogram, help=help, unit=unit, buckets=buckets)

    def register(self, metric: Any) -> Any:
        """Register a pre-built metric object."""
        with self._lock:
            self._metrics[metric.name] = metric
        return metric

    def _get_or_create(self, name: str, factory: Any, **kwargs: Any) -> Any:
        """Return the named metric, constructing it on first use."""
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                return existing
            metric = factory(name, **kwargs)
            self._metrics[name] = metric
            return metric

    def get(self, name: str) -> Any | None:
        """Return one metric by name."""
        with self._lock:
            return self._metrics.get(name)

    def names(self) -> list[str]:
        """Return every registered metric name, sorted."""
        with self._lock:
            return sorted(self._metrics)

    def __len__(self) -> int:
        """Number of registered metrics."""
        with self._lock:
            return len(self._metrics)

    def __contains__(self, name: str) -> bool:
        """Whether a metric is registered."""
        with self._lock:
            return name in self._metrics

    # -- convenience ---------------------------------------------------------

    def incr(self, name: str, amount: float = 1.0, **labels: str) -> float:
        """Increment a counter, creating it if needed."""
        if not self.enabled:
            return 0.0
        return self.counter(name).inc(amount, **labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation, creating it if needed."""
        if not self.enabled:
            return
        self.histogram(name).observe(value, **labels)

    def set_gauge(self, name: str, value: float, **labels: str) -> float:
        """Set a gauge, creating it if needed."""
        if not self.enabled:
            return 0.0
        return self.gauge(name).set(value, **labels)

    def timer(self, name: str, **labels: str) -> Timer:
        """Return a :class:`Timer` bound to a histogram of this name."""
        return Timer(None, name=name, registry=self, **labels)

    def time(self, name: str, **labels: str) -> Timer:
        """Alias for :meth:`timer`, for use directly in a ``with`` block."""
        return self.timer(name, **labels)

    # -- export --------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return every metric as plain data."""
        with self._lock:
            metrics = list(self._metrics.values())
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "uptime_seconds": round(time.time() - self.created_at, 3),
            "metrics": {},
        }
        for metric in metrics:
            if isinstance(metric, Histogram):
                payload["metrics"][metric.name] = {
                    "kind": "histogram", "buckets": list(metric.buckets),
                    "series": metric.series(),
                }
            else:
                payload["metrics"][metric.name] = {"kind": metric.meta.kind,
                                                   "series": metric.series()}
        return payload

    def render(self, *, prometheus: bool = True) -> str:
        """Render metrics as text.

        Args:
            prometheus: use the exposition format; otherwise a readable summary
        """
        with self._lock:
            metrics = list(self._metrics.values())
        if prometheus:
            return self._render_prometheus(metrics)
        return self._render_text(metrics)

    def _render_prometheus(self, metrics: Iterable[Any]) -> str:
        """Emit the Prometheus text exposition format."""
        lines: list[str] = []
        for metric in metrics:
            full = f"{self.prefix}_{metric.name}" if self.prefix else metric.name
            if metric.meta.help:
                lines.append(f"# HELP {full} {_escape_help(metric.meta.help)}")
            lines.append(f"# TYPE {full} {metric.meta.prometheus_type}")
            if isinstance(metric, Histogram):
                for record in metric.series():
                    labels = record["labels"]
                    for bound, count in metric.bucket_counts(**labels):
                        le = "+Inf" if bound == float("inf") else _format_number(bound)
                        lines.append(f"{full}_bucket{{{_labels(labels, extra={'le': le})}}} "
                                     f"{count}")
                    lines.append(f"{full}_sum{{{_labels(labels)}}} "
                                 f"{_format_number(record['sum'])}")
                    lines.append(f"{full}_count{{{_labels(labels)}}} {record['count']}")
            else:
                for record in metric.series():
                    lines.append(f"{full}{{{_labels(record['labels'])}}} "
                                 f"{_format_number(record['value'])}")
        return "\n".join(lines) + "\n" if lines else ""

    def _render_text(self, metrics: Iterable[Any]) -> str:
        """Emit a human-readable summary."""
        lines: list[str] = []
        for metric in metrics:
            if isinstance(metric, Histogram):
                for record in metric.series():
                    suffix = f" {_format_labels(record['labels'])}" if record["labels"] else ""
                    lines.append(
                        f"{metric.name}{suffix}: n={record['count']} "
                        f"mean={_format_number(record['mean'])}s "
                        f"p95={_format_number(record['p95'])}s "
                        f"max={_format_number(record['max'])}s"
                    )
            else:
                for record in metric.series():
                    suffix = f" {_format_labels(record['labels'])}" if record["labels"] else ""
                    lines.append(f"{metric.name}{suffix}: {_format_number(record['value'])}")
        return "\n".join(lines)

    def write(self, path: str, *, prometheus: bool = True) -> str:
        """Write the rendered metrics to a file, returning the path."""
        from pathlib import Path

        target = Path(path).expanduser()
        ensure_dir(target.parent)
        atomic_write_text(target, self.render(prometheus=prometheus))
        return str(target)

    def reset(self) -> None:
        """Clear every metric's values, keeping registrations."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()
        self.created_at = time.time()

    def describe(self) -> str:
        """Return a one-line summary of the registry."""
        with self._lock:
            counts = {"counter": 0, "gauge": 0, "histogram": 0}
            for metric in self._metrics.values():
                counts[metric.meta.kind] = counts.get(metric.meta.kind, 0) + 1
        return (f"{len(self)} metrics ({counts['counter']} counter, {counts['gauge']} gauge, "
                f"{counts['histogram']} histogram) uptime "
                f"{format_duration(time.time() - self.created_at)}")


def standard_registry(*, enabled: bool = True, prefix: str = "magent") -> MetricsRegistry:
    """Return a registry pre-declared with MAgent's core metrics.

    Declaring them up front means an export before any traffic still shows the
    metric with a zero value, which distinguishes "nothing happened" from "this
    build does not collect it".
    """
    registry = MetricsRegistry(enabled=enabled, prefix=prefix)
    registry.counter("llm_requests_total", help="LLM requests sent", unit="requests")
    registry.counter("llm_errors_total", help="LLM requests that failed", unit="requests")
    registry.counter("llm_retries_total", help="LLM requests retried", unit="retries")
    registry.counter("tool_calls_total", help="Tool invocations", unit="calls")
    registry.counter("tool_errors_total", help="Tool invocations that failed", unit="calls")
    registry.counter("tokens_prompt_total", help="Prompt tokens consumed", unit="tokens")
    registry.counter("tokens_completion_total", help="Completion tokens consumed", unit="tokens")
    registry.counter("guardrail_findings_total", help="Guardrail findings by rule", unit="findings")
    registry.counter("cache_hits_total", help="Cache hits", unit="lookups")
    registry.counter("cache_misses_total", help="Cache misses", unit="lookups")
    registry.histogram("llm_latency_seconds", help="LLM request latency")
    registry.histogram("tool_latency_seconds", help="Tool execution latency")
    registry.histogram("run_duration_seconds", help="Whole-run duration")
    registry.gauge("active_runs", help="Runs currently in flight", unit="runs")
    registry.gauge("context_utilization", help="Fraction of the context window used")
    registry.gauge("memory_entries", help="Entries held in the memory store", unit="entries")
    return registry


def _label_key(labels: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return a hashable, order-independent key for a label set."""
    return tuple(sorted((str(key), str(value)) for key, value in labels.items() if value != ""))


def _labels(labels: Mapping[str, str], *, extra: Mapping[str, str] | None = None) -> str:
    """Render a label set in Prometheus syntax."""
    merged = {str(key): str(value) for key, value in dict(labels).items()}
    merged.update({str(key): str(value) for key, value in dict(extra or {}).items()})
    if not merged:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(value)}"'
                        for key, value in sorted(merged.items()))
    return rendered


def _escape_label(value: str) -> str:
    """Escape a label value for the exposition format."""
    return (value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))


def _escape_help(value: str) -> str:
    """Escape help text for the exposition format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _format_labels(labels: Mapping[str, Any]) -> str:
    """Render a label set compactly for human-readable output."""
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _format_number(value: float) -> str:
    """Render a number without trailing noise, keeping ``+Inf`` intact."""
    if value == float("inf"):
        return "+Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


__all__ = [
    "DEFAULT_BUCKETS",
    "METRIC_TYPES",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricMeta",
    "MetricsRegistry",
    "Timer",
    "standard_registry",
]
