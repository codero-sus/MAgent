"""Timing, duration parsing, and deadline arithmetic."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

_DURATION = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ns|us|µs|ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks)",
    re.IGNORECASE,
)
"""Regex matching a single duration component such as ``1.5s``."""

_UNIT_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "w": 604800.0,
    "wk": 604800.0,
    "week": 604800.0,
    "weeks": 604800.0,
}
"""Seconds per unit name."""


def parse_duration(value: str | float | int | None, *, default: float = 0.0) -> float:
    """Parse ``"90s"``, ``"5m"``, ``"1.5h"`` or ``"1d 2h"`` into seconds.

    Bare numbers are treated as seconds. Unparseable input yields ``default``
    instead of raising, because durations usually come from user config.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    matched = False
    for match in _DURATION.finditer(text):
        unit = match.group("unit")
        total += float(match.group("value")) * _UNIT_SECONDS.get(unit, 1.0)
        matched = True
    return total if matched else default


def format_duration(seconds: float, *, precision: int = 2, compact: bool = True) -> str:
    """Format a second count as ``1h 2m 3s`` or ``3.25s``."""
    value = abs(float(seconds))
    sign = "-" if seconds < 0 else ""
    if value < 1e-6:
        return f"{sign}0s"
    if value < 1e-3:
        return f"{sign}{value * 1e6:.0f}us"
    if value < 1.0:
        return f"{sign}{value * 1e3:.0f}ms"
    if compact and value < 60:
        return f"{sign}{value:.{precision}f}s"
    units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    remaining = value
    parts: list[str] = []
    for label, size in units:
        if remaining >= size:
            count = int(remaining // size)
            parts.append(f"{count}{label}")
            remaining -= count * size
    if remaining >= 0.5 and parts and parts[-1].endswith("s"):
        parts[-1] = f"{float(parts[-1][:-1]) + remaining:.0f}s"
    return sign + (" ".join(parts) if parts else f"{value:.{precision}f}s")


def monotonic_ms() -> float:
    """Return a monotonic clock reading in milliseconds."""
    return time.monotonic() * 1000.0


def deadline_from_timeout(timeout: float | None, *, from_time: float | None = None) -> float | None:
    """Convert a relative timeout into an absolute monotonic deadline."""
    if timeout is None or timeout <= 0:
        return None
    base = time.monotonic() if from_time is None else from_time
    return base + timeout


def time_remaining(deadline: float | None) -> float | None:
    """Return seconds left on a monotonic deadline, or ``None`` if open ended."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def is_expired(deadline: float | None) -> bool:
    """Report whether a monotonic deadline has passed."""
    return deadline is not None and time.monotonic() >= deadline


def format_relative(ts: float, *, now: float | None = None) -> str:
    """Format a UNIX timestamp as ``3m ago`` / ``in 2h``."""
    reference = time.time() if now is None else now
    delta = reference - ts
    future = delta < 0
    text = format_duration(abs(delta), precision=0)
    return f"in {text}" if future else f"{text} ago"


def format_timestamp(ts: float | None = None, *, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a UNIX timestamp in local time."""
    return time.strftime(fmt, time.localtime(time.time() if ts is None else ts))


def utc_iso(ts: float | None = None) -> str:
    """Format a UNIX timestamp as ISO-8601 UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if ts is None else ts))


def parse_iso(text: str) -> float:
    """Parse an ISO-8601 timestamp into UNIX seconds, tolerating a ``Z``."""
    cleaned = str(text).strip().replace("Z", "+00:00")
    from datetime import datetime

    try:
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {text}") from exc


@dataclass(slots=True)
class Stopwatch:
    """A pause/resume stopwatch with lap recording.

    Uses the monotonic clock so system time adjustments cannot corrupt
    measurements.
    """

    name: str = "stopwatch"
    _start: float = field(default_factory=time.monotonic)
    _accumulated: float = 0.0
    _running: bool = True
    laps: list[tuple[str, float]] = field(default_factory=list)

    def elapsed(self) -> float:
        """Return elapsed seconds, counting only running intervals."""
        extra = (time.monotonic() - self._start) if self._running else 0.0
        return self._accumulated + extra

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds."""
        return self.elapsed() * 1000.0

    def pause(self) -> float:
        """Stop accumulating time and return the elapsed total."""
        if self._running:
            self._accumulated += time.monotonic() - self._start
            self._running = False
        return self.elapsed()

    def resume(self) -> None:
        """Resume accumulating time after a pause."""
        if not self._running:
            self._start = time.monotonic()
            self._running = True

    def reset(self) -> None:
        """Zero the stopwatch and clear recorded laps."""
        self._accumulated = 0.0
        self._start = time.monotonic()
        self._running = True
        self.laps.clear()

    def lap(self, label: str) -> float:
        """Record a named lap and return the total elapsed time."""
        total = self.elapsed()
        self.laps.append((label, total))
        return total

    def lap_delta(self, label: str) -> float:
        """Record a lap and return the time since the previous lap."""
        previous = self.laps[-1][1] if self.laps else 0.0
        total = self.lap(label)
        return total - previous

    @property
    def running(self) -> bool:
        """Report whether the stopwatch is currently accumulating."""
        return self._running

    def formatted(self) -> str:
        """Return the elapsed time as a human string."""
        return format_duration(self.elapsed())

    def to_dict(self) -> dict[str, object]:
        """Serialize the stopwatch state."""
        return {
            "name": self.name,
            "elapsed_ms": round(self.elapsed_ms(), 3),
            "running": self._running,
            "laps": [{"label": label, "elapsed_ms": round(value * 1000, 3)} for label, value in self.laps],
        }


class RateWindow:
    """Sliding-window counter used for local rate limiting and throughput."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = max(0.001, float(window_seconds))
        self._events: list[float] = []

    def record(self, at: float | None = None) -> None:
        """Record an event at ``at`` (monotonic seconds) or now."""
        self._events.append(time.monotonic() if at is None else at)
        self._prune()

    def _prune(self) -> None:
        """Drop events older than the window."""
        cutoff = time.monotonic() - self.window
        while self._events and self._events[0] < cutoff:
            self._events.pop(0)

    def count(self) -> int:
        """Return how many events fall inside the current window."""
        self._prune()
        return len(self._events)

    def rate_per_second(self) -> float:
        """Return the current event rate per second."""
        self._prune()
        if not self._events:
            return 0.0
        span = max(1e-6, self._events[-1] - self._events[0])
        return round(len(self._events) / span, 4)

    def reset(self) -> None:
        """Clear all recorded events."""
        self._events.clear()


def backoff_schedule(
    attempts: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: float = 0.2,
    seed: float | None = None,
) -> list[float]:
    """Return the delay sequence for exponential backoff with jitter.

    ``seed`` makes the jitter deterministic, which keeps tests reproducible.
    """
    import random

    generator = random.Random(seed)
    schedule: list[float] = []
    for attempt in range(max(0, attempts)):
        delay = min(cap, base * (factor**attempt))
        if jitter > 0:
            delay *= 1.0 + generator.uniform(-jitter, jitter)
        schedule.append(round(max(0.0, delay), 4))
    return schedule


__all__ = [
    "RateWindow",
    "Stopwatch",
    "backoff_schedule",
    "deadline_from_timeout",
    "format_duration",
    "format_relative",
    "format_timestamp",
    "is_expired",
    "monotonic_ms",
    "parse_duration",
    "parse_iso",
    "time_remaining",
    "utc_iso",
]
