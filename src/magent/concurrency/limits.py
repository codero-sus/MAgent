"""Rate limiting, throttling and debouncing for synchronous code.

Three different problems show up around an agent:

*do not exceed a rate*
    :class:`TokenBucket` - requests per second against Cortex, tool calls per
    minute from a policy, API quotas;
*do not repeat too often*
    :class:`Throttle` - at most one call per interval, dropping the rest;
*collapse bursts into one call*
    :class:`Debouncer` - autosave a session after the user stops typing.

:class:`ConcurrencyLimiter` bounds how many threads may be inside a section at
once, which matters when a pool is wider than a resource (a SQLite connection,
one model slot on a GPU).  All of them are thread-safe and non-blocking by
default: ``allow()`` answers immediately, ``acquire()`` waits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..logging_ import get_logger

LOGGER = get_logger("concurrency.limits")
"""Logger for limit decisions."""

DEFAULT_RATE = 10.0
"""Default token refill rate per second."""

DEFAULT_CAPACITY = 10.0
"""Default bucket capacity."""


@dataclass(slots=True)
class LimitStats:
    """Counters shared by the limiters."""

    allowed: int = 0
    denied: int = 0
    waited: float = 0.0
    waits: int = 0
    last_decision_at: float = 0.0
    last_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the counters."""
        return {
            "allowed": self.allowed,
            "denied": self.denied,
            "waited": round(self.waited, 4),
            "waits": self.waits,
            "last_decision_at": self.last_decision_at,
            "last_reason": self.last_reason,
        }

    @property
    def total(self) -> int:
        """Decisions made."""
        return self.allowed + self.denied

    @property
    def denial_rate(self) -> float:
        """Fraction of decisions that were denied."""
        if not self.total:
            return 0.0
        return round(self.denied / self.total, 4)


class TokenBucket:
    """Classic token bucket rate limiter."""

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        capacity: float | None = None,
        *,
        initial: float | None = None,
        name: str = "bucket",
    ) -> None:
        """Create a bucket.

        Args:
            rate: tokens added per second
            capacity: bucket size; defaults to one second of ``rate``
            initial: starting tokens; defaults to a full bucket
            name: identifier used in logs and stats
        """
        self.rate = max(0.0, float(rate))
        self.capacity = float(capacity if capacity is not None else max(self.rate, 1.0))
        self.tokens = float(initial if initial is not None else self.capacity)
        self.name = name
        self.stats = LimitStats()
        self._updated = time.monotonic()
        self._lock = threading.RLock()

    def _refill(self, now: float) -> float:
        """Add tokens accrued since the last call; returns the new balance."""
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        if elapsed and self.rate:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        return self.tokens

    @property
    def available(self) -> float:
        """Tokens available right now."""
        with self._lock:
            return round(self._refill(time.monotonic()), 6)

    def wait_time(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` would be available; 0 when they already are."""
        with self._lock:
            now = time.monotonic()
            balance = self._refill(now)
            if balance >= tokens:
                return 0.0
            if not self.rate:
                return float("inf")
            return (tokens - balance) / self.rate

    def try_acquire(self, tokens: float = 1.0, *, count: bool = True) -> bool:
        """Take tokens when available, without waiting.

        Args:
            tokens: how many to take
            count: record the decision; :meth:`acquire` passes ``False`` while it
                polls, so one logical denial is not counted once per attempt
        """
        with self._lock:
            now = time.monotonic()
            balance = self._refill(now)
            if count:
                self.stats.last_decision_at = time.time()
            if balance >= tokens:
                self.tokens = balance - tokens
                if count:
                    self.stats.allowed += 1
                    self.stats.last_reason = ""
                return True
            if count:
                self.stats.denied += 1
                self.stats.last_reason = f"need {tokens}, have {balance:.3f}"
            return False

    def acquire(self, tokens: float = 1.0, *, timeout: float | None = None) -> bool:
        """Wait until tokens are available.

        Args:
            tokens: how many to take
            timeout: maximum seconds to wait; ``None`` waits as long as needed

        Returns:
            True when the tokens were taken, False on timeout
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        started = time.monotonic()
        while True:
            if self.try_acquire(tokens, count=False):
                waited = time.monotonic() - started
                with self._lock:
                    self.stats.allowed += 1
                    self.stats.last_reason = ""
                    self.stats.last_decision_at = time.time()
                if waited > 0.001:
                    self.stats.waited += waited
                    self.stats.waits += 1
                return True
            delay = self.wait_time(tokens)
            if delay == float("inf"):
                self.stats.denied += 1
                self.stats.last_reason = f"need {tokens}, rate is 0"
                self.stats.last_decision_at = time.time()
                return False
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stats.denied += 1
                    self.stats.last_reason = f"timed out waiting for {tokens} tokens"
                    self.stats.last_decision_at = time.time()
                    return False
                delay = min(delay, remaining)
            time.sleep(min(max(delay, 0.001), 0.25))

    @contextmanager
    def slot(self, tokens: float = 1.0, *, timeout: float | None = None) -> Iterator[bool]:
        """Context manager yielding whether the slot was acquired."""
        yield self.acquire(tokens, timeout=timeout)

    def reset(self, *, full: bool = True) -> None:
        """Reset the bucket, optionally to full."""
        with self._lock:
            self.tokens = self.capacity if full else 0.0
            self._updated = time.monotonic()

    def describe(self) -> dict[str, Any]:
        """Serialise the bucket state."""
        return {
            "name": self.name,
            "rate": self.rate,
            "capacity": self.capacity,
            "available": self.available,
            **self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"TokenBucket(rate={self.rate}/s, capacity={self.capacity}, name={self.name!r})"


class Throttle:
    """Allow at most one call per interval."""

    def __init__(self, interval: float, *, name: str = "throttle", leading: bool = True) -> None:
        """Create a throttle.

        Args:
            interval: minimum seconds between allowed calls
            name: identifier for logs and stats
            leading: allow the first call immediately
        """
        self.interval = max(0.0, float(interval))
        self.name = name
        self.leading = bool(leading)
        self.stats = LimitStats()
        self._last = 0.0 if leading else time.monotonic()
        self._lock = threading.RLock()

    def allow(self) -> bool:
        """True when a call may proceed now."""
        with self._lock:
            now = time.monotonic()
            self.stats.last_decision_at = time.time()
            if now - self._last >= self.interval:
                self._last = now
                self.stats.allowed += 1
                return True
            self.stats.denied += 1
            # Computed inline: ``self.remaining`` would re-enter the same lock.
            self.stats.last_reason = f"next in {max(0.0, self.interval - (now - self._last)):.3f}s"
            return False

    @property
    def remaining(self) -> float:
        """Seconds until the next call is allowed."""
        with self._lock:
            elapsed = time.monotonic() - self._last
            return max(0.0, self.interval - elapsed)

    def wait(self) -> None:
        """Block until a call is allowed, then consume the slot."""
        while not self.allow():
            time.sleep(min(max(self.remaining, 0.001), 0.1))

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a throttled version of ``fn`` that skips suppressed calls."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self.allow():
                return None
            return fn(*args, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "throttled")
        wrapper.__doc__ = (
            f"Throttled to one call per {self.interval}s: {getattr(fn, '__doc__', '')}"
        )
        return wrapper

    def reset(self) -> None:
        """Allow the next call immediately."""
        with self._lock:
            self._last = 0.0

    def describe(self) -> dict[str, Any]:
        """Serialise the throttle state."""
        return {
            "name": self.name,
            "interval": self.interval,
            "remaining": round(self.remaining, 4),
            **self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"Throttle(interval={self.interval}, name={self.name!r})"


class Debouncer:
    """Collapse a burst of calls into one, after the burst quiets down."""

    def __init__(self, delay: float, *, name: str = "debounce", max_wait: float = 0.0) -> None:
        """Create a debouncer.

        Args:
            delay: quiet period in seconds before the call fires
            name: identifier for logs and stats
            max_wait: fire anyway after this many seconds of continuous activity,
                0 to disable
        """
        self.delay = max(0.0, float(delay))
        self.max_wait = max(0.0, float(max_wait))
        self.name = name
        self.stats = LimitStats()
        self._timer: threading.Timer | None = None
        self._first_call = 0.0
        self._lock = threading.RLock()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Schedule ``fn`` to run after the quiet period; returns whether it was scheduled.

        A previous pending call is cancelled, which is the point of debouncing.
        """
        with self._lock:
            now = time.monotonic()
            if self._timer is not None:
                self._timer.cancel()
                self.stats.denied += 1
            if not self._first_call:
                self._first_call = now
            force = bool(self.max_wait and now - self._first_call >= self.max_wait)
            wait = 0.0 if force else self.delay
            timer = threading.Timer(wait, self._fire, args=(fn, args, kwargs))
            timer.daemon = True
            timer.name = f"magent-debounce-{self.name}"
            self._timer = timer
            timer.start()
            self.stats.allowed += 1
            self.stats.last_decision_at = time.time()
            self.stats.last_reason = "forced by max_wait" if force else f"fires in {wait:.3f}s"
            if force:
                self._first_call = 0.0
            return True

    def _fire(self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Run the debounced call, swallowing its errors into the log."""
        with self._lock:
            self._timer = None
            self._first_call = 0.0
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            LOGGER.warning("debounced call %s failed: %s", self.name, exc)

    def flush(self) -> bool:
        """Cancel a pending call without running it."""
        with self._lock:
            if self._timer is None:
                return False
            self._timer.cancel()
            self._timer = None
            self._first_call = 0.0
            return True

    def wait(self, timeout: float | None = None) -> bool:
        """Block until no call is pending; returns whether one fired."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                pending = self._timer is not None
            if not pending:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    @property
    def pending(self) -> bool:
        """True when a call is scheduled."""
        with self._lock:
            return self._timer is not None

    def describe(self) -> dict[str, Any]:
        """Serialise the debouncer state."""
        return {
            "name": self.name,
            "delay": self.delay,
            "max_wait": self.max_wait,
            "pending": self.pending,
            **self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"Debouncer(delay={self.delay}, name={self.name!r})"


class ConcurrencyLimiter:
    """Bound how many threads may be inside a section at once."""

    def __init__(
        self, limit: int = 1, *, name: str = "limiter", timeout: float | None = None
    ) -> None:
        """Create the limiter.

        Args:
            limit: maximum concurrent holders
            name: identifier for logs and stats
            timeout: default seconds to wait for a slot; ``None`` waits forever
        """
        self.limit = max(1, int(limit))
        self.name = name
        self.timeout = timeout
        self.stats = LimitStats()
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self._active = 0
        self._peak = 0
        self._lock = threading.RLock()

    @property
    def active(self) -> int:
        """Holders right now."""
        with self._lock:
            return self._active

    @property
    def peak(self) -> int:
        """Highest number of simultaneous holders seen."""
        return self._peak

    def acquire(self, *, timeout: float | None = None) -> bool:
        """Take a slot, waiting up to ``timeout`` seconds."""
        limit_timeout = self.timeout if timeout is None else timeout
        started = time.monotonic()
        acquired = self._semaphore.acquire(timeout=limit_timeout)  # type: ignore[arg-type]
        waited = time.monotonic() - started
        self.stats.last_decision_at = time.time()
        if acquired:
            with self._lock:
                self._active += 1
                self._peak = max(self._peak, self._active)
            self.stats.allowed += 1
            if waited > 0.001:
                self.stats.waited += waited
                self.stats.waits += 1
        else:
            self.stats.denied += 1
            self.stats.last_reason = f"all {self.limit} slots busy after {waited:.3f}s"
        return acquired

    def release(self) -> None:
        """Give a slot back."""
        with self._lock:
            if self._active:
                self._active -= 1
        try:
            self._semaphore.release()
        except ValueError:
            LOGGER.debug("concurrency limiter %s released without a hold", self.name)

    @contextmanager
    def slot(self, *, timeout: float | None = None) -> Iterator[bool]:
        """Context manager that holds a slot for the duration of the block."""
        acquired = self.acquire(timeout=timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    def map(
        self, fn: Callable[[Any], Any], items: Iterable[Any], *, max_workers: int | None = None
    ) -> list[Any]:
        """Map over items in parallel while respecting the slot limit."""
        from .pool import Task, WorkerPool

        materialised = list(items)
        if not materialised:
            return []
        workers = min(max_workers or self.limit, len(materialised))

        def guarded(item: Any) -> Any:
            with self.slot() as acquired:
                if not acquired:
                    raise TimeoutError(f"no slot on limiter {self.name!r}")
                return fn(item)

        with WorkerPool(max(1, workers), name=self.name) as pool:
            outcomes = pool.run_many(
                [
                    Task(name=f"{self.name}[{index}]", fn=guarded, args=(item,))
                    for index, item in enumerate(materialised)
                ]
            )
        return [outcome.value for outcome in outcomes]

    def describe(self) -> dict[str, Any]:
        """Serialise the limiter state."""
        return {
            "name": self.name,
            "limit": self.limit,
            "active": self.active,
            "peak": self.peak,
            **self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"ConcurrencyLimiter(limit={self.limit}, active={self.active})"


@dataclass(slots=True)
class BatchResult:
    """Outcome of a batched, rate-limited run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    seconds: float = 0.0
    values: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the batch result."""
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 4),
            "errors": list(self.errors[:20]),
        }


def run_rate_limited(
    items: Sequence[Any],
    fn: Callable[[Any], Any],
    *,
    rate: float = 10.0,
    capacity: float | None = None,
    max_workers: int = 1,
    timeout: float = 30.0,
    token: Any = None,
) -> BatchResult:
    """Process items through a token bucket, optionally in parallel.

    Args:
        items: inputs to process
        fn: called with one item
        rate: allowed calls per second
        capacity: bucket size
        max_workers: thread ceiling; 1 runs sequentially
        timeout: per-item timeout
        token: optional :class:`~magent.concurrency.pool.CancellationToken`
    """
    from .pool import Task, WorkerPool

    result = BatchResult(total=len(items))
    started = time.perf_counter()
    bucket = TokenBucket(rate, capacity, name="batch")
    if not items:
        result.seconds = time.perf_counter() - started
        return result

    def call(item: Any) -> Any:
        if token is not None and getattr(token, "cancelled", False):
            raise KeyboardInterrupt(getattr(token, "reason", "cancelled"))
        if not bucket.acquire(timeout=max(timeout, 1.0)):
            raise TimeoutError(f"rate limit {rate}/s not satisfied within {timeout}s")
        return fn(item)

    if max_workers <= 1:
        for index, item in enumerate(items):
            try:
                result.values.append(call(item))
                result.succeeded += 1
            except KeyboardInterrupt as exc:
                result.skipped += len(items) - index
                result.errors.append(str(exc))
                break
            except Exception as exc:
                result.failed += 1
                result.values.append(None)
                result.errors.append(f"{type(exc).__name__}: {exc}")
    else:
        with WorkerPool(max_workers, name="rate-limited", token=token) as pool:
            outcomes = pool.run_many(
                [
                    Task(name=f"item[{index}]", fn=call, args=(item,), timeout=timeout)
                    for index, item in enumerate(items)
                ]
            )
            for outcome in outcomes:
                result.values.append(outcome.value if outcome.ok else None)
                if outcome.ok:
                    result.succeeded += 1
                elif outcome.cancelled:
                    result.skipped += 1
                else:
                    result.failed += 1
                    result.errors.append(f"{outcome.name}: {outcome.error}")
    result.seconds = time.perf_counter() - started
    return result


def interleave(batches: Sequence[Sequence[Any]]) -> list[Any]:
    """Round-robin several sequences into one, for fair scheduling."""
    out: list[Any] = []
    iterators = [iter(batch) for batch in batches]
    while iterators:
        for position, iterator in enumerate(list(iterators)):
            try:
                out.append(next(iterator))
            except StopIteration:
                iterators.pop(position)
    return out


def first_success(outcomes: Iterable[Any]) -> Any:
    """Return the first successful outcome, or ``None``."""
    for outcome in outcomes:
        if getattr(outcome, "ok", False):
            return outcome
    return None


__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_RATE",
    "BatchResult",
    "ConcurrencyLimiter",
    "Debouncer",
    "LimitStats",
    "Throttle",
    "TokenBucket",
    "first_success",
    "interleave",
    "run_rate_limited",
]
