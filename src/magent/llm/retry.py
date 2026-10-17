"""Retry orchestration and circuit breaking.

The transport layer knows *how* to back off; this module decides *whether* to
try again at all. It adds a circuit breaker so a dead Cortex instance fails
fast instead of burning three timeouts per request, plus a decorator for
wrapping arbitrary callables.
"""

from __future__ import annotations

import functools
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from ..errors import MAgentError, RateLimitedError, is_retryable

T = TypeVar("T")


class BreakerState(StrEnum):
    """Lifecycle states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def allows_traffic(self) -> bool:
        """Report whether calls may proceed in this state."""
        return self is not BreakerState.OPEN


@dataclass(slots=True)
class BreakerStats:
    """Counters describing circuit breaker behaviour."""

    successes: int = 0
    failures: int = 0
    rejections: int = 0
    trips: int = 0
    resets: int = 0
    half_open_probes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the counters."""
        return {
            "successes": self.successes,
            "failures": self.failures,
            "rejections": self.rejections,
            "trips": self.trips,
            "resets": self.resets,
            "half_open_probes": self.half_open_probes,
        }

    @property
    def failure_rate(self) -> float:
        """Return the observed failure fraction."""
        total = self.successes + self.failures
        return round(self.failures / total, 4) if total else 0.0


class CircuitBreaker:
    """Trip after repeated failures, then probe before trusting the backend.

    The breaker is time-based: after ``reset_timeout`` it moves to half-open and
    admits a limited number of probe calls. A success closes it; a failure
    re-opens it for another full timeout.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        success_threshold: int = 1,
        half_open_max_calls: int = 2,
        window_seconds: float = 60.0,
        failure_rate_threshold: float = 0.8,
        min_samples: int = 5,
        name: str = "breaker",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = max(0.05, reset_timeout)
        self.success_threshold = max(1, success_threshold)
        self.half_open_max_calls = max(1, half_open_max_calls)
        self.window_seconds = max(1.0, window_seconds)
        self.failure_rate_threshold = failure_rate_threshold
        self.min_samples = max(1, min_samples)
        self.name = name
        self._clock = clock
        self._lock = threading.RLock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._outcomes: deque[tuple[float, bool]] = deque()
        self.stats = BreakerStats()

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Return the current state, promoting OPEN to HALF_OPEN when due."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN once the reset timeout elapses."""
        if self._state is BreakerState.OPEN:
            if self._clock() - self._opened_at >= self.reset_timeout:
                self._state = BreakerState.HALF_OPEN
                self._half_open_calls = 0
                self.stats.resets += 1

    def allow_request(self) -> bool:
        """Report whether a call may proceed right now."""
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.CLOSED:
                return True
            if self._state is BreakerState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    self.stats.half_open_probes += 1
                    return True
                self.stats.rejections += 1
                return False
            self.stats.rejections += 1
            return False

    def record_success(self) -> None:
        """Note a successful call, possibly closing the breaker."""
        with self._lock:
            self._outcomes.append((self._clock(), True))
            self._prune()
            self.stats.successes += 1
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            if self._state is BreakerState.HALF_OPEN:
                if self._consecutive_successes >= self.success_threshold:
                    self._state = BreakerState.CLOSED
                    self._consecutive_successes = 0
            elif self._state is BreakerState.OPEN:
                self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        """Note a failed call, possibly tripping the breaker."""
        with self._lock:
            self._outcomes.append((self._clock(), False))
            self._prune()
            self.stats.failures += 1
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if self._state is BreakerState.HALF_OPEN:
                self._trip()
                return
            if self._consecutive_failures >= self.failure_threshold:
                self._trip()
                return
            total = len(self._outcomes)
            if total >= self.min_samples:
                failures = sum(1 for _, ok in self._outcomes if not ok)
                if failures / total >= self.failure_rate_threshold:
                    self._trip()

    def _trip(self) -> None:
        """Open the breaker and start the reset countdown."""
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._consecutive_failures = 0
        self.stats.trips += 1

    def _prune(self) -> None:
        """Drop outcomes older than the rolling window."""
        cutoff = self._clock() - self.window_seconds
        while self._outcomes and self._outcomes[0][0] < cutoff:
            self._outcomes.popleft()

    def reset(self) -> None:
        """Force the breaker closed and clear counters."""
        with self._lock:
            self._state = BreakerState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._outcomes.clear()
            self._half_open_calls = 0

    def retry_after(self) -> float:
        """Return seconds until the breaker will admit probe traffic."""
        with self._lock:
            if self._state is not BreakerState.OPEN:
                return 0.0
            return max(0.0, self.reset_timeout - (self._clock() - self._opened_at))

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable view of breaker state."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "retry_after_seconds": round(self.retry_after(), 3),
                "window_samples": len(self._outcomes),
                **self.stats.to_dict(),
            }

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Use the breaker as a decorator."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.allow_request():
                from ..errors import BackendUnavailableError

                raise BackendUnavailableError(
                    f"Circuit breaker {self.name!r} is open; retry in {self.retry_after():.1f}s"
                )
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self.record_failure()
                raise exc
            self.record_success()
            return result

        return wrapper


@dataclass(slots=True)
class RetryAttempt:
    """A record of one retry decision, kept for observability."""

    attempt: int
    delay: float
    error: str
    error_code: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attempt record."""
        return {
            "attempt": self.attempt,
            "delay": round(self.delay, 4),
            "error": self.error,
            "error_code": self.error_code,
            "ts": self.ts,
        }


@dataclass(slots=True)
class RetryOutcome:
    """The aggregate result of a retried call."""

    value: Any = None
    attempts: int = 0
    total_delay: float = 0.0
    succeeded: bool = False
    error: str | None = None
    history: list[RetryAttempt] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome including attempt history."""
        return {
            "attempts": self.attempts,
            "total_delay": round(self.total_delay, 4),
            "succeeded": self.succeeded,
            "error": self.error,
            "from_cache": self.from_cache,
            "history": [item.to_dict() for item in self.history],
        }


class RetryExecutor:
    """Configurable retry loop with jitter, budgets, and hooks."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.4,
        factor: float = 2.0,
        cap: float = 20.0,
        jitter: float = 0.25,
        strategy: str = "exponential",
        retry_on: tuple[type[BaseException], ...] | None = None,
        never_retry_on: tuple[type[BaseException], ...] = (),
        breaker: CircuitBreaker | None = None,
        on_attempt: Callable[[RetryAttempt], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        seed: int | None = None,
        total_budget: float | None = None,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.base_delay = max(0.0, base_delay)
        self.factor = max(1.0, factor)
        self.cap = max(self.base_delay, cap)
        self.jitter = max(0.0, jitter)
        self.strategy = strategy
        self.retry_on = retry_on
        self.never_retry_on = never_retry_on
        self.breaker = breaker
        self.on_attempt = on_attempt
        self._sleep = sleep
        self._random = random.Random(seed)
        self.total_budget = total_budget
        self.executions = 0
        self.retries = 0
        self.gave_up = 0

    # -- delay computation ---------------------------------------------------

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Return how long to sleep before the next attempt."""
        if retry_after is not None and retry_after > 0:
            return min(self.cap, retry_after)
        index = max(0, attempt - 1)
        if self.strategy == "constant":
            delay = self.base_delay
        elif self.strategy == "linear":
            delay = self.base_delay * (index + 1)
        elif self.strategy == "fibonacci":
            delay = self.base_delay * _fibonacci(index + 2)
        else:
            delay = self.base_delay * (self.factor**index)
        delay = min(self.cap, delay)
        if self.jitter > 0:
            delay *= 1.0 + self._random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)

    def schedule(self, attempts: int | None = None) -> list[float]:
        """Return the full delay sequence for inspection or testing."""
        count = self.max_attempts if attempts is None else attempts
        return [self.delay_for(attempt) for attempt in range(1, max(1, count))]

    # -- decision ------------------------------------------------------------

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Decide whether another attempt is permitted for ``error``."""
        if attempt >= self.max_attempts:
            return False
        if self.never_retry_on and isinstance(error, self.never_retry_on):
            return False
        if isinstance(error, RateLimitedError):
            return True
        if self.retry_on is not None:
            return isinstance(error, self.retry_on)
        return is_retryable(error)

    # -- execution -----------------------------------------------------------

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Invoke ``fn`` with retries, returning its result or raising."""
        outcome = self.execute(fn, *args, **kwargs)
        if not outcome.succeeded:
            raise (
                outcome.value
                if isinstance(outcome.value, BaseException)
                else MAgentError(str(outcome.error or "retry budget exhausted"))
            )
        return outcome.value  # type: ignore[return-value]

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> RetryOutcome:
        """Invoke ``fn`` and return a full :class:`RetryOutcome`."""
        self.executions += 1
        outcome = RetryOutcome()
        started = time.monotonic()
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            if self.breaker is not None and not self.breaker.allow_request():
                from ..errors import BackendUnavailableError

                last_error = BackendUnavailableError(
                    f"Circuit breaker open; next probe in {self.breaker.retry_after():.1f}s"
                )
                outcome.error = str(last_error)
                break
            outcome.attempts = attempt
            try:
                value = fn(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if self.breaker is not None:
                    self.breaker.record_failure()
                retry_after = exc.retry_after if isinstance(exc, RateLimitedError) else None
                code = exc.code if isinstance(exc, MAgentError) else type(exc).__name__
                if not self.should_retry(exc, attempt):
                    outcome.error = str(exc)
                    outcome.value = exc
                    self.gave_up += 1
                    break
                delay = self.delay_for(attempt, retry_after=retry_after)
                if (
                    self.total_budget is not None
                    and time.monotonic() - started + delay > self.total_budget
                ):
                    outcome.error = f"retry budget exhausted: {exc}"
                    outcome.value = exc
                    self.gave_up += 1
                    break
                record = RetryAttempt(
                    attempt=attempt, delay=delay, error=str(exc)[:300], error_code=code
                )
                outcome.history.append(record)
                outcome.total_delay += delay
                self.retries += 1
                if self.on_attempt is not None:
                    self.on_attempt(record)
                self._sleep(delay)
                continue
            if self.breaker is not None:
                self.breaker.record_success()
            outcome.value = value
            outcome.succeeded = True
            outcome.error = None
            return outcome
        if last_error is not None and outcome.value is None:
            outcome.value = last_error
            outcome.error = str(last_error)
        if not outcome.succeeded:
            self.gave_up += 1
        return outcome

    def stats(self) -> dict[str, Any]:
        """Return executor telemetry."""
        return {
            "executions": self.executions,
            "retries": self.retries,
            "gave_up": self.gave_up,
            "max_attempts": self.max_attempts,
            "strategy": self.strategy,
            "breaker": self.breaker.snapshot() if self.breaker else None,
        }


def _fibonacci(index: int) -> int:
    """Return the ``index``-th Fibonacci number, capped for safety."""
    if index <= 0:
        return 0
    first, second = 0, 1
    for _ in range(min(index, 40)):
        first, second = second, first + second
    return min(first, 1000)


def retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.4,
    **kwargs: Any,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :class:`RetryExecutor`."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        executor = RetryExecutor(max_attempts=max_attempts, base_delay=base_delay, **kwargs)

        @functools.wraps(fn)
        def wrapper(*args: Any, **call_kwargs: Any) -> T:
            return executor.call(fn, *args, **call_kwargs)

        wrapper.executor = executor  # type: ignore[attr-defined]
        return wrapper

    return decorator


def with_timeout(fn: Callable[..., T], timeout: float, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a worker thread and abandon it after ``timeout`` seconds.

    Python cannot kill a thread, so this returns control to the caller and lets
    the worker finish in the background; it is used for tool calls that must not
    stall the agent loop.
    """
    import concurrent.futures

    from ..errors import TimeoutError_

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError_(f"Call exceeded {timeout}s") from exc


def any_success(
    calls: Iterable[Callable[[], T]],
    *,
    on_error: Callable[[BaseException], None] | None = None,
) -> T:
    """Try callables in order and return the first successful result."""
    last: BaseException | None = None
    for call in calls:
        try:
            return call()
        except Exception as exc:
            last = exc
            if on_error is not None:
                on_error(exc)
    if last is not None:
        raise last
    raise MAgentError("no candidates supplied to any_success")


__all__ = [
    "BreakerState",
    "BreakerStats",
    "CircuitBreaker",
    "RetryAttempt",
    "RetryExecutor",
    "RetryOutcome",
    "any_success",
    "retry",
    "with_timeout",
]
