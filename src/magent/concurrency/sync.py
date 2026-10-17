"""Synchronization primitives an agent run actually needs.

Four problems recur in every long-running agent, and each has a subtle failure
mode worth solving once:

*work must be cancellable*
    :class:`CancellationToken` is checked at loop boundaries and raises a typed
    :class:`~magent.errors.CancelledError`, so a Ctrl-C stops a run between
    steps instead of leaving a half-written file.

*a shared resource must not be hammered*
    :class:`ConcurrencyLimiter` bounds simultaneous use with a timeout, rather
    than blocking forever when every slot is taken.

*a burst of events must collapse into one action*
    :class:`Debouncer` runs the trailing call only, which is what a file watcher
    wants when an editor writes a file in three chunks.

*initialization must happen exactly once*
    :func:`once` makes a lazy setup thread-safe without a lock at every call
    site.

These are deliberately small. The heavier patterns - retries and circuit
breaking - already live in :mod:`magent.llm.retry`, and per-caller rate limits
in :mod:`magent.guardrails.policy`; this module does not duplicate them.
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from ..errors import CancelledError, PoolExhaustedError
from ..logging_ import get_logger

LOGGER = get_logger("concurrency.sync")
"""Logger for synchronization faults."""

R = TypeVar("R")
"""Return type used by the decorators in this module."""

DEFAULT_ACQUIRE_TIMEOUT = 30.0
"""Seconds :class:`ConcurrencyLimiter` waits for a slot before giving up."""


class CancellationToken:
    """A cooperative cancellation flag shared across threads.

    The token never kills a thread - Python cannot do that safely - so callers
    check it at the points where stopping is cheap and safe: between loop
    iterations, before a network call, before writing a file.
    """

    def __init__(self, *, reason: str = "") -> None:
        """Create an uncancelled token.

        Args:
            reason: pre-set cancellation reason, which also cancels immediately
        """
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[str], None]] = []
        self.cancelled_at = 0.0
        if reason:
            self.cancel(reason)

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why the token was cancelled."""
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation, returning whether this call changed anything.

        Callbacks registered with :meth:`on_cancel` run once, on the calling
        thread, and a callback that raises is logged rather than propagated -
        cancellation must succeed even if cleanup is imperfect.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = str(reason or "cancelled")
            self.cancelled_at = time.time()
            self._event.set()
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(self._reason)
            except Exception as exc:  # noqa: BLE001 - cleanup must not block cancellation
                LOGGER.debug("cancellation callback raised: %s", exc)
        LOGGER.debug("cancelled: %s", self._reason)
        return True

    def on_cancel(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        """Register a callback invoked once on cancellation.

        A callback registered after cancellation has already happened runs
        immediately, so ordering cannot cause a missed cleanup.
        """
        with self._lock:
            already = self._event.is_set()
            reason = self._reason
            if not already:
                self._callbacks.append(callback)
        if already:
            try:
                callback(reason)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("late cancellation callback raised: %s", exc)
        return callback

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled, returning whether it happened."""
        return self._event.wait(timeout)

    def check(self, *, what: str = "operation") -> None:
        """Raise if cancellation was requested.

        Args:
            what: description included in the error, for a useful message

        Raises:
            CancelledError: the token is cancelled
        """
        if self._event.is_set():
            raise CancelledError(f"{what} cancelled: {self.reason}",
                                 details={"reason": self.reason})

    def reset(self) -> None:
        """Clear the cancellation so the token can be reused."""
        with self._lock:
            self._event.clear()
            self._reason = ""
            self.cancelled_at = 0.0

    def link(self, other: CancellationToken) -> None:
        """Cancel this token whenever ``other`` is cancelled."""
        other.on_cancel(lambda reason: self.cancel(f"linked: {reason}"))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable view."""
        with self._lock:
            return {"cancelled": self._event.is_set(), "reason": self._reason,
                    "cancelled_at": self.cancelled_at, "callbacks": len(self._callbacks)}

    def __bool__(self) -> bool:
        """Truthiness is whether work may continue, for readable call sites."""
        return not self._event.is_set()

    def __repr__(self) -> str:
        """Debug representation."""
        state = f"cancelled({self.reason!r})" if self.cancelled else "active"
        return f"CancellationToken({state})"


class ConcurrencyLimiter:
    """Bounds how many threads may hold a resource at once."""

    def __init__(self, permits: int = 4, *, name: str = "resource",
                 timeout: float = DEFAULT_ACQUIRE_TIMEOUT) -> None:
        """Configure the limiter.

        Args:
            permits: maximum simultaneous holders
            name: label used in errors and statistics
            timeout: seconds to wait for a permit before failing
        """
        self.permits = max(1, int(permits))
        self.name = name
        self.timeout = float(timeout)
        self._semaphore = threading.BoundedSemaphore(self.permits)
        self._lock = threading.Lock()
        self._active = 0
        self.acquired = 0
        self.released = 0
        self.refused = 0
        self.waited_ms = 0.0
        self.max_active = 0

    @property
    def active(self) -> int:
        """How many permits are currently held."""
        with self._lock:
            return self._active

    @property
    def available(self) -> int:
        """How many permits are free."""
        with self._lock:
            return max(0, self.permits - self._active)

    def acquire(self, *, timeout: float | None = None) -> bool:
        """Take a permit, returning whether it was obtained.

        Args:
            timeout: override the configured wait; ``None`` waits forever
        """
        limit = self.timeout if timeout is None else float(timeout)
        # A negative timeout means "wait forever", zero means "do not block".
        wait: float | None = None if limit < 0 else limit
        started = time.perf_counter()
        got = self._semaphore.acquire(timeout=wait)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not got:
            with self._lock:
                self.refused += 1
            return False
        with self._lock:
            self._active += 1
            self.acquired += 1
            self.waited_ms += elapsed
            self.max_active = max(self.max_active, self._active)
        return True

    def release(self) -> None:
        """Give a permit back."""
        with self._lock:
            self._active = max(0, self._active - 1)
            self.released += 1
        self._semaphore.release()

    def require(self, *, timeout: float | None = None, what: str = "") -> None:
        """Acquire a permit or raise.

        Raises:
            PoolExhaustedError: no permit became free within the timeout
        """
        if not self.acquire(timeout=timeout):
            raise PoolExhaustedError(
                f"no {self.name} permit available within "
                f"{self.timeout if timeout is None else timeout}s",
                details={"permits": self.permits, "active": self.active,
                         "what": what or self.name},
            )

    @property
    def held(self) -> _Permit:
        """Return a context manager that holds one permit."""
        return _Permit(self)

    def guard(self, fn: Callable[..., R]) -> Callable[..., R]:
        """Wrap a callable so it runs while holding one permit."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with self.held:
                return fn(*args, **kwargs)

        return wrapper

    def stats(self) -> dict[str, Any]:
        """Return limiter counters."""
        with self._lock:
            active = self._active
        return {
            "name": self.name,
            "permits": self.permits,
            "active": active,
            "available": max(0, self.permits - active),
            "acquired": self.acquired,
            "released": self.released,
            "refused": self.refused,
            "max_active": self.max_active,
            "mean_wait_ms": round(self.waited_ms / self.acquired, 3) if self.acquired else 0.0,
            "timeout": self.timeout,
        }

    def __enter__(self) -> ConcurrencyLimiter:
        """Acquire a permit for a ``with`` block."""
        self.require()
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Release the permit."""
        self.release()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"ConcurrencyLimiter({self.name!r}, {self.active}/{self.permits})"


class _Permit:
    """Context manager holding one limiter permit."""

    def __init__(self, limiter: ConcurrencyLimiter) -> None:
        """Capture the limiter to draw from."""
        self.limiter = limiter
        self.acquired = False

    def __enter__(self) -> ConcurrencyLimiter:
        """Take the permit, raising when none is free."""
        self.limiter.require()
        self.acquired = True
        return self.limiter

    def __exit__(self, *_exc: Any) -> None:
        """Release the permit if it was taken."""
        if self.acquired:
            self.limiter.release()
            self.acquired = False


class Throttle:
    """Enforces a minimum interval between successive calls."""

    def __init__(self, interval: float, *, name: str = "throttle",
                 leading: bool = True) -> None:
        """Configure the throttle.

        Args:
            interval: minimum seconds between calls, 0 to disable
            name: label for logs and statistics
            leading: allow the first call through immediately
        """
        self.interval = max(0.0, float(interval))
        self.name = name
        self.leading = bool(leading)
        self._lock = threading.Lock()
        self._last = 0.0
        self.allowed = 0
        self.suppressed = 0

    def allow(self, *, now: float | None = None) -> bool:
        """Return whether a call may proceed right now.

        Unlike :meth:`wait`, this never blocks, which is what a UI refresh or a
        progress redraw wants: skip the update rather than stall the loop.
        """
        if self.interval <= 0:
            self.allowed += 1
            return True
        moment = time.monotonic() if now is None else now
        with self._lock:
            if self.leading and not self._last:
                self._last = moment
                self.allowed += 1
                return True
            if moment - self._last >= self.interval:
                self._last = moment
                self.allowed += 1
                return True
            self.suppressed += 1
            return False

    def wait(self) -> float:
        """Block until a call is allowed, returning the seconds waited."""
        if self.interval <= 0:
            return 0.0
        started = time.monotonic()
        while not self.allow():
            time.sleep(min(0.01, self.interval / 4))
        return time.monotonic() - started

    def retry_after(self) -> float:
        """Seconds until the next call would be allowed."""
        with self._lock:
            if not self._last:
                return 0.0
            return max(0.0, self.interval - (time.monotonic() - self._last))

    def reset(self) -> None:
        """Forget the last call time."""
        with self._lock:
            self._last = 0.0

    def stats(self) -> dict[str, Any]:
        """Return throttle counters."""
        return {"name": self.name, "interval": self.interval, "allowed": self.allowed,
                "suppressed": self.suppressed, "retry_after": round(self.retry_after(), 4)}

    def __call__(self, fn: Callable[..., R]) -> Callable[..., R]:
        """Wrap a callable so it runs at most once per interval."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> R | None:
            if not self.allow():
                return None
            return fn(*args, **kwargs)

        return wrapper


class Debouncer:
    """Collapses a burst of calls into one trailing call."""

    def __init__(self, delay: float, *, name: str = "debounce",
                 max_wait: float = 0.0) -> None:
        """Configure the debouncer.

        Args:
            delay: quiet period required before the call runs
            name: label for logs
            max_wait: run anyway once this much time has passed since the first
                call in a burst, 0 to always wait for quiet
        """
        self.delay = max(0.0, float(delay))
        self.name = name
        self.max_wait = max(0.0, float(max_wait))
        self._timer: threading.Timer | None = None
        self._pending_call: tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None = None
        self._lock = threading.RLock()
        self._first_call = 0.0
        self.fired = 0
        self.collapsed = 0

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Schedule ``fn`` to run after the quiet period.

        Args:
            fn: callable to run
            args: positional arguments for the trailing call
            kwargs: keyword arguments for the trailing call

        Returns:
            Whether a new timer was scheduled (false means an existing pending
            call was postponed).
        """
        with self._lock:
            now = time.monotonic()
            if self._timer is not None:
                self._timer.cancel()
                self.collapsed += 1
                fresh = False
            else:
                self._first_call = now
                fresh = True
            forced = bool(self.max_wait and now - self._first_call >= self.max_wait)
            wait = 0.0 if forced else self.delay

            self._pending_call = (fn, args, kwargs)

            def run() -> None:
                with self._lock:
                    self._timer = None
                    self._pending_call = None
                    self.fired += 1
                try:
                    fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - a debounce must not kill the caller
                    LOGGER.warning("debounced %s raised: %s", self.name, exc)

            self._timer = threading.Timer(wait, run)
            self._timer.daemon = True
            self._timer.start()
            return fresh

    def pending(self) -> bool:
        """Whether a call is scheduled but has not run."""
        with self._lock:
            return self._timer is not None

    def cancel(self) -> bool:
        """Drop a pending call, returning whether one existed."""
        with self._lock:
            timer = self._timer
            self._timer = None
            self._pending_call = None
        if timer is not None:
            timer.cancel()
            return True
        return False

    def flush(self) -> bool:
        """Run a pending call immediately, returning whether one ran.

        A cancelled :class:`threading.Timer` cannot be made to fire, so the
        callable is replayed here from the arguments captured when it was
        scheduled.
        """
        with self._lock:
            timer = self._timer
            pending = self._pending_call
            self._timer = None
            self._pending_call = None
        if timer is None or pending is None:
            return False
        timer.cancel()
        fn, args, kwargs = pending
        self.fired += 1
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - flushing must not raise into the caller
            LOGGER.warning("flushed %s raised: %s", self.name, exc)
            return True
        return True

    def stats(self) -> dict[str, Any]:
        """Return debouncer counters."""
        return {"name": self.name, "delay": self.delay, "max_wait": self.max_wait,
                "fired": self.fired, "collapsed": self.collapsed,
                "pending": self.pending()}

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callable so bursts collapse into one trailing call."""
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            self.call(fn, *args, **kwargs)

        return wrapper


def once(fn: Callable[..., R]) -> Callable[..., R]:
    """Run a callable at most once, caching its result thread-safely.

    The lock is held only while the callable runs, so concurrent callers block
    until the first finishes and then receive the same value - which is what a
    lazy singleton needs, and what a bare ``if instance is None`` check does not
    provide.

    Args:
        fn: zero-argument callable to memoize
    """
    lock = threading.Lock()
    state: dict[str, Any] = {"done": False, "value": None, "error": None}

    @functools.wraps(fn)
    def wrapper() -> R:
        if state["done"]:
            if state["error"] is not None:
                raise state["error"]
            return state["value"]
        with lock:
            if state["done"]:
                if state["error"] is not None:
                    raise state["error"]
                return state["value"]
            try:
                value = fn()
            except BaseException as exc:
                state["error"] = exc
                state["done"] = True
                raise
            state["value"] = value
            state["done"] = True
            return value

    wrapper.reset = lambda: state.update({"done": False, "value": None, "error": None})  # type: ignore[attr-defined]
    return wrapper


@dataclass(slots=True)
class Barrier:
    """Waits for a fixed number of participants, with a timeout."""

    parties: int
    timeout: float = 30.0
    _barrier: threading.Barrier = field(init=False, repr=False)
    broken: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Create the underlying barrier."""
        self.parties = max(1, int(self.parties))
        self._barrier = threading.Barrier(self.parties)

    def wait(self, *, timeout: float | None = None) -> int:
        """Block until every participant arrives, returning this one's index.

        Raises:
            ConcurrencyError: the wait timed out or the barrier broke
        """
        from ..errors import ConcurrencyError

        try:
            return self._barrier.wait(timeout if timeout is not None else self.timeout)
        except threading.BrokenBarrierError as exc:
            self.broken = True
            raise ConcurrencyError("barrier broken or timed out",
                                   details={"parties": self.parties}) from exc

    def reset(self) -> None:
        """Reset a broken or completed barrier for reuse."""
        self._barrier.reset()
        self.broken = False

    @property
    def waiting(self) -> int:
        """How many participants are currently blocked."""
        return self._barrier.n_waiting

    def abort(self) -> None:
        """Break the barrier, releasing every waiter with an error."""
        self._barrier.abort()
        self.broken = True

    def stats(self) -> dict[str, Any]:
        """Return barrier state."""
        return {"parties": self.parties, "waiting": self.waiting, "broken": self.broken,
                "timeout": self.timeout}


def retry_until(predicate: Callable[[], bool], *, timeout: float = 5.0,
                interval: float = 0.05, cancel: CancellationToken | None = None) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses.

    Args:
        predicate: zero-argument condition to test
        timeout: maximum seconds to wait
        interval: seconds between tests
        cancel: token checked between polls

    Returns:
        Whether the predicate became true.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    step = max(0.001, float(interval))
    while True:
        if cancel is not None:
            cancel.check(what="retry_until")
        try:
            if predicate():
                return True
        except Exception as exc:  # noqa: BLE001 - keep polling through transient faults
            LOGGER.debug("retry_until predicate raised: %s", exc)
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(step, max(0.0, deadline - time.monotonic())))


__all__ = [
    "DEFAULT_ACQUIRE_TIMEOUT",
    "Barrier",
    "CancellationToken",
    "ConcurrencyLimiter",
    "Debouncer",
    "Throttle",
    "once",
    "retry_until",
]
