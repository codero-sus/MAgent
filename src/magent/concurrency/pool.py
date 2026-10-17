"""Bounded parallel execution with cancellation and per-task outcomes.

MAgent is sync-first, so parallelism means threads: parallel tool calls, vote
mode in a team, batch ingestion for RAG, concurrent evaluation cases.  Doing
that with a bare ``ThreadPoolExecutor`` scatters ``try/except`` and timeout
handling across every call site, so this module wraps it:

:class:`WorkerPool`
    named submissions, per-task timeouts, retries with backoff, cancellation;
:func:`run_parallel`
    one-shot helper returning a :class:`TaskOutcome` per task - never raising;
:class:`CancellationToken`
    cooperative cancellation shared with the agent's own ``cancel()``.

A failure in one task is reported in its outcome rather than aborting the batch,
which is what lets an agent run four tools at once and still explain which one
broke.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from ..logging_ import get_logger
from ..utils.timeutil import backoff_schedule

LOGGER = get_logger("concurrency.pool")
"""Logger for pool-level failures."""

DEFAULT_WORKERS = 4
"""Default thread ceiling."""

DEFAULT_TIMEOUT = 60.0
"""Default per-task timeout in seconds."""

MAX_WORKERS = 64
"""Hard ceiling on threads, so a config typo cannot exhaust the machine."""


class CancellationToken:
    """Cooperative cancellation shared across threads."""

    def __init__(self, *, reason: str = "") -> None:
        """Create an uncancelled token.

        Args:
            reason: initial reason text, used when the token is pre-cancelled
        """
        self._event = threading.Event()
        self._reason = reason
        self._lock = threading.Lock()
        self.cancelled_at = 0.0
        if reason:
            self._event.set()
            self.cancelled_at = time.time()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why the token was cancelled."""
        return self._reason

    def cancel(self, reason: str = "cancelled") -> None:
        """Signal cancellation to every holder of this token."""
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self.cancelled_at = time.time()
            self._event.set()

    def reset(self) -> None:
        """Clear the cancellation so the token can be reused."""
        with self._lock:
            self._event.clear()
            self._reason = ""
            self.cancelled_at = 0.0

    def raise_if_cancelled(self) -> None:
        """Raise :class:`~magent.errors.UserAbortError` when cancelled."""
        if self.cancelled:
            from ..errors import UserAbortError

            raise UserAbortError(self._reason or "cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled; returns whether cancellation happened."""
        return self._event.wait(timeout)

    def __bool__(self) -> bool:
        """True while the token is still usable (not cancelled)."""
        return not self.cancelled

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"CancellationToken(cancelled={self.cancelled}, reason={self._reason!r})"


@dataclass(slots=True)
class Task:
    """One unit of parallel work."""

    name: str
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    retries: int = 0
    backoff: float = 0.05
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise argument containers and clamp the timeout."""
        self.args = tuple(self.args or ())
        self.kwargs = dict(self.kwargs or {})
        self.timeout = max(0.1, float(self.timeout))
        self.retries = max(0, int(self.retries))
        if not self.name:
            self.name = getattr(self.fn, "__name__", "task")


@dataclass(slots=True)
class TaskOutcome:
    """What happened to one task."""

    name: str
    ok: bool = False
    value: Any = None
    error: str = ""
    error_type: str = ""
    seconds: float = 0.0
    attempts: int = 1
    timed_out: bool = False
    cancelled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration in milliseconds."""
        return self.seconds * 1000.0

    def raise_for_error(self) -> Any:
        """Return the value, or raise the captured error."""
        if self.ok:
            return self.value
        from ..errors import ToolError

        raise ToolError(
            self.error or f"task {self.name!r} failed",
            details={
                "task": self.name,
                "type": self.error_type,
                "timed_out": self.timed_out,
                "cancelled": self.cancelled,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the outcome, stringifying the value when it is not JSON-safe."""
        value = self.value
        if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
            value = repr(value)
        return {
            "name": self.name,
            "ok": self.ok,
            "value": value,
            "error": self.error,
            "error_type": self.error_type,
            "seconds": round(self.seconds, 4),
            "attempts": self.attempts,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        state = "ok" if self.ok else f"failed: {self.error[:60]}"
        return f"TaskOutcome({self.name!r}, {state}, {self.seconds:.3f}s)"


class WorkerPool:
    """A named, bounded thread pool with per-task outcomes."""

    def __init__(
        self,
        max_workers: int = DEFAULT_WORKERS,
        *,
        name: str = "magent-pool",
        token: CancellationToken | None = None,
    ) -> None:
        """Create the pool lazily.

        Args:
            max_workers: thread ceiling, clamped to :data:`MAX_WORKERS`
            name: thread-name prefix, visible in stack dumps
            token: cancellation token checked before starting each task
        """
        self.max_workers = max(1, min(MAX_WORKERS, int(max_workers)))
        self.name = name
        self.token = token if token is not None else CancellationToken()
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.timeouts = 0
        self.retries = 0
        self._seconds = 0.0

    @property
    def executor(self) -> ThreadPoolExecutor:
        """The underlying executor, created on first use."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix=self.name
                )
            return self._executor

    def submit(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit work and return its future."""
        self.submitted += 1
        return self.executor.submit(_named(name, fn), *args, **kwargs)

    def run(self, task: Task) -> TaskOutcome:
        """Run one task synchronously with retries and a timeout."""
        started = time.perf_counter()
        attempts = 0
        last_error = ""
        last_type = ""
        timed_out = False
        schedule = backoff_schedule(
            task.retries, base=task.backoff, factor=2.0, cap=max(task.backoff * 8, 1.0)
        )
        for attempt in range(task.retries + 1):
            if self.token.cancelled:
                return TaskOutcome(
                    name=task.name,
                    ok=False,
                    cancelled=True,
                    attempts=attempts + 1,
                    seconds=time.perf_counter() - started,
                    error=self.token.reason or "cancelled",
                    error_type="CancelledError",
                    metadata=dict(task.metadata),
                )
            attempts += 1
            future = self.submit(task.name, task.fn, *task.args, **task.kwargs)
            try:
                value = future.result(timeout=task.timeout)
            except FutureTimeout:
                future.cancel()
                self.timeouts += 1
                last_error = f"timed out after {task.timeout}s"
                last_type = "TimeoutError"
                timed_out = True
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                last_type = type(exc).__name__
                timed_out = False
                if not isinstance(exc, task.retry_on):
                    break
            else:
                seconds = time.perf_counter() - started
                self.completed += 1
                self.record_seconds(seconds)
                return TaskOutcome(
                    name=task.name,
                    ok=True,
                    value=value,
                    seconds=seconds,
                    attempts=attempts,
                    metadata=dict(task.metadata),
                )
            if attempt < task.retries:
                self.retries += 1
                delay = schedule[attempt] if attempt < len(schedule) else task.backoff
                if delay:
                    self.token.wait(delay)
                continue
            self.failed += 1
            return TaskOutcome(
                name=task.name,
                ok=False,
                error=last_error,
                error_type=last_type,
                timed_out=timed_out,
                attempts=attempts,
                seconds=time.perf_counter() - started,
                metadata=dict(task.metadata),
            )
        self.failed += 1
        return TaskOutcome(
            name=task.name,
            ok=False,
            error=last_error,
            error_type=last_type,
            attempts=attempts,
            seconds=time.perf_counter() - started,
            metadata=dict(task.metadata),
        )

    def run_many(self, tasks: Sequence[Task], *, timeout: float | None = None) -> list[TaskOutcome]:
        """Run tasks in parallel, returning one outcome per task in input order."""
        if not tasks:
            return []
        started = time.perf_counter()
        futures: list[tuple[Task, Future]] = []
        for task in tasks:
            if self.token.cancelled:
                break
            futures.append((task, self.submit(task.name, _with_retries, task, self)))
        outcomes: dict[int, TaskOutcome] = {}
        budget = timeout if timeout is not None else max(task.timeout for task in tasks) + 1.0
        for task, future in futures:
            remaining = max(0.05, budget - (time.perf_counter() - started))
            try:
                outcome = future.result(timeout=remaining)
            except FutureTimeout:
                future.cancel()
                self.timeouts += 1
                outcome = TaskOutcome(
                    name=task.name,
                    ok=False,
                    timed_out=True,
                    error=f"batch timed out after {budget:.1f}s",
                    error_type="TimeoutError",
                    metadata=dict(task.metadata),
                )
            except Exception as exc:
                self.failed += 1
                outcome = TaskOutcome(
                    name=task.name,
                    ok=False,
                    error=str(exc) or type(exc).__name__,
                    error_type=type(exc).__name__,
                    metadata=dict(task.metadata),
                )
            outcomes[id(task)] = outcome
        ordered: list[TaskOutcome] = []
        for index, (task, _) in enumerate(futures):
            ordered.append(
                outcomes.get(id(task), TaskOutcome(name=task.name, ok=False, error="not run"))
            )
            ordered[-1].metadata["position"] = index
        for task in tasks[len(futures) :]:
            ordered.append(
                TaskOutcome(
                    name=task.name,
                    ok=False,
                    cancelled=True,
                    error=self.token.reason or "cancelled before start",
                )
            )
        return ordered

    def map(
        self,
        fn: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        timeout: float = DEFAULT_TIMEOUT,
        name: str = "map",
    ) -> list[TaskOutcome]:
        """Apply ``fn`` to every item in parallel."""
        tasks = [
            Task(name=f"{name}[{index}]", fn=fn, args=(item,), timeout=timeout)
            for index, item in enumerate(items)
        ]
        return self.run_many(tasks)

    def gather(self, *fns: Callable[[], Any], timeout: float = DEFAULT_TIMEOUT) -> list[Any]:
        """Run zero-argument callables in parallel and return their values.

        Failures become ``None`` in the corresponding position; use
        :meth:`run_many` when the error matters.
        """
        tasks = [
            Task(name=getattr(fn, "__name__", f"task{i}"), fn=fn, timeout=timeout)
            for i, fn in enumerate(fns)
        ]
        return [outcome.value if outcome.ok else None for outcome in self.run_many(tasks)]

    def record_seconds(self, seconds: float) -> None:
        """Add successful task wall time to the pool total."""
        self._seconds += max(0.0, seconds)

    def stats(self) -> dict[str, Any]:
        """Pool counters."""
        return {
            "max_workers": self.max_workers,
            "name": self.name,
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "retries": self.retries,
            "seconds": round(self._seconds, 4),
            "cancelled": self.token.cancelled,
            "running": self._executor is not None,
        }

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shut the pool down."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def cancel(self, reason: str = "cancelled") -> None:
        """Cancel the token and drop queued work."""
        self.token.cancel(reason)
        self.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> WorkerPool:
        """Use the pool as a context manager."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Shut the pool down on exit."""
        self.shutdown()

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"WorkerPool(max_workers={self.max_workers}, name={self.name!r})"


def _named(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a callable so thread dumps show the task name."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        threading.current_thread().name = f"{name}"
        return fn(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _with_retries(task: Task, pool: WorkerPool) -> TaskOutcome:
    """Run a task inside a worker thread, including its retries."""
    schedule = backoff_schedule(
        task.retries, base=task.backoff, factor=2.0, cap=max(task.backoff * 8, 1.0)
    )
    last_error = ""
    last_type = ""
    started = time.perf_counter()
    for attempt in range(task.retries + 1):
        if pool.token.cancelled:
            return TaskOutcome(
                name=task.name,
                cancelled=True,
                attempts=attempt + 1,
                error=pool.token.reason or "cancelled",
                seconds=time.perf_counter() - started,
            )
        try:
            value = task.fn(*task.args, **task.kwargs)
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
            last_type = type(exc).__name__
            if attempt < task.retries and isinstance(exc, task.retry_on):
                pool.retries += 1
                delay = schedule[attempt] if attempt < len(schedule) else task.backoff
                if delay:
                    pool.token.wait(delay)
                continue
            pool.failed += 1
            return TaskOutcome(
                name=task.name,
                ok=False,
                error=last_error,
                error_type=last_type,
                attempts=attempt + 1,
                seconds=time.perf_counter() - started,
                metadata=dict(task.metadata),
            )
        pool.completed += 1
        pool.record_seconds(time.perf_counter() - started)
        return TaskOutcome(
            name=task.name,
            ok=True,
            value=value,
            attempts=attempt + 1,
            seconds=time.perf_counter() - started,
            metadata=dict(task.metadata),
        )
    pool.failed += 1
    return TaskOutcome(
        name=task.name,
        ok=False,
        error=last_error or "exhausted retries",
        error_type=last_type,
        attempts=task.retries + 1,
        seconds=time.perf_counter() - started,
    )


def run_parallel(
    tasks: Sequence[Task] | Sequence[Callable[[], Any]],
    *,
    max_workers: int = DEFAULT_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
    token: CancellationToken | None = None,
    name: str = "parallel",
) -> list[TaskOutcome]:
    """Run several tasks in parallel without raising.

    Args:
        tasks: :class:`Task` objects or zero-argument callables
        max_workers: thread ceiling
        timeout: default per-task timeout for bare callables
        token: shared cancellation token
        name: thread-name prefix

    Returns:
        one :class:`TaskOutcome` per task, in input order
    """
    prepared: list[Task] = []
    for index, item in enumerate(tasks):
        if isinstance(item, Task):
            prepared.append(item)
        elif callable(item):
            prepared.append(
                Task(name=getattr(item, "__name__", f"{name}[{index}]"), fn=item, timeout=timeout)
            )
        else:
            raise TypeError(f"task {index} is neither a Task nor callable: {type(item).__name__}")
    with WorkerPool(max_workers, name=name, token=token) as pool:
        return pool.run_many(prepared)


def gather(
    *fns: Callable[[], Any], max_workers: int = DEFAULT_WORKERS, timeout: float = DEFAULT_TIMEOUT
) -> list[Any]:
    """Run callables in parallel, returning values and ``None`` for failures."""
    outcomes = run_parallel(list(fns), max_workers=max_workers, timeout=timeout)
    return [outcome.value if outcome.ok else None for outcome in outcomes]


def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int = DEFAULT_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
    token: CancellationToken | None = None,
) -> list[TaskOutcome]:
    """Map a function over items in parallel."""
    materialised = list(items)
    tasks = [
        Task(name=f"map[{index}]", fn=fn, args=(item,), timeout=timeout)
        for index, item in enumerate(materialised)
    ]
    return run_parallel(tasks, max_workers=max_workers, token=token)


def chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    """Split a sequence into chunks of at most ``size`` items."""
    step = max(1, int(size))
    return [list(items[index : index + step]) for index in range(0, len(items), step)]


__all__ = [
    "DEFAULT_TIMEOUT",
    "DEFAULT_WORKERS",
    "MAX_WORKERS",
    "CancellationToken",
    "Task",
    "TaskOutcome",
    "WorkerPool",
    "chunked",
    "gather",
    "parallel_map",
    "run_parallel",
]
