"""Bounded parallelism that keeps results, errors, and context intact.

:mod:`concurrent.futures` gives the mechanics but leaves three decisions to
every call site: what to do when one task raises, whether results come back in
submission order, and whether the caller's context (the active trace span, in
MAgent's case) is visible inside the worker. Getting any of them wrong
produces a bug that only shows up under load.

This module fixes those decisions once:

* every task's outcome is captured, so one failure cannot discard the rest,
* results are returned in submission order unless the caller asks otherwise,
* each task runs inside a copy of the submitting context, so spans nest
  correctly across threads.
"""

from __future__ import annotations

import contextvars
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

from ..errors import ConcurrencyError, PoolTimeoutError
from ..logging_ import get_logger

LOGGER = get_logger("concurrency")
"""Logger for pool faults."""

T = TypeVar("T")
"""Type of a task's input item."""

R = TypeVar("R")
"""Type of a task's result."""

DEFAULT_MAX_WORKERS = 8
"""Worker ceiling used when the caller does not specify one."""


@dataclass(slots=True)
class TaskOutcome:
    """The result of one task, whether it succeeded or raised."""

    index: int
    value: Any = None
    error: str | None = None
    exception: BaseException | None = None
    duration_ms: float = 0.0
    started_at: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the task completed without raising."""
        return self.error is None

    def result(self, default: Any = None) -> Any:
        """Return the value, or ``default`` when the task failed."""
        return self.value if self.ok else default

    def raise_if_failed(self) -> Any:
        """Return the value or re-raise the original exception.

        Raises:
            ConcurrencyError: the task failed without a stored exception
        """
        if self.ok:
            return self.value
        if self.exception is not None:
            raise self.exception
        raise ConcurrencyError(self.error or "task failed", details={"index": self.index})

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable view, omitting the exception object."""
        return {
            "index": self.index,
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 3),
        }

    def __repr__(self) -> str:
        """Debug representation."""
        state = "ok" if self.ok else f"error={self.error!r}"
        return f"TaskOutcome(index={self.index}, {state}, {self.duration_ms:.1f}ms)"


@dataclass(slots=True)
class ParallelResult:
    """Everything a parallel pass produced."""

    outcomes: list[TaskOutcome] = field(default_factory=list)
    ordered: bool = True
    submitted: int = 0
    duration_ms: float = 0.0
    cancelled: bool = False

    @property
    def values(self) -> list[Any]:
        """Results of the tasks that succeeded, in order."""
        return [outcome.value for outcome in self.outcomes if outcome.ok]

    @property
    def errors(self) -> list[str]:
        """Error messages from the tasks that failed."""
        return [outcome.error or "unknown error" for outcome in self.outcomes if not outcome.ok]

    @property
    def ok(self) -> bool:
        """Whether every task succeeded."""
        return all(outcome.ok for outcome in self.outcomes)

    @property
    def succeeded(self) -> int:
        """How many tasks completed."""
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def failed(self) -> int:
        """How many tasks raised."""
        return len(self.outcomes) - self.succeeded

    def by_index(self) -> dict[int, TaskOutcome]:
        """Return outcomes keyed by their submission index."""
        return {outcome.index: outcome for outcome in self.outcomes}

    def stats(self) -> dict[str, Any]:
        """Return aggregate timing and success data."""
        durations = [outcome.duration_ms for outcome in self.outcomes]
        return {
            "submitted": self.submitted,
            "completed": len(self.outcomes),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "ordered": self.ordered,
            "cancelled": self.cancelled,
            "wall_ms": round(self.duration_ms, 3),
            "task_ms_total": round(sum(durations), 3),
            "task_ms_max": round(max(durations), 3) if durations else 0.0,
            "speedup": round(sum(durations) / self.duration_ms, 2) if self.duration_ms else 0.0,
        }

    def __len__(self) -> int:
        """Number of outcomes."""
        return len(self.outcomes)

    def __iter__(self) -> Iterator[TaskOutcome]:
        """Iterate outcomes."""
        return iter(self.outcomes)


class WorkerPool:
    """A reusable, bounded thread pool that propagates context into workers.

    Threads are named for the pool so a stack trace or a thread dump says which
    subsystem was running, rather than showing an anonymous
    ``ThreadPoolExecutor-3_0``.
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS, *, name: str = "magent",
                 propagate_context: bool = True) -> None:
        """Create a pool.

        Args:
            max_workers: maximum concurrent threads
            name: label used in thread names and logs
            propagate_context: copy the submitting context into each task
        """
        self.max_workers = max(1, int(max_workers))
        self.name = name
        self.propagate_context = bool(propagate_context)
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.RLock()
        self.submitted = 0
        self.completed = 0
        self.failures = 0
        self._index = 0

    @property
    def executor(self) -> ThreadPoolExecutor:
        """The underlying executor, created on first use."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix=self.name,
                )
            return self._executor

    def _wrap(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Callable[[], R]:
        """Bind a call, optionally capturing the current context."""
        if not self.propagate_context:
            return lambda: fn(*args, **kwargs)
        context = contextvars.copy_context()
        return lambda: context.run(fn, *args, **kwargs)

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future:
        """Submit one task, returning its future."""
        with self._lock:
            self.submitted += 1
        return self.executor.submit(self._wrap(fn, *args, **kwargs))

    def map(self, fn: Callable[[T], R], items: Iterable[T], **kwargs: Any) -> ParallelResult:
        """Run ``fn`` over ``items`` in parallel, preserving order."""
        return parallel_map(fn, list(items), pool=self, **kwargs)

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Stop the pool; a later call recreates it."""
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def stats(self) -> dict[str, Any]:
        """Return pool counters."""
        with self._lock:
            alive = self._executor is not None
        return {
            "name": self.name,
            "max_workers": self.max_workers,
            "submitted": self.submitted,
            "completed": self.completed,
            "failures": self.failures,
            "active": alive,
            "propagate_context": self.propagate_context,
        }

    def __enter__(self) -> WorkerPool:
        """Create the executor eagerly."""
        _ = self.executor
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Shut the pool down."""
        self.shutdown()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"WorkerPool(name={self.name!r}, max_workers={self.max_workers})"


def parallel_map(fn: Callable[[T], R], items: Sequence[T], *, max_workers: int = 0,
                 pool: WorkerPool | None = None, timeout: float = 0.0,
                 ordered: bool = True, stop_on_error: bool = False,
                 on_error: Callable[[TaskOutcome], None] | None = None,
                 chunk_name: str = "") -> ParallelResult:
    """Apply ``fn`` to every item concurrently and collect all outcomes.

    A raised exception becomes a failed :class:`TaskOutcome` instead of
    propagating, so one bad item cannot discard the work already done. Pass
    ``stop_on_error=True`` to cancel the remaining tasks on the first failure.

    Args:
        fn: callable applied to each item
        items: the work to divide
        max_workers: worker ceiling; ignored when ``pool`` is given
        pool: an existing pool to reuse, otherwise one is created and closed
        timeout: total seconds to wait, 0 for no limit
        ordered: return outcomes in submission order
        stop_on_error: cancel pending tasks after the first failure
        on_error: callback invoked with each failed outcome
        chunk_name: label used in thread names and logs

    Returns:
        A :class:`ParallelResult` describing every outcome.

    Raises:
        PoolTimeoutError: the overall timeout elapsed before tasks finished
    """
    work = list(items)
    started = time.perf_counter()
    result = ParallelResult(ordered=ordered, submitted=len(work))
    if not work:
        return result

    label = chunk_name or "magent-map"
    tracker = pool
    owns_executor = tracker is None
    if tracker is None:
        ceiling = max_workers or min(DEFAULT_MAX_WORKERS, len(work))
        tracker = WorkerPool(ceiling, name=label)
        executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=min(tracker.max_workers, len(work)), thread_name_prefix=label
        )
    else:
        executor = tracker.executor

    futures: list[tuple[int, Future]] = []
    halted = threading.Event() if stop_on_error else None
    try:
        for index, item in enumerate(work):
            futures.append((index, executor.submit(tracker._wrap(fn, item))))
            tracker.submitted += 1
        deadline = started + timeout if timeout > 0 else None
        pending: Iterable[tuple[int, Future]] = futures
        if not ordered:
            from concurrent.futures import as_completed

            by_future = {future: index for index, future in futures}
            wait_for = None if deadline is None else max(0.0, deadline - time.perf_counter())
            pending = [(by_future[future], future)
                       for future in as_completed(list(by_future), timeout=wait_for)]
        for index, future in pending:
            if halted is not None and halted.is_set():
                future.cancel()
                continue
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    remaining = 0.0
            task_started = time.perf_counter()
            try:
                value = future.result(timeout=remaining)
                outcome = TaskOutcome(index=index, value=value, started_at=task_started,
                                      duration_ms=(time.perf_counter() - task_started) * 1000.0)
            except FuturesTimeout:
                outcome = TaskOutcome(index=index, error="timeout", started_at=task_started,
                                      duration_ms=(time.perf_counter() - task_started) * 1000.0)
                result.outcomes.append(outcome)
                tracker.completed += 1
                tracker.failures += 1
                _notify(on_error, outcome)
                if deadline is not None:
                    result.cancelled = True
                    for _, other in futures:
                        other.cancel()
                    raise _timeout_error(label, timeout, result, len(work)) from None
            except BaseException as exc:  # noqa: BLE001 - captured per task
                outcome = TaskOutcome(
                    index=index,
                    error=f"{type(exc).__name__}: {exc}",
                    exception=exc,
                    started_at=task_started,
                    duration_ms=(time.perf_counter() - task_started) * 1000.0,
                )
                if halted is not None:
                    halted.set()
            result.outcomes.append(outcome)
            tracker.completed += 1
            if not outcome.ok:
                tracker.failures += 1
                _notify(on_error, outcome)
    except FuturesTimeout:
        result.cancelled = True
        for _, pending_future in futures:
            pending_future.cancel()
        raise _timeout_error(label, timeout, result, len(work)) from None
    finally:
        if owns_executor:
            executor.shutdown(wait=False)
    if ordered:
        result.outcomes.sort(key=lambda item: item.index)
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result


def _timeout_error(label: str, timeout: float, result: ParallelResult,
                   submitted: int) -> PoolTimeoutError:
    """Build a timeout error that carries the work already finished.

    Discarding partial results on a deadline makes a batch all-or-nothing, so
    the completed outcomes travel with the error and the caller can keep them.
    """
    result.outcomes.sort(key=lambda item: item.index)
    result.duration_ms = max(result.duration_ms, 0.0)
    return PoolTimeoutError(
        f"parallel {label} exceeded {timeout}s",
        details={
            "completed": len(result.outcomes),
            "submitted": submitted,
            "timeout": timeout,
            "outcomes": [outcome.to_dict() for outcome in result.outcomes],
        },
    )


def _notify(callback: Callable[[TaskOutcome], None] | None, outcome: TaskOutcome) -> None:
    """Invoke an error callback, never letting it break the pass."""
    if callback is None:
        return
    try:
        callback(outcome)
    except Exception as exc:  # noqa: BLE001 - a bad callback must not lose results
        LOGGER.debug("on_error callback raised: %s", exc)


def gather(tasks: Sequence[Callable[[], R]], *, max_workers: int = 0,
           timeout: float = 0.0, stop_on_error: bool = False) -> ParallelResult:
    """Run zero-argument callables concurrently.

    Args:
        tasks: callables taking no arguments
        max_workers: worker ceiling
        timeout: total seconds to wait, 0 for no limit
        stop_on_error: cancel pending tasks after the first failure
    """
    return parallel_map(lambda task: task(), list(tasks), max_workers=max_workers,
                        timeout=timeout, stop_on_error=stop_on_error, chunk_name="gather")


def first_success(tasks: Sequence[Callable[[], R]], *, timeout: float = 0.0,
                  max_workers: int = 0) -> TaskOutcome | None:
    """Return the first task that succeeds, cancelling the rest.

    Useful for racing redundant paths - several mirrors, a cache and an origin
    - where any one answer is enough and waiting for all of them is waste.

    Args:
        tasks: callables to race
        timeout: total seconds to wait, 0 for no limit
        max_workers: worker ceiling
    """
    if not tasks:
        return None
    started = time.perf_counter()
    workers = max(1, min(max_workers or len(tasks), len(tasks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="magent-race") as executor:
        futures = {executor.submit(contextvars.copy_context().run, task): index
                   for index, task in enumerate(tasks)}
        winner: TaskOutcome | None = None
        try:
            for future in _as_completed(futures, timeout=timeout, started=started):
                index = futures[future]
                try:
                    value = future.result(timeout=0)
                    winner = TaskOutcome(index=index, value=value,
                                         duration_ms=(time.perf_counter() - started) * 1000.0)
                    break
                except BaseException as exc:  # noqa: BLE001 - keep racing
                    LOGGER.debug("race task %d failed: %s", index, exc)
        finally:
            for future in futures:
                future.cancel()
        return winner


def _as_completed(futures: dict[Future, int], *, timeout: float,
                  started: float) -> Iterator[Future]:
    """Yield futures as they finish, honouring an overall deadline."""
    from concurrent.futures import as_completed

    deadline = started + timeout if timeout > 0 else None
    remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
    yield from as_completed(list(futures), timeout=remaining)


def chunked(items: Sequence[T], size: int) -> list[list[T]]:
    """Split a sequence into batches of at most ``size``."""
    step = max(1, int(size))
    return [list(items[index:index + step]) for index in range(0, len(items), step)]


def bounded_chunks(items: Sequence[T], workers: int) -> list[list[T]]:
    """Split work into one batch per worker, balanced by round robin."""
    count = max(1, int(workers))
    buckets: list[list[T]] = [[] for _ in range(min(count, len(items) or 1))]
    for index, item in enumerate(items):
        buckets[index % len(buckets)].append(item)
    return [bucket for bucket in buckets if bucket]


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "ParallelResult",
    "TaskOutcome",
    "WorkerPool",
    "bounded_chunks",
    "chunked",
    "first_success",
    "gather",
    "parallel_map",
]
