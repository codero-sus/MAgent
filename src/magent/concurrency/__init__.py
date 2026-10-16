"""Threads, limits and batching for a sync-first framework.

Public API:

:class:`WorkerPool` / :class:`Task` / :class:`TaskOutcome` / :func:`run_parallel`
    bounded parallel execution where one failure never aborts the batch;
:class:`CancellationToken`
    cooperative cancellation, shared with the agent's own cancel flag;
:class:`TokenBucket` / :class:`Throttle` / :class:`Debouncer` / :class:`ConcurrencyLimiter`
    rate limiting, throttling, debouncing and slot bounding;
:func:`run_rate_limited`
    push a batch through a rate limit, sequentially or in parallel.

>>> from magent.concurrency import run_parallel
>>> outcomes = run_parallel([lambda: 1, lambda: 2])
>>> [outcome.value for outcome in outcomes]
[1, 2]
"""

from __future__ import annotations

from .limits import (DEFAULT_CAPACITY, DEFAULT_RATE, BatchResult, ConcurrencyLimiter, Debouncer,
                     LimitStats, Throttle, TokenBucket, first_success, interleave,
                     run_rate_limited)
from .pool import (DEFAULT_TIMEOUT, DEFAULT_WORKERS, MAX_WORKERS, CancellationToken, Task,
                   TaskOutcome, WorkerPool, chunked, gather, parallel_map, run_parallel)

__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_RATE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_WORKERS",
    "MAX_WORKERS",
    "BatchResult",
    "CancellationToken",
    "ConcurrencyLimiter",
    "Debouncer",
    "LimitStats",
    "Task",
    "TaskOutcome",
    "Throttle",
    "TokenBucket",
    "WorkerPool",
    "chunked",
    "first_success",
    "gather",
    "interleave",
    "parallel_map",
    "run_parallel",
    "run_rate_limited",
]
