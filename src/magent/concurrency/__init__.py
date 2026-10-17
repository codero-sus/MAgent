"""Concurrency: bounded parallel work and the synchronization around it.

Two halves:

- :mod:`magent.concurrency.pool` runs work in parallel and keeps every outcome,
  including the failures, in submission or completion order.
- :mod:`magent.concurrency.sync` provides the coordination an agent run needs:
  cancellation, permit limits, throttling, debouncing, one-shot init.

Parallel tool dispatch, RAG ingestion, and evaluation runs all go through
:func:`parallel_map`, so timeout handling, error capture, and trace-context
propagation are implemented once::

    result = parallel_map(index_file, paths, max_workers=4, timeout=30)
    if not result.ok:
        log(result.errors)
    chunks = result.values

Heavier resilience patterns live elsewhere on purpose: retries and circuit
breaking in :mod:`magent.llm.retry`, per-caller rate limits in
:mod:`magent.guardrails.policy`.
"""

from __future__ import annotations

from .pool import (
    DEFAULT_MAX_WORKERS,
    ParallelResult,
    TaskOutcome,
    WorkerPool,
    bounded_chunks,
    chunked,
    first_success,
    gather,
    parallel_map,
)
from .sync import (
    DEFAULT_ACQUIRE_TIMEOUT,
    Barrier,
    CancellationToken,
    ConcurrencyLimiter,
    Debouncer,
    Throttle,
    once,
    retry_until,
)

__all__ = [
    "DEFAULT_ACQUIRE_TIMEOUT",
    "DEFAULT_MAX_WORKERS",
    "Barrier",
    "CancellationToken",
    "ConcurrencyLimiter",
    "Debouncer",
    "ParallelResult",
    "TaskOutcome",
    "Throttle",
    "WorkerPool",
    "bounded_chunks",
    "chunked",
    "first_success",
    "gather",
    "once",
    "parallel_map",
    "retry_until",
]
