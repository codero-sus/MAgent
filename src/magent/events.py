"""A typed, thread-safe in-process event bus.

Every interesting thing MAgent does is published here: agent steps, tool calls,
guardrail decisions, retries, cache hits, budget warnings. Three kinds of
consumer subscribe:

* the terminal renderer (live progress output),
* the session recorder (transcripts and traces),
* the HTTP server (Server-Sent Events to the browser UI).

Subscribers are invoked synchronously by default so ordering is preserved; a
subscriber that must not slow the agent can register with ``async_dispatch=True``
and it will run on a small worker pool.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .types import AgentEvent, EventKind
from .utils.ids import ulid_like

EventHandler = Callable[[AgentEvent], None]
"""Signature for an event subscriber."""


@dataclass(slots=True)
class Subscription:
    """One registered subscriber with its filter and statistics."""

    handler: EventHandler
    kinds: frozenset[str] = frozenset()
    name: str = ""
    async_dispatch: bool = False
    once: bool = False
    priority: int = 100
    delivered: int = 0
    errors: int = 0
    id: str = field(default_factory=lambda: ulid_like(prefix="sub_"))

    def accepts(self, event: AgentEvent) -> bool:
        """Report whether this subscription wants the given event."""
        if not self.kinds:
            return True
        kind = event.kind.value if isinstance(event.kind, EventKind) else str(event.kind)
        if kind in self.kinds:
            return True
        return any(
            kind.startswith(f"{pattern}.")
            or (pattern.endswith("*") and kind.startswith(pattern[:-1]))
            for pattern in self.kinds
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize subscription metadata (never the handler)."""
        return {
            "id": self.id,
            "name": self.name or getattr(self.handler, "__name__", "handler"),
            "kinds": sorted(self.kinds),
            "async_dispatch": self.async_dispatch,
            "priority": self.priority,
            "delivered": self.delivered,
            "errors": self.errors,
        }


class EventBus:
    """Publish/subscribe hub for :class:`AgentEvent` objects."""

    def __init__(
        self,
        *,
        history_size: int = 500,
        async_workers: int = 2,
        on_error: Callable[[BaseException, Subscription, AgentEvent], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._subscriptions: list[Subscription] = []
        self._history: list[AgentEvent] = []
        self.history_size = max(0, history_size)
        self._queue: queue.Queue[tuple[Subscription, AgentEvent] | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._async_workers = max(0, async_workers)
        self._running = False
        self.on_error = on_error
        self.published = 0
        self.dropped = 0
        self.by_kind: Counter[str] = Counter()
        self.started_at = time.time()

    # -- subscription --------------------------------------------------------

    def subscribe(
        self,
        handler: EventHandler,
        *,
        kinds: Iterable[str | EventKind] = (),
        name: str = "",
        async_dispatch: bool = False,
        once: bool = False,
        priority: int = 100,
    ) -> Subscription:
        """Register a handler and return its :class:`Subscription`."""
        normalized = frozenset(
            item.value if isinstance(item, EventKind) else str(item) for item in kinds
        )
        subscription = Subscription(
            handler=handler,
            kinds=normalized,
            name=name,
            async_dispatch=async_dispatch,
            once=once,
            priority=priority,
        )
        with self._lock:
            self._subscriptions.append(subscription)
            self._subscriptions.sort(key=lambda item: item.priority)
        return subscription

    def unsubscribe(self, subscription: Subscription | str) -> bool:
        """Remove a subscription by object or id; report whether it existed."""
        target = subscription if isinstance(subscription, Subscription) else None
        identifier = None if target else str(subscription)
        with self._lock:
            before = len(self._subscriptions)
            self._subscriptions = [
                item
                for item in self._subscriptions
                if (item is not target if target else item.id != identifier)
            ]
            return len(self._subscriptions) < before

    def clear(self) -> None:
        """Remove every subscription."""
        with self._lock:
            self._subscriptions.clear()

    def subscriptions(self) -> list[Subscription]:
        """Return a snapshot of the subscription list."""
        with self._lock:
            return list(self._subscriptions)

    # -- publishing ----------------------------------------------------------

    def publish(self, event: AgentEvent) -> int:
        """Dispatch one event and return how many handlers received it."""
        kind = event.kind.value if isinstance(event.kind, EventKind) else str(event.kind)
        with self._lock:
            self.published += 1
            self.by_kind[kind] += 1
            if self.history_size:
                self._history.append(event)
                if len(self._history) > self.history_size:
                    overflow = len(self._history) - self.history_size
                    del self._history[:overflow]
                    self.dropped += overflow
            targets = [item for item in self._subscriptions if item.accepts(event)]
        delivered = 0
        for subscription in targets:
            if subscription.async_dispatch:
                self._ensure_workers()
                self._queue.put((subscription, event))
            else:
                self._deliver(subscription, event)
            delivered += 1
            if subscription.once:
                self.unsubscribe(subscription)
        return delivered

    def emit(
        self, kind: EventKind | str, *, agent: str = "main", run_id: str = "", **data: Any
    ) -> int:
        """Build and publish an event in one call."""
        parsed = kind if isinstance(kind, EventKind) else EventKind(str(kind))
        return self.publish(AgentEvent(kind=parsed, data=data, agent=agent, run_id=run_id))

    def _deliver(self, subscription: Subscription, event: AgentEvent) -> None:
        """Invoke one handler, capturing and reporting any exception."""
        try:
            subscription.handler(event)
            subscription.delivered += 1
        except Exception as exc:
            subscription.errors += 1
            if self.on_error is not None:
                try:
                    self.on_error(exc, subscription, event)
                except Exception:
                    pass

    def _ensure_workers(self) -> None:
        """Start async dispatch workers on first use."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for index in range(self._async_workers):
                worker = threading.Thread(
                    target=self._worker_loop, name=f"magent-events-{index}", daemon=True
                )
                worker.start()
                self._workers.append(worker)

    def _worker_loop(self) -> None:
        """Drain the async dispatch queue until a sentinel arrives."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            subscription, event = item
            self._deliver(subscription, event)

    # -- inspection ----------------------------------------------------------

    def history(
        self, *, kinds: Iterable[str | EventKind] = (), limit: int | None = None
    ) -> list[AgentEvent]:
        """Return recorded events, optionally filtered by kind."""
        wanted = {item.value if isinstance(item, EventKind) else str(item) for item in kinds}
        with self._lock:
            events = list(self._history)
        if wanted:
            events = [
                event
                for event in events
                if (event.kind.value if isinstance(event.kind, EventKind) else str(event.kind))
                in wanted
            ]
        if limit is not None and limit >= 0:
            events = events[-limit:]
        return events

    def last(self, kind: EventKind | str) -> AgentEvent | None:
        """Return the most recent event of a given kind."""
        matches = self.history(kinds=[kind], limit=1)
        return matches[-1] if matches else None

    def counts(self) -> dict[str, int]:
        """Return how many events of each kind were published."""
        with self._lock:
            return dict(self.by_kind)

    def stats(self) -> dict[str, Any]:
        """Return bus telemetry."""
        with self._lock:
            return {
                "published": self.published,
                "dropped": self.dropped,
                "history_size": len(self._history),
                "subscriptions": [item.to_dict() for item in self._subscriptions],
                "by_kind": dict(self.by_kind),
                "queue_depth": self._queue.qsize(),
                "uptime_seconds": round(time.time() - self.started_at, 2),
            }

    def wait_for(
        self, predicate: Callable[[AgentEvent], bool], *, timeout: float = 5.0
    ) -> AgentEvent | None:
        """Block until a matching event is published, or the timeout elapses.

        Events already in the history are checked first, so a caller that waited
        for a run to finish before subscribing still finds the event instead of
        timing out on a race it cannot win.
        """
        for event in self.history(limit=None):
            if predicate(event):
                return event
        deadline = time.monotonic() + timeout
        found: list[AgentEvent] = []

        def watcher(event: AgentEvent) -> None:
            """Record the event when it satisfies the predicate."""
            if predicate(event) and not found:
                found.append(event)

        subscription = self.subscribe(watcher, name="wait_for")
        try:
            while time.monotonic() < deadline:
                if found:
                    return found[0]
                time.sleep(0.005)
            return found[0] if found else None
        finally:
            self.unsubscribe(subscription)

    def iter_queue(self) -> Iterator[AgentEvent]:
        """Yield every recorded event, for replaying into a renderer."""
        yield from self.history()

    def drain(self, *, timeout: float = 1.0) -> None:
        """Wait until the async queue is empty, bounded by ``timeout``."""
        deadline = time.monotonic() + timeout
        while self._queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.005)

    def shutdown(self) -> None:
        """Stop async workers and clear subscriptions."""
        with self._lock:
            if not self._running:
                self._subscriptions.clear()
                return
            self._running = False
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._workers.clear()
        self._subscriptions.clear()


class EventRecorder:
    """Subscriber that stores events into a bounded list, for transcripts."""

    def __init__(
        self, bus: EventBus, *, kinds: Iterable[str | EventKind] = (), limit: int = 2000
    ) -> None:
        self.events: list[AgentEvent] = []
        self.limit = limit
        self.kinds = frozenset(
            item.value if isinstance(item, EventKind) else str(item) for item in kinds
        )
        self.subscription = bus.subscribe(self._receive, name="recorder")

    def _receive(self, event: AgentEvent) -> None:
        """Append one event, dropping the oldest when the limit is reached."""
        kind = event.kind.value if isinstance(event.kind, EventKind) else str(event.kind)
        if self.kinds and kind not in self.kinds:
            return
        self.events.append(event)
        if len(self.events) > self.limit:
            del self.events[: len(self.events) - self.limit]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize every recorded event."""
        return [event.to_dict() for event in self.events]

    def clear(self) -> None:
        """Drop recorded events."""
        self.events.clear()

    def __len__(self) -> int:
        """Return how many events were recorded."""
        return len(self.events)


class EventBroadcaster:
    """Fan events out to multiple SSE client queues."""

    def __init__(self, bus: EventBus, *, max_clients: int = 64, queue_size: int = 1000) -> None:
        self._lock = threading.RLock()
        self._clients: dict[str, queue.Queue[AgentEvent | None]] = {}
        self.max_clients = max_clients
        self.queue_size = queue_size
        self.subscription = bus.subscribe(self._broadcast, name="broadcaster", async_dispatch=False)

    def _broadcast(self, event: AgentEvent) -> None:
        """Push one event into every connected client queue."""
        with self._lock:
            clients = list(self._clients.items())
        for client_id, client_queue in clients:
            try:
                client_queue.put_nowait(event)
            except queue.Full:
                self.disconnect(client_id)

    def connect(self, client_id: str = "") -> tuple[str, queue.Queue]:
        """Register a client and return ``(id, queue)``."""
        identifier = client_id or ulid_like(prefix="sse_")
        with self._lock:
            if len(self._clients) >= self.max_clients:
                oldest = next(iter(self._clients))
                self._clients.pop(oldest, None)
            client_queue: queue.Queue[AgentEvent | None] = queue.Queue(maxsize=self.queue_size)
            self._clients[identifier] = client_queue
        return identifier, client_queue

    def disconnect(self, client_id: str) -> None:
        """Remove a client queue."""
        with self._lock:
            self._clients.pop(client_id, None)

    def client_count(self) -> int:
        """Return how many clients are connected."""
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        """Signal every client that the stream is over."""
        with self._lock:
            for client_queue in self._clients.values():
                client_queue.put(None)
            self._clients.clear()


def make_bus(**kwargs: Any) -> EventBus:
    """Create a bus with sensible defaults for an agent run."""
    return EventBus(**kwargs)


__all__ = [
    "EventBroadcaster",
    "EventBus",
    "EventHandler",
    "EventRecorder",
    "Subscription",
    "make_bus",
]
