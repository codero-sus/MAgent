"""Tests for MAgent's typed, thread-safe event bus.

The terminal renderer, trace recorder, metrics, and SSE server all consume the
same event stream. These tests lock down filtering, ordering, one-shot handlers,
error isolation, bounded history, asynchronous delivery, and broadcaster client
management without depending on timing-sensitive sleeps.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest

from magent.events import EventBroadcaster, EventBus, EventRecorder, Subscription
from magent.types import AgentEvent, EventKind


class PublishTests(unittest.TestCase):
    """Synchronous publication preserves order and metadata."""

    def setUp(self) -> None:
        """Build a fresh bus for each test."""
        self.bus = EventBus()

    def tearDown(self) -> None:
        """Stop workers if a test created them."""
        self.bus.shutdown()

    def test_emit_constructs_and_delivers_an_event(self) -> None:
        """emit() is the concise path used throughout the agent loop."""
        seen: list[AgentEvent] = []
        self.bus.subscribe(seen.append)
        delivered = self.bus.emit(EventKind.RUN_STARTED, agent="worker", run_id="r1", task="x")
        self.assertEqual(delivered, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].kind, EventKind.RUN_STARTED)
        self.assertEqual(seen[0].agent, "worker")
        self.assertEqual(seen[0].run_id, "r1")
        self.assertEqual(seen[0].data["task"], "x")

    def test_publish_accepts_a_prebuilt_event(self) -> None:
        """publish() returns the number of matching subscribers."""
        self.bus.subscribe(lambda event: None)
        self.bus.subscribe(lambda event: None)
        count = self.bus.publish(AgentEvent(kind=EventKind.LOG, data={"message": "hello"}))
        self.assertEqual(count, 2)

    def test_delivery_follows_priority(self) -> None:
        """Lower priority numbers run first."""
        order: list[str] = []
        self.bus.subscribe(lambda event: order.append("late"), priority=200)
        self.bus.subscribe(lambda event: order.append("early"), priority=1)
        self.bus.emit(EventKind.LOG)
        self.assertEqual(order, ["early", "late"])

    def test_handler_failure_does_not_stop_the_next_handler(self) -> None:
        """One observer cannot break the agent or other observers."""
        seen: list[str] = []

        def broken(event: AgentEvent) -> None:
            raise RuntimeError("listener failed")

        bad = self.bus.subscribe(broken)
        self.bus.subscribe(lambda event: seen.append("good"))
        self.bus.emit(EventKind.LOG)
        self.assertEqual(seen, ["good"])
        self.assertEqual(bad.errors, 1)

    def test_on_error_receives_listener_failures(self) -> None:
        """Bus errors can be logged without escaping publication."""
        captured: list[tuple[str, str]] = []
        bus = EventBus(on_error=lambda exc, sub, event:
                       captured.append((str(exc), event.kind.value)))
        bus.subscribe(lambda event: (_ for _ in ()).throw(ValueError("bad")))
        bus.emit(EventKind.LOG)
        self.assertEqual(captured, [("bad", "log")])
        bus.shutdown()


class FilterTests(unittest.TestCase):
    """Subscriptions can select exact kinds, dotted prefixes, or wildcards."""

    def setUp(self) -> None:
        self.bus = EventBus()

    def tearDown(self) -> None:
        self.bus.shutdown()

    def test_exact_kind_filter(self) -> None:
        """An exact subscription ignores unrelated events."""
        seen: list[EventKind] = []
        self.bus.subscribe(lambda event: seen.append(event.kind), kinds=[EventKind.TOOL_CALL])
        self.bus.emit(EventKind.RUN_STARTED)
        self.bus.emit(EventKind.TOOL_CALL)
        self.assertEqual(seen, [EventKind.TOOL_CALL])

    def test_dotted_parent_filter(self) -> None:
        """Subscribing to ``tool`` accepts all ``tool.*`` events."""
        seen: list[str] = []
        self.bus.subscribe(lambda event: seen.append(event.kind.value), kinds=["tool"])
        self.bus.emit(EventKind.TOOL_CALL)
        self.bus.emit(EventKind.TOOL_RESULT)
        self.bus.emit(EventKind.LOG)
        self.assertEqual(seen, ["tool.call", "tool.result"])

    def test_wildcard_filter(self) -> None:
        """``llm.*`` matches all LLM events."""
        seen: list[str] = []
        self.bus.subscribe(lambda event: seen.append(event.kind.value), kinds=["llm.*"])
        self.bus.emit(EventKind.LLM_REQUEST)
        self.bus.emit(EventKind.LLM_RESPONSE)
        self.bus.emit(EventKind.RUN_FINISHED)
        self.assertEqual(seen, ["llm.request", "llm.response"])

    def test_once_unsubscribes_after_first_delivery(self) -> None:
        """A one-shot handler is invoked exactly once."""
        seen: list[AgentEvent] = []
        self.bus.subscribe(seen.append, once=True)
        self.bus.emit(EventKind.LOG)
        self.bus.emit(EventKind.LOG)
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.bus.subscriptions(), [])


class SubscriptionTests(unittest.TestCase):
    """Subscriptions expose only safe metadata and can be removed by id."""

    def test_unsubscribe_by_object_and_id(self) -> None:
        """Both public forms remove the target and report success."""
        bus = EventBus()
        first = bus.subscribe(lambda event: None)
        second = bus.subscribe(lambda event: None)
        self.assertTrue(bus.unsubscribe(first))
        self.assertTrue(bus.unsubscribe(second.id))
        self.assertFalse(bus.unsubscribe(second.id))
        bus.shutdown()

    def test_clear_removes_every_subscription(self) -> None:
        """clear() leaves history alone but removes handlers."""
        bus = EventBus()
        bus.subscribe(lambda event: None)
        bus.subscribe(lambda event: None)
        bus.clear()
        self.assertEqual(bus.subscriptions(), [])
        bus.shutdown()

    def test_to_dict_omits_the_handler(self) -> None:
        """Inspection metadata cannot serialize an arbitrary callable."""
        sub = Subscription(handler=lambda event: None, name="test", kinds=frozenset({"log"}))
        payload = sub.to_dict()
        self.assertNotIn("handler", payload)
        self.assertEqual(payload["name"], "test")
        self.assertEqual(payload["kinds"], ["log"])


class HistoryTests(unittest.TestCase):
    """History is bounded and queryable for diagnostics and replay."""

    def test_history_is_bounded(self) -> None:
        """Old events are dropped once the configured limit is exceeded."""
        bus = EventBus(history_size=2)
        bus.emit(EventKind.RUN_STARTED, n=1)
        bus.emit(EventKind.LOG, n=2)
        bus.emit(EventKind.RUN_FINISHED, n=3)
        events = bus.history()
        self.assertEqual([event.data["n"] for event in events], [2, 3])
        self.assertEqual(bus.stats()["dropped"], 1)
        bus.shutdown()

    def test_history_filter_and_limit(self) -> None:
        """Kinds are filtered before the tail limit is applied."""
        bus = EventBus()
        bus.emit(EventKind.LOG, n=1)
        bus.emit(EventKind.RUN_STARTED, n=2)
        bus.emit(EventKind.LOG, n=3)
        events = bus.history(kinds=[EventKind.LOG], limit=1)
        self.assertEqual([event.data["n"] for event in events], [3])
        bus.shutdown()

    def test_last_returns_the_latest_matching_event(self) -> None:
        """last() does not confuse adjacent event kinds."""
        bus = EventBus()
        bus.emit(EventKind.LOG, n=1)
        bus.emit(EventKind.RUN_STARTED, n=2)
        bus.emit(EventKind.LOG, n=3)
        self.assertEqual(bus.last(EventKind.LOG).data["n"], 3)  # type: ignore[union-attr]
        self.assertIsNone(bus.last(EventKind.TOOL_ERROR))
        bus.shutdown()

    def test_counts_track_wire_kind_names(self) -> None:
        """Counters use the dotted values exposed in metrics and JSON."""
        bus = EventBus()
        bus.emit(EventKind.RUN_STARTED)
        bus.emit(EventKind.RUN_STARTED)
        bus.emit(EventKind.RUN_FINISHED)
        self.assertEqual(bus.counts(), {"run.started": 2, "run.finished": 1})
        bus.shutdown()

    def test_iter_queue_replays_history(self) -> None:
        """The iterator returns the same objects in publication order."""
        bus = EventBus()
        bus.emit(EventKind.LOG, n=1)
        bus.emit(EventKind.LOG, n=2)
        self.assertEqual([event.data["n"] for event in bus.iter_queue()], [1, 2])
        bus.shutdown()


class AsyncTests(unittest.TestCase):
    """Slow observers can run off the agent thread."""

    def test_async_handler_is_delivered(self) -> None:
        """drain waits until a queued handler has run."""
        bus = EventBus(async_workers=1)
        done = threading.Event()
        bus.subscribe(lambda event: done.set(), async_dispatch=True)
        bus.emit(EventKind.LOG)
        bus.drain(timeout=1.0)
        self.assertTrue(done.wait(1.0))
        bus.shutdown()

    def test_wait_for_observes_a_future_event(self) -> None:
        """wait_for returns the first future event satisfying its predicate."""
        bus = EventBus()
        result: list[AgentEvent | None] = []
        thread = threading.Thread(
            target=lambda: result.append(bus.wait_for(
                lambda event: event.kind is EventKind.RUN_FINISHED, timeout=1.0)))
        thread.start()
        time.sleep(0.02)
        bus.emit(EventKind.RUN_STARTED)
        bus.emit(EventKind.RUN_FINISHED, answer=42)
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(result[0])
        self.assertEqual(result[0].data["answer"], 42)  # type: ignore[union-attr]
        bus.shutdown()

    def test_wait_for_times_out_cleanly(self) -> None:
        """No match yields None and removes its temporary subscription."""
        bus = EventBus()
        self.assertIsNone(bus.wait_for(lambda event: True, timeout=0.01))
        self.assertEqual(bus.subscriptions(), [])
        bus.shutdown()


class RecorderTests(unittest.TestCase):
    """The recorder stores a bounded transcript independent of bus history."""

    def test_recorder_limit(self) -> None:
        """The oldest recorder entries are dropped at its own limit."""
        bus = EventBus()
        recorder = EventRecorder(bus, limit=2)
        for n in range(3):
            bus.emit(EventKind.LOG, n=n)
        self.assertEqual([event.data["n"] for event in recorder.events], [1, 2])
        self.assertEqual(len(recorder), 2)
        bus.shutdown()

    def test_recorder_filter(self) -> None:
        """A recorder can keep just one kind."""
        bus = EventBus()
        recorder = EventRecorder(bus, kinds=[EventKind.TOOL_ERROR])
        bus.emit(EventKind.LOG)
        bus.emit(EventKind.TOOL_ERROR, message="bad")
        self.assertEqual(len(recorder), 1)
        self.assertEqual(recorder.to_dicts()[0]["kind"], "tool.error")
        recorder.clear()
        self.assertEqual(len(recorder), 0)
        bus.shutdown()


class BroadcasterTests(unittest.TestCase):
    """Each connected SSE client receives the same event."""

    def test_clients_receive_events(self) -> None:
        """Broadcast queues get the exact AgentEvent object."""
        bus = EventBus()
        broadcaster = EventBroadcaster(bus)
        first_id, first = broadcaster.connect("first")
        second_id, second = broadcaster.connect("second")
        bus.emit(EventKind.RUN_STARTED, task="x")
        self.assertEqual(first.get_nowait().data["task"], "x")
        self.assertEqual(second.get_nowait().kind, EventKind.RUN_STARTED)
        self.assertEqual(broadcaster.client_count(), 2)
        broadcaster.disconnect(first_id)
        broadcaster.disconnect(second_id)
        broadcaster.close()
        bus.shutdown()

    def test_connect_evicts_oldest_at_capacity(self) -> None:
        """A bounded broadcaster does not accumulate clients forever."""
        bus = EventBus()
        broadcaster = EventBroadcaster(bus, max_clients=1)
        broadcaster.connect("old")
        broadcaster.connect("new")
        self.assertEqual(broadcaster.client_count(), 1)
        bus.emit(EventKind.LOG)
        broadcaster.close()
        bus.shutdown()

    def test_full_client_is_disconnected(self) -> None:
        """A stalled SSE client cannot block agent publication."""
        bus = EventBus()
        broadcaster = EventBroadcaster(bus, queue_size=1)
        _, client = broadcaster.connect("slow")
        bus.emit(EventKind.LOG, n=1)
        bus.emit(EventKind.LOG, n=2)
        self.assertEqual(broadcaster.client_count(), 0)
        self.assertEqual(client.get_nowait().data["n"], 1)
        broadcaster.close()
        bus.shutdown()


if __name__ == "__main__":
    unittest.main()
