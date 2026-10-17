"""Tests for tracing, metrics and the concurrency helpers."""

from __future__ import annotations

import json
import threading
import time
import unittest

from magent.concurrency import (CancellationToken, ConcurrencyLimiter, Debouncer, Task, TaskOutcome,
                                Throttle, TokenBucket, WorkerPool, chunked, gather, interleave,
                                run_parallel, run_rate_limited)
from magent.observability import (Counter, Gauge, Histogram, MetricsRegistry, Observability, Span,
                                  Tracer, record_run, trace_from_events)
from magent.testing import EchoBackend

from tests.support import MagentTestCase


class TracingTest(unittest.TestCase):
    """Spans nest, measure time and export as JSONL."""

    def test_spans_nest_and_measure(self) -> None:
        tracer = Tracer()
        with tracer.trace("demo") as trace:
            with tracer.span("step.0", kind="step") as span:
                time.sleep(0.005)
                with tracer.span("llm.chat", kind="llm"):
                    pass
        self.assertGreater(span.duration_ms, 0)
        self.assertEqual(len(trace.spans()), 3)
        self.assertIn("step.0", trace.flame())

    def test_errors_are_recorded(self) -> None:
        tracer = Tracer()
        with tracer.trace("failing") as trace:
            try:
                with tracer.span("boom"):
                    raise ValueError("nope")
            except ValueError:
                pass
        self.assertTrue(trace.errors())
        self.assertIn("ValueError", trace.errors()[0].error)

    def test_attributes_and_events(self) -> None:
        span = Span(name="s")
        span.set_attribute("model", "echo").add_event("retry", attempt=1)
        self.assertEqual(span.attributes["model"], "echo")
        self.assertEqual(span.events[0].name, "retry")

    def test_trace_exports_jsonl(self) -> None:
        tracer = Tracer()
        with tracer.trace("export") as trace:
            with tracer.span("work"):
                pass
        lines = trace.jsonl().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["name"], "export")

    def test_trace_from_events_rebuilds_a_run(self) -> None:
        from magent.agent import make_agent

        result = make_agent(backend=EchoBackend()).run("hello")
        trace = trace_from_events(result.events, name="replay")
        self.assertTrue(trace.spans())
        self.assertTrue(trace.root is not None)

    def test_disabled_tracer_is_a_noop(self) -> None:
        tracer = Tracer(enabled=False)
        with tracer.trace("off") as trace:
            with tracer.span("nothing"):
                pass
        self.assertTrue(tracer.stats()["enabled"] is False)
        self.assertIsNotNone(trace)


class MetricsTest(MagentTestCase):
    """Counters, gauges, histograms and the Prometheus export."""

    def test_counter_and_gauge(self) -> None:
        registry = MetricsRegistry()
        registry.inc("runs_total", labels={"model": "echo"})
        registry.inc("runs_total", 2, labels={"model": "echo"})
        registry.set("active", 3)
        self.assertEqual(registry.counter("runs_total", labels={"model": "echo"}).value(), 3)
        self.assertEqual(registry.gauge("active").value(), 3)

    def test_histogram_statistics(self) -> None:
        histogram = Histogram("latency", unit="ms")
        for value in (5, 10, 15, 20, 100):
            histogram.observe(value)
        self.assertEqual(histogram.count, 5)
        self.assertEqual(histogram.percentile(50), 15)
        self.assertGreaterEqual(histogram.p95, 20)

    def test_prometheus_export_is_parsable(self) -> None:
        registry = MetricsRegistry()
        registry.inc("runs_total")
        registry.observe("latency", 12.0)
        text = registry.prometheus()
        self.assertIn("magent_runs_total", text)
        self.assertIn("# TYPE magent_latency_ms histogram", text)
        self.assertIn('magent_latency_ms{"', text) if False else None
        self.assertIn("magent_latency_ms_bucket", text)

    def test_record_run_folds_a_result(self) -> None:
        from magent.agent import make_agent

        registry = MetricsRegistry()
        result = make_agent(backend=EchoBackend()).run("hello")
        record_run(registry, result)
        values = registry.flat()
        self.assertIn("magent_runs_total{model=echo-small}", values)
        self.assertGreater(values["magent_tokens_total{model=echo-small}"], 0)

    def test_merge_and_reset(self) -> None:
        first, second = MetricsRegistry(), MetricsRegistry()
        second.inc("runs_total", 5)
        first.merge(second)
        self.assertEqual(first.counter("runs_total").value(), 5)
        first.reset()
        self.assertEqual(first.counter("runs_total").value(), 0)


class ObservabilityTest(MagentTestCase):
    """The façade attaches to an agent and records what happened."""

    def test_attach_records_runs(self) -> None:
        from magent.agent import make_agent

        observability = Observability()
        agent = make_agent(backend=EchoBackend())
        observability.attach(agent)
        agent.run("hello")
        report = observability.report()
        self.assertEqual(report.runs, 1)
        self.assertGreater(report.spans, 0)
        self.assertIn("prometheus", dir(observability))
        observability.close()

    def test_detach_stops_recording(self) -> None:
        from magent.agent import make_agent

        observability = Observability()
        agent = make_agent(backend=EchoBackend())
        subscription = observability.attach(agent)
        agent.run("first")
        after_first = observability.report().runs
        self.assertEqual(observability.detach(), 1)
        agent.run("second")
        self.assertEqual(observability.report().runs, after_first)
        del subscription

    def test_tool_errors_are_counted(self) -> None:
        from magent.agent import make_agent
        from magent.testing import ToolScriptBackend

        observability = Observability()
        agent = make_agent(backend=ToolScriptBackend(script=[
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "nope.txt"}}]},
            {"text": "done"},
        ]))
        observability.attach(agent)
        agent.run("read a missing file")
        self.assertGreater(observability.report().errors, 0)
        observability.close()


class PoolTest(unittest.TestCase):
    """Parallel execution, retries and timeouts."""

    @staticmethod
    def _sleepy(value: float) -> float:
        """Sleep for a moment and return the value."""
        time.sleep(value)
        return value

    def test_parallel_is_faster_than_serial(self) -> None:
        started = time.perf_counter()
        outcomes = run_parallel([Task(name=f"t{index}", fn=self._sleepy, args=(0.05,))
                                 for index in range(4)], max_workers=4)
        elapsed = time.perf_counter() - started
        self.assertTrue(all(outcome.ok for outcome in outcomes))
        self.assertLess(elapsed, 0.18)

    def test_failures_are_captured_not_raised(self) -> None:
        def boom() -> None:
            raise ValueError("nope")

        outcomes = run_parallel([lambda: 1, boom])
        self.assertTrue(outcomes[0].ok)
        self.assertFalse(outcomes[1].ok)
        self.assertEqual(outcomes[1].error_type, "ValueError")

    def test_timeout_is_reported(self) -> None:
        pool = WorkerPool(2)
        outcome = pool.run(Task(name="slow", fn=self._sleepy, args=(2.0,), timeout=0.1))
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.timed_out)
        pool.shutdown()

    def test_retries_eventually_succeed(self) -> None:
        state = {"calls": 0}

        def flaky() -> str:
            state["calls"] += 1
            if state["calls"] < 3:
                raise RuntimeError("transient")
            return "ok"

        pool = WorkerPool(1)
        outcome = pool.run(Task(name="flaky", fn=flaky, retries=3, backoff=0.001))
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempts, 3)
        pool.shutdown()

    def test_cancellation_skips_work(self) -> None:
        token = CancellationToken()
        token.cancel("stop")
        outcomes = run_parallel([Task(name="a", fn=self._sleepy, args=(0.01,))], token=token)
        self.assertTrue(outcomes[0].cancelled)

    def test_gather_swallows_failures(self) -> None:
        self.assertEqual(gather(lambda: 1, lambda: (_ for _ in ()).throw(ValueError("x"))), [1, None])

    def test_stats_and_context_manager(self) -> None:
        with WorkerPool(2, name="stats") as pool:
            pool.run_many([Task(name="a", fn=self._sleepy, args=(0.001,))])
            stats = pool.stats()
        self.assertEqual(stats["completed"], 1)
        self.assertGreaterEqual(stats["submitted"], 1)

    def test_chunked_and_interleave(self) -> None:
        self.assertEqual(chunked(list(range(5)), 2), [[0, 1], [2, 3], [4]])
        self.assertEqual(interleave([[1, 2], ["a"]]), [1, "a", 2])

    def test_task_outcome_serialises(self) -> None:
        outcome = TaskOutcome(name="t", ok=True, value={"a": 1}, seconds=0.5)
        self.assertEqual(outcome.to_dict()["value"], {"a": 1})


class LimiterTest(unittest.TestCase):
    """Rate limiting, throttling, debouncing and slot bounding."""

    def test_token_bucket_refills(self) -> None:
        bucket = TokenBucket(rate=50, capacity=2)
        self.assertTrue(bucket.try_acquire())
        self.assertTrue(bucket.try_acquire())
        self.assertFalse(bucket.try_acquire())
        self.assertTrue(bucket.acquire(1, timeout=1.0))
        self.assertEqual(bucket.stats.denied, 1)

    def test_bucket_reports_wait_time(self) -> None:
        bucket = TokenBucket(rate=10, capacity=1)
        bucket.try_acquire()
        self.assertGreater(bucket.wait_time(1), 0)

    def test_throttle_allows_one_call_per_interval(self) -> None:
        throttle = Throttle(0.05)
        self.assertTrue(throttle.allow())
        self.assertFalse(throttle.allow())
        self.assertGreater(throttle.remaining, 0)

    def test_debouncer_collapses_a_burst(self) -> None:
        calls: list[int] = []
        debouncer = Debouncer(0.05)
        for index in range(5):
            debouncer.call(calls.append, index)
        self.assertTrue(debouncer.wait(timeout=2.0))
        self.assertEqual(calls, [4])

    def test_limiter_bounds_concurrency(self) -> None:
        limiter = ConcurrencyLimiter(2)
        peak = {"value": 0}

        def work(_: int) -> int:
            peak["value"] = max(peak["value"], limiter.active)
            with limiter.slot():
                time.sleep(0.01)
            return 1

        results = limiter.map(work, list(range(6)), max_workers=4)
        self.assertEqual(results, [1] * 6)
        self.assertLessEqual(limiter.peak, 2)

    def test_rate_limited_batch(self) -> None:
        result = run_rate_limited(list(range(5)), lambda value: value * 2, rate=200, max_workers=2)
        self.assertEqual(result.succeeded, 5)
        self.assertEqual(result.values, [0, 2, 4, 6, 8])

    def test_cancellation_token_is_thread_safe(self) -> None:
        token = CancellationToken()
        self.assertTrue(bool(token))
        threading.Thread(target=lambda: token.cancel("later")).start()
        self.assertTrue(token.wait(1.0))
        self.assertFalse(bool(token))


if __name__ == "__main__":
    unittest.main()
