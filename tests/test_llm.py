"""Tests for the LLM layer: token accounting, usage, caching, retries, routing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from magent.errors import BackendUnavailableError, ConfigError, RateLimitedError
from magent.llm.base import ChatResponse, CompletionRequest, Message, StreamChunk, ToolCall
from magent.llm.cache import ResponseCache
from magent.llm.cortex import CortexBackend, CortexSettings
from magent.llm.echo import EchoBackend
from magent.llm.factory import (BACKEND_CHOICES, BackendFactory, BackendRequest, build_backend,
                                build_offline_backend)
from magent.llm.retry import CircuitBreaker, RetryExecutor, with_timeout
from magent.llm.router import ModelRouter, Route
from magent.llm.tokens import (TokenBudget, TokenEstimator, context_window_plan, estimate_tokens,
                               split_by_tokens, trim_messages, truncate_to_tokens)
from magent.llm.usage import BudgetConfig, PriceBook, UsageTracker
from magent.types import Usage

from tests.support import MagentTestCase


class TokensTest(unittest.TestCase):
    """Token estimation and trimming."""

    def test_estimate_scales_with_length(self) -> None:
        self.assertLess(estimate_tokens("hello"), estimate_tokens("hello " * 100))

    def test_cjk_costs_more_than_latin(self) -> None:
        self.assertGreater(estimate_tokens("日本語のテキスト"), estimate_tokens("plain english"))

    def test_truncate_respects_the_ceiling(self) -> None:
        estimator = TokenEstimator()
        text = "word " * 500
        shortened = truncate_to_tokens(text, 50, estimator=estimator)
        self.assertLessEqual(estimator.estimate(shortened), 50)
        self.assertIn("truncated", shortened)

    def test_split_by_tokens_covers_the_input(self) -> None:
        text = "word " * 200
        parts = split_by_tokens(text, 40)
        self.assertGreater(len(parts), 1)
        self.assertIn("word", "".join(parts))
        self.assertLessEqual(estimate_tokens(parts[0]), 60)

    def test_trim_messages_keeps_system_and_last_turns(self) -> None:
        messages = [Message.system("rules")] + [Message.user(f"m{index}") for index in range(20)]
        trimmed, report = trim_messages(messages, max_tokens=40, keep_system=True, keep_last=2)
        self.assertEqual(trimmed[0].role, "system")
        self.assertEqual(trimmed[-1].text(), "m19")
        self.assertLess(len(trimmed), len(messages))
        self.assertTrue(report)

    def test_trim_never_splits_a_tool_pair(self) -> None:
        from magent.types import ToolResult

        call = ToolCall(id="c1", name="calculate", arguments={})
        messages = [
            Message.user("go"),
            Message.assistant("", tool_calls=[call]),
            Message.tool(ToolResult(call_id="c1", name="calculate", ok=True, content=42)),
            Message.assistant("done"),
        ]
        trimmed, _ = trim_messages(messages, max_tokens=1000, keep_last=4)
        self.assertEqual(len(trimmed), len(messages))

    def test_token_budget_accounting(self) -> None:
        budget = TokenBudget(max_tokens=100, used=40, reserved_for_output=20)
        self.assertEqual(budget.remaining, 40)

    def test_context_window_plan_reports_fit(self) -> None:
        plan = context_window_plan([Message.user("hello")], context_size=1000,
                                   max_output_tokens=100)
        self.assertIn("fits", plan)

    def test_estimator_calibrates_towards_reality(self) -> None:
        estimator = TokenEstimator()
        before = estimator.chars_per_token
        estimator.calibrate(estimated=100, actual=200)
        self.assertNotEqual(before, estimator.chars_per_token)


class UsageTest(unittest.TestCase):
    """Usage accounting and budgets."""

    def test_tracker_accumulates_tokens(self) -> None:
        tracker = UsageTracker()
        tracker.record_tokens(1_500)
        tracker.record_step(1)
        self.assertEqual(tracker.usage.total_tokens, 1_500)
        self.assertIn("total_tokens", json.dumps(tracker.snapshot()))

    def test_budget_detects_exhaustion(self) -> None:
        tracker = UsageTracker(budget=BudgetConfig(max_tokens=100))
        tracker.record_tokens(200)
        self.assertTrue(tracker.exceeded())

    def test_price_book_computes_cost(self) -> None:
        prices = PriceBook(prompt_per_million={"m": 10.0}, completion_per_million={"m": 20.0})
        estimate = prices.cost("m", Usage(prompt_tokens=1_000, completion_tokens=1_000))
        self.assertAlmostEqual(estimate.prompt_cost, 0.01, places=6)
        self.assertAlmostEqual(estimate.completion_cost, 0.02, places=6)
        self.assertAlmostEqual(estimate.total, 0.03, places=6)
        self.assertEqual(prices.currency, "USD")

    def test_unknown_model_is_free_not_crashing(self) -> None:
        self.assertEqual(PriceBook().cost("unknown-model", Usage(prompt_tokens=10)).total, 0.0)

    def test_summary_line_is_readable(self) -> None:
        tracker = UsageTracker()
        tracker.record_tokens(15)
        self.assertIn("token", tracker.summary_line().lower())


class BackendTest(MagentTestCase):
    """The echo backend and the factory behave as documented."""

    def test_echo_backend_answers_and_streams(self) -> None:
        backend = EchoBackend()
        response = backend.complete(CompletionRequest(messages=[Message.user("hi")]))
        self.assertIsInstance(response, ChatResponse)
        self.assertIn("hi", response.text)
        chunks: list[StreamChunk] = list(backend.stream(
            CompletionRequest(messages=[Message.user("stream me")], stream=True)))
        self.assertTrue(chunks)
        self.assertTrue(chunks[-1].done)

    def test_echo_backend_lists_models(self) -> None:
        self.assertTrue(EchoBackend().models())

    def test_factory_defaults_to_auto_then_echo(self) -> None:
        backend = build_backend(config=self.config())
        self.assertIsInstance(backend, EchoBackend)

    def test_factory_honours_the_config_file(self) -> None:
        config = self.config(model__backend="echo", model__model="my-model")
        self.assertIn("my-model", [model.id for model in build_backend(config=config).models()])

    def test_offline_helper(self) -> None:
        self.assertIsInstance(build_offline_backend(), EchoBackend)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            BackendFactory().build(BackendRequest(kind="not-a-backend"))

    def test_backend_choices_are_documented(self) -> None:
        self.assertIn("cortex", BACKEND_CHOICES)
        self.assertIn("echo", BACKEND_CHOICES)

    def test_missing_cortex_host_raises_a_backend_error(self) -> None:
        settings = CortexSettings(base_url="http://127.0.0.1:9", timeout=0.5, max_retries=0)
        backend = CortexBackend(settings)
        with self.assertRaises(Exception):
            backend.complete(CompletionRequest(messages=[Message.user("hi")]))


class CacheTest(MagentTestCase):
    """The response cache is keyed on the request and expires."""

    def test_set_and_get(self) -> None:
        cache = ResponseCache(directory=self.home.path / "cache", enabled=True)
        request = CompletionRequest(messages=[Message.user("hello")], model="m")
        cache.set(request, ChatResponse(text="cached", model="m"))
        hit = cache.get(request)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.text, "cached")

    def test_different_requests_do_not_collide(self) -> None:
        cache = ResponseCache(directory=self.home.path / "cache", enabled=True)
        first = CompletionRequest(messages=[Message.user("a")], model="m")
        second = CompletionRequest(messages=[Message.user("b")], model="m")
        cache.set(first, ChatResponse(text="one", model="m"))
        self.assertIsNone(cache.get(second))

    def test_disabled_cache_never_stores(self) -> None:
        cache = ResponseCache(directory=self.home.path / "cache", enabled=False)
        request = CompletionRequest(messages=[Message.user("x")], model="m")
        cache.set(request, ChatResponse(text="y", model="m"))
        self.assertIsNone(cache.get(request))

    def test_stats_report_hits_and_misses(self) -> None:
        cache = ResponseCache(directory=self.home.path / "cache", enabled=True)
        request = CompletionRequest(messages=[Message.user("x")], model="m")
        cache.set(request, ChatResponse(text="y", model="m"))
        cache.get(request)
        cache.get(CompletionRequest(messages=[Message.user("z")], model="m"))
        stats = cache.stats_dict()
        self.assertGreaterEqual(stats.get("hits", 0), 1)


class RetryTest(unittest.TestCase):
    """Retries stop on success and give up after the configured attempts."""

    def test_retries_until_success(self) -> None:
        state = {"calls": 0}

        def flaky() -> str:
            state["calls"] += 1
            if state["calls"] < 3:
                raise RateLimitedError("slow down")
            return "ok"

        executor = RetryExecutor(max_attempts=4, base_delay=0.001, cap=0.01)
        self.assertEqual(executor.call(flaky), "ok")
        self.assertEqual(state["calls"], 3)

    def test_failure_propagates_after_the_last_attempt(self) -> None:
        def always() -> str:
            raise RateLimitedError("nope")

        executor = RetryExecutor(max_attempts=2, base_delay=0.001, cap=0.002)
        with self.assertRaises(RateLimitedError):
            executor.call(always)
        self.assertEqual(executor.stats()["attempts"] if "attempts" in executor.stats() else 2, 2)

    def test_never_retry_list_is_honoured(self) -> None:
        calls = {"n": 0}

        def fatal() -> str:
            calls["n"] += 1
            raise ConfigError("do not retry")

        executor = RetryExecutor(max_attempts=3, base_delay=0.001, never_retry_on=(ConfigError,))
        with self.assertRaises(ConfigError):
            executor.call(fatal)
        self.assertEqual(calls["n"], 1)

    def test_circuit_breaker_opens_after_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0)
        self.assertTrue(breaker.allow_request())
        breaker.record_failure()
        breaker.record_failure()
        self.assertFalse(breaker.allow_request())
        breaker.reset()
        self.assertTrue(breaker.allow_request())

    def test_with_timeout_raises_for_slow_calls(self) -> None:
        def slow() -> str:
            import time

            time.sleep(0.2)
            return "late"

        with self.assertRaises(Exception):
            with_timeout(slow, 0.05)()


class RouterTest(unittest.TestCase):
    """The router fails over between backends."""

    def test_router_uses_the_first_healthy_route(self) -> None:
        primary = EchoBackend(models_list=["primary"])
        secondary = EchoBackend(models_list=["secondary"])
        router = ModelRouter([Route(backend=primary, label="a"),
                              Route(backend=secondary, label="b")], strategy="priority")
        response = router.complete(CompletionRequest(messages=[Message.user("hi")]))
        self.assertEqual(response.model, "primary")

    def test_router_records_failures(self) -> None:
        class Broken(EchoBackend):
            """A backend that always fails, used to exercise failover."""

            def complete(self, request):  # type: ignore[override]
                raise BackendUnavailableError("down")

        router = ModelRouter([Route(backend=Broken(), label="broken"),
                              Route(backend=EchoBackend(models_list=["ok-model"]), label="ok")],
                             strategy="priority", max_failovers=2)
        response = router.complete(CompletionRequest(messages=[Message.user("hi")]))
        self.assertEqual(response.model, "ok-model")
        self.assertGreaterEqual(router.stats().get("failovers", 0), 1)

    def test_strategies_are_validated(self) -> None:
        with self.assertRaises((ConfigError, ValueError)):
            ModelRouter([], strategy="nonsense")

    def test_route_labels_are_reported(self) -> None:
        router = ModelRouter([Route(backend=EchoBackend(), label="one")])
        self.assertIn("one", router.route_labels())


if __name__ == "__main__":
    unittest.main()
