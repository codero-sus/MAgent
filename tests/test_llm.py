"""Offline tests for backend contracts, token accounting, and routing.

The Cortex-specific wire contract is exercised in ``test_cortex.py``. This file
checks the backend abstraction and all local policies with deterministic echo or
scripted backends: request serialization, streaming assembly, token estimation,
budget arithmetic, circuit breaking, and failover routing.
"""

from __future__ import annotations

import unittest

from magent.errors import ServerError
from magent.llm import (
    STRATEGIES,
    BreakerState,
    CircuitBreaker,
    CompletionRequest,
    EchoBackend,
    ModelRouter,
    Route,
    TokenBudget,
    TokenEstimator,
    collect_stream,
    context_window_plan,
    estimate_message_chars,
    estimate_tokens,
    split_by_tokens,
    trim_messages,
    truncate_to_tokens,
)
from magent.testing import ScriptedBackend, make_backend
from magent.types import ChatResponse, Message, StopReason, ToolCall, Usage


class CompletionRequestTests(unittest.TestCase):
    """CompletionRequest is the backend-neutral OpenAI payload builder."""

    def test_payload_serializes_messages(self) -> None:
        """Role and content use Cortex's OpenAI-compatible shape."""
        request = CompletionRequest(messages=[Message.system("rules"), Message.user("hello")],
                                    model="model-a", temperature=0.2, max_tokens=100)
        payload = request.to_payload()
        self.assertEqual(payload["model"], "model-a")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "rules"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 100)

    def test_payload_omits_unset_optional_fields(self) -> None:
        """None-valued controls are not sent as JSON null."""
        payload = CompletionRequest(messages=[Message.user("x")]).to_payload()
        self.assertNotIn("temperature", payload)
        self.assertNotIn("stop", payload)
        self.assertNotIn("tools", payload)

    def test_payload_can_suppress_tools(self) -> None:
        """A caller can reuse a request for a backend without tool support."""
        request = CompletionRequest(messages=[Message.user("x")], tools=[{
            "type": "function", "function": {"name": "demo", "parameters": {}}}])
        self.assertIn("tools", request.to_payload())
        self.assertNotIn("tools", request.to_payload(include_tools=False))

    def test_clone_changes_only_selected_fields(self) -> None:
        """clone supports retry or routing adjustments without mutation."""
        source = CompletionRequest(messages=[Message.user("x")], model="a", temperature=0.3)
        clone = source.clone(model="b")
        self.assertEqual(source.model, "a")
        self.assertEqual(clone.model, "b")
        self.assertEqual(clone.temperature, 0.3)

    def test_signature_ignores_message_ids(self) -> None:
        """Equivalent requests share a cache signature."""
        left = CompletionRequest(messages=[Message.user("same")], model="m")
        right = CompletionRequest(messages=[Message.user("same")], model="m")
        self.assertEqual(left.signature(), right.signature())


class EchoBackendTests(unittest.TestCase):
    """The bundled backend makes every feature runnable without Cortex."""

    def test_complete_echoes_last_user_message(self) -> None:
        """EchoBackend produces a full ChatResponse."""
        backend = EchoBackend(prefix="answer:")
        response = backend.complete(CompletionRequest(messages=[Message.user("hello")]))
        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.text, "answer:hello")
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreater(response.usage.total_tokens, 0)

    def test_complete_uses_last_user_not_system(self) -> None:
        """System instructions are not echoed as the answer."""
        backend = EchoBackend()
        response = backend.complete(CompletionRequest(messages=[
            Message.user("first"), Message.assistant("middle"), Message.user("last")]))
        self.assertEqual(response.text, "echo:last")

    def test_models_reports_one_running_chat_model(self) -> None:
        """Offline mode still presents a real model catalogue."""
        model = EchoBackend().models()[0]
        self.assertEqual(model.id, "echo-small")
        self.assertTrue(model.default)
        self.assertTrue(model.supports("chat"))

    def test_health_marks_backend_offline_but_healthy(self) -> None:
        """Offline means no network, not broken."""
        health = EchoBackend().health()
        self.assertTrue(health["ok"])
        self.assertTrue(health["offline"])

    def test_stream_reassembles_to_complete_text(self) -> None:
        """collect_stream joins text and keeps terminal usage."""
        backend = EchoBackend(prefix="", chunk_size=3)
        request = CompletionRequest(messages=[Message.user("abcdefgh")])
        response = collect_stream(backend.stream(request))
        self.assertEqual(response.text, "abcdefgh")
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreater(response.usage.total_tokens, 0)

    def test_stream_callback_sees_each_chunk(self) -> None:
        """A renderer receives deltas as they arrive."""
        backend = EchoBackend(prefix="", chunk_size=2)
        seen: list[str] = []
        response = collect_stream(
            backend.stream(CompletionRequest(messages=[Message.user("abcde")])),
            on_chunk=lambda chunk: seen.append(chunk.text),
        )
        self.assertEqual(response.text, "abcde")
        self.assertGreaterEqual(len(seen), 3)

    def test_injected_failure_is_bounded(self) -> None:
        """fail_times makes exactly that many calls fail."""
        backend = EchoBackend(fail_times=1)
        request = CompletionRequest(messages=[Message.user("x")])
        with self.assertRaises(ServerError):
            backend.complete(request)
        self.assertEqual(backend.complete(request).text, "echo:x")

    def test_stats_accumulate_usage(self) -> None:
        """Backend telemetry tracks calls and token usage."""
        backend = EchoBackend()
        backend.complete(CompletionRequest(messages=[Message.user("one")]))
        backend.complete(CompletionRequest(messages=[Message.user("two")]))
        stats = backend.stats()
        self.assertEqual(stats["calls"], 2)
        self.assertGreater(stats["usage"]["total_tokens"], 0)


class ScriptedBackendTests(unittest.TestCase):
    """Scripted doubles make multi-turn agent behavior deterministic."""

    def test_script_returns_items_in_order(self) -> None:
        """Each completion advances the cursor."""
        backend = ScriptedBackend(script=["first", "second"])
        request = CompletionRequest(messages=[Message.user("x")])
        self.assertEqual(backend.complete(request).text, "first")
        self.assertEqual(backend.complete(request).text, "second")

    def test_script_repeats_last_item_when_exhausted(self) -> None:
        """A too-long loop gets deterministic output, not IndexError."""
        backend = ScriptedBackend(script=["only"])
        request = CompletionRequest(messages=[Message.user("x")])
        backend.complete(request)
        self.assertEqual(backend.complete(request).text, "only")

    def test_tool_backend_emits_call_then_final_text(self) -> None:
        """The dedicated fixture implements a complete two-turn tool loop."""
        backend = make_backend(tool_name="calculate",
                               tool_arguments={"expression": "6*7"}, final_text="42")
        first = backend.complete(CompletionRequest(messages=[Message.user("calculate")]))
        self.assertTrue(first.has_tool_calls)
        self.assertEqual(first.tool_calls[0].name, "calculate")
        second = backend.complete(CompletionRequest(messages=[Message.user("again")]))
        self.assertEqual(second.text, "42")


class TokenTests(unittest.TestCase):
    """Token estimates are bounded, monotonic, and useful without a tokenizer."""

    def test_empty_text_costs_no_tokens(self) -> None:
        """The estimator does not invent a token for an empty field."""
        self.assertEqual(estimate_tokens(""), 0)

    def test_estimate_is_monotonic(self) -> None:
        """Appending substantial text never lowers the estimate."""
        self.assertLess(estimate_tokens("hello"), estimate_tokens("hello " * 20))

    def test_cjk_uses_more_tokens_than_same_character_count_latin(self) -> None:
        """The heuristic accounts for dense CJK tokenization."""
        self.assertGreater(estimate_tokens("人工智能模型"), estimate_tokens("abcdef"))

    def test_message_char_count(self) -> None:
        """Message overhead plus bodies is represented."""
        messages = [Message.system("abc"), Message.user("defgh")]
        self.assertGreaterEqual(estimate_message_chars(messages), 8)

    def test_estimator_calibrates_with_observed_usage(self) -> None:
        """Calibration updates the ratio but keeps it within safe bounds."""
        estimator = TokenEstimator(chars_per_token=4.0)
        estimator.calibrate(estimated=25, actual=50)
        self.assertGreater(estimator.samples, 0)
        self.assertGreaterEqual(estimator.chars_per_token, estimator.min_chars_per_token)
        self.assertLessEqual(estimator.chars_per_token, estimator.max_chars_per_token)

    def test_truncate_respects_budget(self) -> None:
        """Truncating a long string returns a shorter marked string."""
        source = "word " * 500
        truncated = truncate_to_tokens(source, 20)
        self.assertLess(len(truncated), len(source))
        self.assertIn("truncated", truncated)

    def test_short_text_is_not_changed(self) -> None:
        """No marker is added when text already fits."""
        self.assertEqual(truncate_to_tokens("short", 100), "short")

    def test_split_reassembles_words(self) -> None:
        """Token chunks retain all source words in order."""
        source = " ".join(f"word{n}" for n in range(100))
        chunks = split_by_tokens(source, 20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(" ".join(chunks).split()), source)

    def test_trim_keeps_system_and_latest_messages(self) -> None:
        """Critical system rules and recent context survive a small budget."""
        messages = [Message.system("rules")] + [Message.user("x" * 100) for _ in range(10)]
        trimmed, report = trim_messages(messages, max_tokens=80, keep_last=2)
        self.assertEqual(trimmed[0].role.value, "system")
        self.assertEqual(trimmed[-1].text(), "x" * 100)
        self.assertGreater(report["dropped"], 0)

    def test_context_window_plan_reserves_output(self) -> None:
        """The plan reports prompt and output allocations separately."""
        plan = context_window_plan([Message.user("hello")], context_size=4096,
                                   max_output_tokens=512)
        self.assertEqual(plan["context_size"], 4096)
        self.assertEqual(plan["max_output_tokens"], 512)
        self.assertGreater(plan["available_input_tokens"], 0)


class TokenBudgetTests(unittest.TestCase):
    """The small mutable budget object guards cumulative use."""

    def test_remaining_and_consumption(self) -> None:
        """consume increments used and decreases remaining."""
        budget = TokenBudget(max_tokens=100, reserved_for_output=20)
        self.assertEqual(budget.remaining, 80)
        budget.consume(30)
        self.assertEqual(budget.used, 30)
        self.assertEqual(budget.remaining, 50)

    def test_unlimited_budget(self) -> None:
        """A zero maximum is explicitly unlimited."""
        budget = TokenBudget()
        self.assertTrue(budget.is_unlimited)
        self.assertFalse(budget.would_exceed(10**9))

    def test_reset(self) -> None:
        """reset clears accumulated usage but keeps the configured cap."""
        budget = TokenBudget(max_tokens=100)
        budget.consume(50)
        budget.reset()
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.max_tokens, 100)


class CircuitBreakerTests(unittest.TestCase):
    """Repeated backend failures open a route and timed probes recover it."""

    def test_failure_threshold_opens_breaker(self) -> None:
        """The configured consecutive failure count trips the breaker."""
        breaker = CircuitBreaker(failure_threshold=2, min_samples=100)
        breaker.record_failure()
        self.assertEqual(breaker.state, BreakerState.CLOSED)
        breaker.record_failure()
        self.assertEqual(breaker.state, BreakerState.OPEN)
        self.assertFalse(breaker.allow_request())

    def test_timeout_moves_open_to_half_open(self) -> None:
        """After reset_timeout, one probe is allowed."""
        now = [0.0]
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10,
                                 min_samples=100, clock=lambda: now[0])
        breaker.record_failure()
        now[0] = 11.0
        self.assertEqual(breaker.state, BreakerState.HALF_OPEN)
        self.assertTrue(breaker.allow_request())

    def test_successful_probe_closes_breaker(self) -> None:
        """One successful half-open call restores the route."""
        now = [0.0]
        breaker = CircuitBreaker(failure_threshold=1, reset_timeout=1,
                                 min_samples=100, clock=lambda: now[0])
        breaker.record_failure()
        now[0] = 2.0
        self.assertTrue(breaker.allow_request())
        breaker.record_success()
        self.assertEqual(breaker.state, BreakerState.CLOSED)

    def test_reset_clears_telemetry_state(self) -> None:
        """An operator can explicitly close and reset a breaker."""
        breaker = CircuitBreaker(failure_threshold=1, min_samples=100)
        breaker.record_failure()
        breaker.reset()
        self.assertEqual(breaker.state, BreakerState.CLOSED)
        self.assertEqual(breaker.snapshot()["window_samples"], 0)


class RouterTests(unittest.TestCase):
    """ModelRouter chooses a route and fails over without changing the API."""

    def request(self, text: str = "hello") -> CompletionRequest:
        return CompletionRequest(messages=[Message.user(text)])

    def test_every_documented_strategy_constructs(self) -> None:
        """No advertised routing strategy is dead configuration."""
        for strategy in STRATEGIES:
            router = ModelRouter([Route(EchoBackend())], strategy=strategy)
            self.assertEqual(router.strategy, strategy)
            router.close()

    def test_priority_chooses_lowest_number(self) -> None:
        """Priority routing starts with the numerically first route."""
        router = ModelRouter([
            Route(EchoBackend(prefix="slow:"), priority=20, label="later"),
            Route(EchoBackend(prefix="fast:"), priority=1, label="first"),
        ], strategy="priority")
        self.assertEqual(router.complete(self.request()).text, "fast:hello")
        router.close()

    def test_failover_uses_next_healthy_route(self) -> None:
        """A transient backend failure is hidden by the fallback route."""
        failed = EchoBackend(name="failed", fail_times=10)
        healthy = EchoBackend(name="healthy", prefix="fallback:")
        attempts: list[object] = []
        router = ModelRouter([
            Route(failed, priority=1, label="failed"),
            Route(healthy, priority=2, label="healthy"),
        ], strategy="priority", breaker_threshold=1, on_failover=attempts.append)
        response = router.complete(self.request())
        self.assertEqual(response.text, "fallback:hello")
        self.assertEqual(router.stats()["failovers"], 1)
        self.assertTrue(attempts)
        self.assertEqual(router.breaker_for("failed").state, BreakerState.OPEN)
        router.close()

    def test_disabled_route_is_not_selected(self) -> None:
        """Operator-disabled routes are removed from candidates."""
        router = ModelRouter([
            Route(EchoBackend(prefix="disabled:"), priority=1, enabled=False, label="off"),
            Route(EchoBackend(prefix="on:"), priority=2, label="on"),
        ])
        self.assertEqual(router.complete(self.request()).text, "on:hello")
        router.close()

    def test_round_robin_rotates_routes(self) -> None:
        """Consecutive calls use each route in turn."""
        router = ModelRouter([
            Route(EchoBackend(prefix="a:"), label="a"),
            Route(EchoBackend(prefix="b:"), label="b"),
        ], strategy="round_robin")
        texts = [router.complete(self.request()).text for _ in range(4)]
        self.assertEqual(texts, ["a:hello", "b:hello", "a:hello", "b:hello"])
        router.close()

    def test_capability_filter(self) -> None:
        """A requested capability removes incompatible routes."""
        router = ModelRouter([
            Route(EchoBackend(prefix="chat:"), capabilities=["chat"], label="chat"),
            Route(EchoBackend(prefix="vision:"), capabilities=["chat", "vision"],
                  label="vision"),
        ], strategy="priority")
        candidates = router.candidates(capability="vision")
        self.assertEqual([route.label for route in candidates], ["vision"])
        router.close()


if __name__ == "__main__":
    unittest.main()
