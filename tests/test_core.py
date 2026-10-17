"""Tests for configuration, secrets, logging, events and the shared types."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from magent.config import (AgentConfig, Config, config_from_dict, default_config, load_config,
                           load_toml, write_example_config)
from magent.errors import ConfigError
from magent.events import EventBus, Subscription
from magent.logging_ import LogCapture, get_logger, setup_logging
from magent.secrets import (SecretStore, contains_secret, generate_api_key, is_secret_reference,
                            redact_text, resolve_secret)
from magent.types import AgentEvent, EventKind, Message, Role, StopReason, ToolResult, Usage

from tests.support import MagentTestCase


class ConfigTest(MagentTestCase):
    """Configuration loading, precedence and validation."""

    def test_defaults_are_valid(self) -> None:
        config = default_config()
        config.validate()
        self.assertEqual(config.model.backend, "auto")
        self.assertTrue(config.memory.enabled)

    def test_layering_prefers_env_over_file(self) -> None:
        path = self.write("magent.toml", "[agent]\nmax_iterations = 3\n")
        os.environ["MAGENT_MAX_ITERATIONS"] = "9"
        try:
            config = load_config(path=path)
        finally:
            os.environ.pop("MAGENT_MAX_ITERATIONS", None)
        self.assertEqual(config.agent.max_iterations, 9)

    def test_cli_overrides_win(self) -> None:
        config = load_config(overrides={"agent.max_iterations": 2, "output.format": "json"})
        self.assertEqual(config.agent.max_iterations, 2)
        self.assertEqual(config.output.format, "json")

    def test_unknown_profile_is_reported(self) -> None:
        path = self.write("magent.toml",
                          '[agent]\nmax_iterations = 3\n\n[profiles.fast.agent]\n'
                          'max_iterations = 1\n')
        with self.assertRaises(ConfigError) as caught:
            load_config(path=path, profile="does-not-exist")
        self.assertIn("fast", str(caught.exception))

    def test_profiles_deep_merge(self) -> None:
        path = self.write("magent.toml",
                          '[agent]\nmax_iterations = 4\n\n'
                          '[profiles.fast.agent]\nmax_iterations = 1\n')
        config = load_config(path=path, profile="fast")
        self.assertEqual(config.agent.max_iterations, 1)

    def test_to_dict_redacts_secrets(self) -> None:
        config = default_config()
        config.server.api_key = "sk-super-secret-value"
        payload = config.to_dict(redact=True)
        self.assertNotIn("sk-super-secret-value", json.dumps(payload))
        self.assertIn("sk-super-secret-value", json.dumps(config.to_dict(redact=False)))

    def test_invalid_values_are_rejected(self) -> None:
        config = default_config()
        config.agent.max_iterations = 0
        with self.assertRaises(ConfigError):
            config.validate()

    def test_write_example_config_round_trips(self) -> None:
        target = self.home.path / "magent.toml"
        write_example_config(target, force=True)
        data = load_toml(target)
        self.assertIsInstance(data, dict)
        self.assertTrue(data)

    def test_config_from_dict_ignores_unknown_sections(self) -> None:
        config = config_from_dict({"agent": {"max_iterations": 5}, "nope": {"x": 1}})
        self.assertEqual(config.agent.max_iterations, 5)

    def test_agent_config_validation_accepts_defaults(self) -> None:
        AgentConfig().validate()


class SecretsTest(MagentTestCase):
    """Secret references, redaction and masking."""

    def test_reference_detection(self) -> None:
        self.assertTrue(is_secret_reference("env:MY_TOKEN"))
        self.assertTrue(is_secret_reference("file:~/.secrets/token"))
        self.assertFalse(is_secret_reference("plain-value"))

    def test_redact_text_masks_known_shapes(self) -> None:
        redacted = redact_text("key sk-abcdefghijklmnopqrstuvwxyz012345 and email a@b.com")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertIn("REDACTED", redacted)

    def test_contains_secret_finds_known_shapes(self) -> None:
        found = contains_secret("token ghp_" + "a" * 36, ["ghp_" + "a" * 36])
        self.assertEqual(len(found), 1)
        self.assertEqual(contains_secret("nothing to see here", ["sk-abc"]), [])

    def test_resolve_secret_from_environment(self) -> None:
        os.environ["MAGENT_TEST_SECRET"] = "value-123"
        try:
            self.assertEqual(resolve_secret("env:MAGENT_TEST_SECRET"), "value-123")
        finally:
            os.environ.pop("MAGENT_TEST_SECRET", None)

    def test_resolve_secret_uses_default_when_missing(self) -> None:
        self.assertEqual(resolve_secret("env:MAGENT_DEFINITELY_MISSING", default="fallback"),
                         "fallback")

    def test_secret_store_round_trip(self) -> None:
        path = self.home.path / "secrets.toml"
        store = SecretStore(path)
        store.set("cortex", "sk-test-1234567890")
        store.save()
        reloaded = SecretStore(path)
        self.assertEqual(reloaded.get("cortex"), "sk-test-1234567890")
        self.assertIn("cortex", reloaded.keys())
        self.assertTrue(reloaded.as_references()["cortex"].startswith("file:"))

    def test_secret_file_is_owner_only(self) -> None:
        path = self.home.path / "secrets.toml"
        store = SecretStore(path)
        store.set("token", "value")
        store.save()
        self.assertEqual(path.stat().st_mode & 0o077, 0)


class EventsTest(unittest.TestCase):
    """The event bus delivers to subscribers and keeps history."""

    def test_publish_and_history(self) -> None:
        bus = EventBus()
        bus.emit(EventKind.RUN_STARTED, run_id="r1")
        bus.emit(EventKind.RUN_FINISHED, run_id="r1")
        history = bus.history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].kind, EventKind.RUN_STARTED)

    def test_subscribe_receives_and_records(self) -> None:
        bus = EventBus()
        seen: list[EventKind] = []
        subscription = bus.subscribe(lambda event: seen.append(event.kind),
                                     kinds=[EventKind.TOOL_CALL], name="collector")
        self.assertIsInstance(subscription, Subscription)
        bus.emit(EventKind.TOOL_CALL, name="calculate")
        bus.emit(EventKind.STEP_STARTED)
        self.assertEqual(seen, [EventKind.TOOL_CALL])
        self.assertEqual(subscription.delivered, 1)

    def test_handler_errors_are_isolated(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        def broken(event: AgentEvent) -> None:
            raise RuntimeError("subscriber bug")

        bus.subscribe(broken, name="broken")
        bus.subscribe(lambda event: seen.append("ok"), name="good")
        bus.emit(EventKind.RUN_STARTED)
        self.assertEqual(seen, ["ok"])

    def test_wait_for_returns_the_matching_event(self) -> None:
        bus = EventBus()
        bus.subscribe(lambda event: None, name="noop")

        def emitter() -> None:
            bus.emit(EventKind.RUN_FINISHED, run_id="r9")

        import threading

        threading.Thread(target=emitter).start()
        event = bus.wait_for(lambda item: item.run_id == "r9", timeout=2.0)
        self.assertIsNotNone(event)

    def test_counts_and_stats(self) -> None:
        bus = EventBus()
        bus.emit(EventKind.RUN_STARTED)
        bus.emit(EventKind.RUN_STARTED)
        self.assertEqual(bus.counts()["run.started"], 2)
        self.assertIn("subscriptions", bus.stats())


class LoggingTest(unittest.TestCase):
    """Logging setup, capture and redaction."""

    def test_setup_logging_is_idempotent(self) -> None:
        first = setup_logging(level="ERROR", force=True)
        second = setup_logging(level="ERROR")
        self.assertIs(first, second)

    def test_log_capture_collects_records(self) -> None:
        with LogCapture(level="DEBUG") as capture:
            get_logger("magent.test").warning("hello capture")
        self.assertTrue(capture.messages())
        self.assertTrue(capture.contains("hello capture"))

    def test_log_level_filtering(self) -> None:
        with LogCapture(level="ERROR") as capture:
            get_logger("magent.test").info("hidden")
            get_logger("magent.test").error("visible")
        self.assertTrue(capture.contains("visible"))
        self.assertFalse(capture.contains("hidden"))

    def test_generate_api_key_is_unique(self) -> None:
        first, second = generate_api_key(), generate_api_key()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("mag"))


class TypesTest(unittest.TestCase):
    """Shared value objects serialise and behave predictably."""

    def test_message_text_is_a_method(self) -> None:
        message = Message(role=Role.USER, content="hello")
        self.assertEqual(message.text(), "hello")

    def test_message_factories(self) -> None:
        self.assertEqual(Message.user("x").role, "user")
        self.assertEqual(Message.assistant("y").role, "assistant")
        self.assertEqual(Message.system("z").role, "system")

    def test_usage_merge_and_totals(self) -> None:
        merged = Usage(prompt_tokens=10, completion_tokens=5).merge(
            Usage(prompt_tokens=1, completion_tokens=2))
        self.assertEqual(merged.prompt_tokens, 11)
        self.assertEqual(merged.total_tokens, 18)

    def test_tool_result_round_trip(self) -> None:
        result = ToolResult(call_id="c1", name="calculate", ok=True, content={"result": 42})
        self.assertTrue(result.ok)
        self.assertEqual(result.content["result"], 42)

    def test_stop_reason_success_flags(self) -> None:
        self.assertTrue(StopReason.COMPLETED.is_success())
        self.assertFalse(StopReason.ERROR.is_success())

    def test_messages_from_dicts(self) -> None:
        messages = Message.from_dict({"role": "user", "content": "hi"})
        self.assertEqual(messages.text(), "hi")

    def test_agent_event_to_dict(self) -> None:
        event = AgentEvent(kind=EventKind.RUN_STARTED, data={"a": 1}, run_id="r1")
        self.assertEqual(event.kind, EventKind.RUN_STARTED)
        self.assertEqual(event.run_id, "r1")


if __name__ == "__main__":
    unittest.main()
