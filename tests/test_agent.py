"""Integration tests for MAgent's synchronous agent loop.

All model turns are supplied by deterministic local backends. The tests cover a
plain answer, a complete tool-call round trip, tool failures, max-iteration and
cancellation stops, event ordering, structured-output parsing and repair, stream
semantics, per-run options, and cumulative telemetry.
"""

from __future__ import annotations

import unittest

from magent.agent import Agent, RunOptions
from magent.errors import AgentError
from magent.events import EventBus
from magent.llm import CompletionRequest, EchoBackend, ScriptedBackend
from magent.testing import make_backend
from magent.tools import default_registry
from magent.types import EventKind, Message, Role, StopReason

from tests.support import MAgentTestCase


class AgentTestCase(MAgentTestCase):
    """Construct agents in a temporary XDG home and workspace."""

    def agent(self, backend: object | None = None, **flat: object) -> Agent:
        """Return a fully configured local agent."""
        config = self.config(flat={f"agent.{name}": value for name, value in flat.items()})
        return Agent(backend=backend or EchoBackend(), config=config.agent, settings=config,
                     registry=default_registry(), workspace=self._home)


class PlainRunTests(AgentTestCase):
    """A one-turn text answer produces a complete, inspectable RunResult."""

    def test_echo_run_completes(self) -> None:
        """The minimal offline run succeeds in one iteration."""
        result = self.agent().run("hello")
        self.assertEqual(result.output, "echo:hello")
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_call_count, 0)
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertGreater(result.duration_ms, 0)

    def test_conversation_contains_system_user_and_assistant(self) -> None:
        """The returned transcript includes every model-visible turn."""
        result = self.agent().run("hello")
        roles = [message.role for message in result.messages]
        self.assertEqual(roles[0], Role.SYSTEM)
        self.assertIn(Role.USER, roles)
        self.assertEqual(roles[-1], Role.ASSISTANT)
        self.assertEqual(result.messages[-1].text(), "echo:hello")

    def test_system_prompt_uses_the_product_name(self) -> None:
        """The product is always written MAgent in human-facing prose."""
        agent = self.agent()
        prompt = agent.system_prompt(RunOptions())
        self.assertIn("You are MAgent", prompt)
        self.assertNotIn("You are Magent", prompt)

    def test_ask_returns_text(self) -> None:
        """ask is a concise wrapper around run."""
        self.assertEqual(self.agent().ask("hi"), "echo:hi")

    def test_chat_accepts_wire_dicts(self) -> None:
        """Existing OpenAI-style history can be continued."""
        result = self.agent().chat([
            {"role": "system", "content": "brief"},
            {"role": "user", "content": "continue"},
        ])
        self.assertEqual(result.output, "echo:continue")

    def test_complete_exposes_the_raw_backend_contract(self) -> None:
        """complete bypasses the loop but still returns ChatResponse."""
        response = self.agent().complete(
            CompletionRequest(messages=[Message.user("direct")]))
        self.assertEqual(response.text, "echo:direct")

    def test_empty_prompt_is_valid(self) -> None:
        """An empty user turn is explicit rather than a crash."""
        result = self.agent().run("")
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)
        self.assertEqual(result.output, "echo:")

    def test_result_round_trip(self) -> None:
        """RunResult serialization preserves its public outcome."""
        original = self.agent().run("serialize")
        restored = type(original).from_dict(original.to_dict())
        self.assertEqual(restored.output, original.output)
        self.assertEqual(restored.stop_reason, original.stop_reason)
        self.assertEqual(restored.run_id, original.run_id)
        self.assertEqual(restored.usage.total_tokens, original.usage.total_tokens)


class ToolLoopTests(AgentTestCase):
    """A model-requested tool executes, feeds back, then yields an answer."""

    def test_calculate_round_trip(self) -> None:
        """The canonical two-turn fixture really executes calculate."""
        backend = make_backend(tool_name="calculate",
                               tool_arguments={"expression": "6*7"},
                               final_text="The answer is 42.")
        result = self.agent(backend).run("calculate six times seven")
        self.assertEqual(result.output, "The answer is 42.")
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.tool_error_count, 0)
        tool_result = result.steps[0].tool_results[0]
        self.assertTrue(tool_result.ok)
        self.assertEqual(tool_result.content["result"], 42)

    def test_tool_result_is_added_as_a_tool_message(self) -> None:
        """The next model turn can match the result to the original call id."""
        backend = make_backend(tool_name="calculate",
                               tool_arguments={"expression": "2+3"}, final_text="five")
        result = self.agent(backend).run("calculate")
        tool_messages = [message for message in result.messages if message.role is Role.TOOL]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "calculate")
        self.assertEqual(tool_messages[0].tool_call_id, "call_00")
        self.assertIn("5", tool_messages[0].text())

    def test_unknown_tool_is_reported_to_model_then_recovers(self) -> None:
        """A hallucinated tool does not crash the loop."""
        backend = make_backend(tool_name="does_not_exist", final_text="I recovered.")
        result = self.agent(backend).run("use an unknown tool")
        self.assertEqual(result.output, "I recovered.")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.tool_error_count, 1)
        self.assertEqual(result.steps[0].tool_results[0].error_code, "tool_not_found")

    def test_dry_run_validates_but_does_not_execute(self) -> None:
        """Per-run dry_run is propagated into ToolContext."""
        backend = make_backend(tool_name="calculate",
                               tool_arguments={"expression": "2+3"}, final_text="done")
        result = self.agent(backend).run("calculate", dry_run=True)
        content = result.steps[0].tool_results[0].content
        self.assertEqual(content["dry_run"], True)
        self.assertEqual(content["tool"], "calculate")

    def test_repeated_tool_calls_hit_max_iterations(self) -> None:
        """A model that never answers is stopped at the configured bound."""
        backend = ScriptedBackend(script=[{
            "tool_calls": [{"id": "repeat", "name": "calculate",
                            "arguments": {"expression": "1+1"}}],
            "finish_reason": "tool_calls",
        }])
        result = self.agent(backend).run("loop", max_iterations=2)
        self.assertEqual(result.stop_reason, StopReason.MAX_ITERATIONS)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.tool_call_count, 2)
        self.assertFalse(result.succeeded)

    def test_tool_events_pair_call_and_result(self) -> None:
        """Every tool.call has a tool.result with the same call id."""
        backend = make_backend(tool_name="calculate",
                               tool_arguments={"expression": "3*3"}, final_text="nine")
        result = self.agent(backend).run("calculate")
        calls = [event for event in result.events if event.kind is EventKind.TOOL_CALL]
        results = [event for event in result.events if event.kind is EventKind.TOOL_RESULT]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(calls[0].data["call_id"], results[0].data["call_id"])


class StopTests(AgentTestCase):
    """Stop reasons are explicit and turn into script-friendly outcomes."""

    def test_pre_cancelled_run_stops_before_model_call(self) -> None:
        """cancel() is observed before step one and is reset afterwards."""
        backend = EchoBackend()
        agent = self.agent(backend)
        agent.cancel(reason="test request")
        result = agent.run("do not run")
        self.assertEqual(result.stop_reason, StopReason.CANCELLED)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(backend.stats()["calls"], 0)
        self.assertFalse(agent.cancelled)

    def test_unknown_loop_mode_raises(self) -> None:
        """A typo in loop mode fails before any model request."""
        with self.assertRaises(AgentError):
            self.agent().run("x", loop_mode="plan_execute")

    def test_unknown_run_option_raises(self) -> None:
        """Misspelled options are not silently ignored."""
        with self.assertRaisesRegex(AgentError, "[Uu]nknown run options"):
            self.agent().run("x", temprature=0.1)

    def test_reset_clears_cancel_and_history(self) -> None:
        """reset returns a reusable agent to a clean conversational state."""
        agent = self.agent()
        agent.run("first")
        agent.cancel()
        agent.reset()
        self.assertFalse(agent.cancelled)
        self.assertEqual(agent.history, [])


class EventTests(AgentTestCase):
    """Agent events provide a complete chronological execution trace."""

    def test_plain_run_event_boundaries(self) -> None:
        """The result trace starts once; the terminal self-reference stays on the bus."""
        agent = self.agent()
        result = agent.run("events")
        kinds = [event.kind for event in result.events]
        self.assertEqual(kinds[0], EventKind.RUN_STARTED)
        self.assertEqual(kinds.count(EventKind.RUN_STARTED), 1)
        # run.finished carries the RunResult itself, so nesting it inside
        # result.events would be recursive. It remains available on the bus and
        # through Agent.stream instead.
        self.assertNotIn(EventKind.RUN_FINISHED, kinds)
        self.assertEqual(agent.events.last(EventKind.RUN_FINISHED).run_id, result.run_id)
        self.assertIn(EventKind.LLM_REQUEST, kinds)
        self.assertIn(EventKind.LLM_RESPONSE, kinds)
        self.assertIn(EventKind.STEP_STARTED, kinds)
        self.assertIn(EventKind.STEP_FINISHED, kinds)

    def test_per_run_callback_sees_same_events(self) -> None:
        """on_event receives each event exactly once and is restored afterwards."""
        seen: list[object] = []
        agent = self.agent()
        result = agent.run("callback", on_event=seen.append)
        self.assertEqual([event.id for event in seen[:-1]],
                         [event.id for event in result.events])
        self.assertEqual(seen[-1].kind, EventKind.RUN_FINISHED)
        count = len(seen)
        second = agent.run("no callback")
        self.assertEqual(len(seen), count)
        self.assertTrue(second.succeeded)

    def test_agent_callback_and_run_callback_both_fire_once(self) -> None:
        """The two callback scopes fan out without duplicate delivery."""
        permanent: list[str] = []
        temporary: list[str] = []
        config = self.config()
        agent = Agent(backend=EchoBackend(), config=config.agent, settings=config,
                      registry=default_registry(), workspace=self._home,
                      on_event=lambda event: permanent.append(event.id))
        result = agent.run("both", on_event=lambda event: temporary.append(event.id))
        expected_prefix = [event.id for event in result.events]
        self.assertEqual(permanent, temporary)
        self.assertEqual(permanent[:-1], expected_prefix)
        self.assertEqual(len(permanent), len(expected_prefix) + 1)

    def test_reused_agent_results_do_not_include_prior_run_events(self) -> None:
        """Each RunResult is filtered by run_id from the shared bus history."""
        agent = self.agent()
        first = agent.run("first")
        second = agent.run("second")
        self.assertTrue(first.events)
        self.assertTrue(second.events)
        self.assertTrue(all(event.run_id == first.run_id for event in first.events))
        self.assertTrue(all(event.run_id == second.run_id for event in second.events))
        self.assertFalse({event.id for event in first.events} &
                         {event.id for event in second.events})

    def test_stream_yields_each_event_once(self) -> None:
        """Regression: stream used to install the same collector twice."""
        events = list(self.agent().stream("streamed"))
        ids = [event.id for event in events]
        self.assertEqual(len(ids), len(set(ids)))
        kinds = [event.kind for event in events]
        self.assertEqual(kinds.count(EventKind.RUN_STARTED), 1)
        self.assertEqual(kinds.count(EventKind.RUN_FINISHED), 1)

    def test_stream_final_event_carries_result(self) -> None:
        """A streaming caller still receives the complete RunResult."""
        final = list(self.agent().stream("streamed"))[-1]
        self.assertEqual(final.kind, EventKind.RUN_FINISHED)
        result = final.data["result"]
        self.assertEqual(result.output, "echo:streamed")


class StructuredOutputTests(AgentTestCase):
    """JSON answers parse, validate, normalize, and repair once."""

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name", "count"],
        "additionalProperties": False,
    }

    def test_plain_json(self) -> None:
        """A valid object is available separately from its formatted output."""
        backend = ScriptedBackend(script=['{"name":"MAgent","count":3}'])
        result = self.agent(backend).run("return json", structured=self.schema)
        self.assertEqual(result.structured, {"name": "MAgent", "count": 3})
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)
        self.assertIn('"name": "MAgent"', result.output)

    def test_fenced_json(self) -> None:
        """Markdown fences from an instruction-following model are removed."""
        backend = ScriptedBackend(script=['```json\n{"name":"MAgent","count":3}\n```'])
        result = self.agent(backend).run("return json", structured=self.schema)
        self.assertEqual(result.structured["count"], 3)

    def test_invalid_first_answer_is_repaired_once(self) -> None:
        """The agent sends schema feedback and accepts a valid second answer."""
        backend = ScriptedBackend(script=[
            '{"name":"MAgent"}',
            '{"name":"MAgent","count":7}',
        ])
        result = self.agent(backend).run("return json", structured=self.schema)
        self.assertEqual(result.structured, {"name": "MAgent", "count": 7})
        self.assertGreaterEqual(len(backend.calls_log), 2)
        repairs = [event for event in result.events
                   if event.kind is EventKind.CUSTOM and
                   event.data.get("kind_label") == "structured.repair"]
        self.assertEqual(len(repairs), 1)


class InspectionTests(AgentTestCase):
    """Describe and stats are safe operator-facing snapshots."""

    def test_describe(self) -> None:
        """The snapshot names MAgent, backend, workspace, and real tool count."""
        described = self.agent().describe()
        self.assertEqual(described["name"], "MAgent")
        self.assertEqual(described["backend"], "echo")
        self.assertEqual(described["tools"], 113)
        self.assertEqual(described["workspace"], str(self._home))

    def test_stats_count_runs(self) -> None:
        """Cumulative stats increase across reusable runs."""
        agent = self.agent()
        agent.run("one")
        agent.run("two")
        stats = agent.stats()
        self.assertEqual(stats["runs"], 2)
        self.assertEqual(stats["agent"], "MAgent")
        self.assertGreater(stats["usage"]["requests"], 0)


if __name__ == "__main__":
    unittest.main()
