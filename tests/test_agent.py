"""Tests for the agent loop: tools, streaming, plans, reflection, approvals."""

from __future__ import annotations

import unittest

from magent.agent import Agent, AgentConfig, Plan, make_agent, parse_reflection
from magent.errors import StructuredOutputError
from magent.testing import EchoBackend, ScriptedBackend, ToolScriptBackend
from magent.types import EventKind, RunResult, StopReason
from tests.support import MAgentTestCase


class AgentLoopTest(MAgentTestCase):
    """The react loop, tool calling and stopping conditions."""

    def _tool_agent(self, *scripts: object) -> Agent:
        backend = ToolScriptBackend(script=list(scripts))
        return self.agent(backend=backend)

    def test_simple_answer(self) -> None:
        result = self.agent().run("say hello")
        self.assertTrue(result.succeeded)
        self.assertIn("hello", result.output)
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)

    def test_tool_call_round_trip(self) -> None:
        agent = self._tool_agent(
            {"tool_calls": [{"name": "calculate", "arguments": {"expression": "6*7"}}]},
            {"text": "The answer is 42."},
        )
        result = agent.run("what is 6*7?")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.tool_call_count, 1)
        calls = [call.name for step in result.steps for call in step.tool_calls]
        self.assertIn("calculate", calls)
        self.assertIn("42", result.output)

    def test_max_iterations_is_enforced(self) -> None:
        agent = self._tool_agent(
            {"tool_calls": [{"name": "calculate", "arguments": {"expression": "1+1"}}]}
        )
        result = agent.run("loop forever", max_iterations=2)
        self.assertEqual(result.stop_reason, StopReason.MAX_ITERATIONS)
        self.assertLessEqual(len(result.steps), 3)

    def test_plan_mode_builds_and_follows_a_plan(self) -> None:
        backend = ScriptedBackend(
            script=[
                '{"steps": [{"goal": "read the file", "tool": "read_file"}, '
                '{"goal": "summarise it"}]}',
                {"text": "done"},
            ]
        )
        agent = self.agent(backend=backend)
        result = agent.run("summarise the file", loop_mode="plan")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.metadata["mode"], "plan")

    def test_reflect_mode_stops_when_satisfied(self) -> None:
        backend = ScriptedBackend(
            script=[{"text": "first answer"}, {"text": "STOP: yes, the goal is met"}]
        )
        agent = self.agent(backend=backend)
        result = agent.run("explain something", loop_mode="reflect")
        self.assertTrue(result.steps)

    def test_single_mode_makes_one_call(self) -> None:
        agent = self.agent()
        result = agent.run("just answer", loop_mode="single")
        self.assertEqual(len(result.steps), 1)
        self.assertTrue(result.succeeded)

    def test_structured_output_is_parsed(self) -> None:
        backend = ScriptedBackend(script=['{"name": "magent", "ok": true}'])
        agent = self.agent(backend=backend)
        result = agent.run("give me json", loop_mode="single", structured=True)
        self.assertTrue(result.structured)
        self.assertEqual(result.structured["name"], "magent")

    def test_structured_output_repairs_and_then_gives_up(self) -> None:
        backend = ScriptedBackend(script=["not json at all", "still not json"])
        agent = self.agent(backend=backend)
        with self.assertRaises(StructuredOutputError):
            agent.run("give me json", loop_mode="single", structured=True)

    def test_tool_error_is_reported_not_fatal(self) -> None:
        agent = self._tool_agent(
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "missing.txt"}}]},
            {"text": "I could not read the file."},
        )
        result = agent.run("read missing.txt")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.tool_error_count, 1)

    def test_unknown_tool_becomes_an_error_result(self) -> None:
        agent = self._tool_agent(
            {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
            {"text": "That tool does not exist."},
        )
        result = agent.run("call a bogus tool")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.tool_error_count, 1)

    def test_usage_and_cost_are_recorded(self) -> None:
        result = self.agent().run("hello")
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertIsNotNone(result.cost)

    def test_run_result_serialises(self) -> None:
        result = self.agent().run("hello")
        payload = result.to_dict()
        self.assertIn("output", payload)
        self.assertIn("usage", payload)

    def test_events_are_emitted(self) -> None:
        agent = self.agent()
        result = agent.run("hello")
        kinds = {event.kind for event in result.events}
        self.assertIn(EventKind.RUN_STARTED, kinds)
        self.assertIn(EventKind.RUN_FINISHED, kinds)

    def test_stream_yields_each_event_once(self) -> None:
        agent = self.agent()
        events = list(agent.stream("hello"))
        kinds = [event.kind for event in events]
        self.assertEqual(kinds.count(EventKind.RUN_STARTED), 1)
        self.assertEqual(kinds.count(EventKind.RUN_FINISHED), 1)
        finished = [event for event in events if event.kind == EventKind.RUN_FINISHED]
        self.assertIsInstance(finished[0].data.get("result"), RunResult)

    def test_cancel_before_a_run_stops_it(self) -> None:
        agent = self.agent()
        agent.cancel(reason="test")
        result = agent.run("do something")
        self.assertEqual(result.stop_reason, StopReason.CANCELLED)
        self.assertFalse(agent.cancelled)

    def test_ask_returns_plain_text(self) -> None:
        self.assertIsInstance(self.agent().ask("hi"), str)

    def test_critique_returns_findings(self) -> None:
        agent = self.agent()
        critique = agent.critique("question", "answer")
        self.assertIsInstance(critique, dict)

    def test_describe_and_stats(self) -> None:
        agent = self.agent()
        agent.run("hello")
        self.assertIn("tools", agent.describe())
        self.assertTrue(agent.stats())

    def test_unknown_run_option_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.agent().run("hello", not_an_option=True)


class ApprovalIntegrationTest(MAgentTestCase):
    """Approvals gate tool calls inside a real run."""

    def test_denied_approval_becomes_a_tool_result(self) -> None:
        from magent.tools import ApprovalManager, ApprovalPolicy
        from magent.tools.approval import deny_all

        config = self.config()
        config.tools.approval_mode = "always"
        manager = ApprovalManager(ApprovalPolicy(mode="always"), prompt=deny_all)
        agent = Agent(
            backend=ToolScriptBackend(
                script=[
                    {
                        "tool_calls": [
                            {
                                "name": "write_file",
                                "arguments": {
                                    "path": str(self.workspace / "x.txt"),
                                    "content": "nope",
                                },
                            }
                        ]
                    },
                    {"text": "I was not allowed to write the file."},
                ]
            ),
            settings=config,
            workspace=self.workspace,
            approval=manager,
        )
        result = agent.run("write a file")
        self.assertTrue(result.succeeded)
        self.assertFalse((self.workspace / "x.txt").exists())
        codes = [item.error_code for step in result.steps for item in step.tool_results]
        self.assertIn("approval_denied", codes)


class AgentConfigTest(unittest.TestCase):
    """Agent configuration validates its own invariants."""

    def test_loop_modes_are_validated(self) -> None:
        config = AgentConfig()
        config.loop_mode = "nonsense"
        with self.assertRaises(Exception):
            config.validate()

    def test_defaults_are_sane(self) -> None:
        config = AgentConfig()
        config.validate()
        self.assertGreater(config.max_iterations, 0)
        self.assertIn(config.loop_mode, {"react", "plan", "reflect", "single"})

    def test_make_agent_helper_builds_an_agent(self) -> None:
        agent = make_agent(backend=EchoBackend(), loop_mode="single")
        self.assertIsInstance(agent, Agent)
        self.assertTrue(agent.run("hi").succeeded)


class ReflectionParsingTest(unittest.TestCase):
    """Reflection text is parsed into a structured decision."""

    def test_stop_token_is_recognised(self) -> None:
        reflection = parse_reflection("Looks complete.\nSTOP: yes", 1)
        self.assertTrue(reflection.stop)

    def test_continue_token_is_recognised(self) -> None:
        reflection = parse_reflection("Not yet.\nSTOP: no\nNext: run the tests", 2)
        self.assertFalse(reflection.stop)
        self.assertEqual(reflection.step, 2)


class PlanModelTest(unittest.TestCase):
    """Plan and PlanStep behave like data."""

    def test_plan_serialises(self) -> None:
        plan = Plan(goal="do the thing")
        payload = plan.to_dict()
        self.assertEqual(payload["goal"], "do the thing")
        self.assertEqual(payload["steps"], [])


if __name__ == "__main__":
    unittest.main()
