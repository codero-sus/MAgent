"""Tests for tool schema inference, registration, discovery, and dispatch.

A tool call crosses the model boundary as untrusted JSON. The registry must
validate that JSON, resolve aliases, inject a constrained context, cap output,
and always return one :class:`~magent.types.ToolResult` rather than leaking a
handler exception into the agent loop.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from magent.errors import ToolError, ToolNotFoundError
from magent.tools import (
    RISK_LEVELS,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    default_registry,
    tool,
)
from magent.types import ToolCall

from tests.support import MAgentTestCase


@tool(category="test", tags=["arithmetic"])
def add_numbers(left: int, right: int = 1) -> int:
    """Add two integers.

    Args:
        left: first integer
        right: second integer
    """
    return left + right


@tool(category="test", output_limit=20)
def long_answer(length: int = 100) -> str:
    """Return a deliberately long answer.

    Args:
        length: number of characters to return
    """
    return "x" * length


@tool(category="test")
def context_name(value: str, context: ToolContext) -> dict[str, str]:
    """Return context metadata to prove it was injected.

    Args:
        value: value to echo
        context: invocation context supplied by the registry
    """
    return {"value": value, "tool": context.tool_name, "call": context.call_id}


@tool(category="test")
def explodes() -> str:
    """Raise to prove handler exceptions become ToolResults."""
    raise RuntimeError("kaboom")


@tool(category="test", parallel_safe=True)
def delayed(value: int, delay: float = 0.01) -> int:
    """Sleep briefly and return a value.

    Args:
        value: value to return
        delay: seconds to sleep
    """
    time.sleep(delay)
    return value


class SchemaInferenceTests(unittest.TestCase):
    """The decorator derives a real JSON schema from Python hints."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.add = self.registry.register(add_numbers)

    def test_name_and_description_are_inferred(self) -> None:
        """Function name and doc summary become model-facing fields."""
        self.assertEqual(self.add.name, "add_numbers")
        self.assertEqual(self.add.description, "Add two integers.")
        self.assertEqual(self.add.category, "test")

    def test_parameter_types_are_inferred(self) -> None:
        """Annotated integers become JSON integer properties."""
        schema = self.add.spec.parameters
        self.assertEqual(schema["properties"]["left"]["type"], "integer")
        self.assertEqual(schema["properties"]["right"]["type"], "integer")

    def test_required_parameters_follow_python_defaults(self) -> None:
        """Only the parameter without a Python default is required."""
        self.assertEqual(self.add.spec.required_parameters(), ["left"])
        self.assertEqual(self.add.spec.parameters["properties"]["right"]["default"], 1)

    def test_google_arg_docs_are_preserved(self) -> None:
        """The model receives the per-argument documentation."""
        props = self.add.spec.parameters["properties"]
        self.assertEqual(props["left"]["description"], "first integer")
        self.assertEqual(props["right"]["description"], "second integer")

    def test_openai_shape(self) -> None:
        """to_openai emits the function wrapper Cortex expects."""
        payload = self.add.to_openai()
        self.assertEqual(payload["type"], "function")
        self.assertEqual(payload["function"]["name"], "add_numbers")
        self.assertIn("parameters", payload["function"])

    def test_spec_round_trip(self) -> None:
        """A ToolSpec serializes and parses without losing policy fields."""
        original = ToolSpec(name="demo", category="tests", risk="write",
                            aliases=["d"], requires_approval=True)
        restored = ToolSpec.from_dict(original.to_dict())
        self.assertEqual(restored.name, "demo")
        self.assertEqual(restored.risk, "write")
        self.assertTrue(restored.requires_approval)
        self.assertEqual(restored.aliases, ["d"])


class RegistryTests(unittest.TestCase):
    """Registration is deterministic and rejects accidental replacement."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_register_function(self) -> None:
        """Decorated functions become Tool instances."""
        registered = self.registry.register(add_numbers)
        self.assertEqual(registered.name, "add_numbers")
        self.assertIs(self.registry.get("add_numbers"), registered)

    def test_duplicate_registration_is_rejected(self) -> None:
        """A later plugin cannot silently shadow an existing tool."""
        self.registry.register(add_numbers)
        with self.assertRaises(ToolError):
            self.registry.register(add_numbers)

    def test_replace_is_explicit(self) -> None:
        """replace=True makes replacement a visible caller decision."""
        first = self.registry.register(add_numbers)
        second = self.registry.register(add_numbers, replace=True)
        # A decorated function owns one Tool object; replace=True permits that
        # object to be deliberately re-bound rather than fabricating a clone.
        self.assertIs(first, second)
        self.assertIs(self.registry.get("add_numbers"), second)

    def test_unregister_reports_presence(self) -> None:
        """unregister returns False once the name is gone."""
        self.registry.register(add_numbers)
        self.assertTrue(self.registry.unregister("add_numbers"))
        self.assertFalse(self.registry.unregister("add_numbers"))

    def test_require_raises_a_specific_error(self) -> None:
        """require() distinguishes a missing tool from a None lookup."""
        with self.assertRaises(ToolNotFoundError):
            self.registry.require("missing")

    def test_names_are_sorted(self) -> None:
        """Stable ordering keeps prompts and snapshots deterministic."""
        self.registry.register(long_answer)
        self.registry.register(add_numbers)
        self.assertEqual(self.registry.names(), sorted(self.registry.names()))


class DiscoveryTests(unittest.TestCase):
    """Models and operators can find the right subset of tools."""

    def setUp(self) -> None:
        self.registry = default_registry()

    def test_builtin_catalogue_is_real_and_large(self) -> None:
        """The standard registry contains the 113 implemented tools."""
        self.assertEqual(self.registry.count(), 113)
        self.assertEqual(len(self.registry.names()), len(set(self.registry.names())))
        self.assertGreaterEqual(len(self.registry.categories()), 16)

    def test_every_builtin_has_a_schema_and_description(self) -> None:
        """No registered tool is a name-only placeholder."""
        for item in self.registry.tools():
            self.assertTrue(item.description.strip(), item.name)
            self.assertEqual(item.spec.parameters.get("type"), "object", item.name)
            self.assertIn(item.spec.risk, RISK_LEVELS, item.name)

    def test_filter_by_category(self) -> None:
        """Filtering returns only the requested category."""
        matches = self.registry.filter(categories=["text"])
        self.assertGreater(len(matches), 3)
        self.assertTrue(all(item.category == "text" for item in matches))

    def test_filter_excludes_network_tools(self) -> None:
        """network=False removes tools that can reach the network."""
        matches = self.registry.filter(network=False)
        self.assertTrue(all(not item.spec.network for item in matches))

    def test_search_text(self) -> None:
        """Free-text search finds a tool by description or name."""
        matches = self.registry.search(text="hash")
        self.assertTrue(matches)
        self.assertTrue(any("hash" in item.name for item in matches))

    def test_suggest_close_name(self) -> None:
        """A typo gets a useful suggestion."""
        self.assertIn("calculate", self.registry.suggest("calculat"))

    def test_schemas_match_count(self) -> None:
        """Every visible built-in contributes one model-facing schema."""
        self.assertEqual(len(self.registry.schemas()), self.registry.count())

    def test_prompt_block_lists_tools(self) -> None:
        """The compact prompt is meaningful, not an empty wrapper."""
        block = self.registry.prompt_block()
        self.assertIn("calculate", block)
        self.assertIn("read_file", block)


class DispatchTests(MAgentTestCase):
    """Dispatch validates, executes, bounds, and reports calls."""

    def setUp(self) -> None:
        super().setUp()
        self.registry = ToolRegistry()
        for fn in (add_numbers, long_answer, context_name, explodes, delayed):
            self.registry.register(fn)
        self.context = ToolContext(workspace=self._home)

    def test_successful_dispatch(self) -> None:
        """A valid call returns its content and metadata."""
        result = self.registry.dispatch(
            ToolCall(id="c1", name="add_numbers", arguments={"left": 2, "right": 3}),
            self.context,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, 5)
        self.assertEqual(result.call_id, "c1")
        self.assertEqual(result.metadata["category"], "test")

    def test_default_argument_is_applied(self) -> None:
        """Schema validation supplies a declared default."""
        result = self.registry.dispatch(
            ToolCall(id="c", name="add_numbers", arguments={"left": 4}), self.context)
        self.assertTrue(result.ok)
        self.assertEqual(result.content, 5)

    def test_numeric_string_is_coerced(self) -> None:
        """Safe schema coercion turns a JSON string into an integer."""
        result = self.registry.dispatch(
            ToolCall(id="c", name="add_numbers", arguments={"left": "4", "right": 2}),
            self.context,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, 6)

    def test_missing_required_argument_is_a_failure_result(self) -> None:
        """Validation never leaks an exception out of dispatch."""
        result = self.registry.dispatch(ToolCall(id="c", name="add_numbers"), self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "tool_validation_error")
        self.assertIn("left", result.error or "")

    def test_unknown_tool_is_a_failure_result_with_suggestion(self) -> None:
        """An unknown call keeps its id and offers the nearest name."""
        result = self.registry.dispatch(ToolCall(id="c9", name="add_number"), self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.call_id, "c9")
        self.assertEqual(result.error_code, "tool_not_found")
        self.assertIn("add_numbers", result.error or "")

    def test_handler_exception_is_a_failure_result(self) -> None:
        """Arbitrary plugin exceptions are contained."""
        result = self.registry.dispatch(ToolCall(id="c", name="explodes"), self.context)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "tool_execution_error")
        self.assertIn("kaboom", result.error or "")

    def test_context_is_injected_and_specialized(self) -> None:
        """The handler sees the resolved name and original call id."""
        result = self.registry.dispatch(
            ToolCall(id="call_77", name="context_name", arguments={"value": "yes"}),
            self.context,
        )
        self.assertEqual(result.content,
                         {"value": "yes", "tool": "context_name", "call": "call_77"})

    def test_output_is_truncated(self) -> None:
        """Oversized output cannot consume the model context."""
        result = self.registry.dispatch(
            ToolCall(id="c", name="long_answer", arguments={"length": 100}), self.context)
        self.assertTrue(result.ok)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.content), 20)

    def test_dry_run_does_not_execute_handler(self) -> None:
        """dry_run validates and reports arguments without side effects."""
        result = self.registry.dispatch(
            ToolCall(id="c", name="explodes"), self.context.sub(dry_run=True))
        self.assertTrue(result.ok)
        self.assertEqual(result.content["dry_run"], True)
        self.assertEqual(result.content["tool"], "explodes")

    def test_stats_count_success_and_failure(self) -> None:
        """Registry telemetry accounts for both outcomes."""
        self.registry.dispatch(ToolCall(id="a", name="add_numbers", arguments={"left": 1}),
                               self.context)
        self.registry.dispatch(ToolCall(id="b", name="missing"), self.context)
        stats = self.registry.stats()
        self.assertEqual(stats["dispatch_count"], 2)
        self.assertEqual(stats["dispatch_errors"], 1)

    def test_dispatch_many_preserves_input_order(self) -> None:
        """Parallel completion order never scrambles call/result pairing."""
        calls = [
            ToolCall(id="slow", name="delayed", arguments={"value": 1, "delay": 0.03}),
            ToolCall(id="fast", name="delayed", arguments={"value": 2, "delay": 0.001}),
        ]
        results = self.registry.dispatch_many(calls, self.context, parallel=True, max_workers=2)
        self.assertEqual([result.call_id for result in results], ["slow", "fast"])
        self.assertEqual([result.content for result in results], [1, 2])

    def test_dispatch_many_invokes_callback_for_every_result(self) -> None:
        """Streaming renderers get one callback per call."""
        seen: list[str] = []
        calls = [ToolCall(id=str(n), name="add_numbers", arguments={"left": n})
                 for n in range(3)]
        results = self.registry.dispatch_many(calls, self.context, on_result=lambda r: seen.append(r.call_id))
        self.assertEqual(len(results), 3)
        self.assertEqual(set(seen), {"0", "1", "2"})


class ContextTests(MAgentTestCase):
    """A ToolContext resolves relative paths and can be specialized safely."""

    def test_relative_path_uses_workspace(self) -> None:
        """A bare filename never resolves against the process cwd."""
        context = ToolContext(workspace=self._home)
        self.assertEqual(context.resolve_path("notes.txt"), self._home / "notes.txt")

    def test_sub_does_not_mutate_parent(self) -> None:
        """Per-call metadata is copied into a specialized context."""
        parent = ToolContext(workspace=self._home, step_index=1)
        child = parent.sub(step_index=2, call_id="c")
        self.assertEqual(parent.step_index, 1)
        self.assertEqual(child.step_index, 2)
        self.assertEqual(child.call_id, "c")

    def test_workspace_path_is_absolute(self) -> None:
        """workspace_path normalizes the public workspace value."""
        context = ToolContext(workspace=Path("."))
        self.assertTrue(context.workspace_path.is_absolute())


if __name__ == "__main__":
    unittest.main()
