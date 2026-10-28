"""Tests for the tool layer: schema inference, registry, sandbox and approval."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Literal

from magent.errors import (
    ApprovalRequiredError,
    PermissionDeniedError,
    ToolNotFoundError,
)
from magent.tools import (
    MODES,
    RISK_LEVELS,
    ApprovalManager,
    ToolContext,
    build_sandbox,
    default_registry,
    string_schema,
    tool,
)
from magent.tools.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    auto_approve,
    build_request,
    deny_all,
)
from magent.tools.base import tool_spec_from_function
from magent.types import ToolCall
from tests.support import MAgentTestCase


class SchemaTest(unittest.TestCase):
    """Tool schemas are inferred from hints, defaults and docstrings."""

    def test_infers_types_and_defaults(self) -> None:
        def sample(path: str, count: int = 3, flag: bool = False) -> str:
            """Do a thing.

            Args:
                path: the file to read
                count: how many lines
                flag: whether to include headers
            """
            return ""

        spec = tool_spec_from_function(sample, name="sample")
        properties = spec.parameters["properties"]
        self.assertEqual(spec.name, "sample")
        self.assertEqual(spec.description.split("\n")[0], "Do a thing.")
        self.assertEqual(properties["path"]["type"], "string")
        self.assertEqual(properties["count"]["type"], "integer")
        self.assertEqual(properties["flag"]["type"], "boolean")
        self.assertEqual(spec.parameters["required"], ["path"])
        self.assertEqual(properties["path"]["description"], "the file to read")

    def test_literal_becomes_an_enum(self) -> None:
        def choose(mode: Literal["fast", "safe"]) -> str:
            """Choose a mode."""
            return mode

        spec = tool_spec_from_function(choose, name="choose")
        self.assertEqual(spec.parameters["properties"]["mode"]["enum"], ["fast", "safe"])

    def test_optional_and_path_types(self) -> None:
        def read(path: Path, limit: int | None = None) -> str:
            """Read something."""
            return ""

        spec = tool_spec_from_function(read, name="read")
        properties = spec.parameters["properties"]
        self.assertEqual(properties["path"]["type"], "string")
        self.assertTrue(properties["limit"]["nullable"])
        self.assertEqual(spec.parameters["required"], ["path"])

    def test_decorator_keeps_the_function_callable(self) -> None:
        @tool(name="double", description="Double a number", category="math")
        def double(value: int) -> int:
            """Double the value."""
            return value * 2

        self.assertEqual(double(4), 8)
        self.assertEqual(double.tool.spec.name, "double")
        self.assertEqual(double.tool.run({"value": 4}).content, 8)

    def test_validation_rejects_bad_input(self) -> None:
        @tool(name="strict", description="Needs an int", category="math")
        def strict(value: int) -> int:
            """Echo an int."""
            return value

        result = strict.tool.run({"value": "not-an-int"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "tool_validation_error")

    def test_schema_helpers_build_json_schema_fragments(self) -> None:
        schema = string_schema("the name", enum=["a", "b"])
        self.assertEqual(schema["type"], "string")
        self.assertEqual(schema["enum"], ["a", "b"])

        from magent.tools import integer_schema, object_schema

        combined = object_schema({"count": integer_schema("counts things")}, ["count"])
        self.assertEqual(combined["properties"]["count"]["type"], "integer")
        self.assertEqual(combined["required"], ["count"])


class RegistryTest(MAgentTestCase):
    """The default registry is complete and introspectable."""

    def test_registry_is_populated(self) -> None:
        registry = default_registry()
        self.assertGreaterEqual(registry.count(), 100)
        self.assertGreaterEqual(len(registry.categories()), 10)

    def test_every_tool_has_a_description_and_schema(self) -> None:
        for item in default_registry().tools():
            self.assertTrue(item.spec.description.strip(), item.spec.name)
            self.assertFalse(item.spec.description.lower().startswith("todo"), item.spec.name)
            self.assertIsInstance(item.to_openai()["function"]["parameters"], dict)

    def test_get_returns_none_but_require_raises(self) -> None:
        registry = default_registry()
        self.assertIsNone(registry.get("definitely_not_a_tool"))
        with self.assertRaises(ToolNotFoundError):
            registry.require("definitely_not_a_tool")
        self.assertIn("read_file", registry.suggest("read_fil"))

    def test_filter_by_category_and_risk(self) -> None:
        registry = default_registry()
        git_tools = registry.filter(categories=["git"])
        self.assertTrue(git_tools)
        self.assertTrue(all(item.spec.category == "git" for item in git_tools))
        read_only = registry.filter(max_risk="read")
        self.assertTrue(all(item.spec.risk in {"safe", "read"} for item in read_only))

    def test_search_matches_name_and_description(self) -> None:
        found = default_registry().search(text="sql")
        self.assertTrue(found)
        self.assertTrue(any("sql" in spec.name for spec in found))

    def test_search_by_category_and_risk(self) -> None:
        specs = default_registry().search(category="git")
        self.assertTrue(specs)
        self.assertTrue(all(spec.category == "git" for spec in specs))

    def test_risk_levels_are_ordered(self) -> None:
        self.assertEqual(RISK_LEVELS[0], "safe")
        self.assertLess(RISK_LEVELS.index("read"), RISK_LEVELS.index("write"))
        self.assertLess(RISK_LEVELS.index("write"), RISK_LEVELS.index("execute"))

    def test_dispatch_runs_a_tool_end_to_end(self) -> None:
        registry = default_registry()
        context = ToolContext(
            config=self.config(),
            registry=registry,
            sandbox=build_sandbox(self.config(), workspace=self.workspace),
        )
        result = registry.dispatch(
            ToolCall(id="c1", name="calculate", arguments={"expression": "6*7"}), context=context
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content["result"], 42)

    def test_dispatch_reports_validation_errors(self) -> None:
        registry = default_registry()
        result = registry.dispatch(
            ToolCall(id="c2", name="calculate", arguments={"expression": "6*"})
        )
        self.assertFalse(result.ok)
        self.assertIn(result.error_code or "", {"tool_validation_error", "tool_execution_error"})

    def test_dispatch_many_runs_in_parallel(self) -> None:
        registry = default_registry()
        calls = [
            ToolCall(id=f"c{index}", name="calculate", arguments={"expression": f"{index}+1"})
            for index in range(4)
        ]
        results = registry.dispatch_many(calls, parallel=True)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(item.ok for item in results))

    def test_stats_count_calls(self) -> None:
        registry = default_registry()
        registry.dispatch(ToolCall(id="c3", name="calculate", arguments={"expression": "1+1"}))
        self.assertTrue(registry.stats())


class SandboxTest(MAgentTestCase):
    """The sandbox denies by default and explains every decision."""

    def setUp(self) -> None:
        super().setUp()
        self.config_ = self.config()
        self.sandbox = build_sandbox(self.config_, workspace=self.workspace)

    def test_reads_inside_the_workspace_are_allowed(self) -> None:
        target = self.write("inside.txt", "ok")
        self.assertTrue(self.sandbox.check_read(target).allowed)

    def test_reads_outside_the_workspace_are_denied(self) -> None:
        with self.assertRaises(PermissionDeniedError) as caught:
            self.sandbox.check_read(Path("/etc/hostname"))
        self.assertIn("outside the workspace", str(caught.exception))

    def test_path_traversal_is_denied(self) -> None:
        sneaky = self.workspace / ".." / ".." / "etc" / "passwd"
        with self.assertRaises(PermissionDeniedError):
            self.sandbox.check_read(sneaky)

    def test_denied_commands_are_refused(self) -> None:
        with self.assertRaises(PermissionDeniedError):
            self.sandbox.check_command("rm -rf /")
        self.assertTrue(self.sandbox.check_command("ls").allowed)

    def test_write_size_ceiling(self) -> None:
        self.config_.tools.max_file_write_bytes = 4
        sandbox = build_sandbox(self.config_, workspace=self.workspace)
        with self.assertRaises(PermissionDeniedError):
            sandbox.check_write(self.workspace / "big.txt", size=100)

    def test_audit_trail_is_kept(self) -> None:
        self.sandbox.check_read(self.write("a.txt", "x"))
        self.assertTrue(self.sandbox.recent_decisions())

    def test_network_policy(self) -> None:
        self.config_.tools.allow_network = False
        sandbox = build_sandbox(self.config_, workspace=self.workspace)
        with self.assertRaises(PermissionDeniedError):
            sandbox.check_url("https://example.com")

    def test_warn_mode_returns_a_decision_instead_of_raising(self) -> None:
        sandbox = build_sandbox(self.config_, workspace=self.workspace)
        sandbox.policy.deny_action = "warn"
        decision = sandbox.check_read(Path("/etc/hostname"))
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.reason)

    def test_stats_describe_decisions(self) -> None:
        self.sandbox.check_read(self.write("b.txt", "x"))
        self.assertTrue(self.sandbox.stats())


class ApprovalTest(MAgentTestCase):
    """Approval modes gate risky tools."""

    def _manager(self, mode: str, prompt=None) -> ApprovalManager:
        return ApprovalManager(ApprovalPolicy(mode=mode), prompt=prompt, events=self._events())

    def _events(self):
        from magent.events import EventBus

        return EventBus()

    def _request(self, name: str):
        registry = default_registry()
        tool_obj = registry.get(name)
        return build_request(
            name,
            {"path": "x"},
            risk=tool_obj.spec.risk,
            destructive=tool_obj.spec.destructive,
            network=tool_obj.spec.network,
            required=tool_obj.spec.requires_approval,
        )

    def test_never_mode_approves_everything(self) -> None:
        manager = self._manager("never")
        needed, _ = manager.needs_approval(self._request("write_file"))
        self.assertFalse(needed)

    def test_dangerous_mode_asks_for_writes(self) -> None:
        manager = self._manager("dangerous")
        self.assertTrue(manager.needs_approval(self._request("write_file"))[0])
        self.assertFalse(manager.needs_approval(self._request("read_file"))[0])

    def test_always_mode_asks_for_everything(self) -> None:
        manager = self._manager("always")
        self.assertTrue(manager.needs_approval(self._request("read_file"))[0])

    def test_explicit_requirement_asks_even_for_read_risk(self) -> None:
        manager = self._manager("dangerous")
        request = build_request("sensitive_read", {}, risk="read", required=True)
        self.assertTrue(manager.needs_approval(request)[0])
        self.assertFalse(self._manager("never").needs_approval(request)[0])

    def test_denial_raises_and_is_remembered(self) -> None:
        manager = self._manager("always", prompt=deny_all)
        with self.assertRaises(ApprovalRequiredError):
            manager.evaluate(self._request("read_file"))
        self.assertTrue(manager.history())
        self.assertEqual(manager.history()[0]["decision"], "deny")
        self.assertTrue(manager.stats())

    def test_approval_callback_grants_and_remembers(self) -> None:
        manager = self._manager("always", prompt=auto_approve)
        record = manager.evaluate(self._request("read_file"))
        self.assertEqual(record.decision, ApprovalDecision.APPROVE)
        self.assertEqual(record.request.tool, "read_file")

    def test_modes_cover_the_documented_set(self) -> None:
        self.assertEqual(set(MODES), {"never", "ask", "always", "dangerous"})


class BuiltinToolsTest(MAgentTestCase):
    """A representative sample of the shipped tools actually works."""

    def _context(self) -> ToolContext:
        """A sandboxed, workspace-scoped context for tool calls."""
        config = self.config()
        return ToolContext(
            config=config,
            registry=default_registry(),
            sandbox=build_sandbox(config, workspace=self.workspace),
        )

    def _run(self, name: str, arguments: dict) -> object:
        """Call one tool through the registry with a sandboxed context."""
        return default_registry().get(name).run(arguments, context=self._context())

    def test_write_read_edit_round_trip(self) -> None:
        target = self.workspace / "note.txt"
        written = self._run("write_file", {"path": str(target), "content": "hello"})
        self.assertTrue(written.ok, written.error)
        read = self._run("read_file", {"path": str(target)})
        self.assertEqual(read.content["content"], "hello")
        edited = self._run(
            "edit_file", {"path": str(target), "old_text": "hello", "new_text": "world"}
        )
        self.assertTrue(edited.ok, edited.error)
        self.assertEqual(target.read_text(encoding="utf-8"), "world")

    def test_grep_finds_matches(self) -> None:
        self.write("src/app.py", "def main():\n    return 1\n")
        result = self._run("grep", {"pattern": "def main", "path": str(self.workspace)})
        self.assertTrue(result.ok, result.error)
        self.assertIn("def main", json.dumps(result.content))

    def test_calculate_handles_math(self) -> None:
        for expression, expected in (("2+2", 4), ("2**10", 1024), ("max(1, 5)", 5)):
            result = self._run("calculate", {"expression": expression})
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.content["result"], expected)

    def test_json_query_reads_a_field(self) -> None:
        result = self._run("json_query", {"data": json.dumps({"a": {"b": 7}}), "path": "a.b"})
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.content["value"], 7)

    def test_csv_read_describes_a_file(self) -> None:
        target = self.write("rows.csv", "name,value\na,1\nb,2\n")
        described = self._run("csv_read", {"path": str(target)})
        self.assertTrue(described.ok, described.error)
        self.assertIn("value", json.dumps(described.content))

    def test_csv_stats_counts_rows(self) -> None:
        target = self.write("rows.csv", "name,value\na,1\nb,2\n")
        stats = self._run("csv_stats", {"path": str(target)})
        self.assertTrue(stats.ok, stats.error)
        self.assertIn("rows", json.dumps(stats.content))

    def test_dangerous_command_is_refused_by_the_sandbox(self) -> None:
        result = self._run("run_command", {"command": "rm -rf /"})
        self.assertFalse(result.ok)

    def test_network_tool_respects_the_offline_policy(self) -> None:
        config = self.config()
        config.tools.allow_network = False
        context = ToolContext(
            config=config,
            registry=default_registry(),
            sandbox=build_sandbox(config, workspace=self.workspace),
        )
        tool_obj = default_registry().get("fetch_page")
        try:
            result = tool_obj.run({"url": "https://example.com"}, context=context)
        except PermissionDeniedError:
            return
        self.assertFalse(result.ok)

    def test_describe_reports_one_tool(self) -> None:
        described = default_registry().describe("calculate")
        self.assertTrue(described)
        self.assertIn("calculate", json.dumps(described))

    def test_prompt_block_lists_tools_compactly(self) -> None:
        block = default_registry().prompt_block()
        self.assertIn("read_file", block)
        self.assertLess(len(block), 200_000)


if __name__ == "__main__":
    unittest.main()
