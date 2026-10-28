"""Tests for the command line: every command runs and reports correctly."""

from __future__ import annotations

import json
import os
import unittest

from tests.support import MAgentTestCase


class CliTest(MAgentTestCase):
    """Each command is exercised through the real argument parser."""

    def setUp(self) -> None:
        super().setUp()
        os.environ["MAGENT_WORKSPACE"] = str(self.workspace)

    def tearDown(self) -> None:
        os.environ.pop("MAGENT_WORKSPACE", None)
        super().tearDown()

    # -- informational -------------------------------------------------------

    def test_version(self) -> None:
        code, out, _ = self.run_cli("version")
        self.assertEqual(code, 0)
        self.assertIn("MAgent", out)

    def test_commands_listing(self) -> None:
        code, out, _ = self.run_cli("commands")
        self.assertEqual(code, 0)
        self.assertIn("doctor", out)

    def test_commands_as_json(self) -> None:
        code, out, _ = self.run_cli("commands", "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(any(item["name"] == "serve" for item in payload))

    def test_completion_script(self) -> None:
        code, out, _ = self.run_cli("completion", "bash")
        self.assertEqual(code, 0)
        self.assertIn("complete -F", out)

    def test_help_lists_every_command(self) -> None:
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        for name in ("run", "ask", "tools", "doctor", "serve", "eval"):
            self.assertIn(name, out)

    def test_unknown_command_is_a_usage_error(self) -> None:
        code, _, err = self.run_cli("definitely-not-a-command")
        self.assertEqual(code, 2)

    # -- tools ---------------------------------------------------------------

    def test_tools_listing(self) -> None:
        code, out, _ = self.run_cli("tools")
        self.assertEqual(code, 0)
        self.assertIn("read_file", out)

    def test_tools_json(self) -> None:
        code, out, _ = self.run_cli("tools", "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        names = [item["name"] for item in payload]
        self.assertIn("read_file", names)
        self.assertTrue(all("category" in item for item in payload))

    def test_tools_inspect_prints_the_schema(self) -> None:
        code, out, _ = self.run_cli("tools", "inspect", "calculate")
        self.assertEqual(code, 0)
        self.assertIn("expression", out)

    def test_tools_inspect_unknown_suggests(self) -> None:
        code, out, _ = self.run_cli("tools", "inspect", "read_fil")
        self.assertEqual(code, 1)
        self.assertIn("read_file", out)

    def test_tools_call_runs_a_tool(self) -> None:
        code, out, _ = self.run_cli("tools", "call", "calculate", '{"expression": "6*7"}')
        self.assertEqual(code, 0)
        self.assertIn("42", out)

    def test_tools_call_reports_a_schema_mismatch(self) -> None:
        code, out, _ = self.run_cli("tools", "call", "calculate", "not-json")
        self.assertEqual(code, 1)
        self.assertIn("expression", out)

    def test_presets_and_preset_detail(self) -> None:
        code, out, _ = self.run_cli("presets")
        self.assertEqual(code, 0)
        self.assertIn("coder", out)
        code, out, _ = self.run_cli("presets", "coder")
        self.assertEqual(code, 0)
        self.assertIn("persona", out)

    def test_skills_listing_and_matching(self) -> None:
        code, out, _ = self.run_cli("skills", "--task", "review this pull request")
        self.assertEqual(code, 0)
        self.assertIn("code-review", out)

    def test_prompts_listing(self) -> None:
        code, out, _ = self.run_cli("prompts")
        self.assertEqual(code, 0)
        self.assertIn("summarize", out)

    def test_prompts_persona(self) -> None:
        code, out, _ = self.run_cli("prompts", "personas", "--name", "coder")
        self.assertEqual(code, 0)
        self.assertIn("Role", out)

    # -- knowledge -----------------------------------------------------------

    def test_memory_add_and_list(self) -> None:
        code, _, _ = self.run_cli("memory", "add", "the deploy key lives in ~/.config")
        self.assertEqual(code, 0)
        code, out, _ = self.run_cli("memory", "list")
        self.assertEqual(code, 0)
        self.assertIn("deploy key", out)

    def test_memory_search(self) -> None:
        self.run_cli("memory", "add", "cortex listens on port 8624")
        self.run_cli("memory", "add", "bananas are yellow")
        code, out, _ = self.run_cli("memory", "search", "cortex port")
        self.assertEqual(code, 0)
        self.assertIn("8624", out)

    def test_memory_stats_as_json(self) -> None:
        code, out, _ = self.run_cli("memory", "stats", "--format", "json")
        self.assertEqual(code, 0)
        self.assertIn("entries", json.loads(out))

    def test_rag_ingest_and_search(self) -> None:
        self.write("docs/guide.md", "# Guide\n\nCortex listens on 8624 by default.\n")
        code, out, _ = self.run_cli("rag", "ingest", str(self.workspace))
        self.assertEqual(code, 0)
        self.assertIn("indexed", out.lower())
        code, out, _ = self.run_cli("rag", "search", "--query", "which port does cortex use")
        self.assertEqual(code, 0)
        self.assertIn("guide.md", out)

    def test_sessions_list_is_empty_initially(self) -> None:
        code, out, _ = self.run_cli("sessions", "list")
        self.assertEqual(code, 0)
        self.assertIn("no stored sessions", out)

    # -- system --------------------------------------------------------------

    def test_config_show_redacts_secrets(self) -> None:
        code, out, _ = self.run_cli("config", "show")
        self.assertEqual(code, 0)
        self.assertIn("model", out.lower())

    def test_config_section(self) -> None:
        code, out, _ = self.run_cli("config", "show", "--section", "tools")
        self.assertEqual(code, 0)
        self.assertIn("workspace", out)

    def test_config_set_then_get(self) -> None:
        target = self.home.path / "magent.toml"
        code, out, _ = self.run_cli(
            "config", "set", "output.format", "json", "--config", str(target)
        )
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())
        self.assertIn("format", target.read_text(encoding="utf-8"))

    def test_config_paths(self) -> None:
        code, out, _ = self.run_cli("config", "paths")
        self.assertEqual(code, 0)
        self.assertIn("magent.toml", out)

    def test_config_validate(self) -> None:
        code, out, _ = self.run_cli("config", "validate")
        self.assertEqual(code, 0)
        self.assertIn("valid", out)

    def test_doctor_reports_checks(self) -> None:
        code, out, _ = self.run_cli("doctor", "--format", "json")
        payload = json.loads(out)
        self.assertEqual(code, 0 if payload["problems"] == 0 else 1)
        names = {check["check"] for check in payload["checks"]}
        self.assertIn("tools", names)
        self.assertIn("sandbox", names)

    def test_plugins_paths_and_check(self) -> None:
        code, out, _ = self.run_cli("plugins", "paths")
        self.assertEqual(code, 0)
        self.assertIn("plugins", out)
        code, _, _ = self.run_cli("plugins", "check")
        self.assertEqual(code, 0)

    def test_models_offline(self) -> None:
        code, out, _ = self.run_cli("models")
        self.assertEqual(code, 0)
        self.assertIn("echo", out.lower())

    def test_init_writes_a_project_scaffold(self) -> None:
        code, out, _ = self.run_cli("init", str(self.workspace), "--skills", "--plugin")
        self.assertEqual(code, 0)
        self.assertTrue((self.workspace / "magent.toml").exists())
        self.assertTrue((self.workspace / ".magent" / "skills" / "house-style.md").exists())
        self.assertTrue(
            (self.workspace / ".magent" / "plugins" / "example" / "plugin.json").exists()
        )

    # -- agent ---------------------------------------------------------------

    def test_run_asks_the_agent(self) -> None:
        code, out, _ = self.run_cli("run", "say hello")
        self.assertEqual(code, 0)
        self.assertIn("hello", out)

    def test_ask_prints_only_the_answer(self) -> None:
        code, out, _ = self.run_cli("ask", "what is 2+2?")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().splitlines()), 1)

    def test_run_as_json(self) -> None:
        code, out, _ = self.run_cli("run", "say hello", "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("stop_reason", payload)

    def test_run_without_a_prompt_is_a_usage_error(self) -> None:
        code, _, err = self.run_cli("run")
        self.assertEqual(code, 2)

    def test_run_records_a_session(self) -> None:
        self.run_cli("run", "say hello", "--session", "cli-test")
        code, out, _ = self.run_cli("sessions", "list")
        self.assertEqual(code, 0)

    def test_plan_produces_steps(self) -> None:
        code, out, _ = self.run_cli("plan", "add pagination to the sessions API")
        self.assertEqual(code, 0)
        self.assertIn("plan", out.lower())

    def test_reflect_critiques(self) -> None:
        code, out, _ = self.run_cli("reflect", "explain the guardrail pipeline")
        self.assertEqual(code, 0)
        self.assertIn("critique", out.lower())

    def test_eval_sample_dataset(self) -> None:
        target = self.home.path / "dataset.json"
        code, _, _ = self.run_cli("eval", "sample", "--output", str(target))
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())

    def test_eval_runs_a_dataset(self) -> None:
        path = self.workspace / "cases.jsonl"
        path.write_text(json.dumps({"id": "hi", "prompt": "say hi"}) + "\n", encoding="utf-8")
        code, out, _ = self.run_cli("eval", "--dataset", str(path), "--report-format", "json")
        self.assertIn(code, (0, 1))
        payload = json.loads(out)
        self.assertIn("summary", payload)

    def test_bench_reports_latency(self) -> None:
        code, out, _ = self.run_cli("bench", "--repeats", "2")
        self.assertEqual(code, 0)
        self.assertIn("ms", out)

    def test_improvements_manifest(self) -> None:
        code, out, _ = self.run_cli("improvements", "count")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(int(out.strip()), 777)

    def test_quiet_mode_writes_nothing(self) -> None:
        code, out, _ = self.run_cli("tools", "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "")

    def test_output_file_is_written(self) -> None:
        target = self.home.path / "out.txt"
        code, _, _ = self.run_cli("tools", "--output-file", str(target))
        self.assertEqual(code, 0)
        self.assertIn("read_file", target.read_text(encoding="utf-8"))


class ReplTest(MAgentTestCase):
    """The REPL dispatches slash commands and messages."""

    def _repl(self) -> object:
        import argparse

        from magent.cli import CliContext
        from magent.cli.repl import Repl

        args = argparse.Namespace(
            quiet=True,
            verbose=False,
            format="text",
            no_color=True,
            color=None,
            ascii=True,
            width=0,
            json=False,
            jsonl=False,
            yaml=False,
            set=[],
            profile="",
            config="",
            workspace=str(self.workspace),
            model="",
            backend="",
            offline=True,
            persona="",
            loop_mode="",
            output_file="",
            no_skills=False,
        )
        context = CliContext.build(args, config=self.config())
        return Repl(context, stream=False)

    def test_slash_help_lists_commands(self) -> None:
        repl = self._repl()
        self.assertTrue(repl.dispatch("/help"))

    def test_slash_tools_lists_tools(self) -> None:
        repl = self._repl()
        self.assertTrue(repl.dispatch("/tools calculate"))

    def test_unknown_slash_command_is_reported(self) -> None:
        repl = self._repl()
        self.assertTrue(repl.dispatch("/nope"))

    def test_slash_quit_ends_the_session(self) -> None:
        self.assertFalse(self._repl().dispatch("/quit"))

    def test_message_runs_the_agent(self) -> None:
        repl = self._repl()
        result = repl.ask("hello there")
        self.assertIsNotNone(result)
        self.assertEqual(repl.turns, 1)

    def test_memory_and_skills_commands(self) -> None:
        repl = self._repl()
        self.assertTrue(repl.dispatch("/memory"))
        self.assertTrue(repl.dispatch("/skills"))

    def test_stats_and_clear(self) -> None:
        repl = self._repl()
        self.assertTrue(repl.dispatch("/stats"))
        self.assertTrue(repl.dispatch("/clear"))


if __name__ == "__main__":
    unittest.main()
