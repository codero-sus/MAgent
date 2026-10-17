"""Subprocess tests for MAgent's argparse grammar and operator output.

Direct parser tests would miss stdout/stderr, exit-code, module-entrypoint, and
configuration behavior. These launch ``python -m magent`` with temporary XDG
paths and no external services. They permanently cover global/group flag
positioning, the two newest command groups, product spelling, JSON output, and
script-friendly status codes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from magent.cli.main import build_parser, command_names

ROOT = Path(__file__).resolve().parents[1]
"""Repository root used as the subprocess working directory."""


class CLITestCase(unittest.TestCase):
    """Run CLI processes in one isolated temporary home."""

    def setUp(self) -> None:
        """Prepare environment variables for source-layout execution."""
        self.home = tempfile.TemporaryDirectory(prefix="magent-cli-tests-")
        self.env = dict(os.environ)
        self.env.update({
            "PYTHONPATH": str(ROOT / "src"),
            "XDG_DATA_HOME": str(Path(self.home.name) / "data"),
            "XDG_CONFIG_HOME": str(Path(self.home.name) / "config"),
            "XDG_CACHE_HOME": str(Path(self.home.name) / "cache"),
            "NO_COLOR": "1",
        })
        self.env.pop("MAGENT_CONFIG", None)

    def tearDown(self) -> None:
        """Remove temporary CLI state."""
        self.home.cleanup()

    def run_cli(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        """Run ``python -m magent`` and capture both output streams."""
        return subprocess.run(
            [sys.executable, "-m", "magent", *args], cwd=ROOT, env=self.env,
            text=True, capture_output=True, timeout=timeout, check=False,
        )


class GrammarTests(CLITestCase):
    """The parser exposes 14 groups and accepts flags at documented positions."""

    def test_command_catalogue(self) -> None:
        """Every registered command appears once in sorted introspection."""
        self.assertEqual(command_names(), [
            "chat", "config", "cortex", "doctor", "eval", "improvements",
            "memory", "plugins", "rag", "run", "serve", "sessions", "tools",
            "version",
        ])

    def test_global_help_lists_all_commands(self) -> None:
        """The module entrypoint renders the complete top-level grammar."""
        completed = self.run_cli("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("MAgent", completed.stdout)
        for name in command_names():
            self.assertIn(name, completed.stdout)

    def test_every_command_help_succeeds(self) -> None:
        """No command registration contains a conflicting argparse option."""
        for name in command_names():
            with self.subTest(command=name):
                completed = self.run_cli(name, "--help")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage: magent", completed.stdout)

    def test_global_format_before_command_is_preserved(self) -> None:
        """Regression: subparser defaults used to overwrite ``-f json``."""
        completed = self.run_cli("-f", "json", "tools", "categories")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertIn("categories", payload)
        self.assertEqual(sum(payload["categories"].values()), 113)

    def test_global_format_after_action_is_preserved(self) -> None:
        """Global flags remain valid at the deepest parser level."""
        completed = self.run_cli("tools", "categories", "-f", "json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(sum(json.loads(completed.stdout)["categories"].values()), 113)

    def test_parser_restores_shared_action_defaults(self) -> None:
        """One parse cannot leak inherited values into the next parse."""
        parser = build_parser()
        first = parser.parse_args(["-f", "json", "tools", "categories"])
        second = parser.parse_args(["tools", "categories"])
        self.assertEqual(first.format, "json")
        self.assertEqual(second.format, "")
        self.assertEqual(first.action, "categories")
        self.assertEqual(second.action, "categories")

    def test_group_flag_works_before_or_after_plugin_action(self) -> None:
        """Group flags are inherited into leaf action parsers."""
        plugin_parent = Path(self.home.name) / "plugins"
        scaffolded = self.run_cli("plugins", "scaffold", str(plugin_parent),
                                  "--name", "weather")
        self.assertEqual(scaffolded.returncode, 0, scaffolded.stderr)
        before = self.run_cli("plugins", "--dir", str(plugin_parent), "list", "-f", "json")
        after = self.run_cli("plugins", "list", "--dir", str(plugin_parent), "-f", "json")
        self.assertEqual(before.returncode, 0, before.stderr)
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertEqual(json.loads(before.stdout)["count"], 1)
        self.assertEqual(json.loads(after.stdout)["count"], 1)

    def test_leaf_help_has_one_options_section(self) -> None:
        """Inherited flags merge into argparse's existing section cleanly."""
        completed = self.run_cli("plugins", "list", "--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        section_lines = [line.strip() for line in completed.stdout.splitlines()]
        self.assertEqual(section_lines.count("options:"), 1)
        self.assertEqual(section_lines.count("global options:"), 1)
        self.assertIn("--dir", completed.stdout)
        self.assertIn("--loaded", completed.stdout)


class OutputAndExitTests(CLITestCase):
    """Commands return stable output and exit status in offline mode."""

    def test_version_uses_product_name(self) -> None:
        """Both version surfaces begin with exactly MAgent."""
        flag = self.run_cli("--version")
        command = self.run_cli("version")
        self.assertEqual(flag.returncode, 0)
        self.assertEqual(command.returncode, 0)
        self.assertTrue(flag.stdout.startswith("MAgent "))
        self.assertEqual(flag.stdout, command.stdout)
        self.assertNotIn("Magent", flag.stdout)

    def test_offline_before_command(self) -> None:
        """--offline survives parsing before the subcommand."""
        completed = self.run_cli("--offline", "run", "hello", "-q")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "echo:hello")

    def test_offline_after_command(self) -> None:
        """--offline also works after the subcommand."""
        completed = self.run_cli("run", "--offline", "hello", "-q")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "echo:hello")

    def test_unknown_command_is_usage_error(self) -> None:
        """argparse reserves exit 2 for an invalid command."""
        completed = self.run_cli("not-a-command")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)

    def test_bad_run_mode_returns_runtime_error(self) -> None:
        """A semantically invalid loop mode reaches the typed agent error path."""
        completed = self.run_cli("--offline", "run", "--mode", "plan_execute", "hello")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)

    def test_doctor_offline_names_magent(self) -> None:
        """Doctor reports MAgent correctly even when Cortex is unavailable."""
        completed = self.run_cli("doctor", "--offline")
        self.assertIn(completed.returncode, {0, 1})
        text = completed.stdout + completed.stderr
        self.assertIn("package MAgent", text)
        self.assertNotIn("package Magent", text)
        self.assertIn("113 tools", text)


class CommandSurfaceTests(CLITestCase):
    """Serve, plugins, config, and improvements have useful no-network modes."""

    def test_serve_routes_json(self) -> None:
        """Route introspection does not bind a socket and reports all 31 routes."""
        completed = self.run_cli("serve", "--routes", "-f", "json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["count"], 31)
        run_route = next(row for row in payload["routes"]
                         if row["method"] == "POST" and row["path"] == "/api/run")
        self.assertEqual(run_route["name"], "api.run")
        self.assertTrue(run_route["auth"])
        self.assertTrue(run_route["body"])

    def test_serve_dry_run_offline(self) -> None:
        """Dry-run resolves config/backend without opening a listening port."""
        completed = self.run_cli("serve", "--dry-run", "--offline", "-f", "json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["backend"], "echo")
        self.assertEqual(payload["routes"], 31)
        self.assertEqual(payload["host"], "127.0.0.1")

    def test_plugin_permissions_and_hooks(self) -> None:
        """Static policy references work without importing plugins."""
        permissions = self.run_cli("plugins", "permissions", "-f", "json")
        hooks = self.run_cli("plugins", "hooks", "-f", "json")
        self.assertEqual(permissions.returncode, 0, permissions.stderr)
        self.assertEqual(hooks.returncode, 0, hooks.stderr)
        self.assertEqual(len(json.loads(permissions.stdout)["permissions"]), 10)
        self.assertEqual(len(json.loads(hooks.stdout)["hooks"]), 10)

    def test_config_init_and_validate(self) -> None:
        """The CLI writes a parseable complete TOML file."""
        path = Path(self.home.name) / "magent.toml"
        initialized = self.run_cli("config", "init", "--file", str(path))
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        self.assertTrue(path.is_file())
        validated = self.run_cli("--config", str(path), "config", "validate")
        self.assertEqual(validated.returncode, 0, validated.stderr)
        self.assertIn("valid", validated.stdout.lower())

    def test_improvements_verify(self) -> None:
        """The CLI wrapper resolves the current generated manifest."""
        completed = self.run_cli("improvements", "--verify", "-f", "json", timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["meets_target"])
        self.assertGreaterEqual(payload["total"], 777)
        self.assertTrue(payload["verification"]["ok"])
        self.assertEqual(payload["verification"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
