"""Regression tests ported while merging the two independent implementations.

These defects were exposed by the complementary test suite: mutable config
clones, dropped extension tables, duplicated token-chunk boundaries, cross-run
event contamination, stale same-name plugins, and product-name casing.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from magent.agent import Agent
from magent.config import config_from_dict, default_config
from magent.llm import EchoBackend, split_by_tokens
from magent.plugins import PluginLoader
from tests.support import MAgentTestCase

ROOT = Path(__file__).resolve().parents[1]
"""Repository root used for naming checks."""


class ConfigRegressionTest(unittest.TestCase):
    """Derived configuration must not mutate or discard caller-owned policy."""

    def test_clone_is_deep_and_independent(self) -> None:
        """Changing nested sections/lists on a clone leaves the source intact."""
        source = default_config()
        source.tools.allow_domains.append("example.test")
        clone = source.clone()
        clone.agent.max_iterations = 2
        clone.tools.allow_domains.append("other.test")
        self.assertEqual(source.agent.max_iterations, 12)
        self.assertEqual(source.tools.allow_domains, ["example.test"])
        self.assertEqual(clone.tools.allow_domains, ["example.test", "other.test"])

    def test_unknown_top_level_table_is_preserved_for_plugins(self) -> None:
        """Extension-owned TOML survives under Config.extra."""
        config = config_from_dict({"my_plugin": {"endpoint": "local", "enabled": True}})
        self.assertEqual(config.extra["my_plugin"], {"endpoint": "local", "enabled": True})

    def test_clone_rejects_unknown_top_level_override(self) -> None:
        """A typo passed to clone is not silently attached or ignored."""
        from magent.errors import ConfigError

        with self.assertRaises(ConfigError):
            default_config().clone(not_a_section=True)


class TokenRegressionTest(unittest.TestCase):
    """LLM request splitting is a partition, unlike overlapping RAG chunks."""

    def test_split_does_not_repeat_boundary_words(self) -> None:
        """Joining chunks recovers every source word exactly once and in order."""
        text = " ".join(f"word{number}" for number in range(100))
        parts = split_by_tokens(text, 20)
        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(" ".join(parts).split()), text)


class AgentEventRegressionTest(MAgentTestCase):
    """A reusable Agent shares a bus but each result owns only its run trace."""

    def test_second_result_excludes_first_run_events(self) -> None:
        """Every event in a result has that result's run id."""
        agent = Agent(backend=EchoBackend(), workspace=self.workspace)
        first = agent.run("first")
        second = agent.run("second")
        self.assertTrue(first.events)
        self.assertTrue(second.events)
        self.assertTrue(all(event.run_id == first.run_id for event in first.events))
        self.assertTrue(all(event.run_id == second.run_id for event in second.events))
        self.assertFalse(
            {event.id for event in first.events} & {event.id for event in second.events}
        )


class PluginIdentityRegressionTest(MAgentTestCase):
    """A process-global plugin module name cannot point at two entry files."""

    def make_plugin(self, parent: Path, marker: str) -> Path:
        """Write a minimal same-name plugin under ``parent``."""
        root = parent / "weather"
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "weather",
                    "version": "1.0.0",
                    "entry": "plugin.py",
                    "description": marker,
                    "provides": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "plugin.py").write_text(f'"""{marker}."""\nMARKER = {marker!r}\n', encoding="utf-8")
        return root

    def test_second_path_is_refused_until_explicit_unload(self) -> None:
        """A checked second file never silently reuses or replaces stale code."""
        first_root = self.make_plugin(self.workspace / "first", "first")
        second_root = self.make_plugin(self.workspace / "second", "second")
        first = PluginLoader()
        record = first.load_directory(first_root)
        self.assertTrue(record.ok, record.error)
        second = PluginLoader()
        collision = second.load_directory(second_root)
        self.assertFalse(collision.ok)
        self.assertIn("already loaded", collision.error)
        first.unload("weather")
        retried = second.load_directory(second_root)
        self.assertTrue(retried.ok, retried.error)
        second.unload("weather")
        sys.modules.pop("magent_plugin_weather", None)


class ProductNamingRegressionTest(unittest.TestCase):
    """Human-facing identifiers use exactly MAgent, never Magent."""

    def test_distribution_display_name_is_magent(self) -> None:
        """Package metadata retains exact display casing."""
        self.assertIn('name = "MAgent"', (ROOT / "pyproject.toml").read_text())

    def test_no_wrong_case_in_source(self) -> None:
        """The source tree has no stale Magent symbols or prose."""
        offenders: list[str] = []
        wrong = "Ma" + "gent"
        for path in (ROOT / "src").rglob("*.py"):
            if wrong in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_server_class_is_named_magent(self) -> None:
        """The public server type follows the product spelling."""
        from magent.server import MAgentServer

        self.assertEqual(MAgentServer.__name__, "MAgentServer")


if __name__ == "__main__":
    unittest.main()
