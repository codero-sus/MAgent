"""Tests for presets, teams, handoffs, skills and plugins."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from magent.agents import (AgentPreset, Handoff, Team, TeamMember, build_handoff, build_preset_agent,
                           get_preset, handoff_block, list_presets, parse_handoff, preset_for_task,
                           preset_names, presets_from_config, register_preset, run_team)
from magent.plugins import (HookRegistry, PluginLoader, PluginManifest, load_plugins, manifest_for,
                            write_manifest)
from magent.skills import (BUILTIN_SKILLS, Skill, SkillLibrary, build_library, dump_skill,
                           get_library, load_skill, merge_skills, parse_skill)
from magent.testing import EchoBackend, ScriptedBackend
from magent.tools import default_registry

from tests.support import MagentTestCase


class PresetTest(MagentTestCase):
    """Presets are data and produce working agents."""

    def test_presets_are_registered(self) -> None:
        names = preset_names()
        for expected in ("coder", "reviewer", "researcher", "writer", "devops", "planner",
                         "tutor", "analyst", "librarian", "minimal"):
            self.assertIn(expected, names)

    def test_every_preset_declares_a_known_persona(self) -> None:
        from magent.prompts.personas import PERSONAS

        for preset in list_presets():
            self.assertIn(preset.persona, PERSONAS, preset.name)
            self.assertTrue(preset.description)

    def test_tool_filters_narrow_the_registry(self) -> None:
        coder = get_preset("coder").build_registry(default_registry())
        self.assertGreater(coder.count(), 0)
        self.assertLess(coder.count(), default_registry().count())
        self.assertTrue(all(item.spec.category in
                            {"files", "text", "search", "git", "python", "shell", "system"}
                            for item in coder.tools()))

    def test_minimal_preset_has_no_tools(self) -> None:
        self.assertEqual(get_preset("minimal").build_registry(default_registry()).count(), 0)

    def test_unknown_preset_suggests_alternatives(self) -> None:
        from magent.errors import ConfigError

        with self.assertRaises(ConfigError) as caught:
            get_preset("coderr")
        self.assertIn("coder", str(caught.exception.details.get("did_you_mean", [])))

    def test_preset_can_be_overridden(self) -> None:
        preset = get_preset("coder").merge(max_iterations=3)
        self.assertEqual(preset.max_iterations, 3)
        self.assertEqual(get_preset("coder").max_iterations, 14)

    def test_build_preset_agent_runs(self) -> None:
        agent = build_preset_agent("minimal", backend=EchoBackend())
        result = agent.run("say something")
        self.assertTrue(result.succeeded)

    def test_task_guessing_picks_a_sensible_preset(self) -> None:
        self.assertEqual(preset_for_task("please review this pull request").name, "reviewer")
        self.assertEqual(preset_for_task("fix the failing test in parser.py").name, "coder")
        self.assertEqual(preset_for_task("hello there").name, "general")

    def test_presets_from_config_overlay_a_builtin(self) -> None:
        config = self.config()
        config.extra = {"presets": {"quick": {"extends": "minimal", "max_iterations": 2}}}
        presets = presets_from_config(config)
        self.assertIn("quick", presets)
        self.assertEqual(presets["quick"].max_iterations, 2)

    def test_register_custom_preset_rejects_duplicates(self) -> None:
        from magent.errors import ConfigError

        with self.assertRaises(ConfigError):
            register_preset(AgentPreset(name="coder", description="duplicate"))


class TeamTest(MagentTestCase):
    """The five orchestration modes run and report."""

    def _members(self, count: int = 2) -> list[TeamMember]:
        return [TeamMember(name=f"m{index}", agent=build_preset_agent("minimal",
                                                                     backend=EchoBackend()))
                for index in range(count)]

    def test_sequential_mode_runs_every_member(self) -> None:
        team = Team(self._members(3), mode="sequential")
        result = team.run("summarise the repository")
        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.turns), 3)
        self.assertTrue(result.handoffs)

    def test_vote_mode_uses_a_judge(self) -> None:
        members = self._members(2)
        judge = TeamMember(name="judge", agent=build_preset_agent("minimal", backend=EchoBackend()),
                           weight=2)
        team = Team([*members, judge], mode="vote", judge="judge")
        result = team.run("what is MAgent?")
        self.assertTrue(result.succeeded)
        self.assertIn("judge", [turn.member for turn in result.turns])

    def test_debate_mode_records_rounds(self) -> None:
        members = self._members(2)
        judge = TeamMember(name="judge", agent=build_preset_agent("minimal", backend=EchoBackend()))
        team = Team([*members, judge], mode="debate", judge="judge", max_rounds=1, max_turns=8)
        result = team.run("is the design sound?")
        self.assertTrue(result.rounds or result.turns)
        self.assertTrue(result.succeeded)

    def test_supervisor_mode_routes_by_name(self) -> None:
        supervisor = TeamMember(name="sup", agent=build_preset_agent(
            "minimal", backend=ScriptedBackend(script=["next: worker", "DONE"])))
        worker = TeamMember(name="worker", agent=build_preset_agent("minimal",
                                                                   backend=EchoBackend()))
        team = Team([supervisor, worker], mode="supervisor", judge="sup", max_turns=6)
        result = team.run("do the thing")
        self.assertTrue(result.succeeded)
        self.assertIn("worker", [turn.member for turn in result.turns])

    def test_max_turns_is_respected(self) -> None:
        team = Team(self._members(2), mode="round_robin", max_rounds=5, max_turns=3)
        result = team.run("keep going")
        self.assertLessEqual(len(result.turns), 4)

    def test_team_needs_at_least_one_member(self) -> None:
        from magent.errors import AgentError

        with self.assertRaises(AgentError):
            Team([])

    def test_result_serialises_and_renders(self) -> None:
        team = Team(self._members(2), mode="sequential")
        result = team.run("do something")
        payload = result.to_dict()
        self.assertIn("turns", payload)
        self.assertIn("Final", result.transcript())
        self.assertTrue(result.member_tokens())
        self.assertIn("team[", result.summary_line())

    def test_run_team_helper(self) -> None:
        result = run_team(self._members(2), "quick task", mode="sequential")
        self.assertTrue(result.succeeded)


class HandoffTest(unittest.TestCase):
    """Handoffs parse, render and survive a bad model reply."""

    def test_parse_json_handoff(self) -> None:
        handoff = parse_handoff('{"task": "t", "summary": "s", "next_steps": ["a", "b"],'
                               ' "confidence": 0.6}', from_agent="a", to_agent="b")
        self.assertEqual(handoff.task, "t")
        self.assertEqual(handoff.next_steps, ["a", "b"])
        self.assertAlmostEqual(handoff.confidence, 0.6)

    def test_parse_prose_handoff(self) -> None:
        handoff = parse_handoff("I fixed the parser but the tests still fail.")
        self.assertIn("parser", handoff.summary)
        self.assertFalse(handoff.to_dict()["metadata"]["parsed"])

    def test_render_mentions_next_steps(self) -> None:
        handoff = Handoff(task="ship it", next_steps=["run tests"], confidence=0.5)
        rendered = handoff_block(handoff)
        self.assertIn("run tests", rendered)
        self.assertIn("50%", rendered)

    def test_build_from_result_records_files(self) -> None:
        from magent.agent import make_agent
        from magent.testing import ToolScriptBackend

        agent = make_agent(backend=ToolScriptBackend(script=[
            {"tool_calls": [{"name": "calculate", "arguments": {"expression": "1+1"}}]},
            {"text": "done"},
        ]))
        result = agent.run("add one and one")
        handoff = build_handoff(result, from_agent="dev", to_agent="reviewer")
        self.assertTrue(handoff.summary)
        self.assertEqual(handoff.from_agent, "dev")


class SkillTest(MagentTestCase):
    """Skills parse, match and merge."""

    def test_builtin_skills_are_shipped(self) -> None:
        self.assertGreaterEqual(len(BUILTIN_SKILLS), 10)
        names = {skill.name for skill in BUILTIN_SKILLS}
        self.assertIn("code-review", names)
        self.assertIn("debug-loop", names)

    def test_parse_front_matter_and_body(self) -> None:
        text = ("---\nname: demo\ndescription: A demo\ntags: [a, b]\ntriggers:\n  - demo\n"
                "priority: 80\n---\nDo the demo thing.\n")
        skill = parse_skill(text, source="demo.md")
        self.assertEqual(skill.name, "demo")
        self.assertEqual(skill.tags, ("a", "b"))
        self.assertEqual(skill.triggers, ("demo",))
        self.assertEqual(skill.priority, 80)
        self.assertIn("demo thing", skill.body)

    def test_matching_prefers_trigger_words(self) -> None:
        library = get_library()
        self.assertEqual(library.for_task("can you review my pull request")[0].name, "code-review")

    def test_no_match_for_unrelated_text(self) -> None:
        self.assertEqual(get_library().for_task("what is the weather in delhi"), [])

    def test_prompt_block_contains_instructions(self) -> None:
        block = get_library().prompt_block("debug this crash")
        self.assertIn("skill", block.lower())

    def test_library_loads_project_skills(self) -> None:
        skill_dir = self.workspace / ".magent" / "skills"
        skill_dir.mkdir(parents=True)
        (skill_dir / "house-style.md").write_text(
            "---\nname: house-style\ndescription: Our conventions\ntriggers: [house style]\n---\n"
            "Match the existing style.\n", encoding="utf-8")
        library = build_library(self.config(), workspace=self.workspace)
        self.assertIn("house-style", library.names())
        self.assertIn("Match the existing style", library.get("house-style").body)

    def test_unknown_skill_reports_known_names(self) -> None:
        from magent.errors import SkillNotFoundError

        with self.assertRaises(SkillNotFoundError) as caught:
            get_library().get("nope")
        self.assertTrue(caught.exception.details["known"])

    def test_round_trip_through_disk(self) -> None:
        skill = Skill(name="round-trip", description="d", body="Do it.")
        path = self.workspace / "round-trip.md"
        path.write_text(dump_skill(skill), encoding="utf-8")
        reloaded = load_skill(path)
        self.assertEqual(reloaded.name, "round-trip")
        self.assertIn("Do it.", reloaded.body)

    def test_merge_combines_bodies(self) -> None:
        merged = merge_skills([Skill(name="a", body="first"), Skill(name="b", body="second")],
                              name="both")
        self.assertIn("first", merged.body)
        self.assertIn("second", merged.body)


class PluginTest(MagentTestCase):
    """Plugins load, contribute and fail loudly but safely."""

    def _plugin_dir(self) -> Path:
        root = self.workspace / "plugins" / "acme"
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(json.dumps({
            "name": "acme", "version": "1.0.0", "entry": "acme.py",
            "description": "Test plugin", "provides": ["tools", "hooks", "skills"],
        }), encoding="utf-8")
        (root / "acme.py").write_text(
            '"""Test plugin."""\n'
            "from magent.skills import Skill\n"
            "from magent.tools import tool\n\n\n"
            '@tool(name="acme_ping", description="Ping the Acme service", category="text")\n'
            'def acme_ping(message: str = "ping") -> str:\n'
            '    """Return the message unchanged."""\n'
            "    return message\n\n\n"
            'def after_run(payload, **context):\n'
            '    """Note that a run finished."""\n'
            "    return None\n\n\n"
            'HOOKS = {"after_run": after_run}\n'
            'SKILLS = [Skill(name="acme-style", description="Acme style", body="Be brief.")]\n',
            encoding="utf-8")
        return root

    def test_manifest_is_inferred_when_absent(self) -> None:
        root = self.workspace / "plugins" / "bare"
        root.mkdir(parents=True)
        (root / "bare.py").write_text("PLUGIN = {}\n", encoding="utf-8")
        manifest = manifest_for(root)
        self.assertEqual(manifest.name, "bare")
        self.assertTrue(manifest.metadata.get("inferred"))

    def test_manifest_validation_reports_unknown_capabilities(self) -> None:
        manifest = PluginManifest.from_dict({"name": "x", "entry": "x.py",
                                            "provides": ["tools", "nope"]})
        self.assertIn("unknown capabilities", " ".join(manifest.validate()))

    def test_loader_registers_tools_hooks_and_skills(self) -> None:
        root = self._plugin_dir()
        registry = default_registry()
        before = registry.count()
        hooks = HookRegistry()
        loader = PluginLoader(registry=registry, hooks=hooks)
        record = loader.load_directory(root)
        self.assertTrue(record.ok, record.error)
        self.assertEqual(registry.count(), before + 1)
        self.assertIn("acme_ping", registry.names())
        self.assertTrue(hooks.handlers("after_run"))
        self.assertIn("acme-style", get_library().names())

    def test_missing_entry_is_reported_not_raised(self) -> None:
        root = self.workspace / "plugins" / "broken"
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(json.dumps({"name": "broken", "entry": "gone.py"}),
                                          encoding="utf-8")
        loader = PluginLoader()
        record = loader.load_directory(root)
        self.assertFalse(record.ok)
        self.assertIn("not found", record.error)

    def test_entry_escaping_the_plugin_directory_is_refused(self) -> None:
        root = self.workspace / "plugins" / "escaping"
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(json.dumps({"name": "escaping",
                                                      "entry": "../../evil.py"}),
                                          encoding="utf-8")
        loader = PluginLoader()
        record = loader.load_directory(root)
        self.assertFalse(record.ok)
        self.assertIn("escapes", record.error)

    def test_strict_mode_raises(self) -> None:
        root = self.workspace / "plugins" / "broken"
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(json.dumps({"name": "broken", "entry": "gone.py"}),
                                          encoding="utf-8")
        from magent.errors import PluginError

        with self.assertRaises(PluginError):
            PluginLoader(strict=True).load_directory(root)

    def test_load_plugins_uses_the_config_directory(self) -> None:
        root = self._plugin_dir().parent
        config = self.config()
        config.tools.plugins_dir = str(root)
        loader = load_plugins(config, registry=default_registry(), workspace=self.workspace)
        self.assertIn("acme", loader.names())

    def test_write_manifest_round_trips(self) -> None:
        target = self.workspace / "written"
        write_manifest(PluginManifest(name="written", version="0.1.0", entry="written.py"), target)
        manifest = manifest_for(target)
        self.assertEqual(manifest.name, "written")
        self.assertEqual(manifest.version, "0.1.0")

    def test_hooks_isolate_failures(self) -> None:
        hooks = HookRegistry()
        seen: list[str] = []
        hooks.register("before_run", lambda payload, **kw: (_ for _ in ()).throw(RuntimeError("x")),
                       name="broken")
        hooks.register("before_run", lambda payload, **kw: seen.append("ok"), name="good")
        hooks.apply("before_run", {})
        self.assertEqual(seen, ["ok"])
        self.assertEqual(len(hooks.errors), 1)


if __name__ == "__main__":
    unittest.main()
