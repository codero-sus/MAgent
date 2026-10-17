"""Tests for manifest-declared plugins, permissions, loading, and hooks.

Plugins are untrusted Python, not a security boundary. MAgent nevertheless makes
loading explicit: a manifest declares its entry point, tools, hooks, checksum,
and permissions; sensitive permissions require operator approval; path traversal
and undeclared exports are refused before or during registration.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from magent.errors import (PluginError, PluginManifestError, PluginNotFoundError,
                           PluginSecurityError)
from magent.plugins import (
    HOOK_NAMES,
    MANIFEST_FILENAME,
    PERMISSIONS,
    SENSITIVE_PERMISSIONS,
    HookRegistry,
    PluginLoader,
    PluginManifest,
    describe_permissions,
    example_manifest,
    hook,
    inspect_plugin,
    known_hooks,
    register_decorated,
    scaffold,
    validate_many,
    write_manifest_template,
)
from magent.tools import ToolContext, ToolRegistry
from magent.types import ToolCall

from tests.support import MAgentTestCase


class ManifestTests(MAgentTestCase):
    """PluginManifest validates identity, paths, declarations, and checksums."""

    def test_example_manifest_is_valid(self) -> None:
        """The programmatic template passes its own validator."""
        manifest = example_manifest("weather")
        self.assertIs(manifest.validate(), manifest)
        self.assertEqual(manifest.name, "weather")
        self.assertEqual(manifest.manifest_version, 1)
        self.assertIn("tools", manifest.permissions)

    def test_round_trip_json_file(self) -> None:
        """save/load preserves every public declaration."""
        root = self.temp_dir("manifest-")
        (root / "plugin.py").write_text("TOOLS = []\n", encoding="utf-8")
        original = example_manifest("demo")
        original.path = str(root)
        original.save(root)
        restored = PluginManifest.load(root)
        self.assertEqual(restored.name, "demo")
        self.assertEqual(restored.tools, original.tools)
        self.assertEqual(restored.permissions, original.permissions)
        self.assertEqual(Path(restored.path), root)

    def test_unknown_permission_is_rejected(self) -> None:
        """A typo cannot silently bypass permission policy."""
        manifest = example_manifest("demo")
        manifest.permissions.append("network.everywhere")
        with self.assertRaises(PluginManifestError):
            manifest.validate()

    def test_unknown_hook_is_rejected(self) -> None:
        """Only lifecycle points implemented by the core can be declared."""
        manifest = example_manifest("demo")
        manifest.hooks.append("during_magic")
        with self.assertRaises(PluginManifestError):
            manifest.validate()

    def test_unsafe_name_is_rejected(self) -> None:
        """Plugin names are identifiers, not filesystem paths."""
        manifest = example_manifest("safe")
        manifest.name = "../../escape"
        with self.assertRaises(PluginManifestError):
            manifest.validate()

    def test_entrypoint_traversal_is_rejected(self) -> None:
        """The entry module must stay inside the plugin directory."""
        manifest = example_manifest("demo")
        manifest.path = str(self.temp_dir("escape-"))
        manifest.entrypoint = "../../evil.py"
        with self.assertRaises(PluginManifestError):
            manifest.validate(directory=manifest.path)

    def test_sensitive_permissions_are_derived(self) -> None:
        """The manifest exposes only the sensitive subset it requested."""
        manifest = example_manifest("demo")
        manifest.permissions = ["tools", "network", "filesystem.write"]
        self.assertEqual(set(manifest.sensitive_permissions),
                         {"network", "filesystem.write"})

    def test_checksum_detects_modified_entrypoint(self) -> None:
        """A checksum pinned in the manifest fails after code changes."""
        root = self.temp_dir("checksum-")
        entry = root / "plugin.py"
        entry.write_text("value = 1\n", encoding="utf-8")
        manifest = example_manifest("demo")
        manifest.path = str(root)
        manifest.checksum = manifest.compute_checksum()
        self.assertTrue(manifest.verify_checksum())
        entry.write_text("value = 2\n", encoding="utf-8")
        self.assertFalse(manifest.verify_checksum())

    def test_validate_many_reports_by_name(self) -> None:
        """Bulk validation records bad manifests without hiding good ones."""
        good = example_manifest("good")
        bad = example_manifest("bad")
        bad.permissions.append("unknown")
        problems = validate_many([good, bad])
        self.assertNotIn("good", problems)
        self.assertIn("bad", problems)
        self.assertTrue(problems["bad"])

    def test_describe_is_human_readable(self) -> None:
        """The manifest summary includes identity and declared permissions."""
        text = example_manifest("demo").describe()
        self.assertIn("demo", text)
        self.assertIn("permissions", text)


class ScaffoldTests(MAgentTestCase):
    """The scaffold creates a plugin that can actually load and run."""

    def test_scaffold_creates_expected_files(self) -> None:
        """The destination is parent/name, with plugin.json and plugin.py."""
        parent = self.temp_dir("scaffold-")
        target = scaffold(parent, "weather")
        self.assertEqual(target, parent / "weather")
        self.assertTrue((target / "plugin.json").is_file())
        self.assertTrue((target / "plugin.py").is_file())

    def test_scaffold_product_name_is_correct(self) -> None:
        """Generated prose says MAgent, not Magent."""
        target = scaffold(self.temp_dir("name-"), "weather")
        text = (target / "plugin.py").read_text(encoding="utf-8")
        self.assertIn("MAgent", text)
        self.assertNotIn("Magent", text)

    def test_scaffold_refuses_to_overwrite(self) -> None:
        """Existing plugin code is safe unless force=True."""
        parent = self.temp_dir("overwrite-")
        target = scaffold(parent, "demo")
        (target / "plugin.py").write_text("mine = True\n", encoding="utf-8")
        with self.assertRaises(PluginError):
            scaffold(parent, "demo")
        scaffold(parent, "demo", force=True)
        self.assertNotEqual((target / "plugin.py").read_text(), "mine = True\n")

    def test_inspect_scaffold(self) -> None:
        """Static inspection reports entrypoint, declarations, and checksum."""
        target = scaffold(self.temp_dir("inspect-"), "demo")
        details = inspect_plugin(target)
        self.assertEqual(details["name"], "demo")
        self.assertTrue(details["entrypoint_exists"])
        self.assertEqual(details["declared_tools"], ["demo_hello"])
        self.assertEqual(details["declared_hooks"], ["before_run"])
        self.assertTrue(details["checksum_ok"])

    def test_write_manifest_template_uses_plugin_json(self) -> None:
        """The helper writes the real manifest filename and valid JSON."""
        root = self.temp_dir("template-")
        path = write_manifest_template(root / MANIFEST_FILENAME, "demo")
        self.assertEqual(path.name, MANIFEST_FILENAME)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], "demo")


class PermissionTests(MAgentTestCase):
    """Loader policy handles allow/deny lists and sensitive capabilities."""

    def manifest(self, permissions: list[str] | None = None) -> PluginManifest:
        """Create one real scaffold and return its manifest."""
        target = scaffold(self.temp_dir("policy-") / "plugins", "demo")
        manifest = PluginManifest.load(target)
        if permissions is not None:
            manifest.permissions = permissions
        return manifest

    def test_all_sensitive_permissions_are_documented(self) -> None:
        """Every sensitive permission is known and has explanatory prose."""
        descriptions = describe_permissions()
        self.assertEqual(set(descriptions), set(PERMISSIONS))
        self.assertTrue(SENSITIVE_PERMISSIONS.issubset(PERMISSIONS))
        for permission in SENSITIVE_PERMISSIONS:
            self.assertTrue(descriptions[permission].strip())

    def test_safe_plugin_is_allowed(self) -> None:
        """The normal tools-only scaffold needs no special approval."""
        self.assertEqual(PluginLoader().check_permission(self.manifest()), "")

    def test_sensitive_permission_is_refused(self) -> None:
        """network does not load merely because it was declared."""
        reason = PluginLoader().check_permission(self.manifest(["tools", "network"]))
        self.assertIn("sensitive", reason)
        self.assertIn("network", reason)

    def test_sensitive_permission_can_be_approved(self) -> None:
        """The operator can narrowly approve one sensitive permission."""
        manifest = self.manifest(["tools", "network"])
        reason = PluginLoader(allow_sensitive=["network"]).check_permission(manifest)
        self.assertEqual(reason, "")

    def test_deny_list_contains_plugin_names(self) -> None:
        """deny blocks by plugin name, not by permission name."""
        reason = PluginLoader(deny=["demo"]).check_permission(self.manifest())
        self.assertIn("deny list", reason)

    def test_allow_list_is_exclusive(self) -> None:
        """A configured allow list skips every other name."""
        reason = PluginLoader(allow=["other"]).check_permission(self.manifest())
        self.assertIn("allow list", reason)

    def test_strict_policy_raises(self) -> None:
        """Strict mode converts a refusal into PluginSecurityError."""
        with self.assertRaises(PluginSecurityError):
            PluginLoader(deny=["demo"], strict=True).check_permission(self.manifest())


class LoaderTests(MAgentTestCase):
    """Discovery imports permitted plugins and registers declared exports."""

    def setUp(self) -> None:
        super().setUp()
        # Plugin modules are process-global; each test owns this name.
        sys.modules.pop("magent_plugin_weather", None)
        self.parent = self.temp_dir("plugins-")
        self.target = scaffold(self.parent, "weather")
        self.registry = ToolRegistry()
        self.hooks = HookRegistry()
        self.loader = PluginLoader([self.parent], registry=self.registry, hooks=self.hooks)

    def tearDown(self) -> None:
        """Remove process-global plugin import state before deleting files."""
        sys.modules.pop("magent_plugin_weather", None)
        super().tearDown()

    def test_discover_is_static_and_stable(self) -> None:
        """Discovery validates manifests without importing plugin.py."""
        import sys

        manifests = self.loader.discover()
        self.assertEqual([manifest.name for manifest in manifests], ["weather"])
        self.assertNotIn("magent_plugin_weather", sys.modules)

    def test_load_all_registers_tool_and_hook(self) -> None:
        """The generated plugin contributes one real tool and one hook."""
        report = self.loader.load_all()
        self.assertEqual(len(report.loaded), 1)
        self.assertEqual(len(report.failed), 0)
        self.assertEqual(report.tool_count, 1)
        self.assertEqual(self.registry.count(), 1)
        self.assertEqual(self.hooks.count("before_run"), 1)
        result = self.registry.dispatch(
            ToolCall(id="c", name="weather_hello", arguments={"name": "Arena"}),
            ToolContext(workspace=self._home),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello, Arena")

    def test_load_report_lookup_and_serialization(self) -> None:
        """Reports expose loaded/skipped/failed partitions and names."""
        report = self.loader.load_all()
        self.assertEqual(report.names(), ["weather"])
        self.assertIsNotNone(report.by_name("weather"))
        self.assertIsNone(report.by_name("missing"))
        payload = report.to_dict()
        self.assertEqual(payload["loaded"], 1)
        self.assertEqual(payload["failed"], 0)
        self.assertIn("weather", report.render())

    def test_get_requires_prior_loaded_name(self) -> None:
        """get raises a specific error before and after unknown lookup."""
        with self.assertRaises(PluginNotFoundError):
            self.loader.get("weather")
        self.loader.load_all()
        self.assertEqual(self.loader.get("weather").name, "weather")
        with self.assertRaises(PluginNotFoundError):
            self.loader.get("missing")

    def test_unload_removes_module_and_hooks(self) -> None:
        """unload drops import state and hook registrations."""
        self.loader.load_all()
        self.assertTrue(self.loader.unload("weather"))
        self.assertEqual(self.hooks.count(), 0)
        self.assertFalse(self.loader.unload("weather"))

    def test_disabled_manifest_is_skipped(self) -> None:
        """enabled=false is visible in the report and never imported."""
        payload = json.loads((self.target / "plugin.json").read_text())
        payload["enabled"] = False
        (self.target / "plugin.json").write_text(json.dumps(payload))
        report = self.loader.load_all()
        self.assertEqual(len(report.loaded), 0)
        self.assertEqual(len(report.skipped), 1)
        self.assertEqual(report.skipped[0].skipped, "disabled in its manifest")

    def test_broken_plugin_is_recorded_not_raised(self) -> None:
        """One syntax error does not abort a non-strict load pass."""
        (self.target / "plugin.py").write_text("this is not python !!!", encoding="utf-8")
        report = self.loader.load_all()
        self.assertEqual(len(report.failed), 1)
        self.assertIn("weather", report.errors)
        self.assertIn("SyntaxError", report.errors["weather"])

    def test_strict_broken_plugin_raises(self) -> None:
        """Strict loader surfaces import failures to CI or operators."""
        (self.target / "plugin.py").write_text("raise RuntimeError('broken')\n")
        strict = PluginLoader([self.parent], registry=self.registry, strict=True)
        with self.assertRaises(PluginError) as caught:
            strict.load_all()
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_with_policy_is_independent_and_shares_registries(self) -> None:
        """A derived policy does not mutate the base loader."""
        extra = self.temp_dir("extra-plugins-")
        derived = self.loader.with_policy(deny=["weather"], directories=[extra],
                                          verify_checksums=False, strict=True)
        self.assertEqual(self.loader.deny, set())
        self.assertEqual(derived.deny, {"weather"})
        self.assertIn(extra, derived.directories)
        self.assertIs(derived.registry, self.registry)
        self.assertIs(derived.hooks, self.hooks)
        self.assertFalse(derived.verify_checksums)
        self.assertTrue(derived.strict)

    def test_same_name_at_a_second_path_is_refused(self) -> None:
        """A cached module cannot masquerade as newly inspected plugin code."""
        self.loader.load_all()
        second_parent = self.temp_dir("other-plugins-")
        scaffold(second_parent, "weather")
        second = PluginLoader([second_parent], registry=ToolRegistry())
        report = second.load_all()
        self.assertEqual(len(report.failed), 1)
        self.assertIn("already loaded", report.failed[0].error)
        self.loader.unload("weather")

    def test_stats_reflect_latest_report(self) -> None:
        """Loader stats are operator-facing and JSON-shaped."""
        self.loader.load_all()
        stats = self.loader.stats()
        self.assertEqual(stats["loaded"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertIn(str(self.parent), stats["directories"])


class HookTests(unittest.TestCase):
    """Hook chains are ordered, transformable, vetoable, and isolated."""

    def test_known_hooks_match_manifest_constant(self) -> None:
        """The introspection helper exposes all ten lifecycle hooks."""
        self.assertEqual(tuple(known_hooks()), tuple(HOOK_NAMES))
        self.assertEqual(len(HOOK_NAMES), 10)

    def test_priority_order(self) -> None:
        """Lower numbers run first and insertion order breaks ties."""
        registry = HookRegistry()
        order: list[str] = []
        registry.register("after_run", lambda payload: order.append("late"), priority=200)
        registry.register("after_run", lambda payload: order.append("early"), priority=1)
        result = registry.emit("after_run", {})
        self.assertEqual(order, ["early", "late"])
        self.assertEqual(result.ran, 2)

    def test_transform_threads_values(self) -> None:
        """Each build_prompt hook receives the prior hook's return value."""
        registry = HookRegistry()
        registry.register("build_prompt", lambda value: value + " one")
        registry.register("build_prompt", lambda value: value + " two")
        self.assertEqual(registry.transform("build_prompt", "start"),
                         "start one two")

    def test_veto_stops_chain(self) -> None:
        """False from before_run blocks and later hooks do not run."""
        registry = HookRegistry()
        seen: list[str] = []
        registry.register("before_run", lambda payload: False)
        registry.register("before_run", lambda payload: seen.append("too late"))
        result = registry.emit("before_run", {"prompt": "x"})
        self.assertTrue(result.blocked)
        self.assertEqual(seen, [])

    def test_once_hook_fires_once(self) -> None:
        """One-shot registrations disable themselves after their first call."""
        registry = HookRegistry()
        seen: list[int] = []
        registry.register("startup", lambda payload: seen.append(1), once=True)
        registry.emit("startup")
        registry.emit("startup")
        self.assertEqual(seen, [1])

    def test_failure_is_recorded_and_chain_continues(self) -> None:
        """A bad plugin does not suppress healthy hooks in non-strict mode."""
        registry = HookRegistry()
        seen: list[str] = []
        registry.register("after_run", lambda payload: (_ for _ in ()).throw(ValueError("bad")),
                          plugin="broken")
        registry.register("after_run", lambda payload: seen.append("ok"), plugin="good")
        result = registry.emit("after_run", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.failed, 1)
        self.assertEqual(seen, ["ok"])
        self.assertIn("broken", result.errors[0])

    def test_strict_failure_raises(self) -> None:
        """Strict mode is useful for tests and plugin CI."""
        registry = HookRegistry(strict=True)
        registry.register("after_run", lambda payload: 1 / 0)
        with self.assertRaises(ZeroDivisionError):
            registry.emit("after_run", {})

    def test_disable_and_unregister_plugin(self) -> None:
        """A plugin's hooks can be paused or removed as a group."""
        registry = HookRegistry()
        registration = registry.register("after_run", lambda payload: None, plugin="demo")
        self.assertEqual(registry.disable("after_run", plugin="demo"), 1)
        self.assertFalse(registration.enabled)
        self.assertEqual(registry.unregister_plugin("demo"), 1)
        self.assertEqual(registry.count(), 0)

    def test_decorator_registration(self) -> None:
        """@hook metadata can be collected from an imported module."""
        @hook("before_run", plugin="demo", priority=5)
        def before(payload: object) -> None:
            return None

        module = SimpleNamespace(before=before)
        registry = HookRegistry()
        count = register_decorated(module, registry, plugin="demo")
        self.assertEqual(count, 1)
        self.assertEqual(registry.hooks("before_run")[0].priority, 5)
        self.assertEqual(registry.hooks("before_run")[0].plugin, "demo")

    def test_unknown_emit_is_reported_not_raised(self) -> None:
        """Emitting an unknown lifecycle name yields an error result."""
        result = HookRegistry().emit("not_a_hook", {})
        self.assertFalse(result.ok)
        self.assertIn("unknown hook", result.errors[0])

    def test_stats_and_description(self) -> None:
        """Hook telemetry reports registrations and executions."""
        registry = HookRegistry()
        registry.register("after_run", lambda payload: None, plugin="demo")
        registry.emit("after_run", {})
        stats = registry.stats()
        self.assertEqual(stats["registered"], 1)
        self.assertEqual(stats["executed"], 1)
        self.assertIn("1 hooks registered", registry.describe())


if __name__ == "__main__":
    unittest.main()
