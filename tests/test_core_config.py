"""Configuration loading, precedence, validation, and redaction tests.

MAgent merges defaults, TOML files, profiles, environment variables, and CLI
overrides. A wrong precedence rule is difficult to diagnose at runtime, so the
suite verifies each layer independently and verifies that derived storage paths
stay under the configured data directory.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from magent.config import (
    ENV_MAP,
    AgentConfig,
    Config,
    HttpServerConfig,
    apply_env,
    apply_overrides,
    config_from_dict,
    default_config,
    load_config,
    load_toml,
    write_example_config,
)
from magent.errors import ConfigFileNotFound, ConfigValidationError

from tests.support import MAgentTestCase


class DefaultTests(MAgentTestCase):
    """Defaults make MAgent usable without a file or an environment."""

    def test_default_config_validates(self) -> None:
        """The shipped defaults are internally consistent."""
        self.assertIs(default_config().validate().__class__, Config)

    def test_core_defaults(self) -> None:
        """The default loop is useful but bounded."""
        config = default_config()
        self.assertEqual(config.agent.loop_mode, "react")
        self.assertEqual(config.agent.max_iterations, 12)
        self.assertEqual(config.model.backend, "auto")
        self.assertTrue(config.model.fallback_to_echo)

    def test_server_defaults_are_loopback_only(self) -> None:
        """Starting the server with no flags does not expose it publicly."""
        server = default_config().http
        self.assertEqual(server.host, "127.0.0.1")
        self.assertEqual(server.port, 8700)
        self.assertTrue(server.serve_ui)

    def test_derived_paths_live_under_data_dir(self) -> None:
        """Session, memory, log, and plugins paths share the isolated root."""
        config = self.config()
        root = Path(config.data_dir).resolve()
        paths = [config.session.directory, config.memory.store_path,
                 config.tools.plugins_dir]
        for value in paths:
            self.assertTrue(Path(value).resolve().is_relative_to(root), value)

    def test_existing_explicit_paths_are_not_overwritten(self) -> None:
        """Path derivation only fills empty fields."""
        explicit = self.temp_dir("sessions-")
        config = config_from_dict({"data_dir": str(self._home / "other"),
                                   "session": {"directory": str(explicit)}}).validate()
        self.assertEqual(Path(config.session.directory), explicit)

    def test_sections_are_dataclasses(self) -> None:
        """Section lookup returns the typed section, not a raw dict."""
        config = default_config()
        self.assertIsInstance(config.section("agent"), AgentConfig)
        self.assertIsInstance(config.section("http"), HttpServerConfig)


class DictTests(MAgentTestCase):
    """Raw mappings become typed Config objects."""

    def test_config_from_dict_sets_known_values(self) -> None:
        """Known keys are converted into section dataclasses."""
        config = config_from_dict({"agent": {"max_iterations": 3},
                                   "http": {"port": 9911}})
        self.assertEqual(config.agent.max_iterations, 3)
        self.assertEqual(config.http.port, 9911)

    def test_unknown_top_level_tables_are_preserved(self) -> None:
        """Extensions can keep their own tables in Config.extra."""
        config = config_from_dict({"my_plugin": {"enabled": True}})
        self.assertEqual(config.extra["my_plugin"], {"enabled": True})

    def test_get_and_set_use_dotted_paths(self) -> None:
        """Dotted access is what CLI ``config get/set`` operates on."""
        config = default_config()
        config.set("agent.max_iterations", 7)
        self.assertEqual(config.get("agent.max_iterations"), 7)
        self.assertEqual(config.agent.max_iterations, 7)

    def test_get_returns_default_for_unknown_key(self) -> None:
        """Missing keys are queryable without raising."""
        self.assertEqual(default_config().get("not.there", "fallback"), "fallback")

    def test_clone_is_independent(self) -> None:
        """Mutating a clone does not mutate the source config."""
        source = default_config()
        clone = source.clone()
        clone.agent.max_iterations = 2
        self.assertEqual(source.agent.max_iterations, 12)
        self.assertEqual(clone.agent.max_iterations, 2)

    def test_to_dict_returns_plain_mappings(self) -> None:
        """Serialization removes dataclass wrappers."""
        payload = default_config().to_dict()
        self.assertIsInstance(payload["agent"], dict)
        self.assertEqual(payload["agent"]["loop_mode"], "react")

    def test_description_lists_every_section(self) -> None:
        """The CLI description has a row for each configured section."""
        keys = {row["key"] for row in default_config().describe()}
        self.assertTrue({"agent.loop_mode", "model.backend", "tools.workspace",
                         "http.host"}.issubset(keys))


class FileTests(MAgentTestCase):
    """TOML files and named profiles load predictably."""

    def write_config(self, text: str) -> Path:
        """Write one temporary config file."""
        path = self.temp_dir("cfg-") / "magent.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_explicit_toml(self) -> None:
        """Values from an explicit file override defaults."""
        path = self.write_config("[agent]\nmax_iterations = 4\n[http]\nport = 8811\n")
        config = load_config(path=path, use_env=False)
        self.assertEqual(config.agent.max_iterations, 4)
        self.assertEqual(config.http.port, 8811)
        self.assertIn(str(path), config.sources)

    def test_profile_overrides_the_base(self) -> None:
        """The requested profile deep-merges over base settings."""
        path = self.write_config(
            "[agent]\nmax_iterations = 10\nloop_mode = 'react'\n"
            "[profiles.fast.agent]\nmax_iterations = 2\n"
        )
        config = load_config(path=path, profile="fast", use_env=False)
        self.assertEqual(config.agent.max_iterations, 2)
        self.assertEqual(config.agent.loop_mode, "react")
        self.assertEqual(config.profile, "fast")

    def test_missing_optional_file_is_ignored(self) -> None:
        """required=False permits a path that is not present."""
        missing = self._home / "missing.toml"
        config = load_config(path=missing, required=False, use_env=False)
        self.assertEqual(config.agent.loop_mode, "react")

    def test_missing_required_file_raises(self) -> None:
        """An explicitly required file cannot silently disappear."""
        with self.assertRaises(ConfigFileNotFound):
            load_config(path=self._home / "missing.toml", required=True, use_env=False)

    def test_load_toml_rejects_invalid_syntax(self) -> None:
        """Malformed TOML becomes a ConfigValidationError."""
        path = self.write_config("[agent\nnot valid")
        from magent.errors import ConfigError

        with self.assertRaises(ConfigError):
            load_toml(path, required=True)

    def test_write_example_refuses_to_clobber(self) -> None:
        """The template writer is safe unless force=True."""
        path = self._home / "example.toml"
        write_example_config(path)
        original = path.read_text(encoding="utf-8")
        path.write_text("mine", encoding="utf-8")
        from magent.errors import ConfigError

        with self.assertRaises(ConfigError):
            write_example_config(path)
        write_example_config(path, force=True)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_generated_example_is_valid_toml(self) -> None:
        """The shipped example parses as a configuration file."""
        path = self._home / "example.toml"
        write_example_config(path)
        parsed = load_toml(path, required=True)
        self.assertIn("agent", parsed)
        self.assertIn("model", parsed)


class PrecedenceTests(MAgentTestCase):
    """Later layers win: file, profile, environment, then CLI overrides."""

    def write_config(self) -> Path:
        """Write a file whose value each later layer can replace."""
        path = self.temp_dir("precedence-") / "magent.toml"
        path.write_text(
            "[agent]\nmax_iterations = 20\n"
            "[profiles.fast.agent]\nmax_iterations = 15\n",
            encoding="utf-8",
        )
        return path

    def test_environment_beats_file_and_profile(self) -> None:
        """MAGENT_AGENT_MAX_ITERATIONS has higher precedence than TOML."""
        path = self.write_config()
        os.environ["MAGENT_MAX_ITERATIONS"] = "9"
        self.addCleanup(os.environ.pop, "MAGENT_MAX_ITERATIONS", None)
        config = load_config(path=path, profile="fast", use_env=True)
        self.assertEqual(config.agent.max_iterations, 9)

    def test_cli_override_beats_environment(self) -> None:
        """Explicit overrides are the final layer."""
        path = self.write_config()
        os.environ["MAGENT_MAX_ITERATIONS"] = "9"
        self.addCleanup(os.environ.pop, "MAGENT_MAX_ITERATIONS", None)
        config = load_config(path=path, use_env=True,
                             overrides={"agent.max_iterations": 3})
        self.assertEqual(config.agent.max_iterations, 3)

    def test_apply_overrides_reports_changed_keys(self) -> None:
        """The caller can audit what an override changed."""
        config = default_config()
        changed = apply_overrides(config, {"agent.max_iterations": 2,
                                           "http.port": 9999})
        self.assertEqual(set(changed), {"agent.max_iterations", "http.port"})

    def test_apply_env_parses_boolean_and_integer(self) -> None:
        """Environment strings become typed values."""
        config = default_config()
        changed = apply_env(config, {
            "MAGENT_ALLOW_NETWORK": "true",
            "MAGENT_HTTP_PORT": "9123",
        })
        self.assertTrue(config.tools.allow_network)
        self.assertEqual(config.http.port, 9123)
        self.assertIn("MAGENT_ALLOW_NETWORK", changed)

    def test_environment_map_is_broad_and_dotted(self) -> None:
        """The documented env surface covers all major sections."""
        self.assertGreaterEqual(len(ENV_MAP), 45)
        self.assertIn("MAGENT_BACKEND", ENV_MAP)
        for variable, target in ENV_MAP.items():
            section, field, caster = target
            self.assertTrue(variable.startswith("MAGENT_"))
            self.assertIn(section, {"server", "model", "generation", "agent", "memory",
                                    "tools", "guardrails", "cache", "output", "session",
                                    "observability", "http"})
            self.assertIsInstance(field, str)
            self.assertIn(caster, {str, int, float, bool, list})


class ValidationTests(MAgentTestCase):
    """Invalid settings fail early with a precise ConfigValidationError."""

    def assert_invalid(self, dotted: str, value: object) -> None:
        """Assert that one dotted override is rejected."""
        config = default_config()
        config.set(dotted, value)
        with self.assertRaises(ConfigValidationError, msg=f"{dotted}={value!r}"):
            config.validate()

    def test_iterations_must_be_positive(self) -> None:
        """An agent with no iteration cannot run."""
        self.assert_invalid("agent.max_iterations", 0)

    def test_loop_mode_is_enumerated(self) -> None:
        """Only react, plan, reflect, and single are accepted."""
        self.assert_invalid("agent.loop_mode", "plan_execute")

    def test_port_is_in_range(self) -> None:
        """TCP ports are 1..65535 (0 is a CLI-only ephemeral request)."""
        self.assert_invalid("http.port", 70000)

    def test_threads_must_be_positive(self) -> None:
        """The server cannot create a zero-worker pool."""
        self.assert_invalid("http.threads", 0)

    def test_valid_loop_modes(self) -> None:
        """Every documented loop mode validates."""
        for mode in ("react", "plan", "reflect", "single"):
            config = default_config()
            config.agent.loop_mode = mode
            self.assertIs(config.validate(), config)


class RedactionTests(MAgentTestCase):
    """Secrets are hidden by default but can be deliberately revealed."""

    def test_to_dict_redacts_api_key(self) -> None:
        """The normal serialized config never contains the server key."""
        config = default_config()
        config.http.api_key = "mag_super_secret_value"
        payload = config.to_dict()
        self.assertNotIn("mag_super_secret_value", str(payload))

    def test_reveal_is_opt_in(self) -> None:
        """redact=False is explicit and returns the real value."""
        config = default_config()
        config.http.api_key = "mag_super_secret_value"
        payload = config.to_dict(redact=False)
        self.assertEqual(payload["http"]["api_key"], "mag_super_secret_value")

    def test_toml_output_is_redacted(self) -> None:
        """Printing TOML for diagnostics does not leak credentials."""
        config = default_config()
        config.http.api_key = "mag_super_secret_value"
        self.assertNotIn("mag_super_secret_value", config.to_toml())


if __name__ == "__main__":
    unittest.main()
