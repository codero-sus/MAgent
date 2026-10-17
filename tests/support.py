"""Shared helpers for the test suite.

The suite runs entirely offline and never touches the user's real directories:
:class:`IsolatedHome` points ``MAGENT_HOME`` at a temporary directory for the
duration of a test, so memories, sessions, caches and logs land in a sandbox that
is deleted afterwards.  Tests that need a live Cortex LLMHoster are skipped
unless ``MAGENT_TEST_CORTEX_URL`` is set, which keeps `make test` green on a
machine with no models running.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORTEX_URL_ENV = "MAGENT_TEST_CORTEX_URL"
"""Set this to run the Cortex integration tests against a live host."""

WORKSPACE_ENV = "MAGENT_TEST_WORKSPACE"
"""Optional workspace override for the tests."""


@dataclass
class IsolatedHome:
    """A temporary ``MAGENT_HOME`` that also stubs the XDG variables."""

    path: Path
    previous: dict[str, str | None]

    @classmethod
    def create(cls, *, with_layout: bool = True) -> IsolatedHome:
        """Create the temporary home and point the environment at it."""
        path = Path(tempfile.mkdtemp(prefix="magent-test-home-"))
        keys = (
            "MAGENT_HOME",
            "MAGENT_DATA_DIR",
            "MAGENT_CACHE_DIR",
            "MAGENT_CONFIG_DIR",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
        )
        previous = {key: os.environ.get(key) for key in keys}
        os.environ["MAGENT_HOME"] = str(path)
        for key in keys[1:]:
            os.environ.pop(key, None)
        if with_layout:
            from magent.paths import ensure_layout

            ensure_layout(with_config=True)
        return cls(path=path, previous=previous)

    def teardown(self) -> None:
        """Restore the environment and delete the directory."""
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.path, ignore_errors=True)

    def file(self, relative: str, text: str) -> Path:
        """Write a file inside the temporary home."""
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target


class MAgentTestCase(unittest.TestCase):
    """Base class that isolates the environment for every test."""

    home: IsolatedHome
    workspace: Path

    def setUp(self) -> None:
        """Create a temporary home and workspace."""
        super().setUp()
        self.home = IsolatedHome.create()
        self.workspace = Path(tempfile.mkdtemp(prefix="magent-test-ws-"))

    def tearDown(self) -> None:
        """Remove temporary paths and process-global plugin import state."""
        import sys

        for name in [item for item in sys.modules if item.startswith("magent_plugin_")]:
            sys.modules.pop(name, None)
        shutil.rmtree(self.workspace, ignore_errors=True)
        self.home.teardown()
        super().tearDown()

    # -- fixtures ------------------------------------------------------------

    def write(self, relative: str, text: str = "") -> Path:
        """Write a workspace file."""
        target = self.workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def config(self, **overrides: Any) -> Any:
        """Build a configuration rooted in the temporary workspace."""
        from magent.config import default_config

        config = default_config()
        config.tools.workspace = str(self.workspace)
        config.data_dir = self.home.path / "data"
        config.cache_dir = self.home.path / "cache"
        config.memory.store_path = str(config.data_dir / "memory.db")
        config.session.directory = str(config.data_dir / "sessions")
        config.observability.log_file = ""
        config.observability.trace_file = ""
        config.guardrails.audit_log = ""
        for key, value in overrides.items():
            section, _, leaf = key.partition("__")
            if not leaf:
                setattr(config, key, value)
                continue
            target = getattr(config, section)
            setattr(target, leaf, value)
        return config

    def backend(self, *script: Any) -> Any:
        """Return an offline backend (echo by default, scripted when given lines)."""
        from magent.testing import EchoBackend, ScriptedBackend

        if script:
            return ScriptedBackend(script=list(script))
        return EchoBackend()

    def agent(self, **kwargs: Any) -> Any:
        """Build an agent wired to this test's workspace and offline backend."""
        from magent.agent import Agent

        config = kwargs.pop("config", None) or self.config()
        workspace = kwargs.pop("workspace", None) or self.workspace
        return Agent(
            backend=kwargs.pop("backend", None) or self.backend(),
            settings=config,
            workspace=workspace,
            **kwargs,
        )

    def registry(self) -> Any:
        """The default tool registry."""
        from magent.tools import default_registry

        return default_registry()

    @contextlib.contextmanager
    def capture(self) -> Iterator[io.StringIO]:
        """Capture stdout and stderr into one buffer."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            yield buffer

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        """Run the CLI offline and return ``(code, stdout, stderr)``."""
        from magent.cli import main

        out, err = io.StringIO(), io.StringIO()
        arguments = [*argv, "--offline"]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(arguments)
        except SystemExit as exc:  # argparse exits on usage errors
            return int(exc.code or 0), out.getvalue(), err.getvalue()
        return code, out.getvalue(), err.getvalue()


def skip_without_cortex() -> unittest.SkipTest:
    """Skip a test that needs a live Cortex LLMHoster."""
    url = os.environ.get(CORTEX_URL_ENV, "")
    if not url:
        raise unittest.SkipTest(f"set {CORTEX_URL_ENV} to run Cortex integration tests")
    return None


def cortex_url() -> str:
    """The Cortex URL to test against."""
    return os.environ.get(CORTEX_URL_ENV, "")


def temp_workspace(files: dict[str, str]) -> Any:
    """Context manager yielding a temporary directory with ``files`` written."""
    from magent.testing.fixtures import SampleCorpus
    from magent.testing.fixtures import temp_workspace as _temp

    return _temp(SampleCorpus(documents=dict(files)))


def assert_no_secrets(test: unittest.TestCase, text: str, *, secrets: list[str]) -> None:
    """Assert that none of the given secret strings appear in the text."""
    for secret in secrets:
        test.assertNotIn(secret, text, f"secret leaked into output: {secret[:8]}...")


__all__ = [
    "CORTEX_URL_ENV",
    "WORKSPACE_ENV",
    "IsolatedHome",
    "MAgentTestCase",
    "assert_no_secrets",
    "cortex_url",
    "skip_without_cortex",
    "temp_workspace",
]
