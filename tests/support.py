"""Shared helpers for the MAgent test suite.

The point of this module is that no test has to reinvent isolation. Every test
that touches configuration, storage, or the filesystem goes through a helper
here, so the suite cannot read or write the developer's real home directory and
cannot be affected by whatever ``~/.config/magent/config.toml`` happens to say.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator

from magent.config import Config, load_config
from magent.testing import make_backend, offline_backend


@contextlib.contextmanager
def isolated_home() -> Iterator[Path]:
    """Point the XDG directories at a temporary home for the duration of a block.

    ``Config`` derives its data, session, memory, log, and plugin paths from
    ``XDG_DATA_HOME``, so redirecting that one variable is enough to keep a test
    run off the real filesystem.
    """
    root = Path(tempfile.mkdtemp(prefix="magent-test-"))
    saved = {name: os.environ.get(name) for name in
             ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "MAGENT_CONFIG")}
    try:
        os.environ["XDG_DATA_HOME"] = str(root / "data")
        os.environ["XDG_CONFIG_HOME"] = str(root / "config")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        os.environ.pop("MAGENT_CONFIG", None)
        yield root
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(root, ignore_errors=True)


def isolated_config(*, path: str | Path | None = None, profile: str = "",
                    flat: dict[str, Any] | None = None) -> Config:
    """Return a validated configuration rooted in a temporary directory.

    Args:
        path: configuration file to load instead of the discovered ones
        profile: ``[profiles.<name>]`` table to merge
        flat: dotted overrides, e.g. ``{"agent.max_iterations": 3}``

    ``load_config`` takes only these keyword arguments, so this helper does not
    invent others: passing a section dataclass here would fail at call time
    rather than at parse time.
    """
    root = Path(tempfile.mkdtemp(prefix="magent-cfg-"))
    saved = {name: os.environ.get(name) for name in
             ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "MAGENT_CONFIG")}
    os.environ["XDG_DATA_HOME"] = str(root)
    os.environ["XDG_CONFIG_HOME"] = str(root)
    os.environ.pop("MAGENT_CONFIG", None)
    try:
        config = load_config(path=path, profile=profile, use_env=False,
                             overrides=dict(flat) if flat else None)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    config.data_dir = root
    return config.validate()


@contextlib.contextmanager
def workspace(files: dict[str, str] | None = None) -> Iterator[Path]:
    """Create a temporary workspace, optionally seeded with files.

    Args:
        files: mapping of relative path to text content; parent directories are
            created as needed.
    """
    root = Path(tempfile.mkdtemp(prefix="magent-ws-"))
    try:
        for relative, content in (files or {}).items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class MAgentTestCase(unittest.TestCase):
    """Base class that isolates storage and cleans up after itself."""

    maxDiff = None
    """Show full diffs: truncated assertion output hides the actual mismatch."""

    def setUp(self) -> None:
        """Create a temporary home and point the XDG directories at it."""
        self._home = Path(tempfile.mkdtemp(prefix="magent-case-"))
        self._saved_env = {name: os.environ.get(name) for name in
                           ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                            "MAGENT_CONFIG")}
        os.environ["XDG_DATA_HOME"] = str(self._home / "data")
        os.environ["XDG_CONFIG_HOME"] = str(self._home / "config")
        os.environ["XDG_CACHE_HOME"] = str(self._home / "cache")
        os.environ.pop("MAGENT_CONFIG", None)
        self._tempdirs: list[Path] = []

    def tearDown(self) -> None:
        """Restore the environment and remove every temporary directory."""
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for directory in self._tempdirs:
            shutil.rmtree(directory, ignore_errors=True)
        shutil.rmtree(self._home, ignore_errors=True)

    def temp_dir(self, prefix: str = "dir-") -> Path:
        """Return a temporary directory that tearDown will remove."""
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(self._home)))
        self._tempdirs.append(path)
        return path

    def config(self, *, path: str | Path | None = None, profile: str = "",
               flat: dict[str, Any] | None = None) -> Config:
        """Return a validated configuration rooted in this test's home."""
        config = load_config(path=path, profile=profile, use_env=False,
                             overrides=dict(flat) if flat else None)
        config.data_dir = self._home / "data"
        return config.validate()

    def offline_agent(self, *, registry: Any = None, backend: Any = None,
                      config: Config | None = None) -> Any:
        """Return an agent wired to the echo backend, with no network access."""
        from magent.agent import Agent

        settings = config or self.config()
        return Agent(backend=backend or offline_backend(), config=settings.agent,
                     settings=settings, registry=registry)

    def tool_backend(self, tool_name: str, arguments: dict[str, Any] | None = None,
                     final_text: str = "done") -> Any:
        """Return a backend that asks for one tool call, then answers."""
        return make_backend(tool_name=tool_name, tool_arguments=arguments or {},
                            final_text=final_text)


__all__ = [
    "MAgentTestCase",
    "isolated_config",
    "isolated_home",
    "make_backend",
    "offline_backend",
    "workspace",
]
