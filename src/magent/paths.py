"""Filesystem layout: where MAgent keeps data, cache, config and docs.

Everything that needs a path goes through here, so the XDG defaults live in one
place and a test can point the whole application at a temporary directory by
setting ``MAGENT_HOME``.  :func:`repo_root` walks up from the package looking for
a checkout marker, which is what lets the CLI find the generated
``improvements.json`` next to the source without hard-coding a location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

APP_NAME = "magent"
"""Directory name under the XDG roots."""

HOME_ENV = "MAGENT_HOME"
"""Environment variable that relocates every MAgent directory at once."""

DATA_ENV = "MAGENT_DATA_DIR"
"""Environment variable for the data directory."""

CACHE_ENV = "MAGENT_CACHE_DIR"
"""Environment variable for the cache directory."""

CONFIG_ENV = "MAGENT_CONFIG_DIR"
"""Environment variable for the configuration directory."""

REPO_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git", "magent.toml")
"""Files that identify a checkout when walking upwards."""

IMPROVEMENTS_JSON = "improvements.json"
"""Machine-readable improvements manifest at the repository root."""

IMPROVEMENTS_MARKDOWN = "IMPROVEMENTS.md"
"""Human-readable improvements document at the repository root."""


def package_dir() -> Path:
    """Directory containing the installed ``magent`` package."""
    return Path(__file__).resolve().parent


def _home() -> Path | None:
    """The ``MAGENT_HOME`` override, when set."""
    value = os.environ.get(HOME_ENV, "").strip()
    return Path(value).expanduser() if value else None


def _xdg(kind: str, fallback: str) -> Path:
    """Resolve an XDG directory, honouring ``MAGENT_HOME``."""
    home = _home()
    if home is not None:
        return home / kind
    base = os.environ.get(f"XDG_{kind.upper()}_HOME", "").strip()
    return (Path(base).expanduser() if base else Path.home() / fallback) / APP_NAME


def data_dir() -> Path:
    """Where memories, sessions and plugins live."""
    override = os.environ.get(DATA_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return _xdg("data", ".local/share")


def cache_dir() -> Path:
    """Where response caches, embeddings and indexes live."""
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return _xdg("cache", ".cache")


def config_dir() -> Path:
    """Where configuration files live."""
    override = os.environ.get(CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return _xdg("config", ".config")


def state_dir() -> Path:
    """Where logs and rotating state live."""
    return _xdg("state", ".local/state")


def session_dir() -> Path:
    """Session records directory."""
    return data_dir() / "sessions"


def memory_dir() -> Path:
    """Long-term memory directory."""
    return data_dir() / "memory"


def plugin_dir() -> Path:
    """User plugin directory."""
    return data_dir() / "plugins"


def skill_dir() -> Path:
    """User skill directory."""
    return data_dir() / "skills"


def log_path() -> Path:
    """Default log file."""
    return state_dir() / "magent.log"


def trace_path() -> Path:
    """Default trace export file."""
    return state_dir() / "traces.jsonl"


def ensure_layout(*, with_config: bool = False) -> dict[str, Path]:
    """Create the standard directories; returns them by name.

    Args:
        with_config: also create the configuration directory
    """
    layout = {
        "data": data_dir(),
        "cache": cache_dir(),
        "sessions": session_dir(),
        "memory": memory_dir(),
        "plugins": plugin_dir(),
        "skills": skill_dir(),
        "state": state_dir(),
    }
    if with_config:
        layout["config"] = config_dir()
    for path in layout.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
    return layout


def repo_root(start: str | Path | None = None, *, markers: Iterable[str] = REPO_MARKERS) -> Path | None:
    """Walk upwards looking for a checkout, returning ``None`` when there is none."""
    current = Path(start).expanduser().resolve() if start else package_dir()
    if current.is_file():
        current = current.parent
    wanted = tuple(markers)
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in wanted):
            return candidate
    return None


def improvements_manifest(*, root: str | Path | None = None) -> Path:
    """Path of the machine-readable improvements manifest.

    Resolves to ``improvements.json`` in the checkout when MAgent is run from a
    source tree, and to the copy inside the installed package otherwise.
    """
    base = Path(root).expanduser() if root else repo_root()
    if base is not None and (base / IMPROVEMENTS_JSON).is_file():
        return base / IMPROVEMENTS_JSON
    packaged = package_dir() / "_data" / IMPROVEMENTS_JSON
    return packaged if packaged.is_file() else (base or package_dir()) / IMPROVEMENTS_JSON


def improvements_markdown(*, root: str | Path | None = None) -> Path:
    """Path of the human-readable improvements document."""
    base = Path(root).expanduser() if root else repo_root()
    if base is not None and (base / IMPROVEMENTS_MARKDOWN).is_file():
        return base / IMPROVEMENTS_MARKDOWN
    return (base or package_dir()) / IMPROVEMENTS_MARKDOWN


def bundled_docs_dir() -> Path:
    """Documentation shipped inside the package, used by ``magent docs``."""
    packaged = package_dir() / "_docs"
    if packaged.is_dir():
        return packaged
    root = repo_root()
    return (root / "docs") if root is not None and (root / "docs").is_dir() else packaged


def is_source_checkout() -> bool:
    """True when MAgent is running from a checkout rather than an install."""
    root = repo_root()
    return bool(root is not None and (root / "src" / "magent").is_dir())


def describe() -> dict[str, str]:
    """Describe the resolved layout, for ``magent doctor`` and bug reports."""
    root = repo_root()
    return {
        "package": str(package_dir()),
        "repo_root": str(root) if root else "",
        "source_checkout": str(is_source_checkout()),
        "data_dir": str(data_dir()),
        "cache_dir": str(cache_dir()),
        "config_dir": str(config_dir()),
        "state_dir": str(state_dir()),
        "sessions": str(session_dir()),
        "memory": str(memory_dir()),
        "plugins": str(plugin_dir()),
        "skills": str(skill_dir()),
        "improvements": str(improvements_manifest()),
        "docs": str(bundled_docs_dir()),
    }


__all__ = [
    "APP_NAME",
    "CACHE_ENV",
    "CONFIG_ENV",
    "DATA_ENV",
    "HOME_ENV",
    "IMPROVEMENTS_JSON",
    "IMPROVEMENTS_MARKDOWN",
    "REPO_MARKERS",
    "bundled_docs_dir",
    "cache_dir",
    "config_dir",
    "data_dir",
    "describe",
    "ensure_layout",
    "improvements_manifest",
    "improvements_markdown",
    "is_source_checkout",
    "log_path",
    "memory_dir",
    "package_dir",
    "plugin_dir",
    "repo_root",
    "session_dir",
    "skill_dir",
    "state_dir",
    "trace_path",
]
