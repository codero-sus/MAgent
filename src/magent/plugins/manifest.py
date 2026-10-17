"""Plugin manifests: metadata and validation before any code is imported.

A plugin directory contains ``plugin.json`` (or a ``PLUGIN`` dict in its entry
module) describing what it provides.  Reading the manifest first means MAgent can
refuse a plugin that targets a newer version, declares no entry point, or asks
for capabilities the configuration forbids - *before* executing its code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .._version import __version__
from ..errors import PluginError
from ..utils.files import read_json
from ..utils.hashing import file_sha256

MANIFEST_NAMES: tuple[str, ...] = ("plugin.json", "magent-plugin.json", "manifest.json")
"""File names recognised as a plugin manifest."""

HOOK_POINTS: tuple[str, ...] = (
    "before_run",
    "after_run",
    "before_prompt",
    "after_prompt",
    "before_tool",
    "after_tool",
    "on_error",
    "on_event",
    "startup",
    "shutdown",
)
"""Hook points a plugin may register for."""

CAPABILITIES: tuple[str, ...] = ("tools", "hooks", "personas", "templates", "skills",
                                 "backends", "metrics")
"""Things a plugin is allowed to contribute."""


@dataclass(slots=True)
class PluginManifest:
    """Declared metadata for one plugin."""

    name: str
    version: str = "0.0.0"
    entry: str = ""
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    provides: tuple[str, ...] = ("tools",)
    hooks: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    min_magent: str = ""
    max_magent: str = ""
    enabled: bool = True
    checksum: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise tuples and default the entry point from the name."""
        self.name = str(self.name or "").strip()
        self.provides = _as_tuple(self.provides)
        self.hooks = _as_tuple(self.hooks)
        self.tools = _as_tuple(self.tools)
        self.requires = _as_tuple(self.requires)
        if not self.entry:
            self.entry = f"{self.name.replace('-', '_')}.py" if self.name else ""

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the manifest is usable.

        Raises:
            PluginError: when the plugin has no name at all
        """
        problems: list[str] = []
        if not self.name:
            raise PluginError("plugin manifest has no name")
        if not self.entry:
            problems.append("no entry point")
        unknown = [item for item in self.provides if item not in CAPABILITIES]
        if unknown:
            problems.append(f"unknown capabilities: {', '.join(unknown)}")
        unknown_hooks = [item for item in self.hooks if item not in HOOK_POINTS]
        if unknown_hooks:
            problems.append(f"unknown hooks: {', '.join(unknown_hooks)}")
        if self.min_magent and not _version_at_least(__version__, self.min_magent):
            problems.append(f"needs MAgent >= {self.min_magent}, have {__version__}")
        if self.max_magent and _version_at_least(__version__, self.max_magent, strict=True):
            problems.append(f"supports MAgent <= {self.max_magent}, have {__version__}")
        return problems

    def is_valid(self) -> bool:
        """True when :meth:`validate` finds nothing."""
        return not self.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "entry": self.entry,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "provides": list(self.provides),
            "hooks": list(self.hooks),
            "tools": list(self.tools),
            "requires": list(self.requires),
            "min_magent": self.min_magent,
            "max_magent": self.max_magent,
            "enabled": self.enabled,
            "checksum": self.checksum,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str | Path = "") -> PluginManifest:
        """Build a manifest from parsed JSON, ignoring unknown keys into metadata."""
        known = set(cls.__dataclass_fields__)
        kwargs: dict[str, Any] = {key: value for key, value in raw.items() if key in known}
        kwargs["metadata"] = {key: value for key, value in raw.items() if key not in known}
        manifest = cls(**kwargs)
        if path:
            manifest.path = str(path)
        return manifest

    @classmethod
    def from_file(cls, path: str | Path) -> PluginManifest:
        """Load a manifest from a JSON file."""
        target = Path(path).expanduser()
        raw = read_json(target)
        if not isinstance(raw, dict):
            raise PluginError(f"plugin manifest {target} is not a JSON object",
                              details={"path": str(target), "type": type(raw).__name__})
        return cls.from_dict(raw, path=target)

    def summary_line(self) -> str:
        """One-line description for ``magent plugins``."""
        state = "on" if self.enabled else "off"
        return f"{self.name} {self.version} [{state}] {self.description}"


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a value to a de-duplicated tuple of strings.

    Unknown entries are kept on purpose: :meth:`PluginManifest.validate` reports
    them, so a typo in ``provides`` is an error the user sees rather than a
    capability that silently disappears.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    elif isinstance(value, Iterable):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [str(value)]
    return tuple(dict.fromkeys(items))


def _version_parts(version: str) -> tuple[int, ...]:
    """Split a version string into comparable integers."""
    parts: list[int] = []
    for chunk in str(version).replace("-", ".").split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _version_at_least(current: str, required: str, *, strict: bool = False) -> bool:
    """Compare two dotted versions."""
    left = _version_parts(current)
    right = _version_parts(required)
    width = max(len(left), len(right))
    left += (0,) * (width - len(left))
    right += (0,) * (width - len(right))
    return left > right if strict else left >= right


def find_manifest(directory: str | Path) -> Path | None:
    """Locate a manifest file inside a plugin directory."""
    root = Path(directory).expanduser()
    if root.is_file():
        return root
    for name in MANIFEST_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def manifest_for(directory: str | Path, *, fallback_name: str = "") -> PluginManifest:
    """Load a directory's manifest, synthesising one when the plugin has none.

    A plugin without a manifest is still loadable: the entry point is guessed
    from the directory name and the manifest is marked as inferred so the CLI can
    say so.  The checksum of the entry file is recorded either way.
    """
    root = Path(directory).expanduser()
    found = find_manifest(root)
    if found is not None:
        try:
            manifest = PluginManifest.from_file(found)
        except (PluginError, OSError, ValueError) as exc:
            raise PluginError(f"invalid plugin manifest in {root}: {exc}",
                              details={"path": str(found)}) from exc
        # ``path`` always names the plugin *directory* - importers join the entry
        # onto it - so the manifest file itself is recorded separately.
        manifest.metadata.setdefault("manifest_file", str(found))
        manifest.path = str(found.parent if found.is_file() else root)
    else:
        name = fallback_name or root.name
        entry = f"{name.replace('-', '_')}.py"
        if not (root / entry).is_file():
            candidates = sorted(root.glob("*.py"))
            entry = candidates[0].name if candidates else entry
        manifest = PluginManifest(name=name, entry=entry, path=str(root),
                                  description="(no manifest)",
                                  metadata={"inferred": True})
    if root.is_dir():
        entry_path = root / manifest.entry
        if entry_path.is_file() and not manifest.checksum:
            manifest.checksum = file_sha256(entry_path)
        manifest.path = manifest.path or str(root)
    return manifest


def verify_checksum(manifest: PluginManifest) -> bool:
    """Check a manifest's declared checksum against its entry file."""
    if not manifest.checksum or not manifest.path:
        return True
    entry = Path(manifest.path)
    target = entry / manifest.entry if entry.is_dir() else entry
    if not target.is_file():
        return False
    return file_sha256(target) == manifest.checksum


def manifests_from(paths: Iterable[str | Path]) -> list[PluginManifest]:
    """Load manifests for several directories, skipping unreadable ones."""
    out: list[PluginManifest] = []
    for path in paths:
        try:
            out.append(manifest_for(path))
        except PluginError:
            continue
    return out


def write_manifest(manifest: PluginManifest, directory: str | Path) -> Path:
    """Write a manifest into a plugin directory (used by ``magent plugins new``)."""
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST_NAMES[0]
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


__all__ = [
    "CAPABILITIES",
    "HOOK_POINTS",
    "MANIFEST_NAMES",
    "PluginManifest",
    "find_manifest",
    "manifest_for",
    "manifests_from",
    "verify_checksum",
    "write_manifest",
]
