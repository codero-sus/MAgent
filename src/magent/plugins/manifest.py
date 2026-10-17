"""Plugin manifests: what a plugin declares before it is allowed to run.

Loading third-party code is the one thing a framework cannot make safe by
cleverness alone, so safety comes from declaration and refusal instead. A
manifest states the plugin's identity, its entry point, the tools and hooks it
wants to register, and - critically - the permissions it needs. Anything not
declared is not granted, and a request outside the permitted set is rejected
before any code from the plugin executes.

The manifest is validated structurally (required fields, types, version
constraints) and semantically (permissions known, entry point inside the plugin
directory), because a path that escapes the plugin folder is how a manifest
turns into an arbitrary file read.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .._version import __version__
from ..errors import PluginManifestError

MANIFEST_FILENAME = "plugin.json"
"""File a plugin directory must contain."""

MANIFEST_VERSION = 1
"""Manifest schema version this MAgent understands."""

PERMISSIONS: tuple[str, ...] = (
    "tools",
    "hooks",
    "filesystem.read",
    "filesystem.write",
    "network",
    "shell",
    "memory.read",
    "memory.write",
    "secrets.read",
    "config.read",
)
"""Every permission a plugin may declare."""

SENSITIVE_PERMISSIONS: frozenset[str] = frozenset({
    "shell", "network", "secrets.read", "filesystem.write",
})
"""Permissions that require explicit operator opt-in."""

HOOK_NAMES: tuple[str, ...] = (
    "before_run",
    "after_run",
    "before_tool",
    "after_tool",
    "on_event",
    "on_error",
    "build_prompt",
    "filter_output",
    "startup",
    "shutdown",
)
"""Lifecycle points a plugin may attach to."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
"""Plugin name shape: lowercase, underscore separated, 2-64 characters."""

_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?([.-][0-9A-Za-z.-]+)?$")
"""Semantic-ish version shape."""


@dataclass(slots=True)
class PluginManifest:
    """A validated description of one plugin."""

    name: str = ""
    version: str = "0.0.0"
    summary: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    entrypoint: str = ""
    manifest_version: int = MANIFEST_VERSION
    min_magent_version: str = ""
    permissions: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    env: list[str] = field(default_factory=list)
    enabled: bool = True
    path: str = ""
    checksum: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize list fields and lower-case permissions."""
        self.permissions = sorted({str(item).strip().lower()
                                   for item in _as_list(self.permissions) if str(item).strip()})
        self.tools = [str(item) for item in _as_list(self.tools) if str(item).strip()]
        self.hooks = [str(item) for item in _as_list(self.hooks) if str(item).strip()]
        self.env = [str(item) for item in _as_list(self.env) if str(item).strip()]
        self.name = str(self.name or "").strip()
        self.entrypoint = str(self.entrypoint or "").strip()

    # -- validation ----------------------------------------------------------

    def validate(self, *, directory: str | Path | None = None) -> PluginManifest:
        """Check the manifest is complete, coherent, and cannot escape its folder.

        Args:
            directory: the plugin's own directory, used to contain the entry point

        Returns:
            This manifest, for chaining.

        Raises:
            PluginManifestError: any rule is violated
        """
        problems: list[str] = []
        if not self.name:
            problems.append("name is required")
        elif not _NAME_RE.match(self.name):
            problems.append(f"name {self.name!r} must match {_NAME_RE.pattern}")
        if not self.version:
            problems.append("version is required")
        elif not _VERSION_RE.match(self.version):
            problems.append(f"version {self.version!r} is not a valid version string")
        if int(self.manifest_version or 0) != MANIFEST_VERSION:
            problems.append(f"manifest_version must be {MANIFEST_VERSION}, "
                            f"got {self.manifest_version!r}")
        if not self.entrypoint:
            problems.append("entrypoint is required")
        unknown = [item for item in self.permissions if item not in PERMISSIONS]
        if unknown:
            problems.append(f"unknown permission(s): {', '.join(unknown)}; "
                            f"known: {', '.join(PERMISSIONS)}")
        unknown_hooks = [item for item in self.hooks if item not in HOOK_NAMES]
        if unknown_hooks:
            problems.append(f"unknown hook(s): {', '.join(unknown_hooks)}; "
                            f"known: {', '.join(HOOK_NAMES)}")
        if self.min_magent_version and not _version_satisfies(__version__,
                                                              self.min_magent_version):
            problems.append(f"requires MAgent >= {self.min_magent_version}, "
                            f"running {__version__}")

        if self.entrypoint:
            escaped = _escapes_directory(self.entrypoint, directory)
            if escaped:
                problems.append(escaped)
        if problems:
            raise PluginManifestError(
                f"invalid manifest for {self.name or '(unnamed)'}: " + "; ".join(problems),
                details={"name": self.name, "problems": problems,
                         "path": str(directory or self.path)},
            )
        if directory:
            self.path = str(Path(directory).expanduser())
        return self

    @property
    def sensitive_permissions(self) -> list[str]:
        """Declared permissions that need explicit operator approval."""
        return [item for item in self.permissions if item in SENSITIVE_PERMISSIONS]

    def grants(self, permission: str) -> bool:
        """Whether the manifest declares one permission."""
        return str(permission).strip().lower() in self.permissions

    def uses_hooks(self) -> bool:
        """Whether the plugin attaches to any lifecycle point."""
        return bool(self.hooks)

    def registers_tools(self) -> bool:
        """Whether the plugin declares tools."""
        return bool(self.tools) or self.grants("tools")

    # -- (de)serialization ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: str = "") -> PluginManifest:
        """Build a manifest from parsed JSON, keeping unknown keys in ``extra``."""
        known = set(cls.__dataclass_fields__)
        aliases = {
            "description": "summary",
            "plugin_version": "version",
            "schema_version": "manifest_version",
            "requires": "min_magent_version",
            "min_magent": "min_magent_version",
            "entry": "entrypoint",
            "main": "entrypoint",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            name = aliases.get(str(key), str(key))
            if name in known:
                kwargs[name] = value
            else:
                extra[key] = value
        manifest = cls(**kwargs)
        manifest.extra = extra
        manifest.path = path or str(kwargs.get("path", "") or "")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form written to ``plugin.json``."""
        payload: dict[str, Any] = {
            "manifest_version": int(self.manifest_version or MANIFEST_VERSION),
            "name": self.name,
            "version": self.version,
            "entrypoint": self.entrypoint,
        }
        for name in ("summary", "author", "homepage", "license", "min_magent_version",
                     "checksum", "path"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        for name in ("permissions", "tools", "hooks", "env"):
            value = getattr(self, name)
            if value:
                payload[name] = list(value)
        if self.config_schema:
            payload["config_schema"] = self.config_schema
        if not self.enabled:
            payload["enabled"] = False
        payload.update(self.extra)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Return the manifest as JSON text."""
        return json.dumps(self.to_dict(), indent=indent or None, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> PluginManifest:
        """Read and validate a manifest file.

        Args:
            path: either the ``plugin.json`` itself or its directory

        Raises:
            PluginManifestError: the file is missing, unparsable, or invalid
        """
        target = Path(path).expanduser()
        if target.is_dir():
            target = target / MANIFEST_FILENAME
        if not target.exists():
            raise PluginManifestError(f"no manifest at {target}",
                                      details={"path": str(target),
                                               "expected": MANIFEST_FILENAME})
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PluginManifestError(f"{target} is not valid JSON: {exc}",
                                      details={"path": str(target)}) from exc
        except OSError as exc:
            raise PluginManifestError(f"cannot read {target}: {exc}",
                                      details={"path": str(target)}) from exc
        if not isinstance(data, dict):
            raise PluginManifestError(f"{target} must contain a JSON object",
                                      details={"path": str(target)})
        manifest = cls.from_dict(data, path=str(target.parent))
        return manifest.validate(directory=target.parent)

    def save(self, directory: str | Path) -> Path:
        """Write the manifest into a plugin directory."""
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        path = target / MANIFEST_FILENAME
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        self.path = str(target)
        return path

    # -- integrity -----------------------------------------------------------

    def entry_path(self) -> Path:
        """Return the absolute path of the declared entry point."""
        base = Path(self.path).expanduser()
        return (base / self.entrypoint).resolve() if self.entrypoint else base

    def compute_checksum(self) -> str:
        """Return a SHA-256 over the plugin's Python files, for tamper checks."""
        base = Path(self.path).expanduser()
        if not base.exists():
            return ""
        digest = hashlib.sha256()
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                digest.update(path.relative_to(base).as_posix().encode("utf-8"))
                digest.update(path.read_bytes())
            except OSError:
                continue
        return digest.hexdigest()

    def verify_checksum(self) -> bool:
        """Whether the plugin still matches its recorded checksum.

        A manifest with no recorded checksum verifies as true, since nothing was
        pinned; that is a deliberate choice so plugins can opt into integrity
        checking rather than being forced to compute it.
        """
        if not self.checksum:
            return True
        return self.compute_checksum() == self.checksum

    def describe(self) -> str:
        """Return a one-line summary for listings."""
        flags = ",".join(self.permissions) or "none"
        return (f"{self.name} {self.version} - {self.summary or '(no summary)'} "
                f"[permissions: {flags}]")

    def __repr__(self) -> str:
        """Debug representation."""
        return f"PluginManifest(name={self.name!r}, version={self.version!r})"


def _escapes_directory(entrypoint: str, directory: str | Path | None) -> str:
    """Return an error message when an entry point leaves its directory."""
    if directory is None:
        return ""
    base = Path(directory).expanduser().resolve()
    candidate = Path(entrypoint)
    if candidate.is_absolute():
        return f"entrypoint must be relative to the plugin directory, got {entrypoint!r}"
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return f"entrypoint {entrypoint!r} resolves outside the plugin directory"
    return ""


def _as_list(value: Any) -> list[Any]:
    """Coerce a scalar or sequence into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a version string."""
    parts: list[int] = []
    for chunk in re.split(r"[.\-+]", str(value or "")):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts) or (0,)


def _version_satisfies(current: str, minimum: str) -> bool:
    """Whether ``current`` is at least ``minimum``, comparing numeric parts."""
    return _version_tuple(current) >= _version_tuple(minimum)


def example_manifest(name: str = "example_plugin") -> PluginManifest:
    """Return a minimal valid manifest, useful for scaffolding."""
    return PluginManifest(
        name=name,
        version="0.1.0",
        summary="An example MAgent plugin",
        author="",
        entrypoint="plugin.py",
        permissions=["tools"],
        tools=["example_tool"],
        hooks=["before_run"],
    )


def validate_many(manifests: Iterable[PluginManifest]) -> dict[str, list[str]]:
    """Validate several manifests, returning problems keyed by plugin name."""
    problems: dict[str, list[str]] = {}
    for manifest in manifests:
        try:
            manifest.validate(directory=manifest.path or None)
        except PluginManifestError as exc:
            problems[manifest.name or "(unnamed)"] = list((exc.details or {}).get("problems")
                                                          or [str(exc)])
    return problems


__all__ = [
    "HOOK_NAMES",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PERMISSIONS",
    "SENSITIVE_PERMISSIONS",
    "PluginManifest",
    "example_manifest",
    "validate_many",
]
