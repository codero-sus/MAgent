"""Plugin loading: discover, validate, import and register extensions.

Plugins are ordinary Python packages or single-file modules in a plugin
directory.  Each one declares a manifest and an entry module exposing either

``register(registry, context)``
    the simple form - add tools and return names;
``PLUGIN = {...}``
    a declarative dict with ``tools``, ``hooks``, ``personas``, ``templates``,
    ``skills`` and an optional ``setup(context)`` callable.

Loading is deliberately paranoid: the entry file must live inside the plugin
directory (no path traversal), it must not exceed the size ceiling, the manifest
must validate, and any exception during import or registration is captured on
the :class:`LoadedPlugin` record instead of propagating.  ``strict=True`` turns
those captures into exceptions, which is what the test suite and ``magent
plugins check`` use.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import PluginError, PluginNotFoundError, PluginSecurityError
from ..logging_ import get_logger
from ..utils.files import ensure_dir, human_size, iter_files
from ..utils.text import slugify
from .hooks import HookRegistry
from .manifest import PluginManifest, manifest_for, verify_checksum

LOGGER = get_logger("plugins.loader")
"""Logger for plugin loading."""

PLUGIN_DIRS: tuple[str, ...] = ("plugins", ".magent/plugins")
"""Directory names searched inside a workspace."""

MAX_ENTRY_BYTES = 2_000_000
"""Refuse to import an entry file larger than this."""


@dataclass(slots=True)
class LoadedPlugin:
    """A plugin that was loaded, with everything it contributed."""

    manifest: PluginManifest
    module: Any = None
    tools: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    seconds: float = 0.0
    ok: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Plugin name."""
        return self.manifest.name

    @property
    def contributions(self) -> int:
        """How many things the plugin registered."""
        return (len(self.tools) + len(self.hooks) + len(self.personas) + len(self.templates)
                + len(self.skills))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record."""
        return {
            "name": self.name,
            "version": self.manifest.version,
            "path": self.manifest.path,
            "ok": self.ok,
            "error": self.error,
            "warnings": list(self.warnings),
            "tools": list(self.tools),
            "hooks": list(self.hooks),
            "personas": list(self.personas),
            "templates": list(self.templates),
            "skills": list(self.skills),
            "contributions": self.contributions,
            "seconds": round(self.seconds, 5),
            "manifest": self.manifest.to_dict(),
        }

    def summary_line(self) -> str:
        """One-line status for ``magent plugins``."""
        state = "ok" if self.ok else f"FAILED: {self.error}"
        return (f"{self.name} {self.manifest.version} - {self.contributions} contributions "
                f"({state})")


class PluginLoader:
    """Discover and load plugins from one or more directories."""

    def __init__(self, *, registry: Any = None, hooks: HookRegistry | None = None,
                 allowed: Sequence[str] = (), blocked: Sequence[str] = (),
                 strict: bool = False, max_entry_bytes: int = MAX_ENTRY_BYTES,
                 check_checksums: bool = False) -> None:
        """Configure the loader.

        Args:
            registry: tool registry receiving plugin tools
            hooks: hook registry receiving plugin hooks
            allowed: when non-empty, only these plugin names load
            blocked: plugin names that never load
            strict: raise instead of recording load failures
            max_entry_bytes: refuse larger entry files
            check_checksums: verify declared entry checksums
        """
        self.registry = registry
        self.hooks = hooks if hooks is not None else HookRegistry()
        self.allowed = {str(item) for item in allowed}
        self.blocked = {str(item) for item in blocked}
        self.strict = bool(strict)
        self.max_entry_bytes = max(1_024, int(max_entry_bytes))
        self.check_checksums = bool(check_checksums)
        self.loaded: dict[str, LoadedPlugin] = {}
        self.paths: list[Path] = []

    # -- discovery -----------------------------------------------------------

    def discover(self, *roots: str | Path) -> list[PluginManifest]:
        """Find plugin manifests under the given roots.

        A root may be a plugin directory itself, a directory containing several
        plugin directories, or a path to a manifest file.
        """
        found: list[PluginManifest] = []
        for root in roots:
            if not root:
                continue
            base = Path(root).expanduser()
            self.paths.append(base)
            if not base.exists():
                continue
            candidates: list[Path] = []
            if base.is_file():
                candidates.append(base.parent)
            else:
                if (base / "plugin.json").is_file() or (base / "__init__.py").is_file() \
                        or any(base.glob("*.py")):
                    candidates.append(base)
                for child in sorted(base.iterdir()):
                    if child.is_dir() and not child.name.startswith((".", "__")):
                        if (child / "plugin.json").is_file() or (child / "__init__.py").is_file() \
                                or any(child.glob("*.py")):
                            candidates.append(child)
                        elif (child / "plugins").is_dir():
                            candidates.append(child / "plugins")
            for candidate in candidates:
                try:
                    manifest = manifest_for(candidate)
                except PluginError as exc:
                    LOGGER.debug("skipping %s: %s", candidate, exc)
                    continue
                if manifest.name in {item.name for item in found}:
                    continue
                found.append(manifest)
        return found

    def available(self, *roots: str | Path) -> list[PluginManifest]:
        """Manifests that pass validation, regardless of the allow/block lists."""
        return [manifest for manifest in self.discover(*roots) if manifest.is_valid()]

    # -- loading -------------------------------------------------------------

    def load_directory(self, directory: str | Path) -> LoadedPlugin:
        """Load one plugin directory, capturing any failure on the record."""
        started = time.perf_counter()
        manifest = manifest_for(directory)
        record = LoadedPlugin(manifest=manifest)
        try:
            self._check(manifest, directory)
            module = self._import(manifest, directory)
            record.module = module
            self._register(module, manifest, record)
            record.ok = True
        except Exception as exc:  # noqa: BLE001 - a broken plugin is data, not a crash
            record.ok = False
            record.error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("plugin %s failed to load: %s", manifest.name, exc)
            if self.strict:
                raise
        record.seconds = time.perf_counter() - started
        self.loaded[manifest.name] = record
        return record

    def load_all(self, *roots: str | Path, include_disabled: bool = False) -> list[LoadedPlugin]:
        """Discover and load every plugin under the given roots."""
        records: list[LoadedPlugin] = []
        for manifest in self.discover(*roots):
            if manifest.name in self.blocked:
                LOGGER.debug("plugin %s is blocked", manifest.name)
                continue
            if self.allowed and manifest.name not in self.allowed:
                continue
            if not manifest.enabled and not include_disabled:
                continue
            records.append(self.load_directory(manifest.path or Path(manifest.name)))
        return records

    def load_named(self, name: str, *roots: str | Path) -> LoadedPlugin:
        """Load one plugin by name.

        Raises:
            PluginNotFoundError: when no plugin with that name is discovered
        """
        for manifest in self.discover(*roots):
            if manifest.name == name:
                return self.load_directory(manifest.path or Path(name))
        raise PluginNotFoundError(f"plugin {name!r} not found",
                                  details={"searched": [str(path) for path in self.paths],
                                           "known": sorted(self.loaded)})

    # -- internals -----------------------------------------------------------

    def _check(self, manifest: PluginManifest, directory: str | Path) -> None:
        """Validate a manifest and the entry file before importing."""
        problems = manifest.validate()
        if problems:
            raise PluginError(f"plugin {manifest.name!r} is invalid: {'; '.join(problems)}",
                              details={"manifest": manifest.to_dict()})
        root = Path(directory).expanduser()
        target = root if root.is_file() else root / manifest.entry
        if root.is_dir():
            try:
                resolved = target.resolve()
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise PluginSecurityError(
                    f"plugin {manifest.name!r} entry escapes its directory",
                    details={"entry": manifest.entry, "root": str(root)},
                ) from exc
        if not target.exists():
            raise PluginError(f"plugin {manifest.name!r} entry {manifest.entry!r} not found",
                              details={"path": str(target)})
        size = target.stat().st_size
        if size > self.max_entry_bytes:
            raise PluginSecurityError(
                f"plugin {manifest.name!r} entry is {human_size(size)}, over the "
                f"{human_size(self.max_entry_bytes)} limit",
                details={"path": str(target), "bytes": size},
            )
        if self.check_checksums and not verify_checksum(manifest):
            raise PluginSecurityError(f"plugin {manifest.name!r} checksum mismatch",
                                      details={"entry": manifest.entry,
                                               "expected": manifest.checksum})

    def _import(self, manifest: PluginManifest, directory: str | Path) -> Any:
        """Import the plugin's entry module under a private name."""
        root = Path(directory).expanduser()
        if root.is_file():
            target = root
        elif (root / "__init__.py").is_file() and manifest.entry in {"", "__init__.py"}:
            target = root / "__init__.py"
        else:
            target = root / manifest.entry
        module_name = f"magent_plugin_{slugify(manifest.name).replace(chr(45), chr(95)) or 'plugin'}"
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            raise PluginError(f"cannot import plugin {manifest.name!r}",
                              details={"path": str(target)})
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def _register(self, module: Any, manifest: PluginManifest, record: LoadedPlugin) -> None:
        """Apply whatever registration style the plugin uses."""
        context = {
            "name": manifest.name,
            "version": manifest.version,
            "manifest": manifest,
            "loader": self,
            "hooks": self.hooks,
            "registry": self.registry,
            "logger": get_logger(f"plugin.{manifest.name}"),
        }
        handled = False
        declared = getattr(module, "PLUGIN", None)
        if isinstance(declared, Mapping):
            self._register_declared(declared, context, record)
            handled = True
        setup = getattr(module, "setup", None)
        if callable(setup) and not handled:
            result = setup(context)
            if isinstance(result, Mapping):
                self._register_declared(result, context, record)
            handled = True
        register = getattr(module, "register", None)
        if callable(register) and not handled:
            result = register(self.registry, context) if self.registry is not None \
                else register(context)
            self._collect_names(result, record.tools)
            handled = True
        if not handled:
            # Single-file plugins usually just decorate functions with @tool and
            # declare HOOKS; discover those instead of demanding boilerplate.
            self._autodiscover(module, manifest, record)
        for hook_name, fn in (getattr(module, "HOOKS", None) or {}).items():
            handler = self.hooks.register(str(hook_name), fn, plugin=manifest.name)
            record.hooks.append(f"{hook_name}:{handler.name}")
        if not handled and not record.contributions:
            record.warnings.append("plugin exposes no tools, hooks or other contributions")

    def _autodiscover(self, module: Any, manifest: PluginManifest, record: LoadedPlugin) -> None:
        """Register module-level ``@tool`` functions, skills, personas and templates."""
        from ..tools.base import Tool

        for attribute, value in sorted(vars(module).items()):
            if attribute.startswith("_"):
                continue
            candidate = getattr(value, "tool", None)
            if isinstance(candidate, Tool):
                if self.registry is not None:
                    self.registry.register(candidate)
                record.tools.append(candidate.spec.name)
                continue
            if isinstance(value, Tool):
                if self.registry is not None:
                    self.registry.register(value)
                record.tools.append(value.spec.name)
        for value in getattr(module, "SKILLS", []) or []:
            from ..skills import register_skill

            register_skill(value)
            record.skills.append(getattr(value, "name", str(value)))
        for value in getattr(module, "PERSONAS", []) or []:
            from ..prompts.personas import register_persona

            register_persona(value)
            record.personas.append(getattr(value, "name", str(value)))
        for value in getattr(module, "TEMPLATES", []) or []:
            from ..prompts.templates import register_template

            register_template(value)
            record.templates.append(getattr(value, "name", str(value)))
        del manifest

    def _register_declared(self, declared: Mapping[str, Any], context: dict[str, Any],
                           record: LoadedPlugin) -> None:
        """Register the entries of a declarative ``PLUGIN`` dictionary."""
        setup = declared.get("setup")
        if callable(setup):
            result = setup(context)
            if isinstance(result, Mapping) and result is not declared:
                self._register_declared(result, context, record)
        if self.registry is not None:
            for item in declared.get("tools", []) or []:
                self.registry.register(item)
                record.tools.append(getattr(getattr(item, "spec", None), "name", "")
                                    or getattr(item, "name", str(item)))
        for hook_name, fn in (declared.get("hooks") or {}).items():
            handler = self.hooks.register(str(hook_name), fn, plugin=context["name"])
            record.hooks.append(f"{hook_name}:{handler.name}")
        for fn in declared.get("personas", []) or []:
            from ..prompts.personas import register_persona

            register_persona(fn)
            record.personas.append(getattr(fn, "name", str(fn)))
        for template in declared.get("templates", []) or []:
            from ..prompts.templates import register_template

            register_template(template)
            record.templates.append(getattr(template, "name", str(template)))
        for skill in declared.get("skills", []) or []:
            from ..skills import register_skill

            register_skill(skill)
            record.skills.append(getattr(skill, "name", str(skill)))

    def _collect_names(self, result: Any, into: list[str]) -> None:
        """Record names returned by a ``register`` callable."""
        if result is None:
            return
        if isinstance(result, Mapping):
            items = list(result.keys())
        elif isinstance(result, (list, tuple, set)):
            items = list(result)
        else:
            items = [result]
        for item in items:
            name = getattr(item, "name", None) or getattr(getattr(item, "spec", None), "name", None)
            into.append(str(name or item))

    # -- views ---------------------------------------------------------------

    def names(self) -> list[str]:
        """Loaded plugin names."""
        return sorted(self.loaded)

    def succeeded(self) -> list[LoadedPlugin]:
        """Plugins that loaded cleanly."""
        return [item for item in self.loaded.values() if item.ok]

    def failed(self) -> list[LoadedPlugin]:
        """Plugins that failed to load."""
        return [item for item in self.loaded.values() if not item.ok]

    def unload(self, name: str) -> bool:
        """Forget a loaded plugin (tools it registered stay registered)."""
        record = self.loaded.pop(name, None)
        if record is None:
            return False
        module_name = getattr(record.module, "__name__", "")
        if module_name in sys.modules:
            sys.modules.pop(module_name, None)
        return True

    def stats(self) -> dict[str, Any]:
        """Counters describing everything loaded."""
        records = list(self.loaded.values())
        return {
            "loaded": len(records),
            "failed": len(self.failed()),
            "tools": sum(len(item.tools) for item in records),
            "hooks": sum(len(item.hooks) for item in records),
            "personas": sum(len(item.personas) for item in records),
            "templates": sum(len(item.templates) for item in records),
            "skills": sum(len(item.skills) for item in records),
            "seconds": round(sum(item.seconds for item in records), 5),
            "paths": [str(path) for path in self.paths],
            "names": self.names(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise every loaded plugin."""
        return {"plugins": {name: record.to_dict() for name, record in sorted(self.loaded.items())},
                "stats": self.stats()}

    def __len__(self) -> int:
        """Number of loaded plugins."""
        return len(self.loaded)

    def __contains__(self, name: object) -> bool:
        """True when a plugin was loaded."""
        return str(name) in self.loaded

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"PluginLoader(loaded={len(self.loaded)}, paths={len(self.paths)})"


def default_plugin_paths(config: Any = None, *, workspace: str | Path | None = None,
                         data_dir: str | Path | None = None) -> list[Path]:
    """Resolve the plugin directories to search.

    Order matters: user plugins, then project plugins, then any explicit
    ``tools.plugins_dir``.  Duplicates are dropped, keeping the first occurrence.
    """
    paths: list[Path] = []
    tools = getattr(config, "tools", None)
    explicit = getattr(tools, "plugins_dir", "") if tools is not None else ""
    home = Path(data_dir or getattr(config, "data_dir", "") or (Path.home() / ".local/share/magent"))
    paths.append(home / "plugins")
    root = Path(workspace or getattr(tools, "workspace", "") or Path.cwd())
    for name in PLUGIN_DIRS:
        paths.append(root / name)
    if explicit:
        paths.append(Path(explicit).expanduser())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_plugins(config: Any = None, *, registry: Any = None, hooks: HookRegistry | None = None,
                 workspace: str | Path | None = None, strict: bool = False,
                 create: bool = False) -> PluginLoader:
    """Load every plugin configured for this workspace.

    Args:
        config: loaded configuration
        registry: tool registry to receive plugin tools
        hooks: hook registry to receive plugin hooks
        workspace: workspace root, overriding the configured one
        strict: raise on the first load failure
        create: create the first plugin directory when it does not exist
    """
    tools = getattr(config, "tools", None)
    enabled = bool(getattr(tools, "autoload_plugins", True)) if tools is not None else True
    loader = PluginLoader(registry=registry, hooks=hooks, strict=strict)
    if not enabled:
        return loader
    paths = default_plugin_paths(config, workspace=workspace)
    if create and paths:
        ensure_dir(paths[0])
    loader.load_all(*paths)
    return loader


def iter_plugin_files(directory: str | Path, *, patterns: Iterable[str] = ("*.py", "*.json",
                                                                          "*.md")) -> list[Path]:
    """List the files that make up a plugin directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return list(iter_files(root, patterns=tuple(patterns), max_depth=2))


def plugin_module(name: str, module: types.ModuleType | Any) -> dict[str, Any]:
    """Describe a plugin module for diagnostics."""
    return {
        "name": name,
        "module": getattr(module, "__name__", ""),
        "file": getattr(module, "__file__", ""),
        "has_register": callable(getattr(module, "register", None)),
        "has_setup": callable(getattr(module, "setup", None)),
        "declared": sorted((getattr(module, "PLUGIN", None) or {}).keys())
        if isinstance(getattr(module, "PLUGIN", None), Mapping) else [],
    }


__all__ = [
    "MAX_ENTRY_BYTES",
    "PLUGIN_DIRS",
    "LoadedPlugin",
    "PluginLoader",
    "default_plugin_paths",
    "iter_plugin_files",
    "load_plugins",
    "plugin_module",
]
