"""Loading plugins: discovery, permission checks, and isolation of failure.

The loader's job is to make third-party code usable without making it trusted.
Concretely:

*discovery is explicit*
    plugins are found only in configured directories, and a directory is scanned
    for ``plugin.json`` rather than executing anything that happens to be there;

*permissions gate registration*
    a plugin that does not declare ``tools`` cannot register tools, and one that
    declares a sensitive permission is skipped unless the operator allowed it;

*integrity is checkable*
    a manifest may pin a SHA-256 over the plugin's sources, and a mismatch is
    refused rather than warned about;

*failure is contained*
    a plugin that raises during import is recorded with its traceback and every
    other plugin still loads, so one broken install does not disable the agent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

from ..errors import PluginError, PluginManifestError, PluginNotFoundError, PluginSecurityError
from ..logging_ import get_logger
from .hooks import HookRegistry, register_decorated
from .manifest import (
    MANIFEST_FILENAME,
    PERMISSIONS,
    SENSITIVE_PERMISSIONS,
    PluginManifest,
    example_manifest,
)

LOGGER = get_logger("plugins.loader")
"""Logger for plugin loading."""

PLUGIN_SUFFIXES: tuple[str, ...] = (".py",)
"""Entry point file extensions that may be imported."""

DEFAULT_TIMEOUT = 30.0
"""Ceiling on a single plugin's import, in seconds; advisory only."""


@dataclass(slots=True)
class LoadedPlugin:
    """A plugin that was imported successfully."""

    manifest: PluginManifest
    module: Any = None
    tools: list[Any] = field(default_factory=list)
    hooks: int = 0
    """Number of hooks this plugin attached.

    A count, not a list: :attr:`tools` holds the registered ``Tool``
    objects because callers inspect them, while hooks belong to the
    :class:`~magent.plugins.hooks.HookRegistry` and are read back from
    there. Asking this field for the registrations is a type error.
    """
    load_ms: float = 0.0
    path: str = ""
    error: str = ""
    skipped: str = ""
    checksum_ok: bool = True

    @property
    def name(self) -> str:
        """The plugin's declared name."""
        return self.manifest.name

    @property
    def ok(self) -> bool:
        """Whether the plugin loaded and was not skipped."""
        return not self.error and not self.skipped

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "summary": self.manifest.summary,
            "path": self.path,
            "permissions": list(self.manifest.permissions),
            "tools": [getattr(item, "name", str(item)) for item in self.tools],
            "hooks": self.hooks,
            "load_ms": round(self.load_ms, 3),
            "ok": self.ok,
            "error": self.error,
            "skipped": self.skipped,
            "checksum_ok": self.checksum_ok,
        }

    def describe(self) -> str:
        """Return a one-line summary for listings."""
        state = "loaded" if self.ok else (f"skipped: {self.skipped}" if self.skipped
                                          else f"failed: {self.error}")
        tools = len(self.tools)
        return (f"{self.manifest.name} {self.manifest.version} - {state} "
                f"({tools} tool(s), {self.hooks} hook(s), {self.load_ms:.0f}ms)")


@dataclass(slots=True)
class LoadReport:
    """What one discovery-and-load pass did."""

    plugins: list[LoadedPlugin] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def loaded(self) -> list[LoadedPlugin]:
        """Plugins that loaded successfully."""
        return [item for item in self.plugins if item.ok]

    @property
    def failed(self) -> list[LoadedPlugin]:
        """Plugins that raised during load."""
        return [item for item in self.plugins if item.error]

    @property
    def skipped(self) -> list[LoadedPlugin]:
        """Plugins refused on policy grounds."""
        return [item for item in self.plugins if item.skipped]

    @property
    def tool_count(self) -> int:
        """Total tools contributed by loaded plugins."""
        return sum(len(item.tools) for item in self.loaded)

    def by_name(self, name: str) -> LoadedPlugin | None:
        """Return one plugin by name."""
        return next((item for item in self.plugins if item.name == name), None)

    def names(self) -> list[str]:
        """Return every plugin name seen, loaded or not."""
        return [item.name for item in self.plugins]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized report."""
        return {
            "directories": self.directories,
            "found": len(self.plugins),
            "loaded": len(self.loaded),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "tools": self.tool_count,
            "duration_ms": round(self.duration_ms, 3),
            "plugins": [item.to_dict() for item in self.plugins],
            "errors": self.errors,
        }

    def render(self, *, width: int = 100) -> str:
        """Return a text summary of the pass."""
        lines = [f"plugins: {len(self.loaded)} loaded, {len(self.skipped)} skipped, "
                 f"{len(self.failed)} failed, {self.tool_count} tools"]
        for plugin in self.plugins:
            lines.append(f"  {plugin.describe()}"[: max(40, width)])
        return "\n".join(lines)


class PluginLoader:
    """Discovers, validates, and loads plugins."""

    def __init__(self, directories: Sequence[str | Path] = (), *,
                 allow_sensitive: Sequence[str] = (), deny: Sequence[str] = (),
                 allow: Sequence[str] | None = None, registry: Any = None,
                 hooks: HookRegistry | None = None, verify_checksums: bool = True,
                 strict: bool = False) -> None:
        """Configure the loader.

        Args:
            directories: where to look for plugin folders
            allow_sensitive: sensitive permissions the operator has approved
            deny: plugin names that must never load
            allow: when set, only these plugin names may load
            registry: tool registry plugins may register into
            hooks: hook registry plugins may attach to
            verify_checksums: refuse a plugin whose checksum does not match
            strict: re-raise load failures instead of recording them
        """
        self.directories = [Path(item).expanduser() for item in directories if str(item)]
        self.allow_sensitive = {str(item).strip().lower() for item in allow_sensitive}
        self.deny = {str(item).strip() for item in deny if str(item).strip()}
        self.allow = {str(item).strip() for item in allow} if allow is not None else None
        self.registry = registry
        self.hooks = hooks if hooks is not None else HookRegistry()
        self.verify_checksums = bool(verify_checksums)
        self.strict = bool(strict)
        self.last_report: LoadReport | None = None
        # Per-instance, not a class attribute: a shared mutable dict would let
        # one loader report another loader's failures.
        self.errors: dict[str, str] = {}

    def with_policy(self, *, allow_sensitive: Sequence[str] | None = None,
                    deny: Sequence[str] | None = None,
                    allow: Sequence[str] | None = None,
                    directories: Sequence[str | Path] | None = None,
                    verify_checksums: bool | None = None,
                    strict: bool | None = None) -> "PluginLoader":
        """Return a copy of this loader with a different policy.

        The registries are shared, so a loader derived for one command still
        registers tools into the same catalogue. Copying rather than mutating
        means a caller can try a stricter policy without losing the original.

        Args:
            allow_sensitive: sensitive permissions to approve, replacing the set
            deny: plugin names to refuse, replacing the set
            allow: when set, only these plugin names may load
            directories: extra directories to scan, appended to the current ones
            verify_checksums: whether to enforce manifest checksums
            strict: whether to re-raise load failures
        """
        copy = PluginLoader(
            list(self.directories) + [Path(item).expanduser()
                                      for item in (directories or ()) if str(item)],
            allow_sensitive=self.allow_sensitive if allow_sensitive is None
            else {str(item).strip().lower() for item in allow_sensitive},
            deny=self.deny if deny is None
            else {str(item).strip() for item in deny if str(item).strip()},
            allow=self.allow if allow is None else list(allow),
            registry=self.registry,
            hooks=self.hooks,
            verify_checksums=self.verify_checksums if verify_checksums is None
            else bool(verify_checksums),
            strict=self.strict if strict is None else bool(strict),
        )
        return copy

    # -- discovery -----------------------------------------------------------

    def discover(self) -> list[PluginManifest]:
        """Find and validate every manifest in the configured directories.

        A directory whose manifest is invalid is recorded and skipped rather
        than aborting discovery, so one broken plugin does not hide the rest.

        Returns:
            Validated manifests, in a stable name order.
        """
        found: list[PluginManifest] = []
        seen: set[str] = set()
        for directory in self.directories:
            if not directory.exists():
                LOGGER.debug("plugin directory does not exist: %s", directory)
                continue
            for manifest_path in sorted(directory.rglob(MANIFEST_FILENAME)):
                if any(part.startswith(".") or part == "__pycache__"
                       for part in manifest_path.parts):
                    continue
                try:
                    manifest = PluginManifest.load(manifest_path.parent)
                except PluginManifestError as exc:
                    LOGGER.warning("skipping %s: %s", manifest_path.parent, exc)
                    continue
                if manifest.name in seen:
                    LOGGER.warning("duplicate plugin name %r at %s; keeping the first",
                                   manifest.name, manifest_path.parent)
                    continue
                seen.add(manifest.name)
                found.append(manifest)
        found.sort(key=lambda item: item.name)
        return found

    # -- policy --------------------------------------------------------------

    def check_permission(self, manifest: PluginManifest) -> str:
        """Return a reason to refuse a plugin, or an empty string to allow it.

        Args:
            manifest: the plugin's declaration

        Raises:
            PluginSecurityError: in strict mode, instead of returning a reason
        """
        if manifest.name in self.deny:
            reason = "plugin is on the deny list"
            return self._refuse(manifest, reason)
        if self.allow is not None and manifest.name not in self.allow:
            reason = "plugin is not on the allow list"
            return self._refuse(manifest, reason)
        if not manifest.enabled:
            return self._refuse(manifest, "manifest declares enabled=false")
        unknown = [item for item in manifest.permissions if item not in PERMISSIONS]
        if unknown:
            return self._refuse(manifest, f"unknown permission(s): {', '.join(unknown)}")
        blocked = [item for item in manifest.sensitive_permissions
                   if item not in self.allow_sensitive]
        if blocked:
            return self._refuse(
                manifest,
                f"needs sensitive permission(s) {', '.join(sorted(blocked))}; "
                f"approve with allow_sensitive",
            )
        if self.verify_checksums and not manifest.verify_checksum():
            return self._refuse(manifest, "checksum does not match the manifest")
        entry = manifest.entry_path()
        if entry.suffix not in PLUGIN_SUFFIXES:
            return self._refuse(manifest, f"entrypoint must be one of {PLUGIN_SUFFIXES}")
        if not entry.exists():
            return self._refuse(manifest, f"entrypoint {manifest.entrypoint} does not exist")
        return ""

    def _refuse(self, manifest: PluginManifest, reason: str) -> str:
        """Record a refusal, raising in strict mode."""
        LOGGER.warning("refusing plugin %s: %s", manifest.name, reason)
        if self.strict:
            raise PluginSecurityError(f"plugin {manifest.name!r} refused: {reason}",
                                      details={"plugin": manifest.name, "reason": reason})
        return reason

    # -- loading -------------------------------------------------------------

    def load(self, manifest: PluginManifest) -> LoadedPlugin:
        """Import one plugin and register what it declares.

        Args:
            manifest: a validated manifest

        Returns:
            A :class:`LoadedPlugin` recording success, refusal, or the error
        """
        import time

        started = time.perf_counter()
        record = LoadedPlugin(manifest=manifest, path=str(Path(manifest.path).expanduser()))
        reason = self.check_permission(manifest)
        if reason:
            record.skipped = reason
            record.load_ms = (time.perf_counter() - started) * 1000.0
            return record
        if self.verify_checksums:
            record.checksum_ok = manifest.verify_checksum()
        try:
            module = self._import(manifest)
        except Exception as exc:  # noqa: BLE001 - contained per plugin
            record.error = f"{type(exc).__name__}: {exc}"
            record.load_ms = (time.perf_counter() - started) * 1000.0
            self._record_error(manifest.name, exc)
            if self.strict:
                raise PluginError(f"plugin {manifest.name!r} failed to load: {exc}",
                                  details={"plugin": manifest.name}) from exc
            return record

        record.module = module
        record.tools = self._register_tools(manifest, module)
        record.hooks = self._register_hooks(manifest, module)
        self._call_lifecycle(module, "register", manifest=manifest, registry=self.registry,
                             hooks=self.hooks)
        record.load_ms = (time.perf_counter() - started) * 1000.0
        LOGGER.info("loaded plugin %s %s (%d tools, %d hooks, %.0fms)", manifest.name,
                    manifest.version, len(record.tools), record.hooks, record.load_ms)
        return record

    def _import(self, manifest: PluginManifest) -> ModuleType:
        """Import a plugin's entry point under a namespaced module name."""
        entry = manifest.entry_path()
        module_name = f"magent_plugin_{manifest.name}"
        existing = sys.modules.get(module_name)
        if existing is not None:
            existing_file = Path(str(getattr(existing, "__file__", "") or "")).resolve()
            if existing_file == entry.resolve():
                return existing
            # Reusing the old module would execute neither the newly inspected
            # file nor its checksum. Replacing it is unsafe too because tools
            # already registered from the old module would remain live. Require
            # an explicit unload before a same-name plugin can move locations.
            raise PluginSecurityError(
                f"plugin {manifest.name!r} is already loaded from {existing_file}; "
                f"refusing a second entrypoint at {entry.resolve()}",
                details={"plugin": manifest.name, "loaded": str(existing_file),
                         "requested": str(entry.resolve())},
            )
        spec = importlib.util.spec_from_file_location(module_name, entry)
        if spec is None or spec.loader is None:
            raise PluginError(f"cannot create an import spec for {entry}",
                              details={"plugin": manifest.name, "entrypoint": str(entry)})
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # A half-initialized module must not linger: a later import would
            # reuse it and report success for code that never ran.
            sys.modules.pop(module_name, None)
            raise
        return module

    def _register_tools(self, manifest: PluginManifest, module: ModuleType) -> list[Any]:
        """Register the plugin's tools, honouring its declared permission."""
        if not manifest.grants("tools"):
            declared = getattr(module, "TOOLS", None)
            if declared:
                LOGGER.warning("plugin %s defines TOOLS but did not declare the "
                               "'tools' permission; not registering", manifest.name)
            return []
        if self.registry is None:
            return []
        registered: list[Any] = []
        declared_names = set(manifest.tools)
        for item in list(getattr(module, "TOOLS", []) or []):
            tool = getattr(item, "tool", item)
            spec = getattr(tool, "spec", None)
            name = getattr(spec, "name", "") if spec is not None else ""
            if declared_names and name and name not in declared_names:
                LOGGER.warning("plugin %s tried to register undeclared tool %r; refused",
                               manifest.name, name)
                continue
            try:
                self.registry.register(tool)
                registered.append(tool)
            except Exception as exc:  # noqa: BLE001 - one bad tool must not fail the plugin
                LOGGER.warning("plugin %s could not register tool %r: %s",
                               manifest.name, name, exc)
        loader = getattr(module, "load", None)
        if callable(loader) and not registered:
            try:
                loader(self.registry)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("plugin %s load() raised: %s", manifest.name, exc)
        return registered

    def _register_hooks(self, manifest: PluginManifest, module: ModuleType) -> int:
        """Register the plugin's hooks, limited to what it declared."""
        if not manifest.grants("hooks") and not manifest.hooks:
            return 0
        allowed = set(manifest.hooks) or set()
        before = self.hooks.count()
        register_decorated(module, self.hooks, plugin=manifest.name)
        if allowed:
            for registration in self.hooks.hooks():
                if registration.plugin == manifest.name and registration.name not in allowed:
                    self.hooks.unregister(registration.name, plugin=manifest.name,
                                          fn=registration.fn)
                    LOGGER.warning("plugin %s attached to undeclared hook %r; removed",
                                   manifest.name, registration.name)
        return self.hooks.count() - before

    def _call_lifecycle(self, module: ModuleType, name: str, **kwargs: Any) -> None:
        """Call an optional lifecycle function if the plugin defines one."""
        fn = getattr(module, name, None)
        if not callable(fn):
            return
        try:
            fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - lifecycle failures are not fatal
            LOGGER.warning("plugin lifecycle %s() raised: %s", name, exc)

    def _record_error(self, name: str, exc: BaseException) -> None:
        """Store a plugin's load failure for later inspection."""
        self.errors[name] = f"{type(exc).__name__}: {exc}\n" + \
            traceback.format_exc(limit=4)

    # -- orchestration -------------------------------------------------------

    def load_all(self) -> LoadReport:
        """Discover and load every permitted plugin."""
        import time

        started = time.perf_counter()
        report = LoadReport(directories=[str(item) for item in self.directories])
        self.errors = {}
        manifests = self.discover()
        for manifest in manifests:
            if not manifest.enabled:
                report.plugins.append(LoadedPlugin(manifest=manifest,
                                                   path=manifest.path,
                                                   skipped="disabled in its manifest"))
                continue
            report.plugins.append(self.load(manifest))
        report.errors = dict(self.errors)
        report.duration_ms = (time.perf_counter() - started) * 1000.0
        self.last_report = report
        LOGGER.info("plugins: %d loaded, %d skipped, %d failed in %.0fms",
                    len(report.loaded), len(report.skipped), len(report.failed),
                    report.duration_ms)
        return report

    def unload(self, name: str) -> bool:
        """Remove a plugin's hooks and forget its module."""
        self.hooks.unregister_plugin(name)
        module_name = f"magent_plugin_{name}"
        existed = sys.modules.pop(module_name, None) is not None
        if self.registry is not None and existed:
            LOGGER.info("plugin %s unloaded; tools it registered remain until the "
                        "registry is rebuilt", name)
        return existed

    def unload_all(self) -> int:
        """Unload every loaded plugin, returning how many were removed."""
        report = self.last_report
        names = [item.name for item in (report.loaded if report else [])]
        return sum(1 for name in names if self.unload(name))

    def get(self, name: str) -> LoadedPlugin:
        """Return one loaded plugin by name.

        Raises:
            PluginNotFoundError: no plugin of that name was loaded
        """
        report = self.last_report
        found = report.by_name(name) if report else None
        if found is None:
            known = report.names() if report else []
            raise PluginNotFoundError(f"no plugin named {name!r}",
                                      details={"name": name, "known": known})
        return found

    def stats(self) -> dict[str, Any]:
        """Return loader statistics."""
        report = self.last_report
        payload: dict[str, Any] = {
            "directories": [str(item) for item in self.directories],
            "allow_sensitive": sorted(self.allow_sensitive),
            "deny": sorted(self.deny),
            "allow": sorted(self.allow) if self.allow is not None else None,
            "verify_checksums": self.verify_checksums,
            "strict": self.strict,
            "hooks": self.hooks.stats(),
        }
        if report is not None:
            payload.update({
                "found": len(report.plugins),
                "loaded": len(report.loaded),
                "skipped": len(report.skipped),
                "failed": len(report.failed),
                "tools": report.tool_count,
                "duration_ms": round(report.duration_ms, 3),
            })
        return payload


def loader_from_config(config: Any, *, registry: Any = None,
                       hooks: HookRegistry | None = None,
                       extra_directories: Iterable[str | Path] = ()) -> PluginLoader:
    """Build a loader from a resolved configuration.

    Args:
        config: configuration carrying a ``tools`` section
        registry: tool registry plugins register into
        hooks: hook registry, created when omitted
        extra_directories: additional directories to scan
    """
    section = getattr(config, "tools", None)
    configured = str(getattr(section, "plugins_dir", "") or "")
    directories: list[Path] = []
    if configured:
        directories.append(Path(configured).expanduser())
    data_dir = str(getattr(config, "data_dir", "") or "")
    if data_dir:
        directories.append(Path(data_dir).expanduser() / "plugins")
    directories.extend(Path(item).expanduser() for item in extra_directories if str(item))
    # De-duplicate while keeping order: [tools] plugins_dir defaults to
    # <data_dir>/plugins, so both branches above can yield the same path, and
    # reporting one directory twice reads as a configuration bug.
    unique: list[Path] = []
    for directory in directories:
        if directory not in unique:
            unique.append(directory)
    directories = unique
    plugins_section = getattr(config, "plugins", None)
    return PluginLoader(
        directories,
        allow_sensitive=list(getattr(plugins_section, "allow_sensitive", []) or []),
        deny=list(getattr(plugins_section, "deny", []) or []),
        allow=list(getattr(plugins_section, "allow", []) or []) or None,
        registry=registry,
        hooks=hooks,
        verify_checksums=bool(getattr(plugins_section, "verify_checksums", True)),
    )


def scaffold(directory: str | Path, name: str = "example_plugin", *,
             force: bool = False) -> Path:
    """Write a minimal working plugin, for ``magent plugins init``.

    Args:
        directory: parent directory to create the plugin in
        name: plugin name
        force: overwrite existing files

    Returns:
        The created plugin directory.

    Raises:
        PluginError: the directory exists and ``force`` was not given
    """
    target = Path(directory).expanduser() / name
    if target.exists() and not force:
        raise PluginError(f"{target} already exists; pass force=True to overwrite",
                          details={"path": str(target)})
    target.mkdir(parents=True, exist_ok=True)
    manifest = example_manifest(name)
    # The declared tool list gates registration, so it must name the tool the
    # template actually defines - otherwise a freshly scaffolded plugin loads
    # cleanly and registers nothing, which reads as a framework bug.
    manifest.tools = [f"{name}_hello"]
    manifest.hooks = ["before_run"]
    manifest.save(target)
    entry = target / "plugin.py"
    entry.write_text(_PLUGIN_TEMPLATE.format(name=name), encoding="utf-8")
    return target


_PLUGIN_TEMPLATE = '''"""The {name} plugin for MAgent."""

from __future__ import annotations

from typing import Any

from magent.plugins import hook
from magent.tools import tool

CALLS: list[str] = []
"""Recorded invocations, so the plugin can be inspected in tests."""


@tool(name="{name}_hello", category="meta", risk="safe",
      description="Greet by name, provided by the {name} plugin.")
def hello(name: str = "world") -> str:
    """Return a greeting.

    Args:
        name: who to greet
    """
    CALLS.append(name)
    return f"hello, {{name}}"


TOOLS = [hello]
"""Tools this plugin registers."""


@hook("before_run", plugin="{name}")
def record_run(payload: Any) -> None:
    """Observe each run without changing it."""
    CALLS.append("run")
'''


def describe_permissions() -> dict[str, str]:
    """Return each permission and what it allows, for ``magent plugins permissions``."""
    meanings = {
        "tools": "register new tools the agent can call",
        "hooks": "attach to agent lifecycle events",
        "filesystem.read": "read files inside the workspace",
        "filesystem.write": "create or modify files inside the workspace",
        "network": "make outbound network requests",
        "shell": "execute shell commands",
        "memory.read": "query long-term memory",
        "memory.write": "write to long-term memory",
        "secrets.read": "resolve configured secrets",
        "config.read": "read the resolved configuration",
    }
    return {name: meanings.get(name, "") for name in PERMISSIONS}


def sensitive_permissions() -> list[str]:
    """Return the permissions that need explicit operator approval."""
    return sorted(SENSITIVE_PERMISSIONS)


def write_manifest_template(path: str | Path, name: str = "example_plugin") -> Path:
    """Write a standalone ``plugin.json`` template."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_manifest(name).to_json() + "\n", encoding="utf-8")
    return target


def inspect_plugin(directory: str | Path) -> dict[str, Any]:
    """Validate a plugin directory and report what it declares.

    Useful before installing a plugin: it answers "what would this be allowed to
    do?" without importing any of its code.
    """
    manifest = PluginManifest.load(directory)
    entry = manifest.entry_path()
    return {
        "name": manifest.name,
        "version": manifest.version,
        "summary": manifest.summary,
        "author": manifest.author,
        "path": manifest.path,
        "entrypoint": manifest.entrypoint,
        "entrypoint_exists": entry.exists(),
        "permissions": list(manifest.permissions),
        "sensitive": manifest.sensitive_permissions,
        "declared_tools": list(manifest.tools),
        "declared_hooks": list(manifest.hooks),
        "checksum": manifest.checksum,
        "computed_checksum": manifest.compute_checksum(),
        "checksum_ok": manifest.verify_checksum(),
        "min_magent_version": manifest.min_magent_version,
    }


__all__ = [
    "DEFAULT_TIMEOUT",
    "MANIFEST_FILENAME",
    "PLUGIN_SUFFIXES",
    "LoadReport",
    "LoadedPlugin",
    "PluginLoader",
    "describe_permissions",
    "inspect_plugin",
    "loader_from_config",
    "scaffold",
    "sensitive_permissions",
    "write_manifest_template",
]
