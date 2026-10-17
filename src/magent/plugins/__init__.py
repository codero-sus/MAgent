"""Plugins: extend MAgent with third-party tools and lifecycle hooks.

A plugin is a directory containing ``plugin.json`` and an entry point. The
manifest declares what the plugin wants - tools, hooks, permissions - and the
loader grants only what was declared::

    from magent.plugins import PluginLoader, HookRegistry
    from magent.tools import ToolRegistry

    registry = ToolRegistry()
    loader = PluginLoader(["~/.local/share/magent/plugins"],
                          registry=registry, hooks=HookRegistry(),
                          allow_sensitive=["network"])
    report = loader.load_all()
    print(report.render())

Nothing here makes third-party code safe. Importing a plugin executes it, so the
defences are procedural and explicit: discovery only in configured directories,
deny-by-default on sensitive permissions, refusal of entry points that resolve
outside the plugin folder, optional SHA-256 integrity checks, and containment of
failures so one broken plugin does not disable the rest. Only install plugins
you trust.
"""

from __future__ import annotations

from .hooks import (
    TRANSFORM_HOOKS,
    VETO_HOOKS,
    HookFn,
    HookRegistration,
    HookRegistry,
    HookResult,
    collect_hooks,
    hook,
    known_hooks,
    register_decorated,
)
from .loader import (
    DEFAULT_TIMEOUT,
    PLUGIN_SUFFIXES,
    LoadReport,
    LoadedPlugin,
    PluginLoader,
    describe_permissions,
    inspect_plugin,
    loader_from_config,
    scaffold,
    sensitive_permissions,
    write_manifest_template,
)
from .manifest import (
    HOOK_NAMES,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    PERMISSIONS,
    SENSITIVE_PERMISSIONS,
    PluginManifest,
    example_manifest,
    validate_many,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "HOOK_NAMES",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "PERMISSIONS",
    "PLUGIN_SUFFIXES",
    "SENSITIVE_PERMISSIONS",
    "TRANSFORM_HOOKS",
    "VETO_HOOKS",
    "HookFn",
    "HookRegistration",
    "HookRegistry",
    "HookResult",
    "LoadReport",
    "LoadedPlugin",
    "PluginLoader",
    "PluginManifest",
    "collect_hooks",
    "describe_permissions",
    "example_manifest",
    "hook",
    "inspect_plugin",
    "known_hooks",
    "loader_from_config",
    "register_decorated",
    "scaffold",
    "sensitive_permissions",
    "validate_many",
    "write_manifest_template",
]
