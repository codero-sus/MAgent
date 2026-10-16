"""Plugins: load third-party tools, hooks, personas, templates and skills.

Public API:

:class:`PluginManifest` / :func:`manifest_for`
    metadata, validation and checksums before any code runs;
:class:`PluginLoader` / :func:`load_plugins`
    discovery, sandboxed import and registration from plugin directories;
:class:`HookRegistry` / :func:`attach_to_agent`
    named extension points (``before_prompt``, ``after_tool``, ...) with ordered,
    isolated handlers.

>>> from magent.plugins import HookRegistry
>>> hooks = HookRegistry()
>>> _ = hooks.register("after_run", lambda payload, **kw: "noted")
>>> hooks.apply("after_run", {})
'noted'
"""

from __future__ import annotations

from .hooks import (DEFAULT_PRIORITY, MAX_HANDLERS_PER_HOOK, HookHandler, HookRegistry,
                    attach_to_agent, chain_handlers, describe_hooks, load_hooks_from)
from .loader import (MAX_ENTRY_BYTES, PLUGIN_DIRS, LoadedPlugin, PluginLoader,
                     default_plugin_paths, iter_plugin_files, load_plugins, plugin_module)
from .manifest import (CAPABILITIES, HOOK_POINTS, MANIFEST_NAMES, PluginManifest, find_manifest,
                       manifest_for, manifests_from, verify_checksum, write_manifest)

__all__ = [
    "CAPABILITIES",
    "DEFAULT_PRIORITY",
    "HOOK_POINTS",
    "MANIFEST_NAMES",
    "MAX_ENTRY_BYTES",
    "MAX_HANDLERS_PER_HOOK",
    "PLUGIN_DIRS",
    "HookHandler",
    "HookRegistry",
    "LoadedPlugin",
    "PluginLoader",
    "PluginManifest",
    "attach_to_agent",
    "chain_handlers",
    "default_plugin_paths",
    "describe_hooks",
    "find_manifest",
    "iter_plugin_files",
    "load_hooks_from",
    "load_plugins",
    "manifest_for",
    "manifests_from",
    "plugin_module",
    "verify_checksum",
    "write_manifest",
]
