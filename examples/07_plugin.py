"""Scaffold, inspect, permission-check, and load a real plugin.

Usage:
    python3 examples/07_plugin.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from magent.plugins import HookRegistry, PluginLoader, inspect_plugin, scaffold
from magent.tools import ToolContext, ToolRegistry
from magent.types import ToolCall


def main() -> None:
    """Load the generated tool and invoke it through the registry."""
    with tempfile.TemporaryDirectory(prefix="magent-plugin-example-") as temporary:
        parent = Path(temporary)
        plugin_dir = scaffold(parent, "weather")
        print("inspect:", inspect_plugin(plugin_dir))

        registry = ToolRegistry()
        hooks = HookRegistry()
        loader = PluginLoader([parent], registry=registry, hooks=hooks)
        report = loader.load_all()
        print(report.render())

        result = registry.dispatch(
            ToolCall(id="example", name="weather_hello", arguments={"name": "Gurugram"}),
            ToolContext(workspace=ROOT),
        )
        print("tool:", result.content)
        print("hooks:", hooks.describe())
        loader.unload_all()


if __name__ == "__main__":
    main()
