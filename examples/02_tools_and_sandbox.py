#!/usr/bin/env python3
"""Example 02: tools, the sandbox and approvals.

Run: ``python examples/02_tools_and_sandbox.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import heading, main_guard, workspace
from magent.config import default_config
from magent.errors import PermissionDeniedError
from magent.tools import ToolContext, build_sandbox, default_registry


def body() -> None:
    """List tools, call one, and show the sandbox refusing a path."""
    registry = default_registry()
    heading(f"{registry.count()} tools in {len(registry.categories())} categories")
    for category, tools in sorted(registry.by_category().items()):
        print(
            f"  {category:<12} {len(tools):>3}  {', '.join(item.spec.name for item in tools[:3])}"
        )
        del tools

    root = workspace()
    config = default_config()
    config.tools.workspace = str(root)
    context = ToolContext(
        config=config, registry=registry, sandbox=build_sandbox(config, workspace=root)
    )

    heading("calling a tool")
    result = registry.get("calculate").run({"expression": "6*7"}, context=context)
    print("calculate 6*7 ->", result.content)

    written = registry.get("write_file").run(
        {"path": str(root / "notes.txt"), "content": "deploy on fridays at 15:00 UTC"},
        context=context,
    )
    print("write_file ->", written.ok, written.error or "")

    read = registry.get("read_file").run({"path": str(root / "notes.txt")}, context=context)
    print("read_file ->", read.content["content"])

    heading("the sandbox refusing a path outside the workspace")
    sandbox = context.sandbox
    try:
        sandbox.check_read("/etc/hostname")
    except PermissionDeniedError as exc:
        print("denied:", exc)

    refused = registry.get("run_command").run({"command": "rm -rf /"}, context=context)
    print("run_command rm -rf / ->", refused.ok, refused.error)

    refused = registry.get("read_file").run({"path": "/etc/hostname"}, context=context)
    print("read_file /etc/hostname ->", refused.ok, refused.error_code)


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
