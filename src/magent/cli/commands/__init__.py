"""Command groups for the MAgent CLI.

Each module here owns one group of subcommands and exposes the same two things:
a ``register(subparsers)`` function that adds them to the parser, and a
``cmd_<group>(args, context)`` handler that dispatches the chosen action.
:mod:`magent.cli.main` imports the modules and calls ``register`` in order, so
the help listing follows this package's import order.

Handlers never call ``sys.exit``. They return an exit code, which keeps the CLI
testable - a test can assert on the returned integer instead of catching
``SystemExit`` - and lets :func:`magent.cli.main.main` own the one place where a
process actually terminates.
"""

from __future__ import annotations

from typing import Any

from . import (config_cmd, cortex, doctor, eval_cmd, memory, misc, plugins, rag,
               run, serve, sessions, tools)

GROUPS: tuple[str, ...] = (
    "run", "chat", "tools", "config", "doctor", "version",
    "cortex", "memory", "rag", "sessions", "eval", "serve", "plugins",
    "improvements",
)
"""Command names this package registers, in help order."""

MODULES: tuple[Any, ...] = (run, tools, config_cmd, doctor, cortex, memory, rag,
                            sessions, eval_cmd, serve, plugins, misc)
"""Command modules, in registration order."""


def register_all(subparsers: Any) -> list[str]:
    """Register every command group, returning the names added.

    Args:
        subparsers: the parser's subcommand action
    """
    registered: list[str] = []
    for module in MODULES:
        register = getattr(module, "register", None)
        if register is None:
            continue
        register(subparsers)
        registered.extend(getattr(module, "COMMAND_NAMES", []) or [])
    doctor.register_version(subparsers)
    return [name for name in GROUPS if name in registered or name == "version"]


__all__ = ["GROUPS", "MODULES", "register_all"]
