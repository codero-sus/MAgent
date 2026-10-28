"""Aggregate every command module into one ordered registry.

Importing the modules here (and nowhere else) keeps the parser, the help text
and ``magent commands`` in agreement: there is exactly one list of commands, and
adding a module means adding one import.
"""

from __future__ import annotations

from . import Command
from .agent_cmds import COMMANDS as AGENT_COMMANDS
from .desktop_cmds import COMMANDS as DESKTOP_COMMANDS
from .knowledge_cmds import COMMANDS as KNOWLEDGE_COMMANDS
from .quality_cmds import COMMANDS as QUALITY_COMMANDS
from .server_cmds import COMMANDS as SERVER_COMMANDS
from .system_cmds import COMMANDS as SYSTEM_COMMANDS
from .tools_cmds import COMMANDS as TOOLS_COMMANDS

COMMAND_MODULES: tuple[str, ...] = (
    "agent_cmds",
    "desktop_cmds",
    "knowledge_cmds",
    "tools_cmds",
    "system_cmds",
    "quality_cmds",
    "server_cmds",
)
"""Command modules, in the order they appear in help output."""


def all_commands() -> list[Command]:
    """Every registered command, deduplicated by name.

    A later module may replace a command of the same name, which is how a
    distribution can shadow a built-in subcommand.
    """
    merged: dict[str, Command] = {}
    for group in (
        AGENT_COMMANDS,
        DESKTOP_COMMANDS,
        KNOWLEDGE_COMMANDS,
        TOOLS_COMMANDS,
        SYSTEM_COMMANDS,
        QUALITY_COMMANDS,
        SERVER_COMMANDS,
    ):
        for item in group:
            merged[item.name] = item
    return list(merged.values())


def find(name: str) -> Command | None:
    """Find a command by name or alias."""
    from . import resolve_command

    return resolve_command(all_commands(), name)


__all__ = [
    "COMMAND_MODULES",
    "all_commands",
    "find",
]
