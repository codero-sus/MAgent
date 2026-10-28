"""The command registry: one declarative object per CLI subcommand.

Each command module exports :data:`COMMANDS`, a tuple of :class:`Command`
objects.  A command owns its arguments (``configure``) and its behaviour
(``run``), so adding a subcommand means adding one object to one tuple - the
parser, the help output and ``magent --help`` all follow from it, and the tests
can call a command's ``run`` directly without going through ``argv``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

CommandRunner = Callable[[Any, argparse.Namespace], int]
"""Signature of a command's implementation."""


@dataclass(slots=True)
class Command:
    """One CLI subcommand."""

    name: str
    help: str
    run: CommandRunner
    description: str = ""
    aliases: tuple[str, ...] = ()
    group: str = "general"
    needs_backend: bool = False
    configure: Callable[[argparse.ArgumentParser], None] | None = None
    examples: tuple[str, ...] = ()
    hidden: bool = False

    def __post_init__(self) -> None:
        """Normalise aliases and default the description to the help text."""
        if isinstance(self.aliases, list):
            self.aliases = tuple(self.aliases)
        if not self.description:
            self.description = self.help

    @property
    def epilog(self) -> str:
        """Examples rendered into the command's help."""
        if not self.examples:
            return ""
        lines = ["examples:"] + [f"  {example}" for example in self.examples]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the command for ``magent commands --format json``."""
        return {
            "name": self.name,
            "help": self.help,
            "description": self.description,
            "aliases": list(self.aliases),
            "group": self.group,
            "needs_backend": self.needs_backend,
            "examples": list(self.examples),
            "hidden": self.hidden,
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"Command(name={self.name!r}, group={self.group!r})"


def command(name: str, help: str, **kwargs: Any) -> Callable[[CommandRunner], Command]:
    """Decorator turning a function into a :class:`Command`."""

    def decorate(fn: CommandRunner) -> Command:
        return Command(name=name, help=help, run=fn, **kwargs)

    return decorate


@dataclass(slots=True)
class CommandGroup:
    """A named set of commands, used to order help output."""

    name: str
    description: str
    commands: list[Command] = field(default_factory=list)

    def add(self, command_: Command) -> Command:
        """Add a command to the group."""
        self.commands.append(command_)
        return command_

    def __len__(self) -> int:
        """Number of commands in the group."""
        return len(self.commands)


GROUPS: tuple[tuple[str, str], ...] = (
    ("agent", "run the agent"),
    ("desktop", "desktop automation, queue and floating control"),
    ("knowledge", "memory, retrieval and sessions"),
    ("models", "backends, models and Cortex control"),
    ("tools", "tools, presets and skills"),
    ("system", "configuration, plugins, diagnostics"),
    ("server", "the HTTP API and web UI"),
    ("quality", "evaluation and improvements"),
)
"""Groups used to organise ``magent --help``."""


def build_groups(commands: Iterable[Command]) -> list[CommandGroup]:
    """Group commands in :data:`GROUPS` order, unknown groups last."""
    known = {name for name, _ in GROUPS}
    groups: dict[str, CommandGroup] = {
        name: CommandGroup(name=name, description=description) for name, description in GROUPS
    }
    extra = CommandGroup(name="other", description="everything else")
    for item in commands:
        target = groups.get(item.group) or (
            extra if item.group not in known else groups[item.group]
        )
        target.add(item)
    result = [groups[name] for name, _ in GROUPS]
    if extra.commands:
        result.append(extra)
    return result


def resolve_command(commands: Sequence[Command], name: str) -> Command | None:
    """Find a command by name or alias."""
    for item in commands:
        if item.name == name or name in item.aliases:
            return item
    return None


def command_names(commands: Sequence[Command]) -> list[str]:
    """Every command name and alias, sorted."""
    names = {item.name for item in commands}
    for item in commands:
        names.update(item.aliases)
    return sorted(names)


def describe_commands(commands: Sequence[Command], *, include_hidden: bool = False) -> str:
    """Render the command list, grouped, for ``magent help``."""
    lines: list[str] = []
    for group in build_groups(commands):
        visible = [item for item in group.commands if include_hidden or not item.hidden]
        if not visible:
            continue
        lines.append(f"{group.name} - {group.description}")
        for item in sorted(visible, key=lambda entry: entry.name):
            lines.append(f"  {item.name:<14} {item.help}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "GROUPS",
    "Command",
    "CommandGroup",
    "CommandRunner",
    "build_groups",
    "command",
    "command_names",
    "describe_commands",
    "resolve_command",
]
