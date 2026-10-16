"""The command line interface: 26 subcommands over one shared context.

Public API:

:func:`main` / :func:`run`
    the entry points wired to the ``magent`` console script;
:class:`CliContext`
    lazy access to the backend, tools, memory, sessions, guardrails, retrieval,
    skills and observability, shared by every command;
:class:`Command` / :func:`all_commands`
    the registry each command module contributes to;
:func:`build_parser`
    the argparse tree, built from that registry.

>>> from magent.cli import all_commands
>>> "doctor" in [item.name for item in all_commands()]
True
"""

from __future__ import annotations

from .commands import Command, CommandGroup, build_groups, command, describe_commands
from .commands.registry import COMMAND_MODULES, all_commands, find
from .context import CliContext, build_context, offline_context
from .main import (ERROR_EXIT_CODES, EXIT_BACKEND, EXIT_CONFIG, EXIT_FAILURE, EXIT_GUARDRAIL,
                   EXIT_INTERRUPTED, EXIT_OK, EXIT_PERMISSION, EXIT_USAGE, exit_code_for, main,
                   run)
from .parser import build_parser, command_help, completions, known_commands, parse_args
from .repl import Repl, SlashCommand, run_repl

__all__ = [
    "COMMAND_MODULES",
    "ERROR_EXIT_CODES",
    "EXIT_BACKEND",
    "EXIT_CONFIG",
    "EXIT_FAILURE",
    "EXIT_GUARDRAIL",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_PERMISSION",
    "EXIT_USAGE",
    "CliContext",
    "Command",
    "CommandGroup",
    "Repl",
    "SlashCommand",
    "all_commands",
    "build_context",
    "build_groups",
    "build_parser",
    "command",
    "command_help",
    "completions",
    "describe_commands",
    "exit_code_for",
    "find",
    "known_commands",
    "main",
    "offline_context",
    "parse_args",
    "run",
    "run_repl",
]
