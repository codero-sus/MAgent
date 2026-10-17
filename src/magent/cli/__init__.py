"""The MAgent command line interface.

Three modules make up the CLI:

- :mod:`magent.cli.args` resolves configuration and output settings once per
  invocation and hands every command a :class:`~magent.cli.args.CliContext`.
- :mod:`magent.cli.main` assembles the parser, dispatches to a command, and owns
  the exit-code contract.
- :mod:`magent.cli.commands` holds one module per command group; each exposes a
  ``register(subparsers)`` function, so adding a command means adding a module
  rather than editing a growing dispatch chain.

Run it as ``magent`` after installing, or as ``python -m magent`` from a source
checkout.
"""

from __future__ import annotations

from .args import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    GLOBAL_FLAGS,
    CliContext,
    add_global_flags,
    build_context,
    confirm,
    fail,
    handle_error,
    parse_overrides,
    read_prompt_text,
)
from .main import DESCRIPTION, EPILOG, PROG, build_parser, command_names, main

__all__ = [
    "DESCRIPTION",
    "EPILOG",
    "EXIT_CONFIG",
    "EXIT_ERROR",
    "EXIT_INTERRUPT",
    "EXIT_OK",
    "EXIT_USAGE",
    "GLOBAL_FLAGS",
    "PROG",
    "CliContext",
    "add_global_flags",
    "build_context",
    "build_parser",
    "command_names",
    "confirm",
    "fail",
    "handle_error",
    "main",
    "parse_overrides",
    "read_prompt_text",
]
