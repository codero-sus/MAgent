"""The ``magent`` command line.

One parser, one dispatch table, one exit-code contract. Subcommands live in
:mod:`magent.cli.commands` and register themselves here, so adding a command
means adding a module - not editing a growing ``if`` chain.

The exit codes are part of the interface, not an afterthought: ``0`` success,
``1`` error, ``2`` bad usage, ``130`` interrupted, and ``3``-``7`` for the
distinct ways an agent run can stop (iteration limit, budget, guardrail, needs
input, tool limit). A script wrapping MAgent can therefore branch on *why* a run
ended rather than parsing prose.

Running with no arguments prints help rather than failing, because that is what
someone trying the tool for the first time wants.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from .._version import __version__, version_string
from ..errors import MAgentError
from .args import (
    EXIT_ERROR,
    EXIT_INTERRUPT,
    EXIT_OK,
    EXIT_USAGE,
    CliContext,
    add_global_flags,
    build_context,
    handle_error,
)
from .commands import cortex as cortex_cmd
from .commands import doctor as doctor_cmd
from .commands import memory as memory_cmd
from .commands import misc as misc_cmd
from .commands import rag as rag_cmd
from .commands import run as run_cmd
from .commands import sessions as sessions_cmd
from .commands import tools as tools_cmd
from .commands import config_cmd

PROG = "magent"
"""Program name shown in help output."""

DESCRIPTION = """\
MAgent - Mr. Agent, Powerful.

An agent framework and CLI for Cortex LLMHoster: it plans, calls tools,
retrieves from your files, remembers across runs, and guards its own inputs and
outputs.

Run 'magent doctor' to check the installation, or 'magent <command> --help'
for one command's options.
"""
"""Top-level help text."""

EPILOG = """\
examples:
  magent run "summarize the README and list the risks"
  magent run --tools read_file,grep -f task.md --format json
  magent chat --resume
  magent rag ingest docs --save .magent/index.json
  magent rag --index .magent/index.json ask "how do approvals work?" --check
  magent tools list --max-risk read
  magent cortex models
  magent doctor --strict

Every command accepts the global options below; -s key=value overrides one
configuration value for that invocation, e.g. -s agent.max_iterations=5.
"""
"""Examples shown at the end of help."""

COMMAND_MODULES: tuple[Any, ...] = (
    run_cmd,
    tools_cmd,
    config_cmd,
    doctor_cmd,
    cortex_cmd,
    memory_cmd,
    rag_cmd,
    sessions_cmd,
    misc_cmd,
)
"""Modules whose ``register`` adds subcommands, in help order."""


def _global_parent() -> argparse.ArgumentParser:
    """Return a help-less parser holding the global flags.

    It is used as a ``parents=`` source so every subcommand accepts them too.
    Building it once and reusing it keeps one definition of the global surface.
    """
    global cached_parent
    if cached_parent is None:
        cached_parent = argparse.ArgumentParser(add_help=False)
        add_global_flags(cached_parent)
    return cached_parent


cached_parent: argparse.ArgumentParser | None = None
"""Memoized global-flag parent parser."""


class _InjectingSubParsers(argparse._SubParsersAction):
    """Subparser action that gives every command the global flags.

    Without this, ``magent tools list --offline`` is rejected because argparse
    only knows ``--offline`` on the root parser - so the accepted position of a
    flag would depend on where the user happened to type it.
    """

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Create a subparser that inherits the global flags."""
        parents = list(kwargs.pop("parents", None) or [])
        parent = _global_parent()
        if parent not in parents:
            parents.append(parent)
        kwargs["parents"] = parents
        # ``parser_class`` is not an add_parser argument: each subparser is
        # already a _HelpfulParser, whose __init__ registers this same action
        # class, so nested subcommands inherit the globals without it.
        return super().add_parser(name, **kwargs)


class _HelpfulParser(argparse.ArgumentParser):
    """Parser that prints full help on an error instead of one terse line."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the parser and make nested subcommands inherit globals too."""
        super().__init__(*args, **kwargs)
        self.register("action", "parsers", _InjectingSubParsers)

    def error(self, message: str) -> Any:
        """Report a usage error with the help text alongside it."""
        sys.stderr.write(f"{self.prog}: error: {message}\n\n")
        self.print_help(sys.stderr)
        raise SystemExit(EXIT_USAGE)


def build_parser() -> argparse.ArgumentParser:
    """Assemble the complete argument parser."""
    parser = _HelpfulParser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=version_string(),
                        help="print the version and exit")
    add_global_flags(parser)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND",
                                       parser_class=_HelpfulParser)
    for module in COMMAND_MODULES:
        register = getattr(module, "register", None)
        if register is None:
            continue
        register(subparsers)
    doctor_cmd.register_version(subparsers)
    return parser


def command_names() -> list[str]:
    """Return every registered command name, for help and completion."""
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    return []


def _apply_global_only(args: argparse.Namespace) -> argparse.Namespace:
    """Lift global flags given after a subcommand up to the namespace.

    ``argparse`` puts a subcommand's own values last, so ``magent run -q "x"``
    and ``magent -q run "x"`` should behave identically. Subcommand parsers
    inherit the global flags, which means the parsed value can land in either
    position; this normalizes it.
    """
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Args:
        argv: arguments to parse, defaulting to ``sys.argv[1:]``
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help(sys.stdout)
        return EXIT_OK
    try:
        args = _apply_global_only(parser.parse_args(arguments))
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_USAGE

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stdout)
        return EXIT_OK

    context: CliContext | None = None
    try:
        context = build_context(args, command=str(getattr(args, "command", "") or ""))
        return int(handler(args, context) or EXIT_OK)
    except KeyboardInterrupt:
        if context is not None:
            context.printer.write("")
            context.printer.warn("interrupted")
        else:
            print("", file=sys.stderr)
        return EXIT_INTERRUPT
    except MAgentError as exc:
        return handle_error(exc, context)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        if code is None:
            return EXIT_OK
        if context is not None:
            context.printer.error(str(code))
        else:
            print(str(code), file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - the CLI must always exit with a code
        return handle_error(exc, context)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COMMAND_MODULES",
    "DESCRIPTION",
    "EPILOG",
    "PROG",
    "build_parser",
    "command_names",
    "main",
]
