"""The CLI entry point: parse, build a context, run the command, exit clean.

``main`` is deliberately thin.  It parses arguments, resolves the command from the
registry, builds a :class:`~magent.cli.context.CliContext` (which loads the
configuration but touches nothing expensive), runs the command and maps failures
onto exit codes:

===  =========================================================
0    success
1    the command ran and reported failure (a failing eval, doctor)
2    usage error (bad arguments, nothing to do)
3    configuration error
4    backend or model unavailable
5    permission, sandbox or approval refusal
6    guardrail block or budget exhaustion
130  interrupted (Ctrl-C)
===  =========================================================

Every command returns an integer, so ``sys.exit(main())`` is all a wrapper needs.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Sequence

from .._version import __version__
from ..errors import MAgentError
from ..logging_ import get_logger, setup_logging
from .commands import Command
from .commands.registry import all_commands
from .context import CliContext
from .parser import build_parser

LOGGER = get_logger("cli.main")
"""Logger for CLI-level failures."""

EXIT_OK = 0
"""Success."""

EXIT_FAILURE = 1
"""The command ran and reported failure."""

EXIT_USAGE = 2
"""Bad arguments or nothing to do."""

EXIT_CONFIG = 3
"""Configuration problem."""

EXIT_BACKEND = 4
"""Backend or model unavailable."""

EXIT_PERMISSION = 5
"""Permission, sandbox or approval refusal."""

EXIT_GUARDRAIL = 6
"""Guardrail block or budget exhaustion."""

EXIT_INTERRUPTED = 130
"""Interrupted by the user."""

ERROR_EXIT_CODES: dict[str, int] = {
    "config_error": EXIT_CONFIG,
    "config_validation_error": EXIT_CONFIG,
    "config_file_not_found": EXIT_CONFIG,
    "secret_resolution_error": EXIT_CONFIG,
    "backend_error": EXIT_BACKEND,
    "backend_unavailable": EXIT_BACKEND,
    "connection_error": EXIT_BACKEND,
    "timeout": EXIT_BACKEND,
    "rate_limited": EXIT_BACKEND,
    "rate_limit_exceeded": EXIT_GUARDRAIL,
    "model_not_found": EXIT_BACKEND,
    "model_not_running": EXIT_BACKEND,
    "capability_not_supported": EXIT_BACKEND,
    "permission_denied": EXIT_PERMISSION,
    "sandbox_violation": EXIT_PERMISSION,
    "approval_required": EXIT_PERMISSION,
    "guardrail_error": EXIT_GUARDRAIL,
    "prompt_injection": EXIT_GUARDRAIL,
    "content_blocked": EXIT_GUARDRAIL,
    "policy_violation": EXIT_GUARDRAIL,
    "budget_exceeded": EXIT_GUARDRAIL,
    "user_abort": EXIT_INTERRUPTED,
    "cancelled": EXIT_INTERRUPTED,
}
"""MAgent error code to process exit code."""


def resolve_command(args: Any, commands: Sequence[Command]) -> Command | None:
    """Find the command the arguments selected."""
    explicit = getattr(args, "_command", None)
    if isinstance(explicit, Command):
        return explicit
    name = getattr(args, "command", "") or ""
    for item in commands:
        if item.name == name or name in item.aliases:
            return item
    return None


def main(argv: Sequence[str] | None = None, *, commands: Sequence[Command] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: arguments without the program name; ``sys.argv[1:]`` when omitted
        commands: command table override, used by the tests

    Returns:
        the process exit code
    """
    items = list(commands if commands is not None else all_commands())
    parser = build_parser(items)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "log_level", ""):
        os.environ["MAGENT_LOG_LEVEL"] = str(args.log_level)
    setup_logging(level=os.environ.get("MAGENT_LOG_LEVEL", "WARNING"),
                  force=False, quiet=bool(getattr(args, "quiet", False)))
    if not getattr(args, "command", ""):
        parser.print_help()
        return EXIT_USAGE
    if args.command == "help":
        parser.print_help()
        return EXIT_OK

    command = resolve_command(args, items)
    if command is None:
        print(f"{parser.prog}: unknown command {args.command!r}", file=sys.stderr)
        return EXIT_USAGE

    context: CliContext | None = None
    try:
        context = CliContext.build(args)
    except MAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return exit_code_for(exc)
    except Exception as exc:  # noqa: BLE001 - never traceback at the user by default
        print(f"error: could not load configuration: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return EXIT_CONFIG

    try:
        return int(command.run(context, args))
    except KeyboardInterrupt:
        if not context.quiet:
            print("", file=sys.stderr)
            print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except MAgentError as exc:
        context.error(str(exc))
        if getattr(args, "verbose", False):
            for key, value in (getattr(exc, "details", None) or {}).items():
                print(f"  {key}: {value}", file=sys.stderr)
            if not getattr(exc, "retryable", False):
                traceback.print_exc()
        return exit_code_for(exc)
    except BrokenPipeError:
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - a crash is a bug report, not a traceback dump
        context.error(f"unexpected {type(exc).__name__}: {exc}")
        print("this is a bug; re-run with --verbose for the traceback", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return EXIT_FAILURE
    finally:
        if command.name not in {"serve", "chat"}:
            _close_quietly(context)


def _close_quietly(context: CliContext | None) -> None:
    """Release the context without letting a close failure change the exit code."""
    if context is None:
        return
    try:
        context.close()
    except Exception as exc:  # noqa: BLE001 - closing must not raise
        LOGGER.debug("context close failed: %s", exc)


def exit_code_for(exc: BaseException) -> int:
    """Map an exception onto a process exit code."""
    code = str(getattr(exc, "code", "") or "")
    if code in ERROR_EXIT_CODES:
        return ERROR_EXIT_CODES[code]
    declared = int(getattr(exc, "exit_code", 0) or 0)
    if declared:
        return declared
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    if isinstance(exc, (ValueError, TypeError)):
        return EXIT_USAGE
    return EXIT_FAILURE


def run() -> None:
    """Console-script wrapper: raise ``SystemExit`` with the command's code."""
    sys.exit(main())


def version() -> str:
    """Version string used by packaging and ``--version``."""
    return __version__


__all__ = [
    "ERROR_EXIT_CODES",
    "EXIT_BACKEND",
    "EXIT_CONFIG",
    "EXIT_FAILURE",
    "EXIT_GUARDRAIL",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_PERMISSION",
    "EXIT_USAGE",
    "exit_code_for",
    "main",
    "resolve_command",
    "run",
    "version",
]
