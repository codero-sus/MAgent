"""Argument parsing for the CLI.

Two details make the difference between a parser that works and one that fights
the user:

* shared flags are accepted *both* before and after the subcommand
  (``magent --format json tools`` and ``magent tools --format json``), which is
  done by giving the subparsers a suppressed-default copy of the common flags so
  they only set a value when the user actually typed it;
* every command's help includes its examples, because ``magent eval --help``
  should show how to run an evaluation rather than just listing flag names.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .._version import __version__
from .commands import Command, build_groups, command_names
from .commands.registry import all_commands

PROGRAM = "magent"
"""Program name shown in usage."""

DESCRIPTION = "MAgent - Mr. Agent, Powerful. A local-first agent framework for Cortex LLMHoster."
"""Top-level description."""

COMMON_FLAGS: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (("--format", "-f"), {"default": "", "help": "output format: text|json|jsonl|yaml|csv|md"}),
    (("--json",), {"action": "store_true", "help": "shorthand for --format json"}),
    (("--jsonl",), {"action": "store_true", "help": "shorthand for --format jsonl"}),
    (("--yaml",), {"action": "store_true", "help": "shorthand for --format yaml"}),
    (("--quiet", "-q"), {"action": "store_true", "help": "suppress non-essential output"}),
    (("--verbose", "-v"), {"action": "store_true", "help": "more detail on stderr"}),
    (("--no-color",), {"action": "store_true", "help": "disable ANSI colour"}),
    (("--color",), {"action": "store_true", "help": "force ANSI colour"}),
    (("--ascii",), {"action": "store_true", "help": "ASCII-only output"}),
    (("--width",), {"type": int, "default": 0, "help": "output width in columns"}),
    (
        ("--output-file",),
        {
            "dest": "output_file",
            "default": "",
            "metavar": "PATH",
            "help": "also write rendered output to this file",
        },
    ),
)
"""Flags shared by every command."""


GLOBAL_FLAGS: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (("--config",), {"default": "", "metavar": "PATH", "help": "configuration file to load"}),
    (("--profile",), {"default": "", "help": "configuration profile to activate"}),
    (("--workspace",), {"default": "", "metavar": "DIR", "help": "workspace root"}),
    (("--backend",), {"default": "", "help": "backend to use (auto|cortex|echo|scripted)"}),
    (("--model",), {"default": "", "help": "model id override"}),
    (("--offline",), {"action": "store_true", "help": "force the offline echo backend"}),
    (
        ("--set",),
        {
            "action": "append",
            "default": [],
            "metavar": "KEY=VALUE",
            "help": "configuration override (repeatable)",
        },
    ),
    (("--log-level",), {"dest": "log_level", "default": "", "help": "log level"}),
)
"""Flags accepted before *and* after the subcommand.

They are declared on the root parser with real defaults and repeated on every
subparser with ``SUPPRESS`` defaults, so ``magent --offline run x`` and
``magent run x --offline`` both work while neither placement clobbers the other.
"""


def _add_flags(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool,
    groups: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (),
) -> None:
    """Add shared flags to a parser, optionally suppressing their defaults."""
    for names, kwargs in (*COMMON_FLAGS, *groups):
        options = dict(kwargs)
        if suppress_defaults:
            options["default"] = argparse.SUPPRESS
        parser.add_argument(*names, **options)


def build_common_parser(
    *, suppress_defaults: bool = False, include_globals: bool = True
) -> argparse.ArgumentParser:
    """Build a parser holding only the shared flags.

    Args:
        suppress_defaults: use ``SUPPRESS`` so the flags only set a value when typed
        include_globals: also accept ``--offline``/``--model``/``--workspace`` style
            flags after the subcommand
    """
    parser = argparse.ArgumentParser(add_help=False)
    _add_flags(
        parser, suppress_defaults=suppress_defaults, groups=GLOBAL_FLAGS if include_globals else ()
    )
    return parser


def build_parser(
    commands: Sequence[Command] | None = None, *, prog: str = PROGRAM
) -> argparse.ArgumentParser:
    """Build the full argument parser.

    Args:
        commands: commands to register; every registered command when omitted
        prog: program name used in usage strings
    """
    items = list(commands if commands is not None else all_commands())
    parser = argparse.ArgumentParser(
        prog=prog,
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(items),
    )
    parser.add_argument("--version", action="version", version=f"{prog} {__version__}")
    _add_flags(parser, suppress_defaults=False, groups=GLOBAL_FLAGS)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    sub_common = build_common_parser(suppress_defaults=True)
    for item in items:
        sub = subparsers.add_parser(
            item.name,
            help=item.help,
            description=item.description,
            epilog=item.epilog or None,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[sub_common],
            aliases=list(item.aliases),
        )
        sub.set_defaults(_command=item)
        if item.configure is not None:
            item.configure(sub)
    subparsers.add_parser(
        "help",
        help="show this help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[sub_common],
    )
    return parser


def _epilog(commands: Sequence[Command]) -> str:
    """Render the grouped command list for ``--help``."""
    lines = ["commands:"]
    for group in build_groups(commands):
        visible = [item for item in group.commands if not item.hidden]
        if not visible:
            continue
        names = "  ".join(sorted(item.name for item in visible))
        lines.append(f"  {group.name:<11} {names}")
    lines.append("")
    lines.append("run `magent <command> --help` for options and examples")
    return "\n".join(lines)


def parse_args(
    argv: Sequence[str] | None = None, *, commands: Sequence[Command] | None = None
) -> argparse.Namespace:
    """Parse ``argv`` and return the namespace."""
    parser = build_parser(commands)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "command", "") == "help":
        parser.print_help()
        raise SystemExit(0)
    return args


def command_help(name: str, *, prog: str = PROGRAM) -> str:
    """Return one command's help text without exiting.

    Raises:
        SystemExit: never; unknown names produce a plain message
    """
    import contextlib
    import io

    parser = build_parser(prog=prog)
    buffer = io.StringIO()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            chosen = action.choices.get(name)
            if chosen is None:
                return f"unknown command {name!r}; try `{prog} --help`"
            with contextlib.redirect_stdout(buffer):
                chosen.print_help()
            return buffer.getvalue()
    return f"no commands are registered in {prog}"


def known_commands(commands: Sequence[Command] | None = None) -> list[str]:
    """Every command name and alias."""
    return command_names(list(commands if commands is not None else all_commands()))


def completions(shell: str = "bash", *, prog: str = PROGRAM) -> str:
    """Emit a completion script for bash or zsh.

    The script is generated from the live command table, so it never drifts from
    the parser.
    """
    names = " ".join(sorted({name for name in known_commands() if len(name) > 1}))
    if shell == "zsh":
        return (
            f"#compdef {prog}\n"
            f"_{prog}() {{\n"
            f"  local -a commands\n"
            f"  commands=({names})\n"
            f"  _describe 'command' commands\n"
            f"}}\n"
            f"compdef _{prog} {prog}\n"
        )
    return (
        f"# bash completion for {prog}\n"
        f"_{prog}_complete() {{\n"
        f"  local cur=${{COMP_WORDS[COMP_CWORD]}}\n"
        f'  COMPREPLY=( $(compgen -W "{names}" -- "$cur") )\n'
        f"}}\n"
        f"complete -F _{prog}_complete {prog}\n"
    )


__all__ = [
    "COMMON_FLAGS",
    "DESCRIPTION",
    "GLOBAL_FLAGS",
    "PROGRAM",
    "build_common_parser",
    "build_parser",
    "command_help",
    "completions",
    "known_commands",
    "parse_args",
]
