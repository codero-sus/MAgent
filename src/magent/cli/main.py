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
from .commands import eval_cmd
from .commands import memory as memory_cmd
from .commands import misc as misc_cmd
from .commands import plugins as plugins_cmd
from .commands import rag as rag_cmd
from .commands import run as run_cmd
from .commands import serve as serve_cmd
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
  magent run --tools read_file,grep --file task.md --format json
  magent chat --resume
  magent rag ingest docs --save .magent/index.json
  magent rag --index .magent/index.json ask "how do approvals work?" --check
  magent tools list --max-risk read
  magent cortex models
  magent eval run --dataset builtin --fail-below 0.6
  magent serve --port 8700
  magent plugins list --loaded
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
    eval_cmd,
    serve_cmd,
    plugins_cmd,
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


class _FlagView:
    """A read-only view of one parser's flags, usable as an argparse parent.

    ``_add_container_actions`` only reads ``_actions``, ``_action_groups``, and
    ``_mutually_exclusive_groups``, so a shim with those three is enough to copy
    selected flags without copying the parser's subcommands. The group is titled
    ``options`` on purpose: a leaf parser already has one, so argparse merges
    into it instead of adding a second section to the help output.
    """

    def __init__(self, actions: Sequence[argparse.Action]) -> None:
        """Wrap the actions that should be inherited."""
        self._actions = list(actions)
        group = argparse._ArgumentGroup.__new__(argparse._ArgumentGroup)
        group.title = "options"
        group.description = None
        group.conflict_handler = "error"
        group._group_actions = self._actions
        self._action_groups = [group]
        self._mutually_exclusive_groups: list[Any] = []
        self._defaults: dict[str, Any] = {}


def _inheritable_flags(owner: argparse.ArgumentParser) -> list[argparse.Action]:
    """Return the flags of a group parser that its actions should also accept.

    Help, version, the subcommand action itself, and the global flags are left
    out: the globals are already injected from :func:`_global_parent`, and
    copying them twice would be a conflicting option string.
    """
    skip = set(id(action) for action in _global_parent()._actions)
    found: list[argparse.Action] = []
    for action in getattr(owner, "_actions", []):
        if not action.option_strings or id(action) in skip:
            continue  # positionals are consumed by the group itself
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction,
                               argparse._SubParsersAction)):
            continue
        found.append(action)
    return found


def _inherit_group_flags(owner: argparse.ArgumentParser | None,
                         leaf: argparse.ArgumentParser) -> None:
    """Teach a leaf subparser its group's flags, once.

    This is done at parse time rather than at ``add_parser`` time so a group
    that adds flags after creating its subcommands still propagates them.
    """
    if owner is None:
        return
    known = set(leaf._option_string_actions)
    wanted = [action for action in _inheritable_flags(owner)
              if not any(option in known for option in action.option_strings)]
    if wanted:
        leaf._add_container_actions(_FlagView(wanted))


class _InjectingSubParsers(argparse._SubParsersAction):
    """Subparser action that makes flag position irrelevant.

    Two argparse behaviours make a CLI with nested commands awkward, and both
    are corrected here:

    * A flag is only recognized by the parser that declares it, so
      ``magent tools list --offline`` and ``magent plugins list --dir X`` were
      rejected even though the flag exists one level up. Every subparser
      therefore inherits the global flags and its own group's flags.
    * ``_SubParsersAction.__call__`` parses the subcommand into a fresh
      namespace and copies *every* key back over the parent's, so a global
      typed before the command was overwritten by the subparser's own default -
      ``magent -f json tools categories`` printed a table. The parent's values
      are seeded into the subparser's defaults for the duration of that parse,
      then restored, so an unspecified flag inherits and a specified one wins.
    """

    owner: argparse.ArgumentParser | None = None
    """The parser whose ``add_subparsers`` created this action."""

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

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> Any:
        """Parse one subcommand, preserving values already on the namespace.

        The inherited value has to be written to ``action.default``, not to the
        parser's ``_defaults``: ``_parse_known_args`` initializes a fresh
        namespace from each action's own ``default`` and only consults
        ``_defaults`` for keys nothing declared. Defaults are restored
        afterwards because ``parents=`` shares one action object between every
        parser that inherited it.
        """
        name = values[0] if values else ""
        leaf = self._name_parser_map.get(str(name))
        if leaf is None:
            return super().__call__(parser, namespace, values, option_string)
        _inherit_group_flags(self.owner, leaf)
        # The subcommand's own dest must not be inherited: it is still unset when
        # this runs, and copying that placeholder down would overwrite the real
        # name on the way back out.
        reserved = {str(action.dest) for source in (self.owner, leaf) if source is not None
                    for action in getattr(source, "_actions", [])
                    if isinstance(action, argparse._SubParsersAction)}
        reserved.add(str(self.dest))
        by_dest: dict[str, list[argparse.Action]] = {}
        for action in leaf._actions:
            if action.dest is not argparse.SUPPRESS:
                by_dest.setdefault(str(action.dest), []).append(action)
        saved: list[tuple[argparse.Action, Any]] = []
        for key, value in vars(namespace).items():
            if key.startswith("_") or key in reserved or key not in by_dest:
                continue
            for action in by_dest[key]:
                if action.default is not value:
                    saved.append((action, action.default))
                    action.default = value
        try:
            return super().__call__(parser, namespace, values, option_string)
        finally:
            for action, default in saved:
                action.default = default


class _HelpfulParser(argparse.ArgumentParser):
    """Parser that prints full help on an error instead of one terse line."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the parser and make nested subcommands inherit globals too."""
        super().__init__(*args, **kwargs)
        self.register("action", "parsers", _InjectingSubParsers)

    def add_subparsers(self, **kwargs: Any) -> argparse.Action:
        """Create the subcommand action, remembering this parser as its owner.

        The back-reference is what lets a leaf action inherit the flags declared
        one level up, without each command module passing them twice.
        """
        action = super().add_subparsers(**kwargs)
        if isinstance(action, _InjectingSubParsers):
            action.owner = self
        return action

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
        args = parser.parse_args(arguments)
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
