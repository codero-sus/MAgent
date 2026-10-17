"""Shared CLI plumbing: global flags, config resolution, and a context object.

Every subcommand needs the same handful of things - a resolved configuration, a
backend, a tool registry, a printer that respects ``--quiet`` and ``--format`` -
and computing them differently in each command is how a CLI ends up with flags
that work in one place and not another.

:func:`build_context` does it once. Command handlers receive a
:class:`CliContext` and ask it for what they need; expensive pieces (the
backend, the registry) are built lazily, so ``magent config show`` never
connects to Cortex.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Config, load_config
from ..errors import MAgentError
from ..logging_ import get_logger, setup_logging
from ..output import OutputOptions, Printer

LOGGER = get_logger("cli")
"""Logger for CLI-level faults."""

GLOBAL_FLAGS: tuple[str, ...] = (
    "--config", "--profile", "--set", "--backend", "--model", "--base-url", "--api-key",
    "--format", "--color", "--no-color", "--quiet", "--verbose", "--debug", "--width",
    "--workspace", "--offline", "--no-tools", "--timeout", "--yes", "--dry-run",
)
"""Flags accepted before or after a subcommand name."""

EXIT_OK = 0
"""Process exit code for success."""

EXIT_ERROR = 1
"""Generic failure."""

EXIT_USAGE = 2
"""Bad arguments or an unknown command."""

EXIT_INTERRUPT = 130
"""The operator pressed Ctrl-C."""

EXIT_CONFIG = 78
"""Configuration could not be resolved (``EX_CONFIG``)."""


def add_global_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the flags every subcommand accepts.

    Args:
        parser: the parser (root or subcommand) to extend
    """
    group = parser.add_argument_group("global options")
    group.add_argument("-c", "--config", metavar="PATH", default="",
                       help="configuration file to load (TOML)")
    group.add_argument("-p", "--profile", metavar="NAME", default="",
                       help="configuration profile to activate")
    group.add_argument("-s", "--set", metavar="KEY=VALUE", action="append", default=[],
                       dest="overrides", help="override one config value, e.g. -s agent.max_iterations=5")
    group.add_argument("--backend", metavar="NAME", default="",
                       help="LLM backend: auto, cortex, echo, router")
    group.add_argument("--model", metavar="ID", default="", help="model id to use")
    group.add_argument("--base-url", metavar="URL", default="",
                       help="Cortex base URL, e.g. http://127.0.0.1:8624")
    group.add_argument("--api-key", metavar="KEY", default="",
                       help="API key; prefer MAGENT_API_KEY over the command line")
    group.add_argument("-f", "--format", metavar="FMT", default="",
                       choices=["", "text", "json", "markdown"], help="output format")
    group.add_argument("--color", dest="color", action="store_true", default=None,
                       help="force colour output")
    group.add_argument("--no-color", dest="color", action="store_false",
                       help="disable colour output")
    group.add_argument("-q", "--quiet", action="store_true", default=False,
                       help="print only the answer")
    group.add_argument("-v", "--verbose", action="store_true", default=False,
                       help="show steps, tools and timings")
    group.add_argument("--debug", action="store_true", default=False,
                       help="enable debug logging and full event output")
    group.add_argument("--width", type=int, default=0, metavar="COLS",
                       help="wrap output at this column width")
    group.add_argument("-w", "--workspace", metavar="DIR", default="",
                       help="working directory the agent may touch")
    group.add_argument("--offline", action="store_true", default=False,
                       help="never contact a server; use the echo backend")
    group.add_argument("--no-tools", action="store_true", default=False,
                       help="disable every tool")
    group.add_argument("--timeout", type=float, default=0.0, metavar="SECONDS",
                       help="per-request timeout override")
    group.add_argument("-y", "--yes", action="store_true", default=False,
                       help="approve destructive actions without asking")
    group.add_argument("--dry-run", action="store_true", default=False,
                       help="show what would happen without doing it")


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Turn ``key=value`` strings into a nested override mapping.

    Dotted keys become nested tables, and values are parsed as TOML so that
    ``-s agent.max_iterations=5`` yields an int rather than the string ``"5"``.

    Args:
        pairs: raw ``key=value`` strings from ``--set``

    Returns:
        A nested dictionary suitable for :func:`magent.config.load_config`.

    Raises:
        MAgentError: a pair has no ``=``, or its value is not valid TOML
    """
    import tomllib

    overrides: dict[str, Any] = {}
    for pair in pairs:
        text = str(pair).strip()
        if not text:
            continue
        if "=" not in text:
            raise MAgentError(f"--set expects key=value, got {pair!r}")
        key, _, raw = text.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise MAgentError(f"--set needs a key before '=', got {pair!r}")
        try:
            value: Any = tomllib.loads(f"value = {raw}")["value"]
        except tomllib.TOMLDecodeError:
            value = raw
        target = overrides
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise MAgentError(f"--set key {key!r} is empty")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise MAgentError(f"--set key {key!r} conflicts with an earlier value")
        target[parts[-1]] = value
    return overrides


@dataclass(slots=True)
class CliContext:
    """Everything a command handler needs, built lazily."""

    args: argparse.Namespace
    config: Config
    options: OutputOptions
    printer: Printer
    command: str = ""
    _backend: Any = field(default=None, repr=False)
    _registry: Any = field(default=None, repr=False)
    _decision: Any = field(default=None, repr=False)
    exit_code: int = EXIT_OK

    # -- lazily built resources ---------------------------------------------

    @property
    def backend(self) -> Any:
        """The LLM backend implied by config and flags, built on first use."""
        if self._backend is None:
            self._backend, self._decision = self._build_backend()
        return self._backend

    @property
    def decision(self) -> Any:
        """The :class:`~magent.llm.factory.BackendDecision` explaining the backend."""
        _ = self.backend
        return self._decision

    @property
    def registry(self) -> Any:
        """The tool registry implied by config and flags, built on first use."""
        if self._registry is None:
            self._registry = self._build_registry()
        return self._registry

    def _build_backend(self) -> tuple[Any, Any]:
        """Resolve the backend from ``--offline``, flags, and configuration."""
        from ..llm.factory import BackendFactory, BackendRequest

        if getattr(self.args, "offline", False):
            from ..llm.echo import EchoBackend

            model = getattr(self.args, "model", "") or self.config.model.model or "echo-small"
            decision = BackendRequest(kind="echo", model=model)
            return EchoBackend(models_list=[model]), decision
        request = BackendRequest.from_config(self.config)
        factory = BackendFactory()
        overrides = {
            "kind": getattr(self.args, "backend", "") or None,
            "model": getattr(self.args, "model", "") or None,
            "base_url": getattr(self.args, "base_url", "") or None,
            "api_key": getattr(self.args, "api_key", "") or None,
            "timeout": getattr(self.args, "timeout", 0.0) or None,
        }
        return factory.build(request, **overrides), factory.last_decision

    def _build_registry(self) -> Any:
        """Build the tool registry, honouring config and ``--no-tools``."""
        from ..tools import ToolRegistry
        from ..tools.builtin import load_all, load_errors

        registry = ToolRegistry()
        if getattr(self.args, "no_tools", False):
            return registry
        include = list(self.config.tools.enabled or []) or None
        exclude = list(self.config.tools.disabled or []) or None
        load_all(registry, include=include, exclude=exclude)
        errors = load_errors()
        if errors:
            LOGGER.debug("tool modules failed to load: %s", errors)
        return registry

    # -- conveniences --------------------------------------------------------

    @property
    def workspace(self) -> Path:
        """Directory the agent is allowed to work in."""
        raw = getattr(self.args, "workspace", "") or self.config.tools.workspace or os.getcwd()
        return Path(raw).expanduser().resolve()

    @property
    def assume_yes(self) -> bool:
        """Whether destructive actions are pre-approved."""
        return bool(getattr(self.args, "yes", False))

    @property
    def dry_run(self) -> bool:
        """Whether the command should only report what it would do."""
        return bool(getattr(self.args, "dry_run", False))

    def emit(self, payload: dict[str, Any], *, text: str = "") -> None:
        """Print a payload in the configured format.

        Args:
            payload: data emitted verbatim when the format is JSON
            text: pre-rendered text used for the text and markdown formats
        """
        if self.options.fmt == "json":
            import json

            self.printer.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                               min_level="quiet")
        else:
            self.printer.write(text if text else _render_pairs(payload), min_level="quiet")

    def make_agent(self, **kwargs: Any) -> Any:
        """Build an :class:`~magent.agent.Agent` wired to this context.

        The sandbox and approval manager are left to the agent, which derives
        both from ``settings`` - building them here too would give the CLI a
        second, drifting copy of that policy. ``--yes`` is expressed the same
        way a config file would: by switching the approval mode off.
        """
        from ..agent import Agent

        settings = kwargs.pop("settings", None) or self.config
        if self.assume_yes and settings.tools.approval_mode != "never":
            settings = settings.clone(tools=replace(settings.tools, approval_mode="never"))
        registry = kwargs.pop("registry", None) or self.registry
        return Agent(backend=self.backend, config=settings.agent, settings=settings,
                     registry=registry, workspace=self.workspace, **kwargs)


def _render_pairs(payload: dict[str, Any]) -> str:
    """Render a flat mapping as aligned ``key : value`` lines."""
    import json

    from ..utils.tables import render_key_value

    pairs: list[tuple[str, Any]] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, default=str)
            pairs.append((str(key), rendered if len(rendered) <= 90 else rendered[:87] + "..."))
        else:
            pairs.append((str(key), value))
    return render_key_value(pairs)


def build_context(args: argparse.Namespace, *, command: str = "") -> CliContext:
    """Resolve configuration and output settings for one invocation.

    Args:
        args: parsed command line
        command: name of the subcommand being run

    Returns:
        A ready :class:`CliContext`.

    Raises:
        MAgentError: the configuration file or an override is invalid
    """
    overrides = parse_overrides(getattr(args, "overrides", []) or [])
    flag_overrides: dict[str, Any] = {}
    if getattr(args, "backend", ""):
        flag_overrides.setdefault("model", {})["backend"] = args.backend
    if getattr(args, "model", ""):
        flag_overrides.setdefault("model", {})["model"] = args.model
    if getattr(args, "base_url", ""):
        flag_overrides.setdefault("server", {})["base_url"] = args.base_url
    if getattr(args, "api_key", ""):
        flag_overrides.setdefault("server", {})["api_key"] = args.api_key
    if getattr(args, "timeout", 0.0):
        flag_overrides.setdefault("server", {})["timeout"] = args.timeout
    if getattr(args, "workspace", ""):
        flag_overrides.setdefault("tools", {})["workspace"] = args.workspace
    if getattr(args, "offline", False):
        flag_overrides.setdefault("model", {})["backend"] = "echo"
    for section, values in flag_overrides.items():
        merged = dict(overrides.get(section) or {})
        merged.update(values)
        overrides[section] = merged

    config = load_config(path=getattr(args, "config", "") or "",
                         profile=getattr(args, "profile", "") or "",
                         overrides=overrides)

    verbosity = "normal"
    if getattr(args, "debug", False):
        verbosity = "debug"
    elif getattr(args, "verbose", False):
        verbosity = "verbose"
    elif getattr(args, "quiet", False):
        verbosity = "quiet"
    options = OutputOptions.from_config(
        config,
        fmt=getattr(args, "format", "") or None,
        color=getattr(args, "color", None),
        width=getattr(args, "width", 0) or None,
        quiet=True if getattr(args, "quiet", False) else None,
        verbosity=verbosity if verbosity != "normal" else None,
    )
    setup_logging(
        level="DEBUG" if getattr(args, "debug", False)
        else ("INFO" if getattr(args, "verbose", False) else "WARNING"),
        fmt=getattr(config.observability, "log_format", "text"),
        color=options.color,
        stream=sys.stderr,
    )
    printer = Printer(options)
    return CliContext(args=args, config=config, options=options, printer=printer, command=command)


def fail(context: CliContext | None, message: str, *, code: int = EXIT_ERROR) -> int:
    """Report an error consistently and return the exit code."""
    if context is None:
        print(f"error: {message}", file=sys.stderr)
        return code
    if context.options.fmt == "json":
        import json

        context.printer.write(json.dumps({"error": message, "exit_code": code}, indent=2),
                              min_level="quiet")
    else:
        context.printer.error(f"error: {message}")
    return code


def handle_error(exc: BaseException, context: CliContext | None = None) -> int:
    """Map an exception onto a useful message and exit code.

    Known MAgent errors are reported by their own message; anything else keeps
    its traceback at ``--debug`` and a one-line summary otherwise, so a crash is
    diagnosable without drowning a normal user.
    """
    if isinstance(exc, KeyboardInterrupt):
        if context is not None:
            context.printer.write("")
            context.printer.warn("interrupted")
        return EXIT_INTERRUPT
    if isinstance(exc, MAgentError):
        message = str(exc) or type(exc).__name__
        details = getattr(exc, "details", None)
        if details and context is not None and context.options.at_least("verbose"):
            message = f"{message} ({details})"
        return fail(context, message, code=getattr(exc, "exit_code", EXIT_ERROR) or EXIT_ERROR)
    if context is not None and context.options.at_least("debug"):
        import traceback

        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    return fail(context, f"{type(exc).__name__}: {exc}")


def confirm(prompt: str, *, context: CliContext | None = None, default: bool = False,
            assume_yes: bool = False) -> bool:
    """Ask the operator a yes/no question.

    A non-interactive stdin answers with the default rather than blocking
    forever, which is what makes the CLI safe to run from a script.

    Args:
        prompt: the question to show
        context: used for colour and quiet settings
        default: answer when input is unavailable
        assume_yes: skip the question and answer yes
    """
    if assume_yes:
        return True
    if context is not None and (context.quiet if hasattr(context, "quiet") else False):
        return default
    if not sys.stdin or not sys.stdin.isatty():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input((context.printer.themed(prompt, "warning") if context else prompt) + suffix)
    except (EOFError, KeyboardInterrupt):
        print("", file=sys.stderr)
        return default
    text = answer.strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "true", "1", "ok"}


def read_prompt_text(args: argparse.Namespace) -> str:
    """Collect the task text from arguments, a file, or stdin."""
    pieces = list(getattr(args, "prompt", []) or [])
    path = getattr(args, "file", "") or ""
    if path:
        target = Path(path).expanduser()
        if not target.exists():
            raise MAgentError(f"prompt file not found: {target}")
        pieces.append(target.read_text(encoding="utf-8"))
    if getattr(args, "stdin", False) or (not pieces and not sys.stdin.isatty()):
        data = sys.stdin.read()
        if data.strip():
            pieces.append(data)
    return "\n\n".join(piece.strip() for piece in pieces if piece.strip()).strip()


__all__ = [
    "EXIT_CONFIG",
    "EXIT_ERROR",
    "EXIT_INTERRUPT",
    "EXIT_OK",
    "EXIT_USAGE",
    "GLOBAL_FLAGS",
    "CliContext",
    "add_global_flags",
    "build_context",
    "confirm",
    "fail",
    "handle_error",
    "parse_overrides",
    "read_prompt_text",
]
