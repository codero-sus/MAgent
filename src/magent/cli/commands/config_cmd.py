"""The ``config`` command: see and change how MAgent is configured.

Configuration arrives from six places at once - defaults, system, user,
project, ``$MAGENT_CONFIG``, environment, then the command line - so the
questions people actually ask are "what is the effective value?" and "which
file did it come from?". ``show`` answers the first and reports the sources
that contributed; ``get`` and ``set`` work on dotted paths; ``init`` writes a
documented starting file.

Secrets are redacted in every rendering unless ``--reveal`` is passed, and that
flag is refused when stdout is not a terminal, so a config dump cannot leak a
key into a log or a pipeline by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...config import (
    SECTION_TYPES,
    Config,
    default_config_paths,
    load_config,
    user_cache_dir,
    user_config_dir,
    user_data_dir,
    write_example_config,
)
from ...errors import MAgentError
from ...secrets import REDACTED
from ...utils.files import atomic_write_text, read_text_safe
from ...utils.tables import render_key_value, render_markdown_table, render_table
from ..args import CliContext, EXIT_ERROR, EXIT_OK


def register(subparsers: Any) -> None:
    """Add the ``config`` subcommand and its actions."""
    parser = subparsers.add_parser("config", help="inspect and edit configuration",
                                   description="Show, query and write MAgent configuration.")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    shown = action.add_parser("show", help="print the resolved configuration")
    shown.add_argument("--section", default="", choices=["", *SECTION_TYPES],
                       help="print only one section")
    shown.add_argument("--reveal", action="store_true",
                       help="do not redact secrets (refused when not a terminal)")
    shown.add_argument("--sources", action="store_true", help="also list contributing files")

    got = action.add_parser("get", help="print one value by dotted path")
    got.add_argument("key", help="dotted path, e.g. agent.max_iterations")
    got.add_argument("--reveal", action="store_true", help="do not redact secrets")

    put = action.add_parser("set", help="write one value to a config file")
    put.add_argument("key", help="dotted path, e.g. agent.max_iterations")
    put.add_argument("value", help="new value, parsed as TOML")
    put.add_argument("--file", default="", metavar="PATH",
                     help="file to write, default the user config")

    action.add_parser("paths", help="show the config, data and cache directories")
    action.add_parser("validate", help="check the resolved configuration")

    init = action.add_parser("init", help="write a starter configuration file")
    init.add_argument("--file", default="", metavar="PATH", help="destination file")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")

    dumped = action.add_parser("dump", help="print the configuration as TOML")
    dumped.add_argument("--reveal", action="store_true", help="do not redact secrets")

    action.add_parser("env", help="show the MAGENT_* environment variables in effect")
    parser.set_defaults(handler=cmd_config, action_default="show")


def cmd_config(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``config`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "show")
    handlers = {
        "show": _show,
        "get": _get,
        "set": _set,
        "paths": _paths,
        "validate": _validate,
        "init": _init,
        "dump": _dump,
        "env": _env,
    }
    return handlers.get(action, _show)(args, context)


def _may_reveal(args: argparse.Namespace, context: CliContext) -> bool:
    """Decide whether unredacted output is allowed.

    Refusing on a non-terminal is deliberate: ``magent config show > log.txt``
    or a pipe into another program is exactly how an API key ends up somewhere
    it should not be.
    """
    if not getattr(args, "reveal", False):
        return False
    if not sys.stdout.isatty():
        context.printer.warn("--reveal refused: stdout is not a terminal")
        return False
    return True


def _show(args: argparse.Namespace, context: CliContext) -> int:
    """Print the resolved configuration."""
    reveal = _may_reveal(args, context)
    config = context.config
    section = getattr(args, "section", "") or ""
    if section:
        payload: Any = config.section(section)
        data = payload.to_dict(redact=not reveal) if hasattr(payload, "to_dict") else dict(payload)
    else:
        data = config.to_dict(redact=not reveal)
    if context.options.fmt == "json":
        out: dict[str, Any] = {"config": data}
        if getattr(args, "sources", False):
            out["sources"] = list(getattr(config, "sources", []) or [])
            out["config_path"] = str(getattr(config, "config_path", "") or "")
        context.emit(out)
        return EXIT_OK
    printer = context.printer
    if section:
        printer.write(printer.themed(f"[{section}]", "heading"), min_level="quiet")
        printer.write(render_key_value(_flat(data)), min_level="quiet")
    else:
        for name in SECTION_TYPES:
            values = data.get(name)
            if not isinstance(values, dict):
                continue
            printer.write(printer.themed(f"[{name}]", "heading"), min_level="quiet")
            printer.write(render_key_value(_flat(values)), min_level="quiet")
            printer.blank(min_level="quiet")
    if getattr(args, "sources", False):
        printer.write(printer.themed("sources", "heading"), min_level="quiet")
        contributed = list(getattr(config, "sources", []) or [])
        printer.write("\n".join(f"  {item}" for item in contributed) or "  (defaults only)",
                      min_level="quiet")
    if not reveal:
        printer.write(printer.themed(f"secrets shown as {REDACTED}; use --reveal in a terminal",
                                     "muted"), min_level="normal")
    return EXIT_OK


def _flat(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Flatten one level of a config section into key/value pairs."""
    pairs: list[tuple[str, Any]] = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, default=str)
            pairs.append((str(key), rendered if len(rendered) <= 70 else rendered[:67] + "..."))
        else:
            pairs.append((str(key), value))
    return pairs


def _get(args: argparse.Namespace, context: CliContext) -> int:
    """Print one configuration value."""
    key = str(getattr(args, "key", "") or "")
    reveal = _may_reveal(args, context)
    value = context.config.get(key, default=_MISSING)
    if value is _MISSING:
        context.printer.error(f"error: unknown configuration key {key!r}")
        _suggest_key(context, key)
        return EXIT_ERROR
    if context.options.fmt == "json":
        context.emit({"key": key, "value": value})
        return EXIT_OK
    if isinstance(value, (dict, list)):
        context.printer.write(json.dumps(value, indent=2, ensure_ascii=False, default=str),
                              min_level="quiet")
        return EXIT_OK
    shown = value if reveal else _redact_if_secret(key, value)
    context.printer.write(str(shown), min_level="quiet")
    return EXIT_OK


class _Missing:
    """Sentinel distinguishing 'absent' from a legitimate ``None`` value."""

    def __repr__(self) -> str:
        """Debug representation."""
        return "<missing>"


_MISSING = _Missing()


def _redact_if_secret(key: str, value: Any) -> Any:
    """Redact a value whose key names a credential."""
    from ...secrets import contains_secret, is_secret_reference

    text = str(value)
    lowered = key.lower()
    if any(hint in lowered for hint in ("key", "token", "secret", "password")):
        if text and not is_secret_reference(text):
            return REDACTED
    if text and contains_secret(text, []):
        return REDACTED
    return value


def _suggest_key(context: CliContext, key: str) -> None:
    """Print close matches for a mistyped dotted key."""
    wanted = key.split(".")
    suggestions: list[str] = []
    for section_name, section_type in SECTION_TYPES.items():
        for field_name in getattr(section_type, "__dataclass_fields__", {}):
            candidate = f"{section_name}.{field_name}"
            if wanted[-1] and wanted[-1] in field_name:
                suggestions.append(candidate)
    if suggestions:
        context.printer.write(context.printer.themed(
            f"did you mean: {', '.join(sorted(suggestions)[:6])}", "muted"), min_level="quiet")


def _set(args: argparse.Namespace, context: CliContext) -> int:
    """Write one value into a TOML configuration file."""
    key = str(getattr(args, "key", "") or "")
    raw = str(getattr(args, "value", ""))
    parts = [part for part in key.split(".") if part]
    if len(parts) != 2:
        context.printer.error("error: key must look like section.field")
        return EXIT_ERROR
    section_name, field_name = parts
    if section_name not in SECTION_TYPES:
        context.printer.error(f"error: unknown section {section_name!r}; "
                              f"known: {', '.join(SECTION_TYPES)}")
        return EXIT_ERROR

    import tomllib

    try:
        value: Any = tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw

    probe = context.config.clone()
    try:
        probe.set(key, value)
        probe.validate()
    except MAgentError as exc:
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR

    target = Path(getattr(args, "file", "") or "").expanduser() or (user_config_dir() / "config.toml")
    if context.dry_run:
        context.printer.write(f"would set [{section_name}] {field_name} = {_toml_value(value)} "
                              f"in {target}", min_level="quiet")
        return EXIT_OK

    existing = _read_toml(target)
    section = dict(existing.get(section_name) or {})
    section[field_name] = value
    existing[section_name] = section
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, _dump_toml(existing))
    context.printer.write(f"set {key} = {_toml_value(value)} in {target}", min_level="quiet")
    return EXIT_OK


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning an empty mapping when it is absent."""
    if not path.exists():
        return {}
    import tomllib

    try:
        with path.open("rb") as handle:
            return dict(tomllib.load(handle))
    except Exception as exc:  # noqa: BLE001 - a broken file must not lose the write
        raise MAgentError(f"cannot parse {path}: {exc}") from exc


def _toml_value(value: Any) -> str:
    """Render one value in TOML syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _dump_toml(data: dict[str, Any]) -> str:
    """Render a mapping as a TOML document with one table per section."""
    lines: list[str] = []
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if scalars:
        lines.append("")
    for name, section in data.items():
        if not isinstance(section, dict):
            continue
        lines.append(f"[{name}]")
        for key, value in section.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _paths(args: argparse.Namespace, context: CliContext) -> int:
    """Show every directory MAgent uses."""
    payload = {
        "config_file": str(getattr(context.config, "config_path", "") or "(none)"),
        "searched": [str(path) for path in default_config_paths()],
        "user_config_dir": str(user_config_dir()),
        "data_dir": str(getattr(context.config, "data_dir", "") or user_data_dir()),
        "cache_dir": str(getattr(context.config, "cache_dir", "") or user_cache_dir()),
        "session_dir": str(context.config.session.directory or "(data_dir/sessions)"),
        "memory_store": str(context.config.memory.store_path or "(data_dir/memory.jsonl)"),
        "workspace": str(context.workspace),
    }
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    rows: list[list[Any]] = []
    for key, value in payload.items():
        if isinstance(value, (list, tuple)):
            rows.append([key, ", ".join(str(item) for item in value) or "-", ""])
        else:
            rows.append([key, str(value),
                         "exists" if value and Path(str(value)).exists() else "-"])
    table = render_markdown_table(rows, headers=["path", "value", "state"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["path", "value", "state"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    return EXIT_OK


def _validate(args: argparse.Namespace, context: CliContext) -> int:
    """Validate the resolved configuration and report every problem."""
    problems: list[str] = []
    config = context.config
    for name in SECTION_TYPES:
        section = config.section(name)
        try:
            section.validate()
        except MAgentError as exc:
            problems.append(f"[{name}] {exc}")
    try:
        config.validate()
    except MAgentError as exc:
        problems.append(str(exc))

    from ...tools.builtin import load_errors

    problems.extend(f"tool module {name}: {message.splitlines()[0]}"
                    for name, message in load_errors().items())

    if context.options.fmt == "json":
        context.emit({"valid": not problems, "problems": problems,
                      "sections": list(SECTION_TYPES)})
        return EXIT_OK if not problems else EXIT_ERROR
    if problems:
        for problem in problems:
            context.printer.error(problem)
        return EXIT_ERROR
    context.printer.write(context.printer.themed(
        f"configuration is valid ({len(SECTION_TYPES)} sections)", "success"), min_level="quiet")
    return EXIT_OK


def _init(args: argparse.Namespace, context: CliContext) -> int:
    """Write a documented starter configuration file."""
    target = Path(getattr(args, "file", "") or "").expanduser() or (user_config_dir() / "config.toml")
    if target.exists() and not getattr(args, "force", False):
        context.printer.error(f"error: {target} exists; pass --force to overwrite")
        return EXIT_ERROR
    if context.dry_run:
        context.printer.write(f"would write a starter config to {target}", min_level="quiet")
        return EXIT_OK
    try:
        written = write_example_config(target, force=bool(getattr(args, "force", False)))
    except MAgentError as exc:
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    context.printer.write(f"wrote {written} ({written.stat().st_size} bytes)", min_level="quiet")
    return EXIT_OK


def _dump(args: argparse.Namespace, context: CliContext) -> int:
    """Print the effective configuration as TOML."""
    reveal = _may_reveal(args, context)
    if reveal:
        text = context.config.to_toml()
    else:
        text = _redact_toml(context.config.to_toml())
    context.printer.write(text, min_level="quiet")
    return EXIT_OK


def _redact_toml(text: str) -> str:
    """Redact credential-looking assignments in rendered TOML."""
    from ...secrets import is_secret_reference

    out: list[str] = []
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            lowered = key.strip().lower()
            if any(hint in lowered for hint in ("key", "token", "secret", "password")):
                bare = value.strip().strip('"')
                if bare and not is_secret_reference(bare):
                    out.append(f'{key.strip()} = "{REDACTED}"')
                    continue
        out.append(line)
    return "\n".join(out)


def _env(args: argparse.Namespace, context: CliContext) -> int:
    """Show the MAGENT_* and CORTEX_* variables currently in effect."""
    import os

    from ...config import ENV_MAP

    active = {name: value for name, value in os.environ.items()
              if name.startswith(("MAGENT_", "CORTEX_"))}
    rows: list[list[Any]] = []
    for name in sorted(active):
        value = active[name]
        mapped = ENV_MAP.get(name)
        target = f"{mapped[0]}.{mapped[1]}" if mapped else "(unused)"
        lowered = name.lower()
        if any(hint in lowered for hint in ("key", "token", "secret", "password")) and value:
            value = REDACTED
        rows.append([name, str(value)[:48], target])
    known = [name for name in ENV_MAP if name not in active]
    if context.options.fmt == "json":
        context.emit({"set": {row[0]: row[1] for row in rows},
                      "unset": sorted(known), "mapped": len(ENV_MAP)})
        return EXIT_OK
    if rows:
        context.printer.write(render_table(rows, headers=["variable", "value", "maps to"],
                                           unicode_borders=context.options.unicode),
                              min_level="quiet")
    else:
        context.printer.write("no MAGENT_* or CORTEX_* variables are set", min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(rows)} set, {len(known)} recognized but unset", "muted"), min_level="normal")
    return EXIT_OK


__all__ = ["REDACTED", "cmd_config", "register"]
