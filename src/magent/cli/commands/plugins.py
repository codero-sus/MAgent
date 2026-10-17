"""The ``plugins`` command: discover, inspect, and manage extensions.

A plugin is a directory with a manifest and an entrypoint. The manifest declares
what the plugin wants - tools it registers, hooks it attaches, and the
permissions those need - so MAgent can refuse a plugin before importing it
rather than discovering what it does afterwards.

That is why this command spends most of its surface on *looking* rather than
*loading*: ``list`` and ``show`` read manifests, ``inspect`` examines a directory
without executing it, and ``permissions`` documents the ten permissions and
which four are sensitive enough to need an explicit opt-in. ``load`` is the only
action that imports code, and it reports exactly what each plugin contributed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...errors import MAgentError
from ...plugins import (
    HOOK_NAMES,
    MANIFEST_FILENAME,
    PERMISSIONS,
    SENSITIVE_PERMISSIONS,
    TRANSFORM_HOOKS,
    VETO_HOOKS,
    HookRegistry,
    PluginLoader,
    describe_permissions,
    inspect_plugin,
    loader_from_config,
    scaffold,
    write_manifest_template,
)
from ...utils.files import human_size
from ...utils.tables import render_key_value, render_table
from ..args import EXIT_ERROR, EXIT_OK, CliContext

COMMAND_NAMES: tuple[str, ...] = ("plugins",)
"""Subcommands this module registers."""

ACTIONS: tuple[str, ...] = (
    "list", "show", "inspect", "scaffold", "permissions", "hooks", "load",
    "manifest", "stats",
)
"""Actions ``plugins`` dispatches."""


def register(subparsers: Any) -> None:
    """Add the ``plugins`` subcommand and its actions."""
    parser = subparsers.add_parser(
        "plugins",
        help="discover, inspect and load extensions",
        description="Manage plugins: manifest-declared extensions that add tools "
                    "and hooks under an explicit permission list.",
        epilog="examples:\n"
               "  magent plugins list\n"
               "  magent plugins scaffold ./plugins --name weather\n"
               "  magent plugins inspect ./plugins/weather\n"
               "  magent plugins load --allow-sensitive network\n"
               "  magent plugins permissions\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir", action="append", default=[], metavar="PATH",
                        help="extra plugin directory; repeatable")
    parser.add_argument("--allow-sensitive", action="append", default=[], metavar="PERMISSION",
                        help=f"permit a sensitive permission: "
                             f"{', '.join(sorted(SENSITIVE_PERMISSIONS))}")
    parser.add_argument("--deny", action="append", default=[], metavar="NAME",
                        help="refuse this plugin by name; repeatable")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="load only these plugins; repeatable")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip manifest checksum verification")
    parser.add_argument("--strict", action="store_true",
                        help="fail the command when any plugin cannot load")

    action = parser.add_subparsers(dest="action", metavar="ACTION")

    listing = action.add_parser("list", help="list discovered plugins")
    listing.add_argument("--loaded", action="store_true",
                         help="also import them and report what they contributed")

    shown = action.add_parser("show", help="print one plugin's manifest")
    shown.add_argument("name", help="plugin name")

    inspected = action.add_parser("inspect", help="examine a directory without importing it")
    inspected.add_argument("path", help="plugin directory")

    created = action.add_parser("scaffold", help="write a working starter plugin")
    created.add_argument("path", help="parent directory to create the plugin in")
    created.add_argument("--name", default="", help="plugin name (default: directory name)")
    created.add_argument("--force", action="store_true", help="overwrite an existing directory")

    action.add_parser("permissions", help="document the permission list")
    action.add_parser("hooks", help="document the hook points")
    action.add_parser("load", help="load every plugin and report what it added")
    action.add_parser("stats", help="show loader statistics")

    templated = action.add_parser("manifest", help="write a manifest template")
    templated.add_argument("path", nargs="?", default="", help="destination file")
    templated.add_argument("--name", default="example_plugin", help="plugin name to use")

    parser.set_defaults(handler=cmd_plugins, action_default="list")


def cmd_plugins(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``plugins`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "list")
    handlers: dict[str, Any] = {
        "list": _list, "show": _show, "inspect": _inspect, "scaffold": _scaffold,
        "permissions": _permissions, "hooks": _hooks, "load": _load,
        "manifest": _manifest, "stats": _stats,
    }
    handler = handlers.get(action)
    if handler is None:
        context.printer.error(f"unknown plugins action: {action}")
        context.printer.write(f"choose one of: {', '.join(ACTIONS)}")
        return EXIT_ERROR
    return handler(args, context)


def _loader(args: argparse.Namespace, context: CliContext, *,
            hooks: HookRegistry | None = None) -> PluginLoader:
    """Build a loader from configuration plus this command's flags."""
    base = loader_from_config(context.config, registry=context.registry, hooks=hooks)
    allow_sensitive = set(base.allow_sensitive) | {
        str(item).strip().lower() for item in (args.allow_sensitive or []) if str(item).strip()}
    deny = set(base.deny) | {str(item).strip() for item in (args.deny or []) if str(item).strip()}
    only = list(args.only) if args.only else None
    if only and base.allow:
        only = sorted(set(only) & set(base.allow))
    return base.with_policy(
        allow_sensitive=sorted(allow_sensitive),
        deny=sorted(deny),
        allow=only if only is not None else base.allow,
        directories=list(args.dir or []),
        verify_checksums=not args.no_checksum,
        strict=bool(args.strict),
    )


def _directories(loader: PluginLoader) -> list[str]:
    """Return the directories a loader searches."""
    return [str(item) for item in getattr(loader, "directories", ()) or ()]


def _manifest_rows(loader: PluginLoader) -> list[dict[str, Any]]:
    """Describe every discovered manifest as plain data."""
    rows: list[dict[str, Any]] = []
    for manifest in loader.discover():
        sensitive = sorted(set(manifest.permissions) & SENSITIVE_PERMISSIONS)
        rows.append({
            "name": manifest.name,
            "version": manifest.version,
            "enabled": bool(manifest.enabled),
            "tools": len(manifest.tools),
            "hooks": len(manifest.hooks),
            "permissions": list(manifest.permissions),
            "sensitive": sensitive,
            "path": str(manifest.path),
            "summary": manifest.summary,
        })
    rows.sort(key=lambda row: row["name"])
    return rows


def _list(args: argparse.Namespace, context: CliContext) -> int:
    """List discovered plugins, optionally loading them too."""
    loader = _loader(args, context)
    rows = _manifest_rows(loader)
    directories = _directories(loader)
    if getattr(args, "loaded", False):
        report = loader.load_all()
        for row in rows:
            loaded = report.by_name(row["name"])
            if loaded is None:
                row["status"] = "not loaded"
            elif loaded.error:
                row["status"] = f"error: {loaded.error}"
            elif loaded.skipped:
                row["status"] = f"skipped: {loaded.skipped}"
            else:
                row["status"] = (f"loaded (+{len(loaded.tools)} tools, "
                                 f"{loaded.hooks} hooks, {loaded.load_ms:.1f} ms)")
        context.emit({"count": len(rows), "directories": directories, "plugins": rows},
                     text=_render_list(rows, directories, report.render()))
        return EXIT_ERROR if args.strict and report.failed else EXIT_OK

    context.emit({"count": len(rows), "directories": directories, "plugins": rows},
                 text=_render_list(rows, directories, ""))
    return EXIT_OK


def _render_list(rows: list[dict[str, Any]], directories: list[str], extra: str) -> str:
    """Render the plugin list as a table."""
    lines = [render_key_value([("directories", ", ".join(directories) or "(none)"),
                               ("plugins", len(rows))])]
    if rows:
        lines.append("")
        lines.append(render_table(
            [(row["name"], row["version"], row["tools"], row["hooks"],
              "yes" if row["enabled"] else "no",
              ",".join(row["sensitive"]) or "-",
              row.get("status", "")) for row in rows],
            headers=("name", "version", "tools", "hooks", "on", "sensitive", "status"),
        ))
    else:
        lines.append("")
        lines.append("no plugins found; try 'magent plugins scaffold ./plugins/my_plugin'")
    if extra:
        lines.append("")
        lines.append(extra)
    return "\n".join(lines)


def _show(args: argparse.Namespace, context: CliContext) -> int:
    """Print one plugin's manifest in full."""
    loader = _loader(args, context)
    wanted = str(args.name)
    manifest = None
    for candidate in loader.discover():
        if candidate.name == wanted:
            manifest = candidate
            break
    if manifest is None:
        context.printer.error(f"no plugin named {wanted!r}")
        known = [row["name"] for row in _manifest_rows(loader)]
        if known:
            context.printer.write(f"available: {', '.join(known)}")
        return EXIT_ERROR
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(vars(manifest))
    if context.options.fmt == "json":
        context.emit(data)
        return EXIT_OK
    context.printer.heading(f"{manifest.name} {manifest.version}")
    context.printer.write(render_key_value([
        ("summary", manifest.summary or "-"),
        ("author", manifest.author or "-"),
        ("license", manifest.license or "-"),
        ("homepage", manifest.homepage or "-"),
        ("entrypoint", manifest.entrypoint),
        ("path", str(manifest.path)),
        ("enabled", manifest.enabled),
        ("permissions", ", ".join(manifest.permissions) or "(none)"),
        ("sensitive", ", ".join(sorted(set(manifest.permissions) & SENSITIVE_PERMISSIONS))
         or "(none)"),
        ("tools", ", ".join(manifest.tools) or "(none)"),
        ("hooks", ", ".join(manifest.hooks) or "(none)"),
        ("min magent", manifest.min_magent_version or "-"),
        ("checksum", manifest.checksum or "(not declared)"),
        ("env", ", ".join(manifest.env) or "(none)"),
    ]), min_level="quiet")
    return EXIT_OK


def _inspect(args: argparse.Namespace, context: CliContext) -> int:
    """Examine a plugin directory without importing anything."""
    target = Path(str(args.path)).expanduser()
    if not target.exists():
        context.printer.error(f"no such path: {target}")
        return EXIT_ERROR
    try:
        report = inspect_plugin(target)
    except MAgentError as exc:
        context.printer.error(str(exc))
        return EXIT_ERROR
    if context.options.fmt == "json":
        context.emit(report)
        return EXIT_OK
    context.printer.heading(f"inspect {target}")
    pairs = [(str(key), value) for key, value in report.items()
             if not isinstance(value, (dict, list))]
    context.printer.write(render_key_value(pairs), min_level="quiet")
    for key, value in report.items():
        if isinstance(value, (dict, list)) and value:
            context.printer.blank()
            context.printer.write(f"{key}:")
            context.printer.write(json.dumps(value, indent=2, default=str))
    return EXIT_OK


def _scaffold(args: argparse.Namespace, context: CliContext) -> int:
    """Write a starter plugin that actually works when loaded."""
    target = Path(str(args.path)).expanduser()
    name = str(args.name or "example_plugin")
    if context.dry_run:
        context.emit({"command": "plugins scaffold", "path": str(target / name),
                      "name": name, "dry_run": True},
                     text=f"would scaffold plugin {name!r} at {target / name}")
        return EXIT_OK
    try:
        created = scaffold(target, name, force=bool(args.force))
    except MAgentError as exc:
        context.printer.error(str(exc))
        return EXIT_ERROR
    context.printer.write(f"scaffolded plugin {name!r} at {created}", min_level="quiet")
    context.printer.blank()
    for line in ("next:",
                 f"  magent plugins inspect {created}",
                 f"  magent plugins list --dir {created.parent}",
                 f"  magent plugins load --dir {created.parent}"):
        context.printer.write(line, min_level="normal")
    return EXIT_OK


def _permissions(args: argparse.Namespace, context: CliContext) -> int:
    """Document the permission list and which permissions are sensitive."""
    described = describe_permissions()
    rows = [{"permission": name,
             "sensitive": name in SENSITIVE_PERMISSIONS,
             "description": described.get(name, "")} for name in PERMISSIONS]
    if context.options.fmt == "json":
        context.emit({"count": len(rows), "sensitive": sorted(SENSITIVE_PERMISSIONS),
                      "permissions": rows})
        return EXIT_OK
    context.printer.heading("plugin permissions")
    context.printer.write(render_table(
        [(row["permission"], "yes" if row["sensitive"] else "no", row["description"])
         for row in rows],
        headers=("permission", "sensitive", "meaning"),
        max_widths=(20, 10, 60),
    ), min_level="quiet")
    context.printer.blank()
    context.printer.write(
        "A sensitive permission is refused unless the plugin is loaded with "
        "--allow-sensitive <permission>, so a manifest cannot grant itself "
        "network or shell access by declaring it.", min_level="normal")
    return EXIT_OK


def _hooks(args: argparse.Namespace, context: CliContext) -> int:
    """Document the hook points and how each one can affect a run."""
    rows = []
    for name in HOOK_NAMES:
        kinds = []
        if name in TRANSFORM_HOOKS:
            kinds.append("transform")
        if name in VETO_HOOKS:
            kinds.append("veto")
        rows.append({"hook": name, "kind": ",".join(kinds) or "observe"})
    if context.options.fmt == "json":
        context.emit({"count": len(rows), "hooks": rows})
        return EXIT_OK
    context.printer.heading("plugin hooks")
    context.printer.write(render_table(
        [(row["hook"], row["kind"]) for row in rows], headers=("hook", "effect"),
    ), min_level="quiet")
    context.printer.blank()
    context.printer.write(
        "observe: sees the value. transform: may rewrite it. veto: may stop the "
        "run or the tool call. Hooks run in registration order and are timed, so "
        "'magent plugins stats' shows which ones are slow.", min_level="normal")
    return EXIT_OK


def _load(args: argparse.Namespace, context: CliContext) -> int:
    """Load every permitted plugin into a registry and report the result."""
    hooks = HookRegistry()
    loader = _loader(args, context, hooks=hooks)
    if context.dry_run:
        rows = _manifest_rows(loader)
        context.emit({"command": "plugins load", "dry_run": True,
                      "would_load": [row["name"] for row in rows]},
                     text=f"would load {len(rows)} plugin(s): "
                          f"{', '.join(row['name'] for row in rows) or '(none)'}")
        return EXIT_OK
    report = loader.load_all()
    payload = report.to_dict()
    payload["hooks_registered"] = hooks.count() if hasattr(hooks, "count") else 0
    payload["registry_tools"] = context.registry.count()
    if context.options.fmt == "json":
        context.emit(payload)
    else:
        context.printer.write(report.render(), min_level="quiet")
        context.printer.blank()
        context.printer.write(render_key_value([
            ("plugins loaded", len(report.loaded)),
            ("plugins failed", len(report.failed)),
            ("plugins skipped", len(report.skipped)),
            ("tools added", report.tool_count),
            ("hooks registered", payload["hooks_registered"]),
            ("registry total", context.registry.count()),
            ("duration", f"{report.duration_ms:.1f} ms"),
        ]), min_level="quiet")
    if report.failed:
        for loaded in report.plugins:
            if loaded.error:
                context.printer.error(f"{loaded.manifest.name}: {loaded.error}")
        if args.strict:
            return EXIT_ERROR
    return EXIT_OK


def _manifest(args: argparse.Namespace, context: CliContext) -> int:
    """Write a manifest template."""
    name = str(args.name or "example_plugin")
    target = Path(str(args.path)).expanduser() if args.path else \
        context.workspace / name / MANIFEST_FILENAME
    if context.dry_run:
        context.emit({"command": "plugins manifest", "path": str(target), "name": name,
                      "dry_run": True}, text=f"would write a manifest template to {target}")
        return EXIT_OK
    try:
        written = write_manifest_template(target, name)
    except MAgentError as exc:
        context.printer.error(str(exc))
        return EXIT_ERROR
    except OSError as exc:
        context.printer.error(f"cannot write {target}: {exc}")
        return EXIT_ERROR
    size = written.stat().st_size if written.exists() else 0
    context.printer.write(f"wrote {written} ({human_size(size)})", min_level="quiet")
    return EXIT_OK


def _stats(args: argparse.Namespace, context: CliContext) -> int:
    """Show loader statistics and per-plugin size."""
    loader = _loader(args, context)
    payload = loader.stats()
    directories = _directories(loader)
    rows = _manifest_rows(loader)
    payload["directories"] = directories
    payload["discovered"] = len(rows)
    total = 0
    for row in rows:
        path = Path(row["path"])
        try:
            total += path.stat().st_size if path.is_file() else \
                sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        except OSError:
            continue
    payload["bytes_on_disk"] = total
    payload["size"] = human_size(total)
    text_payload = dict(payload)
    text_payload["directories"] = ", ".join(directories) or "(none)"
    text_payload["deny"] = ", ".join(sorted(loader.deny)) or "(none)"
    text_payload["allow"] = ", ".join(sorted(loader.allow)) if loader.allow else "(any)"
    text_payload["allow_sensitive"] = ", ".join(sorted(loader.allow_sensitive)) or "(none)"
    hook_stats = payload.get("hooks")
    if isinstance(hook_stats, dict):
        text_payload["hooks"] = (f"{hook_stats.get('registered', 0)} registered, "
                                 f"{hook_stats.get('executed', 0)} executed, "
                                 f"{hook_stats.get('failures', 0)} failed")
    context.emit(payload, text=render_key_value(list(text_payload.items())))
    return EXIT_OK


__all__ = ["ACTIONS", "COMMAND_NAMES", "cmd_plugins", "register"]
