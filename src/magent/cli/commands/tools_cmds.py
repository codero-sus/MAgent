"""Commands for the tool layer: ``tools``, ``presets``, ``skills``, ``plugins``.

Inspection commands are deliberately useful on their own: ``tools --category git``
answers "what can this thing do to my repository", ``tools inspect write_file``
prints the exact JSON schema the model sees, ``tools call calculate`` runs one
tool without involving a model at all - which is how you check a tool's behaviour
when a run misbehaves.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ..commands import Command
from ...errors import MAgentError
from ...utils.text import ellipsize


def _registry(ctx: Any) -> Any:
    """Return the tool registry, building it if needed."""
    return ctx.registry


def tools_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent tools`` - list, search or inspect tools."""
    registry = _registry(ctx)
    action = getattr(args, "action", "") or ""
    if action == "call":
        return _call_tool(ctx, registry, args)
    if action == "inspect":
        return _inspect_tool(ctx, registry, args)
    tools = registry.search(text=args.query) if args.query else registry.tools()
    if getattr(args, "category", ""):
        tools = [tool for tool in tools if tool.spec.category == args.category]
    if getattr(args, "risk", ""):
        tools = [tool for tool in tools if tool.spec.risk == args.risk]
    if args.verbose:
        rows = [(tool.spec.name, tool.spec.category, tool.spec.risk,
                 str(len(tool.spec.parameters)), tool.spec.description.split("\n")[0][:70])
                for tool in tools]
        ctx.table(rows, ["name", "category", "risk", "args", "description"])
    else:
        rows = [(tool.spec.name, tool.spec.category, tool.spec.risk,
                 ellipsize(tool.spec.description.split("\n")[0], 60)) for tool in tools]
        ctx.table(rows, ["name", "category", "risk", "description"],
                  title=f"{len(tools)} tools")
    if ctx.config.output.format == "text":
        ctx.notice(f"{registry.count()} tools in {len(registry.categories())} categories",
                   level="muted")
    return 0


def _inspect_tool(ctx: Any, registry: Any, args: argparse.Namespace) -> int:
    """Print one tool's schema and metadata."""
    name = args.query
    try:
        tool = registry.get(name)
    except MAgentError as exc:
        ctx.error(str(exc))
        suggestions = registry.suggest(name)
        if suggestions:
            ctx.notice("did you mean: " + ", ".join(suggestions), level="warning")
        return 1
    if ctx.config.output.format != "text":
        ctx.emit(tool.to_openai(), fmt=ctx.config.output.format)
        return 0
    spec = tool.spec
    ctx.key_values([
        ("name", spec.name),
        ("category", spec.category),
        ("risk", spec.risk),
        ("destructive", spec.destructive),
        ("network", spec.network),
        ("filesystem", spec.filesystem),
        ("executes_code", spec.executes_code),
        ("parallel_safe", spec.parallel_safe),
        ("requires_approval", spec.requires_approval),
        ("timeout", spec.timeout),
        ("aliases", ", ".join(spec.aliases) or "-"),
        ("tags", ", ".join(spec.tags) or "-"),
    ], title=f"tool {spec.name}")
    ctx.print("")
    ctx.print(spec.description)
    ctx.print("")
    ctx.print("schema:")
    ctx.write(json.dumps(tool.to_openai()["function"]["parameters"], indent=2) + "\n")
    if spec.examples:
        ctx.print("")
        ctx.print("examples:")
        for example in spec.examples:
            ctx.print(f"  {example}")
    return 0


def _call_tool(ctx: Any, registry: Any, args: argparse.Namespace) -> int:
    """Invoke one tool with JSON arguments."""
    name = args.query
    try:
        tool = registry.get(name)
    except MAgentError as exc:
        ctx.error(str(exc))
        return 1
    raw = args.arguments or "{}"
    try:
        arguments = json.loads(raw) if raw.strip().startswith("{") else {"value": raw}
    except ValueError as exc:
        ctx.error(f"arguments must be JSON: {exc}")
        return 2
    if ":" in raw and not raw.strip().startswith("{"):
        key, _, value = raw.partition(":")
        arguments = {key.strip(): value.strip()}
    context = None
    try:
        context = ctx.build_agent().tool_context()
    except Exception:  # noqa: BLE001 - a context is optional for read-only tools
        context = None
    try:
        result = tool.run(arguments, context=context)
    except Exception as exc:  # noqa: BLE001 - a tool failure is the command's output
        ctx.error(f"{type(exc).__name__}: {exc}")
        return 1
    if ctx.config.output.format != "text":
        ctx.emit(result.to_dict() if hasattr(result, "to_dict") else result,
                 fmt=ctx.config.output.format)
        return 0 if result.ok else 1
    ctx.notice(f"{name} {'ok' if result.ok else 'failed'} in {result.duration_ms:.1f} ms",
               level="success" if result.ok else "error")
    if result.ok:
        payload = result.content
        ctx.print(json.dumps(payload, indent=2, default=str) if not isinstance(payload, str)
                  else payload)
    else:
        ctx.error(result.error or "tool failed")
    return 0 if result.ok else 1


def presets_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent presets`` - list agent presets, or show one in detail."""
    from ...agents.presets import get_preset, list_presets

    name = getattr(args, "name", "") or ""
    if name:
        preset = get_preset(name)
        if ctx.config.output.format != "text":
            ctx.emit(preset.to_dict(), fmt=ctx.config.output.format)
            return 0
        ctx.key_values([
            ("name", preset.name),
            ("category", preset.category),
            ("persona", preset.persona),
            ("loop", preset.loop_mode),
            ("iterations", preset.max_iterations),
            ("planning", preset.planning),
            ("reflection", preset.reflection),
            ("temperature", preset.temperature),
            ("max risk", preset.max_risk or "any"),
            ("network", preset.allow_network),
            ("tools", ", ".join(preset.tools) or "(filter below)"),
            ("categories", ", ".join(preset.tool_categories) or "-"),
            ("no tools", preset.no_tools),
        ], title=f"preset {preset.name}")
        if preset.description:
            ctx.print("")
            ctx.print(preset.description)
        if preset.instructions:
            ctx.print("")
            ctx.print(preset.instructions)
        return 0
    rows = [(preset.name, preset.category, preset.persona,
             ellipsize(preset.description, 60)) for preset in list_presets()]
    ctx.table(rows, ["name", "category", "persona", "description"],
              title=f"{len(rows)} presets")
    return 0


def skills_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent skills`` - list skills or match them against a task."""
    library = ctx.skills
    task = getattr(args, "task", "") or ""
    name = getattr(args, "name", "") or ""
    if name:
        skill = library.get(name)
        if ctx.config.output.format != "text":
            ctx.emit(skill.to_dict(include_body=True), fmt=ctx.config.output.format)
            return 0
        ctx.key_values([("name", skill.name), ("tags", ", ".join(skill.tags) or "-"),
                        ("triggers", ", ".join(skill.triggers) or "-"),
                        ("priority", skill.priority), ("words", skill.words),
                        ("source", skill.source or "(built-in)")], title=f"skill {skill.name}")
        ctx.print("")
        ctx.print(skill.render())
        return 0
    if task:
        matches = library.search(task)
        rows = [(match.skill.name, f"{match.score:.2f}", ", ".join(match.reasons[:4]))
                for match in matches]
        ctx.table(rows, ["skill", "score", "why"], title=f"skills matching {task!r}")
        if not matches:
            ctx.notice("no skill matched", level="warning")
        return 0
    rows = [(skill.name, ",".join(skill.tags[:3]), str(skill.priority),
             ellipsize(skill.description, 54)) for skill in library]
    ctx.table(rows, ["name", "tags", "prio", "description"], title=library.describe())
    return 0


def plugins_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent plugins`` - load, list and inspect plugins."""
    from ...plugins import default_plugin_paths, load_plugins, manifest_for

    action = getattr(args, "action", "") or "list"
    paths = default_plugin_paths(ctx.config, workspace=ctx.workspace)
    if action == "paths":
        rows = [(str(path), "yes" if path.exists() else "no") for path in paths]
        ctx.table(rows, ["path", "exists"], title="plugin search path")
        return 0
    if action == "check":
        loader = load_plugins(ctx.config, registry=ctx.registry, hooks=ctx.hooks,
                              workspace=ctx.workspace, strict=False)
        rows = [(record.name, record.manifest.version, "ok" if record.ok else "fail",
                 str(record.contributions), record.error[:50]) for record in loader.loaded.values()]
        ctx.table(rows, ["plugin", "version", "status", "items", "error"],
                  title=f"{loader.stats()['loaded']} plugins")
        return 0 if not loader.failed() else 1
    if action == "inspect" and args.name:
        for path in paths:
            candidate = path / args.name
            if candidate.exists() or (path / "plugins" / args.name).exists():
                manifest = manifest_for(candidate if candidate.exists()
                                        else path / "plugins" / args.name)
                if ctx.config.output.format != "text":
                    ctx.emit(manifest.to_dict(), fmt=ctx.config.output.format)
                    return 0
                ctx.key_values([(key, value) for key, value in manifest.to_dict().items()
                                if not isinstance(value, (dict, list))],
                               title=f"plugin {manifest.name}")
                ctx.print("")
                ctx.print("problems: " + ("; ".join(manifest.validate()) or "none"))
                return 0
        ctx.error(f"plugin {args.name!r} not found in {', '.join(str(p) for p in paths)}")
        return 1
    loader = load_plugins(ctx.config, registry=ctx.registry, hooks=ctx.hooks,
                          workspace=ctx.workspace)
    if not loader.loaded:
        ctx.notice("no plugins found; searched " + ", ".join(str(path) for path in paths))
        return 0
    rows = [(record.name, record.manifest.version,
             "ok" if record.ok else "fail", str(record.contributions),
             ellipsize(record.error or ", ".join(record.tools + record.hooks + record.skills), 50))
            for record in loader.loaded.values()]
    ctx.table(rows, ["plugin", "version", "status", "items", "detail"],
              title=f"{loader.stats()['loaded']} plugins loaded")
    return 0


def configure_tools(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent tools``."""
    parser.add_argument("action", nargs="?", default="", choices=["", "list", "inspect", "call"],
                        help="list (default), inspect one tool, or call one")
    parser.add_argument("query", nargs="?", default="", help="tool name or search text")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON arguments for `call`")
    parser.add_argument("--category", default="", help="filter by category")
    parser.add_argument("--risk", default="", help="filter by risk level")


def configure_presets(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent presets``."""
    parser.add_argument("name", nargs="?", default="", help="preset to describe")


def configure_skills(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent skills``."""
    parser.add_argument("name", nargs="?", default="", help="skill to show")
    parser.add_argument("--task", default="", help="text to match skills against")


def configure_plugins(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent plugins``."""
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "check", "paths", "inspect"], help="what to do")
    parser.add_argument("name", nargs="?", default="", help="plugin name for `inspect`")


COMMANDS: tuple[Command, ...] = (
    Command(
        name="tools",
        help="List, inspect or call tools",
        run=tools_command,
        group="tools",
        aliases=("t",),
        configure=configure_tools,

        description=(
            "Without a name this lists every registered tool. With `inspect` it prints "
            "the exact JSON schema the model receives; with `call` it runs a tool "
            "directly, which is the fastest way to reproduce a tool bug."
        ),
        examples=("magent tools", "magent tools inspect write_file",
                  "magent tools call calculate '{\"expression\": \"6*7\"}'"),
    ),
    Command(
        name="presets",
        help="List or describe agent presets",
        run=presets_command,
        group="tools",
        configure=configure_presets,

        examples=("magent presets", "magent presets coder"),
    ),
    Command(
        name="skills",
        help="List skills or match them to a task",
        run=skills_command,
        group="tools",
        configure=configure_skills,

        examples=("magent skills", "magent skills --task 'review this pull request'",
                  "magent skills code-review"),
    ),
    Command(
        name="plugins",
        help="Load, list and check plugins",
        run=plugins_command,
        group="system",
        configure=configure_plugins,

        examples=("magent plugins paths", "magent plugins check", "magent plugins inspect my-plugin"),
    ),
)


__all__ = [
    "COMMANDS",
    "configure_plugins",
    "configure_presets",
    "configure_skills",
    "configure_tools",
    "plugins_command",
    "presets_command",
    "skills_command",
    "tools_command",
]
