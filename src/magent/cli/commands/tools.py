"""The ``tools`` command: inspect the tool catalogue.

An agent that can call 113 tools is only useful if you can find out what they
are, what they cost in risk terms, and what arguments they take. Every answer
here comes from the live registry - the same objects the agent dispatches - so
the listing cannot drift from the implementation.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...tools import ToolRegistry
from ...tools.base import RISK_LEVELS
from ...utils.tables import render_markdown_table, render_table
from ..args import CliContext, EXIT_ERROR, EXIT_OK


def register(subparsers: Any) -> None:
    """Add the ``tools`` subcommand and its actions."""
    parser = subparsers.add_parser("tools", help="inspect the built-in tool catalogue",
                                   description="List, search and describe tools.")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    listing = action.add_parser("list", help="list tools")
    listing.add_argument("--category", default="", metavar="NAME", help="only this category")
    listing.add_argument("--max-risk", default="", choices=["", *RISK_LEVELS],
                         help="hide tools riskier than this")
    listing.add_argument("--tag", default="", metavar="TAG", help="only tools with this tag")

    cats = action.add_parser("categories", help="list categories with counts")
    cats.add_argument("unused", nargs="*", help=argparse.SUPPRESS)

    shown = action.add_parser("show", help="describe one tool in full")
    shown.add_argument("name", help="tool name")

    found = action.add_parser("search", help="search tools by name, tag or description")
    found.add_argument("query", help="text to look for")
    found.add_argument("--limit", type=int, default=20, help="maximum results")

    schema = action.add_parser("schema", help="print tool schemas as JSON")
    schema.add_argument("names", nargs="*", help="tool names, empty for all")
    schema.add_argument("--category", default="", help="only this category")

    action.add_parser("risks", help="explain the risk ladder")
    parser.set_defaults(handler=cmd_tools, action_default="list")


def cmd_tools(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``tools`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "list")
    registry: ToolRegistry = context.registry
    handlers = {
        "list": _list,
        "categories": _categories,
        "show": _show,
        "search": _search,
        "schema": _schema,
        "risks": _risks,
    }
    handler = handlers.get(action, _list)
    return handler(args, context, registry)


def _filter(registry: ToolRegistry, args: argparse.Namespace) -> list[Any]:
    """Apply the shared list filters.

    ``ToolRegistry.search`` takes empty strings for "no constraint" and calls
    ``.strip()`` on them, so passing ``None`` would raise; and it returns
    :class:`~magent.tools.base.ToolSpec` objects, not ``Tool`` instances.
    """
    category = str(getattr(args, "category", "") or "")
    max_risk = str(getattr(args, "max_risk", "") or "")
    tag = str(getattr(args, "tag", "") or "")
    specs = registry.search(category=category, tag=tag, max_risk=max_risk or None)
    return sorted(specs, key=lambda spec: (spec.category, spec.name))


def _rows(specs: list[Any]) -> list[list[Any]]:
    """Build table rows for a list of tool specifications."""
    return [[spec.name, spec.category, spec.risk,
             "yes" if spec.destructive else "",
             "yes" if spec.network else "",
             (spec.description or "").split("\n")[0][:58]] for spec in specs]


HEADERS = ["name", "category", "risk", "destructive", "network", "summary"]
"""Column headers for the tool listing."""


def _list(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """List tools, optionally filtered."""
    tools = _filter(registry, args)
    if not tools:
        context.printer.warn("no tools matched")
        return EXIT_OK
    if context.options.fmt == "json":
        context.emit({"tools": [spec.to_dict() for spec in tools], "count": len(tools)})
        return EXIT_OK
    rows = _rows(tools)
    if context.options.fmt == "markdown":
        context.printer.write(render_markdown_table(rows, headers=HEADERS), min_level="quiet")
    else:
        context.printer.write(render_table(rows, headers=HEADERS,
                                           unicode_borders=context.options.unicode),
                              min_level="quiet")
        context.printer.write(context.printer.themed(
            f"{len(tools)} tools of {registry.count()} registered", "muted"), min_level="normal")
    return EXIT_OK


def _categories(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """Show tool counts per category."""
    grouped = registry.by_category()
    payload = {name: len(items) for name, items in sorted(grouped.items(),
                                                           key=lambda kv: (-len(kv[1]), kv[0]))}
    if context.options.fmt == "json":
        context.emit({"categories": payload, "total": registry.count()})
        return EXIT_OK
    rows = [[name, count] for name, count in payload.items()]
    table = render_markdown_table(rows, headers=["category", "tools"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["category", "tools"], alignments=["left", "right"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(payload)} categories, {registry.count()} tools", "muted"), min_level="normal")
    return EXIT_OK


def _show(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """Describe one tool completely."""
    name = str(getattr(args, "name", "") or "")
    found = registry.get(name)
    if found is None:
        suggestions = registry.suggest(name, limit=5)
        hint = f" did you mean: {', '.join(suggestions)}" if suggestions else ""
        context.printer.error(f"error: unknown tool {name!r}.{hint}")
        return EXIT_ERROR
    spec = found.spec
    payload = {
        "name": spec.name,
        "category": spec.category,
        "risk": spec.risk,
        "destructive": spec.destructive,
        "network": spec.network,
        "filesystem": spec.filesystem,
        "executes_code": spec.executes_code,
        "parallel_safe": spec.parallel_safe,
        "requires_approval": spec.requires_approval,
        "timeout": spec.timeout,
        "tags": list(spec.tags),
        "aliases": list(getattr(spec, "aliases", []) or []),
        "description": spec.description,
        "parameters": spec.parameters,
        "examples": list(getattr(spec, "examples", []) or []),
    }
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    printer = context.printer
    printer.write(printer.themed(payload["name"], "heading"), min_level="quiet")
    printer.write(printer.themed(
        f"{payload['category']} · risk {payload['risk']}"
        + (" · destructive" if payload["destructive"] else "")
        + (" · network" if payload["network"] else "")
        + (" · needs approval" if payload["requires_approval"] else ""), "muted"),
        min_level="quiet")
    printer.blank(min_level="quiet")
    printer.write(payload["description"], min_level="quiet")
    properties = (payload["parameters"] or {}).get("properties") or {}
    required = set((payload["parameters"] or {}).get("required") or [])
    if properties:
        printer.blank(min_level="quiet")
        printer.write(printer.themed("arguments", "heading"), min_level="quiet")
        rows = [[key, (value or {}).get("type", ""), "required" if key in required else "optional",
                 str((value or {}).get("description", ""))[:52]]
                for key, value in properties.items()]
        printer.write(render_table(rows, headers=["name", "type", "kind", "description"],
                                   unicode_borders=context.options.unicode), min_level="quiet")
    if payload["examples"]:
        printer.blank(min_level="quiet")
        printer.write(printer.themed("examples", "heading"), min_level="quiet")
        for example in payload["examples"]:
            printer.write(f"  {example}", min_level="quiet")
    return EXIT_OK


def _search(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """Search tools by text."""
    query = str(getattr(args, "query", "") or "")
    limit = max(1, int(getattr(args, "limit", 20) or 20))
    found = registry.search(text=query)[:limit]
    if context.options.fmt == "json":
        context.emit({"query": query, "count": len(found),
                      "tools": [{"name": spec.name, "category": spec.category,
                                 "risk": spec.risk,
                                 "summary": (spec.description or "").split("\n")[0]}
                                for spec in found]})
        return EXIT_OK
    if not found:
        context.printer.warn(f"no tools match {query!r}")
        return EXIT_OK
    rows = [[spec.name, spec.category, spec.risk,
             (spec.description or "").split("\n")[0][:60]] for spec in found]
    table = render_markdown_table(rows, headers=["name", "category", "risk", "summary"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["name", "category", "risk", "summary"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    return EXIT_OK


def _schema(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """Print tool schemas as JSON, for the model or for debugging."""
    names = list(getattr(args, "names", []) or [])
    category = getattr(args, "category", "") or ""
    if names:
        schemas = [registry.require(name).to_openai() for name in names]
    elif category:
        schemas = registry.schemas(categories=[category])
    else:
        schemas = registry.schemas()
    context.printer.write(json.dumps(schemas, indent=2, ensure_ascii=False, default=str),
                          min_level="quiet")
    return EXIT_OK


def _risks(args: argparse.Namespace, context: CliContext, registry: ToolRegistry) -> int:
    """Explain the risk ladder and count tools at each level."""
    counts = {level: 0 for level in RISK_LEVELS}
    for spec in registry.search():
        if spec.risk in counts:
            counts[spec.risk] += 1
    rows = [[index, level, counts[level], _risk_meaning(level)]
            for index, level in enumerate(RISK_LEVELS)]
    if context.options.fmt == "json":
        context.emit({"risk_levels": RISK_LEVELS, "counts": counts})
        return EXIT_OK
    table = render_markdown_table(rows, headers=["#", "risk", "tools", "meaning"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["#", "risk", "tools", "meaning"],
                          alignments=["right", "left", "right", "left"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    return EXIT_OK


def _risk_meaning(level: str) -> str:
    """Return a one-line explanation of a risk level."""
    meanings = {
        "safe": "pure computation, no side effects",
        "read": "reads files, environment or state",
        "write": "creates or modifies files",
        "execute": "runs code or commands",
        "destructive": "deletes data or rewrites history",
        "network": "contacts a remote host",
    }
    return meanings.get(level, "")


__all__ = ["HEADERS", "cmd_tools", "register"]
