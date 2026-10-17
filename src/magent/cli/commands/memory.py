"""The ``memory`` command: inspect and maintain long-term memory.

Memory is a JSONL store of embedded entries, ranked at query time by BM25 plus
cosine similarity with time decay. That makes it useful and also makes it easy
to fill with junk - so the command exposes the maintenance operations, not just
reading: dedupe and prune with ``compact``, drop by age with ``prune``, and
remove by match with ``forget``.

Nothing here writes unless the action is explicitly destructive, and the
destructive ones ask first.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...memory import MemoryStore
from ...utils.tables import render_key_value, render_markdown_table, render_table
from ...utils.timeutil import format_timestamp
from ..args import CliContext, EXIT_ERROR, EXIT_OK, confirm


def register(subparsers: Any) -> None:
    """Add the ``memory`` subcommand and its actions."""
    parser = subparsers.add_parser("memory", help="inspect and maintain long-term memory",
                                   description="Search, add and prune stored memories.")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    listing = action.add_parser("list", help="list stored memories")
    listing.add_argument("--kind", default="", help="only this kind")
    listing.add_argument("--limit", type=int, default=20, help="maximum entries")
    listing.add_argument("--order", default="recent", choices=["recent", "important", "used"],
                         help="sort order")

    found = action.add_parser("search", help="search memories by text")
    found.add_argument("query", help="text to look for")
    found.add_argument("--limit", type=int, default=8, help="maximum entries")
    found.add_argument("--kind", default="", help="only this kind")
    found.add_argument("--min-score", type=float, default=0.0, help="score floor")

    added = action.add_parser("add", help="store one memory")
    added.add_argument("text", nargs="+", help="the text to remember")
    added.add_argument("--kind", default="fact", help="entry kind")
    added.add_argument("--tag", action="append", default=[], help="tag (repeatable)")
    added.add_argument("--importance", type=float, default=0.5, help="0.0 to 1.0")
    added.add_argument("--source", default="cli", help="where it came from")

    action.add_parser("stats", help="show store statistics")

    shown = action.add_parser("show", help="print one memory in full")
    shown.add_argument("id", help="entry id or unique prefix")

    removed = action.add_parser("forget", help="delete memories matching a query")
    removed.add_argument("query", help="text to match")
    removed.add_argument("--min-score", type=float, default=0.5, help="score floor")
    removed.add_argument("--kind", default="", help="only this kind")

    deleted = action.add_parser("delete", help="delete one memory by id")
    deleted.add_argument("id", help="entry id or unique prefix")

    pruned = action.add_parser("prune", help="delete memories older than an age")
    pruned.add_argument("--older-than-days", type=float, default=30.0, help="age threshold")
    pruned.add_argument("--kind", default="", help="only this kind")

    compact = action.add_parser("compact", help="deduplicate and prune to a size")
    compact.add_argument("--keep", type=int, default=50, help="entries to keep per kind")
    compact.add_argument("--threshold", type=float, default=0.95, help="dedupe similarity")

    exported = action.add_parser("export", help="export the store to JSONL")
    exported.add_argument("path", help="destination file")
    exported.add_argument("--vectors", action="store_true", help="include embeddings")

    action.add_parser("clear", help="delete every memory")
    parser.set_defaults(handler=cmd_memory, action_default="stats")


def _store(context: CliContext) -> MemoryStore:
    """Open the configured memory store."""
    return MemoryStore.from_config(context.config)


def _entry_rows(entries: list[Any], *, scores: bool = False) -> list[list[Any]]:
    """Build table rows for a list of entries."""
    rows: list[list[Any]] = []
    for entry in entries:
        row: list[Any] = [
            entry.id[:12],
            entry.kind,
            format_timestamp(entry.created_at, fmt="%Y-%m-%d %H:%M"),
            f"{entry.importance:.2f}",
            entry.use_count,
            " ".join((entry.content or "").split())[:56],
        ]
        if scores:
            row.insert(2, f"{getattr(entry, 'score', 0.0):.3f}")
        rows.append(row)
    return rows


HEADERS = ["id", "kind", "created", "importance", "uses", "content"]
"""Columns for a memory listing."""

SCORE_HEADERS = ["id", "kind", "score", "created", "importance", "uses", "content"]
"""Columns for a scored memory search."""


def cmd_memory(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``memory`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "stats")
    if not context.config.memory.enabled:
        context.printer.warn("memory is disabled by [memory] enabled=false")
        return EXIT_OK
    handlers: dict[str, Any] = {
        "list": _list, "search": _search, "add": _add, "stats": _stats,
        "show": _show, "forget": _forget, "delete": _delete, "prune": _prune,
        "compact": _compact, "export": _export, "clear": _clear,
    }
    return handlers.get(action, _stats)(args, context)


def _list(args: argparse.Namespace, context: CliContext) -> int:
    """List stored memories."""
    store = _store(context)
    entries = store.list_entries(limit=int(getattr(args, "limit", 20) or 20),
                                 kind=str(getattr(args, "kind", "") or ""),
                                 order=str(getattr(args, "order", "recent") or "recent"))
    if context.options.fmt == "json":
        context.emit({"count": len(entries),
                      "entries": [entry.to_dict() for entry in entries]})
        return EXIT_OK
    if not entries:
        context.printer.write("no memories stored", min_level="quiet")
        return EXIT_OK
    rows = _entry_rows(entries)
    table = render_markdown_table(rows, headers=HEADERS) if context.options.fmt == "markdown" \
        else render_table(rows, headers=HEADERS,
                          alignments=["left", "left", "left", "right", "right", "left"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(entries)} of {len(store)} entries", "muted"), min_level="normal")
    return EXIT_OK


def _search(args: argparse.Namespace, context: CliContext) -> int:
    """Search memories and show scores."""
    store = _store(context)
    query = " ".join(getattr(args, "query", []) or []) if isinstance(
        getattr(args, "query", None), list) else str(getattr(args, "query", "") or "")
    entries = store.search(query, limit=int(getattr(args, "limit", 8) or 8),
                           kind=str(getattr(args, "kind", "") or ""),
                           min_score=float(getattr(args, "min_score", 0.0) or 0.0))
    if context.options.fmt == "json":
        context.emit({"query": query, "count": len(entries),
                      "results": [{**entry.to_dict(), "score": getattr(entry, "score", 0.0)}
                                  for entry in entries]})
        return EXIT_OK
    if not entries:
        context.printer.write(f"no memories match {query!r}", min_level="quiet")
        return EXIT_OK
    rows = _entry_rows(entries, scores=True)
    table = render_markdown_table(rows, headers=SCORE_HEADERS) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=SCORE_HEADERS,
                          alignments=["left", "left", "right", "left", "right", "right", "left"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    return EXIT_OK


def _add(args: argparse.Namespace, context: CliContext) -> int:
    """Store one memory."""
    text = " ".join(getattr(args, "text", []) or []).strip()
    if not text:
        context.printer.error("error: nothing to remember")
        return EXIT_ERROR
    store = _store(context)
    if context.dry_run:
        context.printer.write(f"would remember {len(text)} chars as "
                              f"{getattr(args, 'kind', 'fact')}", min_level="quiet")
        return EXIT_OK
    entry = store.add(text, kind=str(getattr(args, "kind", "fact") or "fact"),
                      tags=list(getattr(args, "tag", []) or []),
                      importance=float(getattr(args, "importance", 0.5) or 0.5),
                      source=str(getattr(args, "source", "cli") or "cli"))
    store.save()
    if context.options.fmt == "json":
        context.emit(entry.to_dict())
        return EXIT_OK
    context.printer.write(f"remembered {entry.id} ({entry.kind}, "
                          f"importance {entry.importance:.2f})", min_level="quiet")
    return EXIT_OK


def _stats(args: argparse.Namespace, context: CliContext) -> int:
    """Show store statistics."""
    store = _store(context)
    payload = store.stats()
    payload["path"] = str(getattr(store, "path", "") or "")
    payload["entries"] = len(store)
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    context.printer.write(render_key_value([(str(key), value) for key, value in payload.items()]),
                          min_level="quiet")
    return EXIT_OK


def _show(args: argparse.Namespace, context: CliContext) -> int:
    """Print one memory in full."""
    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    entry = _resolve(store, wanted)
    if entry is None:
        context.printer.error(f"error: no memory matches {wanted!r}")
        return EXIT_ERROR
    if context.options.fmt == "json":
        context.emit(entry.to_dict())
        return EXIT_OK
    printer = context.printer
    printer.write(printer.themed(entry.id, "heading"), min_level="quiet")
    printer.write(render_key_value([
        ("kind", entry.kind),
        ("created", format_timestamp(entry.created_at)),
        ("importance", f"{entry.importance:.2f}"),
        ("uses", entry.use_count),
        ("tags", ", ".join(entry.tags) or "-"),
        ("source", entry.source or "-"),
        ("embedding", f"{len(entry.embedding or [])} dims"),
    ]), min_level="quiet")
    printer.blank(min_level="quiet")
    printer.write(entry.content, min_level="quiet")
    if entry.metadata:
        printer.blank(min_level="quiet")
        printer.write(printer.themed("metadata", "muted"), min_level="normal")
        printer.write(json.dumps(entry.metadata, indent=2, ensure_ascii=False, default=str),
                      min_level="normal")
    return EXIT_OK


def _resolve(store: MemoryStore, wanted: str) -> Any | None:
    """Find an entry by id or by a unique prefix."""
    if not wanted:
        return None
    direct = store.get(wanted) if hasattr(store, "get") else None
    if direct is not None:
        return direct
    matches = [entry for entry in store.list_entries(limit=100000)
               if entry.id.startswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def _forget(args: argparse.Namespace, context: CliContext) -> int:
    """Delete memories matching a query."""
    store = _store(context)
    query = str(getattr(args, "query", "") or "")
    matches = store.search(query, limit=50, kind=str(getattr(args, "kind", "") or ""),
                           min_score=float(getattr(args, "min_score", 0.5) or 0.0))
    if not matches:
        context.printer.write("nothing matched", min_level="quiet")
        return EXIT_OK
    if context.dry_run:
        context.printer.write(f"would forget {len(matches)} memories:", min_level="quiet")
        for entry in matches[:10]:
            context.printer.write(f"  {entry.id[:12]} {' '.join(entry.content.split())[:60]}",
                                  min_level="quiet")
        return EXIT_OK
    if not confirm(f"forget {len(matches)} memories matching {query!r}?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    removed = sum(1 for entry in matches if store.delete(entry.id))
    store.save()
    context.printer.write(f"forgot {removed} memories", min_level="quiet")
    return EXIT_OK


def _delete(args: argparse.Namespace, context: CliContext) -> int:
    """Delete one memory by id."""
    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    entry = _resolve(store, wanted)
    if entry is None:
        context.printer.error(f"error: no memory matches {wanted!r}")
        return EXIT_ERROR
    if context.dry_run:
        context.printer.write(f"would delete {entry.id}", min_level="quiet")
        return EXIT_OK
    if not confirm(f"delete memory {entry.id}: {' '.join(entry.content.split())[:60]}?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    store.delete(entry.id)
    store.save()
    context.printer.write(f"deleted {entry.id}", min_level="quiet")
    return EXIT_OK


def _prune(args: argparse.Namespace, context: CliContext) -> int:
    """Delete memories older than a threshold."""
    store = _store(context)
    days = float(getattr(args, "older_than_days", 30.0) or 30.0)
    seconds = max(0.0, days * 86400.0)
    if context.dry_run:
        context.printer.write(f"would delete memories older than {days} days", min_level="quiet")
        return EXIT_OK
    if not confirm(f"delete memories older than {days} days?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    removed = store.delete_older_than(seconds, kind=str(getattr(args, "kind", "") or ""))
    store.save()
    context.printer.write(f"pruned {removed} memories older than {days} days", min_level="quiet")
    return EXIT_OK


def _compact(args: argparse.Namespace, context: CliContext) -> int:
    """Deduplicate and prune the store."""
    store = _store(context)
    before = len(store)
    keep = int(getattr(args, "keep", 50) or 50)
    threshold = float(getattr(args, "threshold", 0.95) or 0.95)
    if context.dry_run:
        context.printer.write(f"would compact to {keep} entries per kind "
                              f"(dedupe above {threshold}); {before} stored now",
                              min_level="quiet")
        return EXIT_OK
    report = store.compact(keep=keep, dedupe_threshold=threshold)
    store.save()
    payload = {"before": before, "after": len(store), **report}
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    context.printer.write(render_key_value([(str(key), value) for key, value in payload.items()]),
                          min_level="quiet")
    return EXIT_OK


def _export(args: argparse.Namespace, context: CliContext) -> int:
    """Export the store to JSONL."""
    store = _store(context)
    target = str(getattr(args, "path", "") or "")
    if not target:
        context.printer.error("error: a destination path is required")
        return EXIT_ERROR
    if context.dry_run:
        context.printer.write(f"would export {len(store)} memories to {target}", min_level="quiet")
        return EXIT_OK
    written = store.export_jsonl(target, include_embeddings=bool(getattr(args, "vectors", False)))
    context.printer.write(f"exported {len(store)} memories to {written}", min_level="quiet")
    return EXIT_OK


def _clear(args: argparse.Namespace, context: CliContext) -> int:
    """Delete every memory."""
    store = _store(context)
    count = len(store)
    if not count:
        context.printer.write("the store is already empty", min_level="quiet")
        return EXIT_OK
    if context.dry_run:
        context.printer.write(f"would delete all {count} memories", min_level="quiet")
        return EXIT_OK
    if not confirm(f"delete all {count} memories? this cannot be undone",
                   context=context, default=False, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    store.clear()
    store.save()
    context.printer.write(f"deleted {count} memories", min_level="quiet")
    return EXIT_OK


__all__ = ["HEADERS", "SCORE_HEADERS", "cmd_memory", "register"]
