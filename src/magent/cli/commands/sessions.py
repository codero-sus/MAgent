"""The ``sessions`` command: list, inspect, resume and prune transcripts.

Sessions are how a conversation survives the process that had it. The command
exposes the whole lifecycle - find one, read it, export it for a bug report,
resume it, and reclaim the disk it occupies - because a transcript you cannot
get at is only a disk cost.

``resume`` hands the tail of a transcript back to the agent as history, which
is what makes "continue where we left off" work without re-explaining context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...output import render_session_info
from ...session import SessionStore, cleanup
from ...utils.files import human_size
from ...utils.tables import render_key_value
from ..args import CliContext, EXIT_ERROR, EXIT_OK, confirm


def register(subparsers: Any) -> None:
    """Add the ``sessions`` subcommand and its actions."""
    parser = subparsers.add_parser("sessions", help="inspect stored conversations",
                                   description="List, read, export, resume and prune sessions.")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    listing = action.add_parser("list", help="list stored sessions")
    listing.add_argument("--limit", type=int, default=20, help="maximum sessions")
    listing.add_argument("--tag", default="", help="only sessions with this tag")
    listing.add_argument("--query", default="", help="filter by title text")

    shown = action.add_parser("show", help="print one transcript")
    shown.add_argument("id", help="session id or unique prefix")
    shown.add_argument("--fmt", default="", choices=["", "markdown", "text", "json", "jsonl"],
                       help="rendering to use")
    shown.add_argument("--limit", type=int, default=0, help="only the last N entries")
    shown.add_argument("--metadata", action="store_true", help="include per-entry metadata")

    exported = action.add_parser("export", help="write one transcript to a file")
    exported.add_argument("id", help="session id or unique prefix")
    exported.add_argument("path", help="destination file")
    exported.add_argument("--fmt", default="", choices=["", "markdown", "text", "json", "jsonl"],
                          help="rendering to use")

    resumed = action.add_parser("resume", help="continue a session in chat")
    resumed.add_argument("id", nargs="?", default="", help="session id, empty for the newest")
    resumed.add_argument("--keep", type=int, default=20, help="messages to replay")

    deleted = action.add_parser("delete", help="delete one session")
    deleted.add_argument("id", help="session id or unique prefix")

    pruned = action.add_parser("prune", help="delete the oldest sessions")
    pruned.add_argument("--keep", type=int, default=0, help="how many to keep")

    found = action.add_parser("search", help="search transcript contents")
    found.add_argument("query", nargs="+", help="text to look for")
    found.add_argument("--limit", type=int, default=10, help="maximum matches")

    action.add_parser("stats", help="show store statistics")

    cleared = action.add_parser("clear", help="delete every session")
    cleared.add_argument("--older-than-days", type=float, default=0.0,
                         help="only sessions idle this long, 0 for all")

    parser.set_defaults(handler=cmd_sessions, action_default="list")


def _store(context: CliContext) -> SessionStore:
    """Build the configured session store."""
    return SessionStore.from_config(context.config)


def cmd_sessions(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``sessions`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "list")
    if not context.config.session.enabled:
        context.printer.warn("sessions are disabled by [session] enabled=false")
        return EXIT_OK
    handlers: dict[str, Any] = {
        "list": _list, "show": _show, "export": _export, "resume": _resume,
        "delete": _delete, "prune": _prune, "search": _search, "stats": _stats,
        "clear": _clear,
    }
    return handlers.get(action, _list)(args, context)


def _list(args: argparse.Namespace, context: CliContext) -> int:
    """List stored sessions, newest first."""
    store = _store(context)
    infos = store.list(limit=int(getattr(args, "limit", 20) or 20),
                       tag=str(getattr(args, "tag", "") or ""),
                       query=str(getattr(args, "query", "") or ""))
    if context.options.fmt == "json":
        context.emit({"count": len(infos), "directory": str(store.directory),
                      "sessions": [info.to_dict() for info in infos]})
        return EXIT_OK
    if not infos:
        context.printer.write(f"no sessions in {store.directory}", min_level="quiet")
        return EXIT_OK
    context.printer.write(render_session_info(infos, options=context.options), min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(infos)} of {store.count()} sessions, {human_size(store.total_bytes())} total",
        "muted"), min_level="normal")
    return EXIT_OK


def _show(args: argparse.Namespace, context: CliContext) -> int:
    """Print one transcript."""
    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    try:
        session = store.open(wanted)
    except Exception as exc:  # noqa: BLE001 - an unknown id is a normal failure
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    limit = int(getattr(args, "limit", 0) or 0)
    if limit and limit < len(session):
        session.transcript.entries = session.transcript.entries[-limit:]
    fmt = str(getattr(args, "fmt", "") or "")
    if fmt or context.options.fmt in {"json", "markdown"}:
        active = fmt or ("markdown" if context.options.fmt == "markdown" else "json")
        context.printer.write(session.render(active,
                                             include_metadata=bool(getattr(args, "metadata", False))),
                              min_level="quiet")
        return EXIT_OK
    if context.options.fmt == "json":
        context.emit(session.stats())
        return EXIT_OK
    context.printer.write(session.render("text",
                                         include_metadata=bool(getattr(args, "metadata", False))),
                          min_level="quiet")
    return EXIT_OK


def _export(args: argparse.Namespace, context: CliContext) -> int:
    """Write one transcript to a file."""
    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    target = str(getattr(args, "path", "") or "")
    if not target:
        context.printer.error("error: a destination path is required")
        return EXIT_ERROR
    if context.dry_run:
        context.printer.write(f"would export {wanted} to {target}", min_level="quiet")
        return EXIT_OK
    try:
        written = store.export(wanted, target, fmt=str(getattr(args, "fmt", "") or ""))
    except Exception as exc:  # noqa: BLE001
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    context.printer.write(f"exported to {written} ({written.stat().st_size} bytes)",
                          min_level="quiet")
    return EXIT_OK


def _resume(args: argparse.Namespace, context: CliContext) -> int:
    """Continue a session by handing its tail to the chat loop."""
    from .run import cmd_chat

    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    session = store.open(wanted) if wanted else store.latest()
    if session is None:
        context.printer.error("error: no session to resume")
        return EXIT_ERROR
    keep = int(getattr(args, "keep", 20) or 20)
    replayed = session.last_messages(keep)
    context.printer.write(context.printer.themed(
        f"resuming {session.id} - {len(session)} turns, replaying {len(replayed)}", "info"),
        min_level="normal")
    args.resume = False
    args.session_id = session.id
    args.no_memory = getattr(args, "no_memory", False)
    args.tools = getattr(args, "tools", [])
    args.max_steps = getattr(args, "max_steps", 0)
    args.mode = getattr(args, "mode", "")
    args.persona = getattr(args, "persona", "")
    args.prompt = []
    code = cmd_chat(args, context)
    return code


def _delete(args: argparse.Namespace, context: CliContext) -> int:
    """Delete one session."""
    store = _store(context)
    wanted = str(getattr(args, "id", "") or "")
    try:
        resolved = store.resolve(wanted)
    except Exception as exc:  # noqa: BLE001
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    if context.dry_run:
        context.printer.write(f"would delete {resolved}", min_level="quiet")
        return EXIT_OK
    if not confirm(f"delete session {resolved}?", context=context,
                   assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    removed = store.delete(resolved)
    context.printer.write(f"deleted {resolved}" if removed else f"{resolved} was already gone",
                          min_level="quiet")
    return EXIT_OK if removed else EXIT_ERROR


def _prune(args: argparse.Namespace, context: CliContext) -> int:
    """Delete the oldest sessions beyond a keep count."""
    store = _store(context)
    keep = int(getattr(args, "keep", 0) or 0) or store.keep_last
    before = store.count()
    if before <= keep:
        context.printer.write(f"nothing to prune ({before} sessions, keeping {keep})",
                              min_level="quiet")
        return EXIT_OK
    if context.dry_run:
        context.printer.write(f"would prune {before - keep} sessions, keeping {keep}",
                              min_level="quiet")
        return EXIT_OK
    if not confirm(f"delete the {before - keep} oldest sessions, keeping {keep}?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    report = store.prune(keep_last=keep)
    if context.options.fmt == "json":
        context.emit(report)
        return EXIT_OK
    context.printer.write(f"removed {report['removed']} sessions, reclaimed "
                          f"{human_size(report['bytes_freed'])}", min_level="quiet")
    return EXIT_OK


def _search(args: argparse.Namespace, context: CliContext) -> int:
    """Search transcript contents."""
    store = _store(context)
    query = " ".join(getattr(args, "query", []) or [])
    matches = store.search(query, limit=int(getattr(args, "limit", 10) or 10))
    if context.options.fmt == "json":
        context.emit({"query": query, "count": len(matches), "matches": matches})
        return EXIT_OK
    if not matches:
        context.printer.write(f"no transcript contains {query!r}", min_level="quiet")
        return EXIT_OK
    for match in matches:
        context.printer.write(context.printer.themed(
            f"{match['session_id'][:12]} #{match['index']} {match['role']}", "heading"),
            min_level="quiet")
        context.printer.write(f"  {match['snippet']}", min_level="quiet")
        if match.get("title"):
            context.printer.write(context.printer.themed(f"  {match['title']}", "muted"),
                                  min_level="normal")
    return EXIT_OK


def _stats(args: argparse.Namespace, context: CliContext) -> int:
    """Show store statistics."""
    store = _store(context)
    payload = store.stats()
    payload["human_bytes"] = human_size(int(payload.get("bytes", 0) or 0))
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    context.printer.write(render_key_value([(str(key), value) for key, value in payload.items()]),
                          min_level="quiet")
    return EXIT_OK


def _clear(args: argparse.Namespace, context: CliContext) -> int:
    """Delete sessions, optionally only the idle ones."""
    store = _store(context)
    days = float(getattr(args, "older_than_days", 0.0) or 0.0)
    before = store.count()
    if context.dry_run:
        report = cleanup(store.directory, older_than_seconds=days * 86400.0, dry_run=True)
        context.printer.write(f"would remove {report['removed']} sessions "
                              f"({report['human_freed']})", min_level="quiet")
        return EXIT_OK
    prompt = (f"delete sessions idle for more than {days} days?" if days
              else f"delete all {before} sessions? this cannot be undone")
    if not confirm(prompt, context=context, default=False, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    if days:
        report = cleanup(store.directory, older_than_seconds=days * 86400.0)
    else:
        removed = store.clear()
        report = {"removed": removed, "kept": 0, "bytes_freed": 0, "human_freed": "0B"}
    context.printer.write(f"removed {report['removed']} sessions "
                          f"({report.get('human_freed', human_size(report.get('bytes_freed', 0)))})",
                          min_level="quiet")
    return EXIT_OK


__all__ = ["cmd_sessions", "register"]
