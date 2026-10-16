"""Commands over stored knowledge: ``memory``, ``rag``, ``sessions``.

These are the commands that make MAgent's state inspectable.  Anything an agent
remembered, indexed or recorded can be listed, searched, exported and deleted
from the CLI without touching files by hand - which is also how you verify that
memory and retrieval are actually working rather than quietly empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..commands import Command
from ...errors import MAgentError
from ...utils.text import ellipsize


# -- memory ------------------------------------------------------------------


def memory_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent memory`` - inspect, search and edit long-term memory."""
    store = ctx.memory
    if store is None:
        ctx.notice("memory is disabled in the configuration", level="warning")
        return 1
    action = getattr(args, "action", "") or "list"
    if action == "add":
        if not args.text:
            ctx.error("nothing to remember")
            return 2
        entry = store.add(args.text, kind=args.kind or "fact", tags=list(args.tag or []))
        store.save()
        ctx.notice(f"stored {entry.id}", level="success")
        if ctx.config.output.format != "text":
            ctx.emit(entry.to_dict(), fmt=ctx.config.output.format)
        return 0
    if action == "forget":
        if args.id:
            removed = store.delete(args.id)
            store.save()
            ctx.notice(f"{'deleted' if removed else 'not found'}: {args.id}",
                       level="success" if removed else "warning")
            return 0 if removed else 1
        count = store.delete_matching(args.text or "", kind=args.kind or "",
                                      min_score=args.min_score or 0.0, limit=args.limit or 0)
        store.save()
        ctx.notice(f"deleted {count} entries", level="success")
        return 0 if count else 1
    if action == "stats":
        stats = store.stats()
        if ctx.config.output.format != "text":
            ctx.emit(stats, fmt=ctx.config.output.format)
            return 0
        ctx.key_values([(key, value) for key, value in stats.items()
                        if not isinstance(value, (dict, list))], title="memory")
        kinds = stats.get("kinds") or {}
        if kinds:
            ctx.print("")
            ctx.table([(kind, str(count)) for kind, count in sorted(kinds.items())],
                      ["kind", "entries"])
        return 0
    if action == "compact":
        merged = store.compact(threshold=args.threshold or 0.95)
        store.save()
        ctx.notice(f"merged {merged} near-duplicate entries", level="success")
        return 0
    entries = (store.search(args.text, limit=args.limit or 20) if args.text
               else (store.recent(args.limit or 20) if action == "recent"
                     else store.list_entries(limit=args.limit or 20)))
    if ctx.config.output.format != "text":
        ctx.emit([entry.to_dict() for entry in entries], fmt=ctx.config.output.format)
        return 0
    if not entries:
        ctx.notice("no entries")
        return 0
    rows = [(entry.id[:12], entry.kind, f"{entry.score:.2f}" if entry.score else "-",
             ellipsize(entry.content, 62)) for entry in entries]
    ctx.table(rows, ["id", "kind", "score", "content"],
              title=f"{len(entries)} entries in {store.path}")
    return 0


# -- retrieval ---------------------------------------------------------------


def rag_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent rag`` - index files and query the retrieval corpus."""
    retriever = ctx.retriever
    action = getattr(args, "action", "") or "status"
    if action == "ingest":
        raw_paths = list(args.paths or []) or [str(ctx.workspace)]
        paths: list[Path] = []
        for item in raw_paths:
            candidate = Path(item).expanduser()
            if not candidate.is_absolute() and not args.no_root:
                candidate = ctx.workspace / candidate
            paths.append(candidate)
        report = retriever.ingest(paths)
        path = ctx.save_index()
        if ctx.config.output.format != "text":
            ctx.emit({"report": report.to_dict(), "index": str(path or "")},
                     fmt=ctx.config.output.format)
            return 0
        ctx.notice(f"indexed {report.chunks} chunks from {report.files} files "
                   f"in {report.seconds * 1000:.0f} ms", level="success")
        if report.skipped:
            ctx.notice(f"skipped {len(report.skipped)} files", level="muted")
        if path:
            ctx.notice(f"index saved to {path}", level="muted")
        return 0
    if action == "search":
        if not args.query:
            ctx.error("search needs a query")
            return 2
        result = retriever.retrieve(args.query, limit=args.limit or 5)
        if ctx.config.output.format != "text":
            ctx.emit({"query": args.query, "chunks": [chunk.to_dict() for chunk in result.chunks]},
                     fmt=ctx.config.output.format)
            return 0
        if not result.chunks:
            ctx.notice("no matches; run `magent rag ingest` first", level="warning")
            return 1
        rows = [(f"{index}", f"{chunk.source}:{chunk.start_line}-{chunk.end_line}",
                 f"{chunk.score:.2f}", ellipsize(chunk.text.replace('\n', ' '), 60))
                for index, chunk in enumerate(result.chunks, start=1)]
        ctx.table(rows, ["#", "location", "score", "text"])
        return 0
    if action == "context":
        if not args.query:
            ctx.error("context needs a query")
            return 2
        ctx.write(retriever.context(args.query, limit=args.limit or 5) + "\n")
        return 0
    stats = retriever.stats()
    if ctx.config.output.format != "text":
        ctx.emit(stats, fmt=ctx.config.output.format)
        return 0
    ctx.key_values([(key, value) for key, value in stats.items()
                    if not isinstance(value, (dict, list))], title="retrieval index")
    return 0


# -- sessions ----------------------------------------------------------------


def sessions_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent sessions`` - list, show, export, search and delete sessions."""
    from ...session import render_transcript

    sessions = ctx.sessions
    action = getattr(args, "action", "") or "list"
    if action == "list":
        infos = sessions.list(limit=args.limit or 20, tag=args.tag or "", query=args.query or "")
        if ctx.config.output.format != "text":
            ctx.emit([info.to_dict() for info in infos], fmt=ctx.config.output.format)
            return 0
        if not infos:
            ctx.notice("no stored sessions")
            return 0
        rows = [(info.age, info.id[:18], str(info.messages), str(info.tokens),
                 ellipsize(info.title or info.preview, 44)) for info in infos]
        ctx.table(rows, ["age", "id", "msgs", "tokens", "title"],
                  title=f"{len(infos)} sessions in {sessions.store.directory}")
        return 0
    if action == "show":
        record = sessions.load(args.id) if args.id else getattr(sessions, "current", None)
        if record is None:
            ctx.error(f"unknown session {args.id!r}")
            return 1
        if ctx.config.output.format != "text":
            ctx.emit(record.to_dict(), fmt=ctx.config.output.format)
            return 0
        ctx.write(render_transcript(record, fmt=args.transcript_format or "markdown",
                                    redact=ctx.config.session.redact_secrets))
        return 0
    if action == "export":
        if not args.id:
            ctx.error("export needs a session id")
            return 2
        target = Path(args.output or f"{args.id}.md").expanduser()
        path = sessions.store.export(args.id, target, fmt=args.transcript_format or "markdown")
        ctx.notice(f"wrote {path}", level="success")
        return 0
    if action == "delete":
        if not args.id:
            ctx.error("delete needs a session id")
            return 2
        removed = sessions.delete(args.id)
        ctx.notice(f"{'deleted' if removed else 'not found'}: {args.id}",
                   level="success" if removed else "warning")
        return 0 if removed else 1
    if action == "search":
        if not args.query:
            ctx.error("search needs a query")
            return 2
        hits = sessions.search(args.query, limit=args.limit or 10)
        if ctx.config.output.format != "text":
            ctx.emit([{"session": info.to_dict(), "matches": lines} for info, lines in hits],
                     fmt=ctx.config.output.format)
            return 0
        for info, lines in hits:
            ctx.print(f"{info.id[:18]}  {ellipsize(info.title, 50)}")
            for line in lines:
                ctx.print(f"    {line}")
        if not hits:
            ctx.notice("no matches")
        return 0
    if action == "prune":
        removed = sessions.store.prune(args.keep or 10)
        ctx.notice(f"removed {removed} old sessions", level="success")
        return 0
    stats = sessions.store.stats()
    if ctx.config.output.format != "text":
        ctx.emit(stats, fmt=ctx.config.output.format)
        return 0
    ctx.key_values([(key, value) for key, value in stats.items()
                    if not isinstance(value, (dict, list))], title="sessions")
    return 0


# -- observation -------------------------------------------------------------


def trace_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent trace`` - show spans for the last run, or from a trace file."""
    if args.path:
        path = Path(args.path).expanduser()
        if not path.is_file():
            ctx.error(f"no trace file at {path}")
            return 1
        lines = path.read_text(encoding="utf-8").splitlines()
        if ctx.config.output.format != "text":
            ctx.emit([line for line in lines if line.strip()], fmt="jsonl")
            return 0
        for line in lines[-args.limit or -50:]:
            if line.strip():
                ctx.print(ellipsize(line, 160))
        return 0
    observability = ctx.observability
    trace = observability.tracer.last_trace()
    if trace is None:
        ctx.notice("no traces recorded yet; run something first", level="warning")
        return 1
    if ctx.config.output.format != "text":
        ctx.emit(trace.to_dict(), fmt=ctx.config.output.format)
        return 0
    ctx.print(trace.flame(width=args.graph_width or ctx.config.output.width or 60))
    ctx.print("")
    ctx.key_values([("trace", trace.trace_id), ("spans", len(trace.spans())),
                    ("errors", len(trace.errors())),
                    ("duration", f"{trace.duration_ms:.1f} ms")])
    return 0


def _configure_memory(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent memory``."""
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "search", "recent", "add", "forget", "compact", "stats"],
                        help="what to do")
    parser.add_argument("text", nargs="?", default="", help="content or query")
    parser.add_argument("--id", default="", help="entry id to forget")
    parser.add_argument("--kind", default="", help="entry kind or filter")
    parser.add_argument("--tag", action="append", default=None, help="tag (repeatable)")
    parser.add_argument("--limit", type=int, default=20, help="maximum entries")
    parser.add_argument("--min-score", dest="min_score", type=float, default=0.0,
                        help="forget only entries above this score")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="similarity threshold for compact")


def _configure_rag(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent rag``."""
    parser.add_argument("action", nargs="?", default="status",
                        choices=["status", "ingest", "search", "context"], help="what to do")
    parser.add_argument("paths", nargs="*", default=[], help="paths to ingest")
    parser.add_argument("--query", default="", help="search text")
    parser.add_argument("--limit", type=int, default=5, help="chunks to return")
    parser.add_argument("--no-root", dest="no_root", action="store_true",
                        help="treat paths as given instead of relative to the workspace")


def _configure_sessions(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent sessions``."""
    parser.add_argument("action", nargs="?", default="list",
                        choices=["list", "show", "export", "delete", "search", "prune", "stats"],
                        help="what to do")
    parser.add_argument("id", nargs="?", default="", help="session id")
    parser.add_argument("--query", default="", help="search text")
    parser.add_argument("--tag", default="", help="filter by tag")
    parser.add_argument("--limit", type=int, default=20, help="maximum sessions")
    parser.add_argument("--transcript-format", dest="transcript_format", default="markdown",
                        help="transcript format: markdown|text|json|html")
    parser.add_argument("--output", default="", help="output path for export")
    parser.add_argument("--keep", type=int, default=10, help="sessions kept by prune")


def _configure_trace(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent trace``."""
    parser.add_argument("--path", default="", help="read traces from this JSONL file")
    parser.add_argument("--limit", type=int, default=50, help="lines to show")
    parser.add_argument("--graph-width", dest="graph_width", type=int, default=0,
                        help="flame graph width (0 uses the output width)")


COMMANDS: tuple[Command, ...] = (
    Command(
        name="memory",
        help="Inspect, search and edit long-term memory",
        run=memory_command,
        group="knowledge",
        aliases=("m",),
        configure=_configure_memory,
        description=(
            "Memory is a hybrid BM25/vector store. `list`, `search`, `recent`, `add`, "
            "`forget`, `compact` and `stats` cover everything the agent can do to it."
        ),
        examples=("magent memory list", "magent memory search 'deployment steps'",
                  "magent memory add 'the API key lives in ~/.config/magent/secrets.toml'"),
    ),
    Command(
        name="rag",
        help="Index files and query the retrieval corpus",
        run=rag_command,
        group="knowledge",
        configure=_configure_rag,
        description=(
            "Ingest documents, then search them. Chunks keep their source and line "
            "range, so an answer can always be traced back to the file it came from."
        ),
        examples=("magent rag ingest docs/", "magent rag search 'how do guardrails work'",
                  "magent rag status"),
    ),
    Command(
        name="sessions",
        help="List, show, export and search sessions",
        run=sessions_command,
        group="knowledge",
        configure=_configure_sessions,
        examples=("magent sessions list", "magent sessions show --id ses_1234",
                  "magent sessions search 'flaky test'"),
    ),
    Command(
        name="trace",
        help="Show the spans of the last run",
        run=trace_command,
        group="system",
        configure=_configure_trace,
        examples=("magent trace", "magent trace --path ~/.cache/magent/traces.jsonl"),
    ),
)


__all__ = [
    "COMMANDS",
    "memory_command",
    "rag_command",
    "sessions_command",
    "trace_command",
]
