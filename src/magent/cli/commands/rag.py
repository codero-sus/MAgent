"""The ``rag`` command: index files and ask questions grounded in them.

Retrieval-augmented answers are only trustworthy when you can see what they were
built from, so every action here surfaces the evidence: ``search`` shows scores
and the file-and-line each chunk came from, and ``ask`` returns the answer with
its citations plus a verification of whether those citations actually appear in
the text.

An index can be saved and reloaded, which matters because embedding a large
corpus is the slow part - ``ingest --save`` once, then ``--index`` on every
later question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...rag import Retriever, cited_context
from ...utils.tables import render_key_value, render_markdown_table, render_table
from ..args import CliContext, EXIT_ERROR, EXIT_OK


def register(subparsers: Any) -> None:
    """Add the ``rag`` subcommand and its actions."""
    parser = subparsers.add_parser("rag", help="index files and answer from them",
                                   description="Retrieval over your own documents.")
    parser.add_argument("--index", default="", metavar="PATH",
                        help="load a previously saved index")
    parser.add_argument("--mode", default="", choices=["", "hybrid", "bm25", "vector", "none"],
                        help="retrieval mode")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    ingested = action.add_parser("ingest", help="index files or directories")
    ingested.add_argument("paths", nargs="+", help="files or directories to index")
    ingested.add_argument("--save", default="", metavar="PATH", help="save the index here")
    ingested.add_argument("--pattern", action="append", default=[],
                          help="glob patterns to include (repeatable)")

    found = action.add_parser("search", help="search the index")
    found.add_argument("query", nargs="+", help="text to look for")
    found.add_argument("--limit", type=int, default=5, help="maximum chunks")
    found.add_argument("--min-score", type=float, default=0.0, help="score floor")
    found.add_argument("--source", default="", help="restrict to one source path")
    found.add_argument("--no-rerank", action="store_true", help="skip the reranking pipeline")
    found.add_argument("--full", action="store_true", help="print whole chunk text")

    asked = action.add_parser("ask", help="answer a question from the index")
    asked.add_argument("question", nargs="+", help="the question")
    asked.add_argument("--limit", type=int, default=5, help="chunks to retrieve")
    asked.add_argument("--context-only", action="store_true",
                       help="print the retrieved context without asking a model")
    asked.add_argument("--check", action="store_true", help="verify the answer's citations")

    action.add_parser("stats", help="show index statistics")

    listed = action.add_parser("sources", help="list indexed sources")
    listed.add_argument("--limit", type=int, default=50, help="maximum sources")

    removed = action.add_parser("remove", help="drop one source from the index")
    removed.add_argument("source", help="source path as indexed")

    saved = action.add_parser("save", help="save the current index to a file")
    saved.add_argument("path", help="destination file")

    action.add_parser("clear", help="empty the index")
    parser.set_defaults(handler=cmd_rag, action_default="stats")


def _retriever(context: CliContext, args: argparse.Namespace) -> Retriever:
    """Load a saved index when asked for, otherwise build an empty retriever."""
    path = str(getattr(args, "index", "") or "")
    if path:
        target = Path(path).expanduser()
        if not target.exists():
            raise FileNotFoundError(f"no such index: {target}")
        retriever = Retriever.load(target)
        context.printer.debug(f"loaded index with {len(retriever)} chunks from {target}")
        return retriever
    mode = str(getattr(args, "mode", "") or "") or context.config.memory.retrieval_mode
    return Retriever(name="cli", mode=mode)


def cmd_rag(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``rag`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "stats")
    try:
        retriever = _retriever(context, args)
    except FileNotFoundError as exc:
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    handlers: dict[str, Any] = {
        "ingest": _ingest, "search": _search, "ask": _ask, "stats": _stats,
        "sources": _sources, "remove": _remove, "save": _save, "clear": _clear,
    }
    return handlers.get(action, _stats)(args, context, retriever)


def _ingest(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Index files and directories, optionally saving the result."""
    paths = [Path(item).expanduser() for item in (getattr(args, "paths", []) or [])]
    missing = [str(item) for item in paths if not item.exists()]
    if missing:
        context.printer.error(f"error: no such path(s): {', '.join(missing)}")
        return EXIT_ERROR
    patterns = list(getattr(args, "pattern", []) or []) or None
    report = retriever.ingest(paths, patterns=patterns) if patterns else retriever.ingest(paths)
    payload = {
        "indexed": len(retriever),
        "files": getattr(report, "files", 0),
        "chunks": getattr(report, "chunks", 0),
        "bytes": getattr(report, "bytes", 0),
        "embedded": getattr(report, "embedded", 0),
        "sources": len(getattr(report, "sources", {}) or {}),
        "skipped": list(getattr(report, "skipped", []) or [])[:10],
        "errors": list(getattr(report, "errors", []) or [])[:10],
        "seconds": round(float(getattr(report, "seconds", 0.0) or 0.0), 3),
    }
    saved_to = ""
    target = str(getattr(args, "save", "") or "")
    if target:
        if context.dry_run:
            payload["would_save"] = target
        else:
            saved_to = str(retriever.save(target))
            payload["saved"] = saved_to
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    printer = context.printer
    printer.write(printer.themed(
        f"indexed {payload['files']} file(s) into {payload['chunks']} chunks "
        f"({payload['indexed']} total) in {payload['seconds']}s", "success"),
        min_level="quiet")
    if saved_to:
        printer.write(f"saved index to {saved_to}", min_level="quiet")
    if payload["skipped"]:
        printer.write(printer.themed(
            f"skipped {len(payload['skipped'])}: {', '.join(payload['skipped'][:5])}", "muted"),
            min_level="normal")
    if payload["errors"]:
        for error in payload["errors"]:
            printer.warn(str(error))
    return EXIT_OK


def _search(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Search the index and show scores and provenance."""
    if not len(retriever):
        context.printer.warn("the index is empty; run 'magent rag ingest <paths>' first")
        return EXIT_OK
    query = " ".join(getattr(args, "query", []) or [])
    sources = [str(getattr(args, "source", ""))] if getattr(args, "source", "") else None
    outcome = retriever.retrieve(
        query,
        limit=int(getattr(args, "limit", 5) or 5),
        min_score=float(getattr(args, "min_score", 0.0) or 0.0),
        sources=sources,
        rerank_results=not getattr(args, "no_rerank", False),
    )
    chunks = list(getattr(outcome, "chunks", []) or [])
    citations = list(getattr(outcome.citations, "citations", []) or []) \
        if getattr(outcome, "citations", None) else []
    if context.options.fmt == "json":
        context.emit({
            "query": query,
            "mode": getattr(outcome, "mode", ""),
            "count": len(chunks),
            "seconds": round(float(getattr(outcome, "seconds", 0.0) or 0.0), 4),
            "reranked": bool(getattr(outcome, "reranked", False)),
            "citations": [item.to_dict() if hasattr(item, "to_dict") else str(item)
                          for item in citations],
            "results": [chunk.to_dict() if hasattr(chunk, "to_dict") else
                        {"source": chunk.source, "score": chunk.score, "text": chunk.text}
                        for chunk in chunks],
        })
        return EXIT_OK
    if not chunks:
        context.printer.write(f"nothing matched {query!r}", min_level="quiet")
        return EXIT_OK
    printer = context.printer
    if getattr(args, "full", False):
        for position, chunk in enumerate(chunks, start=1):
            printer.write(printer.themed(
                f"[{position}] {chunk.source}:{chunk.start_line}-{chunk.end_line} "
                f"score {chunk.score:.3f}", "heading"), min_level="quiet")
            printer.write(chunk.text.rstrip(), min_level="quiet")
            printer.blank(min_level="quiet")
        return EXIT_OK
    rows = [[position, f"{chunk.score:.3f}", _short(chunk.source, 34),
             f"{chunk.start_line}-{chunk.end_line}",
             _short(chunk.heading or " ".join(chunk.text.split()), 44)]
            for position, chunk in enumerate(chunks, start=1)]
    table = render_markdown_table(rows, headers=["#", "score", "source", "lines", "excerpt"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["#", "score", "source", "lines", "excerpt"],
                          alignments=["right", "right", "left", "left", "left"],
                          unicode_borders=context.options.unicode)
    printer.write(table, min_level="quiet")
    mode = str(getattr(outcome, "mode", "") or "")
    printer.write(printer.themed(
        f"{len(chunks)} chunk(s), mode {mode or 'default'}", "muted"), min_level="normal")
    return EXIT_OK


def _ask(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Answer a question from the index, with citations."""
    if not len(retriever):
        context.printer.warn("the index is empty; run 'magent rag ingest <paths>' first")
        return EXIT_OK
    question = " ".join(getattr(args, "question", []) or [])
    limit = int(getattr(args, "limit", 5) or 5)
    if getattr(args, "context_only", False):
        outcome = retriever.retrieve(question, limit=limit)
        text = outcome.context or cited_context(list(outcome.chunks))
        references = list(getattr(outcome.citations, "citations", []) or []) \
            if getattr(outcome, "citations", None) else []
        if context.options.fmt == "json":
            context.emit({"question": question, "context": text, "chunks": len(outcome.chunks),
                          "citations": [item.to_dict() for item in references]})
            return EXIT_OK
        context.printer.write(text or "(no context retrieved)", min_level="quiet")
        return EXIT_OK

    # --model is a global flag, so it arrives on the shared namespace.
    payload = retriever.grounded_answer(question, backend=context.backend,
                                        model=str(getattr(args, "model", "") or ""),
                                        limit=limit)
    if payload.get("error"):
        context.printer.warn(str(payload["error"]))
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    printer = context.printer
    printer.write(str(payload.get("answer", "")).rstrip() or "(no answer)", min_level="quiet")
    references = str(payload.get("references", "") or "")
    if references:
        printer.blank(min_level="normal")
        printer.write(printer.themed("sources", "heading"), min_level="normal")
        for line in references.splitlines():
            printer.write(printer.themed(f"  {line.strip()}", "muted"), min_level="normal")
    sources = list(payload.get("sources", []) or [])
    if sources and not references:
        printer.write(printer.themed(f"sources: {', '.join(str(item) for item in sources[:6])}",
                                     "muted"), min_level="normal")
    if getattr(args, "check", False):
        check = payload.get("citation_check") or {}
        coverage = float(check.get("coverage", 0.0) or 0.0)
        dangling = list(check.get("dangling", []) or [])
        printer.write(printer.themed(
            f"citation coverage {coverage:.0%}" + (f", dangling: {dangling}" if dangling else ""),
            "success" if not dangling else "warning"), min_level="normal")
    printer.write(printer.themed(
        f"{payload.get('chunks', 0)} chunk(s) in "
        f"{payload.get('retrieval_seconds', 0.0)}s", "muted"), min_level="normal")
    return EXIT_OK


def _stats(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Show index statistics."""
    payload = retriever.stats()
    payload["chunks"] = len(retriever)
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    context.printer.write(render_key_value([(str(key), value) for key, value in payload.items()]),
                          min_level="quiet")
    return EXIT_OK


def _sources(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """List the sources currently indexed."""
    limit = int(getattr(args, "limit", 50) or 50)
    known = retriever.index.sources()
    items = sorted(known.items() if isinstance(known, dict) else ((str(item), 0) for item in known))
    rows = [[_short(source, 60), count] for source, count in items[:limit]]
    if context.options.fmt == "json":
        context.emit({"count": len(items), "sources": dict(items)})
        return EXIT_OK
    if not rows:
        context.printer.write("the index is empty", min_level="quiet")
        return EXIT_OK
    table = render_markdown_table(rows, headers=["source", "chunks"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["source", "chunks"], alignments=["left", "right"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(items)} sources, {len(retriever)} chunks", "muted"), min_level="normal")
    return EXIT_OK


def _remove(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Drop one source from the index."""
    source = str(getattr(args, "source", "") or "")
    if context.dry_run:
        context.printer.write(f"would remove {source}", min_level="quiet")
        return EXIT_OK
    removed = retriever.remove(source)
    if context.options.fmt == "json":
        context.emit({"source": source, "removed_chunks": removed, "remaining": len(retriever)})
        return EXIT_OK
    if removed:
        context.printer.write(f"removed {removed} chunk(s) from {source}", min_level="quiet")
    else:
        context.printer.warn(f"{source} was not in the index")
    return EXIT_OK


def _save(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Save the index to a file."""
    target = str(getattr(args, "path", "") or "")
    if not target:
        context.printer.error("error: a destination path is required")
        return EXIT_ERROR
    if not len(retriever):
        context.printer.warn("the index is empty; nothing to save")
        return EXIT_OK
    if context.dry_run:
        context.printer.write(f"would save {len(retriever)} chunks to {target}", min_level="quiet")
        return EXIT_OK
    written = retriever.save(target)
    context.printer.write(f"saved {len(retriever)} chunks to {written} "
                          f"({written.stat().st_size} bytes)", min_level="quiet")
    return EXIT_OK


def _clear(args: argparse.Namespace, context: CliContext, retriever: Retriever) -> int:
    """Empty the in-memory index."""
    before = len(retriever)
    if context.dry_run:
        context.printer.write(f"would clear {before} chunks", min_level="quiet")
        return EXIT_OK
    retriever.clear()
    context.printer.write(f"cleared {before} chunks", min_level="quiet")
    return EXIT_OK


def _short(text: str, limit: int) -> str:
    """Clip text to a width, keeping the tail for paths."""
    flat = str(text or "")
    if len(flat) <= limit:
        return flat
    return "..." + flat[-(limit - 3):]


__all__ = ["cmd_rag", "register"]
