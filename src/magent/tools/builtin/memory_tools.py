"""Memory tools: durable notes the agent can keep across runs.

The memory layer is a separate package (:mod:`magent.memory`); these tools are
the agent-facing surface on top of it. They let a run record a fact, search
what it already knows, and list or forget entries. Entries are stored as
newline-delimited JSON with an embedding cache, so search can be lexical or
semantic depending on whether a backend is configured.
"""

from __future__ import annotations

import json
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.json_utils import loads_lenient
from ...utils.text import ellipsize


def _store(context: ToolContext) -> Any:
    """Resolve the memory store from the tool context."""
    store = context.memory
    if store is None:
        raise ToolExecutionError(
            "No memory store is attached to this run; enable it with memory.enabled=true"
        )
    return store


@tool(category="memory", risk="write", filesystem=True, tags=["memory", "remember", "core"])
def remember(content: str, kind: str = "note", tags: str = "", importance: float = 0.5,
             source: str = "", metadata: str = "", context: ToolContext | None = None) -> dict[str, Any]:
    """Store a durable memory entry.

    Args:
        content: the text to remember
        kind: entry type, e.g. note, fact, decision, preference, task
        tags: comma separated tags for later filtering
        importance: 0.0 to 1.0, higher ranks the entry first in recall
        source: where the information came from
        metadata: JSON object of extra fields
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    if not content.strip():
        raise ToolExecutionError("Nothing to remember: content is empty")
    extra = loads_lenient(metadata) if metadata else {}
    if extra is not None and not isinstance(extra, dict):
        raise ToolExecutionError("metadata must be a JSON object")
    tag_list = [item.strip() for item in tags.replace(";", ",").split(",") if item.strip()]
    entry = store.add(
        content,
        kind=kind,
        tags=tag_list,
        importance=max(0.0, min(1.0, float(importance))),
        source=source,
        metadata=extra or {},
    )
    return {
        "ok": True,
        "id": getattr(entry, "id", None),
        "kind": getattr(entry, "kind", kind),
        "tags": tag_list,
        "importance": getattr(entry, "importance", importance),
        "chars": len(content),
        "stored": True,
    }


@tool(category="memory", risk="read", filesystem=True, tags=["memory", "recall", "core"])
def recall(query: str = "", limit: int = 8, kind: str = "", tags: str = "",
           semantic: bool = True, min_score: float = 0.0,
           context: ToolContext | None = None) -> dict[str, Any]:
    """Search stored memories by text, tags, or semantic similarity.

    Args:
        query: search text, empty to list the most recent entries
        limit: maximum entries to return
        kind: restrict to one entry kind
        tags: comma separated tags; entries must carry at least one
        semantic: use embeddings when a backend is available
        min_score: drop matches scoring below this
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    tag_list = [item.strip() for item in tags.replace(";", ",").split(",") if item.strip()]
    results = store.search(
        query,
        limit=max(1, min(limit, 100)),
        kind=kind or None,
        tags=tag_list or None,
        semantic=semantic and bool(ctx.backend is not None),
        min_score=min_score,
    )
    entries: list[dict[str, Any]] = []
    for item in results:
        entries.append({
            "id": getattr(item, "id", None),
            "content": getattr(item, "content", str(item)),
            "kind": getattr(item, "kind", ""),
            "tags": list(getattr(item, "tags", []) or []),
            "importance": getattr(item, "importance", 0.0),
            "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
            "created_at": getattr(item, "created_at", ""),
            "source": getattr(item, "source", ""),
        })
    text = "\n".join(
        f"- [{entry['kind']}] {ellipsize(entry['content'], 200)}" for entry in entries
    )
    return {
        "query": query,
        "count": len(entries),
        "entries": entries,
        "text": text or "(no memories matched)",
        "semantic_used": semantic and bool(ctx.backend is not None),
    }


@tool(category="memory", risk="read", filesystem=True, tags=["memory", "list"])
def list_memories(limit: int = 20, kind: str = "", order: str = "recent",
                  context: ToolContext | None = None) -> dict[str, Any]:
    """List stored memories without running a search.

    Args:
        limit: maximum entries to return
        kind: restrict to one entry kind
        order: recent, oldest, or importance
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    if order not in {"recent", "oldest", "importance"}:
        raise ToolExecutionError("order must be recent, oldest, or importance")
    entries = store.list_entries(limit=max(1, min(limit, 500)), kind=kind or None, order=order)
    records = [
        {
            "id": getattr(entry, "id", index),
            "content": ellipsize(getattr(entry, "content", str(entry)), 300),
            "kind": getattr(entry, "kind", ""),
            "importance": getattr(entry, "importance", 0.0),
            "created_at": getattr(entry, "created_at", ""),
            "tags": list(getattr(entry, "tags", []) or []),
        }
        for index, entry in enumerate(entries, start=1)
    ]
    stats = store.stats() if hasattr(store, "stats") else {}
    return {"count": len(records), "order": order, "entries": records, "stats": stats}


@tool(category="memory", risk="destructive", destructive=True, filesystem=True,
      requires_approval=True, tags=["memory", "forget"])
def forget(entry_id: str = "", query: str = "", kind: str = "", purge_all: bool = False,
           min_score: float = 0.25, limit: int = 100,
           context: ToolContext | None = None) -> dict[str, Any]:
    """Delete one or more stored memories.

    Args:
        entry_id: exact entry identifier to delete
        query: delete entries matching this search
        kind: restrict a query-based delete to one kind
        purge_all: delete every entry
        min_score: only delete query matches scoring at least this high
        limit: maximum entries a query-based delete may remove
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    if ctx.dry_run:
        return {"dry_run": True, "would_delete": entry_id or query or ("all" if purge_all else "nothing")}
    if purge_all:
        removed = store.clear()
        return {"deleted": removed, "mode": "purge_all"}
    if entry_id:
        ok = store.delete(entry_id)
        if not ok:
            raise ToolExecutionError(f"No memory with id {entry_id!r}")
        return {"deleted": 1, "mode": "by_id", "id": entry_id}
    if query:
        removed = store.delete_matching(query, kind=kind or None, min_score=min_score, limit=limit)
        return {"deleted": removed, "mode": "by_query", "query": query, "min_score": min_score}
    raise ToolExecutionError("Provide entry_id, query, or purge_all")


@tool(category="memory", risk="read", filesystem=True, tags=["memory", "export"])
def export_memories(path: str = "", format: str = "jsonl", kind: str = "",
                    context: ToolContext | None = None) -> dict[str, Any]:
    """Export every memory as JSONL or JSON, optionally to a file.

    Args:
        path: destination file inside the workspace, empty to return the text
        format: jsonl or json
        kind: restrict the export to one entry kind
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    entries = store.list_entries(limit=100000, kind=kind or None, order="recent")
    records = [entry.to_dict() if hasattr(entry, "to_dict") else {"content": str(entry)} for entry in entries]
    if format == "json":
        text = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    elif format == "jsonl":
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    else:
        raise ToolExecutionError("format must be json or jsonl")
    payload: dict[str, Any] = {"count": len(records), "format": format, "chars": len(text)}
    if path:
        from ...utils.files import atomic_write_text, ensure_dir

        target = ctx.resolve_path(path)
        if ctx.sandbox is not None:
            ctx.sandbox.check_write(target)
        if ctx.dry_run:
            payload.update({"dry_run": True, "path": str(target)})
            return payload
        ensure_dir(target.parent)
        atomic_write_text(target, text)
        payload["path"] = str(target)
        payload["written"] = True
    else:
        payload["text"] = text[: ctx.output_limit]
    return payload


@tool(category="memory", risk="safe", filesystem=True, tags=["memory", "compact"])
def compact_memories(kind: str = "", keep: int = 50, context: ToolContext | None = None) -> dict[str, Any]:
    """Merge near-duplicate memories and drop the least important ones.

    Args:
        kind: restrict compaction to one entry kind
        keep: maximum entries to retain per kind
    """
    ctx = context or ToolContext()
    store = _store(ctx)
    if not hasattr(store, "compact"):
        raise ToolExecutionError("The configured memory store does not support compaction")
    report = store.compact(kind=kind or None, keep=max(1, keep))
    return {"report": report, "kind": kind or "all", "keep": keep}


TOOLS: list[Any] = [remember, recall, list_memories, forget, export_memories, compact_memories]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every memory tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "compact_memories",
    "export_memories",
    "forget",
    "list_memories",
    "load",
    "recall",
    "remember",
]
