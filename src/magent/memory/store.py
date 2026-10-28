"""Durable memory: a JSONL-backed store with hybrid retrieval.

The store is append-first. Adding an entry writes one JSON line and updates an
in-memory index, so a crash loses at most the entry being written. Deletions and
compaction rewrite the file atomically through
:func:`~magent.utils.files.atomic_write_text`, which means the store is never
left half-written.

Retrieval blends BM25, vector similarity, importance, and recency
(:mod:`magent.memory.ranking`), and falls back to lexical-only when no embedder
is available - so memory works with or without a running model.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ..errors import MemoryError_, StoreError
from ..logging_ import get_logger
from ..types import new_id
from ..utils.files import atomic_write_text, ensure_dir, human_size
from ..utils.text import collapse_whitespace, ellipsize, similarity
from .embeddings import Embedder, build_embedder
from .entry import DEFAULT_KIND, MemoryEntry
from .ranking import DecayPolicy, RankingWeights, rank_entries
from .vector import VectorIndex

LOGGER = get_logger("memory")
"""Logger for memory operations."""

ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"https?://[^\s)\"'<>]+"),
    "path": re.compile(
        r"(?:~|\.{1,2}|[A-Za-z]:)?/(?:[\w.-]+/)*[\w.-]+|(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,8}"
    ),
    "identifier": re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b"),
    "number": re.compile(r"\b\d+(?:\.\d+)?\b"),
}
"""Regexes used to extract lightweight entities from stored content."""

CODE_LIKE = re.compile(r"[_0-9.]|[a-z][A-Z]")
"""Test for an identifier that looks like code rather than an English word."""

MAX_ENTITIES_PER_ENTRY = 20
"""Cap on how many entities one entry records."""


class MemoryStore:
    """A thread-safe, file-backed memory with hybrid retrieval."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        embedder: Embedder | None = None,
        backend: Any = None,
        config: Any = None,
        autosave: bool = True,
        dedupe: bool = True,
        entity_tracking: bool = True,
        weights: RankingWeights | None = None,
        decay: DecayPolicy | None = None,
        embedding_model: str = "",
        vector_buckets: int = 0,
    ) -> None:
        """Open a store, loading any existing entries from ``path``.

        Args:
            path: JSONL file; ``None`` keeps everything in memory only
            embedder: explicit embedder, overriding automatic selection
            backend: LLM backend used to build a :class:`BackendEmbedder`
            config: a :class:`~magent.config.MemoryConfig` for defaults
            autosave: append to disk on every write
            dedupe: skip an add whose content fingerprint already exists
            entity_tracking: extract entities from content on add
            weights: retrieval signal weights
            decay: recency decay policy
            embedding_model: embedding model override for the backend
            vector_buckets: bucket count for approximate vector search
        """
        self.config = config
        memory_config = config if config is not None else None
        self.path = Path(path).expanduser() if path else None
        if self.path is None and memory_config is not None:
            configured = str(getattr(memory_config, "store_path", "") or "")
            self.path = Path(configured).expanduser() if configured else None
        if memory_config is not None:
            self.autosave = bool(autosave and getattr(memory_config, "persistence", True))
        else:
            self.autosave = bool(autosave)
        self.dedupe = bool(
            dedupe if memory_config is None else getattr(memory_config, "dedupe", dedupe)
        )
        self.entity_tracking = bool(
            entity_tracking
            if memory_config is None
            else getattr(memory_config, "entity_tracking", entity_tracking)
        )
        self.retrieval_mode = str(getattr(memory_config, "retrieval_mode", "hybrid") or "hybrid")
        self.top_k = int(getattr(memory_config, "retrieval_top_k", 5) or 5)
        self.embedder = embedder or build_embedder(
            backend,
            model=embedding_model or str(getattr(memory_config, "summary_model", "") or ""),
            dim=int(getattr(memory_config, "vector_dim", 0) or 256),
            cache=bool(getattr(memory_config, "cache_embeddings", True)),
            batch_size=int(getattr(memory_config, "embedding_batch_size", 16) or 16),
        )
        self.weights = weights or RankingWeights()
        self.decay = decay or DecayPolicy()
        self.vector_index = VectorIndex(
            dim=int(getattr(memory_config, "vector_dim", 0) or 0), buckets=vector_buckets
        )
        self._entries: dict[str, MemoryEntry] = {}
        self._order: list[str] = []
        self._fingerprints: dict[str, str] = {}
        self._lock = threading.RLock()
        self.added = 0
        self.duplicates_skipped = 0
        self.deleted = 0
        self.searches = 0
        self.search_ms = 0.0
        if self.path is not None and self.path.is_file():
            self.load()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Any, *, backend: Any = None, root: str | Path | None = None
    ) -> MemoryStore:
        """Build a store from a loaded MAgent configuration.

        Args:
            config: a full :class:`~magent.config.Config` or a ``MemoryConfig``
            backend: optional LLM backend for real embeddings
            root: directory override for the store file
        """
        memory_config = getattr(config, "memory", config)
        data_root = Path(root or getattr(config, "data_dir", "") or Path.cwd()).expanduser()
        path = str(getattr(memory_config, "store_path", "") or "")
        target = Path(path).expanduser() if path else data_root / "memory" / "memories.jsonl"
        if not getattr(memory_config, "enabled", True):
            return cls(path=None, config=memory_config, backend=backend, autosave=False)
        store = cls(path=target, config=memory_config, backend=backend)
        if not getattr(memory_config, "vector_enabled", True):
            store.retrieval_mode = "bm25"
        return store

    # -- writes --------------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        kind: str = DEFAULT_KIND,
        tags: Sequence[str] = (),
        importance: float = 0.5,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        entry_id: str = "",
        embedding: Sequence[float] | None = None,
    ) -> MemoryEntry:
        """Store one memory, embedding it when an embedder is available.

        Args:
            content: the text to remember
            kind: entry kind, e.g. fact, decision, preference
            tags: tags for filtering and a small ranking bonus
            importance: 0.0 to 1.0; higher ranks first and decays slower
            source: where the information came from
            metadata: extra fields to persist
            entry_id: reuse an identifier, e.g. when importing
            embedding: precomputed vector, skipping the embedder
        """
        text = collapse_whitespace(str(content or "")).strip()
        if not text:
            raise MemoryError_("Cannot store an empty memory")
        entry = MemoryEntry(
            id=entry_id or new_id("mem"),
            content=text,
            kind=kind,
            tags=list(tags or ()),
            importance=importance,
            source=source,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if self.dedupe:
                existing = self._fingerprints.get(entry.fingerprint)
                if existing and existing in self._entries:
                    self.duplicates_skipped += 1
                    previous = self._entries[existing]
                    previous.touch()
                    previous.importance = max(previous.importance, entry.importance)
                    return previous
            if self.entity_tracking:
                entry.metadata.setdefault("entities", extract_entities(text))
            if embedding is None:
                embedding = self._embed([text])[0]
            entry.embedding = [float(value) for value in (embedding or [])]
            self._entries[entry.id] = entry
            self._order.append(entry.id)
            self._fingerprints[entry.fingerprint] = entry.id
            if entry.embedding:
                self.vector_index.add(entry.id, entry.embedding)
            self.added += 1
        if self.autosave and self.path is not None:
            self._append(entry)
        return entry

    def add_many(
        self,
        items: Iterable[dict[str, Any] | str],
        *,
        kind: str = DEFAULT_KIND,
        importance: float = 0.5,
        source: str = "",
    ) -> list[MemoryEntry]:
        """Store several memories, embedding them in one batch."""
        payloads: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                payloads.append({"content": item})
            elif isinstance(item, dict):
                payloads.append(item)
            else:
                payloads.append({"content": str(item)})
        texts = [str(payload.get("content", "")) for payload in payloads]
        vectors = self._embed([text for text in texts if text.strip()])
        cursor = 0
        stored: list[MemoryEntry] = []
        for payload in payloads:
            text = str(payload.get("content", ""))
            vector = None
            if text.strip():
                vector = vectors[cursor] if cursor < len(vectors) else None
                cursor += 1
            stored.append(
                self.add(
                    text,
                    kind=str(payload.get("kind", kind)),
                    tags=list(payload.get("tags") or ()),
                    importance=float(payload.get("importance", importance)),
                    source=str(payload.get("source", source)),
                    metadata=dict(payload.get("metadata") or {}),
                    embedding=vector,
                )
            )
        return stored

    def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Update an existing entry and re-embed it when the content changed."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise MemoryError_(f"No memory with id {entry_id!r}")
            old_fingerprint = entry.fingerprint
            entry.update(content, importance=importance, tags=tags, metadata=metadata)
            self._fingerprints.pop(old_fingerprint, None)
            self._fingerprints[entry.fingerprint] = entry.id
            if content is not None:
                entry.embedding = self._embed([entry.content])[0]
                self.vector_index.add(entry.id, entry.embedding)
        self.save()
        return entry

    def touch(self, entry_id: str) -> bool:
        """Record an access on one entry without changing its content."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.touch()
            return True

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, degrading to empty vectors when embedding fails."""
        if not texts or self.retrieval_mode in {"bm25", "none"}:
            return [[] for _ in texts]
        try:
            return self.embedder.embed(list(texts))
        except Exception as exc:
            LOGGER.warning("embedding failed, falling back to lexical search: %s", exc)
            return [[] for _ in texts]

    # -- reads ---------------------------------------------------------------

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Return one entry by identifier."""
        with self._lock:
            return self._entries.get(entry_id)

    def search(
        self,
        query: str = "",
        *,
        limit: int | None = None,
        kind: str = "",
        tags: Sequence[str] | None = None,
        semantic: bool = True,
        min_score: float = 0.0,
        mode: str = "",
        before: float = 0.0,
        after: float = 0.0,
        touch: bool = True,
    ) -> list[MemoryEntry]:
        """Rank stored memories against a query.

        Args:
            query: search text; empty returns the best-scoring recent entries
            limit: maximum entries to return
            kind: restrict to one entry kind
            tags: entries must carry at least one of these tags
            semantic: allow the vector term when an embedder is available
            min_score: drop entries scoring below this
            mode: override the configured retrieval mode
            before: only entries created before this timestamp
            after: only entries created after this timestamp
            touch: record an access on every returned entry
        """
        started = time.perf_counter()
        active_mode = (mode or self.retrieval_mode or "hybrid").lower()
        if not semantic and active_mode in {"hybrid", "vector"}:
            active_mode = "bm25"
        wanted = [str(tag) for tag in (tags or ()) if str(tag).strip()]
        with self._lock:
            candidates = [
                entry
                for entry in (self._entries[key] for key in self._order)
                if (not kind or entry.kind == kind.lower())
                and entry.matches_tags(wanted)
                and (not before or entry.created_at <= before)
                and (not after or entry.created_at >= after)
            ]
            ranked = rank_entries(
                candidates,
                query,
                embedder=self.embedder if active_mode in {"hybrid", "vector"} else None,
                weights=self.weights,
                decay=self.decay,
                tags=wanted,
                mode=active_mode,
            )
            top = [entry for entry in ranked if entry.score >= min_score][: (limit or self.top_k)]
            if touch:
                for entry in top:
                    entry.touch()
        self.searches += 1
        self.search_ms += (time.perf_counter() - started) * 1000.0
        return top

    def search_similar(self, entry_id: str, *, limit: int = 5) -> list[MemoryEntry]:
        """Find entries closest to an existing one, for dedupe review."""
        entry = self.get(entry_id)
        if entry is None or not entry.embedding:
            return []
        neighbours = self.vector_index.search(entry.embedding, limit=limit + 1)
        results = []
        for key, score in neighbours:
            if key == entry_id:
                continue
            candidate = self._entries.get(key)
            if candidate is not None:
                candidate.score = round(score, 6)
                results.append(candidate)
        return results[:limit]

    def list_entries(
        self, *, limit: int = 50, kind: str = "", order: str = "recent", tag: str = ""
    ) -> list[MemoryEntry]:
        """List entries without running a search.

        Args:
            limit: maximum entries to return
            kind: restrict to one entry kind
            order: ``recent``, ``oldest``, ``importance``, or ``accessed``
            tag: restrict to entries carrying this tag
        """
        with self._lock:
            items = [
                entry
                for entry in (self._entries[key] for key in self._order)
                if (not kind or entry.kind == kind.lower()) and (not tag or entry.has_tag(tag))
            ]
        if order == "oldest":
            items.sort(key=lambda entry: entry.created_at)
        elif order == "importance":
            items.sort(key=lambda entry: (entry.importance, entry.created_at), reverse=True)
        elif order == "accessed":
            items.sort(key=lambda entry: entry.access_count, reverse=True)
        else:
            items.sort(key=lambda entry: entry.created_at, reverse=True)
        return items[: max(1, limit)]

    def recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Return the most recently created entries."""
        return self.list_entries(limit=limit, order="recent")

    def by_kind(self) -> dict[str, int]:
        """Count entries per kind."""
        with self._lock:
            return dict(Counter(entry.kind for entry in self._entries.values()))

    def tags(self) -> dict[str, int]:
        """Count entries per tag."""
        with self._lock:
            counter: Counter[str] = Counter()
            for entry in self._entries.values():
                counter.update(entry.tags)
            return dict(counter.most_common())

    def entities(self, *, kind: str = "", limit: int = 30) -> dict[str, list[str]]:
        """Return the most common tracked entities, grouped by type."""
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for entry in self.list_entries(limit=100000, kind=kind):
            for name, values in (entry.metadata.get("entities") or {}).items():
                grouped[name].update(values)
        return {
            name: [value for value, _ in counter.most_common(limit)]
            for name, counter in grouped.items()
        }

    # -- deletion ------------------------------------------------------------

    def delete(self, entry_id: str) -> bool:
        """Delete one entry; report whether it existed."""
        with self._lock:
            entry = self._entries.pop(entry_id, None)
            if entry is None:
                return False
            self._order = [key for key in self._order if key != entry_id]
            self._fingerprints.pop(entry.fingerprint, None)
            self.vector_index.remove(entry_id)
            self.deleted += 1
        self.save()
        return True

    def delete_matching(
        self, query: str, *, kind: str = "", min_score: float = 0.25, limit: int = 100
    ) -> int:
        """Delete entries matching a query above a score threshold.

        Deliberately conservative: a low threshold would let a vague query wipe
        the store, so results below ``min_score`` are kept.
        """
        matches = self.search(query, limit=limit, kind=kind, min_score=min_score, touch=False)
        removed = 0
        for entry in matches:
            if self.delete(entry.id):
                removed += 1
        return removed

    def delete_older_than(self, seconds: float, *, kind: str = "") -> int:
        """Delete entries older than ``seconds``, preserving important ones."""
        cutoff = time.time() - max(0.0, seconds)
        victims = [
            entry.id
            for entry in self.list_entries(limit=100000, kind=kind, order="oldest")
            if entry.created_at < cutoff and entry.importance < 0.8
        ]
        return sum(1 for entry_id in victims if self.delete(entry_id))

    def clear(self) -> int:
        """Delete every entry; return how many were removed."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._order.clear()
            self._fingerprints.clear()
            self.vector_index.clear()
            self.deleted += count
        if self.path is not None and self.autosave:
            ensure_dir(self.path.parent)
            atomic_write_text(self.path, "")
        return count

    def compact(
        self, *, kind: str = "", keep: int = 50, dedupe_threshold: float = 0.95
    ) -> dict[str, Any]:
        """Merge near-duplicates and drop the weakest entries per kind.

        Args:
            kind: restrict compaction to one kind, empty for all
            keep: maximum entries to retain per kind
            dedupe_threshold: text similarity above which two entries merge
        """
        report: dict[str, Any] = {
            "before": len(self._entries),
            "merged": 0,
            "dropped": 0,
            "after": 0,
        }
        kinds = [kind.lower()] if kind else sorted(self.by_kind())
        for active in kinds:
            entries = self.list_entries(limit=100000, kind=active, order="importance")
            survivors: list[MemoryEntry] = []
            for entry in entries:
                duplicate_of = next(
                    (
                        other
                        for other in survivors
                        if similarity(other.content, entry.content) >= dedupe_threshold
                    ),
                    None,
                )
                if duplicate_of is not None:
                    duplicate_of.importance = max(duplicate_of.importance, entry.importance)
                    duplicate_of.access_count += entry.access_count
                    duplicate_of.metadata["merged_ids"] = list(
                        duplicate_of.metadata.get("merged_ids") or []
                    ) + [entry.id]
                    for tag in entry.tags:
                        if tag not in duplicate_of.tags:
                            duplicate_of.tags.append(tag)
                    self.delete(entry.id)
                    report["merged"] += 1
                    continue
                if len(survivors) >= max(1, keep):
                    self.delete(entry.id)
                    report["dropped"] += 1
                    continue
                survivors.append(entry)
        report["after"] = len(self._entries)
        self.save()
        return report

    # -- persistence ---------------------------------------------------------

    def _append(self, entry: MemoryEntry) -> None:
        """Append one entry to the JSONL file."""
        if self.path is None:
            return
        try:
            ensure_dir(self.path.parent)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(entry.to_json() + "\n")
        except OSError as exc:
            LOGGER.warning("could not append memory to %s: %s", self.path, exc)

    def save(self) -> Path | None:
        """Rewrite the whole store atomically; return the path written."""
        if self.path is None or not self.autosave:
            return None
        try:
            ensure_dir(self.path.parent)
            with self._lock:
                lines = [
                    self._entries[key].to_json() for key in self._order if key in self._entries
                ]
            atomic_write_text(self.path, "".join(line + "\n" for line in lines))
        except OSError as exc:
            raise StoreError(f"could not write memory store {self.path}: {exc}") from exc
        return self.path

    def load(self) -> int:
        """Read the JSONL file into memory; return how many entries loaded."""
        if self.path is None or not self.path.is_file():
            return 0
        loaded = 0
        skipped = 0
        with self._lock:
            self._entries.clear()
            self._order.clear()
            self._fingerprints.clear()
            self.vector_index.clear()
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = MemoryEntry.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    skipped += 1
                    continue
                self._entries[entry.id] = entry
                self._order.append(entry.id)
                self._fingerprints[entry.fingerprint] = entry.id
                if entry.embedding:
                    self.vector_index.add(entry.id, entry.embedding)
                loaded += 1
        if skipped:
            LOGGER.warning("skipped %d malformed memory line(s) in %s", skipped, self.path)
        return loaded

    def reindex_embeddings(self, *, batch_size: int = 32) -> int:
        """Recompute every embedding, e.g. after switching embedding models."""
        entries = self.list_entries(limit=100000)
        count = 0
        for start in range(0, len(entries), max(1, batch_size)):
            batch = entries[start : start + max(1, batch_size)]
            vectors = self._embed([entry.content for entry in batch])
            for entry, vector in zip(batch, vectors, strict=False):
                entry.embedding = list(vector)
                if vector:
                    self.vector_index.add(entry.id, vector)
                    count += 1
        self.save()
        return count

    def export_jsonl(self, path: str | Path, *, include_embeddings: bool = False) -> Path:
        """Write every entry to a JSONL file."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        with self._lock:
            lines = [
                self._entries[key].to_json(include_embedding=include_embeddings)
                for key in self._order
                if key in self._entries
            ]
        atomic_write_text(target, "".join(line + "\n" for line in lines))
        return target

    def import_jsonl(self, path: str | Path, *, replace: bool = False) -> int:
        """Import entries from a JSONL file; return how many were added."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise StoreError(f"No such memory export: {source}")
        if replace:
            self.clear()
        count = 0
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = MemoryEntry.from_dict(raw)
            self.add(
                entry.content,
                kind=entry.kind,
                tags=entry.tags,
                importance=entry.importance,
                source=entry.source,
                metadata=entry.metadata,
                entry_id=entry.id,
                embedding=entry.embedding or None,
            )
            count += 1
        return count

    def backup(self, suffix: str = "") -> Path | None:
        """Copy the store file aside before a destructive operation."""
        if self.path is None or not self.path.is_file():
            return None
        stamp = suffix or time.strftime("%Y%m%d-%H%M%S")
        target = self.path.with_suffix(f".{stamp}.bak")
        atomic_write_text(target, self.path.read_text(encoding="utf-8", errors="replace"))
        return target

    # -- introspection -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return store telemetry for ``magent memory stats``."""
        with self._lock:
            entries = list(self._entries.values())
            size = self.path.stat().st_size if self.path and self.path.is_file() else 0
            kinds = Counter(entry.kind for entry in entries)
            tag_counter: Counter[str] = Counter()
            for entry in entries:
                tag_counter.update(entry.tags)
            embedded = sum(1 for entry in entries if entry.embedding)
        total = len(entries)
        return {
            "entries": total,
            "path": str(self.path) if self.path else None,
            "bytes": size,
            "size_human": human_size(size),
            "kinds": dict(kinds.most_common()),
            "top_tags": dict(tag_counter.most_common(10)),
            "embedded": embedded,
            "embedding_coverage": round(embedded / total, 4) if total else 0.0,
            "average_importance": round(sum(entry.importance for entry in entries) / total, 4)
            if total
            else 0.0,
            "oldest_seconds": round(
                max((entry.age_seconds() for entry in entries), default=0.0), 1
            ),
            "added": self.added,
            "duplicates_skipped": self.duplicates_skipped,
            "deleted": self.deleted,
            "searches": self.searches,
            "average_search_ms": round(self.search_ms / self.searches, 4) if self.searches else 0.0,
            "retrieval_mode": self.retrieval_mode,
            "vector": self.vector_index.stats(),
            "embedder": self.embedder.stats() if hasattr(self.embedder, "stats") else {},
        }

    def to_dict(self, *, limit: int = 100) -> dict[str, Any]:
        """Serialize the store summary plus a page of entries."""
        return {
            "stats": self.stats(),
            "entries": [
                entry.to_dict(include_embedding=False) for entry in self.list_entries(limit=limit)
            ],
        }

    def summary_text(self, *, limit: int = 8, max_chars: int = 1200) -> str:
        """Render the top entries as a prompt-ready block."""
        entries = self.list_entries(limit=limit, order="importance")
        if not entries:
            return ""
        return ellipsize("\n".join(entry.summary_line() for entry in entries), max_chars)

    def __len__(self) -> int:
        """Return how many entries are stored."""
        return len(self._entries)

    def __contains__(self, entry_id: object) -> bool:
        """Report whether an identifier is stored."""
        return str(entry_id) in self._entries

    def __iter__(self) -> Iterator[MemoryEntry]:
        """Iterate entries in insertion order."""
        with self._lock:
            return iter([self._entries[key] for key in list(self._order) if key in self._entries])

    def __enter__(self) -> MemoryStore:
        """Support ``with MemoryStore(...) as store:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Flush the store when leaving a ``with`` block."""
        self.save()

    def __repr__(self) -> str:
        """Show the entry count and store path."""
        return f"<MemoryStore entries={len(self._entries)} path={self.path}>"


def extract_entities(text: str, *, limit: int = MAX_ENTITIES_PER_ENTRY) -> dict[str, list[str]]:
    """Pull lightweight entities out of content for later filtering.

    Emails, URLs, file paths, identifiers, and numbers are captured. This is
    deliberately shallow: it exists so an agent can ask "what did I note about
    ``config.py``" without a full NER model.
    """
    found: dict[str, list[str]] = {}
    source = collapse_whitespace(text or "")
    for name, pattern in ENTITY_PATTERNS.items():
        if name == "path":
            # Paths inside a URL are already captured by the url entity, and
            # reporting "/example.com/a" as a file path is just noise.
            source = ENTITY_PATTERNS["url"].sub(" ", source)
        matches = list(dict.fromkeys(pattern.findall(source)))
        if name == "identifier":
            # Only keep identifiers that look like code: a plain English word
            # would otherwise dominate the entity list with noise.
            matches = [item for item in matches if CODE_LIKE.search(item)][:limit]
        if matches:
            found[name] = matches[:limit]
    return found


def build_store(
    config: Any = None,
    *,
    backend: Any = None,
    path: str | Path | None = None,
    root: str | Path | None = None,
) -> MemoryStore:
    """Build a :class:`MemoryStore` from configuration, or a sensible default."""
    memory_config = getattr(config, "memory", config)
    if path is not None:
        return MemoryStore(path=path, config=memory_config, backend=backend)
    if config is None:
        return MemoryStore(path=None, backend=backend, autosave=False)
    return MemoryStore.from_config(config, backend=backend, root=root)


__all__ = [
    "CODE_LIKE",
    "ENTITY_PATTERNS",
    "MAX_ENTITIES_PER_ENTRY",
    "MemoryStore",
    "build_store",
    "extract_entities",
]
