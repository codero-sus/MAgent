"""A persistable hybrid index over :class:`~magent.rag.chunker.Chunk` objects.

The index keeps three views of the same corpus: the chunks themselves, a BM25
posting structure for lexical search, and an optional vector index for semantic
search. A query blends them and reports both terms, which is what makes a bad
retrieval diagnosable instead of mysterious.

It is deliberately simple and exact: brute-force cosine over a few thousand
chunks costs microseconds, and there is no third-party vector database to
install, run, or keep in sync.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import IndexError_
from ..logging_ import get_logger
from ..memory.embeddings import DEFAULT_DIM, Embedder, build_embedder
from ..memory.ranking import BM25Scorer, tokenize
from ..memory.vector import VectorIndex
from ..utils.files import atomic_write_text, ensure_dir, human_size
from ..utils.numbers import cosine_similarity
from .chunker import Chunk

LOGGER = get_logger("rag.index")
"""Logger for index operations."""

INDEX_VERSION = 1
"""Schema version written into a saved index."""


def augmented_text(chunk: Chunk) -> str:
    """Return the text BM25 scores for a chunk.

    Tiny corpora give rare tokens (a project name, a lone heading word) a huge
    IDF, so a short document containing one can outrank the document that
    actually answers the query.  Repeating the heading, the file stem and the
    symbol name in the scored text spreads those tokens across every chunk of
    the same file and gives queries that mention a filename or function a fair
    chance to match.  The stored ``chunk.text`` is left untouched: augmentation
    affects ranking only, never the context handed to the model.
    """
    parts: list[str] = []
    if chunk.heading:
        parts.extend([chunk.heading, chunk.heading])
    stem = Path(chunk.source).stem
    if stem:
        parts.append(stem)
    symbol = str(chunk.metadata.get("symbol") or "")
    if symbol:
        parts.append(symbol)
    parts.append(chunk.text)
    return "\n".join(part for part in parts if part)


class ChunkIndex:
    """Hybrid lexical and vector index over document chunks."""

    def __init__(self, *, embedder: Embedder | None = None, backend: Any = None,
                 dim: int = DEFAULT_DIM, buckets: int = 0, embed_on_add: bool = True,
                 embedding_batch_size: int = 16) -> None:
        """Create an empty index.

        Args:
            embedder: embedder used for the semantic term
            backend: LLM backend, used to build an embedder when none is given
            dim: vector width for the offline embedder
            buckets: bucket count for approximate vector search
            embed_on_add: embed each chunk as it is added
            embedding_batch_size: texts per embedding request
        """
        self.embedder = embedder or build_embedder(backend, dim=dim, batch_size=embedding_batch_size)
        self.embed_on_add = bool(embed_on_add)
        self.batch_size = max(1, int(embedding_batch_size))
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, int] = {}
        self._scorer: BM25Scorer | None = None
        self._dirty_scorer = True
        self.vectors = VectorIndex(buckets=buckets)
        self._lock = threading.RLock()
        self.added = 0
        self.searches = 0
        self.search_ms = 0.0
        self.created_at = time.time()

    # -- building ------------------------------------------------------------

    def add_chunk(self, chunk: Chunk, *, embed: bool | None = None) -> Chunk:
        """Add one chunk, embedding it when configured to."""
        should_embed = self.embed_on_add if embed is None else embed
        with self._lock:
            if chunk.id in self._by_id:
                position = self._by_id[chunk.id]
                self._chunks[position] = chunk
            else:
                self._by_id[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
            self._dirty_scorer = True
            self.added += 1
        if should_embed and not chunk.metadata.get("embedded"):
            self._embed_chunks([chunk])
        return chunk

    def add_chunks(self, chunks: Iterable[Chunk], *, embed: bool | None = None) -> int:
        """Add many chunks, embedding them in batches."""
        items = list(chunks)
        for chunk in items:
            self.add_chunk(chunk, embed=False)
        should_embed = self.embed_on_add if embed is None else embed
        if should_embed:
            self._embed_chunks(items)
        return len(items)

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed a batch of chunks and index their vectors."""
        pending = [chunk for chunk in chunks if chunk.text and not chunk.metadata.get("embedded")]
        if not pending:
            return 0
        embedded = 0
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            try:
                vectors = self.embedder.embed([chunk.text for chunk in batch])
            except Exception as exc:  # noqa: BLE001 - lexical search must still work
                LOGGER.warning("embedding failed for %d chunk(s): %s", len(batch), exc)
                return embedded
            for chunk, vector in zip(batch, vectors, strict=False):
                if not vector:
                    continue
                chunk.metadata["embedded"] = True
                chunk.metadata["dim"] = len(vector)
                self.vectors.add(chunk.id, vector)
                embedded += 1
        return embedded

    def build_embeddings(self) -> int:
        """Embed every chunk that does not have a vector yet."""
        with self._lock:
            pending = [chunk for chunk in self._chunks if not chunk.metadata.get("embedded")]
        return self._embed_chunks(pending)

    def remove_source(self, source: str) -> int:
        """Drop every chunk from one source; return how many were removed."""
        with self._lock:
            kept = [chunk for chunk in self._chunks if chunk.source != source]
            removed = len(self._chunks) - len(kept)
            if not removed:
                return 0
            for chunk in self._chunks:
                if chunk.source == source:
                    self.vectors.remove(chunk.id)
            self._chunks = kept
            self._by_id = {chunk.id: index for index, chunk in enumerate(kept)}
            self._dirty_scorer = True
        return removed

    def clear(self) -> None:
        """Empty the index."""
        with self._lock:
            self._chunks.clear()
            self._by_id.clear()
            self._scorer = None
            self._dirty_scorer = True
            self.vectors.clear()

    # -- searching -----------------------------------------------------------

    def _lexical_scores(self, query: str) -> list[float]:
        """Return normalized BM25 scores, rebuilding the scorer when needed."""
        with self._lock:
            if self._dirty_scorer or self._scorer is None:
                self._scorer = BM25Scorer([augmented_text(c) for c in self._chunks])
                self._dirty_scorer = False
            scorer = self._scorer
        if scorer.total != len(self._chunks):
            scorer = BM25Scorer([augmented_text(c) for c in self._chunks])
            with self._lock:
                self._scorer = scorer
                self._dirty_scorer = False
        return scorer.scores(query)

    def search(self, query: str, *, limit: int = 5, mode: str = "hybrid", min_score: float = 0.0,
               sources: Sequence[str] | None = None, lexical_weight: float = 0.5,
               heading_boost: float = 0.1, embed_query: bool = True) -> list[Chunk]:
        """Find the chunks most relevant to a query.

        Args:
            query: the search text
            limit: maximum chunks to return
            mode: ``hybrid``, ``bm25``, ``vector``, or ``none``
            min_score: drop results below this blended score
            sources: restrict to these source paths
            lexical_weight: weight of the BM25 term; the semantic term gets the rest
            heading_boost: bonus for a query token appearing in a heading or symbol
            embed_query: allow the vector term when embeddings exist
        """
        started = time.perf_counter()
        with self._lock:
            candidates = list(self._chunks)
        if not candidates:
            return []
        if sources:
            allowed = {str(item) for item in sources}
            candidates = [chunk for chunk in candidates if chunk.source in allowed]
            if not candidates:
                return []
        active = mode if mode in {"hybrid", "bm25", "vector", "none"} else "hybrid"
        lexical: dict[str, float] = {}
        if active in {"hybrid", "bm25"} and query.strip():
            scorer = BM25Scorer([augmented_text(c) for c in candidates])
            for chunk, score in zip(candidates, scorer.scores(query), strict=True):
                lexical[chunk.id] = score
        semantic: dict[str, float] = {}
        if active in {"hybrid", "vector"} and query.strip() and embed_query:
            try:
                query_vector = self.embedder.embed([query])[0]
            except Exception as exc:  # noqa: BLE001 - fall back to lexical
                LOGGER.debug("query embedding failed: %s", exc)
                query_vector = []
            if query_vector:
                hits = self.vectors.search(query_vector, limit=max(limit * 10, 50))
                semantic = {key: score for key, score in hits}
        query_tokens = set(tokenize(query))
        semantic_weight = max(0.0, 1.0 - lexical_weight)
        scored: list[tuple[float, Chunk]] = []
        for chunk in candidates:
            base = (lexical_weight * lexical.get(chunk.id, 0.0)
                    + semantic_weight * max(0.0, semantic.get(chunk.id, 0.0)))
            if active == "none":
                base = 0.0
            bonus = 0.0
            if heading_boost and query_tokens:
                heading_tokens = set(tokenize(chunk.heading)) | set(
                    tokenize(str(chunk.metadata.get("symbol", "")))
                )
                if heading_tokens & query_tokens:
                    bonus = heading_boost
            total = round(min(1.0, base + bonus), 6)
            if total >= min_score and (base > 0 or bonus > 0 or active == "none"):
                chunk.score = total
                chunk.metadata["lexical"] = round(lexical.get(chunk.id, 0.0), 6)
                chunk.metadata["semantic"] = round(semantic.get(chunk.id, 0.0), 6)
                scored.append((total, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        self.searches += 1
        self.search_ms += (time.perf_counter() - started) * 1000.0
        return [chunk for _, chunk in scored[: max(1, limit)]]

    def get(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by identifier."""
        with self._lock:
            position = self._by_id.get(chunk_id)
            return self._chunks[position] if position is not None else None

    def chunks(self) -> list[Chunk]:
        """Return every chunk, in insertion order."""
        with self._lock:
            return list(self._chunks)

    def sources(self) -> dict[str, int]:
        """Count chunks per source."""
        counts: dict[str, int] = {}
        for chunk in self.chunks():
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return counts

    def tokens(self) -> int:
        """Return the total character count divided by four, as a token estimate."""
        return sum(chunk.chars for chunk in self.chunks()) // 4

    def __len__(self) -> int:
        """Return how many chunks are indexed."""
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        """Report whether a chunk identifier is indexed."""
        return str(chunk_id) in self._by_id

    # -- persistence ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return index telemetry."""
        with self._lock:
            total = len(self._chunks)
            embedded = sum(1 for chunk in self._chunks if chunk.metadata.get("embedded"))
            chars = sum(chunk.chars for chunk in self._chunks)
        return {
            "chunks": total,
            "sources": len(self.sources()),
            "embedded": embedded,
            "embedding_coverage": round(embedded / total, 4) if total else 0.0,
            "chars": chars,
            "size_human": human_size(chars),
            "added": self.added,
            "searches": self.searches,
            "average_search_ms": round(self.search_ms / self.searches, 4) if self.searches else 0.0,
            "vector": self.vectors.stats(),
            "created_at": self.created_at,
        }

    def save(self, path: str | Path, *, include_vectors: bool = True) -> Path:
        """Write the index to disk atomically."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        payload = {
            "version": INDEX_VERSION,
            "created_at": self.created_at,
            "saved_at": time.time(),
            "stats": self.stats(),
            "chunks": [chunk.to_dict() for chunk in self.chunks()],
        }
        if include_vectors:
            payload["vectors"] = self.vectors.to_dict()
        atomic_write_text(target, json.dumps(payload, ensure_ascii=False))
        return target

    @classmethod
    def load(cls, path: str | Path, *, embedder: Embedder | None = None,
             backend: Any = None) -> ChunkIndex:
        """Rebuild an index from a file written by :meth:`save`."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise IndexError_(f"No RAG index at {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        index = cls(embedder=embedder, backend=backend, embed_on_add=False)
        index.created_at = float(payload.get("created_at") or time.time())
        chunks = [Chunk.from_dict(raw) for raw in (payload.get("chunks") or [])]
        index.add_chunks(chunks, embed=False)
        vectors = payload.get("vectors") or {}
        for entry in vectors.get("entries") or []:
            index.vectors.add(str(entry.get("key", "")), entry.get("vector") or [])
        return index

    def __repr__(self) -> str:
        """Show the chunk and source counts."""
        return f"<ChunkIndex chunks={len(self._chunks)} sources={len(self.sources())}>"


__all__ = ["INDEX_VERSION", "ChunkIndex", "augmented_text"]
