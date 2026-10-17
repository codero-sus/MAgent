"""Retrieval-augmented generation over a local corpus.

:class:`Retriever` is the one object a caller needs: point it at files or a
directory, then ask it for context. It chunks, indexes, embeds, searches,
reranks, and renders a cited context block, and it can persist the index so a
second run does not re-read the corpus.

Everything works offline. Without a backend, embeddings come from the
deterministic hashing embedder and retrieval is lexical-plus-semantic over
hashed n-grams. With a Cortex backend attached, embeddings come from
``/v1/embeddings`` and reranking can use a cross-encoder model when one is
serving.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from ..memory.embeddings import Embedder, build_embedder
from ..utils.files import human_size, iter_files
from ..utils.text import ellipsize
from .chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_files,
    chunk_text_blocks,
)
from .citations import CitationSet, cited_context, references_block, verify_citations
from .index import ChunkIndex
from .reranker import rerank

LOGGER = get_logger("rag")
"""Logger for retrieval activity."""

DEFAULT_PATTERNS: tuple[str, ...] = ("*.md", "*.txt", "*.py", "*.rst", "*.json", "*.yaml", "*.yml")
"""File patterns ingested by default."""

DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)
"""Directories skipped during a walk."""


@dataclass(slots=True)
class IngestReport:
    """What one ingest pass did."""

    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    bytes: int = 0
    seconds: float = 0.0
    sources: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    embedded: int = 0

    @property
    def average_chunk_chars(self) -> float:
        """Return the mean chunk size."""
        return round(self.bytes / self.chunks, 1) if self.chunks else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "files": self.files,
            "chunks": self.chunks,
            "skipped": self.skipped[:50],
            "skipped_count": len(self.skipped),
            "bytes": self.bytes,
            "size_human": human_size(self.bytes),
            "seconds": round(self.seconds, 3),
            "sources": self.sources,
            "errors": self.errors[:20],
            "embedded": self.embedded,
            "average_chunk_chars": self.average_chunk_chars,
        }

    def summary(self) -> str:
        """Return a one-line human summary."""
        return (
            f"ingested {self.files} file(s) into {self.chunks} chunk(s) "
            f"({human_size(self.bytes)}) in {self.seconds:.2f}s"
            + (f", skipped {len(self.skipped)}" if self.skipped else "")
        )


@dataclass(slots=True)
class RetrievalResult:
    """The outcome of one retrieval, with everything needed to cite it."""

    query: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    citations: CitationSet | None = None
    mode: str = "hybrid"
    seconds: float = 0.0
    tokens: int = 0
    reranked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Return the rendered, cited context block."""
        return self.citations.context if self.citations else ""

    @property
    def sources(self) -> list[str]:
        """Return the distinct sources behind the result."""
        return (
            self.citations.sources()
            if self.citations
            else list(dict.fromkeys(chunk.source for chunk in self.chunks))
        )

    def text(self, *, with_references: bool = True) -> str:
        """Return the context block, optionally with the reference list."""
        if self.citations is None:
            return "\n\n".join(chunk.text for chunk in self.chunks)
        return (
            self.citations.render(include_context=not with_references)
            if with_references
            else self.context
        )

    def verify(self, answer: str) -> dict[str, Any]:
        """Check an answer's citations against this result's sources."""
        if self.citations is None:
            return {"grounded": False, "reason": "no citations"}
        return verify_citations(answer, self.citations.citations).to_dict()

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize the result for the API, CLI, and evaluation harness."""
        return {
            "query": self.query,
            "mode": self.mode,
            "count": len(self.chunks),
            "seconds": round(self.seconds, 4),
            "tokens": self.tokens,
            "reranked": self.reranked,
            "sources": self.sources,
            "chunks": [chunk.to_dict(include_text=include_text) for chunk in self.chunks],
            "citations": self.citations.to_dict() if self.citations else None,
            "metadata": self.metadata,
        }


class Retriever:
    """Ingest a corpus and retrieve cited context from it."""

    def __init__(
        self,
        *,
        index: ChunkIndex | None = None,
        backend: Any = None,
        embedder: Embedder | None = None,
        strategy: str = "",
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        max_files: int = 500,
        limit: int = 5,
        mode: str = "hybrid",
        lexical_weight: float = 0.5,
        use_mmr: bool = True,
        use_cross_encoder: bool = False,
        recency: bool = False,
        dedupe: bool = True,
        dim: int = 256,
        name: str = "rag",
    ) -> None:
        """Configure retrieval.

        Args:
            index: existing :class:`ChunkIndex` to reuse
            backend: LLM backend for real embeddings and cross-encoder reranking
            embedder: explicit embedder, overriding automatic selection
            strategy: force a chunking strategy, empty to pick per file
            max_chars: target chunk size
            overlap: overlap between chunks
            max_files: ceiling on files per ingest pass
            limit: default number of chunks to retrieve
            mode: ``hybrid``, ``bm25``, ``vector``, or ``none``
            lexical_weight: BM25 weight; the semantic term gets the remainder
            use_mmr: apply Maximal Marginal Relevance
            use_cross_encoder: rerank with the backend's reranking model
            recency: boost recently modified sources
            dedupe: drop near-duplicate chunks from a result
            dim: vector width for the offline embedder
            name: retriever name, used in logs and stats
        """
        self.name = name
        self.embedder = embedder or build_embedder(backend, dim=dim)
        self.backend = backend
        self.index = index if index is not None else ChunkIndex(embedder=self.embedder)
        self.strategy = strategy
        self.max_chars = max(64, int(max_chars))
        self.overlap = max(0, int(overlap))
        self.max_files = max(1, int(max_files))
        self.limit = max(1, int(limit))
        self.mode = mode
        self.lexical_weight = float(lexical_weight)
        self.use_mmr = bool(use_mmr)
        self.use_cross_encoder = bool(use_cross_encoder)
        self.recency = bool(recency)
        self.dedupe = bool(dedupe)
        self.retrievals = 0
        self.retrieval_ms = 0.0

    # -- ingestion -----------------------------------------------------------

    def ingest(
        self,
        paths: Iterable[str | Path],
        *,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        exclude: Sequence[str] = DEFAULT_EXCLUDE,
        recursive: bool = True,
        max_files: int | None = None,
        embed: bool = True,
        strategy: str = "",
    ) -> IngestReport:
        """Ingest files and directories into the index.

        Args:
            paths: files or directories to read
            patterns: glob patterns applied to directory walks
            exclude: directory names to skip
            recursive: descend into subdirectories
            max_files: ceiling for this pass
            embed: compute embeddings during ingest
            strategy: force a chunking strategy
        """
        started = time.perf_counter()
        ceiling = max(1, int(max_files or self.max_files))
        files: list[Path] = []
        skipped: list[str] = []
        errors: list[str] = []
        for raw in paths:
            target = Path(raw).expanduser()
            if target.is_file():
                files.append(target)
            elif target.is_dir():
                found = list(
                    iter_files(
                        target,
                        patterns=patterns,
                        exclude=exclude,
                        max_depth=None if recursive else 1,
                        limit=ceiling,
                    )
                )
                files.extend(found)
                if not found:
                    skipped.append(str(target))
            else:
                skipped.append(str(target))
        selected = files[:ceiling]
        skipped.extend(str(item) for item in files[ceiling:])
        chunks, missing = chunk_files(
            selected,
            strategy=strategy or self.strategy,
            max_chars=self.max_chars,
            overlap=self.overlap,
            max_files=ceiling,
        )
        skipped.extend(missing)
        total_bytes = 0
        for path in selected:
            try:
                total_bytes += path.stat().st_size
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        if chunks:
            with_mtime(chunks)
            self.index.add_chunks(chunks, embed=embed)
        embedded = self.index.build_embeddings() if embed else 0
        report = IngestReport(
            files=len(selected),
            chunks=len(chunks),
            skipped=skipped,
            bytes=total_bytes,
            seconds=time.perf_counter() - started,
            sources=self.index.sources(),
            errors=errors,
            embedded=embedded,
        )
        LOGGER.debug("%s ingest: %s", self.name, report.summary())
        return report

    def ingest_text(
        self,
        text: str,
        *,
        source: str = "inline",
        strategy: str = "",
        metadata: dict[str, Any] | None = None,
        embed: bool = True,
    ) -> int:
        """Index a string directly, e.g. a pasted document or a fetched page."""
        chunks = chunk_text_blocks(
            text,
            source=source,
            strategy=strategy or self.strategy or "paragraph",
            max_chars=self.max_chars,
            overlap=self.overlap,
            metadata=metadata,
        )
        if not chunks:
            return 0
        self.index.add_chunks(chunks, embed=embed)
        return len(chunks)

    def ingest_directory(self, root: str | Path, **kwargs: Any) -> IngestReport:
        """Ingest one directory tree."""
        return self.ingest([root], **kwargs)

    def refresh(self, paths: Iterable[str | Path] = (), **kwargs: Any) -> IngestReport:
        """Re-ingest the corpus, dropping sources that disappeared."""
        known = set(self.index.sources())
        report = self.ingest(paths, **kwargs) if paths else IngestReport()
        if not paths:
            report.sources = dict((known and self.index.sources()) or {})
            return report
        for source in known - set(report.sources):
            self.index.remove_source(source)
        return report

    def remove(self, source: str) -> int:
        """Drop one source from the index."""
        return self.index.remove_source(source)

    def clear(self) -> None:
        """Empty the index."""
        self.index.clear()

    # -- retrieval -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        mode: str = "",
        min_score: float = 0.0,
        sources: Sequence[str] | None = None,
        rerank_results: bool = True,
        cross_encoder: bool | None = None,
        use_mmr: bool | None = None,
        max_context_chars: int = 12_000,
        instruction: str = "",
    ) -> RetrievalResult:
        """Retrieve and rank chunks for a query.

        Args:
            query: the search text
            limit: how many chunks to return
            mode: override the configured retrieval mode
            min_score: drop results below this blended score
            sources: restrict to these source paths
            rerank_results: apply the reranking pipeline
            cross_encoder: override the cross-encoder setting
            use_mmr: override the MMR setting
            max_context_chars: ceiling for the rendered context block
            instruction: extra instruction appended to the context block
        """
        started = time.perf_counter()
        wanted = max(1, int(limit or self.limit))
        active_mode = (mode or self.mode or "hybrid").lower()
        pool = self.index.search(
            query,
            limit=max(wanted * 4, wanted),
            mode=active_mode,
            min_score=min_score,
            sources=sources,
            lexical_weight=self.lexical_weight,
        )
        reranked = False
        if rerank_results and pool:
            pool = rerank(
                pool,
                query,
                limit=wanted,
                embedder=self.embedder if self.use_mmr else None,
                backend=self.backend,
                use_cross_encoder=self.use_cross_encoder
                if cross_encoder is None
                else cross_encoder,
                use_mmr=self.use_mmr if use_mmr is None else use_mmr,
                recency=self.recency,
                dedupe=self.dedupe,
            )
            reranked = True
        selected = pool[:wanted]
        citations = cited_context(selected, max_chars=max_context_chars, instruction=instruction)
        result = RetrievalResult(
            query=query,
            chunks=selected,
            citations=citations,
            mode=active_mode,
            seconds=time.perf_counter() - started,
            tokens=sum(chunk.chars for chunk in selected) // 4,
            reranked=reranked,
            metadata={"index": self.index.stats(), "retriever": self.name},
        )
        self.retrievals += 1
        self.retrieval_ms += result.seconds * 1000.0
        return result

    def context(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_chars: int = 12_000,
        style: str = "numbered",
        instruction: str = "",
        **kwargs: Any,
    ) -> str:
        """Return just the rendered context block for a query."""
        result = self.retrieve(
            query, limit=limit, max_context_chars=max_chars, instruction=instruction, **kwargs
        )
        if style == "plain":
            return "\n\n".join(chunk.text for chunk in result.chunks)[:max_chars]
        if style == "references":
            return references_block(result.citations.citations if result.citations else [])
        return result.context

    def citation_set(self, query: str, **kwargs: Any) -> CitationSet:
        """Return the citations for a query without the full result."""
        result = self.retrieve(query, **kwargs)
        return result.citations or CitationSet()

    def answer_prompt(
        self,
        question: str,
        *,
        limit: int | None = None,
        max_chars: int = 12_000,
        instruction: str = "",
    ) -> str:
        """Build a complete grounded-answer prompt for one question."""
        guidance = instruction or (
            "Answer using only the retrieved context above. Cite each claim with its "
            "bracketed number. If the context is insufficient, say what is missing."
        )
        context = self.context(question, limit=limit, max_chars=max_chars, instruction=guidance)
        return f"{context}\n\nQuestion: {question}" if context else f"Question: {question}"

    def grounded_answer(
        self,
        question: str,
        *,
        backend: Any = None,
        model: str = "",
        limit: int | None = None,
        max_chars: int = 12_000,
        system: str = "",
    ) -> dict[str, Any]:
        """Retrieve, answer, and verify citations in one call.

        Returns the answer text, the sources used, and the citation check, which
        is exactly what the evaluation harness scores for groundedness.
        """
        from ..llm.base import CompletionRequest
        from ..types import Message

        active = backend or self.backend
        result = self.retrieve(question, limit=limit, max_context_chars=max_chars)
        prompt = self.answer_prompt(question, limit=limit, max_chars=max_chars)
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(prompt))
        payload: dict[str, Any] = {
            "question": question,
            "sources": result.sources,
            "chunks": len(result.chunks),
            "retrieval_seconds": round(result.seconds, 4),
        }
        if active is None:
            payload.update({"answer": "", "error": "no backend available"})
            return payload
        response = active.complete(
            CompletionRequest(messages=messages, model=model, temperature=0.2)
        )
        payload.update(
            {
                "answer": response.text,
                "model": response.model,
                "usage": response.usage.to_dict(),
                "citation_check": result.verify(response.text or ""),
                "references": references_block(
                    result.citations.citations if result.citations else []
                ),
            }
        )
        return payload

    # -- persistence and telemetry ------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the index so the next run skips ingestion."""
        return self.index.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Any = None,
        embedder: Embedder | None = None,
        **kwargs: Any,
    ) -> Retriever:
        """Rebuild a retriever from a saved index."""
        index = ChunkIndex.load(path, embedder=embedder, backend=backend)
        return cls(index=index, backend=backend, embedder=embedder or index.embedder, **kwargs)

    def stats(self) -> dict[str, Any]:
        """Return retriever and index telemetry."""
        return {
            "name": self.name,
            "index": self.index.stats(),
            "retrievals": self.retrievals,
            "average_retrieval_ms": round(self.retrieval_ms / self.retrievals, 3)
            if self.retrievals
            else 0.0,
            "mode": self.mode,
            "lexical_weight": self.lexical_weight,
            "mmr": self.use_mmr,
            "cross_encoder": self.use_cross_encoder,
            "dedupe": self.dedupe,
            "recency": self.recency,
            "embedder": self.embedder.stats() if hasattr(self.embedder, "stats") else {},
        }

    def describe(self) -> str:
        """Return a one-line description for the CLI."""
        stats = self.index.stats()
        return (
            f"{self.name}: {stats['chunks']} chunks from {stats['sources']} source(s), "
            f"mode={self.mode}, {human_size(stats['chars'])}"
        )

    def __len__(self) -> int:
        """Return how many chunks are indexed."""
        return len(self.index)

    def __repr__(self) -> str:
        """Show the retriever name and index size."""
        return f"<Retriever {self.name!r} chunks={len(self.index)} mode={self.mode!r}>"


def with_mtime(chunks: Sequence[Chunk]) -> None:
    """Record each source's modification time on its chunks, for recency ranking."""
    seen: dict[str, float] = {}
    for chunk in chunks:
        source = chunk.source
        if not source:
            continue
        if source not in seen:
            try:
                seen[source] = Path(source).stat().st_mtime
            except OSError:
                seen[source] = 0.0
        if seen[source]:
            chunk.metadata["mtime"] = seen[source]


def search_files(
    query: str,
    paths: Iterable[str | Path],
    *,
    limit: int = 5,
    backend: Any = None,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict[str, Any]]:
    """One-shot retrieval over a set of paths, without keeping an index.

    Convenient for a CLI query or a single tool call; for repeated queries build
    a :class:`Retriever` and reuse its index.
    """
    retriever = Retriever(backend=backend, max_chars=max_chars, limit=limit)
    retriever.ingest(paths, patterns=patterns)
    result = retriever.retrieve(query, limit=limit)
    return [
        {
            "rank": index,
            "source": chunk.source,
            "location": chunk.location,
            "heading": chunk.heading,
            "score": chunk.score,
            "preview": ellipsize(" ".join(chunk.text.split()), 300),
            "matches": _matching_lines(chunk.text, query),
        }
        for index, chunk in enumerate(result.chunks, start=1)
    ]


def _matching_lines(text: str, query: str, *, limit: int = 3) -> list[str]:
    """Return the lines of a chunk that contain a query term."""
    terms = [term for term in query.lower().split() if len(term) > 2]
    if not terms:
        return []
    hits: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(term in lowered for term in terms):
            hits.append(ellipsize(line.strip(), 200))
        if len(hits) >= limit:
            break
    return hits


def pattern_matches(name: str, patterns: Sequence[str]) -> bool:
    """Report whether a file name matches any of the patterns."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_PATTERNS",
    "IngestReport",
    "RetrievalResult",
    "Retriever",
    "pattern_matches",
    "search_files",
    "with_mtime",
]
