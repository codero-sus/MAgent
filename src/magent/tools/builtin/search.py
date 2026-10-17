"""Retrieval tools: lexical, ranked, and semantic search over a corpus.

``search_documents`` builds an in-memory index on first use and caches it for
the lifetime of the run, so repeated queries over the same corpus do not re-read
the files. Ranking blends BM25-style term frequency, field weighting, freshness,
and (when a backend is configured) cosine similarity over embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.files import read_text_safe
from ...utils.numbers import cosine_similarity
from ...utils.text import ellipsize, normalize_newlines

WORD = re.compile(r"[a-z0-9_]+")
"""Tokenizer for the lexical index."""

INDEX_CACHE_KEY = "__magent_search_index__"
"""Metadata key under which a run caches its built index."""

MAX_INDEXED_CHARS = 2_000_000
"""Ceiling on total indexed text per run, to bound memory."""


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, dropping pure stopwords."""
    tokens = WORD.findall(normalize_newlines(text).lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "not",
        "but",
        "you",
        "your",
        "our",
        "its",
        "their",
        "them",
        "then",
        "than",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "when",
        "where",
        "which",
        "what",
        "who",
    }
)
"""Common English words excluded from the index."""


class DocumentIndex:
    """A small in-memory inverted index with BM25 scoring.

    Built from a list of ``{"id", "title", "text", "path", "tags"}`` documents.
    Everything is plain Python data structures so an index can be pickled into
    a cache or rebuilt cheaply at the start of a run.
    """

    def __init__(
        self, documents: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        """Index ``documents`` with the given BM25 tuning parameters."""
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        self.term_vectors: list[Counter[str]] = []
        for position, document in enumerate(documents):
            tokens = tokenize(
                f"{document.get('title', '')} {document.get('title', '')} "
                f"{document.get('text', '')}"
            )
            counts = Counter(tokens)
            self.term_vectors.append(counts)
            self.lengths.append(len(tokens))
            for term in counts:
                self.postings.setdefault(term, []).append((position, counts[term]))
        self.total = len(documents)
        self.average_length = (sum(self.lengths) / self.total) if self.total else 0.0

    def idf(self, term: str) -> float:
        """Inverse document frequency for one term."""
        hits = len(self.postings.get(term, ()))
        if not hits or not self.total:
            return 0.0
        return math.log(1 + (self.total - hits + 0.5) / (hits + 0.5))

    def search(
        self, query: str, limit: int = 10, tag: str = "", kind: str = "", min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Rank documents against ``query`` using BM25."""
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = {}
        for term in terms:
            weight = self.idf(term)
            if weight <= 0:
                continue
            for position, frequency in self.postings.get(term, ()):
                length = self.lengths[position] or 1
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.average_length or 1)
                )
                scores[position] = (
                    scores.get(position, 0.0) + weight * (frequency * (self.k1 + 1)) / denominator
                )
        results: list[dict[str, Any]] = []
        for position, score in scores.items():
            document = self.documents[position]
            if tag and tag.lower() not in [item.lower() for item in document.get("tags", [])]:
                continue
            if kind and document.get("kind", "") != kind:
                continue
            if score < min_score:
                continue
            results.append(
                {
                    "id": document.get("id", position),
                    "title": document.get("title", ""),
                    "path": document.get("path", ""),
                    "kind": document.get("kind", ""),
                    "score": round(score, 4),
                    "matched_terms": sorted(
                        {term for term in terms if term in self.term_vectors[position]}
                    ),
                    "chars": len(document.get("text", "")),
                    "snippet": make_snippet(document.get("text", ""), terms),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def stats(self) -> dict[str, Any]:
        """Report index size and vocabulary statistics."""
        return {
            "documents": self.total,
            "terms": len(self.postings),
            "average_length": round(self.average_length, 2),
            "longest_document": max(self.lengths) if self.lengths else 0,
            "indexed_chars": sum(len(document.get("text", "")) for document in self.documents),
        }


def make_snippet(text: str, terms: list[str], radius: int = 90) -> str:
    """Extract the most relevant sentence-sized window around a matched term."""
    if not text:
        return ""
    lowered = text.lower()
    best_position = -1
    for term in terms:
        position = lowered.find(term)
        if position >= 0 and (best_position < 0 or position < best_position):
            best_position = position
    if best_position < 0:
        return ellipsize(normalize_newlines(text).replace("\n", " ").strip(), radius * 2)
    start = max(0, best_position - radius)
    end = min(len(text), best_position + radius)
    snippet = normalize_newlines(text[start:end]).replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _load_corpus(
    context: ToolContext, path: str, patterns: str, max_files: int
) -> list[dict[str, Any]]:
    """Read files from the workspace into index documents."""
    root = context.resolve_path(path or ".")
    if context.sandbox is not None:
        context.sandbox.check_read(root)
    if not root.exists():
        raise ToolExecutionError(f"No such path: {root}")
    globs = [item.strip() for item in patterns.split(",") if item.strip()] or ["**/*"]
    candidates: list[Path] = []
    if root.is_file():
        candidates = [root]
    else:
        for pattern in globs:
            candidates.extend(candidate for candidate in root.glob(pattern) if candidate.is_file())
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= max_files:
            break
    documents: list[dict[str, Any]] = []
    total = 0
    for index, candidate in enumerate(sorted(unique), start=1):
        try:
            text = read_text_safe(candidate, limit=1_000_000)
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip():
            continue
        total += len(text)
        if total > MAX_INDEXED_CHARS:
            break
        documents.append(
            {
                "id": str(index),
                "title": candidate.name,
                "path": context.sandbox.relative(candidate) if context.sandbox else str(candidate),
                "kind": candidate.suffix.lstrip(".").lower() or "file",
                "tags": [candidate.suffix.lstrip(".").lower()],
                "text": text,
            }
        )
    return documents


@tool(
    category="search", risk="read", filesystem=True, timeout=120.0, tags=["search", "bm25", "core"]
)
def search_documents(
    query: str,
    path: str = ".",
    patterns: str = "**/*.md,**/*.txt,**/*.py",
    limit: int = 8,
    max_files: int = 500,
    tag: str = "",
    kind: str = "",
    semantic: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Search files in the workspace and return ranked matches with snippets.

    The index is cached for the run, so a second query over the same path is
    nearly free.

    Args:
        query: search text
        path: directory or file to search, relative to the workspace
        patterns: comma separated glob patterns
        limit: maximum results
        max_files: maximum files to index
        tag: only return documents carrying this tag
        kind: only return documents of this file extension
        semantic: rerank with embeddings when a backend is configured
    """
    ctx = context or ToolContext()
    if not query.strip():
        raise ToolExecutionError("Empty query")
    cache_key = f"{INDEX_CACHE_KEY}:{path}:{patterns}:{max_files}"
    index = ctx.metadata.get(cache_key) if ctx.metadata else None
    if not isinstance(index, DocumentIndex):
        documents = _load_corpus(ctx, path, patterns, max_files)
        index = DocumentIndex(documents)
        if ctx.metadata is not None:
            ctx.metadata[cache_key] = index
    results = index.search(query, limit=max(1, min(limit, 50)), tag=tag, kind=kind)
    if semantic and results and ctx.backend is not None:
        results = _semantic_rerank(ctx, index, query, results)
    return {
        "query": query,
        "count": len(results),
        "results": results,
        "index": index.stats(),
        "semantic": semantic and ctx.backend is not None,
        "text": "\n".join(
            f"{item['score']:.3f}  {item['path']}: {item['snippet']}" for item in results
        )
        or "(no matches)",
    }


def _semantic_rerank(
    context: ToolContext, index: DocumentIndex, query: str, results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Blend embedding similarity into the lexical scores."""
    backend = context.backend
    if backend is None or not hasattr(backend, "embed"):
        return results
    try:
        vectors = backend.embed([query] + [item["snippet"] for item in results])
    except Exception:
        return results
    if len(vectors) < len(results) + 1:
        return results
    query_vector = vectors[0]
    for item, vector in zip(results, vectors[1:], strict=False):
        similarity = cosine_similarity(query_vector, vector)
        item["semantic_score"] = round(similarity, 4)
        item["score"] = round(item["score"] * 0.5 + similarity * 2.0, 4)
    results.sort(key=lambda entry: entry["score"], reverse=True)
    return results


@tool(category="search", risk="safe", tags=["search", "similarity"])
def similar_texts(text: str, candidates: str, limit: int = 5) -> dict[str, Any]:
    """Rank candidate strings by similarity to a query string.

    Uses a token-overlap and edit-distance blend, so it works offline and is
    useful for deduplication and fuzzy matching.

    Args:
        text: the reference text
        candidates: JSON array of candidate strings, or newline separated text
        limit: maximum results
    """
    from ...utils.json_utils import loads_lenient
    from ...utils.text import similarity

    parsed = loads_lenient(candidates)
    if isinstance(parsed, str):
        items = [line for line in normalize_newlines(parsed).split("\n") if line.strip()]
    elif isinstance(parsed, list):
        items = [str(item) for item in parsed]
    else:
        raise ToolExecutionError("candidates must be a JSON array or newline separated text")
    scored = sorted(
        (
            {
                "text": ellipsize(item, 300),
                "index": position,
                "score": round(similarity(text, item), 4),
            }
            for position, item in enumerate(items)
        ),
        key=lambda entry: entry["score"],
        reverse=True,
    )
    return {
        "query": ellipsize(text, 200),
        "candidates": len(items),
        "results": scored[: max(1, limit)],
        "best": scored[0]["text"] if scored else "",
    }


TOOLS: list[Any] = [search_documents, similar_texts]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every search tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "INDEX_CACHE_KEY",
    "MAX_INDEXED_CHARS",
    "STOPWORDS",
    "TOOLS",
    "DocumentIndex",
    "load",
    "make_snippet",
    "search_documents",
    "similar_texts",
    "tokenize",
]
