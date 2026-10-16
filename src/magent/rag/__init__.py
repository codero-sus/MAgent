"""Retrieval-augmented generation over local documents.

Pipeline: :mod:`magent.rag.chunker` splits documents into structure-aware
chunks, :mod:`magent.rag.index` keeps a hybrid lexical + vector index over them,
:mod:`magent.rag.reranker` improves the ordering (MMR, cross-encoder, recency,
dedupe), and :mod:`magent.rag.citations` turns the result into a numbered,
verifiable context block.

:class:`Retriever` wires the four together and is what callers use.
"""

from __future__ import annotations

from .chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    STRATEGIES,
    TEXT_SUFFIXES,
    Chunk,
    chunk_file,
    chunk_files,
    chunk_markdown,
    chunk_python,
    chunk_text_blocks,
    choose_strategy,
)
from .citations import (
    CITATION,
    Citation,
    CitationCheck,
    CitationSet,
    build_citations,
    citation_counts,
    cited_context,
    extract_citations,
    references_block,
    render_context,
    strip_citations,
    verify_citations,
)
from .index import INDEX_VERSION, ChunkIndex
from .reranker import (
    DEFAULT_LAMBDA,
    cross_encoder_rerank,
    dedupe_chunks,
    mmr_select,
    recency_boost,
    rerank,
)
from .retriever import (
    DEFAULT_EXCLUDE,
    DEFAULT_PATTERNS,
    IngestReport,
    RetrievalResult,
    Retriever,
    pattern_matches,
    search_files,
)

__all__ = [
    "CITATION",
    "Chunk",
    "ChunkIndex",
    "Citation",
    "CitationCheck",
    "CitationSet",
    "DEFAULT_EXCLUDE",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP",
    "DEFAULT_PATTERNS",
    "INDEX_VERSION",
    "IngestReport",
    "RetrievalResult",
    "Retriever",
    "STRATEGIES",
    "TEXT_SUFFIXES",
    "build_citations",
    "chunk_file",
    "chunk_files",
    "chunk_markdown",
    "chunk_python",
    "chunk_text_blocks",
    "choose_strategy",
    "citation_counts",
    "cited_context",
    "cross_encoder_rerank",
    "dedupe_chunks",
    "extract_citations",
    "mmr_select",
    "pattern_matches",
    "recency_boost",
    "references_block",
    "render_context",
    "rerank",
    "search_files",
    "strip_citations",
    "verify_citations",
]
