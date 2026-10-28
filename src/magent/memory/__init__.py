"""MAgent memory: durable entries, embeddings, ranking, and the context window.

Two layers, one package:

:mod:`magent.memory.store`
    Long-term memory. JSONL-backed entries with hybrid (BM25 + vector +
    importance + recency) retrieval, dedupe, compaction, and entity tracking.
:mod:`magent.memory.short_term`
    The conversation window. Token-aware trimming that summarizes evicted turns
    instead of losing them.

Retrieval never requires a running model: :class:`HashingEmbedder` provides a
deterministic offline embedding, and :class:`BackendEmbedder` upgrades to real
embeddings from Cortex LLMHoster when a backend is attached.
"""

from __future__ import annotations

from .embeddings import (
    DEFAULT_DIM,
    NGRAM_RANGE,
    BackendEmbedder,
    CachingEmbedder,
    Embedder,
    HashingEmbedder,
    build_embedder,
)
from .entry import DEFAULT_KIND, KINDS, MemoryEntry
from .ranking import (
    BM25Scorer,
    DecayPolicy,
    RankingWeights,
    entry_text,
    hybrid_score,
    rank_entries,
    tokenize,
)
from .short_term import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_TOKENS,
    ConversationWindow,
    ShortTermMemory,
)
from .store import ENTITY_PATTERNS, MemoryStore, build_store, extract_entities
from .vector import VectorIndex

__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_KIND",
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_TOKENS",
    "ENTITY_PATTERNS",
    "KINDS",
    "NGRAM_RANGE",
    "BM25Scorer",
    "BackendEmbedder",
    "CachingEmbedder",
    "ConversationWindow",
    "DecayPolicy",
    "Embedder",
    "HashingEmbedder",
    "MemoryEntry",
    "MemoryStore",
    "RankingWeights",
    "ShortTermMemory",
    "VectorIndex",
    "build_embedder",
    "build_store",
    "entry_text",
    "extract_entities",
    "hybrid_score",
    "rank_entries",
    "tokenize",
]
