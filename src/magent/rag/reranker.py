"""Re-ranking retrieved chunks: diversity, cross-encoder, and recency.

A pure similarity ranking often returns five near-copies of the same paragraph.
The rerankers here fix that:

:func:`mmr_select`
    Maximal Marginal Relevance - keep relevance but penalize chunks that repeat
    what is already selected.
:func:`cross_encoder_rerank`
    Ask Cortex's reranking model to score ``(query, chunk)`` pairs directly, when
    a model with the ``reranking`` capability is available.
:func:`recency_boost`
    Nudge newer sources up, for corpora where the latest file wins.

Each returns a new list; none of them mutate the caller's ordering in place.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..logging_ import get_logger
from ..memory.embeddings import Embedder
from ..utils.numbers import cosine_similarity
from .chunker import Chunk

LOGGER = get_logger("rag.rerank")
"""Logger for reranking fallbacks."""

DEFAULT_LAMBDA = 0.7
"""Default MMR trade-off: 0.7 relevance, 0.3 novelty."""


def mmr_select(chunks: Sequence[Chunk], query: str, *, embedder: Embedder | None = None,
               limit: int = 5, lambda_: float = DEFAULT_LAMBDA,
               vectors: dict[str, list[float]] | None = None) -> list[Chunk]:
    """Select chunks by Maximal Marginal Relevance.

    Args:
        chunks: candidates, already scored by the index
        query: the search text
        embedder: embedder used to compute the query vector
        limit: how many chunks to return
        lambda_: relevance versus novelty trade-off in ``0..1``
        vectors: precomputed chunk vectors keyed by chunk id
    """
    items = [chunk for chunk in chunks if chunk.text.strip()]
    if not items or limit <= 0:
        return []
    if len(items) <= 1:
        return items[: max(1, limit)]
    lookup = dict(vectors or {})
    query_vector: list[float] = []
    if embedder is not None:
        missing = [chunk for chunk in items if chunk.id not in lookup]
        if missing:
            try:
                computed = embedder.embed([chunk.text for chunk in missing])
                for chunk, vector in zip(missing, computed, strict=False):
                    if vector:
                        lookup[chunk.id] = vector
            except Exception as exc:  # noqa: BLE001 - fall back to score ordering
                LOGGER.debug("mmr embedding failed: %s", exc)
        try:
            query_vector = embedder.embed([query])[0]
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("mmr query embedding failed: %s", exc)
    if not query_vector and not lookup:
        # Nothing to measure novelty against, so plain relevance order is right.
        return sorted(items, key=lambda chunk: chunk.score, reverse=True)[: max(1, limit)]
    # Relevance stays the score the retrieval pipeline already computed. MMR's
    # job is to add diversity, not to re-judge relevance with a weaker signal:
    # recomputing it as a raw cosine discarded the hybrid blend (and mixed
    # incomparable scales, since chunks without a vector fell back to a 0..1
    # score), which let a weakly related chunk outrank the correct answer.
    # The cosine is still used below, where it belongs: the novelty term.
    relevance = {chunk.id: chunk.score for chunk in items}
    selected: list[Chunk] = []
    remaining = list(items)
    while remaining and len(selected) < max(1, limit):
        best: Chunk | None = None
        best_value = float("-inf")
        for chunk in remaining:
            novelty = 1.0
            for chosen in selected:
                if chunk.id in lookup and chosen.id in lookup:
                    novelty = min(novelty, 1.0 - max(0.0, cosine_similarity(lookup[chunk.id], lookup[chosen.id])))
            value = lambda_ * relevance.get(chunk.id, 0.0) + (1.0 - lambda_) * novelty
            if value > best_value:
                best_value, best = value, chunk
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


def cross_encoder_rerank(chunks: Sequence[Chunk], query: str, backend: Any, *, limit: int = 5,
                         model: str = "", top_n: int | None = None) -> list[Chunk]:
    """Rerank with a cross-encoder model when the backend offers one.

    Falls back to the incoming order when the backend has no reranking route or
    the call fails, so retrieval never depends on an optional model.
    """
    items = list(chunks)
    if not items or backend is None or not hasattr(backend, "rerank"):
        return items[: max(1, limit)]
    documents = [chunk.text for chunk in items]
    try:
        payload = backend.rerank(query=query, documents=documents, model=model,
                                 top_n=top_n or max(1, limit))
    except TypeError:
        try:
            payload = backend.rerank(query, documents)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("rerank call failed: %s", exc)
            return items[: max(1, limit)]
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("rerank call failed: %s", exc)
        return items[: max(1, limit)]
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return items[: max(1, limit)]
    ordered: list[Chunk] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        score = float(entry.get("relevance_score") or entry.get("score") or 0.0)
        if isinstance(index, int) and 0 <= index < len(items):
            chunk = items[index]
            chunk.score = round(score, 6)
            chunk.metadata["rerank_score"] = round(score, 6)
            ordered.append(chunk)
    return ordered[: max(1, limit)] or items[: max(1, limit)]


def recency_boost(chunks: Sequence[Chunk], *, half_life_days: float = 30.0,
                  weight: float = 0.1, now: float | None = None) -> list[Chunk]:
    """Add a small bonus for recently modified sources and re-sort."""
    moment = now if now is not None else time.time()
    scored: list[Chunk] = []
    for chunk in chunks:
        modified = float(chunk.metadata.get("mtime") or chunk.metadata.get("created_at") or 0.0)
        bonus = 0.0
        if modified and half_life_days > 0:
            age_days = max(0.0, (moment - modified) / 86400.0)
            bonus = weight * (0.5 ** (age_days / half_life_days))
        chunk.score = round(min(1.0, chunk.score + bonus), 6)
        chunk.metadata["recency_bonus"] = round(bonus, 6)
        scored.append(chunk)
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def dedupe_chunks(chunks: Sequence[Chunk], *, threshold: float = 0.97) -> list[Chunk]:
    """Drop chunks whose text is nearly identical to one already kept."""
    from ..utils.text import similarity

    kept: list[Chunk] = []
    for chunk in chunks:
        if any(similarity(chunk.text, other.text) >= threshold for other in kept):
            continue
        kept.append(chunk)
    return kept


def rerank(chunks: Sequence[Chunk], query: str, *, limit: int = 5, embedder: Embedder | None = None,
           backend: Any = None, use_cross_encoder: bool = False, use_mmr: bool = True,
           recency: bool = False, dedupe: bool = True,
           score_fn: Callable[[Chunk], float] | None = None) -> list[Chunk]:
    """Apply the configured reranking pipeline in one call.

    Args:
        chunks: candidates from the index
        query: the search text
        limit: how many to return
        embedder: embedder for MMR
        backend: backend for cross-encoder reranking
        use_cross_encoder: try the reranking model first
        use_mmr: apply MMR after scoring
        recency: apply the recency bonus
        dedupe: drop near-duplicate chunks
        score_fn: optional custom scorer applied before ordering
    """
    items = list(chunks)
    if score_fn is not None:
        for chunk in items:
            chunk.score = float(score_fn(chunk))
        items.sort(key=lambda chunk: chunk.score, reverse=True)
    if dedupe:
        items = dedupe_chunks(items)
    if use_cross_encoder and backend is not None:
        items = cross_encoder_rerank(items, query, backend, limit=max(limit, limit * 3))
    if recency:
        items = recency_boost(items)
    if use_mmr and embedder is not None:
        items = mmr_select(items, query, embedder=embedder, limit=limit)
    return items[: max(1, limit)]


__all__ = [
    "DEFAULT_LAMBDA",
    "cross_encoder_rerank",
    "dedupe_chunks",
    "mmr_select",
    "recency_boost",
    "rerank",
]
