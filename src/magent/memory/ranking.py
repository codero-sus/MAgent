"""Ranking for memory retrieval: BM25, vectors, recency, and importance.

Retrieval quality comes from blending signals rather than trusting one. The
blend here is explicit and tunable, and each term is reported in the result so
a bad ranking can be diagnosed instead of guessed at.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..utils.numbers import cosine_similarity
from .entry import MemoryEntry

STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "will", "would", "can", "could", "should", "not",
    "but", "you", "your", "our", "its", "their", "them", "then", "than",
})
"""Words excluded from the lexical signal."""


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, dropping stopwords and single letters."""
    import re

    return [
        token for token in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def entry_text(item: Any) -> str:
    """Return the searchable text of an entry, chunk, or plain string.

    :class:`BM25Scorer` is shared by the memory store and the RAG index, whose
    records name their body differently, so the text is resolved here once.
    """
    if isinstance(item, str):
        return item
    for attribute in ("content", "text", "body"):
        value = getattr(item, attribute, None)
        if isinstance(value, str) and value:
            return value
        if callable(value):
            try:
                called = value()
            except Exception:  # noqa: BLE001 - fall through to the next attribute
                continue
            if isinstance(called, str) and called:
                return called
    return ""


@dataclass(slots=True)
class DecayPolicy:
    """Exponential recency decay with an importance floor.

    ``half_life_days`` is how long until an untouched entry's recency term falls
    to 0.5. Important entries decay more slowly, so a high-importance decision
    survives longer than a passing note.
    """

    half_life_days: float = 14.0
    importance_weight: float = 0.6
    min_score: float = 0.05

    def factor(self, entry: MemoryEntry, *, now: float | None = None) -> float:
        """Return the recency multiplier for one entry, in ``0..1``."""
        if self.half_life_days <= 0:
            return 1.0
        idle_days = entry.idle_seconds(now=now) / 86400.0
        effective_half_life = self.half_life_days * (1.0 + self.importance_weight * entry.importance)
        decayed = 0.5 ** (idle_days / effective_half_life)
        return max(self.min_score, min(1.0, decayed))


@dataclass(slots=True)
class RankingWeights:
    """How much each retrieval signal contributes to the final score."""

    lexical: float = 0.45
    semantic: float = 0.35
    importance: float = 0.12
    recency: float = 0.08

    def normalized(self) -> "RankingWeights":
        """Return a copy whose weights sum to 1."""
        total = self.lexical + self.semantic + self.importance + self.recency
        if total <= 0:
            return RankingWeights()
        return RankingWeights(
            lexical=self.lexical / total,
            semantic=self.semantic / total,
            importance=self.importance / total,
            recency=self.recency / total,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize the weights."""
        return {"lexical": self.lexical, "semantic": self.semantic,
                "importance": self.importance, "recency": self.recency}


class BM25Scorer:
    """Okapi BM25 over a collection of memory entries."""

    def __init__(self, entries: Sequence[MemoryEntry], *, k1: float = 1.5, b: float = 0.75) -> None:
        """Index ``entries`` for repeated queries.

        Args:
            entries: the corpus to index
            k1: term-frequency saturation parameter
            b: length normalization parameter
        """
        self.entries = list(entries)
        self.k1 = k1
        self.b = b
        self.vectors = [Counter(tokenize(entry_text(entry))) for entry in self.entries]
        self.lengths = [sum(counter.values()) for counter in self.vectors]
        self.total = len(self.entries)
        self.average_length = (sum(self.lengths) / self.total) if self.total else 0.0
        self.document_frequency: Counter[str] = Counter()
        for counter in self.vectors:
            self.document_frequency.update(counter.keys())

    def idf(self, term: str) -> float:
        """Inverse document frequency for one term."""
        hits = self.document_frequency.get(term, 0)
        if not hits or not self.total:
            return 0.0
        return math.log(1 + (self.total - hits + 0.5) / (hits + 0.5))

    def scores(self, query: str) -> list[float]:
        """Return one BM25 score per entry, in corpus order."""
        terms = tokenize(query)
        out = [0.0] * self.total
        if not terms or not self.total:
            return out
        for index, counter in enumerate(self.vectors):
            length = self.lengths[index] or 1
            total = 0.0
            for term in terms:
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                weight = self.idf(term)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.average_length or 1)
                )
                total += weight * (frequency * (self.k1 + 1)) / denominator
            out[index] = total
        peak = max(out) if out else 0.0
        if peak > 0:
            out = [value / peak for value in out]
        return out

    def matched_terms(self, index: int, query: str) -> list[str]:
        """Return which query terms an entry actually contains."""
        counter = self.vectors[index] if index < len(self.vectors) else Counter()
        return sorted(term for term in set(tokenize(query)) if counter.get(term))


def hybrid_score(*, lexical: float = 0.0, semantic: float = 0.0, importance: float = 0.0,
                 recency: float = 1.0, tag_bonus: float = 0.0,
                 weights: RankingWeights | None = None) -> float:
    """Blend the retrieval signals into one score in ``0..1``.

    Args:
        lexical: normalized BM25 score
        semantic: cosine similarity of embeddings
        importance: the entry's declared importance
        recency: the decay factor for the entry
        tag_bonus: extra credit for a tag match
        weights: signal weights, defaults to :class:`RankingWeights`
    """
    active = (weights or RankingWeights()).normalized()
    total = (
        active.lexical * max(0.0, min(1.0, lexical))
        + active.semantic * max(0.0, min(1.0, semantic))
        + active.importance * max(0.0, min(1.0, importance))
        + active.recency * max(0.0, min(1.0, recency))
        + max(0.0, tag_bonus)
    )
    return round(max(0.0, min(1.0, total)), 6)


def rank_entries(entries: Iterable[MemoryEntry], query: str, *, embedder: Any = None,
                 weights: RankingWeights | None = None, decay: DecayPolicy | None = None,
                 tags: Sequence[str] | None = None, mode: str = "hybrid",
                 now: float | None = None) -> list[MemoryEntry]:
    """Score and sort entries for a query, filling ``entry.score``.

    Args:
        entries: candidates to rank
        query: the search text
        embedder: optional embedder for the semantic term
        weights: signal weights
        decay: recency policy
        tags: tags that earn a bonus when matched
        mode: ``hybrid``, ``bm25``, ``vector``, or ``none``
        now: reference timestamp for recency
    """
    items = list(entries)
    if not items:
        return []
    active_decay = decay or DecayPolicy()
    active_weights = weights or RankingWeights()
    lexical_scores: list[float] = [0.0] * len(items)
    if mode in {"hybrid", "bm25"} and query:
        lexical_scores = BM25Scorer(items).scores(query)
    semantic_scores: list[float] = [0.0] * len(items)
    if mode in {"hybrid", "vector"} and query and embedder is not None:
        try:
            vectors = [entry.embedding for entry in items if entry.embedding]
            if vectors and len(vectors) == len(items):
                query_vector = embedder.embed([query])[0]
                semantic_scores = [cosine_similarity(query_vector, vector) for vector in vectors]
        except Exception:  # noqa: BLE001 - semantic ranking is best effort
            semantic_scores = [0.0] * len(items)
    wanted = {str(tag).strip().lower() for tag in (tags or ()) if str(tag).strip()}
    for index, entry in enumerate(items):
        tag_bonus = 0.05 if wanted and entry.matches_tags(wanted) else 0.0
        if mode == "none":
            entry.score = round(entry.importance, 6)
        else:
            entry.score = hybrid_score(
                lexical=lexical_scores[index],
                semantic=semantic_scores[index],
                importance=entry.importance,
                recency=active_decay.factor(entry, now=now),
                tag_bonus=tag_bonus,
                weights=active_weights,
            )
    items.sort(key=lambda item: item.score, reverse=True)
    return items


__all__ = [
    "BM25Scorer",
    "entry_text",
    "DecayPolicy",
    "RankingWeights",
    "STOPWORDS",
    "hybrid_score",
    "rank_entries",
    "tokenize",
]
