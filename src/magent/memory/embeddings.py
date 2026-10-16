"""Embedding providers for memory retrieval.

Three embedders share one interface, ``embed(list[str]) -> list[list[float]]``:

:class:`HashingEmbedder`
    Deterministic, offline, and dependency-free. It hashes character n-grams
    into a fixed-width vector, which is enough for lexical-semantic recall in a
    personal agent and makes the whole memory stack testable without a model.
:class:`BackendEmbedder`
    Delegates to a real backend's ``/v1/embeddings`` route, i.e. Cortex
    LLMHoster when it is running.
:class:`CachingEmbedder`
    Wraps either of the above with an LRU cache keyed by content hash, so
    re-embedding a stable corpus costs nothing.

The store picks one automatically: a backend embedder when a backend is
attached, otherwise the hashing embedder, so retrieval always works.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from typing import Any, Protocol, Sequence

from ..utils.hashing import content_hash
from ..utils.numbers import cosine_similarity, l2_normalize
from ..utils.text import normalize_newlines

DEFAULT_DIM = 256
"""Default vector width for the offline hashing embedder."""

NGRAM_RANGE = (3, 5)
"""Character n-gram sizes the hashing embedder folds into the vector."""


class Embedder(Protocol):
    """The interface every embedder satisfies."""

    dim: int
    """Width of the vectors this embedder returns."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


class HashingEmbedder:
    """A deterministic, offline embedding built from hashed character n-grams.

    This is a real embedding, not a stub: similar strings share n-grams and
    therefore land close together in cosine space. It needs no model, no
    network, and no third-party package, which is what lets MAgent's memory
    layer work on a fresh interpreter.
    """

    def __init__(self, *, dim: int = DEFAULT_DIM, ngram_range: tuple[int, int] = NGRAM_RANGE,
                 word_weight: float = 1.5, normalize: bool = True) -> None:
        """Configure the vector width and the features that are hashed.

        Args:
            dim: vector width; larger is more precise and more memory
            ngram_range: inclusive ``(min, max)`` character n-gram sizes
            word_weight: extra weight for whole-word features
            normalize: L2-normalize each vector so cosine equals dot product
        """
        self.dim = max(8, int(dim))
        self.ngram_range = (max(1, ngram_range[0]), max(ngram_range[0], ngram_range[1]))
        self.word_weight = float(word_weight)
        self.normalize = bool(normalize)
        self.name = "hashing"
        self.embedded = 0

    def _features(self, text: str) -> list[tuple[str, float]]:
        """Extract weighted features from one string."""
        lowered = normalize_newlines(text or "").lower()
        features: list[tuple[str, float]] = []
        minimum, maximum = self.ngram_range
        for size in range(minimum, maximum + 1):
            for index in range(0, max(0, len(lowered) - size + 1)):
                features.append((f"c{size}:{lowered[index:index + size]}", 1.0))
        for word in lowered.replace("\n", " ").split():
            cleaned = "".join(char for char in word if char.isalnum())
            if cleaned:
                features.append((f"w:{cleaned}", self.word_weight))
        return features

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        vector = [0.0] * self.dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        self.embedded += 1
        if self.normalize:
            normalized = l2_normalize(vector)
            return normalized if normalized else vector
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        return [self.embed_one(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        """Return the cosine similarity of two strings."""
        return cosine_similarity(self.embed_one(left), self.embed_one(right))

    def stats(self) -> dict[str, Any]:
        """Return embedder telemetry."""
        return {"name": self.name, "dim": self.dim, "embedded": self.embedded,
                "ngram_range": list(self.ngram_range)}


class BackendEmbedder:
    """Embed through an LLM backend's embeddings route (Cortex in production)."""

    def __init__(self, backend: Any, *, model: str = "", dim: int = 0, batch_size: int = 16,
                 timeout: float = 60.0) -> None:
        """Wrap a backend that exposes ``embed``.

        Args:
            backend: any object with ``embed(texts, model=..., batch_size=...)``
            model: embedding model override
            dim: expected vector width, 0 to accept whatever the backend returns
            batch_size: texts per request
            timeout: not used directly; the backend applies its own timeout
        """
        self.backend = backend
        self.model = model
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self.name = f"backend:{getattr(backend, 'name', type(backend).__name__)}"
        self.embedded = 0
        self.failures = 0
        self._resolved_dim = int(dim)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch through the backend, padding or truncating to ``dim``."""
        items = [text for text in texts]
        if not items:
            return []
        try:
            vectors = self.backend.embed(items, model=self.model, batch_size=self.batch_size)
        except TypeError:
            vectors = self.backend.embed(items)
        except Exception:  # noqa: BLE001 - callers fall back to lexical search
            self.failures += 1
            raise
        self.embedded += len(vectors)
        if vectors:
            self._resolved_dim = self._resolved_dim or len(vectors[0])
        if self.dim and self._resolved_dim != self.dim:
            vectors = [_fit(vector, self.dim) for vector in vectors]
        return [list(map(float, vector)) for vector in vectors]

    def stats(self) -> dict[str, Any]:
        """Return embedder telemetry."""
        return {"name": self.name, "model": self.model, "dim": self._resolved_dim,
                "embedded": self.embedded, "failures": self.failures}


def _fit(vector: Sequence[float], dim: int) -> list[float]:
    """Pad or truncate a vector to an exact width."""
    values = [float(item) for item in vector]
    if len(values) >= dim:
        return values[:dim]
    return values + [0.0] * (dim - len(values))


class CachingEmbedder:
    """An LRU cache in front of any other embedder."""

    def __init__(self, inner: Embedder, *, max_entries: int = 4096) -> None:
        """Wrap ``inner`` with a bounded cache.

        Args:
            inner: the embedder to cache
            max_entries: how many distinct texts to remember
        """
        self.inner = inner
        self.dim = inner.dim
        self.max_entries = max(16, int(max_entries))
        self.name = f"cache:{getattr(inner, 'name', 'inner')}"
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        """Return the cache key for one text."""
        return content_hash("embed", getattr(self.inner, "model", ""), text)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, only calling through for cache misses."""
        keys = [self._key(text) for text in texts]
        results: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        with self._lock:
            for index, key in enumerate(keys):
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    results[index] = cached
                    self.hits += 1
                else:
                    # Repeated texts inside one batch are embedded once, so a
                    # caller passing duplicates does not pay for them twice.
                    if key not in missing:
                        self.misses += 1
                    missing.setdefault(key, []).append(index)
        if missing:
            unique_keys = list(missing)
            unique_texts = [texts[positions[0]] for positions in (missing[key] for key in unique_keys)]
            computed = self.inner.embed(unique_texts)
            with self._lock:
                for key, vector in zip(unique_keys, computed, strict=False):
                    for index in missing[key]:
                        results[index] = vector
                    self._cache[key] = vector
                    self._cache.move_to_end(key)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
        return [vector if vector is not None else [] for vector in results]

    def clear(self) -> None:
        """Drop every cached vector."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache telemetry, including the inner embedder's own stats."""
        with self._lock:
            size = len(self._cache)
        total = self.hits + self.misses
        inner_stats = self.inner.stats() if hasattr(self.inner, "stats") else {}
        return {"name": self.name, "entries": size, "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "inner": inner_stats}


def build_embedder(backend: Any = None, *, model: str = "", dim: int = DEFAULT_DIM,
                   cache: bool = True, max_entries: int = 4096,
                   batch_size: int = 16) -> Embedder:
    """Return the best available embedder for the current environment.

    Args:
        backend: an LLM backend with an ``embed`` method, or ``None``
        model: embedding model override for the backend
        dim: vector width for the offline embedder
        cache: wrap the result in an LRU cache
        max_entries: cache capacity
        batch_size: texts per backend request
    """
    embedder: Embedder
    if backend is not None and hasattr(backend, "embed"):
        embedder = BackendEmbedder(backend, model=model, dim=dim or 0, batch_size=batch_size)
    else:
        embedder = HashingEmbedder(dim=dim or DEFAULT_DIM)
    return CachingEmbedder(embedder, max_entries=max_entries) if cache else embedder


__all__ = [
    "DEFAULT_DIM",
    "NGRAM_RANGE",
    "BackendEmbedder",
    "CachingEmbedder",
    "Embedder",
    "HashingEmbedder",
    "build_embedder",
]
