"""A brute-force vector index with optional coarse bucketing.

For a personal agent's memory - thousands to tens of thousands of entries -
exact cosine search over a Python list is fast enough and has no dependencies.
The index keeps that promise explicit: it is exact, it is serializable, and it
reports how long a search took so a caller can tell when the corpus has grown
past what brute force should handle.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import IndexError_
from ..utils.files import atomic_write_text, ensure_dir
from ..utils.numbers import cosine_similarity


class VectorIndex:
    """Exact nearest-neighbour search over ``(key, vector)`` pairs."""

    def __init__(self, *, dim: int = 0, buckets: int = 0) -> None:
        """Create an empty index.

        Args:
            dim: vector width; 0 accepts whatever the first insertion provides
            buckets: when positive, cluster vectors into this many buckets and
                search only the closest ones (approximate, much faster)
        """
        self.dim = int(dim)
        self.buckets = max(0, int(buckets))
        self._keys: list[str] = []
        self._vectors: list[list[float]] = []
        self._positions: dict[str, int] = {}
        self._centroids: list[list[float]] = []
        self._assignments: list[int] = []
        self._lock = threading.RLock()
        self._dirty = True
        self.searches = 0
        self.search_ms = 0.0

    # -- mutation ------------------------------------------------------------

    def add(self, key: str, vector: Sequence[float]) -> None:
        """Insert or replace one vector."""
        values = [float(item) for item in vector]
        if not values:
            return
        with self._lock:
            if self.dim and len(values) != self.dim:
                raise IndexError_(f"vector for {key!r} has dim {len(values)}, expected {self.dim}")
            self.dim = self.dim or len(values)
            position = self._positions.get(key)
            if position is None:
                self._positions[key] = len(self._keys)
                self._keys.append(key)
                self._vectors.append(values)
            else:
                self._vectors[position] = values
            self._dirty = True

    def add_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> int:
        """Insert several vectors; return how many were added."""
        count = 0
        for key, vector in items:
            self.add(key, vector)
            count += 1
        return count

    def remove(self, key: str) -> bool:
        """Drop one vector; report whether it existed."""
        with self._lock:
            position = self._positions.pop(key, None)
            if position is None:
                return False
            del self._keys[position]
            del self._vectors[position]
            self._positions = {name: index for index, name in enumerate(self._keys)}
            self._dirty = True
            return True

    def clear(self) -> None:
        """Empty the index."""
        with self._lock:
            self._keys.clear()
            self._vectors.clear()
            self._positions.clear()
            self._centroids.clear()
            self._assignments.clear()
            self._dirty = True

    # -- search --------------------------------------------------------------

    def search(self, vector: Sequence[float], *, limit: int = 5,
               min_score: float = 0.0) -> list[tuple[str, float]]:
        """Return the closest keys with their cosine similarities.

        Args:
            vector: the query vector
            limit: maximum neighbours to return
            min_score: drop results below this similarity
        """
        query = [float(item) for item in vector]
        started = time.perf_counter()
        with self._lock:
            candidates = self._candidate_positions(query)
            scored = [
                (self._keys[position], cosine_similarity(query, self._vectors[position]))
                for position in candidates
            ]
        results = [item for item in scored if item[1] >= min_score]
        results.sort(key=lambda item: item[1], reverse=True)
        self.searches += 1
        self.search_ms += (time.perf_counter() - started) * 1000.0
        return results[: max(1, limit)]

    def _candidate_positions(self, query: Sequence[float]) -> list[int]:
        """Return the positions to score, honoring bucket pruning."""
        total = len(self._keys)
        if not self.buckets or total <= self.buckets * 4:
            return list(range(total))
        if self._dirty or not self._centroids:
            self._rebuild_buckets()
        nearest = sorted(
            range(len(self._centroids)),
            key=lambda index: cosine_similarity(query, self._centroids[index]),
            reverse=True,
        )
        probe = max(1, self.buckets // 4)
        wanted = set(nearest[:probe])
        return [position for position, bucket in enumerate(self._assignments) if bucket in wanted]

    def _rebuild_buckets(self) -> None:
        """Recompute bucket centroids with a single k-means-style pass."""
        total = len(self._vectors)
        count = min(self.buckets, total)
        if count <= 1:
            self._centroids = []
            self._assignments = [0] * total
            self._dirty = False
            return
        stride = max(1, total // count)
        self._centroids = [list(self._vectors[index]) for index in range(0, total, stride)][:count]
        self._assignments = [
            max(range(len(self._centroids)),
                key=lambda index: cosine_similarity(vector, self._centroids[index]))
            for vector in self._vectors
        ]
        for bucket in range(len(self._centroids)):
            members = [self._vectors[i] for i, assignment in enumerate(self._assignments)
                       if assignment == bucket]
            if members:
                width = len(members[0])
                self._centroids[bucket] = [
                    sum(member[axis] for member in members) / len(members) for axis in range(width)
                ]
        self._dirty = False

    # -- introspection -------------------------------------------------------

    def __len__(self) -> int:
        """Return how many vectors are indexed."""
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        """Report whether a key is indexed."""
        return str(key) in self._positions

    def keys(self) -> list[str]:
        """Return every indexed key."""
        with self._lock:
            return list(self._keys)

    def get(self, key: str) -> list[float] | None:
        """Return one stored vector."""
        with self._lock:
            position = self._positions.get(key)
            return list(self._vectors[position]) if position is not None else None

    def magnitude(self, vector: Sequence[float]) -> float:
        """Return the L2 norm of a vector."""
        return math.sqrt(sum(float(item) ** 2 for item in vector))

    def stats(self) -> dict[str, Any]:
        """Return index size and search telemetry."""
        with self._lock:
            size = len(self._keys)
        return {
            "entries": size,
            "dim": self.dim,
            "buckets": self.buckets,
            "searches": self.searches,
            "average_search_ms": round(self.search_ms / self.searches, 4) if self.searches else 0.0,
            "total_search_ms": round(self.search_ms, 3),
        }

    # -- persistence ---------------------------------------------------------

    def to_dict(self, *, precision: int = 6) -> dict[str, Any]:
        """Serialize the index to a JSON-compatible dictionary."""
        with self._lock:
            return {
                "dim": self.dim,
                "buckets": self.buckets,
                "entries": [
                    {"key": key, "vector": [round(value, precision) for value in vector]}
                    for key, vector in zip(self._keys, self._vectors, strict=True)
                ],
            }

    def save(self, path: str | Path) -> Path:
        """Write the index to disk atomically."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        atomic_write_text(target, json.dumps(self.to_dict(), ensure_ascii=False))
        return target

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Rebuild an index from a file written by :meth:`save`."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise IndexError_(f"No vector index at {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        index = cls(dim=int(data.get("dim", 0)), buckets=int(data.get("buckets", 0)))
        for entry in data.get("entries") or []:
            index.add(str(entry.get("key", "")), entry.get("vector") or [])
        return index

    @classmethod
    def from_items(cls, items: Iterable[tuple[str, Sequence[float]]], *, buckets: int = 0) -> VectorIndex:
        """Build an index from an iterable of ``(key, vector)`` pairs."""
        index = cls(buckets=buckets)
        index.add_many(items)
        return index


__all__ = ["VectorIndex"]
