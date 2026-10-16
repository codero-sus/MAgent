"""Disk-backed response and embedding caches.

Caching is the single biggest win for local inference: repeated system prompts,
tool schemas, and RAG queries hit the same model with the same input. MAgent
stores results in a sharded directory of JSON files plus an optional SQLite
index, both dependency-free.

Safety properties:

* Keys are content hashes over the *whole* semantic request (model, messages,
  tools, sampling params), so a different temperature never returns a stale
  answer.
* Entries expire by TTL and the store enforces a byte budget with LRU eviction.
* Streaming responses are only cached once fully drained.
* Secrets are never part of a key and never stored in a value.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import StoreError
from ..types import ChatResponse, Usage
from ..utils.files import ensure_dir, human_size
from ..utils.hashing import content_hash

DEFAULT_TTL_SECONDS = 3600.0
"""Default cache entry lifetime."""

DEFAULT_MAX_ENTRIES = 4096
"""Default in-memory LRU capacity."""

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
"""Default on-disk cache budget (256 MiB)."""

SCHEMA_VERSION = 1
"""Version of the SQLite index schema."""


@dataclass(slots=True)
class CacheStats:
    """Counters describing cache effectiveness."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    expired: int = 0
    bytes_stored: int = 0
    errors: int = 0

    @property
    def lookups(self) -> int:
        """Return total lookup attempts."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Return the fraction of lookups served from cache."""
        total = self.lookups
        return round(self.hits / total, 4) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize including derived rates."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "expired": self.expired,
            "errors": self.errors,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
            "bytes_stored": self.bytes_stored,
            "bytes_stored_human": human_size(self.bytes_stored),
        }

    def reset(self) -> None:
        """Zero every counter."""
        self.hits = self.misses = self.writes = self.evictions = self.expired = self.errors = 0
        self.bytes_stored = 0


class MemoryLRU:
    """Thread-safe LRU map with optional per-entry expiry."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, ttl: float | None = None) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` when absent or expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            value, stored_at = entry
            if self.ttl is not None and time.time() - stored_at > self.ttl:
                del self._data[key]
                self.stats.expired += 1
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store a value, evicting the least recently used entry if needed."""
        with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            self.stats.writes += 1
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1
        if ttl is not None:
            self.ttl = ttl

    def delete(self, key: str) -> bool:
        """Remove a key; report whether it existed."""
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        """Return the number of live entries."""
        with self._lock:
            return len(self._data)

    def keys(self) -> list[str]:
        """Return a snapshot of the stored keys."""
        with self._lock:
            return list(self._data)

    def peek(self, key: str) -> tuple[Any, float] | None:
        """Return ``(value, stored_at)`` without affecting LRU order."""
        with self._lock:
            return self._data.get(key)


@dataclass
class DiskCache:
    """Sharded JSON-file cache with an LRU eviction budget.

    Files live at ``<root>/<prefix>/<aa>/<bb>/<hash>.json`` so no single
    directory ever holds more than 256 entries, which keeps ``readdir`` fast on
    every filesystem.
    """

    root: Path
    prefix: str = "responses"
    ttl: float | None = DEFAULT_TTL_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_entries: int = 8192
    memory: MemoryLRU = field(default_factory=lambda: MemoryLRU(max_entries=512, ttl=DEFAULT_TTL_SECONDS))
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        """Create the cache directory tree."""
        self.root = Path(self.root).expanduser()
        ensure_dir(self.base_dir)

    @property
    def base_dir(self) -> Path:
        """Return the directory holding this cache's shards."""
        return self.root / self.prefix

    def path_for(self, key: str) -> Path:
        """Return the shard file path for a cache key."""
        digest = content_hash(key)
        return self.base_dir / digest[:2] / digest[2:4] / f"{digest}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached payload, or ``None`` when absent or expired."""
        cached = self.memory.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        path = self.path_for(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            self.stats.misses += 1
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.stats.errors += 1
            self.stats.misses += 1
            path.unlink(missing_ok=True)
            return None
        stored_at = float(payload.get("stored_at") or 0.0)
        if self.ttl is not None and time.time() - stored_at > self.ttl:
            self.stats.expired += 1
            self.stats.misses += 1
            path.unlink(missing_ok=True)
            return None
        self.stats.hits += 1
        value = payload.get("value")
        self.memory.set(key, value)
        return value

    def set(self, key: str, value: Any, *, extra: dict[str, Any] | None = None) -> None:
        """Persist a value to disk and the memory tier."""
        payload = {
            "key": key,
            "stored_at": time.time(),
            "ttl": self.ttl,
            "value": value,
            "extra": extra or {},
        }
        try:
            text = json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            self.stats.errors += 1
            raise StoreError(f"Cache value for {key[:32]} is not serializable: {exc}") from exc
        path = self.path_for(key)
        with self._lock:
            try:
                ensure_dir(path.parent)
                path.write_text(text, encoding="utf-8")
            except OSError as exc:
                self.stats.errors += 1
                raise StoreError(f"Could not write cache entry {path}: {exc}") from exc
            self.stats.writes += 1
            self.stats.bytes_stored += len(text.encode("utf-8"))
        self.memory.set(key, value)
        self._enforce_budget()

    def delete(self, key: str) -> bool:
        """Remove one cache entry."""
        self.memory.delete(key)
        return self.path_for(key).unlink(missing_ok=True)

    def clear(self) -> None:
        """Delete every entry in this cache."""
        import shutil

        with self._lock:
            self.memory.clear()
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir, ignore_errors=True)
            ensure_dir(self.base_dir)
            self.stats.bytes_stored = 0

    def _enforce_budget(self) -> None:
        """Evict oldest entries until the byte and entry budgets are met."""
        entries = self._scan_entries()
        total = sum(size for _, size, _ in entries)
        if total <= self.max_bytes and len(entries) <= self.max_entries:
            return
        entries.sort(key=lambda item: item[2])
        with self._lock:
            for path, size, _ in entries:
                if total <= self.max_bytes and len(entries) <= self.max_entries:
                    break
                try:
                    path.unlink()
                    total -= size
                    self.stats.evictions += 1
                    entries = [(p, s, m) for p, s, m in entries if p != path]
                except OSError:
                    self.stats.errors += 1

    def _scan_entries(self) -> list[tuple[Path, int, float]]:
        """Return ``(path, size, mtime)`` for every cached file."""
        out: list[tuple[Path, int, float]] = []
        if not self.base_dir.exists():
            return out
        for shard_a in self.base_dir.iterdir():
            if not shard_a.is_dir():
                continue
            for shard_b in shard_a.iterdir():
                if not shard_b.is_dir():
                    continue
                for entry in shard_b.iterdir():
                    if entry.suffix != ".json":
                        continue
                    try:
                        stat = entry.stat()
                    except OSError:
                        continue
                    out.append((entry, stat.st_size, stat.st_mtime))
        return out

    def entry_count(self) -> int:
        """Return how many entries are stored on disk."""
        return len(self._scan_entries())

    def disk_usage(self) -> int:
        """Return total bytes consumed on disk."""
        return sum(size for _, size, _ in self._scan_entries())

    def prune_expired(self) -> int:
        """Delete expired entries eagerly and return how many were removed."""
        if self.ttl is None:
            return 0
        cutoff = time.time() - self.ttl
        removed = 0
        for path, _, mtime in self._scan_entries():
            if mtime < cutoff:
                if path.unlink(missing_ok=True):
                    removed += 1
                    self.stats.expired += 1
        return removed


class ResponseCache:
    """Cache of :class:`ChatResponse` objects keyed by request signature."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        enabled: bool = True,
        ttl: float = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        prefix: str = "responses",
    ) -> None:
        self.enabled = enabled
        self.stats = CacheStats()
        self._disk: DiskCache | None = None
        if enabled and directory is not None:
            self._disk = DiskCache(
                root=Path(directory), prefix=prefix, ttl=ttl, max_bytes=max_bytes
            )
            self.stats = self._disk.stats
        self._memory = MemoryLRU(max_entries=DEFAULT_MAX_ENTRIES, ttl=ttl if enabled else None)

    @staticmethod
    def key_for(signature: str, namespace: str = "chat") -> str:
        """Return the cache key for a request signature."""
        return content_hash(namespace, signature)

    def get(self, signature: str, *, namespace: str = "chat") -> ChatResponse | None:
        """Look up a cached response."""
        if not self.enabled:
            return None
        key = self.key_for(signature, namespace)
        payload = None
        if self._disk is not None:
            payload = self._disk.get(key)
        if payload is None:
            payload = self._memory.get(key)
        if payload is None:
            return None
        try:
            response = ChatResponse(
                text=str(payload.get("text", "")),
                model=str(payload.get("model", "")),
                usage=Usage.from_dict(payload.get("usage")),
                finish_reason=str(payload.get("finish_reason", "stop")),
                raw=dict(payload.get("raw") or {}),
                cache_hit=True,
            )
            return response
        except (TypeError, ValueError):
            return None

    def set(self, signature: str, response: ChatResponse, *, namespace: str = "chat") -> None:
        """Store a response in both cache tiers."""
        if not self.enabled or response.has_tool_calls:
            return
        payload = {
            "text": response.text,
            "model": response.model,
            "usage": response.usage.to_dict(),
            "finish_reason": response.finish_reason,
            "raw": {k: v for k, v in (response.raw or {}).items() if k in {"id", "object", "created"}},
            "cached_at": time.time(),
        }
        key = self.key_for(signature, namespace)
        self._memory.set(key, payload)
        if self._disk is not None:
            self._disk.set(key, payload)

    def clear(self) -> None:
        """Empty both cache tiers."""
        self._memory.clear()
        if self._disk is not None:
            self._disk.clear()

    def stats_dict(self) -> dict[str, Any]:
        """Return combined cache telemetry."""
        return {
            "enabled": self.enabled,
            "persistent": self._disk is not None,
            "memory_entries": len(self._memory),
            "disk_entries": self._disk.entry_count() if self._disk else 0,
            "disk_bytes": self._disk.disk_usage() if self._disk else 0,
            **self.stats.to_dict(),
        }


class EmbeddingCache:
    """SQLite-backed cache of embedding vectors.

    Embeddings are expensive to recompute and change only when the model or the
    text changes, so they are cached by ``(model, text)`` hash with no TTL by
    default.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ttl: float | None = None,
        table: str = "embeddings",
    ) -> None:
        self.path = Path(path).expanduser()
        self.ttl = ttl
        self.table = table
        ensure_dir(self.path.parent)
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._connection = self._connect()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        """Open the SQLite connection with sane pragmas."""
        try:
            connection = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10.0)
        except sqlite3.Error as exc:
            raise StoreError(f"Could not open embedding cache at {self.path}: {exc}") from exc
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _migrate(self) -> None:
        """Create the cache table and version row if missing."""
        with self._lock:
            self._connection.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.table} (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    hits INTEGER DEFAULT 0
                )"""
            )
            self._connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_model ON {self.table}(model)"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._connection.commit()

    @staticmethod
    def key_for(model: str, text: str) -> str:
        """Return the cache key for one embedding."""
        return content_hash("embed", model, text)

    @staticmethod
    def _pack(vector: list[float]) -> bytes:
        """Serialize a float vector to compact bytes."""
        import array

        data = array.array("f", vector)
        return data.tobytes()

    @staticmethod
    def _unpack(blob: bytes) -> list[float]:
        """Deserialize a float vector from bytes."""
        import array

        data = array.array("f")
        data.frombytes(blob)
        return list(data)

    def get(self, model: str, text: str) -> list[float] | None:
        """Return a cached vector, or ``None``."""
        key = self.key_for(model, text)
        with self._lock:
            row = self._connection.execute(
                f"SELECT vector, created_at, dim FROM {self.table} WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.stats.misses += 1
                return None
            if self.ttl is not None and time.time() - float(row[1]) > self.ttl:
                self._connection.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
                self._connection.commit()
                self.stats.expired += 1
                self.stats.misses += 1
                return None
            self._connection.execute(
                f"UPDATE {self.table} SET hits = hits + 1 WHERE key = ?", (key,)
            )
            self._connection.commit()
            self.stats.hits += 1
            try:
                return self._unpack(row[0])
            except (ValueError, TypeError):
                self.stats.errors += 1
                return None

    def get_many(self, model: str, texts: list[str]) -> dict[str, list[float]]:
        """Return cached vectors for several texts in one query."""
        if not texts:
            return {}
        keys = {self.key_for(model, text): text for text in texts}
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT key, vector FROM {self.table} WHERE key IN ({placeholders})",
                tuple(keys),
            ).fetchall()
        out: dict[str, list[float]] = {}
        for key, blob in rows:
            self.stats.hits += 1
            out[keys[key]] = self._unpack(blob)
        self.stats.misses += len(texts) - len(out)
        return out

    def set(self, model: str, text: str, vector: list[float], *, token_count: int = 0) -> None:
        """Store one embedding vector."""
        if not vector:
            return
        key = self.key_for(model, text)
        blob = self._pack(vector)
        with self._lock:
            self._connection.execute(
                f"""INSERT OR REPLACE INTO {self.table}
                    (key, model, dim, vector, token_count, created_at, hits)
                    VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (key, model, len(vector), blob, token_count, time.time()),
            )
            self._connection.commit()
            self.stats.writes += 1
            self.stats.bytes_stored += len(blob)

    def set_many(self, model: str, texts: list[str], vectors: list[list[float]]) -> int:
        """Store many vectors in one transaction; return the count written."""
        rows = [
            (self.key_for(model, text), model, len(vector), self._pack(vector), 0, time.time())
            for text, vector in zip(texts, vectors, strict=False)
            if vector
        ]
        if not rows:
            return 0
        with self._lock:
            self._connection.executemany(
                f"""INSERT OR REPLACE INTO {self.table}
                    (key, model, dim, vector, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self._connection.commit()
            self.stats.writes += len(rows)
        return len(rows)

    def delete_model(self, model: str) -> int:
        """Drop every cached vector for one model; return the count."""
        with self._lock:
            cursor = self._connection.execute(
                f"DELETE FROM {self.table} WHERE model = ?", (model,)
            )
            self._connection.commit()
            return cursor.rowcount or 0

    def clear(self) -> None:
        """Empty the cache table."""
        with self._lock:
            self._connection.execute(f"DELETE FROM {self.table}")
            self._connection.commit()
            self.stats.reset()

    def count(self) -> int:
        """Return how many vectors are cached."""
        with self._lock:
            row = self._connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()
            return int(row[0]) if row else 0

    def models(self) -> list[str]:
        """Return the distinct model names present in the cache."""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT DISTINCT model FROM {self.table} ORDER BY model"
            ).fetchall()
            return [str(row[0]) for row in rows]

    def vacuum(self) -> None:
        """Reclaim space in the database file."""
        with self._lock:
            self._connection.execute("VACUUM")

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    def stats_dict(self) -> dict[str, Any]:
        """Return cache telemetry including the row count."""
        data = self.stats.to_dict()
        data["rows"] = self.count()
        data["models"] = self.models()
        data["path"] = str(self.path)
        return data


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_TTL_SECONDS",
    "CacheStats",
    "DiskCache",
    "EmbeddingCache",
    "MemoryLRU",
    "ResponseCache",
    "SCHEMA_VERSION",
]
