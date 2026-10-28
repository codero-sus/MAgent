"""Append-only audit log for guardrail decisions.

Every verdict - allowed, warned, rewritten, or blocked - is written as one JSON
line so an operator can answer "what did the guardrails do during that run?"
without re-running anything. Secrets are scrubbed on the way in, the file is
size-capped with rotation, and reading never requires the writer to be running.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from ..secrets import redact_text
from ..types import new_id
from ..utils.files import atomic_write_text, ensure_dir, human_size

LOGGER = get_logger("guardrails.audit")
"""Logger for audit write failures, which must never break a run."""

DEFAULT_MAX_BYTES = 8 * 1024 * 1024
"""Default size cap before the log rotates."""

DEFAULT_KEEP_ROTATED = 3
"""How many rotated files to keep."""


class AuditLog:
    """A JSONL audit trail with rotation and redaction."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        enabled: bool = True,
        redact: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        keep_rotated: int = DEFAULT_KEEP_ROTATED,
        serializer: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Open an audit log.

        Args:
            path: JSONL file; ``None`` keeps entries in memory only
            enabled: when false, nothing is recorded
            redact: scrub secrets from every entry before writing
            max_bytes: size at which the log rotates
            keep_rotated: rotated files to retain
            serializer: custom line serializer, defaults to compact JSON
        """
        self.path = Path(path).expanduser() if path else None
        self.enabled = bool(enabled)
        self.redact = bool(redact)
        self.max_bytes = max(1024, int(max_bytes))
        self.keep_rotated = max(0, int(keep_rotated))
        self.serializer = serializer
        self._lock = threading.RLock()
        self._memory: list[dict[str, Any]] = []
        self.recorded = 0
        self.skipped = 0
        self.rotations = 0

    # -- writing -------------------------------------------------------------

    def record(
        self,
        kind: str,
        data: dict[str, Any] | None = None,
        *,
        agent: str = "",
        run_id: str = "",
        severity: str = "info",
    ) -> dict[str, Any]:
        """Append one audit entry and return it."""
        if not self.enabled:
            self.skipped += 1
            return {}
        entry = {
            "id": new_id("aud"),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": str(kind),
            "severity": str(severity),
            "agent": agent,
            "run_id": run_id,
            "data": dict(data or {}),
        }
        if self.redact:
            entry["data"] = _redact_structure(entry["data"])
        with self._lock:
            self._memory.append(entry)
            if len(self._memory) > 2000:
                del self._memory[: len(self._memory) - 2000]
            self.recorded += 1
            if self.path is not None:
                self._append(entry)
        return entry

    def _append(self, entry: dict[str, Any]) -> None:
        """Write one entry to disk, rotating first when the file is too large."""
        if self.path is None:
            return
        try:
            ensure_dir(self.path.parent)
            if self.path.is_file() and self.path.stat().st_size >= self.max_bytes:
                self.rotate()
            line = (
                self.serializer(entry)
                if self.serializer
                else json.dumps(entry, ensure_ascii=False, default=str, separators=(",", ":"))
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError as exc:
            self.skipped += 1
            LOGGER.warning("could not write audit entry to %s: %s", self.path, exc)

    def rotate(self, *, suffix: str = "") -> Path | None:
        """Move the current log aside and start a fresh one."""
        if self.path is None or not self.path.is_file():
            return None
        stamp = suffix or time.strftime("%Y%m%d-%H%M%S")
        target = self.path.with_suffix(f".{stamp}.log")
        with self._lock:
            try:
                atomic_write_text(target, self.path.read_text(encoding="utf-8", errors="replace"))
                atomic_write_text(self.path, "")
            except OSError as exc:
                LOGGER.warning("audit rotation failed: %s", exc)
                return None
            self.rotations += 1
            self._prune_rotated()
        return target

    def _prune_rotated(self) -> None:
        """Delete the oldest rotated logs beyond ``keep_rotated``."""
        if self.path is None or self.keep_rotated <= 0:
            return
        rotated = sorted(
            self.path.parent.glob(self.path.stem + ".*.log"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in rotated[self.keep_rotated :]:
            try:
                stale.unlink()
            except OSError:
                continue

    def record_many(self, entries: Iterable[tuple[str, dict[str, Any]]], **kwargs: Any) -> int:
        """Record several ``(kind, data)`` pairs."""
        return sum(1 for kind, data in entries if self.record(kind, data, **kwargs))

    # -- reading -------------------------------------------------------------

    def entries(
        self, *, limit: int = 100, kind: str = "", severity: str = "", since: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return recorded entries, newest last, filtered as requested."""
        raw: list[dict[str, Any]] = []
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            raw = list(self._memory)
        selected = [
            entry
            for entry in raw
            if (not kind or entry.get("kind") == kind)
            and (not severity or entry.get("severity") == severity)
            and (not since or float(entry.get("ts") or 0) >= since)
        ]
        return selected[-max(1, limit) :]

    def tail(self, count: int = 10) -> list[dict[str, Any]]:
        """Return the most recent entries."""
        return self.entries(limit=count)

    def find(
        self, predicate: Callable[[dict[str, Any]], bool], *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return entries matching a predicate."""
        return [entry for entry in self.entries(limit=100000) if predicate(entry)][: max(1, limit)]

    def count(self) -> int:
        """Return how many entries are on disk (or in memory)."""
        if self.path is not None and self.path.is_file():
            with self.path.open("r", encoding="utf-8", errors="replace") as stream:
                return sum(1 for line in stream if line.strip())
        return len(self._memory)

    def size(self) -> int:
        """Return the log file size in bytes."""
        if self.path is not None and self.path.is_file():
            return self.path.stat().st_size
        return 0

    def clear(self) -> int:
        """Empty the log; return how many entries were removed."""
        removed = self.count()
        with self._lock:
            self._memory.clear()
            if self.path is not None and self.path.is_file():
                atomic_write_text(self.path, "")
        return removed

    def export(self, path: str | Path, *, kind: str = "") -> Path:
        """Copy the log to another file, optionally filtered by kind."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        entries = self.entries(limit=1000000, kind=kind)
        atomic_write_text(
            target,
            "".join(json.dumps(entry, ensure_ascii=False, default=str) + "\n" for entry in entries),
        )
        return target

    def stats(self) -> dict[str, Any]:
        """Return audit telemetry."""
        kinds: dict[str, int] = {}
        for entry in self.entries(limit=100000):
            name = str(entry.get("kind", "?"))
            kinds[name] = kinds.get(name, 0) + 1
        return {
            "path": str(self.path) if self.path else None,
            "enabled": self.enabled,
            "recorded": self.recorded,
            "skipped": self.skipped,
            "rotations": self.rotations,
            "entries": self.count(),
            "bytes": self.size(),
            "size_human": human_size(self.size()),
            "by_kind": kinds,
        }

    def __len__(self) -> int:
        """Return the entry count."""
        return self.count()

    def __repr__(self) -> str:
        """Show the path and entry count."""
        return f"<AuditLog path={self.path} entries={self.count()} enabled={self.enabled}>"


def _redact_structure(value: Any) -> Any:
    """Recursively scrub secrets from a structure before it is written."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): _redact_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_structure(item) for item in value]
    return value


__all__ = ["DEFAULT_KEEP_ROTATED", "DEFAULT_MAX_BYTES", "AuditLog"]
