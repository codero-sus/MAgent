"""Workspace context: turn the files around the agent into prompt material.

Two things come out of here:

:func:`describe_workspace`
    a compact tree plus size/age facts, cheap enough to include on every run;
:func:`file_context_blocks`
    the contents of the handful of files that actually matter - ones the user
    named, ones matching the query, and repository conventions such as
    ``AGENTS.md`` or ``README.md``.

Everything goes through the tool sandbox, so context assembly cannot be used to
read a path the agent would otherwise be refused.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..llm.tokens import estimate_tokens
from ..logging_ import get_logger
from ..utils.files import human_size, iter_files, read_text_safe
from ..utils.mime import is_binary
from ..utils.text import ellipsize
from .block import ContextBlock, files_block

LOGGER = get_logger("context.files")
"""Logger for workspace scanning."""

CONVENTION_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    "MAGENT.md",
    ".magent/instructions.md",
    "CONTRIBUTING.md",
    "README.md",
)
"""Files that describe project conventions, read automatically when present."""

DEFAULT_PATTERNS: tuple[str, ...] = ("*.md", "*.txt", "*.py", "*.json", "*.toml", "*.yaml", "*.yml")
"""Extensions considered for query-driven context."""

SKIP_DIRS: tuple[str, ...] = (".git", "__pycache__", "node_modules", ".venv", "venv", "dist",
                              "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox")
"""Directories never walked."""

MAX_FILE_CHARS = 6_000
"""Hard ceiling on characters read from one file."""

MAX_FILES_SCANNED = 2_000
"""Ceiling on files walked, so a huge tree cannot stall a run."""


@dataclass(slots=True)
class FileRef:
    """One workspace file considered for context."""

    path: Path
    relative: str
    size: int = 0
    modified: float = 0.0
    tokens: int = 0
    text: str = ""
    reason: str = ""
    score: float = 0.0
    included: bool = False
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Seconds since the file was last modified."""
        return max(0.0, time.time() - self.modified) if self.modified else 0.0

    def header(self) -> str:
        """One-line description used in the workspace listing."""
        age = _relative_age(self.age_seconds)
        return f"{self.relative} ({human_size(self.size)}, {age})"

    def to_block(self, *, max_tokens: int = 900, title: str = "") -> ContextBlock:
        """Render the file as a context block."""
        heading = title or f"File: {self.relative}"
        body = self.text.strip()
        if not body:
            return ContextBlock(title=heading, body=f"(empty or unreadable: {self.error or 'no text'})",
                                kind="files", priority=3, source=str(self.path))
        tokens = estimate_tokens(body)
        if tokens > max_tokens:
            body = ellipsize(body, max_tokens * 4)
        return ContextBlock(title=heading, body=body, kind="files", priority=4,
                            source=str(self.path), max_tokens=max_tokens,
                            metadata={"reason": self.reason, "score": round(self.score, 4),
                                      "path": self.relative})

    def to_dict(self) -> dict[str, Any]:
        """Serialise the reference without the file text."""
        return {
            "path": self.relative,
            "size": self.size,
            "modified": self.modified,
            "tokens": self.tokens,
            "reason": self.reason,
            "score": round(self.score, 4),
            "included": self.included,
            "error": self.error,
        }


@dataclass(slots=True)
class WorkspaceSnapshot:
    """Result of scanning a workspace."""

    root: Path
    files: list[FileRef] = field(default_factory=list)
    directories: int = 0
    total_bytes: int = 0
    scanned: int = 0
    seconds: float = 0.0
    conventions: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def count(self) -> int:
        """Number of files found."""
        return len(self.files)

    def listing(self, *, limit: int = 60, patterns: Sequence[str] = ()) -> str:
        """Render a short tree-style listing of the workspace."""
        selected = [item for item in self.files
                    if not patterns or any(item.relative.endswith(p.lstrip("*")) or
                                           Path(item.relative).suffix == p.lstrip("*")
                                           for p in patterns)]
        selected = selected[: max(1, limit)]
        lines = [item.header() for item in selected]
        if self.truncated or len(lines) < len(self.files):
            lines.append(f"... {self.count - len(lines)} more files")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the snapshot metadata."""
        return {
            "root": str(self.root),
            "files": self.count,
            "directories": self.directories,
            "total_bytes": self.total_bytes,
            "scanned": self.scanned,
            "seconds": round(self.seconds, 4),
            "conventions": list(self.conventions),
            "truncated": self.truncated,
        }


def _relative_age(seconds: float) -> str:
    """Format an age in seconds as a short human phrase."""
    if seconds < 60:
        return "just now"
    if seconds < 3_600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86_400:
        return f"{int(seconds // 3_600)}h ago"
    if seconds < 86_400 * 30:
        return f"{int(seconds // 86_400)}d ago"
    return f"{int(seconds // (86_400 * 30))}mo ago"


def scan_workspace(root: str | Path, *, patterns: Iterable[str] = ("*",),
                   max_files: int = MAX_FILES_SCANNED, max_depth: int | None = 4,
                   sandbox: Any = None) -> WorkspaceSnapshot:
    """Walk a workspace and collect file metadata without reading contents.

    Args:
        root: directory to scan
        patterns: glob patterns to keep
        max_files: stop after this many files
        max_depth: how deep to descend
        sandbox: optional sandbox; paths it refuses are skipped
    """
    started = time.perf_counter()
    base = Path(root or ".").expanduser()
    snapshot = WorkspaceSnapshot(root=base)
    if not base.exists():
        LOGGER.debug("workspace %s does not exist", base)
        snapshot.seconds = time.perf_counter() - started
        return snapshot
    dirs: set[str] = set()
    for path in iter_files(base, patterns=tuple(patterns), exclude=SKIP_DIRS,
                           max_depth=max_depth, limit=max_files + 1):
        if snapshot.scanned >= max_files:
            snapshot.truncated = True
            break
        snapshot.scanned += 1
        relative = _relative_to(base, path)
        if sandbox is not None:
            try:
                decision = sandbox.check_read(path)
            except Exception as exc:  # noqa: BLE001 - a broken sandbox must not break scanning
                LOGGER.debug("sandbox check failed for %s: %s", path, exc)
                continue
            if not getattr(decision, "allowed", True):
                continue
        try:
            stat = path.stat()
        except OSError as exc:
            LOGGER.debug("stat failed for %s: %s", path, exc)
            continue
        if is_binary(path):
            continue
        dirs.update(part for part in Path(relative).parts[:-1])
        snapshot.total_bytes += stat.st_size
        snapshot.files.append(FileRef(path=path, relative=relative, size=stat.st_size,
                                      modified=stat.st_mtime))
    snapshot.directories = len(dirs)
    for name in CONVENTION_FILES:
        candidate = base / name
        if candidate.is_file():
            snapshot.conventions.append(name)
    snapshot.files.sort(key=lambda item: (-item.size, item.relative))
    snapshot.seconds = time.perf_counter() - started
    return snapshot


def _relative_to(base: Path, path: Path) -> str:
    """Return ``path`` relative to ``base``, falling back to the absolute path."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def read_file_ref(ref: FileRef, *, max_chars: int = MAX_FILE_CHARS,
                  sandbox: Any = None) -> FileRef:
    """Read one file into its reference, recording any failure."""
    if sandbox is not None:
        try:
            decision = sandbox.check_read(ref.path)
        except Exception as exc:  # noqa: BLE001
            ref.error = f"sandbox error: {exc}"
            return ref
        if not getattr(decision, "allowed", False):
            ref.error = getattr(decision, "reason", "denied by sandbox")
            return ref
    try:
        text = read_text_safe(ref.path, limit=max_chars)
    except Exception as exc:  # noqa: BLE001 - unreadable files are reported, not raised
        ref.error = str(exc)
        return ref
    ref.text = text
    ref.tokens = estimate_tokens(text)
    return ref


def rank_files(refs: Iterable[FileRef], query: str, *, limit: int = 5) -> list[FileRef]:
    """Score file references against a query by name and content overlap.

    This is deliberately cheap - a keyword overlap, not an embedding - because
    it runs on every turn to decide which open files are worth quoting.
    """
    terms = {term for term in query.lower().replace("?", " ").replace(",", " ").split()
             if len(term) > 2}
    scored: list[FileRef] = []
    for ref in refs:
        if not terms:
            continue
        name_terms = set(Path(ref.relative).stem.lower().replace("_", " ").replace("-", " ").split())
        haystack = f"{ref.relative}\n{ref.text[:4_000]}".lower()
        hits = sum(1 for term in terms if term in haystack)
        if not hits:
            continue
        score = hits / len(terms)
        score += 0.15 * len(name_terms & terms)
        if ref.age_seconds < 86_400:
            score += 0.05
        ref.score = round(score, 4)
        scored.append(ref)
    scored.sort(key=lambda item: (-item.score, -item.tokens))
    return scored[: max(1, limit)]


def file_context_blocks(root: str | Path, *, query: str = "", named: Sequence[str] = (),
                        include_conventions: bool = True, limit: int = 4,
                        max_chars: int = MAX_FILE_CHARS, max_tokens: int = 900,
                        sandbox: Any = None) -> tuple[list[ContextBlock], dict[str, Any]]:
    """Build context blocks from the files most relevant to a turn.

    Args:
        root: workspace root
        query: the user's text, used to rank candidate files
        named: paths the user explicitly mentioned; always included
        include_conventions: read ``README.md``/``AGENTS.md`` style files
        limit: how many query-ranked files to include
        max_chars: characters read per file
        max_tokens: token ceiling per block
        sandbox: optional sandbox used to authorize every read

    Returns:
        the blocks plus stats about what was scanned
    """
    base = Path(root or ".").expanduser()
    blocks: list[ContextBlock] = []
    stats: dict[str, Any] = {"root": str(base), "named": [], "conventions": [], "ranked": []}
    if not base.is_dir():
        return blocks, stats

    for name in named:
        candidate = (base / name).expanduser() if not Path(name).is_absolute() else Path(name)
        if not candidate.is_file():
            continue
        ref = read_file_ref(FileRef(path=candidate, relative=_relative_to(base, candidate),
                                    size=candidate.stat().st_size,
                                    modified=candidate.stat().st_mtime, reason="named by user"),
                            max_chars=max_chars, sandbox=sandbox)
        ref.included = True
        blocks.append(ref.to_block(max_tokens=max_tokens))
        stats["named"].append(ref.to_dict())

    if include_conventions:
        for name in CONVENTION_FILES:
            candidate = base / name
            if not candidate.is_file():
                continue
            ref = read_file_ref(FileRef(path=candidate, relative=name,
                                        size=candidate.stat().st_size,
                                        modified=candidate.stat().st_mtime,
                                        reason="project convention"),
                                max_chars=min(max_chars, 4_000), sandbox=sandbox)
            ref.included = True
            blocks.append(ref.to_block(max_tokens=min(max_tokens, 500),
                                       title=f"Project conventions: {name}"))
            stats["conventions"].append(ref.to_dict())

    if query.strip():
        snapshot = scan_workspace(base, patterns=DEFAULT_PATTERNS, max_files=400, max_depth=4,
                                  sandbox=sandbox)
        already = {str(item.metadata.get("path", "")) for item in blocks}
        already |= {str(block.source) for block in blocks if block.source}
        candidates: list[FileRef] = []
        for ref in snapshot.files:
            if str(ref.path) in already or ref.relative in already:
                continue
            read_file_ref(ref, max_chars=min(max_chars, 4_000), sandbox=sandbox)
            candidates.append(ref)
        for ref in rank_files(candidates, query, limit=limit):
            ref.reason = "matches query"
            ref.included = True
            blocks.append(ref.to_block(max_tokens=max_tokens))
            stats["ranked"].append(ref.to_dict())
        stats["scanned"] = snapshot.scanned

    stats["blocks"] = len(blocks)
    stats["tokens"] = sum(item.tokens for item in blocks)
    return blocks, stats


def describe_workspace(root: str | Path, *, limit: int = 40, include_listing: bool = True,
                       sandbox: Any = None) -> str:
    """Return a short textual description of a workspace for the system prompt."""
    base = Path(root or ".").expanduser()
    if not base.is_dir():
        return f"Workspace: {base} (missing)"
    snapshot = scan_workspace(base, patterns=("*",), max_files=400, max_depth=3, sandbox=sandbox)
    lines = [
        f"Workspace: {base}",
        f"Files: {snapshot.count} scanned ({snapshot.scanned}), "
        f"{human_size(snapshot.total_bytes)}, {snapshot.directories} directories",
    ]
    if snapshot.conventions:
        lines.append("Convention files: " + ", ".join(snapshot.conventions))
    if include_listing and snapshot.files:
        lines.append("Largest files:")
        lines.extend(f"  {item.header()}" for item in snapshot.files[:limit])
    lines.append(f"Scan time: {snapshot.seconds * 1000:.1f} ms")
    return "\n".join(lines)


def open_editor_files(*, limit: int = 10) -> list[str]:
    """Best-effort list of files open in common editors (used by ``include_files``).

    Reads no editor state that is not already on disk; returns an empty list
    when nothing can be determined.
    """
    found: list[str] = []
    home = Path(os.path.expanduser("~"))
    candidates = [
        home / ".local/state/nvim/swap",
        home / ".viminfo",
        home / ".local/share/recently-used.xbel",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name.endswith(".xbel"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _href_re().finditer(text):
                target = match.group(1)
                if target.startswith("file://") and len(found) < limit:
                    found.append(target[len("file://"):])
        elif path.is_dir():
            for swap in sorted(path.glob("*.sw?"), key=lambda p: -p.stat().st_mtime)[:limit]:
                found.append(swap.name)
    return found[:limit]


def _href_re() -> Any:
    """Compile the href matcher lazily so module import stays cheap."""
    import re

    return re.compile(r'href="([^"]+)"')


def files_listing_block(root: str | Path, *, limit: int = 40, sandbox: Any = None,
                        priority: int = 4) -> ContextBlock:
    """Build a block listing workspace files with sizes and ages."""
    snapshot = scan_workspace(root, max_files=400, max_depth=3, sandbox=sandbox)
    return files_block([item.header() for item in snapshot.files[:limit]],
                       title="Workspace files", priority=priority,
                       metadata={"scanned": snapshot.scanned})


def touched_recently(root: str | Path, *, hours: float = 24.0, limit: int = 10,
                     sandbox: Any = None) -> list[FileRef]:
    """Return files modified within the last ``hours``."""
    snapshot = scan_workspace(root, max_files=MAX_FILES_SCANNED, max_depth=5, sandbox=sandbox)
    cutoff = time.time() - hours * 3_600
    recent = [item for item in snapshot.files if item.modified >= cutoff]
    recent.sort(key=lambda item: -item.modified)
    return recent[: max(1, limit)]


__all__ = [
    "CONVENTION_FILES",
    "DEFAULT_PATTERNS",
    "MAX_FILE_CHARS",
    "MAX_FILES_SCANNED",
    "SKIP_DIRS",
    "FileRef",
    "WorkspaceSnapshot",
    "describe_workspace",
    "file_context_blocks",
    "files_listing_block",
    "open_editor_files",
    "rank_files",
    "read_file_ref",
    "scan_workspace",
    "touched_recently",
]
