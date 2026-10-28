"""Splitting documents into chunks that are worth retrieving.

Chunking decides retrieval quality more than the ranking math does, so the
strategies here are structure-aware rather than fixed-width:

``paragraph``
    Split on blank lines, merging until the size limit - the default, and the
    right choice for prose.
``markdown``
    Split on headings, keeping the heading path with each chunk so a retrieved
    fragment still says where it came from.
``sentence``
    Split on sentence boundaries for fine-grained factual lookup.
``code``
    Split a Python module on top-level definitions using its AST, so a chunk is
    a whole function or class instead of an arbitrary 1200 characters.
``fixed``
    Fixed size with overlap, for text with no usable structure.

Every chunk keeps its provenance (source, start offset, line span) so an answer
can be cited back to a file and a line range.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.files import read_text_safe
from ..utils.hashing import content_hash
from ..utils.text import chunk_text, ellipsize, normalize_newlines, sentence_split

STRATEGIES: tuple[str, ...] = ("paragraph", "markdown", "sentence", "code", "fixed")
"""Chunking strategies this module implements."""

DEFAULT_MAX_CHARS = 1200
"""Default chunk size in characters."""

DEFAULT_OVERLAP = 120
"""Default overlap between consecutive chunks, in characters."""

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
"""Markdown ATX heading."""

TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
        ".html",
        ".css",
        ".sh",
        ".sql",
        ".tex",
    }
)
"""File extensions treated as text for chunking."""


@dataclass(slots=True)
class Chunk:
    """One retrievable fragment of a document."""

    text: str
    source: str = ""
    index: int = 0
    start: int = 0
    end: int = 0
    start_line: int = 1
    end_line: int = 1
    heading: str = ""
    strategy: str = "paragraph"
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def id(self) -> str:
        """Return a stable identifier for this chunk."""
        return content_hash("chunk", self.source, self.start, self.text)[:16]

    @property
    def chars(self) -> int:
        """Return the chunk length."""
        return len(self.text)

    @property
    def location(self) -> str:
        """Return a human-readable citation location."""
        if not self.source:
            return f"chunk {self.index}"
        if self.start_line == self.end_line:
            return f"{self.source}:{self.start_line}"
        return f"{self.source}:{self.start_line}-{self.end_line}"

    def preview(self, limit: int = 120) -> str:
        """Return a one-line preview of the chunk."""
        return ellipsize(" ".join(self.text.split()), limit)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Serialize the chunk for an index file or an API response."""
        data: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "heading": self.heading,
            "strategy": self.strategy,
            "chars": self.chars,
            "metadata": dict(self.metadata),
            "score": round(self.score, 6),
        }
        if include_text:
            data["text"] = self.text
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Chunk:
        """Rebuild a chunk from a serialized dictionary."""
        return cls(
            text=str(raw.get("text", "")),
            source=str(raw.get("source", "")),
            index=int(raw.get("index", 0)),
            start=int(raw.get("start", 0)),
            end=int(raw.get("end", 0)),
            start_line=int(raw.get("start_line", 1)),
            end_line=int(raw.get("end_line", 1)),
            heading=str(raw.get("heading", "")),
            strategy=str(raw.get("strategy", "paragraph")),
            metadata=dict(raw.get("metadata") or {}),
        )


def _strip_leading_heading(body: str, headings: Sequence[str]) -> str:
    """Drop the section's own heading line when the heading trail is prepended.

    Without this the title appears twice in one chunk, which inflates its term
    frequency and makes short documents dominate BM25 rankings.
    """
    if not headings:
        return body
    first, _, rest = body.partition("\n")
    match = HEADING.match(first)
    if match and match.group(2).strip() == headings[-1]:
        return rest.strip()
    return body


def _line_of(text: str, offset: int) -> int:
    """Return the 1-based line number containing a character offset."""
    if offset <= 0:
        return 1
    return text.count("\n", 0, min(offset, len(text))) + 1


def chunk_text_blocks(
    text: str,
    *,
    source: str = "",
    strategy: str = "paragraph",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    keep_headings: bool = True,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split one string into :class:`Chunk` objects.

    Args:
        text: the document body
        source: file name or URL recorded on each chunk
        strategy: one of :data:`STRATEGIES`
        max_chars: target chunk size
        overlap: characters repeated between consecutive chunks
        keep_headings: prefix Markdown chunks with their heading path
        metadata: extra fields copied onto every chunk
    """
    body = normalize_newlines(text or "")
    if not body.strip():
        return []
    active = strategy if strategy in STRATEGIES else "paragraph"
    if active == "code" and source.endswith(".py"):
        return chunk_python(body, source=source, max_chars=max_chars, metadata=metadata)
    if active == "markdown":
        return chunk_markdown(
            body, source=source, max_chars=max_chars, keep_headings=keep_headings, metadata=metadata
        )
    if active == "sentence":
        pieces = _merge(sentence_split(body), max_chars)
    elif active == "fixed":
        pieces = [
            (match.start(), match.group(0))
            for match in re.finditer(rf"[\s\S]{{1,{max(64, max_chars)}}}", body)
        ]
        pieces = _add_overlap(body, pieces, overlap)
    else:
        pieces = _merge(_paragraph_spans(body), max_chars)
    chunks: list[Chunk] = []
    for index, (start, piece) in enumerate(pieces):
        end = start + len(piece)
        chunks.append(
            Chunk(
                text=piece.strip(),
                source=source,
                index=index,
                start=start,
                end=end,
                start_line=_line_of(body, start),
                end_line=max(_line_of(body, end - 1), _line_of(body, start)),
                strategy=active,
                metadata=dict(metadata or {}),
            )
        )
    return [chunk for chunk in chunks if chunk.text]


def _paragraph_spans(text: str) -> list[tuple[int, str]]:
    """Return ``(offset, paragraph)`` pairs for a blank-line separated document."""
    spans: list[tuple[int, str]] = []
    position = 0
    for block in re.split(r"\n\s*\n", text):
        index = text.find(block, position)
        start = index if index >= 0 else position
        spans.append((start, block))
        position = start + len(block)
    return [(start, block) for start, block in spans if block.strip()]


def _merge(
    spans: Sequence[tuple[int, str]] | Sequence[str], max_chars: int
) -> list[tuple[int, str]]:
    """Merge small pieces until each is close to ``max_chars``."""
    normalized: list[tuple[int, str]] = []
    for item in spans:
        if isinstance(item, tuple):
            normalized.append(item)
        else:
            normalized.append((0, item))
    merged: list[tuple[int, str]] = []
    buffer_start, buffer = -1, ""
    for start, piece in normalized:
        if not piece.strip():
            continue
        if buffer and len(buffer) + len(piece) + 2 > max_chars:
            merged.append((buffer_start, buffer))
            buffer_start, buffer = start, piece.strip()
        else:
            if buffer_start < 0:
                buffer_start = start
            buffer = f"{buffer}\n\n{piece.strip()}" if buffer else piece.strip()
    if buffer:
        merged.append((buffer_start, buffer))
    if not merged and normalized:
        return [(0, " ".join(piece.strip() for _, piece in normalized))]
    return merged


def _add_overlap(text: str, spans: list[tuple[int, str]], overlap: int) -> list[tuple[int, str]]:
    """Repeat the tail of each span at the head of the next one."""
    if overlap <= 0 or len(spans) < 2:
        return spans
    out: list[tuple[int, str]] = [spans[0]]
    for index in range(1, len(spans)):
        start, piece = spans[index]
        tail = text[max(0, start - overlap) : start]
        out.append((max(0, start - overlap), (tail + piece) if tail else piece))
    return out


def chunk_markdown(
    text: str,
    *,
    source: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    keep_headings: bool = True,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split Markdown on headings, carrying the heading path into each chunk."""
    lines = normalize_newlines(text or "").split("\n")
    sections: list[tuple[int, str, list[str]]] = []
    path: list[str] = []
    current: list[str] = []
    section_start = 0
    offset = 0
    offsets: list[int] = []
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
        match = HEADING.match(line)
        if match:
            if current:
                sections.append((section_start, "\n".join(current), list(path)))
            level = len(match.group(1))
            title = match.group(2).strip()
            del path[level - 1 :]
            path.append(title)
            current = [line]
            section_start = offsets[-1]
        else:
            current.append(line)
    if current:
        sections.append((section_start, "\n".join(current), list(path)))
    chunks: list[Chunk] = []
    for index, (start, body, headings) in enumerate(sections):
        if not body.strip():
            continue
        trail = " > ".join(headings)
        body_text = _strip_leading_heading(body, headings)
        rendered = f"{trail}\n\n{body_text}" if (keep_headings and trail) else body
        for piece_index, piece in enumerate(chunk_text(rendered, max_chars=max_chars, overlap=0)):
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source=source,
                    index=index * 100 + piece_index,
                    start=start,
                    end=min(len(text or ""), start + len(body)),
                    start_line=_line_of(text, start),
                    end_line=_line_of(text, min(len(text), start + len(body))),
                    heading=trail,
                    strategy="markdown",
                    metadata=dict(metadata or {}),
                )
            )
    return [chunk for chunk in chunks if chunk.text]


def _safe_docstring(node: ast.AST) -> str:
    """Return a node's docstring, tolerating nodes that cannot carry one."""
    try:
        return ast.get_docstring(node) or ""
    except (TypeError, ValueError):  # ast.Expr / non-docstring first statement
        return ""


def chunk_python(
    source_code: str,
    *,
    source: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split a Python module into one chunk per top-level definition.

    Falls back to paragraph chunking when the file does not parse, so a syntax
    error never makes a file unsearchable.
    """
    text = normalize_newlines(source_code or "")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text_blocks(
            text, source=source, strategy="paragraph", max_chars=max_chars, metadata=metadata
        )
    lines = text.split("\n")
    chunks: list[Chunk] = []
    for index, node in enumerate(tree.body):
        if not hasattr(node, "lineno"):
            continue
        end_line = getattr(node, "end_lineno", node.lineno) or node.lineno
        body = "\n".join(lines[node.lineno - 1 : end_line])
        if not body.strip():
            continue
        name = getattr(node, "name", "") or type(node).__name__
        docstring = _safe_docstring(node)
        for piece_index, piece in enumerate(chunk_text(body, max_chars=max_chars, overlap=0)):
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source=source,
                    index=index * 100 + piece_index,
                    start=text.find(body),
                    end=text.find(body) + len(body),
                    start_line=node.lineno,
                    end_line=end_line,
                    heading=name,
                    strategy="code",
                    metadata={
                        **(metadata or {}),
                        "symbol": name,
                        "symbol_type": type(node).__name__,
                        "docstring": ellipsize(docstring, 300),
                    },
                )
            )
    return [chunk for chunk in chunks if chunk.text]


def chunk_file(
    path: str | Path,
    *,
    strategy: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    limit: int = 2_000_000,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Read one file and chunk it, choosing a strategy from the extension."""
    target = Path(path).expanduser()
    if not target.is_file():
        return []
    text = read_text_safe(target, limit=limit)
    if not text.strip():
        return []
    active = strategy
    if not active:
        active = (
            "markdown"
            if target.suffix.lower() in {".md", ".markdown", ".rst"}
            else ("code" if target.suffix.lower() == ".py" else "paragraph")
        )
    return chunk_text_blocks(
        text,
        source=str(target),
        strategy=active,
        max_chars=max_chars,
        overlap=overlap,
        metadata={
            **(metadata or {}),
            "suffix": target.suffix.lower(),
            "bytes": target.stat().st_size,
        },
    )


def chunk_files(
    paths: Iterable[str | Path],
    *,
    strategy: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    max_files: int = 500,
    limit: int = 2_000_000,
) -> tuple[list[Chunk], list[str]]:
    """Chunk many files; return the chunks and the paths that were skipped.

    Args:
        paths: files to read
        strategy: force one strategy, empty to pick per file
        max_chars: target chunk size
        overlap: overlap between chunks
        max_files: stop after this many files, so a stray home directory does
            not become an indexing job
        limit: per-file byte ceiling
    """
    chunks: list[Chunk] = []
    skipped: list[str] = []
    for position, path in enumerate(paths):
        if position >= max_files:
            skipped.append(str(path))
            continue
        target = Path(path).expanduser()
        if not target.is_file():
            skipped.append(str(target))
            continue
        if target.suffix.lower() not in TEXT_SUFFIXES and strategy != "fixed":
            skipped.append(str(target))
            continue
        chunks.extend(
            chunk_file(target, strategy=strategy, max_chars=max_chars, overlap=overlap, limit=limit)
        )
    return chunks, skipped


def choose_strategy(path: str | Path) -> str:
    """Return the chunking strategy that fits a file's extension."""
    suffix = Path(path).suffix.lower()
    if suffix in {".md", ".markdown", ".rst"}:
        return "markdown"
    if suffix == ".py":
        return "code"
    if suffix in TEXT_SUFFIXES:
        return "paragraph"
    return "fixed"


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP",
    "HEADING",
    "STRATEGIES",
    "TEXT_SUFFIXES",
    "Chunk",
    "choose_strategy",
    "chunk_file",
    "chunk_files",
    "chunk_markdown",
    "chunk_python",
    "chunk_text_blocks",
]
