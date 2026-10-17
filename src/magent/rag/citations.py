"""Citations: turning retrieved chunks into numbered, verifiable references.

An answer that says "the config lives in ``config.py``" is uncheckable. An
answer that says "the config lives in ``config.py`` [3]" points at a chunk the
caller can open. This module builds the numbered context block the model sees,
extracts the citation markers from its answer, and verifies that every marker
resolves to a real source - which is how the evaluation harness scores
groundedness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..utils.text import ellipsize
from .chunker import Chunk

CITATION = re.compile(r"\[(\d{1,3})\]")
"""A bracketed citation marker such as ``[3]``."""

MAX_CONTEXT_CHARS = 12_000
"""Default ceiling for a rendered context block."""


@dataclass(slots=True)
class Citation:
    """One numbered reference to a retrieved chunk."""

    number: int
    source: str
    heading: str = ""
    start_line: int = 0
    end_line: int = 0
    chunk_id: str = ""
    preview: str = ""
    score: float = 0.0

    def label(self) -> str:
        """Return the human-readable label, e.g. ``docs/guide.md:12-18``."""
        if not self.source:
            return f"[{self.number}]"
        if self.start_line and self.end_line and self.start_line != self.end_line:
            return f"{self.source}:{self.start_line}-{self.end_line}"
        if self.start_line:
            return f"{self.source}:{self.start_line}"
        return self.source

    def to_dict(self) -> dict[str, Any]:
        """Serialize the citation."""
        return {
            "number": self.number,
            "label": self.label(),
            "source": self.source,
            "heading": self.heading,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_id": self.chunk_id,
            "preview": self.preview,
            "score": round(self.score, 6),
        }

    def __str__(self) -> str:
        """Render the citation as it appears in a reference list."""
        head = f" ({self.heading})" if self.heading else ""
        return f"[{self.number}] {self.label()}{head}: {self.preview}"


@dataclass(slots=True)
class CitationSet:
    """The numbered references for one retrieval, plus the context block."""

    citations: list[Citation] = field(default_factory=list)
    context: str = ""
    style: str = "numbered"

    def by_number(self, number: int) -> Citation | None:
        """Return one citation by its number."""
        return next((item for item in self.citations if item.number == number), None)

    def labels(self) -> list[str]:
        """Return every label, for a footer or a UI tooltip."""
        return [item.label() for item in self.citations]

    def sources(self) -> list[str]:
        """Return the distinct source paths."""
        return list(dict.fromkeys(item.source for item in self.citations if item.source))

    def render(self, *, include_context: bool = True) -> str:
        """Render the reference list, optionally preceded by the context block."""
        references = "\n".join(str(item) for item in self.citations)
        if not include_context:
            return f"References:\n{references}" if references else ""
        body = self.context or ""
        return f"{body}\n\nReferences:\n{references}".strip() if references else body

    def to_dict(self) -> dict[str, Any]:
        """Serialize the set for an API response or transcript."""
        return {
            "style": self.style,
            "count": len(self.citations),
            "citations": [item.to_dict() for item in self.citations],
            "sources": self.sources(),
            "chars": len(self.context),
            "context": self.context,
        }


def build_citations(chunks: Sequence[Chunk], *, preview_chars: int = 200) -> list[Citation]:
    """Number a sequence of chunks, starting at 1."""
    return [
        Citation(
            number=index,
            source=chunk.source,
            heading=chunk.heading,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_id=chunk.id,
            preview=chunk.preview(preview_chars),
            score=chunk.score,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def render_context(chunks: Sequence[Chunk], *, style: str = "numbered",
                   max_chars: int = MAX_CONTEXT_CHARS, include_heading: bool = True,
                   instruction: str = "") -> str:
    """Render chunks into a context block for a prompt.

    Args:
        chunks: the retrieved chunks, in rank order
        style: ``numbered`` for cited retrieval, ``plain`` for a plain digest,
            ``markdown`` for a document-shaped block
        max_chars: ceiling for the whole block
        include_heading: prepend a heading that tells the model what this is
        instruction: extra instruction appended after the block
    """
    if not chunks:
        return ""
    header = include_heading and style == "numbered"
    lines: list[str] = []
    if header:
        lines.append(
            "Retrieved context. Cite the sources you use with the bracketed number, "
            "for example [2]. If the context does not answer the question, say so."
        )
        lines.append("")
    budget = max_chars - sum(len(line) + 1 for line in lines)
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.location
        heading = f" - {chunk.heading}" if (include_heading and chunk.heading) else ""
        prefix = f"[{index}] {location}{heading}" if style == "numbered" else f"--- {location}{heading}"
        body = chunk.text.strip()
        remaining = budget - len(prefix) - 4
        if remaining <= 80:
            lines.append(f"{prefix}\n  (truncated to fit the context budget)")
            break
        lines.append(prefix)
        lines.append(ellipsize(body, remaining) if style != "markdown" else _fence(chunk, remaining))
        lines.append("")
        budget -= len(prefix) + len(body) + 4
    if instruction:
        lines.append(instruction)
    return "\n".join(lines).strip()[:max_chars]


def _fence(chunk: Chunk, limit: int) -> str:
    """Fence a code chunk so the model does not confuse it with prose."""
    language = "python" if chunk.source.endswith(".py") else ""
    body = ellipsize(chunk.text.strip(), max(40, limit - 12))
    return f"```{language}\n{body}\n```"


def cited_context(chunks: Sequence[Chunk], *, max_chars: int = MAX_CONTEXT_CHARS,
                  preview_chars: int = 200, instruction: str = "") -> CitationSet:
    """Build a numbered context block and its citation list together."""
    citations = build_citations(chunks, preview_chars=preview_chars)
    context = render_context(chunks, style="numbered", max_chars=max_chars, instruction=instruction)
    return CitationSet(citations=citations, context=context, style="numbered")


def extract_citations(text: str) -> list[int]:
    """Return the citation numbers used in an answer, in order of appearance."""
    return [int(match.group(1)) for match in CITATION.finditer(text or "")]


def citation_counts(text: str) -> dict[int, int]:
    """Count how often each citation number appears."""
    counts: dict[int, int] = {}
    for number in extract_citations(text):
        counts[number] = counts.get(number, 0) + 1
    return counts


@dataclass(slots=True)
class CitationCheck:
    """The result of verifying an answer's citations against its sources."""

    used: list[int] = field(default_factory=list)
    valid: list[int] = field(default_factory=list)
    dangling: list[int] = field(default_factory=list)
    unused: list[int] = field(default_factory=list)
    coverage: float = 0.0
    grounded: bool = True

    @property
    def ok(self) -> bool:
        """Report whether every used citation resolves to a real source."""
        return not self.dangling

    def to_dict(self) -> dict[str, Any]:
        """Serialize the check for evaluation output."""
        return {
            "used": self.used,
            "valid": self.valid,
            "dangling": self.dangling,
            "unused": self.unused,
            "coverage": round(self.coverage, 4),
            "grounded": self.grounded,
            "ok": self.ok,
        }

    def summary(self) -> str:
        """Return a one-line human summary."""
        if self.ok and self.used:
            return f"{len(self.used)} citation(s) verified against {len(self.valid)} source(s)"
        if self.dangling:
            return f"{len(self.dangling)} dangling citation(s): {self.dangling}"
        if not self.used:
            return "no citations in the answer"
        return f"{len(self.used)} citation(s), coverage {self.coverage:.0%}"


def verify_citations(text: str, citations: Sequence[Citation]) -> CitationCheck:
    """Check an answer's citations against the sources it was given.

    ``coverage`` is the fraction of supplied sources the answer actually cited,
    which distinguishes "answered from the context" from "answered from memory
    and cited nothing".
    """
    used = extract_citations(text)
    known = {item.number for item in citations}
    valid = sorted({number for number in used if number in known})
    dangling = sorted({number for number in used if number not in known})
    unused = sorted(known - set(used))
    coverage = len(valid) / len(known) if known else 0.0
    return CitationCheck(used=sorted(set(used)), valid=valid, dangling=dangling, unused=unused,
                         coverage=coverage, grounded=not dangling)


def strip_citations(text: str) -> str:
    """Remove citation markers, for output formats that should not carry them."""
    cleaned = CITATION.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def references_block(citations: Sequence[Citation], *, title: str = "Sources") -> str:
    """Render the reference list that goes under an answer."""
    if not citations:
        return ""
    lines = [f"{title}:"]
    lines.extend(f"  {item}" for item in citations)
    return "\n".join(lines)


__all__ = [
    "CITATION",
    "MAX_CONTEXT_CHARS",
    "Citation",
    "CitationCheck",
    "CitationSet",
    "build_citations",
    "citation_counts",
    "cited_context",
    "extract_citations",
    "references_block",
    "render_context",
    "strip_citations",
    "verify_citations",
]
