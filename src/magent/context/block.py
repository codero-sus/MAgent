"""Context blocks: the unit the context builder assembles and prunes.

A context block is a titled slice of information that may be injected into the
system prompt - retrieved documents, remembered facts, open files, a session
summary, environment details.  Giving every slice a priority and a token count
is what makes budget management possible: when the window is tight the builder
drops the lowest-priority blocks whole, then truncates the survivors, instead of
silently cutting the middle of the prompt.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.tokens import estimate_tokens, truncate_to_tokens
from ..utils.text import ellipsize, normalize_newlines, word_wrap

BLOCK_KINDS: tuple[str, ...] = (
    "environment",
    "files",
    "rag",
    "memory",
    "session",
    "task",
    "skills",
    "tools",
    "custom",
)
"""Recognised block kinds, most to least structural."""

DEFAULT_PRIORITY = 5
"""Priority used when a caller does not state one."""

MAX_PRIORITY = 10
"""Highest priority; blocks at this level are never dropped for budget."""


@dataclass(slots=True)
class ContextBlock:
    """One titled slice of context with a priority and a measured cost."""

    title: str
    body: str
    kind: str = "custom"
    priority: int = DEFAULT_PRIORITY
    source: str = ""
    tokens: int = 0
    required: bool = False
    max_tokens: int = 0
    truncated: bool = False
    dropped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the body, clamp the priority and measure the tokens."""
        self.body = normalize_newlines(self.body or "").strip()
        self.title = (self.title or self.kind or "Context").strip()
        self.kind = self.kind if self.kind in BLOCK_KINDS else "custom"
        self.priority = max(0, min(MAX_PRIORITY, int(self.priority)))
        self.max_tokens = max(0, int(self.max_tokens))
        if not self.tokens:
            self.tokens = estimate_tokens(self.body)
        if self.max_tokens and self.tokens > self.max_tokens:
            self.body = truncate_to_tokens(self.body, self.max_tokens)
            self.tokens = estimate_tokens(self.body)
            self.truncated = True

    @property
    def empty(self) -> bool:
        """True when the block carries no text at all."""
        return not self.body.strip()

    def render(self, *, style: str = "heading", width: int = 0) -> str:
        """Render the block as prompt text.

        Args:
            style: ``heading`` (``## Title``), ``plain`` (bare body), or
                ``tagged`` (``<context kind="rag">...</context>``)
            width: wrap the body at this column count, 0 to leave it alone
        """
        body = word_wrap(self.body, width) if width else self.body
        if style == "plain":
            return body
        if style == "tagged":
            return f'<context kind="{self.kind}" title="{self.title}">\n{body}\n</context>'
        suffix = ""
        if self.truncated:
            suffix = " (truncated)"
        return f"## {self.title}{suffix}\n\n{body}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the block for logs and session transcripts."""
        return {
            "title": self.title,
            "kind": self.kind,
            "priority": self.priority,
            "source": self.source,
            "tokens": self.tokens,
            "required": self.required,
            "truncated": self.truncated,
            "dropped": self.dropped,
            "chars": len(self.body),
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:
        """Number of characters in the body."""
        return len(self.body)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (
            f"ContextBlock(kind={self.kind!r}, title={self.title!r}, "
            f"tokens={self.tokens}, priority={self.priority})"
        )


def block(title: str, body: str, **kwargs: Any) -> ContextBlock:
    """Convenience constructor for a :class:`ContextBlock`."""
    return ContextBlock(title=title, body=body, **kwargs)


def environment_block(text: str, *, priority: int = 7) -> ContextBlock:
    """Build the block describing workspace, platform and tool availability."""
    return ContextBlock(
        title="Environment", body=text, kind="environment", priority=priority, required=True
    )


def files_block(
    entries: Sequence[str],
    *,
    title: str = "Workspace files",
    priority: int = 4,
    max_tokens: int = 900,
) -> ContextBlock:
    """Build a block listing files, one per line."""
    body = "\n".join(line for line in entries if line)
    return ContextBlock(
        title=title, body=body, kind="files", priority=priority, max_tokens=max_tokens
    )


def rag_block(
    chunks: Iterable[Any],
    *,
    title: str = "Retrieved context",
    priority: int = 8,
    max_tokens: int = 2_500,
) -> ContextBlock:
    """Build a block from retrieved chunks, keeping their citations.

    Args:
        chunks: objects with ``text`` plus ``source``/``start_line`` attributes
        title: heading used in the prompt
        priority: high by default - retrieved evidence should survive trimming
        max_tokens: hard ceiling for the block
    """
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        text = str(getattr(chunk, "text", chunk) or "").strip()
        if not text:
            continue
        source = str(getattr(chunk, "source", "") or "")
        start = getattr(chunk, "start_line", 0) or 0
        end = getattr(chunk, "end_line", 0) or 0
        where = f"{source}:{start}-{end}" if source and start else source
        label = f"[{index}]" + (f" {where}" if where else "")
        lines.append(f"{label} {ellipsize(text, 1_200)}")
    return ContextBlock(
        title=title, body="\n\n".join(lines), kind="rag", priority=priority, max_tokens=max_tokens
    )


def memory_block(
    entries: Iterable[Any],
    *,
    title: str = "Relevant memories",
    priority: int = 6,
    max_tokens: int = 900,
) -> ContextBlock:
    """Build a block from memory entries, newest facts first."""
    lines: list[str] = []
    for entry in entries:
        text = str(getattr(entry, "content", entry) or "").strip()
        if not text:
            continue
        kind = str(getattr(entry, "kind", "") or "")
        score = getattr(entry, "score", 0.0) or 0.0
        prefix = f"- ({kind}, {score:.2f}) " if kind and score else "- "
        lines.append(f"{prefix}{ellipsize(text, 400)}")
    return ContextBlock(
        title=title, body="\n".join(lines), kind="memory", priority=priority, max_tokens=max_tokens
    )


def session_block(
    summary: str, *, transcript: str = "", priority: int = 6, max_tokens: int = 1_200
) -> ContextBlock:
    """Build a block carrying the running summary of the current session."""
    parts = [part for part in (summary.strip(), transcript.strip()) if part]
    return ContextBlock(
        title="Session so far",
        body="\n\n".join(parts),
        kind="session",
        priority=priority,
        max_tokens=max_tokens,
    )


def task_block(text: str, *, priority: int = 9) -> ContextBlock:
    """Build the block holding the user's stated task."""
    return ContextBlock(title="Task", body=text, kind="task", priority=priority, required=True)


def custom_block(
    title: str, body: str, *, priority: int = DEFAULT_PRIORITY, kind: str = "custom", **kwargs: Any
) -> ContextBlock:
    """Build a caller-defined block."""
    return ContextBlock(title=title, body=body, kind=kind, priority=priority, **kwargs)


def total_tokens(blocks: Sequence[ContextBlock]) -> int:
    """Sum the token estimates of the blocks that are still in play."""
    return sum(item.tokens for item in blocks if not item.dropped)


def render_blocks(
    blocks: Sequence[ContextBlock],
    *,
    style: str = "heading",
    separator: str = "\n\n",
    width: int = 0,
) -> str:
    """Render a sequence of blocks into one prompt-ready string."""
    kept = [item for item in blocks if not item.dropped and not item.empty]
    if not kept:
        return ""
    return separator.join(item.render(style=style, width=width) for item in kept)


def sort_blocks(blocks: Iterable[ContextBlock]) -> list[ContextBlock]:
    """Order blocks by descending priority, keeping required blocks first."""
    return sorted(blocks, key=lambda item: (-int(item.required), -item.priority, item.kind))


__all__ = [
    "BLOCK_KINDS",
    "DEFAULT_PRIORITY",
    "MAX_PRIORITY",
    "ContextBlock",
    "block",
    "custom_block",
    "environment_block",
    "files_block",
    "memory_block",
    "rag_block",
    "render_blocks",
    "session_block",
    "sort_blocks",
    "task_block",
    "total_tokens",
]
