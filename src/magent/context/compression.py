"""Context compression: fit more signal into a fixed token window.

Retrieval and memory both want to say a lot.  The compressor decides what the
model actually sees, in this order of aggressiveness:

1. drop whole blocks, lowest priority first (required blocks are exempt);
2. truncate the survivors to their own ``max_tokens`` ceilings;
3. strip boilerplate - repeated lines, navigation cruft, over-long code spans;
4. optionally replace a block's text with a model-written summary.

Every decision is recorded on the block (``dropped`` / ``truncated``) and in the
returned :class:`CompressionReport`, so a run can explain why a document did not
make it into the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..llm.tokens import TokenEstimator, estimate_tokens, truncate_to_tokens
from ..logging_ import get_logger
from ..utils.text import collapse_whitespace, ellipsize, normalize_newlines
from .block import ContextBlock, render_blocks, sort_blocks

LOGGER = get_logger("context.compression")
"""Logger for compression decisions."""

NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[ \t]*[-*+][ \t]*$", re.MULTILINE),
    re.compile(r"^[ \t]*(page|next|previous|back to top|table of contents)\b.*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*\[\s*(edit|view source|sign in|log in)\s*\].*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"\n{4,}"),
)
"""Lines that carry no information and are removed before token counting."""

MAX_LINE_REPEATS = 3
"""How many times an identical line may repeat before the extras are cut."""


@dataclass(slots=True)
class CompressionReport:
    """What the compressor did, for logs and the ``context`` event."""

    before_tokens: int = 0
    after_tokens: int = 0
    budget: int = 0
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    summarized: list[str] = field(default_factory=list)
    passes: int = 0
    fits: bool = True

    @property
    def saved_tokens(self) -> int:
        """Tokens removed from the prompt."""
        return max(0, self.before_tokens - self.after_tokens)

    @property
    def ratio(self) -> float:
        """After/before token ratio; 1.0 means nothing was compressed."""
        if not self.before_tokens:
            return 1.0
        return round(self.after_tokens / self.before_tokens, 4)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report."""
        return {
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "budget": self.budget,
            "saved_tokens": self.saved_tokens,
            "ratio": self.ratio,
            "dropped": list(self.dropped),
            "truncated": list(self.truncated),
            "summarized": list(self.summarized),
            "passes": self.passes,
            "fits": self.fits,
        }

    def summary(self) -> str:
        """One human-readable line describing the compression."""
        if not self.dropped and not self.truncated and not self.summarized:
            return f"context fits budget ({self.after_tokens} tokens)"
        parts = [f"{self.before_tokens}->{self.after_tokens} tokens"]
        if self.dropped:
            parts.append(f"dropped {len(self.dropped)}: {', '.join(self.dropped[:4])}")
        if self.truncated:
            parts.append(f"truncated {len(self.truncated)}")
        if self.summarized:
            parts.append(f"summarized {len(self.summarized)}")
        return "; ".join(parts)


def strip_noise(text: str) -> str:
    """Remove navigation cruft, empty bullets and runaway blank lines."""
    out = normalize_newlines(text or "")
    for pattern in NOISE_PATTERNS:
        out = pattern.sub("\n", out)
    return collapse_whitespace(out).strip()


def dedupe_lines(text: str, *, max_repeats: int = MAX_LINE_REPEATS) -> str:
    """Collapse lines that repeat more than ``max_repeats`` times in a row."""
    kept: list[str] = []
    run = 0
    previous: str | None = None
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if stripped and stripped == previous:
            run += 1
            if run >= max_repeats:
                continue
        else:
            run = 0
        previous = stripped
        kept.append(line)
    return "\n".join(kept)


def compress_text(text: str, *, max_tokens: int = 0, max_chars: int = 0,
                  strip_boilerplate: bool = True, estimator: TokenEstimator | None = None,
                  keep_tail: float = 0.0) -> str:
    """Shrink one string towards a token or character ceiling.

    Args:
        text: the input
        max_tokens: token ceiling, 0 for no limit
        max_chars: character ceiling applied after token truncation
        strip_boilerplate: run :func:`strip_noise` and :func:`dedupe_lines`
        estimator: estimator used for token accounting
        keep_tail: fraction (0..0.4) of the budget reserved for the ending
    """
    out = strip_noise(text) if strip_boilerplate else normalize_newlines(text or "")
    out = dedupe_lines(out) if strip_boilerplate else out
    tokens = estimator.estimate(out) if estimator else estimate_tokens(out)
    if max_tokens and tokens > max_tokens:
        keep_tail = max(0.0, min(0.4, keep_tail))
        if keep_tail > 0:
            head_budget = int(max_tokens * (1 - keep_tail))
            tail_budget = max_tokens - head_budget
            head = truncate_to_tokens(out, head_budget, estimator=estimator, marker="\n[...]\n")
            tail_chars = int(tail_budget * 4)
            tail = out[-tail_chars:] if tail_chars else ""
            out = f"{head.rstrip()}\n[...]\n{tail.lstrip()}" if tail else head
        else:
            out = truncate_to_tokens(out, max_tokens, estimator=estimator)
        if not out.strip() or out.strip() in {"[truncated]", "[...]"}:
            # A budget too small for the marker still deserves the leading text.
            out = ellipsize(strip_noise(text), max(16, max_tokens * 4))
    if max_chars and len(out) > max_chars:
        out = ellipsize(out, max_chars)
    return out.strip()


def summarize_with(summarizer: Callable[[str], str], text: str, *, max_words: int = 120) -> str:
    """Summarize ``text`` with a caller-supplied function, falling back safely."""
    if not text.strip():
        return ""
    try:
        summary = summarizer(text).strip()
    except Exception as exc:  # noqa: BLE001 - a broken summarizer must not break a run
        LOGGER.debug("summarizer failed, keeping original text: %s", exc)
        return text
    if not summary:
        return text
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + " [...]"
    return summary


@dataclass(slots=True)
class ContextCompressor:
    """Fit a set of blocks into a token budget."""

    budget: int = 4_000
    min_block_tokens: int = 120
    strip_boilerplate: bool = True
    summarizer: Callable[[str], str] | None = None
    summary_kinds: tuple[str, ...] = ("rag", "files", "session")
    estimator: TokenEstimator | None = None
    style: str = "heading"

    def tokens(self, text: str) -> int:
        """Token estimate for one string, honouring the configured estimator."""
        return self.estimator.estimate(text) if self.estimator else estimate_tokens(text)

    def clean(self, blocks: Sequence[ContextBlock]) -> int:
        """Strip boilerplate from every block in place; returns tokens saved."""
        saved = 0
        for item in blocks:
            if not self.strip_boilerplate or not item.body:
                continue
            before = item.tokens
            item.body = strip_noise(dedupe_lines(item.body))
            item.tokens = self.tokens(item.body)
            saved += max(0, before - item.tokens)
        return saved

    def fit(self, blocks: Iterable[ContextBlock], *, budget: int | None = None) -> tuple[list[ContextBlock], CompressionReport]:
        """Compress blocks until they fit the budget.

        Args:
            blocks: candidates, in any order
            budget: token ceiling; defaults to ``self.budget``

        Returns:
            the surviving blocks in priority order, plus a report
        """
        limit = int(budget or self.budget or 0)
        ordered = sort_blocks(blocks)
        report = CompressionReport(budget=limit)
        report.before_tokens = sum(item.tokens for item in ordered)
        self.clean(ordered)
        report.before_tokens = sum(item.tokens for item in ordered)
        if not limit:
            report.after_tokens = report.before_tokens
            report.fits = True
            return [item for item in ordered if not item.empty], report

        # pass 1: hard per-block ceilings
        for item in ordered:
            ceiling = item.max_tokens
            if ceiling and item.tokens > ceiling:
                item.body = compress_text(item.body, max_tokens=ceiling,
                                          strip_boilerplate=self.strip_boilerplate,
                                          estimator=self.estimator)
                item.tokens = self.tokens(item.body)
                item.truncated = True
                report.truncated.append(item.title)
        report.passes += 1

        # pass 2: drop the cheapest blocks until the total fits
        while sum(item.tokens for item in ordered if not item.dropped) > limit:
            victim = self._pick_victim(ordered, limit)
            if victim is None:
                break
            victim.dropped = True
            report.dropped.append(victim.title)
        report.passes += 1

        # pass 3: summarize the bulky survivors when a summarizer exists
        if self.summarizer and sum(item.tokens for item in ordered if not item.dropped) > limit:
            for item in self._by_size(ordered):
                if item.dropped or item.required or item.kind not in self.summary_kinds:
                    continue
                summary = summarize_with(self.summarizer, item.body)
                if self.tokens(summary) < item.tokens:
                    item.body = summary
                    item.tokens = self.tokens(summary)
                    item.metadata["summarized"] = True
                    report.summarized.append(item.title)
                if sum(x.tokens for x in ordered if not x.dropped) <= limit:
                    break
            report.passes += 1

        # pass 4: proportional truncation of whatever is still over
        total = sum(item.tokens for item in ordered if not item.dropped)
        if total > limit:
            shrinkable = [item for item in ordered if not item.dropped and not item.required]
            excess = total - limit
            for item in self._by_size(shrinkable):
                if excess <= 0:
                    break
                share = max(self.min_block_tokens, item.tokens - excess)
                if share >= item.tokens:
                    continue
                item.body = compress_text(item.body, max_tokens=share,
                                          strip_boilerplate=self.strip_boilerplate,
                                          estimator=self.estimator)
                saved = item.tokens - self.tokens(item.body)
                item.tokens = self.tokens(item.body)
                item.truncated = True
                excess -= max(0, saved)
                if item.title not in report.truncated:
                    report.truncated.append(item.title)
            report.passes += 1

        kept = [item for item in ordered if not item.dropped and not item.empty]
        report.after_tokens = sum(item.tokens for item in kept)
        report.fits = report.after_tokens <= limit or limit <= 0
        return kept, report

    def render(self, blocks: Iterable[ContextBlock], *, budget: int | None = None) -> str:
        """Compress and render in one call."""
        kept, _ = self.fit(blocks, budget=budget)
        return render_blocks(kept, style=self.style)

    def _pick_victim(self, blocks: Sequence[ContextBlock], limit: int) -> ContextBlock | None:
        """Choose the least valuable droppable block.

        Priority dominates: a low-priority block goes before a high-priority one
        no matter how small it is.  Inside one priority tier the block with the
        weakest relevance score is dropped, and ties break towards the most
        expensive block so one drop frees as much budget as possible.
        """
        candidates = [item for item in blocks if not item.dropped and not item.required]
        live = [item for item in blocks if not item.dropped]
        if not candidates or len(live) <= 1:
            return None
        return min(candidates, key=_drop_key)


def _drop_key(item: ContextBlock) -> tuple[int, float, int, str]:
    """Sort key ranking blocks from first-dropped to last-dropped."""
    score = float(item.metadata.get("score") or 0.0)
    return (item.priority, score, -item.tokens, item.title)

    def _by_size(self, blocks: Sequence[ContextBlock]) -> list[ContextBlock]:
        """Largest blocks first."""
        return sorted(blocks, key=lambda item: -item.tokens)


def compress_messages(messages: Sequence[Any], *, max_tokens: int,
                      keep_last: int = 2, keep_system: bool = True,
                      strategy: str = "middle") -> tuple[list[Any], dict[str, Any]]:
    """Trim a message history to a token budget (thin wrapper for callers)."""
    from ..llm.tokens import trim_messages

    return trim_messages(messages, max_tokens=max_tokens, keep_system=keep_system,
                         keep_last=keep_last, strategy=strategy)


__all__ = [
    "MAX_LINE_REPEATS",
    "NOISE_PATTERNS",
    "CompressionReport",
    "ContextCompressor",
    "compress_messages",
    "compress_text",
    "dedupe_lines",
    "strip_noise",
    "summarize_with",
]
