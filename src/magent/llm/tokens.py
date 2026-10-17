"""Token estimation, budgeting, and context trimming.

MAgent cannot ship a real BPE tokenizer without a dependency, so it uses a
calibrated heuristic that is accurate to a few percent for English text and
conservative (over-estimating) for code and CJK. When a backend returns real
``usage`` numbers those always win, and the estimator self-calibrates against
them via :meth:`TokenEstimator.calibrate`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import Message, Role

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]+|\s+")
"""Tokenizer-ish splitter used by the heuristic estimator."""

_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
"""Range covering CJK and full-width characters, which tokenize per glyph."""

_CHARS_PER_TOKEN = 4.0
"""Default characters-per-token for Latin prose."""

_CJK_CHARS_PER_TOKEN = 1.4
"""CJK glyphs average fewer characters per token."""

_MESSAGE_OVERHEAD = 4
"""Per-message framing tokens in the OpenAI chat format."""

_REPLY_OVERHEAD = 3
"""Tokens reserved for the assistant reply priming."""


def count_cjk(text: str) -> int:
    """Return how many CJK/full-width characters a string contains."""
    return len(_CJK_RE.findall(text))


def estimate_tokens(text: str, *, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    """Estimate the token count of ``text`` with a calibrated heuristic.

    Latin prose is divided into word/number/punctuation/whitespace runs and
    long words are split into 4-character subwords, which tracks BPE behaviour
    closely. CJK characters are counted near-per-glyph.
    """
    if not text:
        return 0
    cjk = count_cjk(text)
    latin = len(text) - cjk
    runs = _WORD_RE.findall(text)
    units = 0.0
    for run in runs:
        if _CJK_RE.match(run):
            continue
        length = len(run)
        if run.isalpha():
            units += max(1.0, math.ceil(length / 4.2)) if length > 4 else 1.0
        elif run.isdigit():
            units += max(1.0, math.ceil(length / 3.0))
        elif run.isspace():
            units += 0.25
        else:
            units += max(1.0, length * 0.6)
    total = units * (4.0 / max(1.0, chars_per_token)) + cjk / _CJK_CHARS_PER_TOKEN
    if latin == 0:
        return max(1, int(math.ceil(total)))
    return max(1, int(math.ceil(total)))


@dataclass
class TokenEstimator:
    """Self-calibrating token estimator with observed accuracy tracking.

    Feed it real ``(estimated, actual)`` pairs from backend usage reports and
    it adjusts ``chars_per_token`` toward the observed ratio.
    """

    chars_per_token: float = _CHARS_PER_TOKEN
    samples: int = 0
    total_estimated: int = 0
    total_actual: int = 0
    min_chars_per_token: float = 2.0
    max_chars_per_token: float = 8.0
    history: list[tuple[int, int]] = field(default_factory=list)
    max_history: int = 200

    def estimate(self, text: str) -> int:
        """Estimate tokens for one string."""
        return estimate_tokens(text, chars_per_token=self.chars_per_token)

    def estimate_messages(
        self, messages: Sequence[Message], *, include_overhead: bool = True
    ) -> int:
        """Estimate tokens for a full chat message list."""
        total = 0
        for message in messages:
            total += self.estimate(message.text())
            for call in message.tool_calls:
                total += self.estimate(call.name) + self.estimate(call.signature())
            if include_overhead:
                total += _MESSAGE_OVERHEAD
        if include_overhead:
            total += _REPLY_OVERHEAD
        return total

    def calibrate(self, estimated: int, actual: int) -> float:
        """Adjust the heuristic from one observed pair and return the new ratio."""
        if estimated <= 0 or actual <= 0:
            return self.chars_per_token
        self.samples += 1
        self.total_estimated += estimated
        self.total_actual += actual
        self.history.append((estimated, actual))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        ratio = self.total_actual / self.total_estimated
        adjusted = self.chars_per_token * ratio
        self.chars_per_token = round(
            min(self.max_chars_per_token, max(self.min_chars_per_token, adjusted)), 4
        )
        return self.chars_per_token

    @property
    def error_ratio(self) -> float:
        """Return how far off estimates have been (1.0 means perfect)."""
        if not self.total_estimated:
            return 1.0
        return round(self.total_actual / self.total_estimated, 4)

    def stats(self) -> dict[str, Any]:
        """Return calibration telemetry."""
        return {
            "chars_per_token": self.chars_per_token,
            "samples": self.samples,
            "total_estimated": self.total_estimated,
            "total_actual": self.total_actual,
            "error_ratio": self.error_ratio,
        }


@dataclass(slots=True)
class TokenBudget:
    """A hard ceiling on tokens for a run, a step, or a session."""

    max_tokens: int = 0
    used: int = 0
    reserved_for_output: int = 0

    @property
    def remaining(self) -> int:
        """Return how many input tokens are still available.

        An unlimited budget reports :data:`sys.maxsize` so callers can compare
        without special casing.
        """
        import sys

        if self.is_unlimited:
            return sys.maxsize
        return max(0, self.max_tokens - self.used - self.reserved_for_output)

    @property
    def is_unlimited(self) -> bool:
        """Report whether no ceiling is configured."""
        return self.max_tokens <= 0

    def consume(self, count: int) -> int:
        """Record usage and return the remaining allowance."""
        self.used += max(0, count)
        return self.remaining

    def would_exceed(self, count: int) -> bool:
        """Report whether spending ``count`` tokens would break the budget."""
        if self.is_unlimited:
            return False
        return self.used + count + self.reserved_for_output > self.max_tokens

    def utilization(self) -> float:
        """Return budget consumption as a 0..1 fraction."""
        if self.is_unlimited:
            return 0.0
        return round(min(1.0, self.used / self.max_tokens), 4)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the budget state."""
        return {
            "max_tokens": self.max_tokens,
            "used": self.used,
            "reserved_for_output": self.reserved_for_output,
            "remaining": None if self.is_unlimited else self.remaining,
            "utilization": self.utilization(),
            "unlimited": self.is_unlimited,
        }

    def reset(self) -> None:
        """Zero the consumed counter."""
        self.used = 0


def trim_messages(
    messages: Sequence[Message],
    *,
    max_tokens: int,
    estimator: TokenEstimator | None = None,
    keep_system: bool = True,
    keep_last: int = 2,
    strategy: str = "middle",
) -> tuple[list[Message], dict[str, Any]]:
    """Drop messages until the list fits ``max_tokens``.

    Strategies:

    ``middle``
        Keep the leading system prompt and the most recent ``keep_last``
        messages, evicting from the middle outwards. Preserves both the agent's
        instructions and the immediate conversational context.
    ``oldest``
        Evict strictly from the front.
    ``newest``
        Evict strictly from the back (useful for "first message wins" prompts).

    Never splits a tool-call / tool-result pair, which would produce an
    invalid request the backend would reject.
    """
    counter = estimator or TokenEstimator()
    items = list(messages)
    total = counter.estimate_messages(items)
    dropped: list[int] = []
    if total <= max_tokens or not items:
        return items, {
            "dropped": 0,
            "tokens_before": total,
            "tokens_after": total,
            "strategy": strategy,
            "indices": dropped,
        }

    protected: set[int] = set()
    if keep_system:
        for index, message in enumerate(items):
            if message.role is Role.SYSTEM:
                protected.add(index)
    tail_start = max(0, len(items) - keep_last)
    protected.update(range(tail_start, len(items)))

    def pair_indices() -> set[int]:
        """Return indices that belong to a tool call/result pair."""
        pairs: set[int] = set()
        for index, message in enumerate(items):
            if message.tool_calls:
                pairs.add(index)
                for offset in range(1, len(message.tool_calls) + 2):
                    candidate = index + offset
                    if candidate < len(items) and items[candidate].role is Role.TOOL:
                        pairs.add(candidate)
            if message.role is Role.TOOL:
                pairs.add(index)
        return pairs

    pairs = pair_indices()
    while total > max_tokens:
        candidate = _pick_victim(items, protected, pairs, strategy)
        if candidate is None:
            break
        removed = items.pop(candidate)
        dropped.append(candidate)
        total -= counter.estimate(removed.text()) + _MESSAGE_OVERHEAD
        protected = {index for index in protected if index < candidate} | {
            index - 1 if index > candidate else index for index in protected if index != candidate
        }
        pairs = pair_indices()
    return items, {
        "dropped": len(dropped),
        "tokens_before": counter.estimate_messages(messages),
        "tokens_after": counter.estimate_messages(items),
        "strategy": strategy,
        "indices": dropped,
    }


def _pick_victim(
    items: Sequence[Message],
    protected: set[int],
    pairs: set[int],
    strategy: str,
) -> int | None:
    """Choose the next message index to evict, honoring protections."""
    order: Iterable[int]
    if strategy == "oldest":
        order = range(len(items))
    elif strategy == "newest":
        order = reversed(range(len(items)))
    else:
        middle = len(items) // 2
        order = sorted(range(len(items)), key=lambda index: abs(index - middle))
    for index in order:
        if index in protected or index in pairs:
            continue
        if items[index].role is Role.SYSTEM:
            continue
        return index
    for index in order:
        if index in protected:
            continue
        return index
    return None


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    *,
    estimator: TokenEstimator | None = None,
    marker: str = "\n[truncated]\n",
) -> str:
    """Truncate a string so its estimated token count fits ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    counter = estimator or TokenEstimator()
    if counter.estimate(text) <= max_tokens:
        return text
    ratio = max_tokens / max(1, counter.estimate(text))
    cut = int(len(text) * ratio * 0.92)
    while cut > 0 and counter.estimate(text[:cut] + marker) > max_tokens:
        cut = int(cut * 0.9)
    return text[: max(0, cut)] + marker


def split_by_tokens(
    text: str, max_tokens: int, *, estimator: TokenEstimator | None = None
) -> list[str]:
    """Split text into pieces that each fit within ``max_tokens``."""
    if max_tokens <= 0:
        return [text] if text else []
    counter = estimator or TokenEstimator()
    if counter.estimate(text) <= max_tokens:
        return [text]
    from ..utils.text import chunk_text

    approx_chars = max(64, int(max_tokens * counter.chars_per_token * 0.9))
    # This is a partitioning helper for LLM requests, not a retrieval chunker:
    # overlap duplicated boundary words when callers reassembled its output.
    pieces = chunk_text(text, max_chars=approx_chars, overlap=0)
    out: list[str] = []
    for piece in pieces:
        while counter.estimate(piece) > max_tokens and len(piece) > approx_chars:
            approx_chars = int(approx_chars * 0.85)
            piece = piece[:approx_chars]
        out.append(piece)
    return out


def context_window_plan(
    messages: Sequence[Message],
    *,
    context_size: int,
    max_output_tokens: int = 1024,
    estimator: TokenEstimator | None = None,
) -> dict[str, Any]:
    """Compute how much room a request has and whether trimming is needed."""
    counter = estimator or TokenEstimator()
    prompt_tokens = counter.estimate_messages(messages)
    available = max(0, context_size - max_output_tokens)
    return {
        "context_size": context_size,
        "max_output_tokens": max_output_tokens,
        "prompt_tokens": prompt_tokens,
        "available_input_tokens": available,
        "fits": prompt_tokens <= available,
        "overflow_tokens": max(0, prompt_tokens - available),
        "utilization": round(prompt_tokens / context_size, 4) if context_size else 0.0,
    }


__all__ = [
    "TokenBudget",
    "TokenEstimator",
    "context_window_plan",
    "count_cjk",
    "estimate_tokens",
    "split_by_tokens",
    "trim_messages",
    "truncate_to_tokens",
]
