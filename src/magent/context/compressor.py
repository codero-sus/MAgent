"""Compressing a conversation so it fits a window without losing the point.

Four strategies, each with a real trade-off:

``trim``
    Drop the oldest messages. Free, instant, and lossy in the worst way - the
    beginning of a conversation is where the task was stated.
``window``
    Keep only the last N exchanges. Predictable cost, same blind spot as trim.
``truncate``
    Cut the longest messages in place, usually tool output. Keeps the shape of
    the conversation and loses detail rather than turns.
``summarize``
    Replace dropped turns with a short summary. Costs a model call but is the
    only strategy that preserves *what happened* rather than *what was said
    last*.

The summarizer degrades gracefully: with a backend it asks the model, without
one it falls back to an extractive summary that scores sentences by term
frequency. That keeps every code path testable offline, which matters because
a summarizer that only works against a live server is a summarizer nobody can
regression-test.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from ..llm.tokens import TokenEstimator, trim_messages, truncate_to_tokens
from ..logging_ import get_logger
from ..types import Message

LOGGER = get_logger("context.compress")
"""Logger for compression decisions."""

MIN_TRUNCATE_TOKENS = 16
"""Messages shorter than this are left alone; truncating them saves nothing."""

SUMMARY_PREFIX = "[earlier in this conversation] "
"""Marker prepended to a generated summary so it is never mistaken for user text."""

SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
"""Sentence splitter used by the extractive fallback."""

STOPWORDS = frozenset("""
a about an and are as at be but by can do for from had has have how i if in is it its
me my not of on or so that the their them then there these they this to was we were what
when which who will with you your
""".split())
"""Words that carry no topical signal when scoring sentences."""


class Summarizer(Protocol):
    """Anything that can turn messages into a short summary string."""

    def summarize(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Return a summary of ``messages`` no longer than ``max_tokens``."""
        ...


@dataclass(slots=True)
class CompressionResult:
    """What a compression pass did, and what it cost."""

    messages: list[Message] = field(default_factory=list)
    strategy: str = "trim"
    before_tokens: int = 0
    after_tokens: int = 0
    budget: int = 0
    dropped: int = 0
    truncated: int = 0
    summarized: bool = False
    summary_tokens: int = 0
    fits: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def saved_tokens(self) -> int:
        """Tokens removed by this pass."""
        return max(0, self.before_tokens - self.after_tokens)

    @property
    def ratio(self) -> float:
        """Fraction of the original size that remains."""
        if not self.before_tokens:
            return 1.0
        return round(self.after_tokens / self.before_tokens, 4)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for events and logs."""
        return {
            "strategy": self.strategy,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "budget": self.budget,
            "dropped": self.dropped,
            "truncated": self.truncated,
            "summarized": self.summarized,
            "summary_tokens": self.summary_tokens,
            "fits": self.fits,
            "saved_tokens": self.saved_tokens,
            "ratio": self.ratio,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """Return a one-line human-readable account of the pass."""
        parts = [f"{self.strategy}: {self.before_tokens}->{self.after_tokens} tokens"]
        if self.dropped:
            parts.append(f"dropped {self.dropped}")
        if self.truncated:
            parts.append(f"truncated {self.truncated}")
        if self.summarized:
            parts.append(f"summarized ({self.summary_tokens} tok)")
        if not self.fits:
            parts.append("STILL OVER BUDGET")
        return ", ".join(parts)


def extractive_summary(messages: Sequence[Message], *, max_tokens: int = 200,
                       estimator: TokenEstimator | None = None) -> str:
    """Summarize messages offline by picking their most informative sentences.

    Sentences are scored by how often their non-stopword terms appear across
    the whole conversation, with a small bonus for being early - the task is
    usually stated first. This is a real extractive summarizer, not a
    placeholder: it needs no model, so compression stays available (and
    testable) when no backend is reachable.

    Args:
        messages: the turns to compress
        max_tokens: ceiling for the returned text
        estimator: token counter, defaults to a fresh :class:`TokenEstimator`

    Returns:
        A single-paragraph summary in the original sentence order.
    """
    counter = estimator or TokenEstimator()
    sentences: list[tuple[int, str]] = []
    for position, message in enumerate(messages):
        text = message.text() if callable(getattr(message, "text", None)) else str(message)
        for match in SENTENCE.finditer(text or ""):
            sentence = " ".join(match.group(0).split())
            if len(sentence) >= 12:
                sentences.append((position, sentence))
    if not sentences:
        return ""
    frequencies: Counter[str] = Counter()
    for _, sentence in sentences:
        frequencies.update(term for term in re.findall(r"[a-z0-9_]{3,}", sentence.lower())
                           if term not in STOPWORDS)
    if not frequencies:
        return ""
    peak = max(frequencies.values()) or 1
    scored: list[tuple[float, int, str]] = []
    for order, (position, sentence) in enumerate(sentences):
        terms = [term for term in re.findall(r"[a-z0-9_]{3,}", sentence.lower())
                 if term not in STOPWORDS]
        if not terms:
            continue
        density = sum(frequencies[term] for term in terms) / (len(terms) * peak)
        early = 1.0 / (1.0 + order * 0.05)
        role_bonus = 1.15 if messages[position].role in {"user", "system"} else 1.0
        scored.append((density * early * role_bonus, order, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    budget = max(24, int(max_tokens))
    chosen: list[tuple[int, str]] = []
    used = 0
    for _, order, sentence in scored:
        cost = counter.estimate(sentence) + 2
        if used + cost > budget:
            continue
        chosen.append((order, sentence))
        used += cost
        if len(chosen) >= 8:
            break
    chosen.sort(key=lambda item: item[0])
    return " ".join(sentence for _, sentence in chosen).strip()


class ModelSummarizer:
    """Summarize with an LLM backend, falling back to the extractive summary."""

    def __init__(self, backend: Any, *, model: str = "", temperature: float = 0.2,
                 instruction: str = "", timeout: float = 60.0) -> None:
        """Wrap a backend exposing ``complete`` or ``chat``.

        Args:
            backend: the LLM backend used to write the summary
            model: model override, empty to use the backend default
            temperature: sampling temperature; low keeps summaries factual
            instruction: extra guidance appended to the summarization prompt
            timeout: per-call timeout in seconds
        """
        self.backend = backend
        self.model = model
        self.temperature = float(temperature)
        self.instruction = instruction
        self.timeout = float(timeout)
        self.calls = 0
        self.failures = 0

    def prompt(self, messages: Sequence[Message], max_tokens: int) -> str:
        """Build the summarization prompt for a set of turns."""
        transcript = "\n".join(
            f"{message.role}: {message.text() if callable(getattr(message, 'text', None)) else message}"
            for message in messages
        )[-12_000:]
        guidance = f"\n{self.instruction}" if self.instruction else ""
        return (
            "Compress this conversation transcript into a dense factual summary "
            f"of at most {max_tokens} tokens. Keep: the user's goal, every "
            "decision made, file paths, commands run, tool results, numbers, and "
            "anything still pending. Drop pleasantries and repetition. Write only "
            f"the summary.{guidance}\n\n{transcript}"
        )

    def summarize(self, messages: Sequence[Message], *, max_tokens: int = 200) -> str:
        """Return a model-written summary, or an extractive one on any failure."""
        if not messages or self.backend is None:
            return extractive_summary(messages, max_tokens=max_tokens)
        from ..llm.base import CompletionRequest

        request = CompletionRequest(
            messages=[Message.user(self.prompt(messages, max_tokens))],
            model=self.model,
            temperature=self.temperature,
            max_tokens=max(32, int(max_tokens)),
            timeout=self.timeout,
        )
        try:
            self.calls += 1
            response = self.backend.complete(request)
            text = (getattr(response, "text", "") or "").strip()
            if text:
                return text
            self.failures += 1
            LOGGER.debug("summarizer returned empty text; using extractive fallback")
        except Exception as exc:  # noqa: BLE001 - never let summarizing break a run
            self.failures += 1
            LOGGER.debug("summarization failed (%s); using extractive fallback", exc)
        return extractive_summary(messages, max_tokens=max_tokens)


def compress_messages(messages: Sequence[Message], *, budget: int, strategy: str = "trim",
                      estimator: TokenEstimator | None = None, keep_last: int = 2,
                      summarizer: Summarizer | None = None,
                      summary_tokens: int = 200) -> CompressionResult:
    """Compress ``messages`` until they fit ``budget`` tokens.

    The system message and the final turns always survive; everything else is
    dropped, truncated, or summarized according to ``strategy``. A pass that
    still does not fit escalates to the next strategy rather than giving up, so
    the caller can rely on getting the smallest legal conversation available.

    Args:
        messages: the conversation to compress
        budget: target token ceiling for the returned messages
        strategy: starting strategy, one of :data:`STRATEGIES`
        estimator: token counter shared with the caller
        keep_last: turns protected from dropping
        summarizer: used by the ``summarize`` strategy
        summary_tokens: ceiling for a generated summary

    Returns:
        A :class:`CompressionResult` describing the new messages and the cost.
    """
    counter = estimator or TokenEstimator()
    items = list(messages)
    start = counter.estimate_messages(items)
    result = CompressionResult(messages=items, strategy=strategy, before_tokens=start,
                               budget=budget, after_tokens=start)
    if budget <= 0 or start <= budget:
        result.fits = True
        result.messages = items
        return result

    order = [strategy] + [name for name in ("summarize", "truncate", "trim", "window")
                          if name != strategy]
    current = items
    for attempt in order:
        result.strategy = attempt
        if attempt == "window":
            kept, info = trim_messages(current, max_tokens=budget, estimator=counter,
                                       keep_last=max(keep_last, 2), strategy="oldest")
        elif attempt == "truncate":
            kept = _truncate_longest(current, budget=budget, estimator=counter,
                                     keep_last=keep_last)
            info = {"truncated": sum(1 for a, b in zip(current, kept, strict=False)
                                     if a.content != b.content)}
        elif attempt == "summarize":
            kept, info = _summarize(current, budget=budget, estimator=counter,
                                    keep_last=keep_last, summarizer=summarizer,
                                    summary_tokens=summary_tokens)
        else:
            kept, info = trim_messages(current, max_tokens=budget, estimator=counter,
                                       keep_last=keep_last, strategy="middle")
        tokens = counter.estimate_messages(kept)
        result.messages = kept
        result.after_tokens = tokens
        result.dropped = max(0, len(current) - len(kept))
        result.truncated = int(info.get("truncated", 0))
        result.summarized = bool(info.get("summarized"))
        result.summary_tokens = int(info.get("summary_tokens", 0))
        result.fits = tokens <= budget
        result.notes.append(f"{attempt}: {tokens} tokens ({'fits' if result.fits else 'over'})")
        if tokens <= budget:
            return result
        current = kept
    result.notes.append("escalated through every strategy; returning the smallest result")
    return result


def _truncate_longest(messages: Sequence[Message], *, budget: int,
                      estimator: TokenEstimator, keep_last: int = 2) -> list[Message]:
    """Shorten the largest messages in place until the total fits.

    Each pass halves the longest unprotected message. The loop is bounded twice
    over - by a hard pass count and by a no-progress check - because a message
    already at the floor cannot shrink further, and without that check the pass
    would spin forever on a conversation that cannot be made to fit.
    """
    items = [Message.from_dict(message.to_dict()) for message in messages]
    protected = max(0, len(items) - max(1, keep_last))
    previous = -1
    for _ in range(64):
        total = estimator.estimate_messages(items)
        if total <= budget or total == previous:
            break
        previous = total
        candidates = [(index, estimator.estimate(item.content or ""))
                      for index, item in enumerate(items)
                      if index < protected or item.role == "system"]
        candidates = [entry for entry in candidates if entry[1] > MIN_TRUNCATE_TOKENS]
        if not candidates:
            break
        index, tokens = max(candidates, key=lambda entry: entry[1])
        target = items[index]
        allowed = max(MIN_TRUNCATE_TOKENS, tokens // 2)
        target.content = truncate_to_tokens(target.content or "", allowed, estimator=estimator)
    return items


def _summarize(messages: Sequence[Message], *, budget: int, estimator: TokenEstimator,
               keep_last: int = 2, summarizer: Summarizer | None = None,
               summary_tokens: int = 200) -> tuple[list[Message], dict[str, Any]]:
    """Replace the oldest turns with one summary message."""
    items = list(messages)
    system = [message for message in items if message.role == "system"]
    body = [message for message in items if message.role != "system"]
    tail_count = min(max(1, keep_last), len(body))
    head, tail = body[: len(body) - tail_count], body[len(body) - tail_count:]
    if not head:
        return items, {"summarized": False, "summary_tokens": 0, "truncated": 0}
    room = max(64, budget - estimator.estimate_messages(system + tail) - 8)
    ceiling = min(summary_tokens, room)
    worker = summarizer or _ExtractiveSummarizer()
    text = worker.summarize(head, max_tokens=ceiling) or extractive_summary(head, max_tokens=ceiling)
    if not text.strip():
        kept, info = trim_messages(items, max_tokens=budget, estimator=estimator,
                                   keep_last=keep_last, strategy="middle")
        info.update({"summarized": False})
        return kept, info
    summary = Message.system(f"{SUMMARY_PREFIX}{text.strip()}")
    summary.metadata["compressed"] = True
    summary.metadata["replaces"] = len(head)
    return system + [summary] + tail, {
        "summarized": True,
        "summary_tokens": estimator.estimate(summary.content or ""),
        "truncated": 0,
    }


class _ExtractiveSummarizer:
    """Default summarizer used when the caller supplies no backend."""

    def summarize(self, messages: Sequence[Message], *, max_tokens: int = 200) -> str:
        """Return an extractive summary of ``messages``."""
        return extractive_summary(messages, max_tokens=max_tokens)


def summarize_messages(messages: Sequence[Message], *, max_tokens: int = 200,
                       backend: Any = None, model: str = "") -> str:
    """One-shot summary helper used by the CLI and the session compactor."""
    worker: Summarizer = ModelSummarizer(backend, model=model) if backend is not None \
        else _ExtractiveSummarizer()
    return worker.summarize(messages, max_tokens=max_tokens)


__all__ = [
    "MIN_TRUNCATE_TOKENS",
    "STOPWORDS",
    "SUMMARY_PREFIX",
    "CompressionResult",
    "ModelSummarizer",
    "Summarizer",
    "compress_messages",
    "extractive_summary",
    "summarize_messages",
]
