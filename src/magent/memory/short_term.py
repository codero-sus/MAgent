"""Short-term memory: the conversation window and its summarization.

Long-term memory (:class:`~magent.memory.store.MemoryStore`) is what an agent
knows across sessions. Short-term memory is what it can hold in one context
window, and it has a hard budget. This module owns that budget:

- messages are appended per session,
- the window is trimmed by count and by token estimate, never splitting a
  tool-call / tool-result pair,
- when the window overflows a threshold, the evicted turns are summarized and
  the summary is carried forward as a system message, so information survives
  the trim instead of vanishing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.tokens import TokenEstimator, trim_messages
from ..logging_ import get_logger
from ..prompts.templates import CONTEXT_HANDOFF, SUMMARIZE_PROMPT
from ..types import Message, Role
from ..utils.text import ellipsize

LOGGER = get_logger("memory.short_term")
"""Logger for window management."""

DEFAULT_MAX_MESSAGES = 60
"""Default number of turns kept in the window."""

DEFAULT_MAX_TOKENS = 6000
"""Default token ceiling for the visible window."""


@dataclass
class ConversationWindow:
    """One session's message window plus its rolling summary."""

    session_id: str = "default"
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    summarized_messages: int = 0
    trims: int = 0
    system_prompt: str = ""
    pending: list[str] = field(default_factory=list)

    # -- writes --------------------------------------------------------------

    def append(self, message: Message) -> None:
        """Add one message to the window."""
        if message.role is Role.SYSTEM and not self.messages:
            self.system_prompt = message.text()
        self.messages.append(message)

    def extend(self, messages: Iterable[Message]) -> int:
        """Add several messages; return how many were appended."""
        items = list(messages)
        for message in items:
            self.append(message)
        return len(items)

    def set_summary(self, summary: str, *, consumed: int = 0) -> None:
        """Replace the rolling summary and record how many turns it covers."""
        self.summary = str(summary or "").strip()
        if consumed:
            self.summarized_messages += consumed

    def queue_evicted(self, messages: Iterable[Message], *, limit: int = 200) -> int:
        """Buffer evicted turns until they are worth a summarization call.

        Summarizing on every single eviction costs one model call per message
        and produces a summary of a summary. Buffering keeps the text safe while
        batching the work.

        Args:
            messages: the turns being evicted
            limit: how many buffered turns to retain before dropping the oldest
        """
        added = 0
        for message in messages:
            role = message.role.value if isinstance(message.role, Role) else str(message.role)
            body = ellipsize(message.text(), 400).strip()
            if body:
                self.pending.append(f"{role}: {body}")
                added += 1
        if len(self.pending) > max(1, limit):
            del self.pending[: len(self.pending) - max(1, limit)]
        return added

    def clear(self, *, keep_system: bool = True) -> int:
        """Empty the window; return how many messages were dropped."""
        dropped = len(self.messages)
        if keep_system and self.system_prompt:
            self.messages = [Message.system(self.system_prompt)]
            dropped -= 1
        else:
            self.messages = []
        return max(0, dropped)

    # -- reads ---------------------------------------------------------------

    def visible(self, *, include_summary: bool = True) -> list[Message]:
        """Return what should be sent to the model right now.

        The system prompt leads, then the rolling summary as a system message,
        then the retained turns.
        """
        out: list[Message] = []
        if self.system_prompt:
            out.append(Message.system(self.system_prompt))
        elif self.messages and self.messages[0].role is Role.SYSTEM:
            out.append(self.messages[0])
        if include_summary and self.summary:
            out.append(Message.system(CONTEXT_HANDOFF.render(summary=self.summary)))
        start = 1 if (self.messages and self.messages[0].role is Role.SYSTEM) else 0
        out.extend(self.messages[start:])
        return out

    def transcript(self, *, limit: int = 4000, include_summary: bool = False) -> str:
        """Render the window as plain text."""
        lines: list[str] = []
        if include_summary and self.summary:
            lines.append(f"[summary] {self.summary}")
        for message in self.messages:
            role = message.role.value if isinstance(message.role, Role) else str(message.role)
            lines.append(f"{role}: {ellipsize(message.text(), 600)}")
            for call in message.tool_calls:
                lines.append(f"  -> {call.name}(...)")
        return ellipsize("\n".join(lines), limit)

    def last_user_message(self) -> Message | None:
        """Return the most recent user turn."""
        for message in reversed(self.messages):
            if message.role is Role.USER:
                return message
        return None

    def token_estimate(self, *, estimator: TokenEstimator | None = None) -> int:
        """Estimate the token cost of the visible window."""
        counter = estimator or TokenEstimator()
        return counter.estimate_messages(self.visible())

    def to_dict(self, *, include_messages: bool = True) -> dict[str, Any]:
        """Serialize the window for session persistence."""
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "messages": len(self.messages),
            "summary": self.summary,
            "summarized_messages": self.summarized_messages,
            "trims": self.trims,
            "system_prompt": self.system_prompt,
            "pending": len(self.pending),
        }
        if include_messages:
            data["history"] = [message.to_dict() for message in self.messages]
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConversationWindow:
        """Rebuild a window from a persisted dictionary."""
        window = cls(
            session_id=str(raw.get("session_id", "default")),
            summary=str(raw.get("summary", "")),
            summarized_messages=int(raw.get("summarized_messages", 0)),
            trims=int(raw.get("trims", 0)),
            system_prompt=str(raw.get("system_prompt", "")),
        )
        window.pending = [str(item) for item in (raw.get("pending_text") or [])]
        window.messages = [
            Message.from_dict(item) for item in (raw.get("history") or []) if isinstance(item, dict)
        ]
        return window

    def __len__(self) -> int:
        """Return how many messages the window holds."""
        return len(self.messages)

    def __repr__(self) -> str:
        """Show the session, message count, and summary length."""
        return (
            f"<ConversationWindow session={self.session_id!r} messages={len(self.messages)} "
            f"summary_chars={len(self.summary)}>"
        )


class ShortTermMemory:
    """Per-session conversation windows with token-aware trimming."""

    def __init__(
        self,
        *,
        config: Any = None,
        backend: Any = None,
        estimator: TokenEstimator | None = None,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        trim_strategy: str = "middle",
        summarize: bool = True,
        summary_threshold: int = 24,
        summary_model: str = "",
        keep_last: int = 6,
        summary_batch: int = 4,
    ) -> None:
        """Configure the window policy.

        Args:
            config: a :class:`~magent.config.MemoryConfig` supplying defaults
            backend: LLM backend used to write summaries
            estimator: token estimator for budget accounting
            max_messages: turns kept before trimming
            max_tokens: token ceiling for the visible window
            trim_strategy: ``middle``, ``oldest``, or ``newest``
            summarize: summarize evicted turns instead of dropping them
            summary_threshold: message count that triggers summarization
            summary_model: model override for the summarization call
            keep_last: turns always kept verbatim
            summary_batch: buffered evictions before a summarization call
        """
        self.config = config
        self.backend = backend
        self.estimator = estimator or TokenEstimator()
        self.max_messages = int(getattr(config, "max_messages", max_messages) or max_messages)
        self.max_tokens = int(getattr(config, "max_context_tokens", max_tokens) or max_tokens)
        self.trim_strategy = str(getattr(config, "trim_strategy", trim_strategy) or trim_strategy)
        self.summarize_enabled = bool(getattr(config, "summarize", summarize))
        self.summary_threshold = int(
            getattr(config, "summary_threshold", summary_threshold) or summary_threshold
        )
        self.summary_model = str(getattr(config, "summary_model", summary_model) or summary_model)
        self.keep_last = max(2, int(keep_last))
        self.summary_batch = max(1, int(summary_batch))
        self._windows: dict[str, ConversationWindow] = {}
        self._lock = threading.RLock()
        self.summaries_written = 0
        self.messages_trimmed = 0

    # -- access --------------------------------------------------------------

    def window(self, session_id: str = "default") -> ConversationWindow:
        """Return the window for a session, creating it on first use."""
        with self._lock:
            existing = self._windows.get(session_id)
            if existing is None:
                existing = ConversationWindow(session_id=session_id)
                self._windows[session_id] = existing
            return existing

    def sessions(self) -> list[str]:
        """Return every known session identifier."""
        with self._lock:
            return sorted(self._windows)

    def add(self, message: Message, *, session_id: str = "default") -> ConversationWindow:
        """Append a message and apply the window policy."""
        window = self.window(session_id)
        window.append(message)
        self.enforce(session_id)
        return window

    def extend(
        self, messages: Sequence[Message], *, session_id: str = "default"
    ) -> ConversationWindow:
        """Append several messages and apply the window policy once."""
        window = self.window(session_id)
        window.extend(messages)
        self.enforce(session_id)
        return window

    def messages(
        self, session_id: str = "default", *, include_summary: bool = True
    ) -> list[Message]:
        """Return the messages to send for a session."""
        return self.window(session_id).visible(include_summary=include_summary)

    def transcript(self, session_id: str = "default", *, limit: int = 4000) -> str:
        """Return the session transcript as text."""
        return self.window(session_id).transcript(limit=limit, include_summary=True)

    def clear(self, session_id: str | None = None) -> int:
        """Clear one session, or every session when ``session_id`` is ``None``."""
        with self._lock:
            if session_id is None:
                count = sum(len(window) for window in self._windows.values())
                self._windows.clear()
                return count
            window = self._windows.get(session_id)
            return window.clear() if window else 0

    # -- policy --------------------------------------------------------------

    def enforce(self, session_id: str = "default") -> dict[str, Any]:
        """Trim and summarize a window until it fits the budget.

        Returns a report of what happened, which the CLI prints under
        ``--verbose`` and the HTTP API exposes for debugging.
        """
        window = self.window(session_id)
        report: dict[str, Any] = {
            "session_id": session_id,
            "trimmed": 0,
            "summarized": False,
            "messages": len(window),
            "tokens": 0,
        }
        if len(window) > max(self.keep_last, self.max_messages):
            report.update(self._trim_by_count(window))
        if self.max_tokens > 0:
            report.update(self._trim_by_tokens(window))
        report["messages"] = len(window)
        report["tokens"] = window.token_estimate(estimator=self.estimator)
        return report

    def _trim_by_count(self, window: ConversationWindow) -> dict[str, Any]:
        """Evict the oldest turns, summarizing them first when enabled."""
        limit = max(self.keep_last, self.max_messages)
        overflow = len(window.messages) - limit
        if overflow <= 0:
            return {"trimmed": 0, "summarized": False}
        head = (
            window.messages[:1]
            if window.messages and window.messages[0].role is Role.SYSTEM
            else []
        )
        body = window.messages[len(head) :]
        evicted = body[:overflow]
        kept = body[overflow:]
        summarized = False
        if self.summarize_enabled:
            window.queue_evicted(evicted)
            summarized = self._maybe_summarize(window)
        window.messages = head + kept
        window.trims += 1
        self.messages_trimmed += len(evicted)
        LOGGER.debug("trimmed %d message(s) from session %s", len(evicted), window.session_id)
        return {"trimmed": len(evicted), "summarized": summarized}

    def _trim_by_tokens(self, window: ConversationWindow) -> dict[str, Any]:
        """Trim until the visible window fits the token ceiling."""
        trimmed, report = trim_messages(
            window.messages,
            max_tokens=self.max_tokens,
            estimator=self.estimator,
            keep_system=True,
            keep_last=self.keep_last,
            strategy=self.trim_strategy,
        )
        dropped = int(report.get("dropped") or 0)
        if dropped:
            summarized = False
            if self.summarize_enabled:
                removed = [message for message in window.messages if message not in trimmed]
                window.queue_evicted(removed)
                summarized = self._maybe_summarize(window)
            report["summarized"] = summarized
            window.messages = trimmed
            window.trims += 1
            self.messages_trimmed += dropped
        return {
            "trimmed": dropped,
            "tokens_before": report.get("tokens_before", 0),
            "tokens_after": report.get("tokens_after", 0),
            "strategy": report.get("strategy", ""),
        }

    def _maybe_summarize(self, window: ConversationWindow) -> bool:
        """Summarize the eviction buffer once it is large enough to be worth it."""
        if self.backend is None or len(window.pending) < self.summary_batch:
            return False
        text = self.write_summary_text(list(window.pending), existing=window.summary)
        if not text:
            return False
        consumed = len(window.pending)
        window.pending.clear()
        window.set_summary(text, consumed=consumed)
        self.summaries_written += 1
        return True

    def write_summary_text(
        self, lines: Sequence[str], *, existing: str = "", max_words: int = 300
    ) -> str:
        """Summarize buffered transcript lines into the rolling summary.

        The previous summary is included so the new one supersedes it, but it is
        length-capped: without that cap a rolling summary grows without bound
        across many trims and eventually costs more than the turns it replaced.
        """
        transcript = "\n".join(line.strip() for line in lines if line.strip())
        if not transcript:
            return existing
        prompt = SUMMARIZE_PROMPT.render(
            transcript=ellipsize(transcript, 12000), max_words=str(max_words)
        )
        if existing:
            prompt = f"Previous summary (supersede it, do not repeat it):\n{ellipsize(existing, 1200)}\n\n{prompt}"
        return self._call_summarizer(prompt, existing, max_words)

    def write_summary(
        self, messages: Sequence[Message], *, existing: str = "", max_words: int = 300
    ) -> str:
        """Summarize evicted turns into the rolling summary."""
        transcript = "\n".join(
            f"{message.role.value}: {ellipsize(message.text(), 500)}" for message in messages
        ).strip()
        if not transcript:
            return existing
        prompt = SUMMARIZE_PROMPT.render(
            transcript=ellipsize(transcript, 12000), max_words=str(max_words)
        )
        if existing:
            prompt = f"Previous summary (supersede it, do not repeat it):\n{ellipsize(existing, 1200)}\n\n{prompt}"
        return self._call_summarizer(prompt, existing, max_words)

    def _call_summarizer(self, prompt: str, existing: str, max_words: int) -> str:
        """Call the backend for a summary, keeping the old one on failure."""
        if self.backend is None:
            return existing
        try:
            from ..llm.base import CompletionRequest

            response = self.backend.complete(
                CompletionRequest(
                    messages=[Message.user(prompt)],
                    model=self.summary_model,
                    temperature=0.2,
                    max_tokens=max_words * 2,
                )
            )
        except Exception as exc:
            LOGGER.warning("summarization failed, keeping the old summary: %s", exc)
            return existing
        text = (getattr(response, "text", "") or "").strip()
        return text or existing

    # -- persistence and telemetry ------------------------------------------

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        """Serialize every window."""
        with self._lock:
            return {
                "sessions": {
                    key: window.to_dict(include_messages=include_messages)
                    for key, window in self._windows.items()
                },
                "policy": {
                    "max_messages": self.max_messages,
                    "max_tokens": self.max_tokens,
                    "trim_strategy": self.trim_strategy,
                    "summarize": self.summarize_enabled,
                    "summary_threshold": self.summary_threshold,
                    "keep_last": self.keep_last,
                    "summary_batch": self.summary_batch,
                },
                "summaries_written": self.summaries_written,
                "messages_trimmed": self.messages_trimmed,
            }

    def load_dict(self, raw: dict[str, Any]) -> int:
        """Restore windows from :meth:`to_dict` output."""
        sessions = raw.get("sessions") or {}
        count = 0
        with self._lock:
            for key, value in sessions.items():
                if isinstance(value, dict):
                    self._windows[str(key)] = ConversationWindow.from_dict(value)
                    count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """Return window telemetry."""
        with self._lock:
            windows = list(self._windows.values())
        return {
            "sessions": len(windows),
            "messages": sum(len(window) for window in windows),
            "summaries": sum(1 for window in windows if window.summary),
            "summaries_written": self.summaries_written,
            "messages_trimmed": self.messages_trimmed,
            "trims": sum(window.trims for window in windows),
            "pending": sum(len(window.pending) for window in windows),
            "tokens": sum(window.token_estimate(estimator=self.estimator) for window in windows),
        }

    def __len__(self) -> int:
        """Return how many sessions are tracked."""
        return len(self._windows)

    def __repr__(self) -> str:
        """Show session count and policy."""
        return (
            f"<ShortTermMemory sessions={len(self._windows)} "
            f"max_messages={self.max_messages} max_tokens={self.max_tokens}>"
        )


__all__ = [
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_TOKENS",
    "ConversationWindow",
    "ShortTermMemory",
]
