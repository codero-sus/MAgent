"""Assembling one model request out of sections that compete for the window.

A prompt is not one string. It is a system identity, a persona, environment
facts, retrieved documents, remembered entries, the tool schema block, and the
conversation so far - each produced by a different subsystem, each happy to be
as long as it likes. :class:`ContextBuilder` is the place where those demands
meet a hard number.

The builder never silently drops a required section. It measures, compresses
the sections that can be compressed, and reports exactly what happened, so a
run that loses its retrieved context says so in the result instead of quietly
answering without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..llm.tokens import TokenEstimator, truncate_to_tokens
from ..logging_ import get_logger
from ..types import Message
from .compressor import CompressionResult, Summarizer, compress_messages
from .policy import ContextPolicy, SectionPolicy, policy_from_config

LOGGER = get_logger("context.build")
"""Logger for context assembly decisions."""


@dataclass(slots=True)
class ContextSection:
    """One named block of prompt text awaiting a budget decision."""

    name: str
    text: str = ""
    required: bool = False
    tokens: int = 0
    budget: int = 0
    dropped: bool = False
    truncated: bool = False
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for events and debug output."""
        return {
            "name": self.name,
            "tokens": self.tokens,
            "budget": self.budget,
            "required": self.required,
            "dropped": self.dropped,
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class ContextBuild:
    """The outcome of one assembly pass."""

    messages: list[Message] = field(default_factory=list)
    system: str = ""
    sections: dict[str, ContextSection] = field(default_factory=dict)
    tokens: int = 0
    budget: int = 0
    history_tokens: int = 0
    history_dropped: int = 0
    compression: CompressionResult | None = None
    fits: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        """Fraction of the input budget actually used."""
        if not self.budget:
            return 0.0
        return round(self.tokens / self.budget, 4)

    def dropped_sections(self) -> list[str]:
        """Names of sections that were removed entirely."""
        return [name for name, section in self.sections.items() if section.dropped]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for logging and the web UI."""
        return {
            "tokens": self.tokens,
            "budget": self.budget,
            "utilization": self.utilization,
            "fits": self.fits,
            "history_tokens": self.history_tokens,
            "history_dropped": self.history_dropped,
            "sections": {name: section.to_dict() for name, section in self.sections.items()},
            "compression": self.compression.to_dict() if self.compression else None,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """Return a one-line summary suitable for a verbose CLI."""
        kept = [name for name, section in self.sections.items()
                if section.tokens and not section.dropped]
        parts = [f"{self.tokens}/{self.budget} tokens ({self.utilization:.0%})",
                 f"{len(self.messages)} messages"]
        if kept:
            parts.append("sections: " + ",".join(kept))
        dropped = self.dropped_sections()
        if dropped:
            parts.append("dropped: " + ",".join(dropped))
        if self.history_dropped:
            parts.append(f"history -{self.history_dropped}")
        return " | ".join(parts)


class ContextBuilder:
    """Collect prompt sections and fit them, plus history, into a window."""

    def __init__(self, policy: ContextPolicy | None = None, *, config: Any = None,
                 estimator: TokenEstimator | None = None,
                 summarizer: Summarizer | None = None,
                 context_size: int = 0) -> None:
        """Create a builder.

        Args:
            policy: explicit policy; built from ``config`` when omitted
            config: a resolved :class:`~magent.config.Config`
            estimator: token counter, shared across sections when provided
            summarizer: used when the history strategy is ``summarize``
            context_size: override the window size taken from the policy
        """
        self.policy = policy or policy_from_config(config, context_size=context_size)
        if context_size:
            self.policy = self.policy.clone(context_size=context_size)
        self.estimator = estimator or TokenEstimator()
        self.summarizer = summarizer
        self._sections: dict[str, ContextSection] = {}
        self._order = 0

    # -- collecting ----------------------------------------------------------

    def set(self, name: str, text: str, *, required: bool | None = None) -> ContextBuilder:
        """Replace one section's text, returning the builder for chaining.

        Args:
            name: section name, matching a :class:`SectionPolicy` when possible
            text: the section body; empty clears it
            required: override the policy's required flag for this section
        """
        policy = self.policy.section(name)
        existing = self._sections.get(name)
        self._sections[name] = ContextSection(
            name=name,
            text=text or "",
            required=policy.required if required is None else bool(required),
            order=existing.order if existing else self._next_order(name),
        )
        return self

    def add(self, name: str, text: str, *, separator: str = "\n\n") -> ContextBuilder:
        """Append to a section, creating it when absent."""
        if not text:
            return self
        current = self._sections.get(name)
        if current is None or not current.text:
            return self.set(name, text)
        return self.set(name, f"{current.text}{separator}{text}")

    def extend(self, name: str, items: Iterable[str], *, separator: str = "\n") -> ContextBuilder:
        """Add many pieces to one section at once."""
        pieces = [str(item).strip() for item in items if str(item).strip()]
        return self.add(name, separator.join(pieces), separator=separator) if pieces else self

    def clear(self, name: str = "") -> ContextBuilder:
        """Drop one section, or all of them when no name is given."""
        if name:
            self._sections.pop(name, None)
        else:
            self._sections.clear()
        return self

    def _next_order(self, name: str) -> int:
        """Return the layout position for a newly added section."""
        from .policy import SECTION_NAMES

        if name in SECTION_NAMES:
            return SECTION_NAMES.index(name)
        self._order += 1
        return len(SECTION_NAMES) + self._order

    # -- measuring -----------------------------------------------------------

    def measure(self) -> dict[str, int]:
        """Return the uncompressed token count of every section."""
        return {name: self.estimator.estimate(section.text)
                for name, section in self._sections.items()}

    def total_tokens(self) -> int:
        """Return the tokens of all sections plus nothing else."""
        return sum(self.measure().values())

    def system_text(self) -> str:
        """Return the assembled system prompt from the static sections."""
        ordered = sorted((section for section in self._sections.values()
                          if section.name != "history" and section.text.strip()),
                         key=lambda section: section.order)
        return "\n\n".join(section.text.strip() for section in ordered).strip()

    # -- building ------------------------------------------------------------

    def build(self, history: Sequence[Message] | None = None, *,
              keep_last: int = 2) -> ContextBuild:
        """Fit every section plus ``history`` into the window.

        Required sections are compressed rather than dropped; optional sections
        are truncated to their share and dropped entirely if nothing would
        survive. History is compressed last, because it is the only part that
        grows without bound during a run.

        Args:
            history: the conversation so far, system message optional
            keep_last: turns of history protected from compression

        Returns:
            A :class:`ContextBuild` with the final messages and a full account.
        """
        build = ContextBuild(budget=self.policy.input_tokens)
        budgets = self.policy.budgets()
        history = list(history or [])

        # Static sections first: they are bounded and mostly required.
        for name, section in sorted(self._sections.items(), key=lambda item: item[1].order):
            if name == "history":
                continue
            ceiling = budgets.get(name, 0)
            tokens = self.estimator.estimate(section.text)
            section.budget = ceiling
            section.tokens = tokens
            policy = self.policy.section(name) if name in self.policy.sections \
                else SectionPolicy(name=name, share=0.0, required=section.required)
            if tokens > ceiling:
                if ceiling <= 0:
                    # No share at all: the section must not reach the prompt.
                    section.dropped = True
                    section.text = ""
                    section.tokens = 0
                    build.notes.append(f"{name}: dropped (no budget for {tokens} tokens)")
                else:
                    # Partial context beats none, so trim to the share instead of
                    # discarding the section. Clearing the text on a dropped
                    # section matters: system_text() reads the text, and a
                    # section marked dropped but still carrying its body would
                    # quietly consume the window it was refused.
                    section.text = truncate_to_tokens(section.text, ceiling,
                                                      estimator=self.estimator)
                    section.tokens = self.estimator.estimate(section.text)
                    section.truncated = True
                    build.notes.append(f"{name}: truncated {tokens}->{section.tokens} "
                                       f"({policy.strategy})")
            build.sections[name] = section

        build.system = self.system_text()
        messages: list[Message] = []
        if build.system:
            messages.append(Message.system(build.system))

        # History gets whatever the static sections did not use.
        static_tokens = self.estimator.estimate_messages(messages)
        room = max(256, build.budget - static_tokens)
        history_messages = [message for message in history if message.role != "system"]
        if history_messages:
            result = compress_messages(
                history_messages, budget=room,
                strategy=self.policy.strategy_for("history"),
                estimator=self.estimator, keep_last=keep_last,
                summarizer=self.summarizer,
                summary_tokens=min(400, max(64, room // 4)),
            )
            build.compression = result
            build.history_tokens = result.after_tokens
            build.history_dropped = result.dropped
            messages.extend(result.messages)
            if not result.fits:
                build.notes.append(f"history still over budget by "
                                   f"{result.after_tokens - room} tokens")
        build.messages = messages
        build.tokens = self.estimator.estimate_messages(messages)
        build.fits = build.tokens <= build.budget
        if not build.fits:
            build.notes.append(f"context over budget: {build.tokens} > {build.budget}")
        LOGGER.debug("context built: %s", build.describe())
        return build

    def fits(self, history: Sequence[Message] | None = None) -> bool:
        """Return whether the current sections plus history fit uncompressed."""
        tokens = self.total_tokens() + self.estimator.estimate_messages(list(history or []))
        return tokens <= self.policy.input_tokens

    def headroom(self, history: Sequence[Message] | None = None) -> int:
        """Return how many tokens remain for new content."""
        used = self.total_tokens() + self.estimator.estimate_messages(list(history or []))
        return max(0, self.policy.input_tokens - used)

    def report(self, history: Sequence[Message] | None = None) -> dict[str, Any]:
        """Return a diagnostic view without mutating any section."""
        build = self.build(history)
        payload = build.to_dict()
        payload["policy"] = self.policy.to_dict()
        return payload


def build_context(policy: ContextPolicy | None = None, *, sections: dict[str, str] | None = None,
                  history: Sequence[Message] | None = None, config: Any = None,
                  context_size: int = 0, output_reserve: int = 0,
                  summarizer: Summarizer | None = None) -> ContextBuild:
    """Convenience wrapper: assemble sections and history in one call.

    Args:
        policy: explicit policy, or derived from ``config``
        sections: mapping of section name to text
        history: the conversation so far
        config: resolved configuration used when no policy is given
        context_size: window override
        output_reserve: tokens held back for the model's reply
        summarizer: summarizer for the ``summarize`` history strategy

    Returns:
        The :class:`ContextBuild` produced by the assembled builder.
    """
    active = policy
    if context_size or output_reserve:
        base = active or policy_from_config(config, context_size=context_size)
        overrides: dict[str, Any] = {}
        if context_size:
            overrides["context_size"] = context_size
        if output_reserve:
            overrides["output_reserve"] = output_reserve
        active = base.clone(**overrides)
    builder = ContextBuilder(active, config=config if active is None else None,
                             summarizer=summarizer)
    for name, text in (sections or {}).items():
        builder.set(name, text)
    return builder.build(history)


__all__ = [
    "ContextBuild",
    "ContextBuilder",
    "ContextSection",
    "build_context",
]
