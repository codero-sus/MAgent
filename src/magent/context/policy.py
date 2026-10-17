"""How a fixed context window is divided between competing demands.

An agent run has to fit a system prompt, retrieved documents, remembered
facts, and the conversation so far into one model window, and those four want
the same tokens. Without an explicit policy the last one written wins, which
in practice means a long tool output silently evicts the instructions that
tell the agent how to behave.

:class:`ContextPolicy` makes the division explicit and ordered: each section
gets a share of the window, required sections are satisfied first, and
whatever is left over flows to the sections that can still use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..errors import ConfigValidationError

STRATEGIES: tuple[str, ...] = ("trim", "summarize", "truncate", "window")
"""Compression strategies a section may use when it does not fit."""

SECTION_NAMES: tuple[str, ...] = (
    "system",
    "instructions",
    "tools",
    "retrieved",
    "memory",
    "history",
)
"""Sections in the order they are laid out in a prompt."""

DEFAULT_SHARES: dict[str, float] = {
    "system": 0.18,
    "instructions": 0.07,
    "tools": 0.10,
    "retrieved": 0.25,
    "memory": 0.10,
    "history": 0.30,
}
"""Default fraction of the *input* window each section may occupy."""

DEFAULT_CONTEXT_SIZE = 8192
"""Window size assumed when neither the model nor the config states one."""

DEFAULT_OUTPUT_RESERVE = 1024
"""Tokens held back for the reply when nothing else specifies a maximum."""

REQUIRED_SECTIONS: frozenset[str] = frozenset({"system", "history"})
"""Sections that are never dropped entirely, only compressed."""


@dataclass(slots=True)
class SectionPolicy:
    """Budget and compression rules for one prompt section."""

    name: str
    share: float = 0.1
    strategy: str = "trim"
    required: bool = False
    min_tokens: int = 64
    max_tokens: int = 0
    keep_last: int = 0
    compress_above: float = 0.9

    def __post_init__(self) -> None:
        """Clamp values into usable ranges."""
        self.name = str(self.name)
        self.share = min(1.0, max(0.0, float(self.share)))
        if self.strategy not in STRATEGIES:
            self.strategy = "trim"
        self.required = bool(self.required)
        self.min_tokens = max(0, int(self.min_tokens))
        self.max_tokens = max(0, int(self.max_tokens))
        self.keep_last = max(0, int(self.keep_last))
        self.compress_above = min(1.0, max(0.0, float(self.compress_above)))

    def budget(self, available: int) -> int:
        """Return this section's token allowance for a window of ``available``.

        Args:
            available: tokens left for the whole prompt after the output reserve

        Returns:
            The section's ceiling, honouring ``min_tokens`` and ``max_tokens``.
        """
        share_tokens = int(available * self.share)
        ceiling = min(share_tokens, self.max_tokens) if self.max_tokens else share_tokens
        return max(self.min_tokens, ceiling) if available else 0

    def needs_compression(self, used: int, budget: int) -> bool:
        """Return whether ``used`` tokens should trigger compression."""
        if budget <= 0:
            return used > 0
        return used >= budget * self.compress_above

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for logging and debugging."""
        return {
            "name": self.name,
            "share": round(self.share, 4),
            "strategy": self.strategy,
            "required": self.required,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "keep_last": self.keep_last,
        }


@dataclass(slots=True)
class ContextPolicy:
    """A complete window budget: sizes, reserves, and per-section rules."""

    context_size: int = 8192
    output_reserve: int = 1024
    safety_margin: float = 0.05
    sections: dict[str, SectionPolicy] = field(default_factory=dict)
    strategy: str = "trim"
    summarize_model: str = ""
    drop_tool_results_first: bool = True

    def __post_init__(self) -> None:
        """Fill in any section the caller did not specify."""
        self.context_size = max(512, int(self.context_size))
        self.output_reserve = max(0, int(self.output_reserve))
        self.safety_margin = min(0.5, max(0.0, float(self.safety_margin)))
        if self.strategy not in STRATEGIES:
            self.strategy = "trim"
        if not self.sections:
            self.sections = default_sections()
        for name in SECTION_NAMES:
            self.sections.setdefault(name, SectionPolicy(name=name,
                                                         share=DEFAULT_SHARES[name],
                                                         required=name in REQUIRED_SECTIONS))
        self.sections["system"].required = True
        self.sections["history"].required = True

    def validate(self) -> ContextPolicy:
        """Reject a policy that cannot leave room for anything.

        Raises:
            ConfigValidationError: the output reserve consumes the whole window
        """
        if self.output_reserve >= self.context_size:
            raise ConfigValidationError(
                "context output_reserve must be smaller than context_size",
                details={"output_reserve": self.output_reserve, "context_size": self.context_size},
            )
        return self

    @property
    def input_tokens(self) -> int:
        """Tokens available for the prompt after reserving room for output."""
        usable = self.context_size - self.output_reserve
        return max(0, int(usable * (1.0 - self.safety_margin)))

    def budgets(self) -> dict[str, int]:
        """Return each section's token allowance.

        Shares are normalized so they sum to at most the input window, and any
        slack left by a section that is not present is redistributed to the
        sections that can still absorb it.
        """
        available = self.input_tokens
        total_share = sum(section.share for section in self.sections.values()) or 1.0
        scale = min(1.0, 1.0 / total_share) if total_share > 1.0 else 1.0
        raw = {name: int(available * section.share * scale)
               for name, section in self.sections.items()}
        assigned = sum(raw.values())
        slack = max(0, available - assigned)
        if slack:
            # Give the remainder to history, the section that always wants more.
            raw["history"] = raw.get("history", 0) + slack
        return raw

    def section(self, name: str) -> SectionPolicy:
        """Return one section's policy, creating a neutral default if absent."""
        if name not in self.sections:
            self.sections[name] = SectionPolicy(name=name, share=0.0)
        return self.sections[name]

    def strategy_for(self, name: str) -> str:
        """Return the compression strategy for one section."""
        return self.section(name).strategy or self.strategy

    def clone(self, **overrides: Any) -> ContextPolicy:
        """Return a copy with selected fields replaced."""
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view including every section budget."""
        return {
            "context_size": self.context_size,
            "output_reserve": self.output_reserve,
            "safety_margin": self.safety_margin,
            "input_tokens": self.input_tokens,
            "strategy": self.strategy,
            "budgets": self.budgets(),
            "sections": {name: section.to_dict() for name, section in self.sections.items()},
        }


def default_sections() -> dict[str, SectionPolicy]:
    """Return the section policies MAgent uses out of the box.

    The defaults encode three lessons: instructions are small but must survive,
    retrieved context is the most useful thing to cut when space is tight, and
    history is the only section worth summarizing rather than dropping.
    """
    return {
        "system": SectionPolicy(name="system", share=DEFAULT_SHARES["system"],
                                strategy="truncate", required=True, min_tokens=256),
        "instructions": SectionPolicy(name="instructions", share=DEFAULT_SHARES["instructions"],
                                      strategy="truncate", required=True, min_tokens=128),
        "tools": SectionPolicy(name="tools", share=DEFAULT_SHARES["tools"],
                               strategy="trim", min_tokens=128),
        "retrieved": SectionPolicy(name="retrieved", share=DEFAULT_SHARES["retrieved"],
                                   strategy="trim", min_tokens=0, compress_above=0.8),
        "memory": SectionPolicy(name="memory", share=DEFAULT_SHARES["memory"],
                                strategy="trim", min_tokens=0, compress_above=0.8),
        "history": SectionPolicy(name="history", share=DEFAULT_SHARES["history"],
                                 strategy="trim", required=True, min_tokens=512, keep_last=4),
    }


def policy_from_config(config: Any, *, context_size: int = 0) -> ContextPolicy:
    """Build a :class:`ContextPolicy` from a resolved :class:`~magent.config.Config`.

    Anything the configuration file does not mention keeps its default, and an
    unrecognized ``[context]`` table is read out of ``config.extra`` so users
    can tune shares without MAgent having to version its config schema.

    Args:
        config: a resolved :class:`~magent.config.Config`, or ``None``
        context_size: override for the model's window size

    Returns:
        A validated policy.
    """
    window = context_size
    extra: dict[str, Any] = {}
    if config is not None:
        model = getattr(config, "model", None)
        if not window:
            window = int(getattr(model, "context_size", 0) or 0)
        generation = getattr(config, "generation", None)
        extra = dict(getattr(config, "extra", {}) or {})
        extra.setdefault("_max_tokens", int(getattr(generation, "max_tokens", 0) or 0))
    table = dict(extra.get("context") or {})
    table.pop("_max_tokens", None)
    window = int(table.pop("context_size", 0) or window or DEFAULT_CONTEXT_SIZE)
    wanted = int(table.pop("output_reserve", 0) or extra.get("_max_tokens", 0)
                 or DEFAULT_OUTPUT_RESERVE)
    # A reserve larger than the window leaves nothing to prompt with, and small
    # models are common enough that clamping beats raising here: the caller may
    # still be about to override either number.
    reserve = max(1, min(wanted, window - max(64, window // 8)))
    sections: dict[str, SectionPolicy] = {}
    for name, value in (table.pop("sections", {}) or {}).items():
        if isinstance(value, dict):
            sections[str(name)] = SectionPolicy(name=str(name), **value)
    kwargs = {key: value for key, value in table.items()
              if key in {"safety_margin", "strategy", "summarize_model", "drop_tool_results_first"}}
    return ContextPolicy(context_size=window, output_reserve=reserve,
                         sections=sections, **kwargs).validate()


__all__ = [
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_OUTPUT_RESERVE",
    "DEFAULT_SHARES",
    "REQUIRED_SECTIONS",
    "SECTION_NAMES",
    "STRATEGIES",
    "ContextPolicy",
    "SectionPolicy",
    "default_sections",
    "policy_from_config",
]
