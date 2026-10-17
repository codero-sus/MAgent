"""Composable system prompt construction with a token budget.

A system prompt is assembled from named sections in a fixed order, each with a
weight that decides what gets trimmed when the budget is tight. That makes the
prompt inspectable (:meth:`SystemPromptBuilder.to_dict` shows exactly which
section contributed what) and reproducible, instead of one f-string built in
five places.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.tokens import TokenEstimator, truncate_to_tokens
from ..utils.text import ellipsize
from .personas import DEFAULT_PERSONA, Persona, get_persona
from .templates import (
    SYSTEM_ENVIRONMENT,
    SYSTEM_IDENTITY,
    SYSTEM_OUTPUT_RULES,
    SYSTEM_SAFETY_RULES,
    SYSTEM_TOOL_POLICY,
)

SECTION_ORDER: tuple[str, ...] = (
    "identity",
    "persona",
    "environment",
    "rules",
    "tools",
    "context",
    "memory",
    "task",
    "examples",
    "extra",
)
"""Canonical section order for a system prompt."""

SECTION_WEIGHTS: dict[str, int] = {
    "identity": 100,
    "persona": 80,
    "environment": 60,
    "rules": 95,
    "tools": 70,
    "context": 50,
    "memory": 30,
    "task": 90,
    "examples": 20,
    "extra": 40,
}
"""Default trim priority per section; higher survives longer."""


@dataclass(slots=True)
class PromptSection:
    """One named block of a system prompt."""

    name: str
    text: str = ""
    weight: int = 50
    required: bool = False
    max_chars: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        """Trim the body to ``max_chars`` when a ceiling was declared."""
        if self.max_chars and len(self.text) > self.max_chars:
            self.text = ellipsize(self.text, self.max_chars)

    @property
    def chars(self) -> int:
        """Return the length of the section body."""
        return len(self.text)

    def is_empty(self) -> bool:
        """Report whether the section contributes nothing."""
        return not self.text.strip()

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Serialize the section, optionally omitting the body."""
        data: dict[str, Any] = {
            "name": self.name,
            "weight": self.weight,
            "required": self.required,
            "chars": self.chars,
            "source": self.source,
        }
        if include_text:
            data["text"] = self.text
        return data


class SystemPromptBuilder:
    """Assemble a system prompt from weighted, ordered sections."""

    def __init__(self, *, estimator: TokenEstimator | None = None, max_tokens: int = 0) -> None:
        """Create a builder with an optional token ceiling.

        Args:
            estimator: token estimator used for budget accounting
            max_tokens: when positive, :meth:`build` trims to fit
        """
        self.estimator = estimator or TokenEstimator()
        self.max_tokens = max(0, int(max_tokens))
        self._sections: dict[str, PromptSection] = {}

    # -- construction --------------------------------------------------------

    def add(
        self,
        name: str,
        text: str,
        *,
        weight: int | None = None,
        required: bool = False,
        max_chars: int = 0,
        source: str = "",
    ) -> SystemPromptBuilder:
        """Add or replace one section; returns ``self`` for chaining."""
        if not str(text or "").strip():
            self._sections.pop(name, None)
            return self
        self._sections[name] = PromptSection(
            name=name,
            text=str(text).strip(),
            weight=SECTION_WEIGHTS.get(name, 50) if weight is None else int(weight),
            required=required,
            max_chars=max_chars,
            source=source or name,
        )
        return self

    def extend(self, sections: Iterable[tuple[str, str]], **kwargs: Any) -> SystemPromptBuilder:
        """Add several ``(name, text)`` sections at once."""
        for name, text in sections:
            self.add(name, text, **kwargs)
        return self

    def remove(self, name: str) -> bool:
        """Drop a section; report whether it existed."""
        return self._sections.pop(name, None) is not None

    def identity(
        self,
        *,
        agent_name: str = "MAgent",
        tagline: str = "a capable local agent",
        backend: str = "Cortex LLMHoster",
        language: str = "the user's language",
    ) -> SystemPromptBuilder:
        """Set the identity section from the standard template."""
        return self.add(
            "identity",
            SYSTEM_IDENTITY.render(
                agent_name=agent_name, tagline=tagline, backend=backend, language=language
            ),
            required=True,
            source="template:system_identity",
        )

    def persona(self, persona: Persona | str = DEFAULT_PERSONA) -> SystemPromptBuilder:
        """Set the persona section from a :class:`Persona` or a name."""
        resolved = persona if isinstance(persona, Persona) else get_persona(str(persona))
        return self.add("persona", resolved.prompt(), source=f"persona:{resolved.name}")

    def environment(
        self, *, workspace: str | Path | None = None, model: str = "", extras: str = ""
    ) -> SystemPromptBuilder:
        """Set the environment section with runtime facts."""
        return self.add(
            "environment",
            SYSTEM_ENVIRONMENT.render(
                workspace=str(workspace or Path.cwd()),
                platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
                date=time.strftime("%Y-%m-%d %H:%M %Z").strip(),
                model=model or "the configured model",
                extras=extras,
            ),
            source="runtime",
        )

    def rules(
        self, *, tools: bool = True, safety: bool = True, output: bool = True
    ) -> SystemPromptBuilder:
        """Set the rules section from the standard rule templates."""
        blocks = []
        if tools:
            blocks.append(SYSTEM_TOOL_POLICY.render())
        if safety:
            blocks.append(SYSTEM_SAFETY_RULES.render())
        if output:
            blocks.append(SYSTEM_OUTPUT_RULES.render())
        return self.add("rules", "\n\n".join(blocks), required=True, source="template:rules")

    def tools(self, block: str, *, max_chars: int = 6000) -> SystemPromptBuilder:
        """Set the tools section, typically a registry prompt block."""
        return self.add("tools", block, max_chars=max_chars, source="registry")

    def context(self, block: str, *, max_chars: int = 8000) -> SystemPromptBuilder:
        """Set retrieved task context."""
        return self.add("context", block, max_chars=max_chars, source="rag")

    def memory(self, block: str, *, max_chars: int = 3000) -> SystemPromptBuilder:
        """Set recalled memories."""
        return self.add("memory", block, max_chars=max_chars, source="memory")

    def task(self, block: str) -> SystemPromptBuilder:
        """Set a task-specific instruction block."""
        return self.add("task", block, source="task")

    def examples(self, block: str, *, max_chars: int = 2000) -> SystemPromptBuilder:
        """Set few-shot examples."""
        return self.add("examples", block, max_chars=max_chars, source="few_shot")

    def extra(self, block: str) -> SystemPromptBuilder:
        """Set user supplied extra instructions."""
        return self.add("extra", block, source="config")

    # -- inspection ----------------------------------------------------------

    def sections(self) -> list[PromptSection]:
        """Return the sections in canonical order, unknown names last."""
        known = [name for name in SECTION_ORDER if name in self._sections]
        unknown = sorted(name for name in self._sections if name not in SECTION_ORDER)
        return [self._sections[name] for name in known + unknown]

    def names(self) -> list[str]:
        """Return the section names in render order."""
        return [section.name for section in self.sections()]

    def get(self, name: str) -> PromptSection | None:
        """Return one section by name."""
        return self._sections.get(name)

    def text(self) -> str:
        """Return the untrimmed prompt body."""
        return "\n\n".join(section.text for section in self.sections() if not section.is_empty())

    def token_estimate(self) -> int:
        """Estimate the prompt's token count."""
        return self.estimator.estimate(self.text())

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize the prompt structure for debugging and the HTTP API."""
        sections = self.sections()
        return {
            "sections": [section.to_dict(include_text=include_text) for section in sections],
            "chars": sum(section.chars for section in sections),
            "tokens": self.token_estimate(),
            "max_tokens": self.max_tokens,
            "text": self.text() if include_text else "",
        }

    # -- budgeting -----------------------------------------------------------

    def trim_to(self, max_tokens: int) -> int:
        """Trim the prompt to a token budget; return tokens removed.

        The lowest-weight non-required sections are truncated or dropped first,
        so identity and rules survive even in a very small window.
        """
        target = max(0, int(max_tokens))
        if not target:
            return 0
        before = self.token_estimate()
        if before <= target:
            return 0
        ordered = sorted(
            (section for section in self.sections() if not section.required),
            key=lambda section: (section.weight, -section.chars),
        )
        for section in ordered:
            if self.token_estimate() <= target:
                break
            current = self.estimator.estimate(section.text)
            if current <= 0:
                continue
            overshoot = self.token_estimate() - target
            keep = max(0, current - overshoot)
            if keep < 40:
                self.remove(section.name)
                continue
            truncated = truncate_to_tokens(section.text, keep, estimator=self.estimator)
            self.add(
                section.name, truncated, weight=section.weight, max_chars=0, source=section.source
            )
        return max(0, before - self.token_estimate())

    def build(self, *, max_tokens: int | None = None) -> str:
        """Render the final prompt, applying the token budget when set."""
        ceiling = self.max_tokens if max_tokens is None else int(max_tokens)
        if ceiling:
            self.trim_to(ceiling)
        return self.text()

    def __len__(self) -> int:
        """Return how many sections are present."""
        return len(self._sections)

    def __repr__(self) -> str:
        """Show the section names and the estimated token count."""
        return f"<SystemPromptBuilder sections={self.names()} tokens={self.token_estimate()}>"


def describe_environment(
    workspace: str | Path | None = None,
    *,
    model: str = "",
    include_files: bool = False,
    max_files: int = 15,
    include_git: bool = True,
) -> str:
    """Build the extras block of runtime facts about the workspace.

    Args:
        workspace: the agent's working directory
        model: the model serving this run
        include_files: list a few workspace files for orientation
        max_files: how many files to list
        include_git: report the current git branch when the workspace is a repo
    """
    root = Path(workspace or os.getcwd()).expanduser()
    lines: list[str] = []
    try:
        entries = sorted(entry.name for entry in root.iterdir() if not entry.name.startswith("."))
    except OSError:
        entries = []
    lines.append(f"- Workspace entries: {len(entries)}")
    if include_files and entries:
        lines.append(
            f"- Top level: {', '.join(entries[:max_files])}"
            + (" ..." if len(entries) > max_files else "")
        )
    if include_git:
        head = root / ".git" / "HEAD"
        try:
            if head.is_file():
                ref = head.read_text(encoding="utf-8").strip()
                if ref.startswith("ref: "):
                    lines.append(f"- Git branch: {ref[5:].split('/')[-1]}")
                else:
                    lines.append(f"- Git detached at: {ref[:12]}")
        except OSError:
            pass
    if model:
        lines.append(f"- Active model: {model}")
    python_version = platform.python_version()
    lines.append(f"- Python: {python_version}")
    return "\n".join(lines)


def tool_prompt_block(
    registry: Any, *, style: str = "compact", max_chars: int = 6000, include_hidden: bool = False
) -> str:
    """Render a registry's tools into a prompt block with a size ceiling.

    Falls back to a one-line-per-tool listing when the verbose rendering would
    not fit, so a large catalogue never crowds out the rules.
    """
    if registry is None:
        return ""
    block = registry.prompt_block(include_hidden=include_hidden, style=style)
    if len(block) <= max_chars:
        return block
    compact = registry.prompt_block(include_hidden=include_hidden, style="compact")
    if len(compact) <= max_chars:
        return compact
    names = registry.names()
    header = (
        f"{len(names)} tools available: {', '.join(names)}\n"
        "Call describe_tool(name) for the exact arguments of any of them."
    )
    return ellipsize(header, max_chars)


def build_system_prompt(
    *,
    agent_name: str = "MAgent",
    tagline: str = "Mr. Agent, Powerful.",
    persona: Persona | str = DEFAULT_PERSONA,
    backend_name: str = "Cortex LLMHoster",
    model: str = "",
    workspace: str | Path | None = None,
    registry: Any = None,
    context_block: str = "",
    memory_block: str = "",
    task_block: str = "",
    examples_block: str = "",
    extra_instructions: str = "",
    system_prompt_override: str = "",
    language: str = "the user's language",
    include_files: bool = False,
    max_tokens: int = 0,
    tools_style: str = "compact",
) -> str:
    """Assemble a complete system prompt from the standard parts.

    Args:
        agent_name: name the agent refers to itself by
        tagline: one-line description used in the identity block
        persona: persona name or instance
        backend_name: the inference backend description
        model: active model identifier
        workspace: working directory to describe
        registry: tool registry whose tools are advertised
        context_block: retrieved context to inject
        memory_block: recalled memories to inject
        task_block: task-specific instructions
        examples_block: few-shot examples
        extra_instructions: operator supplied additions
        system_prompt_override: replaces the whole prompt when non-empty
        language: language the agent should answer in
        include_files: list workspace files in the environment block
        max_tokens: trim the prompt to this many tokens, 0 for no ceiling
        tools_style: compact or verbose tool rendering
    """
    if system_prompt_override.strip():
        return system_prompt_override.strip()
    builder = SystemPromptBuilder(max_tokens=max_tokens)
    builder.identity(
        agent_name=agent_name, tagline=tagline, backend=backend_name, language=language
    )
    builder.persona(persona)
    builder.environment(
        workspace=workspace,
        model=model,
        extras=describe_environment(workspace, model=model, include_files=include_files),
    )
    builder.rules()
    if registry is not None:
        builder.tools(tool_prompt_block(registry, style=tools_style))
    if context_block:
        builder.context(context_block)
    if memory_block:
        builder.memory(memory_block)
    if task_block:
        builder.task(task_block)
    if examples_block:
        builder.examples(examples_block)
    if extra_instructions:
        builder.extra(extra_instructions)
    return builder.build()


__all__ = [
    "SECTION_ORDER",
    "SECTION_WEIGHTS",
    "PromptSection",
    "SystemPromptBuilder",
    "build_system_prompt",
    "describe_environment",
    "tool_prompt_block",
]
