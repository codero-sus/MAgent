"""Agent presets: named, tuned starting points.

A preset bundles the choices that make an agent good at one kind of job - the
persona, which tools it may use, whether it plans or reflects, how hot its
sampling runs and how long it may work.  ``magent run --preset coder`` and
:func:`build_preset_agent` both go through this table, so the CLI, the API and
the tests all exercise the same configuration.

Presets are data, not code: :meth:`AgentPreset.merge` returns a copy with
overrides applied, so a caller can start from ``researcher`` and narrow the tool
set without touching the table.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..config import AgentConfig
from ..errors import ConfigError
from ..prompts.personas import PERSONAS

PRESET_CATEGORIES: tuple[str, ...] = ("general", "code", "research", "writing", "operations")
"""Groups used to organise presets in help output."""


@dataclass(slots=True, frozen=True)
class AgentPreset:
    """A reusable agent configuration."""

    name: str
    description: str
    persona: str = "assistant"
    category: str = "general"
    tools: tuple[str, ...] = ()
    disabled_tools: tuple[str, ...] = ()
    tool_categories: tuple[str, ...] = ()
    max_risk: str = ""
    no_tools: bool = False
    allow_network: bool | None = None
    loop_mode: str = "react"
    max_iterations: int = 8
    temperature: float | None = None
    planning: bool | None = None
    reflection: bool | None = None
    parallel_tools: bool | None = None
    model_role: str = ""
    instructions: str = ""
    examples: tuple[str, ...] = ()
    structured: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, **overrides: Any) -> AgentPreset:
        """Return a copy with fields overridden.

        Raises:
            ConfigError: when an override names a field this preset does not have
        """
        known = set(self.__dataclass_fields__)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ConfigError(
                f"unknown preset overrides: {', '.join(unknown)}",
                details={"known": sorted(known), "preset": self.name},
            )
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)

    def agent_config(self, base: AgentConfig | None = None) -> AgentConfig:
        """Project the preset onto an :class:`~magent.config.AgentConfig`."""
        config = base if base is not None else AgentConfig()
        config.persona = self.persona
        config.loop_mode = self.loop_mode
        config.max_iterations = self.max_iterations
        if self.planning is not None:
            config.planning = self.planning
        if self.reflection is not None:
            config.reflection = self.reflection
        if self.parallel_tools is not None:
            config.parallel_tools = self.parallel_tools
        if self.instructions:
            extra = config.extra_instructions or ""
            config.extra_instructions = f"{extra}\n{self.instructions}".strip()
        return config

    def tool_filter(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`~magent.tools.base.ToolRegistry.filter`."""
        out: dict[str, Any] = {}
        if self.tools:
            out["enabled"] = list(self.tools)
        if self.disabled_tools:
            out["disabled"] = list(self.disabled_tools)
        if self.tool_categories:
            out["categories"] = list(self.tool_categories)
        if self.max_risk:
            out["max_risk"] = self.max_risk
        if self.allow_network is not None:
            out["network"] = self.allow_network
        return out

    def build_registry(self, registry: Any = None) -> Any:
        """Return a registry restricted to this preset's tools.

        Args:
            registry: source registry; the default registry when omitted
        """
        from ..tools import ToolRegistry, default_registry

        if self.no_tools:
            return ToolRegistry()
        source: ToolRegistry = registry if registry is not None else default_registry()
        allowed = source.filter(**self.tool_filter())
        scoped = ToolRegistry()
        scoped.register_all(allowed)
        return scoped

    def to_dict(self) -> dict[str, Any]:
        """Serialise the preset for ``magent presets --format json``."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "persona": self.persona,
            "tools": list(self.tools),
            "disabled_tools": list(self.disabled_tools),
            "tool_categories": list(self.tool_categories),
            "max_risk": self.max_risk,
            "no_tools": self.no_tools,
            "allow_network": self.allow_network,
            "loop_mode": self.loop_mode,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "planning": self.planning,
            "reflection": self.reflection,
            "parallel_tools": self.parallel_tools,
            "model_role": self.model_role,
            "structured": self.structured,
            "instructions": self.instructions,
            "examples": list(self.examples),
            "metadata": dict(self.metadata),
        }

    def summary_line(self) -> str:
        """One-line description for help output."""
        bits = [self.persona, self.loop_mode, f"{self.max_iterations} iters"]
        if self.tools:
            bits.append(f"{len(self.tools)} tools")
        elif self.tool_categories:
            bits.append("/".join(self.tool_categories))
        return f"{self.name:<12} {self.description} ({', '.join(bits)})"


PRESETS: dict[str, AgentPreset] = {}
"""Registry of built-in presets, keyed by name."""


def _register(preset: AgentPreset) -> AgentPreset:
    """Add a preset to :data:`PRESETS` and return it."""
    PRESETS[preset.name] = preset
    return preset


_register(
    AgentPreset(
        name="general",
        description="Balanced default: reads the workspace, uses tools, explains its work",
        persona="assistant",
        category="general",
        loop_mode="react",
        max_iterations=8,
        disabled_tools=("python_exec", "shell_exec"),
    )
)

_register(
    AgentPreset(
        name="coder",
        description="Edits code in a repository: read first, small diffs, run the tests",
        persona="coder",
        category="code",
        tool_categories=("files", "text", "search", "git", "python", "shell", "system"),
        loop_mode="react",
        max_iterations=14,
        planning=True,
        parallel_tools=False,
        instructions=(
            "Read the file before editing it. Prefer targeted edits over rewrites. "
            "After a change, run the project's tests or a syntax check and report the result."
        ),
    )
)

_register(
    AgentPreset(
        name="reviewer",
        description="Reviews code or text and reports findings with severity and evidence",
        persona="reviewer",
        category="code",
        tool_categories=("files", "text", "search", "git"),
        max_risk="read",
        loop_mode="reflect",
        max_iterations=6,
        temperature=0.2,
        structured=True,
        instructions=(
            "Report findings as a list with severity, file, line and a concrete suggestion. "
            "Do not modify any file."
        ),
    )
)

_register(
    AgentPreset(
        name="researcher",
        description="Gathers evidence from the web and files, then cites what it used",
        persona="researcher",
        category="research",
        tools=(
            "web_fetch",
            "web_search",
            "read_file",
            "grep",
            "list_directory",
            "write_file",
            "memory_add",
            "memory_search",
            "rag_search",
            "rag_ingest",
            "calculate",
            "datetime_now",
            "notes_append",
        ),
        loop_mode="plan",
        max_iterations=12,
        planning=True,
        reflection=True,
        allow_network=True,
        temperature=0.3,
        instructions=(
            "Collect evidence before concluding. Cite the source URL or file path for every "
            "claim. Distinguish what you verified from what you inferred."
        ),
    )
)

_register(
    AgentPreset(
        name="analyst",
        description="Turns data files into findings: query, aggregate, chart, conclude",
        persona="analyst",
        category="research",
        tool_categories=("data", "files", "math", "text", "output"),
        loop_mode="plan",
        max_iterations=10,
        temperature=0.2,
        instructions=(
            "Show the numbers behind every claim. State the row count, the time range and any "
            "rows you excluded."
        ),
    )
)

_register(
    AgentPreset(
        name="writer",
        description="Drafts and edits prose with a consistent voice and structure",
        persona="writer",
        category="writing",
        tool_categories=("files", "text", "output"),
        max_risk="write",
        loop_mode="reflect",
        max_iterations=6,
        temperature=0.8,
        instructions=(
            "Match the requested tone and length. Revise once against your own draft before "
            "presenting the final text."
        ),
    )
)

_register(
    AgentPreset(
        name="devops",
        description="Inspects and operates systems carefully, asking before anything destructive",
        persona="devops",
        category="operations",
        tool_categories=("shell", "system", "files", "git", "web"),
        loop_mode="react",
        max_iterations=12,
        parallel_tools=False,
        allow_network=True,
        temperature=0.1,
        instructions=(
            "Inspect before changing anything. Never run a destructive command without an "
            "explicit approval, and always state the rollback step first."
        ),
    )
)

_register(
    AgentPreset(
        name="planner",
        description="Produces a step-by-step plan without executing it",
        persona="planner",
        category="general",
        tool_categories=("files", "search", "text"),
        max_risk="read",
        loop_mode="plan",
        max_iterations=4,
        temperature=0.3,
        structured=True,
        instructions=(
            "Output a numbered plan. For each step give the goal, the tools to use, the "
            "expected result and how to verify it. Do not execute the steps."
        ),
    )
)

_register(
    AgentPreset(
        name="tutor",
        description="Explains concepts step by step and checks understanding",
        persona="tutor",
        category="writing",
        tools=("read_file", "grep", "calculate", "datetime_now", "memory_search", "memory_add"),
        max_risk="read",
        loop_mode="single",
        max_iterations=3,
        temperature=0.6,
        instructions=(
            "Explain in small steps, define every term, and finish with one question that "
            "checks understanding."
        ),
    )
)

_register(
    AgentPreset(
        name="librarian",
        description="Indexes documents and answers only from retrieved, cited context",
        persona="researcher",
        category="research",
        tools=(
            "rag_ingest",
            "rag_search",
            "rag_status",
            "read_file",
            "list_directory",
            "directory_tree",
            "grep",
            "memory_search",
            "grounded_answer",
        ),
        max_risk="read",
        loop_mode="react",
        max_iterations=8,
        temperature=0.1,
        instructions=(
            "Answer only from retrieved context and cite the chunk numbers. Say so plainly "
            "when the corpus does not contain the answer."
        ),
    )
)

_register(
    AgentPreset(
        name="minimal",
        description="No tools, no planning: a single fast completion",
        persona="minimal",
        category="general",
        no_tools=True,
        loop_mode="single",
        max_iterations=1,
        temperature=0.4,
    )
)

_register(
    AgentPreset(
        name="sandboxed",
        description="Read-only reconnaissance: no writes, no network, no execution",
        persona="assistant",
        category="operations",
        max_risk="read",
        allow_network=False,
        disabled_tools=("python_exec", "shell_exec", "write_file", "edit_file", "delete_path"),
        loop_mode="react",
        max_iterations=6,
        instructions="You may only read. Report what you found and what you would change.",
    )
)


def list_presets(category: str = "") -> list[AgentPreset]:
    """Return presets, optionally filtered by category."""
    items = list(PRESETS.values())
    if category:
        items = [item for item in items if item.category == category]
    return sorted(items, key=lambda item: (item.category, item.name))


def preset_names() -> list[str]:
    """Names of every registered preset."""
    return sorted(PRESETS)


def get_preset(name: str) -> AgentPreset:
    """Look up a preset by name, with suggestions on a typo.

    Raises:
        ConfigError: when no preset matches
    """
    key = (name or "").strip().lower()
    if key in PRESETS:
        return PRESETS[key]
    suggestions = _closest(key, list(PRESETS))
    raise ConfigError(
        f"unknown preset {name!r}", details={"known": preset_names(), "did_you_mean": suggestions}
    )


def _closest(needle: str, options: Iterable[str], limit: int = 3) -> list[str]:
    """Return the option names closest to ``needle``."""
    from ..utils.text import similarity

    scored = sorted(((similarity(needle, option), option) for option in options), reverse=True)
    return [option for score, option in scored[:limit] if score > 0.4]


def register_preset(preset: AgentPreset, *, replace_existing: bool = False) -> AgentPreset:
    """Register a custom preset.

    Args:
        preset: the preset to add
        replace_existing: allow overwriting a built-in name

    Raises:
        ConfigError: when the name is taken and replacing is not allowed
    """
    if not preset.name:
        raise ConfigError("preset name is required")
    if preset.name in PRESETS and not replace_existing:
        raise ConfigError(
            f"preset {preset.name!r} already exists",
            details={"hint": "pass replace_existing=True to override"},
        )
    if preset.persona not in PERSONAS:
        raise ConfigError(
            f"preset {preset.name!r} uses unknown persona {preset.persona!r}",
            details={"known": sorted(PERSONAS)},
        )
    PRESETS[preset.name] = preset
    return preset


def presets_from_config(config: Any = None) -> dict[str, AgentPreset]:
    """Load presets declared under ``[presets.<name>]`` in a config file.

    A declared preset may ``extend`` a built-in one, in which case only the
    listed fields change.  Unknown keys are reported rather than silently
    ignored, so a typo in ``magent.toml`` is visible.
    """
    raw = getattr(config, "extra", None) or {}
    section = raw.get("presets") if isinstance(raw, dict) else None
    out: dict[str, AgentPreset] = {}
    if not isinstance(section, dict):
        return out
    text_fields = (
        "persona",
        "category",
        "loop_mode",
        "instructions",
        "max_risk",
        "model_role",
        "description",
    )
    int_fields = ("max_iterations",)
    float_fields = ("temperature",)
    bool_fields = (
        "planning",
        "reflection",
        "parallel_tools",
        "allow_network",
        "structured",
        "no_tools",
    )
    tuple_fields = ("tools", "disabled_tools", "tool_categories", "examples")
    for name, values in section.items():
        if not isinstance(values, dict):
            continue
        unknown = sorted(
            set(values)
            - {"extends"}
            - set(text_fields)
            - set(int_fields)
            - set(float_fields)
            - set(bool_fields)
            - set(tuple_fields)
        )
        base = PRESETS.get(str(values.get("extends", ""))) or AgentPreset(
            name=str(name), description=str(values.get("description", name))
        )
        overrides: dict[str, Any] = {}
        for key in text_fields:
            if key in values:
                overrides[key] = str(values[key])
        for key in int_fields:
            if key in values:
                overrides[key] = int(values[key])
        for key in float_fields:
            if key in values:
                overrides[key] = float(values[key])
        for key in bool_fields:
            if key in values:
                overrides[key] = bool(values[key])
        for key in tuple_fields:
            if key in values:
                overrides[key] = tuple(str(item) for item in values[key])
        preset = base.merge(**overrides)
        preset = replace(
            preset, name=str(name), description=overrides.get("description", base.description)
        )
        if unknown:
            preset.metadata["unknown_keys"] = unknown
        out[str(name)] = register_preset(preset, replace_existing=True)
    return out


def build_preset_agent(
    name: str, *, backend: Any = None, registry: Any = None, config: Any = None, **overrides: Any
) -> Any:
    """Build an :class:`~magent.agent.Agent` from a preset.

    Args:
        name: preset name
        backend: LLM backend; resolved from configuration when omitted
        registry: tool registry to scope; the default registry when omitted
        config: loaded configuration
        **overrides: preset fields to override (``max_iterations``, ``temperature``, ...)
    """
    from ..agent import Agent
    from ..llm.factory import build_backend

    preset = get_preset(name).merge(
        **{
            key: value
            for key, value in overrides.items()
            if key in AgentPreset.__dataclass_fields__
        }
    )
    extra = {
        key: value
        for key, value in overrides.items()
        if key not in AgentPreset.__dataclass_fields__
    }
    active_backend = backend if backend is not None else build_backend(config=config)
    agent_config = preset.agent_config()
    generation = None
    if preset.temperature is not None:
        from ..config import GenerationConfig

        generation = GenerationConfig(temperature=preset.temperature)
    return Agent(
        backend=active_backend,
        config=agent_config,
        generation=generation,
        registry=preset.build_registry(registry),
        settings=config,
        name=extra.pop("name", f"MAgent[{preset.name}]"),
        workspace=extra.pop("workspace", None),
        memory=extra.pop("memory", None),
        session=extra.pop("session", None),
        guardrails=extra.pop("guardrails", None),
        events=extra.pop("events", None),
        prompt_defaults={
            "examples_block": "\n\n".join(preset.examples),
            **extra.pop("prompt_defaults", {}),
        },
        **extra,
    )


def describe_presets() -> list[dict[str, Any]]:
    """Serialise every preset, for CLI tables and docs."""
    return [preset.to_dict() for preset in list_presets()]


def preset_table() -> Sequence[tuple[str, str, str, str]]:
    """Rows of ``(name, category, persona, description)`` for a CLI table."""
    return [
        (preset.name, preset.category, preset.persona, preset.description)
        for preset in list_presets()
    ]


def preset_for_task(task: str) -> AgentPreset:
    """Pick a sensible preset from keywords in a task description.

    This is a convenience for ``magent run`` without ``--preset``: the guess is
    recorded in the run metadata so the user can see why it was chosen.
    """
    text = (task or "").lower()
    rules = (
        ("reviewer", ("review", "audit", "critique", "pull request", "pr ")),
        ("devops", ("deploy", "server", "docker", "kubernetes", "nginx", "systemd", "ci pipeline")),
        (
            "analyst",
            ("csv", "dataset", "analys", "analyz", "metrics", "sql", "chart", "statistics"),
        ),
        ("librarian", ("document", "knowledge base", "index these", "cite", "rag")),
        ("researcher", ("research", "investigate", "find out", "compare", "web", "survey")),
        ("writer", ("write", "draft", "blog", "essay", "rewrite", "copy", "readme")),
        ("planner", ("plan", "roadmap", "design doc", "break down", "estimate")),
        ("tutor", ("explain", "teach", "how does", "learn", "understand")),
        ("coder", ("fix", "bug", "implement", "refactor", "function", "class", "test", "code")),
    )
    for name, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return PRESETS[name]
    return PRESETS["general"]


__all__ = [
    "PRESETS",
    "PRESET_CATEGORIES",
    "AgentPreset",
    "build_preset_agent",
    "describe_presets",
    "get_preset",
    "list_presets",
    "preset_for_task",
    "preset_names",
    "preset_table",
    "presets_from_config",
    "register_preset",
]
