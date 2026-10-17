"""MAgent - Mr. Agent, Powerful.

A local-first agent framework for **Cortex LLMHoster**: ``pip install`` nothing
mandatory, talk to the models running on your own machine, and get tools,
memory, retrieval, guardrails, teams, sessions, evaluation, tracing, plugins and
a CLI/HTTP API out of the box.

Typical use::

    from magent import Agent, build_backend

    agent = Agent(backend=build_backend())
    print(agent.run("summarise the docs directory").output)

The package keeps import time low by loading subpackages lazily: the names below
are resolved on first access, so ``magent.Agent`` imports the agent module and
nothing else.  ``import magent`` itself only pulls in the error hierarchy, the
shared types, the paths module and logging.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__
from .errors import MAgentError
from .logging_ import get_logger
from .paths import cache_dir, config_dir, data_dir, repo_root
from .types import (
    AgentEvent,
    ChatResponse,
    CostEstimate,
    EventKind,
    Message,
    ModelInfo,
    Role,
    RunResult,
    StepRecord,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "ChatResponse",
    "Config",
    "ContextBuilder",
    "CostEstimate",
    "EvalRunner",
    "EventBus",
    "EventKind",
    "Guardrails",
    "MAgentError",
    "MemoryEntry",
    "MemoryStore",
    "Message",
    "MetricsRegistry",
    "ModelInfo",
    "PluginLoader",
    "PromptTemplate",
    "Repl",
    "ResultRenderer",
    "Retriever",
    "Role",
    "RunResult",
    "SessionManager",
    "SkillLibrary",
    "StepRecord",
    "StopReason",
    "Team",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "Tracer",
    "Usage",
    "__version__",
    "build_backend",
    "build_guardrails",
    "build_store",
    "cache_dir",
    "config_dir",
    "data_dir",
    "default_registry",
    "get_logger",
    "load_config",
    "make_agent",
    "repo_root",
    "run_team",
]

_LAZY: dict[str, tuple[str, str]] = {
    # agent
    "Agent": ("magent.agent", "Agent"),
    "AgentConfig": ("magent.config", "AgentConfig"),
    "make_agent": ("magent.agent", "make_agent"),
    # configuration
    "Config": ("magent.config", "Config"),
    "load_config": ("magent.config", "load_config"),
    # events
    "EventBus": ("magent.events", "EventBus"),
    # backend
    "build_backend": ("magent.llm.factory", "build_backend"),
    # tools
    "default_registry": ("magent.tools", "default_registry"),
    "ToolRegistry": ("magent.tools", "ToolRegistry"),
    "build_sandbox": ("magent.tools", "build_sandbox"),
    # memory
    "MemoryStore": ("magent.memory", "MemoryStore"),
    "MemoryEntry": ("magent.memory", "MemoryEntry"),
    "build_store": ("magent.memory", "build_store"),
    # retrieval
    "Retriever": ("magent.rag", "Retriever"),
    # guardrails
    "Guardrails": ("magent.guardrails", "Guardrails"),
    "build_guardrails": ("magent.guardrails", "build_guardrails"),
    # prompts
    "PromptTemplate": ("magent.prompts", "PromptTemplate"),
    # context
    "ContextBuilder": ("magent.context", "ContextBuilder"),
    # output
    "ResultRenderer": ("magent.output", "ResultRenderer"),
    # sessions
    "SessionManager": ("magent.session", "SessionManager"),
    # teams and presets
    "Team": ("magent.agents", "Team"),
    "run_team": ("magent.agents", "run_team"),
    # skills
    "SkillLibrary": ("magent.skills", "SkillLibrary"),
    # plugins
    "PluginLoader": ("magent.plugins", "PluginLoader"),
    # evaluation
    "EvalRunner": ("magent.eval", "EvalRunner"),
    # observability
    "Tracer": ("magent.observability", "Tracer"),
    "MetricsRegistry": ("magent.observability", "MetricsRegistry"),
    # CLI
    "Repl": ("magent.cli.repl", "Repl"),
}

_SUBPACKAGES: tuple[str, ...] = (
    "agent",
    "agents",
    "cli",
    "concurrency",
    "context",
    "errors",
    "eval",
    "events",
    "guardrails",
    "http",
    "llm",
    "logging_",
    "memory",
    "observability",
    "output",
    "paths",
    "plugins",
    "prompts",
    "rag",
    "server",
    "session",
    "skills",
    "testing",
    "tools",
    "types",
    "utils",
)


def __getattr__(name: str) -> Any:
    """Resolve lazily exported names (PEP 562).

    Raises:
        AttributeError: when the name is neither an exported symbol nor a subpackage
    """
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    if name in _SUBPACKAGES:
        module = import_module(f"magent.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'magent' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the public names, including the lazy ones."""
    return sorted({*globals(), *__all__, *_LAZY, *_SUBPACKAGES})


def version_tuple() -> tuple[int, ...]:
    """Version as a tuple of integers, for feature checks."""
    parts: list[int] = []
    for chunk in __version__.split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def capabilities() -> dict[str, Any]:
    """Describe what this installation can do, without importing much.

    Used by ``magent doctor``, the ``GET /api/info`` endpoint and packaging
    metadata.  Every value here is derived from the installed source tree rather
    than hard-coded, so it cannot drift.
    """
    from .paths import bundled_docs_dir, improvements_manifest
    from .paths import describe as describe_paths

    manifest = improvements_manifest()
    return {
        "version": __version__,
        "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "paths": describe_paths(),
        "docs": str(bundled_docs_dir()),
        "improvements": str(manifest),
        "improvements_present": manifest.is_file(),
        "packages": list(_SUBPACKAGES),
        "lazy_exports": sorted(_LAZY),
    }
