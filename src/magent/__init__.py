"""MAgent - Mr. Agent, Powerful.

A zero-dependency Python agent framework built for `Cortex LLMHoster`_.

MAgent plans, calls tools, remembers, retrieves, guards its own inputs and
outputs, and explains what it did. It talks to local models through Cortex's
OpenAI-compatible API, and it runs entirely offline against a built-in echo
backend when no server is reachable, so the whole framework is testable without
a GPU, a network, or an API key.

Start here
----------

One-shot question::

    import magent

    agent = magent.make_agent()                 # Cortex, or echo if unreachable
    result = agent.run("Summarize the README in this directory.")
    print(result.output)

With explicit control::

    from magent import Agent, Config, load_config, default_registry

    settings = load_config()                    # files, env, profiles
    agent = Agent(backend=..., config=settings.agent, registry=default_registry())
    result = agent.run("Refactor utils/text.py and run the tests.")

Retrieval over your own files::

    from magent import Retriever

    rag = Retriever()
    rag.ingest_directory("docs")
    print(rag.context("how do approvals work?"))

Everything below is re-exported for convenience. Subsystems that are heavier or
rarely needed - the HTTP server, the evaluation harness, the plugin loader -
live in their own modules and are *not* imported here, so ``import magent``
stays fast.

.. _Cortex LLMHoster: https://github.com/codero-sus/Cortex_LLMHoster
"""

from __future__ import annotations

from ._version import (
    BUILD_CHANNEL,
    CODENAME,
    CORTEX_REPO,
    HOMEPAGE,
    IMPROVEMENT_COUNT_TARGET,
    MIN_CORTEX_VERSION,
    MIN_PYTHON,
    RELEASE_NAME,
    USER_AGENT,
    __version__,
    __version_info__,
    is_supported_python,
    version_string,
)
from .agent import Agent, AgentConfig, Plan, PlanStep, Reflection, RunOptions, make_agent
from .config import (
    CacheConfig,
    Config,
    EvalConfig,
    GenerationConfig,
    GuardrailsConfig,
    HttpServerConfig,
    MemoryConfig,
    ModelConfig,
    ObservabilityConfig,
    OutputConfig,
    ServerConfig,
    SessionConfig,
    ToolsConfig,
    default_config,
    load_config,
)
from .context import (
    CompressionResult,
    ContextBuild,
    ContextBuilder,
    ContextPolicy,
    build_context,
    compress_messages,
)
from .errors import (
    ERROR_REGISTRY,
    MAgentError,
    classify_status,
    error_from_code,
    is_retryable,
)
from .events import AgentEvent, EventBus, EventKind, EventRecorder, make_bus
from .guardrails import Guardrails, PIIScanner, PolicyEngine, Verdict, build_guardrails
from .llm import (
    CompletionRequest,
    CortexAdmin,
    CortexBackend,
    CortexSettings,
    EchoBackend,
    LLMBackend,
    ModelRouter,
    StreamChunk,
    TokenEstimator,
    UsageTracker,
    build_backend,
)
from .memory import MemoryEntry, MemoryStore, ShortTermMemory, build_embedder
from .output import OutputOptions, Printer, render_result
from .prompts import (
    PERSONAS,
    PromptTemplate,
    SystemPromptBuilder,
    build_system_prompt,
    get_persona,
)
from .rag import Chunk, ChunkIndex, Citation, Retriever, RetrievalResult
from .session import Session, SessionStore, Transcript
from .tools import (
    Tool,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    default_registry,
    tool,
)
from .types import (
    ChatResponse,
    Embedding,
    Message,
    Role,
    RunResult,
    StepRecord,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "BUILD_CHANNEL",
    "CODENAME",
    "CORTEX_REPO",
    "CacheConfig",
    "ChatResponse",
    "Chunk",
    "ChunkIndex",
    "Citation",
    "CompletionRequest",
    "CompressionResult",
    "Config",
    "ContextBuild",
    "ContextBuilder",
    "ContextPolicy",
    "CortexAdmin",
    "CortexBackend",
    "CortexSettings",
    "EchoBackend",
    "Embedding",
    "EvalConfig",
    "EventBus",
    "EventKind",
    "EventRecorder",
    "GenerationConfig",
    "Guardrails",
    "GuardrailsConfig",
    "HOMEPAGE",
    "HttpServerConfig",
    "IMPROVEMENT_COUNT_TARGET",
    "LLMBackend",
    "MAgentError",
    "MIN_CORTEX_VERSION",
    "MIN_PYTHON",
    "MemoryConfig",
    "MemoryEntry",
    "MemoryStore",
    "Message",
    "ModelConfig",
    "ModelRouter",
    "ObservabilityConfig",
    "OutputConfig",
    "OutputOptions",
    "PERSONAS",
    "PIIScanner",
    "Plan",
    "PlanStep",
    "PolicyEngine",
    "Printer",
    "PromptTemplate",
    "RELEASE_NAME",
    "Reflection",
    "RetrievalResult",
    "Retriever",
    "Role",
    "RunOptions",
    "RunResult",
    "ServerConfig",
    "Session",
    "SessionConfig",
    "SessionStore",
    "ShortTermMemory",
    "StepRecord",
    "StopReason",
    "StreamChunk",
    "SystemPromptBuilder",
    "TokenEstimator",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ToolsConfig",
    "Transcript",
    "USER_AGENT",
    "Usage",
    "UsageTracker",
    "Verdict",
    "Agent",
    "AgentConfig",
    "AgentEvent",
    "__version__",
    "__version_info__",
    "build_backend",
    "build_context",
    "build_embedder",
    "build_guardrails",
    "build_system_prompt",
    "compress_messages",
    "default_config",
    "default_registry",
    "ERROR_REGISTRY",
    "classify_status",
    "error_from_code",
    "is_retryable",
    "get_persona",
    "is_supported_python",
    "load_config",
    "make_agent",
    "make_bus",
    "render_result",
    "tool",
    "version_string",
]
