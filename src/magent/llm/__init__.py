"""LLM backends, routing, caching, token accounting, and retry policy.

The default backend is :class:`~magent.llm.cortex.CortexBackend`, which speaks to
`Cortex LLMHoster <https://github.com/codero-sus/Cortex_LLMHoster>`_. Every other
module in MAgent depends only on the abstract types re-exported here.
"""

from __future__ import annotations

from .base import (
    BaseBackend,
    CapabilityRequest,
    CompletionRequest,
    GenerationParams,
    LLMBackend,
    StreamChunk,
    StreamHandler,
    collect_stream,
    estimate_message_chars,
)
from .cache import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TTL_SECONDS,
    CacheStats,
    DiskCache,
    EmbeddingCache,
    MemoryLRU,
    ResponseCache,
)
from .cortex import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_API_PREFIX,
    DEFAULT_CORTEX_URL,
    EMBEDDINGS_PATH,
    HEALTH_PATH,
    MODELS_PATH,
    READY_PATH,
    CortexBackend,
    CortexSettings,
    build_cortex_backend,
    cortex_health,
)
from .cortex_admin import CortexAdmin, CortexMetrics, CortexState
from .echo import EchoBackend, LambdaBackend, ScriptedBackend, ToolScriptBackend
from .factory import (
    BACKEND_CHOICES,
    BackendDecision,
    BackendFactory,
    BackendRequest,
    build_backend,
    build_offline_backend,
)
from .retry import (
    BreakerState,
    CircuitBreaker,
    RetryAttempt,
    RetryExecutor,
    RetryOutcome,
    retry,
    with_timeout,
)
from .router import STRATEGIES, ModelRouter, Route, RouteAttempt, single_backend_router
from .tokens import (
    TokenBudget,
    TokenEstimator,
    context_window_plan,
    estimate_tokens,
    split_by_tokens,
    trim_messages,
    truncate_to_tokens,
)
from .usage import BudgetConfig, BudgetViolation, PriceBook, UsageTracker

__all__ = [
    "BACKEND_CHOICES",
    "BreakerState",
    "BaseBackend",
    "BudgetConfig",
    "BudgetViolation",
    "CHAT_COMPLETIONS_PATH",
    "CacheStats",
    "CapabilityRequest",
    "CircuitBreaker",
    "CompletionRequest",
    "CortexAdmin",
    "CortexBackend",
    "CortexMetrics",
    "CortexSettings",
    "CortexState",
    "DEFAULT_API_PREFIX",
    "DEFAULT_CORTEX_URL",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TTL_SECONDS",
    "DiskCache",
    "EMBEDDINGS_PATH",
    "EchoBackend",
    "EmbeddingCache",
    "GenerationParams",
    "HEALTH_PATH",
    "LLMBackend",
    "LambdaBackend",
    "MODELS_PATH",
    "MemoryLRU",
    "ModelRouter",
    "PriceBook",
    "READY_PATH",
    "ResponseCache",
    "RetryAttempt",
    "RetryExecutor",
    "RetryOutcome",
    "Route",
    "RouteAttempt",
    "STRATEGIES",
    "ScriptedBackend",
    "StreamChunk",
    "StreamHandler",
    "TokenBudget",
    "TokenEstimator",
    "ToolScriptBackend",
    "UsageTracker",
    "BackendDecision",
    "BackendFactory",
    "BackendRequest",
    "build_backend",
    "build_cortex_backend",
    "build_offline_backend",
    "collect_stream",
    "context_window_plan",
    "cortex_health",
    "estimate_message_chars",
    "estimate_tokens",
    "retry",
    "single_backend_router",
    "split_by_tokens",
    "trim_messages",
    "truncate_to_tokens",
    "with_timeout",
]
