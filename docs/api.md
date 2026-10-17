# Python API

This page maps the primary public objects. The package is typed (`py.typed`) and supports CPython 3.11+.

## Top-level imports

```python
from magent import (
    Agent,
    Config,
    CortexBackend,
    CortexSettings,
    EchoBackend,
    MemoryStore,
    Message,
    RunResult,
    ToolCall,
    ToolResult,
    default_registry,
    load_config,
    tool,
)
```

Specialized surfaces are imported from their packages:

```python
from magent.llm import CortexAdmin, ModelRouter, Route
from magent.rag import ChunkIndex, Retriever
from magent.plugins import HookRegistry, PluginLoader
from magent.server import MAgentServer, build_context
```

## Configuration

```python
config = load_config(
    path=None,
    profile="",
    overrides={"agent.max_iterations": 8},
    use_env=True,
    required=False,
)
```

`Config` exposes typed sections, `get/set` dotted access, `section`, `describe`, redacted `to_dict`, TOML rendering, validation, and deep `clone`.

## Backend

`LLMBackend` protocol:

```python
response = backend.complete(request)
chunks = backend.stream(request)
models = backend.models()
health = backend.health()
backend.close()
```

`CompletionRequest` holds messages, model/generation controls, tool schemas/choice, response format, timeout, and metadata. `ChatResponse` holds text, model, usage, tool calls, finish reason, raw payload, latency, and cache state.

Cortex:

```python
settings = CortexSettings.from_env()
backend = CortexBackend(settings)
admin = CortexAdmin(backend=backend)
```

Offline/testing:

```python
backend = EchoBackend(prefix="echo:")
```

Routing:

```python
router = ModelRouter([
    Route(primary_backend, priority=1, label="primary"),
    Route(fallback_backend, priority=2, label="fallback"),
], strategy="priority")
```

Strategies: priority, round-robin, least-used, fastest, cheapest, random. Each route owns a circuit breaker.

## Agent

```python
agent = Agent(
    backend=backend,
    config=config.agent,
    settings=config,
    registry=default_registry(),
    workspace=".",
)
result: RunResult = agent.run("task", max_iterations=8)
```

Conveniences: `ask`, `chat`, `stream`, `complete`, `plan`, `reflect`, `critique`, `summarize`, `describe`, `stats`, `cancel`, `reset`, and `close`.

## Messages and results

Factories:

```python
Message.system("rules")
Message.user("request")
Message.assistant("answer", tool_calls=[])
Message.tool(tool_result)
```

Every value type has JSON-friendly `to_dict`; persisted types provide `from_dict`. `Usage` derives a missing total from prompt + completion consistently for direct and parsed construction.

## Tools

```python
@tool(category="math", risk="safe")
def square(value: float) -> float:
    """Square a number.

    Args:
        value: number to square
    """
    return value * value

registry = default_registry()
registry.register(square)
result = registry.dispatch(
    ToolCall(id="c1", name="square", arguments={"value": 3}),
)
```

`ToolContext` provides workspace resolution, sandbox, approvals, events, config, memory, backend, run/step/call identity, time/output limits, secrets for redaction, dry-run, logger, and extension data.

## Memory

```python
store = MemoryStore(path="memory.db")
record = store.add("fact", kind="fact", tags=["tag"], importance=0.8)
results = store.search("fact", limit=5, semantic=True)
store.export_jsonl("memory.jsonl")
```

`HashingEmbedder`, `BackendEmbedder`, and `CachingEmbedder` satisfy the same embedding interface.

## RAG

```python
retriever = Retriever(mode="hybrid", limit=5)
report = retriever.ingest(["README.md", "docs/"])
result = retriever.retrieve("query")
context = result.context
check = result.verify("answer with [1]")
```

Use `ChunkIndex` directly for custom ingestion and scoring.

## Events

```python
subscription = agent.events.subscribe(
    handler,
    kinds=["tool", "run.*"],
    priority=100,
    async_dispatch=False,
)
agent.events.unsubscribe(subscription)
```

`EventRecorder` keeps a bounded transcript. `EventBroadcaster` fans out into bounded queues for SSE clients.

## Guardrails

```python
from magent.guardrails import build_guardrails

verdict = build_guardrails().inspect("text", stage="prompt")
```

The pipeline, findings, and verdicts are typed. Budget and approval systems are separate objects so applications can replace policy without replacing the agent.

## Plugins

```python
registry = default_registry()
hooks = HookRegistry()
loader = PluginLoader(["plugins"], registry=registry, hooks=hooks)
report = loader.load_all()
```

Inspect a plugin statically with `inspect_plugin(path)` before loading.

## Server

```python
context = build_context(config, agent=agent, registry=registry, backend=backend)
server = MAgentServer(context, host="127.0.0.1", port=8700)
server.start()
server.stop()
```

`Router`, `Request`, `Response`, and `Route` are available for embedding/custom routes, while the standard 31 routes are registered by `build_context`/server construction.

## Testing

```python
from magent.testing import make_backend, start_fake_cortex, temp_workspace

backend = make_backend(
    tool_name="calculate",
    tool_arguments={"expression": "6*7"},
    final_text="42",
)
```

The testing API is public and deliberately dependency-free.
