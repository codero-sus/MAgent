# Cortex LLMHoster integration

Cortex LLMHoster is MAgent's primary model hosting layer. The two projects remain separate: MAgent is a pure HTTP client and does not vendor, patch, subclass, or redistribute Cortex.

## Connection settings

The default root is `http://127.0.0.1:8624`.

```python
from magent import CortexBackend, CortexSettings

settings = CortexSettings.from_env()
backend = CortexBackend(settings)
assert backend.ping()
```

Environment precedence:

| Purpose | Preferred | Fallback |
|---|---|---|
| root URL | `MAGENT_CORTEX_URL` | `CORTEX_URL` |
| API key | `MAGENT_CORTEX_API_KEY` | `CORTEX_API_KEY` |
| model | `MAGENT_CORTEX_MODEL` | `CORTEX_DEFAULT_MODEL` |
| timeout | `MAGENT_CORTEX_TIMEOUT` | `CORTEX_TIMEOUT` |

Additional variables control API prefix, retries, embedding model, TLS verification, and whether authentication is required.

Settings are redacted by default:

```python
print(settings.to_dict())               # masked API key
print(settings.to_dict(redact=False))   # explicit reveal
```

## Exact route layout

Root probes:

- `GET /health` returns liveness;
- `GET /ready` confirms a default/running model.

OpenAI-compatible routes under `/v1`:

- `GET /v1/models`;
- `POST /v1/chat/completions`;
- `POST /v1/completions`;
- `POST /v1/embeddings`;
- `POST /v1/responses`;
- audio transcription/speech, image generation/editing, video generation, reranking, and moderation when the selected Cortex runtime supports them.

Administrative routes under `/admin/api` cover state, metrics, runtime, hardware, authentication options, configuration, model files, capability detection, model start/stop/restart/test, cookbook entries, and updates.

## Authentication

MAgent sends:

```http
Authorization: Bearer <key>
```

A Cortex 401 becomes `AuthenticationError` with guidance to set `MAGENT_CORTEX_API_KEY`. Keys in URLs and error details are redacted. MAgent does not log request/response bodies by default.

Cortex can be started with authentication disabled for a trusted loopback-only development environment; consult Cortex's documentation for its `--no-auth` and environment controls.

## Chat

```python
from magent import CompletionRequest, Message

request = CompletionRequest(
    model="",
    messages=[
        Message.system("Answer briefly."),
        Message.user("What is MAgent?"),
    ],
    temperature=0.2,
    max_tokens=300,
)
response = backend.complete(request)
print(response.text)
print(response.model, response.usage.total_tokens)
```

Convenience methods:

```python
text = backend.ask("Reply with one word")
response = backend.chat([Message.user("Hello")])
legacy = backend.completion("Complete this sentence")
```

## Streaming

`CortexBackend.stream()` parses SSE incrementally. It handles text deltas, terminal usage, `[DONE]`, and fragmented function-call arguments:

```python
from magent.llm import collect_stream

chunks = backend.stream(CompletionRequest(
    messages=[Message.user("Explain streaming")],
    stream=True,
))
response = collect_stream(chunks, on_chunk=lambda chunk: print(chunk.text, end=""))
```

Tool-call deltas are merged by index and id before JSON arguments are parsed. HTTP connections are closed correctly when a stream ends or a caller stops early.

## Models and capabilities

```python
for model in backend.models():
    print(model.id, model.runtime, model.capabilities, model.default)

from magent.types import Capability

model = backend.get_model("model-id")
backend.assert_capability("model-id", Capability.VISION)
```

Capabilities are used by tools and the router to avoid sending incompatible work to a model. A missing capability raises `CapabilityNotSupportedError` before an expensive call.

## Embeddings

```python
vectors = backend.embed(
    ["first document", "second document"],
    model="embedding-model-id",
    batch_size=16,
)
vector = backend.embed_one("query", model="embedding-model-id")
```

Set `MAGENT_EMBEDDING_MODEL` when the default chat model does not support embeddings. `dimensions=` is passed through for runtimes that implement OpenAI's dimension control.

## Admin client

```python
from magent.llm import CortexAdmin

admin = CortexAdmin(backend=backend)
state = admin.state()
print(state.version)
print(state.running_models())
print(admin.hardware())
print(admin.metrics().to_dict())

admin.start_model("model-id", wait=True, timeout=120)
admin.test_model("model-id", prompt="ping")
admin.restart_model("model-id")
admin.stop_model("model-id")
```

The CLI exposes the same operations:

```bash
magent cortex status
magent cortex state -f json
magent cortex runtime
magent cortex hardware
magent cortex start MODEL_ID
magent cortex test MODEL_ID --prompt ping
magent cortex stop MODEL_ID
```

## Backend selection

The backend factory applies one deterministic chain:

1. explicit backend object/name;
2. `MAGENT_BACKEND`;
3. probe Cortex `GET /health` at the configured URL;
4. `EchoBackend` when `fallback_to_echo=true`.

`--offline` bypasses the probe and guarantees that no model-server socket is opened.

## Errors and retries

The stdlib HTTP client classifies 401/403, 404, 413, 429, other 4xx, and 5xx responses into typed errors. Retries use bounded exponential backoff with jitter and honor `Retry-After`. A `ModelRouter` can fail over to another Cortex route, with a circuit breaker per route.

## Contract testing

`magent.testing.FakeCortexServer` is a real threaded loopback HTTP server implementing the route, auth, JSON, SSE, tool-call, embedding, metrics, and supported admin shapes. Tests therefore verify serialization and parsing end to end without copying Cortex code or downloading a model.
