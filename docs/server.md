# Embedded HTTP server and web console

MAgent includes a stdlib `ThreadingHTTPServer`, router, bounded executor, JSON API, SSE endpoint, static-file handler, and one self-contained web console. No WSGI/ASGI framework, Node build, or CDN is required.

## Start

```bash
magent serve
```

Defaults:

- `127.0.0.1:8700`;
- console enabled at `/`;
- 16 request workers;
- 32 MiB request body limit;
- 300-second request timeout;
- wildcard CORS from the default `cors_origins=["*"]`;
- no server key on loopback.

Preview the exact route table without binding:

```bash
magent serve --routes
magent serve --routes -f json
```

Use an ephemeral port:

```bash
magent serve --port 0 --offline
```

## Authentication and bind safety

Generate a key:

```bash
magent serve --generate-key
```

A non-loopback bind requires a key:

```bash
export MAGENT_SERVER_KEY='mag_...'
magent serve --host 0.0.0.0 --server-key "$MAGENT_SERVER_KEY"
```

Without one, construction fails before a socket is exposed.

Clients can use:

```http
Authorization: Bearer mag_...
```

or:

```http
X-API-Key: mag_...
```

Keys are compared with `hmac.compare_digest`. `/health` and `/ready` remain open for probes.

The global CLI `--api-key` configures the upstream Cortex key. The server key is deliberately named `--server-key` so the two credentials cannot be confused.

## Routes

### Probes

- `GET /health`
- `GET /ready`

### Information

- `GET /api/info`
- `GET /api/routes`
- `GET /api/config`
- `GET /api/improvements`
- `GET /api/stats`

### Tools

- `GET /api/tools`
- `GET /api/tools/{name}`
- `GET /api/tools/{name}/schema`
- `POST /api/tools/{name}/call`

### Runs/events

- `POST /api/run`
- `POST /api/run/async`
- `GET /api/runs`
- `GET /api/runs/{id}`
- `POST /api/runs/{id}/cancel`
- `GET /api/events`

### Memory/RAG/guardrails

- `GET|POST /api/memory`
- `GET /api/memory/search`
- `GET /api/memory/stats`
- `POST /api/rag/search`
- `POST /api/rag/ingest`
- `POST /api/guardrails/check`

### Sessions/metrics/Cortex

- `GET /api/sessions`
- `GET|DELETE /api/sessions/{id}`
- `GET /api/metrics`
- `GET /api/cortex/status`
- `GET /api/cortex/models`

There are 31 registered method/path pairs. `GET /api/routes` is the source of truth at runtime.

## Run example

```bash
curl -sS http://127.0.0.1:8700/api/run \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Calculate 6 * 7",
    "loop_mode": "react",
    "max_iterations": 4
  }'
```

Accepted run options are validated. Aliases such as `max_steps` normalize to canonical fields; unknown names return HTTP 400 with `details.accepted`.

Async:

```bash
run_id=$(curl -sS -X POST http://127.0.0.1:8700/api/run/async \
  -H 'Content-Type: application/json' -d '{"prompt":"hello"}' |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')
curl -sS "http://127.0.0.1:8700/api/runs/$run_id"
```

## Direct tool call

```bash
curl -sS -X POST http://127.0.0.1:8700/api/tools/calculate/call \
  -H 'Content-Type: application/json' \
  -d '{"arguments":{"expression":"(6*7)+1"}}'
```

Approval-required tools return HTTP 428 unless explicit approval is present. Handler/validation failures remain a `ToolResult` JSON body.

## SSE

```bash
curl -N http://127.0.0.1:8700/api/events?seconds=30
```

The endpoint uses HTTP/1.1 `Transfer-Encoding: chunked`, `text/event-stream`, no-cache, keep-alive comments, dotted `EventKind` names, and connected/closed terminal events. Streams are capped at 300 seconds and clients are bounded/disconnected if their queue fills.

## CORS

`cors_origins=["*"]` enables wildcard mode and returns `Access-Control-Allow-Origin: *` on normal, error, and `OPTIONS` responses. An explicit list uses exact origin matching. Preflight returns 204 and advertises allowed headers/methods.

## Static console

The 40+ KiB embedded `index.html` has tabs for Run, Tools, Memory, RAG, Sessions, Events, Metrics, and Settings. It uses relative URLs, so it works behind the same-origin Arena/reverse proxy. Its event listeners use actual dotted event values such as `run.started`, `tool.call`, and `agent.text_delta`.

Static serving resolves paths under one directory and rejects traversal. Unknown API paths always return JSON rather than falling into static HTML.

## Python

```python
from magent import Agent, EchoBackend, load_config
from magent.server import MAgentServer, build_context, free_port

config = load_config(use_env=False)
backend = EchoBackend()
agent = Agent(backend=backend, config=config.agent, settings=config)
context = build_context(config, agent=agent, backend=backend)
server = MAgentServer(context, host="127.0.0.1", port=free_port())
server.start()
print(server.url)
server.stop()
```

Use `server.start(block=True)` for a foreground process. The CLI uses non-blocking start plus a poll so `KeyboardInterrupt` maps to exit 130 reliably.
