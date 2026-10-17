# HTTP API and web UI

`magent serve` starts a stdlib `ThreadingHTTPServer` exposing the same objects
the CLI uses, plus a single-page UI at `/`.

```bash
magent serve --host 127.0.0.1 --port 8765
curl -s localhost:8765/api/health | python -m json.tool
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api` | the route table |
| GET | `/api/health` | liveness, version, uptime (public) |
| GET | `/api/info` | configuration summary and counts |
| GET | `/api/models` | models the backend reports, plus health |
| GET | `/api/tools` | tools with their schemas (`?category=`, `?q=`) |
| POST | `/api/tools/{name}` | dispatch one tool |
| POST | `/api/chat` | run the agent on a prompt |
| GET | `/api/chat/stream` | server-sent events for one run |
| GET | `/api/presets` | agent presets |
| GET | `/api/skills` | skills, or matches for `?task=` |
| GET | `/api/sessions` | stored sessions (`?limit=`, `?q=`, `?tag=`) |
| GET | `/api/sessions/{id}` | one session plus a transcript |
| DELETE | `/api/sessions/{id}` | delete a session |
| GET/POST | `/api/memory` | search or store a memory |
| GET | `/api/metrics` | metrics snapshot, or Prometheus text with `?format=prometheus` |
| GET | `/api/rag/search` | retrieve cited chunks (`?q=`, `?limit=`) |
| POST | `/api/rag/ingest` | index paths or inline text |
| GET | `/` | the web UI |

## Examples

```bash
curl -s -X POST localhost:8765/api/chat -H 'Content-Type: application/json' \
  -d '{"prompt": "summarise the docs directory", "session_id": "demo"}' | python -m json.tool

curl -N 'localhost:8765/api/chat/stream?prompt=write%20a%20haiku'

curl -s -X POST localhost:8765/api/tools/calculate -H 'Content-Type: application/json' \
  -d '{"expression": "6*7"}'

curl -s 'localhost:8765/api/metrics?format=prometheus'
```

## Authentication and origins

Set `[http].api_key` (or `MAGENT_HTTP_API_KEY`) to require
`Authorization: Bearer <token>` or `X-API-Key` on everything except `/api/health`
and the UI. Origins come from `[http].cors_origins`, or `allow_origins_any =
true` for local use. Preflights are answered, and no framing headers are sent, so
the UI works behind a proxy.

## Client code

```python
import json, urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:8765/api/chat",
    data=json.dumps({"prompt": "hello"}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(json.load(response)["output"])
```

The web UI uses relative URLs, so it works unchanged whether you open
`http://localhost:8765/` or reach the same server through a proxy.
