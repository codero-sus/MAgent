"""A complete in-process fake of the Cortex LLMHoster HTTP API.

This is not a mock object; it is a real HTTP server speaking the same routes,
auth scheme, SSE framing, error envelope, and admin payload shapes as Cortex
LLMHoster. MAgent's test suite, ``magent demo``, and ``magent doctor --offline``
all run against it, which means the Cortex integration is verified end to end
without installing Cortex, downloading weights, or owning a GPU.

Implemented routes
------------------
``GET  /health``                  liveness
``GET  /ready``                   readiness (503 until the default model runs)
``GET  /v1/models``               model list with capabilities
``POST /v1/chat/completions``     chat, streaming SSE, tool calls
``POST /v1/completions``          legacy completion
``POST /v1/embeddings``           deterministic hash embeddings
``POST /v1/responses``            Responses API dialect
``POST /v1/audio/transcriptions`` multipart transcription
``GET  /admin/api/state``         dashboard state
``GET  /admin/api/metrics``       inference counters
``GET  /admin/api/runtime``       runtime status
``POST /admin/api/models/start``  start a model runtime
``POST /admin/api/models/stop``   stop a model runtime
``POST /admin/api/models/restart``restart a model runtime
``POST /admin/api/models/test``   model smoke test
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

EMBEDDING_DIM = 32
"""Dimensionality of the fake embedding vectors."""

DEFAULT_MODEL = "tiny-llama-q4"
"""Model id the fake hoster serves by default."""

EMBEDDING_MODEL = "nomic-embed-q4"
"""Model id used for fake embeddings."""


@dataclass
class FakeModel:
    """One model record held by the fake hoster."""

    id: str
    runtime: str = "llama.cpp"
    capabilities: list[str] = field(default_factory=lambda: ["chat", "tool_calling"])
    context_size: int = 8192
    running: bool = True
    model_path: str = "/models/fake.gguf"
    gpu_layers: int = 0
    optimization_preset: str = "balanced"

    def to_public(self, default: bool = False) -> dict[str, Any]:
        """Return the shape used by ``GET /v1/models``."""
        return {
            "id": self.id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "cortex-llmhoster",
            "runtime": self.runtime,
            "capabilities": list(self.capabilities),
            "default": default,
        }

    def to_state(self, default: bool = False) -> dict[str, Any]:
        """Return the richer shape used by ``GET /admin/api/state``."""
        return {
            "id": self.id,
            "upstream_model": self.id,
            "default": default,
            "runtime": self.runtime,
            "capabilities": list(self.capabilities),
            "runtime_command": [],
            "api_base_path": "/v1",
            "health_path": "/health",
            "optimization_level": 3,
            "optimization_preset": self.optimization_preset,
            "model_path": self.model_path,
            "threads": 0,
            "threads_batch": 0,
            "gpu_layers": self.gpu_layers,
            "context_size": self.context_size,
            "batch_size": 512,
            "ubatch_size": 512,
            "parallel": 1,
            "flash_attn": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "use_mmap": True,
            "mlock": False,
            "embedding": "embedding" in self.capabilities,
            "mmproj_path": "",
        }

    def to_runtime_state(self) -> dict[str, Any]:
        """Return the shape used by the runtime manager status map."""
        return {
            "id": self.id,
            "status": "running" if self.running else "stopped",
            "pid": 4242 if self.running else None,
            "base_url": f"http://127.0.0.1:9001/{self.id}" if self.running else None,
            "logs": [],
        }


def fake_embedding(text: str, model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a deterministic pseudo-embedding derived from token hashes.

    Similar strings get genuinely similar vectors (shared token hashes
    contribute shared dimensions), so cosine-similarity tests over the fake
    backend are meaningful rather than random.
    """
    vector = [0.0] * dim
    tokens = [token for token in text.lower().replace(".", " ").split() if token]
    if not tokens:
        tokens = [text.lower() or "empty"]
    for token in tokens:
        digest = hashlib.blake2b(f"{model}:{token}".encode("utf-8"), digest_size=dim).digest()
        for index, byte in enumerate(digest):
            vector[index] += (byte / 255.0) - 0.5
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


class FakeCortexState:
    """Mutable state for the fake hoster, shared across request threads."""

    def __init__(
        self,
        *,
        api_key: str = "",
        auth_enabled: bool = False,
        models: list[FakeModel] | None = None,
        default_model: str = DEFAULT_MODEL,
        chunk_delay: float = 0.0,
        fail_models: dict[str, int] | None = None,
    ) -> None:
        self.api_key = api_key
        self.auth_enabled = auth_enabled or bool(api_key)
        self.models: dict[str, FakeModel] = {
            model.id: model for model in (models or _default_models())
        }
        self.default_model = default_model
        self.chunk_delay = chunk_delay
        self.fail_models: dict[str, int] = dict(fail_models or {})
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.requests = Counter()
        self.status_codes = Counter()
        self.by_model = Counter()
        self.total_latency_ms = 0.0
        self.completed = 0
        self.failed = 0
        self.in_flight = 0
        self.transcripts: list[dict[str, Any]] = []
        self.admin_calls: list[str] = []
        self.version = "0.2.0-fake"

    def metrics(self) -> dict[str, Any]:
        """Return the inference metrics snapshot in Cortex's shape."""
        average = self.total_latency_ms / self.completed if self.completed else 0.0
        return {
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "in_flight": self.in_flight,
            "total_requests": sum(self.requests.values()),
            "completed_requests": self.completed,
            "failed_requests": self.failed,
            "average_latency_ms": round(average, 2),
            "status_codes": dict(self.status_codes),
            "requests_by_model": dict(self.by_model),
        }

    def public_settings(self) -> dict[str, Any]:
        """Return the ``server`` + ``models`` section of admin state."""
        return {
            "server": {
                "host": "127.0.0.1",
                "port": 8624,
                "workers": 1,
                "api_auth_enabled": self.auth_enabled,
                "api_key_env": "CORTEX_API_KEY",
                "max_inference_requests": 8,
            },
            "models": [
                model.to_state(default=model.id == self.default_model)
                for model in self.models.values()
            ],
        }

    def state(self) -> dict[str, Any]:
        """Return the full ``/admin/api/state`` payload."""
        return {
            **self.public_settings(),
            "api": {
                "format": "openai",
                "version": 1,
                "base_path": "/v1",
                "models_path": "/models",
                "models_endpoint": "/v1/models",
            },
            "metrics": self.metrics(),
            "runtime": {
                "hardware": {
                    "cpu_count": 8,
                    "total_memory_bytes": 16 * 1024**3,
                    "gpus": [],
                    "platform": "fake",
                },
                "models": {
                    model_id: model.to_runtime_state() for model_id, model in self.models.items()
                },
            },
            "version": self.version,
            "env_overrides": [],
            "admin_config_path": "cortex.toml",
        }


def _default_models() -> list[FakeModel]:
    """Return the model set the fake hoster ships with."""
    return [
        FakeModel(id=DEFAULT_MODEL, capabilities=["chat", "tool_calling"], context_size=8192),
        FakeModel(id="vision-model-q4", capabilities=["chat", "vision", "ocr"], context_size=4096),
        FakeModel(
            id=EMBEDDING_MODEL,
            capabilities=["embedding"],
            context_size=2048,
            model_path="/models/embed.gguf",
        ),
        FakeModel(id="stopped-model-q4", capabilities=["chat"], running=False),
    ]


def _canned_reply(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Produce a deterministic reply, including a tool call when appropriate."""
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                last_user = " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            else:
                last_user = str(content or "")
            break
    lowered = last_user.lower()
    if tools:
        names = {
            str((tool.get("function") or {}).get("name", ""))
            for tool in tools
            if isinstance(tool, dict)
        }
        for keyword, tool_name in (
            ("calculate", "calculator"),
            ("math", "calculator"),
            ("read file", "read_file"),
            ("list files", "list_dir"),
            ("search", "grep"),
            ("time", "current_time"),
        ):
            if keyword in lowered and tool_name in names:
                return {
                    "tool_calls": [
                        {
                            "id": f"call_{hashlib.blake2b(lowered.encode(), digest_size=4).hexdigest()}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(_fake_arguments(tool_name, last_user)),
                            },
                        }
                    ],
                    "finish_reason": "tool_calls",
                    "text": "",
                }
        if "use a tool" in lowered and names:
            chosen = sorted(names)[0]
            return {
                "tool_calls": [
                    {
                        "id": "call_fake_0",
                        "type": "function",
                        "function": {"name": chosen, "arguments": json.dumps(_fake_arguments(chosen, last_user))},
                    }
                ],
                "finish_reason": "tool_calls",
                "text": "",
            }
    tool_results = [
        message for message in messages if message.get("role") == "tool"
    ]
    if tool_results:
        payload = tool_results[-1].get("content", "")
        return {
            "text": f"Based on the tool result: {str(payload)[:200]}",
            "finish_reason": "stop",
        }
    if lowered.startswith("echo:"):
        return {"text": last_user[5:], "finish_reason": "stop"}
    return {
        "text": f"Fake Cortex reply to: {last_user[:200]}",
        "finish_reason": "stop",
    }


def _fake_arguments(tool_name: str, prompt: str) -> dict[str, Any]:
    """Return plausible arguments for a fake tool call."""
    if tool_name == "calculator":
        digits = "".join(char if char in "0123456789+-*/(). " else "" for char in prompt)
        return {"expression": digits.strip() or "2+2"}
    if tool_name == "read_file":
        return {"path": "README.md"}
    if tool_name == "list_dir":
        return {"path": "."}
    if tool_name == "grep":
        return {"pattern": prompt.split()[-1] if prompt.split() else "TODO", "path": "."}
    if tool_name == "current_time":
        return {"timezone": "UTC"}
    return {"input": prompt[:100]}


class _FakeCortexHandler(BaseHTTPRequestHandler):
    """Request handler implementing the fake Cortex routes."""

    server_version = "FakeCortex/0.2"
    protocol_version = "HTTP/1.1"

    # -- plumbing ------------------------------------------------------------

    @property
    def cortex(self) -> FakeCortexState:
        """Return the shared fake state."""
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log."""
        if getattr(self.server, "verbose", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        """Send a JSON response with correct length headers."""
        body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400, *, code: str = "invalid_request") -> None:
        """Send a Cortex-shaped error envelope."""
        self._send_json({"error": {"message": message, "type": "invalid_request_error", "code": code}}, status)

    def _read_body(self) -> bytes:
        """Read the request body honoring Content-Length."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(min(length, 64 * 1024 * 1024))

    def _read_json(self) -> dict[str, Any]:
        """Read and parse a JSON request body."""
        raw = self._read_body()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _authorized(self) -> bool:
        """Check the bearer key when auth is enabled."""
        if not self.cortex.auth_enabled:
            return True
        header = self.headers.get("Authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
        return bool(token) and token == self.cortex.api_key

    def _track(self, name: str, status: int, started: float, model: str = "") -> None:
        """Update request counters exactly as Cortex's metrics do."""
        with self.cortex.lock:
            self.cortex.requests[name] += 1
            self.cortex.status_codes[str(status)] += 1
            self.cortex.total_latency_ms += (time.perf_counter() - started) * 1000.0
            self.cortex.completed += 1
            if status >= 500:
                self.cortex.failed += 1
            if model:
                self.cortex.by_model[model] += 1

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Dispatch GET requests."""
        started = time.perf_counter()
        path = self.path.split("?")[0]
        with self.cortex.lock:
            self.cortex.in_flight += 1
        try:
            if path == "/health":
                self._send_json({"status": "ok"})
                return
            if path == "/ready":
                self._handle_ready()
                return
            if not self._authorized():
                self._send_error("Invalid or missing API key.", 401, code="invalid_api_key")
                return
            if path == "/v1/models":
                self._handle_models()
                return
            if path == "/admin/api/state":
                self.cortex.admin_calls.append("state")
                self._send_json(self.cortex.state())
                return
            if path == "/admin/api/metrics":
                self.cortex.admin_calls.append("metrics")
                self._send_json(self.cortex.metrics())
                return
            if path == "/admin/api/runtime":
                self.cortex.admin_calls.append("runtime")
                self._send_json(
                    {"models": {k: v.to_runtime_state() for k, v in self.cortex.models.items()}}
                )
                return
            if path == "/admin/api/models/files":
                self._send_json({"files": [{"name": "fake.gguf", "size": 1024, "path": "/models/fake.gguf"}]})
                return
            if path == "/admin/api/config/export":
                self._send_json(self.cortex.public_settings())
                return
            if path == "/admin/api/auth/options":
                self._send_json(
                    {"api_key_enabled": self.cortex.auth_enabled, "account_enabled": False}
                )
                return
            if path == "/admin/api/update/check":
                self._send_json({"update_available": False, "current": self.cortex.version})
                return
            if path == "/":
                body = b"<html><body>Fake Cortex LLMHoster</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_error(f"No route for GET {path}", 404, code="not_found")
        finally:
            with self.cortex.lock:
                self.cortex.in_flight = max(0, self.cortex.in_flight - 1)
            self._track("GET " + path, self._status_sent, started, getattr(self, "_request_model", ""))

    def do_POST(self) -> None:  # noqa: N802
        """Dispatch POST requests."""
        started = time.perf_counter()
        path = self.path.split("?")[0]
        with self.cortex.lock:
            self.cortex.in_flight += 1
        try:
            if not self._authorized():
                self._send_error("Invalid or missing API key.", 401, code="invalid_api_key")
                return
            if path == "/v1/chat/completions":
                self._handle_chat(started)
                return
            if path == "/v1/completions":
                self._handle_legacy_completion()
                return
            if path == "/v1/embeddings":
                self._handle_embeddings()
                return
            if path == "/v1/responses":
                self._handle_responses()
                return
            if path == "/v1/audio/transcriptions":
                self._handle_transcription()
                return
            if path == "/v1/rerank":
                self._handle_rerank()
                return
            if path == "/v1/moderations":
                self._send_json({"id": "mod_fake", "results": [{"flagged": False, "categories": {}}]})
                return
            if path.startswith("/admin/api/models/"):
                self._handle_admin_model_action(path)
                return
            if path == "/admin/api/config/reload":
                self._send_json({"ok": True, "reloaded": True})
                return
            if path == "/admin/api/metrics/reset":
                with self.cortex.lock:
                    self.cortex.status_codes.clear()
                    self.cortex.by_model.clear()
                    self.cortex.total_latency_ms = 0.0
                    self.cortex.completed = 0
                    self.cortex.failed = 0
                self._send_json({"ok": True})
                return
            self._send_error(f"No route for POST {path}", 404, code="not_found")
        finally:
            with self.cortex.lock:
                self.cortex.in_flight = max(0, self.cortex.in_flight - 1)
            self._track("POST " + path, self._status_sent, started, getattr(self, "_request_model", ""))

    def do_PUT(self) -> None:  # noqa: N802
        """Dispatch PUT requests (admin config updates)."""
        started = time.perf_counter()
        path = self.path.split("?")[0]
        try:
            if not self._authorized():
                self._send_error("Invalid or missing API key.", 401, code="invalid_api_key")
                return
            payload = self._read_json()
            if path == "/admin/api/config":
                self._send_json({"ok": True, "applied": sorted(payload)})
                return
            self._send_error(f"No route for PUT {path}", 404, code="not_found")
        finally:
            self._track("PUT " + path, self._status_sent, started)

    def do_DELETE(self) -> None:  # noqa: N802
        """Dispatch DELETE requests."""
        self._send_error("Method not supported by the fake hoster", 405, code="method_not_allowed")

    def do_HEAD(self) -> None:  # noqa: N802
        """Answer HEAD with headers only."""
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    _status_sent = 200

    def send_response(self, code: int, message: str | None = None) -> None:
        """Record the status actually sent so metrics stay accurate."""
        self._status_sent = code
        super().send_response(code, message)

    # -- handlers ------------------------------------------------------------

    def _handle_ready(self) -> None:
        """Mirror Cortex readiness: 503 unless the default model is running."""
        models = self.cortex.models
        if not models:
            self._send_json({"status": "not_ready", "reason": "no_models_configured"}, 503)
            return
        default = models.get(self.cortex.default_model)
        if default is not None and not default.running:
            self._send_json(
                {
                    "status": "not_ready",
                    "reason": "default_local_model_not_running",
                    "model": default.id,
                    "runtime_status": "stopped",
                },
                503,
            )
            return
        self._send_json({"status": "ready", "models": len(models)})

    def _handle_models(self) -> None:
        """Return the OpenAI-shaped model list."""
        self._send_json(
            {
                "object": "list",
                "data": [
                    model.to_public(default=model.id == self.cortex.default_model)
                    for model in self.cortex.models.values()
                ],
            }
        )

    def _handle_chat(self, started: float) -> None:
        """Serve chat completions, streaming or not."""
        payload = self._read_json()
        model_id = str(payload.get("model") or self.cortex.default_model)
        messages = payload.get("messages") or []
        if model_id not in self.cortex.models:
            self._send_error(f"The model `{model_id}` does not exist", 404, code="model_not_found")
            return
        model = self.cortex.models[model_id]
        if not model.running:
            self._send_error(
                f"Model {model_id} is configured but its runtime is not running",
                503,
                code="model_not_running",
            )
            return
        with self.cortex.lock:
            remaining_failures = self.cortex.fail_models.get(model_id, 0)
            if remaining_failures > 0:
                self.cortex.fail_models[model_id] = remaining_failures - 1
                self._send_error("Injected fake runtime failure", 503, code="injected_failure")
                return
            self.cortex.transcripts.append({"model": model_id, "messages": messages})
        self._request_model = model_id
        reply = _canned_reply([m for m in messages if isinstance(m, dict)], payload.get("tools"))
        usage = {
            "prompt_tokens": max(1, sum(len(json.dumps(m)) for m in messages) // 4),
            "completion_tokens": max(1, len(reply.get("text", "")) // 4 + 4 * len(reply.get("tool_calls") or [])),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        if payload.get("stream"):
            self._stream_reply(model_id, reply, usage)
            return
        body: dict[str, Any] = {
            "id": f"chatcmpl-fake-{int(time.time() * 1000) % 100000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply.get("text", ""),
                        **({"tool_calls": reply["tool_calls"]} if reply.get("tool_calls") else {}),
                    },
                    "finish_reason": reply.get("finish_reason", "stop"),
                }
            ],
            "usage": usage,
        }
        self._send_json(body)

    def _stream_reply(self, model_id: str, reply: dict[str, Any], usage: dict[str, int]) -> None:
        """Emit the reply as SSE chunks exactly like an OpenAI-compatible host."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunk_id = f"chatcmpl-fake-stream-{int(time.time() * 1000) % 100000}"

        def write(event: dict[str, Any]) -> None:
            """Write one SSE data frame using chunked transfer encoding."""
            frame = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(b"%X\r\n" % len(frame))
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        base = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
        }
        write({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        text = reply.get("text", "")
        size = 12
        for start in range(0, len(text), size):
            if self.cortex.chunk_delay:
                time.sleep(self.cortex.chunk_delay)
            write({
                **base,
                "choices": [{"index": 0, "delta": {"content": text[start : start + size]}, "finish_reason": None}],
            })
        for index, call in enumerate(reply.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = str(function.get("arguments") or "")
            first = {
                "index": index,
                "id": call.get("id"),
                "type": "function",
                "function": {"name": function.get("name", ""), "arguments": arguments[:8]},
            }
            write({**base, "choices": [{"index": 0, "delta": {"tool_calls": [first]}, "finish_reason": None}]})
            if len(arguments) > 8:
                rest = {"index": index, "function": {"arguments": arguments[8:]}}
                write({**base, "choices": [{"index": 0, "delta": {"tool_calls": [rest]}, "finish_reason": None}]})
        write({
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": reply.get("finish_reason", "stop")}],
            "usage": usage,
        })
        done = b"data: [DONE]\n\n"
        self.wfile.write(b"%X\r\n" % len(done))
        self.wfile.write(done)
        self.wfile.write(b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _handle_legacy_completion(self) -> None:
        """Serve ``POST /v1/completions``."""
        payload = self._read_json()
        prompt = str(payload.get("prompt") or "")
        model_id = str(payload.get("model") or self.cortex.default_model)
        text = f"fake completion for: {prompt[:120]}"
        self._send_json(
            {
                "id": "cmpl-fake",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
                "usage": {
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max(1, len(text) // 4),
                    "total_tokens": max(2, (len(prompt) + len(text)) // 4),
                },
            }
        )

    def _handle_embeddings(self) -> None:
        """Serve ``POST /v1/embeddings`` with deterministic vectors."""
        payload = self._read_json()
        model_id = str(payload.get("model") or EMBEDDING_MODEL)
        raw_input = payload.get("input")
        texts = raw_input if isinstance(raw_input, list) else [str(raw_input or "")]
        data = [
            {"object": "embedding", "index": index, "embedding": fake_embedding(str(text), model_id)}
            for index, text in enumerate(texts)
        ]
        self._send_json(
            {
                "object": "list",
                "data": data,
                "model": model_id,
                "usage": {"prompt_tokens": sum(len(str(t)) // 4 + 1 for t in texts), "total_tokens": 0},
            }
        )

    def _handle_responses(self) -> None:
        """Serve the Responses API dialect."""
        payload = self._read_json()
        text = f"fake responses reply to {str(payload.get('input'))[:80]}"
        self._send_json(
            {
                "id": "resp_fake",
                "object": "response",
                "status": "completed",
                "model": payload.get("model") or self.cortex.default_model,
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                "output_text": text,
                "usage": {"input_tokens": 8, "output_tokens": max(1, len(text) // 4)},
            }
        )

    def _handle_transcription(self) -> None:
        """Serve multipart audio transcription."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(length, 32 * 1024 * 1024)) if length else b""
        digest = hashlib.sha256(raw).hexdigest()[:12]
        self._send_json(
            {"task": "transcribe", "language": "english", "duration": 1.5, "text": f"fake transcript {digest}"}
        )

    def _handle_rerank(self) -> None:
        """Serve reranking with a naive overlap scorer."""
        payload = self._read_json()
        query = str(payload.get("query") or "").lower().split()
        documents = payload.get("documents") or []
        scored = []
        for index, document in enumerate(documents):
            words = str(document).lower().split()
            overlap = len(set(query) & set(words))
            scored.append({"index": index, "relevance_score": round(overlap / max(1, len(query)), 4)})
        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        top_n = int(payload.get("top_n") or len(scored))
        self._send_json({"id": "rerank_fake", "model": payload.get("model"), "results": scored[:top_n]})

    def _handle_admin_model_action(self, path: str) -> None:
        """Serve model start/stop/restart/test admin actions."""
        payload = self._read_json()
        model_id = str(payload.get("id") or payload.get("model") or "")
        self.cortex.admin_calls.append(path)
        with self.cortex.lock:
            if path.endswith("/stop-all"):
                for model in self.cortex.models.values():
                    model.running = False
                self._send_json({"ok": True, "stopped": sorted(self.cortex.models)})
                return
            if model_id not in self.cortex.models:
                self._send_error(f"Unknown model {model_id}", 404, code="model_not_found")
                return
            model = self.cortex.models[model_id]
            if path.endswith("/start"):
                model.running = True
            elif path.endswith("/stop"):
                model.running = False
            elif path.endswith("/restart"):
                model.running = False
                model.running = True
            elif path.endswith("/test"):
                self._send_json(
                    {"ok": model.running, "model": model_id, "reply": "pong" if model.running else ""}
                )
                return
            elif path.endswith("/detect-capabilities"):
                self._send_json({"capabilities": ["chat", "tool_calling"]})
                return
            self._send_json({"ok": True, "model": model.to_runtime_state()})


class FakeCortexServer:
    """A threaded HTTP server hosting :class:`FakeCortexState`.

    Use it as a context manager to get an ephemeral port::

        with FakeCortexServer() as server:
            backend = CortexBackend(CortexSettings(base_url=server.url))
    """

    def __init__(
        self,
        state: FakeCortexState | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        verbose: bool = False,
    ) -> None:
        self.state = state or FakeCortexState()
        self.host = host
        self.verbose = verbose
        self._httpd = ThreadingHTTPServer((host, port), _FakeCortexHandler)
        self._httpd.state = self.state  # type: ignore[attr-defined]
        self._httpd.verbose = verbose  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Return the bound port (ephemeral when constructed with ``port=0``)."""
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        """Return the base URL clients should use."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> FakeCortexServer:
        """Start serving in a background daemon thread."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="fake-cortex", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the socket."""
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self._httpd.server_close()

    def reset(self) -> None:
        """Clear recorded requests and transcripts."""
        with self.state.lock:
            self.state.requests.clear()
            self.state.status_codes.clear()
            self.state.by_model.clear()
            self.state.transcripts.clear()
            self.state.admin_calls.clear()
            self.state.completed = 0
            self.state.failed = 0
            self.state.total_latency_ms = 0.0

    def fail_next(self, model_id: str, times: int = 1) -> None:
        """Make the next ``times`` requests for a model return HTTP 503."""
        with self.state.lock:
            self.state.fail_models[model_id] = times

    def __enter__(self) -> FakeCortexServer:
        """Start the server for a ``with`` block."""
        return self.start()

    def __exit__(self, *exc: object) -> None:
        """Stop the server when leaving the ``with`` block."""
        self.stop()


def start_fake_cortex(**kwargs: Any) -> FakeCortexServer:
    """Start a fake Cortex server and return it already running."""
    return FakeCortexServer(**kwargs).start()


__all__ = [
    "DEFAULT_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "FakeCortexServer",
    "FakeCortexState",
    "FakeModel",
    "fake_embedding",
    "start_fake_cortex",
]
