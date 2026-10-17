"""The embedded HTTP server: REST API plus the bundled web UI.

``magent serve`` starts a :class:`MagentServer` - a stdlib
``ThreadingHTTPServer`` with a small JSON API over the same objects the CLI
uses.  It exists so MAgent can be driven from a browser, from a script in
another language, or from a CI job, without adding a web framework dependency.

Endpoints (see :meth:`MagentServer.routes` for the live list):

``GET  /api/health``      liveness, version and uptime
``GET  /api/info``        configuration summary and capability counts
``GET  /api/models``      models the backend reports
``GET  /api/tools``       every registered tool with its JSON schema
``POST /api/tools/{name}``dispatch one tool
``POST /api/chat``        run the agent on a prompt
``GET  /api/chat/stream`` server-sent events for one run
``GET  /api/sessions``    stored sessions
``GET  /api/metrics``     Prometheus text or a JSON snapshot
``GET  /api/skills``      the skill library
``GET  /api/presets``     agent presets
``POST /api/rag/ingest``  index files for retrieval
``GET  /api/rag/search``  retrieve cited chunks

Auth is a bearer token when ``http.api_key`` (or ``MAGENT_HTTP_API_KEY``) is set;
origins come from ``http.cors_origins`` unless ``allow_origins_any`` is set, which
is the default for local use.  The server binds to ``0.0.0.0`` when asked, sends
no framing headers, and answers ``OPTIONS`` preflights, so it works behind the
sandbox's port proxy as well as on localhost.
"""

from __future__ import annotations

import hmac
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .._version import __version__
from ..errors import MAgentError, ServerError_, ToolNotFoundError, ToolValidationError
from ..logging_ import get_logger
from ..types import EventKind
from ..utils.files import iter_files, read_text_safe
from ..utils.ids import ulid_like
from ..utils.text import ellipsize
from .routes import Router, RouteMatch

LOGGER = get_logger("server")
"""Logger for HTTP requests and errors."""

DEFAULT_HOST = "127.0.0.1"
"""Default bind address - loopback unless the user opts out."""

DEFAULT_PORT = 8765
"""Default port; deliberately not Cortex's 8624."""

STATIC_DIR = Path(__file__).parent / "static"
"""Directory holding the bundled UI."""

INDEX_FILE = "index.html"
"""File served at ``/``."""

SSE_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Type", "text/event-stream; charset=utf-8"),
    ("Cache-Control", "no-cache, no-transform"),
    ("Connection", "keep-alive"),
    ("X-Accel-Buffering", "no"),
)
"""Headers required for a working server-sent-events stream."""

PUBLIC_ROUTES: tuple[str, ...] = ("/api/health", "/", "/index.html", "/favicon.ico", "/api")
"""Paths that never require a token."""


@dataclass(slots=True)
class ServerStats:
    """Request counters kept by the handler."""

    started_at: float = field(default_factory=time.time)
    requests: int = 0
    errors: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    paths: dict[str, int] = field(default_factory=dict)
    bytes_out: int = 0
    runs: int = 0
    last_request_at: float = 0.0

    def record(self, path: str, status: int, size: int) -> None:
        """Record one completed request."""
        self.requests += 1
        self.statuses[str(status)] = self.statuses.get(str(status), 0) + 1
        self.paths[path] = self.paths.get(path, 0) + 1
        self.bytes_out += max(0, size)
        self.last_request_at = time.time()
        if status >= 400:
            self.errors += 1

    @property
    def uptime(self) -> float:
        """Seconds since the server started."""
        return max(0.0, time.time() - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the counters."""
        return {
            "uptime_seconds": round(self.uptime, 3),
            "requests": self.requests,
            "errors": self.errors,
            "statuses": dict(self.statuses),
            "top_paths": sorted(self.paths.items(), key=lambda item: -item[1])[:10],
            "bytes_out": self.bytes_out,
            "runs": self.runs,
            "last_request_at": self.last_request_at,
        }


def _preset_names() -> list[str]:
    """Names of the registered agent presets, without importing them eagerly."""
    from ..agents.presets import preset_names

    return preset_names()


class MagentServer:
    """Owns the shared agent dependencies and the HTTP server socket."""

    def __init__(self, config: Any = None, *, backend: Any = None, registry: Any = None,
                 memory: Any = None, guardrails: Any = None, sessions: Any = None,
                 retriever: Any = None, skills: Any = None, observability: Any = None,
                 agent_factory: Any = None) -> None:
        """Wire the server.

        Args:
            config: loaded configuration
            backend: LLM backend shared by every request
            registry: tool registry; the default one when omitted
            memory: long-term memory store
            guardrails: guardrail manager
            sessions: session manager
            retriever: RAG retriever
            skills: skill library
            observability: observability façade, used for metrics
            agent_factory: callable ``(session_id, options) -> Agent`` overriding the default
        """
        self.config = config
        self.backend = backend
        self.registry = registry
        self.memory = memory
        self.guardrails = guardrails
        self.sessions = sessions
        self.retriever = retriever
        self.skills = skills
        self.observability = observability
        self.agent_factory = agent_factory
        self.stats = ServerStats()
        self.started_at = time.time()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.router = Router()
        self._register_routes()

    # -- configuration -------------------------------------------------------

    @property
    def http_config(self) -> Any:
        """The HTTP section of the configuration."""
        return getattr(self.config, "http", None)

    @property
    def host(self) -> str:
        """Bind address."""
        return str(getattr(self.http_config, "host", "") or DEFAULT_HOST)

    @property
    def port(self) -> int:
        """Bind port."""
        return int(getattr(self.http_config, "port", 0) or DEFAULT_PORT)

    @property
    def api_key(self) -> str:
        """Bearer token required when set."""
        return str(getattr(self.http_config, "api_key", "") or "")

    @property
    def max_body_bytes(self) -> int:
        """Maximum accepted request body size."""
        return int(getattr(self.http_config, "max_body_bytes", 0) or 1_000_000)

    @property
    def serve_ui(self) -> bool:
        """Whether the bundled web UI is served at ``/``."""
        return bool(getattr(self.http_config, "serve_ui", True))

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Origins allowed to call the API."""
        if bool(getattr(self.http_config, "allow_origins_any", True)):
            return ("*",)
        raw = getattr(self.http_config, "cors_origins", ()) or ()
        if isinstance(raw, str):
            return tuple(part.strip() for part in raw.split(",") if part.strip())
        return tuple(str(item) for item in raw)

    @property
    def url(self) -> str:
        """Base URL of a running server, using the port actually bound."""
        host = self.host if self.host not in {"0.0.0.0", ""} else "127.0.0.1"
        port = self.bound_port if self.running else self.port
        return f"http://{host}:{port}"

    # -- agent construction --------------------------------------------------

    def build_agent(self, session_id: str = "", **options: Any) -> Any:
        """Create an agent for one request.

        A fresh agent per request keeps concurrent callers from sharing a
        conversation; tools, memory, guardrails and sessions are shared because
        they are explicitly thread-safe.
        """
        if self.agent_factory is not None:
            return self.agent_factory(session_id, options)
        from ..agent import Agent
        from ..llm.factory import build_backend

        backend = self.backend if self.backend is not None else build_backend(config=self.config)
        registry = self.registry
        if registry is None:
            from ..tools import default_registry

            registry = default_registry()
        return Agent(backend=backend, settings=self.config, registry=registry,
                     memory=self.memory, guardrails=self.guardrails, session=None,
                     name="MAgent[api]",
                     workspace=getattr(getattr(self.config, "tools", None), "workspace", None))

    def run_prompt(self, prompt: str, *, session_id: str = "", **options: Any) -> dict[str, Any]:
        """Run one prompt and return a JSON-friendly result.

        Raises:
            MAgentError: when the prompt is empty
        """
        from ..errors import ConfigError

        if not prompt.strip():
            raise ConfigError("prompt must not be empty")
        agent = self.build_agent(session_id, **options)
        started = time.perf_counter()
        result = agent.run(prompt, session_id=session_id or "", **options)
        self.stats.runs += 1
        if self.sessions is not None:
            try:
                self.sessions.record_run(result, session_id=session_id or "")
            except Exception as exc:  # noqa: BLE001 - persistence is best effort
                LOGGER.debug("session record failed: %s", exc)
        return {
            "run_id": result.run_id,
            "output": result.output,
            "stop_reason": str(getattr(result.stop_reason, "value", result.stop_reason)),
            "model": result.model,
            "usage": result.usage.to_dict(),
            "cost": result.cost.to_dict(),
            "steps": [
                {"index": step.index, "thought": ellipsize(step.thought, 400),
                 "tools": [call.name for call in step.tool_calls],
                 "seconds": round(step.duration_ms / 1000, 4)}
                for step in result.steps
            ],
            "seconds": round(time.perf_counter() - started, 4),
            "error": result.error,
            "session_id": session_id,
        }

    # -- routes --------------------------------------------------------------

    def _register_routes(self) -> None:
        """Populate the router."""
        router = self.router
        router.get("/api", self.api_index, description="List every route", auth=False)
        router.get("/api/health", self.api_health, description="Liveness and version", auth=False)
        router.get("/api/info", self.api_info, description="Configuration summary")
        router.get("/api/models", self.api_models, description="Models the backend reports")
        router.get("/api/tools", self.api_tools, description="Registered tools and schemas")
        router.post("/api/tools/{name}", self.api_tool_call, description="Dispatch one tool")
        router.post("/api/chat", self.api_chat, description="Run the agent on a prompt")
        router.get("/api/chat/stream", self.api_chat_stream, description="Server-sent event stream",
                   auth=True, stream=True)
        router.get("/api/presets", self.api_presets, description="Agent presets")
        router.get("/api/skills", self.api_skills, description="Skill library")
        router.get("/api/sessions", self.api_sessions, description="Stored sessions")
        router.get("/api/sessions/{id}", self.api_session, description="One stored session")
        router.delete("/api/sessions/{id}", self.api_session_delete, description="Delete a session")
        router.get("/api/memory", self.api_memory, description="Search long-term memory")
        router.post("/api/memory", self.api_memory_add, description="Store a memory")
        router.get("/api/metrics", self.api_metrics, description="Metrics snapshot")
        router.get("/api/rag/search", self.api_rag_search, description="Retrieve cited chunks")
        router.post("/api/rag/ingest", self.api_rag_ingest, description="Index files")
        router.get("/", self.serve_index, description="Web UI", auth=False)
        router.get("/index.html", self.serve_index, description="Web UI", auth=False)
        router.get("/favicon.ico", self.serve_favicon, description="Favicon", auth=False)

    # -- handlers ------------------------------------------------------------

    def api_index(self, request: "Request") -> tuple[int, Any]:
        """Return the route table."""
        del request
        return 200, {"name": "MAgent HTTP API", "version": __version__,
                     "routes": self.router.describe()}

    def api_health(self, request: "Request") -> tuple[int, Any]:
        """Report liveness, version and uptime."""
        del request
        return 200, {"status": "ok", "version": __version__, "uptime_seconds": self.stats.uptime,
                     "runs": self.stats.runs, "started_at": self.stats.started_at}

    def api_info(self, request: "Request") -> tuple[int, Any]:
        """Summarise the running configuration."""
        del request
        library = self.skills
        if library is None:
            from ..skills import build_library

            library = build_library(self.config)
            self.skills = library
        sessions = self._sessions()
        payload: dict[str, Any] = {
            "version": __version__,
            "backend": type(self.backend).__name__ if self.backend is not None else "lazy",
            "workspace": str(getattr(getattr(self.config, "tools", None), "workspace", "")),
            "tools": self._registry().count(),
            "tool_categories": len(self._registry().categories()),
            "sessions": len(sessions.store),
            "memory": bool(self.memory),
            "retriever": bool(self.retriever),
            "skills": len(library),
            "guardrails": bool(self.guardrails),
            "presets": len(_preset_names()),
            "stats": self.stats.to_dict(),
        }
        if getattr(sessions, "current", None) is not None:
            payload["current_session"] = sessions.current.id
        return 200, payload

    def api_models(self, request: "Request") -> tuple[int, Any]:
        """List models from the backend, with the Cortex admin view when available."""
        del request
        backend = self.backend
        if backend is None:
            from ..llm.factory import build_backend

            backend = build_backend(config=self.config)
        models: list[dict[str, Any]] = []
        try:
            for model in backend.models() if hasattr(backend, "models") else []:
                models.append({"id": getattr(model, "id", str(model)),
                               "capabilities": list(getattr(model, "capabilities", []) or []),
                               "runtime": getattr(model, "runtime", "")})
        except Exception as exc:  # noqa: BLE001 - a probe failure is not an API failure
            LOGGER.debug("model listing failed: %s", exc)
        health: dict[str, Any] = {}
        if hasattr(backend, "health"):
            try:
                health = backend.health()
            except Exception as exc:  # noqa: BLE001
                health = {"ok": False, "error": str(exc)}
        return 200, {"models": models, "backend": type(backend).__name__, "health": health}

    def api_tools(self, request: "Request") -> tuple[int, Any]:
        """List tools, optionally filtered by category or a text query."""
        registry = self._registry()
        category = request.query.get("category", "")
        search = request.query.get("q", "")
        if search:
            return 200, {"tools": [item.to_openai() for item in registry.search(text=search)],
                         "query": search}
        tools = registry.filter(categories=[category] if category else None)
        return 200, {"tools": [item.to_openai() for item in tools], "count": len(tools),
                     "categories": sorted(registry.categories())}

    def api_tool_call(self, request: "Request") -> tuple[int, Any]:
        """Dispatch a single tool call."""
        registry = self._registry()
        name = request.params.get("name", "")
        try:
            tool = registry.require(name)
        except ToolNotFoundError as exc:
            return 404, {"error": str(exc), "suggestions": registry.suggest(name)}
        arguments = request.json or {}
        context = self._tool_context()
        try:
            result = tool.run(arguments, context=context)
        except ToolValidationError as exc:
            return 400, {"error": str(exc), "tool": name, "schema": tool.to_openai()}
        return 200 if result.ok else 400, {"tool": name, "ok": result.ok,
                                           "content": result.content, "error": result.error,
                                           "error_code": result.error_code,
                                           "seconds": round(result.duration_ms / 1000, 4)}

    def api_chat(self, request: "Request") -> tuple[int, Any]:
        """Run the agent on one prompt."""
        payload = request.json or {}
        prompt = str(payload.get("prompt") or request.query.get("prompt", ""))
        session_id = str(payload.get("session_id") or request.query.get("session_id", ""))
        options = {key: value for key, value in payload.items()
                   if key in {"loop_mode", "max_iterations", "temperature", "model", "persona",
                              "include_files", "structured"}}
        return 200, self.run_prompt(prompt, session_id=session_id, **options)

    def api_chat_stream(self, request: "Request") -> tuple[int, Any]:
        """Stream a run as server-sent events."""
        prompt = str(request.query.get("prompt", ""))
        session_id = str(request.query.get("session_id", ""))
        if not prompt.strip():
            return 400, {"error": "prompt must not be empty"}
        agent = self.build_agent(session_id)
        stream_events: list[dict[str, Any]] = []

        def generator() -> Iterable[bytes]:
            started = time.perf_counter()
            yield _sse("open", {"run": "started"})
            try:
                for event in agent.stream(prompt, session_id=session_id):
                    payload = _event_payload(event)
                    stream_events.append(payload)
                    yield _sse(str(getattr(event.kind, "value", event.kind)), payload)
            except Exception as exc:  # noqa: BLE001 - errors are part of the stream
                yield _sse("error", {"message": str(exc), "type": type(exc).__name__})
            self.stats.runs += 1
            yield _sse("done", {"seconds": round(time.perf_counter() - started, 4)})

        return 200, {"__sse__": generator()}

    def api_presets(self, request: "Request") -> tuple[int, Any]:
        """List agent presets."""
        del request
        from ..agents.presets import describe_presets

        return 200, {"presets": describe_presets()}

    def api_skills(self, request: "Request") -> tuple[int, Any]:
        """List skills, optionally matching a task."""
        library = self.skills
        if library is None:
            from ..skills import build_library

            library = build_library(self.config)
        query = request.query.get("task", "")
        if query:
            matches = [{"skill": match.skill.to_dict(), "score": match.score}
                       for match in library.search(query)]
            return 200, {"matches": matches, "task": query}
        return 200, {"skills": [skill.to_dict() for skill in library],
                     "stats": library.stats()}

    def api_sessions(self, request: "Request") -> tuple[int, Any]:
        """List stored sessions."""
        sessions = self._sessions()
        limit = _int(request.query.get("limit", 25), 25)
        query = request.query.get("q", "")
        tag = request.query.get("tag", "")
        return 200, {"sessions": [info.to_dict() for info in sessions.list(limit=limit, tag=tag,
                                                                          query=query)],
                     "stats": sessions.store.stats() if hasattr(sessions, "store") else {}}

    def api_session(self, request: "Request") -> tuple[int, Any]:
        """Return one session with its transcript."""
        from ..session import render_transcript

        sessions = self._sessions()
        record = sessions.load(request.params.get("id", ""))
        if record is None:
            return 404, {"error": "unknown session"}
        fmt = request.query.get("format", "markdown")
        return 200, {"session": record.to_dict(), "transcript": render_transcript(record, fmt=fmt)}

    def api_session_delete(self, request: "Request") -> tuple[int, Any]:
        """Delete one session."""
        sessions = self._sessions()
        removed = sessions.delete(request.params.get("id", ""))
        return (200 if removed else 404), {"deleted": removed}

    def api_memory(self, request: "Request") -> tuple[int, Any]:
        """Search long-term memory."""
        if self.memory is None:
            return 200, {"entries": [], "enabled": False}
        query = request.query.get("q", "")
        limit = _int(request.query.get("limit", 10), 10)
        entries = self.memory.search(query, limit=limit) if query else self.memory.recent(limit)
        return 200, {"entries": [entry.to_dict() for entry in entries], "query": query}

    def api_memory_add(self, request: "Request") -> tuple[int, Any]:
        """Store one memory."""
        if self.memory is None:
            return 409, {"error": "memory is disabled"}
        payload = request.json or {}
        content = str(payload.get("content", "")).strip()
        if not content:
            return 400, {"error": "content is required"}
        entry = self.memory.add(content, kind=str(payload.get("kind", "fact")),
                                tags=list(payload.get("tags", []) or []))
        return 201, {"entry": entry.to_dict()}

    def api_metrics(self, request: "Request") -> tuple[int, Any]:
        """Return metrics as Prometheus text or a JSON snapshot."""
        want = request.query.get("format", "json")
        if self.observability is not None:
            if want == "prometheus":
                return 200, {"__text__": self.observability.prometheus(),
                             "__content_type__": "text/plain; version=0.0.4; charset=utf-8"}
            return 200, {"metrics": self.observability.report().to_dict(),
                         "values": self.observability.metrics.flat()}
        if want == "prometheus":
            return 200, {"__text__": _prometheus_from_stats(self.stats),
                         "__content_type__": "text/plain; charset=utf-8"}
        return 200, {"metrics": self.stats.to_dict()}

    def api_rag_search(self, request: "Request") -> tuple[int, Any]:
        """Retrieve cited chunks for a query."""
        if self.retriever is None:
            return 200, {"chunks": [], "enabled": False}
        query = request.query.get("q", "")
        if not query.strip():
            return 400, {"error": "q is required"}
        limit = _int(request.query.get("limit", 5), 5)
        result = self.retriever.retrieve(query, limit=limit)
        return 200, {"query": query, "chunks": [chunk.to_dict() for chunk in result.chunks],
                     "context": result.context, "stats": self.retriever.stats()}

    def api_rag_ingest(self, request: "Request") -> tuple[int, Any]:
        """Index files or inline text for retrieval."""
        if self.retriever is None:
            return 200, {"ingested": 0, "enabled": False}
        payload = request.json or {}
        text = str(payload.get("text", ""))
        if text:
            chunks = self.retriever.ingest_text(text, source=str(payload.get("source", "inline")))
            return 201, {"chunks": chunks, "source": payload.get("source", "inline")}
        paths = list(payload.get("paths", []) or [])
        report = self.retriever.ingest(paths, root=str(payload.get("root", "") or "") or None)
        return 201, {"report": report.to_dict(), "stats": self.retriever.stats()}

    def serve_index(self, request: "Request") -> tuple[int, Any]:
        """Serve the bundled web UI (or a JSON hint when it is disabled)."""
        del request
        if not self.serve_ui:
            return 404, {"error": "the web UI is disabled (http.serve_ui = false)"}
        index = STATIC_DIR / INDEX_FILE
        if not index.is_file():
            return 500, {"error": "the bundled UI file is missing",
                         "path": str(index)}
        return 200, {"__text__": read_text_safe(index, limit=2_000_000),
                     "__content_type__": "text/html; charset=utf-8"}

    def serve_favicon(self, request: "Request") -> tuple[int, Any]:
        """Answer the favicon request with an inline SVG-in-a-PNG-free 204."""
        del request
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
               "<rect width='32' height='32' rx='7' fill='#0f1115'/>"
               "<text x='16' y='22' font-size='16' text-anchor='middle' fill='#7fd1ff'>M</text>"
               "</svg>")
        return 200, {"__text__": svg, "__content_type__": "image/svg+xml"}

    # -- helpers -------------------------------------------------------------

    def _registry(self) -> Any:
        """Return the tool registry, building the default one if needed."""
        if self.registry is None:
            from ..tools import default_registry

            self.registry = default_registry()
        return self.registry

    def _sessions(self) -> Any:
        """Return the session manager, building one if needed."""
        if self.sessions is None:
            from ..session import build_session_manager

            self.sessions = build_session_manager(self.config)
        return self.sessions

    def _tool_context(self) -> Any:
        """Build a tool context for API-dispatched calls."""
        from ..tools import ToolContext, build_sandbox

        return ToolContext(config=self.config, registry=self.registry, sandbox=build_sandbox(
            self.config, workspace=getattr(getattr(self.config, "tools", None), "workspace", None)))

    # -- serving -------------------------------------------------------------

    def create_httpd(self, host: str = "", port: int = 0) -> ThreadingHTTPServer:
        """Create (but do not start) the HTTP server."""
        server = self
        bind_host = host or self.host
        bind_port = port or self.port

        class Handler(BaseHTTPRequestHandler):
            """Request handler bound to one :class:`MagentServer`."""

            server_version = f"MAgent/{__version__}"
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                """Route the stdlib access log into MAgent's logger."""
                LOGGER.debug("http %s - %s", self.address_string(), fmt % args)

            def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
                """Handle GET."""
                server._handle(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                """Handle POST."""
                server._handle(self, "POST")

            def do_PUT(self) -> None:  # noqa: N802
                """Handle PUT."""
                server._handle(self, "PUT")

            def do_PATCH(self) -> None:  # noqa: N802
                """Handle PATCH."""
                server._handle(self, "PATCH")

            def do_DELETE(self) -> None:  # noqa: N802
                """Handle DELETE."""
                server._handle(self, "DELETE")

            def do_HEAD(self) -> None:  # noqa: N802
                """Handle HEAD like GET without a body."""
                server._handle(self, "HEAD")

            def do_OPTIONS(self) -> None:  # noqa: N802
                """Answer CORS preflight requests."""
                server._send_options(self)

        httpd = ThreadingHTTPServer((bind_host, bind_port), Handler)
        httpd.daemon_threads = True
        httpd.allow_reuse_address = True
        try:
            httpd.daemon_threads = True
            httpd.request_queue_size = int(getattr(self.http_config, "threads", 0) or 16)
        except Exception:  # noqa: BLE001 - tuning is best effort
            pass
        self._httpd = httpd
        self._bound_port = int(httpd.server_address[1])
        return httpd

    @property
    def bound_port(self) -> int:
        """Port actually bound, which matters when 0 was requested."""
        return int(getattr(self, "_bound_port", self.port))

    def serve_forever(self, *, host: str = "", port: int = 0) -> None:
        """Run the server in the calling thread until interrupted."""
        httpd = self.create_httpd(host, port)
        LOGGER.info("serving MAgent API on http://%s:%s", self.host, self.bound_port)
        try:
            httpd.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            LOGGER.info("shutting down")
        finally:
            httpd.server_close()

    def start_background(self, *, host: str = "", port: int = 0) -> str:
        """Start the server on a daemon thread and return its URL."""
        httpd = self.create_httpd(host, port)
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.25},
                                        name="magent-http", daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.bound_port}"

    def stop(self) -> None:
        """Stop the server and wait for the thread to finish."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    @property
    def running(self) -> bool:
        """True while a server socket is open."""
        return self._httpd is not None

    def describe(self) -> dict[str, Any]:
        """Describe the server configuration."""
        return {
            "host": self.host,
            "port": self.bound_port if self.running else self.port,
            "url": self.url,
            "serve_ui": self.serve_ui,
            "auth": bool(self.api_key),
            "cors_origins": list(self.allowed_origins),
            "max_body_bytes": self.max_body_bytes,
            "routes": len(self.router),
            "running": self.running,
            "stats": self.stats.to_dict(),
        }

    # -- request handling ----------------------------------------------------

    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """Resolve and dispatch one request."""
        parsed = urlparse(handler.path)
        path = unquote(parsed.path or "/")
        query = {key: values[0] if values else "" for key, values in
                 parse_qs(parsed.query, keep_blank_values=True).items()}
        match = self.router.resolve(method, path)
        origin = handler.headers.get("Origin", "")
        if method == "OPTIONS":
            self._send_options(handler)
            return
        if not match.found:
            if match.method_not_allowed:
                self._send_json(handler, 405, {"error": f"method {method} not allowed",
                                               "allowed": list(match.allowed)}, origin)
            else:
                self._send_json(handler, 404, {"error": f"no route for {path}"}, origin)
            return
        assert match.route is not None
        route = match.route
        if route.auth and not self._authorized(handler):
            self._send_json(handler, 401, {"error": "missing or invalid API key"}, origin)
            return
        try:
            body, too_large = self._read_body(handler)
            if too_large:
                self._send_json(handler, 413, {"error": "request body too large"}, origin)
                return
            request = Request(method=method, path=path, query=query, params=match.params,
                              json=body, headers=dict(handler.headers), origin=origin,
                              client=handler.client_address[0] if handler.client_address else "")
            status, payload = route.handler(request)
        except MAgentError as exc:
            LOGGER.debug("api error on %s: %s", path, exc)
            self._send_json(handler, _status_for(exc), {"error": str(exc),
                                                        "type": type(exc).__name__,
                                                        "details": getattr(exc, "details", {})}, origin)
            self.stats.record(path, _status_for(exc), 0)
            return
        except Exception as exc:  # noqa: BLE001 - the server must never die on one request
            LOGGER.warning("unhandled error on %s: %s", path, exc)
            self._send_json(handler, 500, {"error": "internal error", "type": type(exc).__name__,
                                           "message": str(exc)}, origin)
            self.stats.record(path, 500, 0)
            return
        size = self._send(handler, status, payload, origin, head=(method == "HEAD"))
        self.stats.record(path, status, size)

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        """Check the bearer token, when one is configured."""
        token = self.api_key
        if not token:
            return True
        header = handler.headers.get("Authorization", "")
        supplied = ""
        if header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        supplied = supplied or handler.headers.get("X-API-Key", "").strip()
        return bool(supplied) and hmac.compare_digest(supplied, token)

    def _read_body(self, handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any], bool]:
        """Read and parse a JSON request body."""
        length = _int(handler.headers.get("Content-Length", 0), 0)
        if length <= 0:
            return {}, False
        if length > self.max_body_bytes:
            return {}, True
        raw = handler.rfile.read(length)
        if not raw.strip():
            return {}, False
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise ServerError_(f"invalid JSON body: {exc}") from exc
        return (payload if isinstance(payload, dict) else {"value": payload}), False

    def _cors_headers(self, origin: str) -> list[tuple[str, str]]:
        """CORS headers for an origin."""
        allowed = self.allowed_origins
        if not allowed:
            return []
        if "*" in allowed:
            return [("Access-Control-Allow-Origin", "*")]
        if origin and origin in allowed:
            return [("Access-Control-Allow-Origin", origin), ("Vary", "Origin"),
                    ("Access-Control-Allow-Credentials", "true")]
        return []

    def _send_options(self, handler: BaseHTTPRequestHandler) -> None:
        """Send a CORS preflight response."""
        handler.send_response(204)
        for key, value in self._cors_headers(handler.headers.get("Origin", "")):
            handler.send_header(key, value)
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        handler.send_header("Access-Control-Max-Age", "600")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, payload: Any,
                   origin: str = "") -> int:
        """Send a JSON response."""
        return self._send(handler, status, payload, origin)

    def _send(self, handler: BaseHTTPRequestHandler, status: int, payload: Any, origin: str = "",
              *, head: bool = False) -> int:
        """Write a response, handling SSE, raw text and JSON payloads."""
        if isinstance(payload, dict) and "__sse__" in payload:
            # The stream ends when the generator ends, so the connection must not
            # be kept alive afterwards: without this the client waits for a
            # Content-Length that never arrives.
            handler.close_connection = True
            handler.send_response(status)
            for key, value in SSE_HEADERS:
                handler.send_header(key, value)
            for key, value in self._cors_headers(origin):
                handler.send_header(key, value)
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.end_headers()
            written = 0
            try:
                for chunk in payload["__sse__"]:
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                    written += len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.debug("client disconnected from an event stream")
            finally:
                handler.close_connection = True
            return written
        if isinstance(payload, dict) and "__text__" in payload:
            body = str(payload["__text__"]).encode("utf-8")
            content_type = str(payload.get("__content_type__", "text/plain; charset=utf-8"))
        else:
            body = json.dumps(_jsonable(payload), ensure_ascii=False, default=str).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("Cache-Control", "no-store")
        for key, value in self._cors_headers(origin):
            handler.send_header(key, value)
        handler.end_headers()
        if not head:
            try:
                handler.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.debug("client disconnected")
        return len(body)


@dataclass(slots=True)
class Request:
    """A parsed API request handed to a route handler."""

    method: str
    path: str
    query: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    origin: str = ""
    client: str = ""

    def header(self, name: str, default: str = "") -> str:
        """Case-insensitive header lookup."""
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default

    def to_dict(self) -> dict[str, Any]:
        """Serialise the request without its body."""
        return {"method": self.method, "path": self.path, "query": dict(self.query),
                "params": dict(self.params), "origin": self.origin, "client": self.client}


def _status_for(exc: MAgentError) -> int:
    """Map an error to an HTTP status code."""
    code = getattr(exc, "code", "") or ""
    mapping = {
        "config_error": 400,
        "config_validation_error": 400,
        "tool_not_found": 404,
        "tool_validation_error": 400,
        "tool_execution_error": 400,
        "tool_timeout": 504,
        "permission_denied": 403,
        "sandbox_violation": 403,
        "approval_required": 403,
        "guardrail_error": 400,
        "prompt_injection": 400,
        "policy_violation": 403,
        "budget_exceeded": 429,
        "rate_limited": 429,
        "rate_limit_exceeded": 429,
        "content_blocked": 400,
        "not_found": 404,
        "model_not_found": 404,
        "model_not_running": 409,
        "session_not_found": 404,
        "skill_not_found": 404,
        "backend_unavailable": 503,
        "timeout": 504,
        "cancelled": 499,
        "user_abort": 499,
    }
    if code in mapping:
        return mapping[code]
    status = getattr(exc, "http_status", 0)
    return int(status) if status else 500


def _jsonable(value: Any) -> Any:
    """Convert a payload to JSON-safe primitives."""
    from ..output.format import to_plain

    return to_plain(value)


def _int(value: Any, default: int) -> int:
    """Parse an integer, falling back to a default."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _sse(event: str, payload: Mapping[str, Any]) -> bytes:
    """Format one server-sent event."""
    body = json.dumps(_jsonable(payload), ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {body}\n\n".encode("utf-8")


def _event_payload(event: Any) -> dict[str, Any]:
    """Serialise an agent event for the stream."""
    kind = str(getattr(getattr(event, "kind", ""), "value", getattr(event, "kind", "")))
    data: dict[str, Any] = {}
    for key, value in dict(getattr(event, "data", {}) or {}).items():
        if key in {"result", "response", "message", "call", "history"}:
            continue
        data[key] = _jsonable(value)
    return {"kind": kind, "ts": getattr(event, "ts", time.time()), "run_id": getattr(event, "run_id", ""),
            "data": data}


def _prometheus_from_stats(stats: ServerStats) -> str:
    """Minimal Prometheus output when no metrics registry is attached."""
    lines = [
        "# HELP magent_http_requests_total HTTP requests served",
        "# TYPE magent_http_requests_total counter",
        f"magent_http_requests_total {stats.requests}",
        "# TYPE magent_http_errors_total counter",
        f"magent_http_errors_total {stats.errors}",
        "# TYPE magent_runs_total counter",
        f"magent_runs_total {stats.runs}",
        "# TYPE magent_uptime_seconds gauge",
        f"magent_uptime_seconds {stats.uptime:.3f}",
    ]
    return "\n".join(lines) + "\n"


def build_server(config: Any = None, *, host: str = "", port: int = 0, **kwargs: Any) -> MagentServer:
    """Create a server, applying host/port overrides to its configuration.

    Args:
        config: loaded configuration
        host: bind address override
        port: bind port override
        **kwargs: passed to :class:`MagentServer`
    """
    if config is not None and (host or port):
        config.http.host = host or config.http.host
        config.http.port = port or config.http.port
    return MagentServer(config, **kwargs)


def free_port(host: str = "127.0.0.1") -> int:
    """Ask the operating system for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, *, timeout: float = 5.0) -> bool:
    """Poll ``/api/health`` until it answers or the timeout expires."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/api/health", timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.05)
    return False


def static_files() -> list[Path]:
    """List the bundled static assets."""
    return list(iter_files(STATIC_DIR, patterns=("*",), max_depth=2)) if STATIC_DIR.is_dir() else []


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "INDEX_FILE",
    "PUBLIC_ROUTES",
    "SSE_HEADERS",
    "STATIC_DIR",
    "MagentServer",
    "Request",
    "ServerStats",
    "build_server",
    "free_port",
    "static_files",
    "wait_for_health",
]
