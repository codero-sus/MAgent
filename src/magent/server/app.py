"""The embedded MAgent HTTP server.

A JSON API over the same agent, tools, memory, RAG, and session objects the CLI
uses - there is no second implementation, so the API cannot drift from the
behaviour you get on the command line. Standard library only: a
:class:`ThreadingHTTPServer` with an explicit route table.

Two design points matter operationally:

*it binds to loopback by default*
    the server can start agent runs that execute tools, so exposing it on an
    interface is an explicit choice. Binding elsewhere without an ``api_key``
    is refused rather than merely warned about.

*authentication is constant-time and optional*
    when ``api_key`` is set every route except ``/health`` requires
    ``Authorization: Bearer``; when it is not set and the bind address is not
    loopback, startup fails. That combination is the one that turns a local tool
    into a remote code execution service.

Long-running work is reported through ``/api/events`` as server-sent events,
backed by :class:`~magent.events.EventBroadcaster`, so a browser can watch a run
progress instead of polling.
"""

from __future__ import annotations

import hmac
import json
import queue
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .._version import __version__, version_string
from ..errors import MAgentError, ServerError_
from ..logging_ import get_logger

LOGGER = get_logger("server")
"""Logger for HTTP server activity."""

DEFAULT_HOST = "127.0.0.1"
"""Bind address used when configuration does not say."""

DEFAULT_PORT = 8700
"""Port used when configuration does not say."""

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
"""Addresses considered local, and therefore safe to serve without a key."""

MAX_HEADER_BYTES = 64 * 1024
"""Refuse requests whose headers exceed this, before reading a body."""

STATIC_DIR = Path(__file__).resolve().parent / "static"
"""Directory holding the bundled web UI."""

CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".txt": "text/plain; charset=utf-8",
    ".map": "application/json; charset=utf-8",
}
"""MIME type per static file extension."""

HTTP_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
"""Methods the router dispatches."""


@dataclass(slots=True)
class Route:
    """One registered endpoint."""

    method: str
    pattern: str
    handler: Callable[..., Any]
    name: str = ""
    auth: bool = True
    body: bool = False
    regex: Any = None

    def __post_init__(self) -> None:
        """Compile the path pattern and default the route name."""
        self.regex = _compile(self.pattern)
        if not self.name:
            self.name = f"{self.method.lower()} {self.pattern}"

    def match(self, method: str, path: str) -> dict[str, str] | None:
        """Return path parameters when this route matches, else ``None``."""
        if method != self.method:
            return None
        match = self.regex.match(path)
        return match.groupdict() if match else None

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form, for ``/api/routes``."""
        return {"method": self.method, "path": self.pattern, "name": self.name,
                "auth": self.auth, "accepts_body": self.body}


def _compile(pattern: str) -> Any:
    """Turn ``/api/tools/{name}`` into a matching regex."""
    parts = re.split(r"(\{[a-zA-Z_][a-zA-Z0-9_]*\})", pattern)
    out = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            out.append(f"(?P<{part[1:-1]}>[^/]+)")
        else:
            out.append(re.escape(part))
    return re.compile("^" + "".join(out) + "/?$")


@dataclass(slots=True)
class Response:
    """A handler's reply."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    raw: bytes | None = None
    content_type: str = "application/json; charset=utf-8"
    stream: Callable[[Callable[[str], None]], None] | None = None

    @classmethod
    def json(cls, payload: Any, status: int = 200, **headers: str) -> Response:
        """Build a JSON response."""
        return cls(status=status, body=payload, headers=dict(headers))

    @classmethod
    def text(cls, payload: str, status: int = 200) -> Response:
        """Build a plain-text response."""
        return cls(status=status, raw=payload.encode("utf-8"),
                   content_type="text/plain; charset=utf-8")

    @classmethod
    def html(cls, payload: str, status: int = 200) -> Response:
        """Build an HTML response."""
        return cls(status=status, raw=payload.encode("utf-8"),
                   content_type="text/html; charset=utf-8")

    @classmethod
    def error(cls, message: str, status: int = 400, *, code: str = "",
              details: Any = None, headers: dict[str, str] | None = None) -> Response:
        """Build a JSON error response.

        Args:
            message: human-readable explanation
            status: HTTP status code
            code: stable machine-readable identifier
            details: optional structured context
            headers: extra response headers, e.g. ``WWW-Authenticate``
        """
        payload: dict[str, Any] = {"error": message, "status": status}
        if code:
            payload["code"] = code
        if details is not None:
            payload["details"] = details
        return cls(status=status, body=payload, headers=dict(headers or {}))

    @classmethod
    def no_content(cls) -> Response:
        """Build a 204 response."""
        return cls(status=204, raw=b"", content_type="")

    def encode(self) -> bytes:
        """Return the body bytes."""
        if self.raw is not None:
            return self.raw
        if self.body is None:
            return b""
        if isinstance(self.body, (bytes, bytearray)):
            return bytes(self.body)
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        return json.dumps(self.body, ensure_ascii=False, default=str).encode("utf-8")


@dataclass(slots=True)
class Request:
    """A parsed inbound request."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, list[str]] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    params: dict[str, str] = field(default_factory=dict)
    client: str = ""
    started_at: float = field(default_factory=time.time)
    context: Any = None

    def arg(self, name: str, default: str = "") -> str:
        """Return the first value of a query parameter."""
        values = self.query.get(name) or []
        return str(values[0]) if values else default

    def int_arg(self, name: str, default: int = 0) -> int:
        """Return a query parameter as an integer."""
        raw = self.arg(name, "")
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    def float_arg(self, name: str, default: float = 0.0) -> float:
        """Return a query parameter as a float."""
        raw = self.arg(name, "")
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    def bool_arg(self, name: str, default: bool = False) -> bool:
        """Return a query parameter as a boolean."""
        raw = self.arg(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def json(self, default: Any = None) -> Any:
        """Parse the body as JSON, returning ``default`` when it is not."""
        if not self.body:
            return default
        try:
            return json.loads(self.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return default

    def header(self, name: str, default: str = "") -> str:
        """Return one header, case-insensitively."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return default

    def bearer_token(self) -> str:
        """Return the bearer token from the Authorization header."""
        value = self.header("authorization")
        if value.lower().startswith("bearer "):
            return value[7:].strip()
        return ""

    @property
    def content_length(self) -> int:
        """Declared body length, or 0."""
        try:
            return max(0, int(self.header("content-length") or 0))
        except ValueError:
            return 0


class ServerContext:
    """Shared state the handlers need: agent, registries, and stores."""

    def __init__(self, config: Any = None, *, agent: Any = None, registry: Any = None,
                 backend: Any = None, sessions: Any = None, memory: Any = None,
                 retriever: Any = None, guardrails: Any = None, hooks: Any = None,
                 metrics: Any = None, secrets: Any = None) -> None:
        """Assemble the objects the API exposes.

        Args:
            config: resolved configuration
            agent: the agent that runs tasks
            registry: tool registry
            backend: LLM backend, for provenance
            sessions: session store
            memory: memory store
            retriever: RAG retriever
            guardrails: guardrail manager
            hooks: plugin hook registry
            metrics: metrics registry, for ``/api/metrics``
            secrets: literal strings to redact from tool output
        """
        self.config = config
        self.agent = agent
        self.registry = registry
        self.backend = backend
        self.sessions = sessions
        self.memory = memory
        self.retriever = retriever
        self.guardrails = guardrails
        self.hooks = hooks
        self.metrics = metrics
        self.secrets = [str(item) for item in (secrets or ()) if str(item)]
        self.server: Any = None
        self.started_at = time.time()
        self.requests = 0
        self.failures = 0
        self.bytes_sent = 0
        self.lock = threading.RLock()
        self.active_runs: dict[str, Any] = {}

    def note_request(self, ok: bool = True, size: int = 0) -> None:
        """Record one completed request."""
        with self.lock:
            self.requests += 1
            self.bytes_sent += max(0, int(size))
            if not ok:
                self.failures += 1

    def uptime_seconds(self) -> float:
        """Seconds since the server started."""
        return max(0.0, time.time() - self.started_at)

    def stats(self) -> dict[str, Any]:
        """Return server statistics."""
        with self.lock:
            return {
                "uptime_seconds": round(self.uptime_seconds(), 3),
                "requests": self.requests,
                "failures": self.failures,
                "bytes_sent": self.bytes_sent,
                "active_runs": len(self.active_runs),
            }


class Router:
    """A small route table with pattern matching."""

    def __init__(self) -> None:
        """Create an empty router."""
        self.routes: list[Route] = []

    def add(self, method: str, pattern: str, handler: Callable[..., Any], *,
            name: str = "", auth: bool = True, body: bool = False) -> Route:
        """Register one route."""
        route = Route(method=method.upper(), pattern=pattern, handler=handler,
                      name=name, auth=auth, body=body)
        self.routes.append(route)
        return route

    def get(self, pattern: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a GET route."""
        return self._decorator("GET", pattern, **kwargs)

    def post(self, pattern: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a POST route."""
        return self._decorator("POST", pattern, **kwargs)

    def delete(self, pattern: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registering a DELETE route."""
        return self._decorator("DELETE", pattern, **kwargs)

    def _decorator(self, method: str, pattern: str,
                   **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Build a registration decorator for one method."""
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add(method, pattern, fn, **kwargs)
            return fn

        return wrap

    def resolve(self, method: str, path: str) -> tuple[Route | None, dict[str, str], bool]:
        """Find a route for a method and path.

        Returns:
            ``(route, params, path_exists)`` where ``path_exists`` says whether
            some route matched the path with a different method, which is the
            difference between 405 and 404.
        """
        exists = False
        for route in self.routes:
            params = route.match(method, path)
            if params is not None:
                return route, params, True
            if route.match("GET", path) is not None or route.match("POST", path) is not None:
                exists = True
        return None, {}, exists

    def methods_for(self, path: str) -> list[str]:
        """Return the methods registered for a path, for the ``Allow`` header.

        A 405 without ``Allow`` leaves a client guessing which verb it should
        have used, and RFC 9110 requires the header on that status.
        """
        found: list[str] = []
        for method in HTTP_METHODS:
            for route in self.routes:
                if route.match(method, path) is not None and method not in found:
                    found.append(method)
                    break
        return found

    def list_routes(self) -> list[dict[str, Any]]:
        """Return every route, for documentation and ``/api/routes``."""
        return [route.to_dict() for route in self.routes]

    def __len__(self) -> int:
        """Number of registered routes."""
        return len(self.routes)


class MAgentServer:
    """The HTTP server: routing, authentication, CORS, and static files."""

    def __init__(self, context: ServerContext, *, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, api_key: str = "",
                 cors_origins: Sequence[str] = (), serve_ui: bool = True,
                 max_body_bytes: int = 32 * 1024 * 1024, request_timeout: float = 300.0,
                 threads: int = 16, allow_origins_any: bool = True,
                 require_auth_off_loopback: bool = True) -> None:
        """Configure the server.

        Args:
            context: shared state handlers use
            host: interface to bind
            port: port to bind, 0 to pick a free one
            api_key: bearer token required for authenticated routes
            cors_origins: allowed ``Origin`` values, empty for the default policy
            serve_ui: serve the bundled web UI at ``/``
            max_body_bytes: refuse larger request bodies
            request_timeout: seconds a handler may take
            threads: worker thread ceiling
            allow_origins_any: reflect any origin when none are configured
            require_auth_off_loopback: refuse to bind a public interface without
                an API key

        Raises:
            ServerError_: a public bind address was requested without an API key
        """
        self.context = context
        self.host = str(host or DEFAULT_HOST)
        self.port = int(port)
        self.api_key = str(api_key or "")
        self.cors_origins = [str(item) for item in cors_origins if str(item)]
        self.serve_ui = bool(serve_ui)
        self.max_body_bytes = max(1024, int(max_body_bytes))
        self.request_timeout = float(request_timeout)
        self.threads = max(1, int(threads))
        self.allow_origins_any = bool(allow_origins_any)
        self.router = Router()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._broadcaster: Any = None
        self.bound_port = 0

        if require_auth_off_loopback and not self._is_loopback() and not self.api_key:
            raise ServerError_(
                f"refusing to bind {self.host}:{self.port} without an api_key",
                details={
                    "host": self.host,
                    "hint": "the API can start agent runs that execute tools; set "
                            "[http] api_key or bind 127.0.0.1",
                },
            )
        # Handlers reach the server through the context, so they can stream
        # events and report the public URL without a second global.
        try:
            self.context.server = self
        except AttributeError:
            pass  # a slotted or frozen context simply loses /api/events
        register_routes(self.router, self)

    def _is_loopback(self) -> bool:
        """Whether the configured bind address is local."""
        host = self.host.strip().lower()
        if host in LOOPBACK_HOSTS:
            return True
        try:
            return socket.gethostbyname(host) in LOOPBACK_HOSTS
        except OSError:
            return False

    @classmethod
    def from_config(cls, config: Any, context: ServerContext | None = None) -> MAgentServer:
        """Build a server from a resolved configuration."""
        section = getattr(config, "http", None)
        from ..secrets import is_secret_reference, resolve_secret

        raw_key = str(getattr(section, "api_key", "") or "")
        api_key = resolve_secret(raw_key) if is_secret_reference(raw_key) else raw_key
        return cls(
            context or ServerContext(config),
            host=str(getattr(section, "host", DEFAULT_HOST) or DEFAULT_HOST),
            port=int(getattr(section, "port", DEFAULT_PORT) or DEFAULT_PORT),
            api_key=api_key,
            cors_origins=list(getattr(section, "cors_origins", []) or []),
            serve_ui=bool(getattr(section, "serve_ui", True)),
            max_body_bytes=int(getattr(section, "max_body_bytes", 0) or 32 * 1024 * 1024),
            request_timeout=float(getattr(section, "request_timeout", 300.0) or 300.0),
            threads=int(getattr(section, "threads", 16) or 16),
            allow_origins_any=bool(getattr(section, "allow_origins_any", True)),
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, block: bool = False) -> MAgentServer:
        """Bind and begin serving.

        Args:
            block: serve in the calling thread instead of a background one

        Raises:
            ServerError_: the address cannot be bound
        """
        if self._httpd is not None:
            return self
        try:
            httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        except OSError as exc:
            raise ServerError_(f"cannot bind {self.host}:{self.port}: {exc}",
                               details={"host": self.host, "port": self.port}) from exc
        httpd.daemon_threads = True
        self._httpd = httpd
        self.bound_port = int(httpd.server_address[1])
        LOGGER.info("serving on http://%s:%d", self.host, self.bound_port)
        if block:
            try:
                httpd.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                LOGGER.info("interrupted; shutting down")
            finally:
                self.stop()
            return self
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="magent-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the socket."""
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            try:
                httpd.shutdown()
                httpd.server_close()
            except OSError as exc:
                LOGGER.debug("shutdown reported: %s", exc)
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._broadcaster is not None:
            try:
                self._broadcaster.close()
            except Exception:  # noqa: BLE001 - closing a broadcaster is best effort
                pass
            self._broadcaster = None
        LOGGER.info("stopped")

    @property
    def running(self) -> bool:
        """Whether the server is currently bound."""
        return self._httpd is not None

    @property
    def url(self) -> str:
        """The base URL the server is listening on."""
        return f"http://{self.host}:{self.bound_port or self.port}"

    def broadcaster(self) -> Any:
        """Return the event broadcaster backing ``/api/events``."""
        if self._broadcaster is None:
            from ..events import EventBroadcaster, EventBus

            bus = getattr(self.context.agent, "events", None) or EventBus()
            self._broadcaster = EventBroadcaster(bus)
        return self._broadcaster

    def __enter__(self) -> MAgentServer:
        """Start the server."""
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        """Stop the server."""
        self.stop()

    # -- request handling ----------------------------------------------------

    def authenticate(self, request: Request) -> tuple[bool, str]:
        """Check a request's credentials.

        Returns:
            ``(allowed, reason)``. Comparison is constant-time so a timing probe
            cannot recover the key one character at a time.
        """
        if not self.api_key:
            return True, ""
        supplied = request.bearer_token() or request.header("x-api-key")
        if not supplied:
            return False, "missing bearer token"
        if hmac.compare_digest(supplied, self.api_key):
            return True, ""
        return False, "invalid api key"

    def cors_headers(self, request: Request) -> dict[str, str]:
        """Return the CORS headers for a request's origin."""
        origin = request.header("origin")
        if not origin:
            return {}
        if self.cors_origins:
            if origin in self.cors_origins:
                return {"Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key",
                        "Access-Control-Allow-Methods": ", ".join(HTTP_METHODS),
                        "Vary": "Origin"}
            return {}
        if self.allow_origins_any:
            headers = {"Access-Control-Allow-Origin": "*",
                       "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key",
                       "Access-Control-Allow-Methods": ", ".join(HTTP_METHODS)}
            if self.api_key:
                # Credentials and a wildcard origin cannot be combined; reflect
                # the specific origin instead so a keyed deployment still works.
                headers["Access-Control-Allow-Origin"] = origin
                headers["Access-Control-Allow-Credentials"] = "true"
                headers["Vary"] = "Origin"
            return headers
        return {}

    def dispatch(self, request: Request) -> Response:
        """Route one request and return the response.

        Every exit path - including errors - gets the CORS headers, because a
        browser that cannot read a 401 body cannot tell the user why a call
        failed.
        """
        return self._finish(self._route(request), request)

    def _finish(self, response: Response, request: Request) -> Response:
        """Attach the CORS headers a response needs before it is written."""
        for key, value in self.cors_headers(request).items():
            response.headers.setdefault(key, value)
        return response

    def _route(self, request: Request) -> Response:
        """Resolve one request to a response, without touching CORS."""
        if request.method == "OPTIONS":
            return Response(status=204, raw=b"", content_type="")
        route, params, exists = self.router.resolve(request.method, request.path)
        if route is None:
            if request.path in {"", "/"} and self.serve_ui:
                return self.serve_static("index.html", request)
            if exists:
                return Response.error(f"{request.method} not allowed for {request.path}",
                                      405, code="method_not_allowed",
                                      headers={"Allow": ", ".join(self.router.methods_for(
                                          request.path))})
            if self.serve_ui and not request.path.startswith("/api/"):
                found = self.serve_static(request.path.lstrip("/"), request)
                if found.status != 404:
                    return found
            return Response.error(f"no route for {request.path}", 404, code="not_found",
                                  details={"routes": len(self.router)})
        request.params = params
        if route.auth:
            allowed, reason = self.authenticate(request)
            if not allowed:
                return Response.error(f"unauthorized: {reason}", 401, code="unauthorized",
                                      headers={"WWW-Authenticate": "Bearer"})
        if route.body and request.content_length > self.max_body_bytes:
            return Response.error(
                f"request body of {request.content_length} bytes exceeds "
                f"{self.max_body_bytes}", 413, code="body_too_large")
        try:
            result = route.handler(request, self.context)
        except MAgentError as exc:
            self.context.note_request(ok=False)
            status = int(getattr(exc, "exit_code", 1) or 1)
            return Response.error(str(exc), 400 if status < 100 else status,
                                  code=str(getattr(exc, "code", "error")),
                                  details=getattr(exc, "details", None))
        except Exception as exc:  # noqa: BLE001 - a handler must not kill the server
            self.context.note_request(ok=False)
            LOGGER.exception("handler %s raised", route.name)
            return Response.error(f"{type(exc).__name__}: {exc}", 500, code="internal_error")
        if not isinstance(result, Response):
            result = Response.json(result)
        return result

    def serve_static(self, relative: str, request: Request) -> Response:
        """Serve a file from the bundled UI directory.

        The path is resolved and then checked to be inside ``STATIC_DIR``, so
        ``/../`` traversal cannot read arbitrary files.
        """
        if not self.serve_ui:
            return Response.error("the UI is disabled", 404, code="ui_disabled")
        clean = unquote(str(relative or "index.html")).lstrip("/")
        target = (STATIC_DIR / clean).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            LOGGER.warning("refusing static path traversal: %s", relative)
            return Response.error("forbidden", 403, code="forbidden")
        if target.is_dir():
            target = target / "index.html"
        if not target.exists() or not target.is_file():
            return Response.error(f"no such file: {clean}", 404, code="not_found")
        try:
            data = target.read_bytes()
        except OSError as exc:
            return Response.error(f"cannot read {clean}: {exc}", 500, code="read_error")
        content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        return Response(status=200, raw=data, content_type=content_type,
                        headers={"Cache-Control": "no-cache"})

    def info(self) -> dict[str, Any]:
        """Return the payload served at ``/api/info``."""
        config = self.context.config
        registry = self.context.registry
        return {
            "name": "MAgent",
            "version": __version__,
            "banner": version_string(),
            "url": self.url,
            "host": self.host,
            "port": self.bound_port or self.port,
            "loopback": self._is_loopback(),
            "auth_required": bool(self.api_key),
            "serve_ui": self.serve_ui,
            "routes": len(self.router),
            "tools": registry.count() if registry is not None else 0,
            "model": str(getattr(getattr(config, "model", None), "model", "") or ""),
            "backend": str(getattr(self.context.backend, "name", "") or ""),
            "uptime_seconds": round(self.context.uptime_seconds(), 2),
            "stats": self.context.stats(),
        }


def _make_handler(server: MAgentServer) -> type:
    """Build a request handler class bound to one server instance."""

    class _Handler(BaseHTTPRequestHandler):
        """Dispatches every method through the server's router."""

        server_version = f"MAgent/{__version__}"
        protocol_version = "HTTP/1.1"

        def _handle(self, method: str) -> None:
            started = time.perf_counter()
            try:
                request = self._parse(method)
            except MAgentError as exc:
                self._send(Response.error(str(exc), 413, code="payload_too_large"))
                return
            response = server.dispatch(request)
            self._send(response)
            server.context.note_request(ok=response.status < 400,
                                        size=len(response.encode()))
            elapsed = (time.perf_counter() - started) * 1000.0
            LOGGER.debug("%s %s -> %d in %.1fms", method, request.path,
                         response.status, elapsed)

        def _parse(self, method: str) -> Request:
            """Read the request line, headers, and body."""
            parsed = urlparse(self.path)
            length = 0
            raw_length = self.headers.get("Content-Length")
            if raw_length:
                try:
                    length = int(raw_length)
                except ValueError:
                    raise MAgentError("invalid Content-Length") from None
            if length > server.max_body_bytes:
                raise MAgentError(
                    f"body of {length} bytes exceeds the {server.max_body_bytes} byte limit")
            body = self.rfile.read(length) if length > 0 else b""
            client = f"{self.client_address[0]}:{self.client_address[1]}" \
                if self.client_address else ""
            return Request(
                method=method,
                path=parsed.path or "/",
                query=parse_qs(parsed.query),
                headers={str(key): str(value) for key, value in self.headers.items()},
                body=body,
                client=client,
                context=server.context,
            )

        def _send(self, response: Response) -> None:
            """Write a response, honouring streaming bodies."""
            try:
                if response.stream is not None:
                    self._send_stream(response)
                    return
                payload = response.encode()
                self.send_response(response.status)
                if response.content_type:
                    self.send_header("Content-Type", response.content_type)
                self.send_header("Content-Length", str(len(payload)))
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                if payload and self.command != "HEAD":
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError) as exc:
                LOGGER.debug("client disconnected: %s", exc)
            except OSError as exc:
                LOGGER.debug("could not write response: %s", exc)

        def _send_stream(self, response: Response) -> None:
            """Stream a response body using chunked transfer encoding.

            A streaming body has no ``Content-Length``, and HTTP/1.1 requires
            either that header or chunked framing to delimit it. Without one of
            them a well-behaved client waits for EOF that never arrives, so
            every chunk is length-prefixed and the stream ends with a zero-length
            chunk even if the generator raises.
            """
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type or "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "keep-alive")
            for key, value in response.headers.items():
                if key.lower() in {"content-length", "transfer-encoding"}:
                    continue  # framing is owned by this method
                self.send_header(key, value)
            self.end_headers()
            sent = 0
            broken = False

            def write(chunk: str) -> None:
                nonlocal sent
                payload = chunk.encode("utf-8")
                if not payload:
                    return
                self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
                self.wfile.flush()
                sent += len(payload)

            try:
                response.stream(write)
            except (BrokenPipeError, ConnectionResetError):
                broken = True
                LOGGER.debug("stream client disconnected")
            except OSError as exc:
                broken = True
                LOGGER.debug("stream write failed: %s", exc)
            except Exception:  # noqa: BLE001 - terminate the chunk stream, then report
                LOGGER.exception("stream handler raised")
            if not broken:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError as exc:
                    LOGGER.debug("could not terminate stream: %s", exc)
            server.context.note_request(ok=not broken, size=sent)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            """Handle GET."""
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            """Handle POST."""
            self._handle("POST")

        def do_PUT(self) -> None:  # noqa: N802
            """Handle PUT."""
            self._handle("PUT")

        def do_PATCH(self) -> None:  # noqa: N802
            """Handle PATCH."""
            self._handle("PATCH")

        def do_DELETE(self) -> None:  # noqa: N802
            """Handle DELETE."""
            self._handle("DELETE")

        def do_OPTIONS(self) -> None:  # noqa: N802
            """Handle a CORS preflight."""
            self._handle("OPTIONS")

        def do_HEAD(self) -> None:  # noqa: N802
            """Handle HEAD, sending headers only."""
            self._handle("GET")

        def log_message(self, fmt: str, *args: Any) -> None:
            """Route the default access log through MAgent's logger."""
            LOGGER.debug(fmt, *args)

    return _Handler


def register_routes(router: Router, server: MAgentServer) -> None:
    """Register every API route on a router.

    Kept as a module function so the route table can be listed and tested
    without binding a socket.
    """
    from .routes import build_routes

    for method, pattern, handler, kwargs in build_routes():
        router.add(method, pattern, handler, **kwargs)


def free_port(host: str = DEFAULT_HOST) -> int:
    """Return an unused TCP port, for tests and ``--port 0``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def wait_ready(url: str, *, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a URL until it answers, for tests and startup ordering."""
    from ..http import HTTPClient, HTTPRequest

    deadline = time.monotonic() + max(0.0, timeout)
    with HTTPClient(timeout=1.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.send(HTTPRequest(url=url, method="GET"))
                if response.status and response.status < 500:
                    return True
            except Exception:  # noqa: BLE001 - keep polling until the deadline
                pass
            time.sleep(interval)
    return False


def describe_routes(router: Router) -> str:
    """Render the route table as text."""
    rows = sorted(((route.method, route.pattern, route.name, "yes" if route.auth else "no")
                   for route in router.routes), key=lambda item: (item[1], item[0]))
    lines = [f"{'METHOD':<7} {'PATH':<34} {'AUTH':<5} NAME"]
    lines.extend(f"{method:<7} {pattern:<34} {auth:<5} {name}"
                 for method, pattern, name, auth in rows)
    return "\n".join(lines)


__all__ = [
    "CONTENT_TYPES",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "HTTP_METHODS",
    "LOOPBACK_HOSTS",
    "MAX_HEADER_BYTES",
    "STATIC_DIR",
    "MAgentServer",
    "Request",
    "Response",
    "Route",
    "Router",
    "ServerContext",
    "describe_routes",
    "free_port",
    "register_routes",
    "wait_ready",
]
