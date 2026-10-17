"""The embedded MAgent HTTP server and its bundled web UI.

Start it from Python::

    from magent.server import MAgentServer, ServerContext

    server = MAgentServer.from_config(settings, ServerContext(settings, agent=agent))
    server.start()
    print(server.url)          # http://127.0.0.1:8700
    ...
    server.stop()

or from the command line with ``magent serve``.

The API mirrors the CLI: ``/api/run`` executes a task, ``/api/tools`` lists the
catalogue, ``/api/sessions`` and ``/api/memory`` expose the stores,
``/api/events`` streams agent progress as server-sent events, and ``/`` serves a
small single-file UI. Handlers delegate to the same objects the CLI uses, so
there is one behaviour to maintain.

Binding is loopback by default. Serving on another interface without an
``api_key`` is refused at construction, because this API can start runs that
execute tools.
"""

from __future__ import annotations

from typing import Any

from ..errors import ServerError_
from .app import (
    CONTENT_TYPES,
    DEFAULT_HOST,
    DEFAULT_PORT,
    HTTP_METHODS,
    LOOPBACK_HOSTS,
    MAX_HEADER_BYTES,
    STATIC_DIR,
    MAgentServer,
    Request,
    Response,
    Route,
    Router,
    ServerContext,
    describe_routes,
    free_port,
    register_routes,
    wait_ready,
)
from .routes import build_routes


def serve(config: Any = None, *, agent: Any = None, registry: Any = None,
          backend: Any = None, host: str = "", port: int = 0, api_key: str = "",
          block: bool = True, context: ServerContext | None = None) -> MAgentServer:
    """Build and start a server from configuration in one call.

    Args:
        config: resolved configuration
        agent: agent that runs tasks; built from ``config`` when omitted
        registry: tool registry; built from ``config`` when omitted
        backend: LLM backend; resolved from ``config`` when omitted
        host: override the configured bind address
        port: override the configured port
        api_key: override the configured key
        block: serve in the calling thread until interrupted
        context: pre-built context, overriding the individual arguments

    Returns:
        The started :class:`MAgentServer`.
    """
    active = context
    if active is None:
        active = build_context(config, agent=agent, registry=registry, backend=backend)
    server = MAgentServer.from_config(config, active)
    if host:
        server.host = host
    if port:
        server.port = int(port)
    if api_key:
        server.api_key = api_key
    if host or port or api_key:
        if not server._is_loopback() and not server.api_key:
            raise ServerError_(
                f"refusing to bind {server.host}:{server.port} without an api_key",
                details={"host": server.host, "port": server.port},
            )
    active.server = server
    return server.start(block=block)


def build_context(config: Any = None, *, agent: Any = None, registry: Any = None,
                  backend: Any = None, sessions: Any = None, memory: Any = None,
                  retriever: Any = None, guardrails: Any = None,
                  metrics: Any = None) -> ServerContext:
    """Assemble a :class:`ServerContext`, building missing pieces from config.

    Anything already supplied is used as-is; the rest is derived so a caller can
    wire up as much or as little as it wants.
    """
    active_agent = agent
    active_registry = registry
    active_backend = backend

    if active_registry is None:
        from ..tools import default_registry

        active_registry = default_registry()
    if active_backend is None:
        from ..llm.factory import BackendFactory, BackendRequest

        factory = BackendFactory()
        request = BackendRequest.from_config(config) if config is not None else \
            BackendRequest.from_env()
        active_backend = factory.build(request)
    if active_agent is None and config is not None:
        from ..agent import Agent

        active_agent = Agent(backend=active_backend, config=config.agent, settings=config,
                             registry=active_registry)

    active_sessions = sessions
    if active_sessions is None and config is not None and config.session.enabled:
        from ..session import SessionStore

        active_sessions = SessionStore.from_config(config)

    active_memory = memory
    if active_memory is None and config is not None and config.memory.enabled:
        from ..memory import MemoryStore

        try:
            active_memory = MemoryStore.from_config(config)
        except Exception:  # noqa: BLE001 - an unwritable store must not stop the server
            active_memory = None

    active_guardrails = guardrails
    if active_guardrails is None and config is not None:
        from ..guardrails import build_guardrails

        try:
            active_guardrails = build_guardrails(config)
        except Exception:  # noqa: BLE001 - guardrails must not stop the server
            active_guardrails = None

    active_metrics = metrics
    if active_metrics is None and config is not None and \
            getattr(config.observability, "metrics_enabled", True):
        from ..observability.metrics import standard_registry

        try:
            active_metrics = standard_registry()
        except Exception:  # noqa: BLE001
            active_metrics = None

    context = ServerContext(
        config, agent=active_agent, registry=active_registry, backend=active_backend,
        sessions=active_sessions, memory=active_memory, retriever=retriever,
        guardrails=active_guardrails, metrics=active_metrics,
        secrets=_redaction_list(config),
    )
    return context


def _redaction_list(config: Any = None) -> list[str]:
    """Return the literal secrets the server itself knows about.

    Tool output can echo an environment variable or a config file back to the
    caller, so the keys this process resolved are scrubbed from it. Only values
    the server already holds are listed; nothing is read from disk here.
    """
    if config is None:
        return []
    from ..secrets import is_secret_reference, resolve_secret

    found: list[str] = []
    for section_name in ("server", "http"):
        section = getattr(config, section_name, None)
        raw = str(getattr(section, "api_key", "") or "")
        if not raw:
            continue
        value = raw
        if is_secret_reference(raw):
            try:
                value = resolve_secret(raw)
            except Exception:  # noqa: BLE001 - an unresolvable ref cannot leak either
                continue
        if value and len(value) >= 4:
            found.append(value)
    # De-duplicate while keeping order, so the log of what is scrubbed is stable.
    return list(dict.fromkeys(found))


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
    "build_context",
    "build_routes",
    "describe_routes",
    "free_port",
    "register_routes",
    "serve",
    "wait_ready",
]
