"""HTTP route handlers for the MAgent server.

Every handler is a plain function taking ``(request, context)`` and returning a
:class:`~magent.server.app.Response`. Keeping them as functions rather than
methods means :func:`build_routes` is the whole API surface, and it can be
listed, tested, and documented without binding a socket.

The handlers delegate to the same objects the CLI uses - agent, tool registry,
memory store, session store, retriever, guardrails - so there is one behaviour
to maintain and the API cannot disagree with the command line.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Callable

from .._version import CORTEX_REPO, HOMEPAGE, RELEASE_NAME, __version__, version_string
from ..errors import MAgentError
from ..logging_ import get_logger
from ..types import ToolCall
from .app import Request, Response

LOGGER = get_logger("server.routes")
"""Logger for API handler activity."""

MAX_EVENT_SECONDS = 300.0
"""Ceiling on how long one SSE connection stays open."""

EVENT_POLL_SECONDS = 0.25
"""How often an idle SSE connection checks for shutdown."""

LOOP_MODES = ("react", "plan", "reflect", "single")
"""Loop modes :meth:`~magent.agent.Agent.run` accepts."""

_OPTION_FIELDS: frozenset[str] | None = None
"""Cached set of :class:`~magent.agent.RunOptions` field names."""


def run_option_names() -> frozenset[str]:
    """Return the option names ``Agent.run`` accepts.

    Derived from :class:`~magent.agent.RunOptions` rather than hardcoded, so the
    API tracks the agent instead of drifting away from it.
    """
    global _OPTION_FIELDS  # noqa: PLW0603 - a read-mostly cache
    if _OPTION_FIELDS is None:
        from ..agent import RunOptions

        _OPTION_FIELDS = frozenset(field.name for field in fields(RunOptions))
    return _OPTION_FIELDS


def build_routes() -> list[tuple[str, str, Callable[..., Any], dict[str, Any]]]:
    """Return the route table as ``(method, pattern, handler, kwargs)`` tuples.

    ``auth=False`` marks the two endpoints a load balancer must reach without
    credentials; ``body=True`` marks routes that read a JSON request body.
    """
    return [
        ("GET", "/health", health, {"auth": False, "name": "health"}),
        ("GET", "/ready", ready, {"auth": False, "name": "ready"}),
        ("GET", "/api/info", info, {"name": "api.info"}),
        ("GET", "/api/routes", routes, {"name": "api.routes"}),
        ("GET", "/api/config", config_view, {"name": "api.config"}),
        ("GET", "/api/stats", stats, {"name": "api.stats"}),
        ("GET", "/api/tools", tools_list, {"name": "api.tools.list"}),
        ("GET", "/api/tools/{name}", tools_show, {"name": "api.tools.show"}),
        ("GET", "/api/tools/{name}/schema", tools_schema, {"name": "api.tools.schema"}),
        ("POST", "/api/tools/{name}/call", tools_call, {"name": "api.tools.call", "body": True}),
        ("POST", "/api/run", run, {"name": "api.run", "body": True}),
        ("POST", "/api/run/async", run_async, {"name": "api.run.async", "body": True}),
        ("GET", "/api/runs", runs_list, {"name": "api.runs.list"}),
        ("GET", "/api/runs/{run_id}", runs_show, {"name": "api.runs.show"}),
        ("POST", "/api/runs/{run_id}/cancel", runs_cancel, {"name": "api.runs.cancel"}),
        ("GET", "/api/events", events, {"name": "api.events"}),
        ("GET", "/api/sessions", sessions_list, {"name": "api.sessions.list"}),
        ("GET", "/api/sessions/{session_id}", sessions_show, {"name": "api.sessions.show"}),
        ("DELETE", "/api/sessions/{session_id}", sessions_delete,
         {"name": "api.sessions.delete"}),
        ("GET", "/api/memory", memory_list, {"name": "api.memory.list"}),
        ("GET", "/api/memory/search", memory_search, {"name": "api.memory.search"}),
        ("GET", "/api/memory/stats", memory_stats, {"name": "api.memory.stats"}),
        ("POST", "/api/memory", memory_add, {"name": "api.memory.add", "body": True}),
        ("POST", "/api/rag/search", rag_search, {"name": "api.rag.search", "body": True}),
        ("POST", "/api/rag/ingest", rag_ingest, {"name": "api.rag.ingest", "body": True}),
        ("GET", "/api/rag/stats", rag_stats, {"name": "api.rag.stats"}),
        ("POST", "/api/guardrails/check", guardrails_check,
         {"name": "api.guardrails.check", "body": True}),
        ("GET", "/api/metrics", metrics, {"name": "api.metrics"}),
        ("GET", "/api/cortex/status", cortex_status, {"name": "api.cortex.status"}),
        ("GET", "/api/cortex/models", cortex_models, {"name": "api.cortex.models"}),
        ("GET", "/api/improvements", improvements, {"name": "api.improvements"}),
    ]


def _server(context: Any) -> Any:
    """Return the owning server, or ``None`` when a handler runs standalone."""
    return getattr(context, "server", None)


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary event data into JSON-safe structures."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:  # noqa: BLE001 - fall back to a readable string
            return str(value)
    return str(value)


# -- liveness ---------------------------------------------------------------- #

def health(request: Request, context: Any) -> Response:
    """Report that the process is alive and serving."""
    return Response.json({"status": "ok", "version": __version__,
                          "uptime_seconds": round(context.uptime_seconds(), 2)})


def ready(request: Request, context: Any) -> Response:
    """Report whether the server can actually do work.

    Liveness and readiness are separate questions: the process can be up while
    no backend is reachable, and a load balancer needs to tell them apart.
    """
    if context.agent is None:
        return Response.json({"status": "not_ready", "reason": "no_agent"}, 503)
    backend = context.backend
    name = str(getattr(backend, "name", "") or "")
    if backend is None or not name:
        return Response.json({"status": "not_ready", "reason": "no_backend"}, 503)
    registry = context.registry
    payload = {
        "status": "ready",
        "backend": name,
        "tools": registry.count() if registry is not None else 0,
        "offline": bool(name.startswith("echo")),
        "version": __version__,
    }
    try:
        payload["health"] = backend.health()
    except Exception:  # noqa: BLE001 - an unhealthy backend still answers readiness
        payload["health"] = {"ok": False}
    return Response.json(payload)


def info(request: Request, context: Any) -> Response:
    """Return server, agent, and build metadata."""
    server = _server(context)
    config = context.config
    payload: dict[str, Any] = {
        "name": "MAgent",
        "release_name": RELEASE_NAME,
        "version": __version__,
        "banner": version_string(),
        "homepage": HOMEPAGE,
        "cortex_repo": CORTEX_REPO,
        "tools": context.registry.count() if context.registry is not None else 0,
        "backend": str(getattr(context.backend, "name", "") or ""),
        "model": str(getattr(getattr(config, "model", None), "model", "") or ""),
        "profile": str(getattr(config, "profile", "") or ""),
        "uptime_seconds": round(context.uptime_seconds(), 3),
        "stats": context.stats(),
        "capabilities": {
            "agent": context.agent is not None,
            "sessions": context.sessions is not None,
            "memory": context.memory is not None,
            "retrieval": context.retriever is not None,
            "guardrails": context.guardrails is not None,
            "metrics": getattr(context, "metrics", None) is not None,
            "events": server is not None,
        },
    }
    if server is not None:
        payload["url"] = server.url
        payload["loopback"] = server._is_loopback()
        payload["auth_required"] = bool(server.api_key)
        payload["serve_ui"] = bool(server.serve_ui)
        payload["routes"] = len(server.router)
    return Response.json(payload)


def routes(request: Request, context: Any) -> Response:
    """List the registered API routes."""
    server = _server(context)
    if server is None:
        return Response.json({"routes": [], "count": 0})
    return Response.json({"routes": server.router.list_routes(),
                          "count": len(server.router)})


def config_view(request: Request, context: Any) -> Response:
    """Return the resolved configuration with secrets redacted.

    There is deliberately no unredacted mode: an endpoint that can dump
    credentials is a credential leak with an extra step.
    """
    config = context.config
    if config is None:
        return Response.json({})
    if request.bool_arg("reveal", False):
        return Response.error("configuration secrets are never exposed over HTTP", 403,
                              code="forbidden")
    section = request.arg("section", "")
    if section:
        try:
            value = config.section(section)
        except MAgentError as exc:
            return Response.error(str(exc), 404, code="unknown_section")
        except Exception as exc:  # noqa: BLE001 - an unknown table name is a client error
            return Response.error(str(exc), 404, code="unknown_section")
        body = value.to_dict(redact=True) if hasattr(value, "to_dict") else _jsonable(value)
        return Response.json({section: body})
    return Response.json(config.to_dict(redact=True))


def stats(request: Request, context: Any) -> Response:
    """Return aggregate server, agent, and tool statistics."""
    payload: dict[str, Any] = {"server": context.stats()}
    agent = context.agent
    if agent is not None:
        try:
            payload["agent"] = agent.stats()
        except Exception as exc:  # noqa: BLE001 - statistics must not fail a request
            payload["agent_error"] = str(exc)
    registry = context.registry
    if registry is not None:
        try:
            payload["tools"] = registry.stats()
        except Exception:  # noqa: BLE001
            payload["tools"] = {"count": registry.count()}
    memory = context.memory
    if memory is not None:
        try:
            payload["memory"] = memory.stats()
        except Exception:  # noqa: BLE001
            pass
    return Response.json(payload)


# -- tools ------------------------------------------------------------------- #

def tools_list(request: Request, context: Any) -> Response:
    """List tools, optionally filtered by category, tag, risk, or text."""
    registry = context.registry
    if registry is None:
        return Response.json({"tools": [], "count": 0, "total": 0, "categories": {}})
    specs = registry.search(category=request.arg("category", ""),
                            tag=request.arg("tag", ""),
                            max_risk=request.arg("max_risk", "") or None,
                            text=request.arg("q", ""))
    limit = request.int_arg("limit", 0)
    if limit > 0:
        specs = specs[:limit]
    return Response.json({
        "count": len(specs),
        "total": registry.count(),
        "categories": {name: len(items) for name, items in registry.by_category().items()},
        "tools": [spec.to_dict() for spec in specs],
    })


def _tool_or_error(context: Any, name: str) -> tuple[Any, Response | None]:
    """Resolve one tool by name, or produce the response to send instead."""
    registry = context.registry
    if registry is None:
        return None, Response.error("no tool registry is configured", 503, code="no_registry")
    tool = registry.get(name)
    if tool is None:
        return None, Response.error(
            f"unknown tool {name!r}", 404, code="tool_not_found",
            details={"suggestions": registry.suggest(name, limit=5)})
    return tool, None


def tools_show(request: Request, context: Any) -> Response:
    """Describe one tool, including its runtime statistics."""
    name = str(request.params.get("name", ""))
    tool, error = _tool_or_error(context, name)
    if error is not None:
        return error
    return Response.json(tool.to_dict())


def tools_schema(request: Request, context: Any) -> Response:
    """Return one tool's OpenAI function-calling schema."""
    name = str(request.params.get("name", ""))
    tool, error = _tool_or_error(context, name)
    if error is not None:
        return error
    return Response.json(tool.to_openai())


def tools_call(request: Request, context: Any) -> Response:
    """Call one tool directly, bypassing the agent.

    Useful for debugging a tool in isolation. It goes through the same
    validation, sandbox, and approval path a real run uses, so it cannot do
    anything an agent could not.
    """
    from ..tools import ToolContext

    registry = context.registry
    name = str(request.params.get("name", ""))
    tool, error = _tool_or_error(context, name)
    if error is not None:
        return error
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    raw_arguments = payload.get("arguments")
    arguments = raw_arguments if isinstance(raw_arguments, dict) else payload
    if tool.spec.requires_approval and not request.bool_arg("approve", False):
        return Response.error(
            f"tool {name!r} requires approval; pass ?approve=true to confirm",
            428, code="approval_required",
            details={"risk": tool.spec.risk, "destructive": tool.spec.destructive,
                     "network": tool.spec.network})
    call = ToolCall(id=str(payload.get("id") or f"api_{int(time.time() * 1000)}"),
                    name=name, arguments=dict(arguments or {}))
    workspace = str(payload.get("workspace") or "").strip()
    tool_context = ToolContext(workspace=Path(workspace).expanduser() if workspace else Path.cwd())
    tool_context.dry_run = request.bool_arg("dry_run", False) or bool(payload.get("dry_run"))
    tool_context.secrets = list(getattr(context, "secrets", []) or [])
    result = registry.dispatch(call, tool_context)
    return Response.json(result.to_dict(), 200 if result.ok else 400)


# -- runs -------------------------------------------------------------------- #

def _run_options(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract agent run options from a request body.

    Returns ``(options, rejected)``: only names :class:`~magent.agent.RunOptions`
    declares are forwarded, and anything else is reported back rather than
    silently dropped, because a misspelled ``max_iteration`` would otherwise run
    an unbounded loop.
    """
    allowed = run_option_names()
    options: dict[str, Any] = {}
    rejected: list[str] = []
    aliases = {"context": "context_block", "memory": "memory_block", "task": "task_block",
               "examples": "examples_block", "max_steps": "max_iterations",
               "steps": "max_iterations", "max_tool_calls": "max_iterations"}
    for key, value in payload.items():
        if value is None:
            continue
        target = aliases.get(key, key)
        if target in allowed:
            options[target] = value
        elif key in {"prompt", "task", "q", "text", "query", "timeout", "include_steps",
                     "include_messages", "run_id", "source"}:
            continue  # handled by the endpoint itself
        else:
            rejected.append(key)
    for key in ("tools", "disabled_tools"):
        value = options.get(key)
        if isinstance(value, str):
            options[key] = [item.strip() for item in value.split(",") if item.strip()]
    for key in ("max_iterations", "max_tokens"):
        if key in options:
            try:
                options[key] = int(options[key])
            except (TypeError, ValueError):
                rejected.append(key)
                options.pop(key, None)
    if "temperature" in options:
        try:
            options["temperature"] = float(options["temperature"])
        except (TypeError, ValueError):
            rejected.append("temperature")
            options.pop("temperature", None)
    mode = str(options.get("loop_mode") or "").strip().lower()
    if mode:
        if mode not in LOOP_MODES:
            rejected.append("loop_mode")
            options.pop("loop_mode", None)
        else:
            options["loop_mode"] = mode
    options.pop("stream", None)  # the sync endpoint cannot stream a response body
    return options, rejected


def _result_payload(result: Any, *, include_steps: bool = True,
                    include_messages: bool = False) -> dict[str, Any]:
    """Serialize a run result, trimming the heavy sections by default."""
    payload = _jsonable(result.to_dict())
    if not isinstance(payload, dict):
        return {"output": str(payload)}
    if not include_steps:
        payload.pop("steps", None)
    if not include_messages:
        payload.pop("messages", None)
    return payload


def _prompt_of(payload: dict[str, Any]) -> str:
    """Pull the task text out of a request body."""
    for key in ("prompt", "task", "q", "text", "query"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def run(request: Request, context: Any) -> Response:
    """Run one task synchronously and return the full result."""
    agent = context.agent
    if agent is None:
        return Response.error("no agent is configured", 503, code="no_agent")
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    prompt = _prompt_of(payload)
    if not prompt:
        return Response.error("a 'prompt' is required", 400, code="missing_prompt")
    options, rejected = _run_options(payload)
    if rejected:
        return Response.error(
            f"unknown run options: {', '.join(sorted(rejected))}", 400,
            code="unknown_option",
            details={"accepted": sorted(run_option_names()), "loop_modes": list(LOOP_MODES)})
    timeout = float(payload.get("timeout") or 0.0)
    holder: dict[str, Any] = {}
    failures: list[str] = []
    started = time.perf_counter()

    def execute() -> None:
        try:
            holder["result"] = agent.run(prompt, **options)
        except Exception as exc:  # noqa: BLE001 - reported as a 500, not a dead thread
            failures.append(f"{type(exc).__name__}: {exc}")

    worker = threading.Thread(target=execute, name="magent-api-run", daemon=True)
    worker.start()
    worker.join(timeout=timeout if timeout > 0 else None)
    if worker.is_alive():
        try:
            agent.cancel(reason=f"api timeout after {timeout}s")
        except Exception:  # noqa: BLE001 - cancellation is best effort
            pass
        return Response.error(f"the run exceeded {timeout}s", 504, code="timeout",
                              details={"elapsed": round(time.perf_counter() - started, 3)})
    if failures:
        return Response.error(failures[0], 500, code="run_failed")
    result = holder.get("result")
    if result is None:
        return Response.error("the agent produced no result", 500, code="no_result")
    body = _result_payload(
        result,
        include_steps=request.bool_arg("steps", True) and bool(payload.get("include_steps", True)),
        include_messages=bool(payload.get("include_messages")))
    body["server_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return Response.json(body)


def run_async(request: Request, context: Any) -> Response:
    """Start a run in the background and return its identifier immediately."""
    agent = context.agent
    if agent is None:
        return Response.error("no agent is configured", 503, code="no_agent")
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    prompt = _prompt_of(payload)
    if not prompt:
        return Response.error("a 'prompt' is required", 400, code="missing_prompt")
    options, rejected = _run_options(payload)
    if rejected:
        return Response.error(f"unknown run options: {', '.join(sorted(rejected))}", 400,
                              code="unknown_option")
    run_id = str(payload.get("run_id") or f"api_{int(time.time() * 1000)}")
    record: dict[str, Any] = {"run_id": run_id, "status": "running",
                              "prompt": prompt[:500], "started_at": time.time(),
                              "result": None, "error": "", "finished_at": None}
    with context.lock:
        context.active_runs[run_id] = record

    def execute() -> None:
        try:
            result = agent.run(prompt, **options)
            record["result"] = _result_payload(result)
            record["status"] = "completed" if getattr(result, "succeeded", False) else "failed"
            if result.error:
                record["error"] = str(result.error)
        except Exception as exc:  # noqa: BLE001 - recorded on the run, never raised
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "failed"
        finally:
            record["finished_at"] = time.time()

    threading.Thread(target=execute, name=f"magent-run-{run_id}", daemon=True).start()
    return Response.json({"run_id": run_id, "status": "running",
                          "poll": f"/api/runs/{run_id}",
                          "events": "/api/events"}, 202)


def runs_list(request: Request, context: Any) -> Response:
    """List background runs, newest first."""
    limit = request.int_arg("limit", 50)
    status_filter = request.arg("status", "")
    with context.lock:
        items = list(context.active_runs.values())
    if status_filter:
        items = [record for record in items if record.get("status") == status_filter]
    items.sort(key=lambda record: float(record.get("started_at") or 0.0), reverse=True)
    return Response.json({
        "count": len(items),
        "active": sum(1 for record in items if record.get("status") == "running"),
        "runs": [{"run_id": record["run_id"], "status": record["status"],
                  "started_at": record.get("started_at"),
                  "finished_at": record.get("finished_at"),
                  "prompt": record.get("prompt", ""),
                  "error": record.get("error", "")} for record in items[:limit]],
    })


def runs_show(request: Request, context: Any) -> Response:
    """Return one background run's state, including its result when finished."""
    run_id = str(request.params.get("run_id", ""))
    with context.lock:
        record = context.active_runs.get(run_id)
        known = list(context.active_runs)
    if record is None:
        return Response.error(f"no such run: {run_id}", 404, code="run_not_found",
                              details={"known": known[:20]})
    return Response.json(_jsonable(record))


def runs_cancel(request: Request, context: Any) -> Response:
    """Ask the agent to stop the run it is working on."""
    agent = context.agent
    if agent is None or not hasattr(agent, "cancel"):
        return Response.error("the agent cannot be cancelled", 400, code="not_cancellable")
    run_id = str(request.params.get("run_id", ""))
    with context.lock:
        record = context.active_runs.get(run_id)
    agent.cancel(reason=f"cancelled via API ({run_id or 'current'})")
    if record is not None:
        record["status"] = "cancelling"
    return Response.json({"run_id": run_id, "status": "cancelling",
                          "note": "cancellation takes effect at the next loop boundary"})


def events(request: Request, context: Any) -> Response:
    """Stream agent events as server-sent events.

    The connection self-closes after ``seconds`` (capped) so a dropped client
    cannot hold a thread forever.
    """
    server = _server(context)
    if server is None:
        return Response.error("event streaming is unavailable", 503, code="no_broadcaster")
    broadcaster = server.broadcaster()
    limit = request.int_arg("limit", 0)
    kinds = {item.strip() for item in request.arg("kinds", "").split(",") if item.strip()}
    seconds = request.float_arg("seconds", 60.0)
    timeout = min(MAX_EVENT_SECONDS, max(1.0, seconds))

    def stream(write: Callable[[str], None]) -> None:
        client_id, event_queue = broadcaster.connect()
        write(f"event: connected\ndata: {json.dumps({'client': client_id})}\n\n")
        deadline = time.monotonic() + timeout
        sent = 0
        try:
            while time.monotonic() < deadline:
                try:
                    event = event_queue.get(timeout=EVENT_POLL_SECONDS)
                except queue.Empty:
                    write(": keep-alive\n\n")
                    continue
                if event is None:
                    break
                kind = str(getattr(event.kind, "value", event.kind))
                if kinds and kind not in kinds:
                    continue
                body = _jsonable(event.to_dict() if hasattr(event, "to_dict") else event)
                write(f"event: {kind}\ndata: {json.dumps(body, default=str)}\n\n")
                sent += 1
                if limit and sent >= limit:
                    break
        finally:
            broadcaster.disconnect(client_id)
            write(f"event: closed\ndata: {json.dumps({'sent': sent})}\n\n")

    return Response(status=200, content_type="text/event-stream", stream=stream,
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- sessions ---------------------------------------------------------------- #

def sessions_list(request: Request, context: Any) -> Response:
    """List stored sessions."""
    store = context.sessions
    if store is None:
        return Response.json({"sessions": [], "count": 0, "enabled": False})
    infos = store.list(limit=request.int_arg("limit", 50), tag=request.arg("tag", ""),
                       query=request.arg("q", ""))
    return Response.json({"count": len(infos), "enabled": True,
                          "directory": str(getattr(store, "directory", "")),
                          "sessions": [info.to_dict() for info in infos]})


def sessions_show(request: Request, context: Any) -> Response:
    """Return one session's metadata and transcript."""
    store = context.sessions
    if store is None:
        return Response.error("sessions are disabled", 404, code="sessions_disabled")
    session_id = str(request.params.get("session_id", ""))
    try:
        session = store.open(session_id)
    except MAgentError as exc:
        return Response.error(str(exc), 404, code="session_not_found")
    except Exception as exc:  # noqa: BLE001 - a corrupt file is a client-visible 404
        return Response.error(str(exc), 404, code="session_not_found")
    limit = request.int_arg("limit", 0)
    entries = session.entries[-limit:] if limit > 0 else session.entries
    return Response.json({
        "id": session.id,
        "title": session.title,
        "model": session.model,
        "path": str(session.path),
        "stats": session.stats(),
        "entry_count": len(session.entries),
        "returned": len(entries),
        "entries": [_jsonable(entry.to_dict()) for entry in entries],
    })


def sessions_delete(request: Request, context: Any) -> Response:
    """Delete one session file."""
    store = context.sessions
    if store is None:
        return Response.error("sessions are disabled", 404, code="sessions_disabled")
    session_id = str(request.params.get("session_id", ""))
    try:
        removed = store.delete(session_id)
    except MAgentError as exc:
        return Response.error(str(exc), 404, code="session_not_found")
    if not removed:
        return Response.error(f"no such session: {session_id}", 404, code="session_not_found")
    return Response.json({"deleted": session_id})


# -- memory ------------------------------------------------------------------ #

def memory_list(request: Request, context: Any) -> Response:
    """List stored memories."""
    store = context.memory
    if store is None:
        return Response.json({"entries": [], "count": 0, "enabled": False})
    entries = store.list_entries(limit=request.int_arg("limit", 50),
                                 kind=request.arg("kind", ""),
                                 order=request.arg("order", "recent"),
                                 tag=request.arg("tag", ""))
    return Response.json({"count": len(entries), "total": len(store), "enabled": True,
                          "kinds": store.by_kind() if hasattr(store, "by_kind") else {},
                          "entries": [_jsonable(entry.to_dict()) for entry in entries]})


def memory_search(request: Request, context: Any) -> Response:
    """Search memories by text and optional kind."""
    store = context.memory
    if store is None:
        return Response.error("memory is disabled", 404, code="memory_disabled")
    query = request.arg("q", "") or request.arg("query", "")
    if not query:
        return Response.error("a 'q' query parameter is required", 400, code="missing_query")
    entries = store.search(query, limit=request.int_arg("limit", 8),
                           kind=request.arg("kind", ""),
                           min_score=request.float_arg("min_score", 0.0),
                           semantic=request.bool_arg("semantic", True))
    return Response.json({"query": query, "count": len(entries),
                          "results": [_jsonable(entry.to_dict()) for entry in entries]})


def memory_add(request: Request, context: Any) -> Response:
    """Store one memory entry."""
    store = context.memory
    if store is None:
        return Response.error("memory is disabled", 400, code="memory_disabled")
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    content = str(payload.get("content") or payload.get("text") or "").strip()
    if not content:
        return Response.error("'content' is required", 400, code="missing_content")
    tags = payload.get("tags") or []
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(",") if item.strip()]
    try:
        importance = float(payload.get("importance", 0.5))
    except (TypeError, ValueError):
        return Response.error("'importance' must be a number between 0 and 1", 400,
                              code="bad_importance")
    entry = store.add(content, kind=str(payload.get("kind") or "fact"), tags=list(tags),
                      importance=max(0.0, min(1.0, importance)),
                      source=str(payload.get("source") or "api"),
                      metadata=payload.get("metadata") if isinstance(
                          payload.get("metadata"), dict) else None)
    if hasattr(store, "save"):
        store.save()
    return Response.json(_jsonable(entry.to_dict()), 201)


def memory_stats(request: Request, context: Any) -> Response:
    """Return memory store statistics."""
    store = context.memory
    if store is None:
        return Response.json({"enabled": False})
    payload = store.stats()
    payload["entries"] = len(store)
    payload["enabled"] = True
    return Response.json(_jsonable(payload))


# -- retrieval --------------------------------------------------------------- #

def rag_search(request: Request, context: Any) -> Response:
    """Search the retrieval index."""
    retriever = context.retriever
    if retriever is None:
        return Response.error("retrieval is not configured", 503, code="no_retriever")
    if not len(retriever):
        return Response.error("the index is empty; POST /api/rag/ingest first", 404,
                              code="empty_index")
    payload = request.json({})
    if not isinstance(payload, dict):
        payload = {}
    query = str(payload.get("query") or payload.get("q") or request.arg("q", "")).strip()
    if not query:
        return Response.error("'query' is required", 400, code="missing_query")
    limit = int(payload.get("limit", request.int_arg("limit", 5)) or 5)
    result = retriever.retrieve(query, limit=limit,
                                mode=str(payload.get("mode") or ""),
                                min_score=float(payload.get("min_score", 0.0) or 0.0))
    return Response.json({
        "query": query,
        "mode": result.mode,
        "count": len(result.chunks),
        "seconds": round(result.seconds, 4),
        "tokens": result.tokens,
        "reranked": result.reranked,
        "context": result.context,
        "citations": _jsonable(result.citations.to_dict())
        if hasattr(result.citations, "to_dict") else [],
        "results": [{"source": chunk.source, "score": round(float(chunk.score), 4),
                     "heading": chunk.heading, "start_line": chunk.start_line,
                     "end_line": chunk.end_line, "strategy": chunk.strategy,
                     "text": chunk.text[:2000]} for chunk in result.chunks],
    })


def rag_ingest(request: Request, context: Any) -> Response:
    """Ingest paths and/or raw text into the retrieval index."""
    retriever = context.retriever
    if retriever is None:
        return Response.error("retrieval is not configured", 503, code="no_retriever")
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    paths = payload.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    text = str(payload.get("text") or "")
    if not paths and not text.strip():
        return Response.error("'paths' or 'text' is required", 400, code="missing_input")
    body: dict[str, Any] = {"before": len(retriever)}
    if text.strip():
        body["chunks_added"] = retriever.ingest_text(
            text, source=str(payload.get("source") or "api"))
    if paths:
        report = retriever.ingest([Path(str(item)).expanduser() for item in paths])
        body["report"] = _jsonable(report.to_dict()) if hasattr(report, "to_dict") \
            else str(report)
    body["after"] = len(retriever)
    return Response.json(body)


def rag_stats(request: Request, context: Any) -> Response:
    """Return retrieval index statistics."""
    retriever = context.retriever
    if retriever is None:
        return Response.json({"enabled": False})
    payload = retriever.stats()
    payload["chunks"] = len(retriever)
    payload["enabled"] = True
    return Response.json(_jsonable(payload))


# -- guardrails -------------------------------------------------------------- #

def guardrails_check(request: Request, context: Any) -> Response:
    """Run the guardrails over supplied text without executing anything."""
    guardrails = context.guardrails
    if guardrails is None:
        return Response.error("guardrails are not configured", 503, code="no_guardrails")
    payload = request.json({})
    if not isinstance(payload, dict):
        return Response.error("the body must be a JSON object", 400, code="bad_request")
    text = str(payload.get("text") or "")
    stage = str(payload.get("stage") or "prompt").strip().lower()
    session_id = str(payload.get("session_id") or "")
    verdict = guardrails.inspect(text, stage=stage, session_id=session_id,
                                 agent=str(payload.get("agent") or "api"))
    return Response.json({
        "stage": stage,
        "ok": verdict.ok,
        "blocked": verdict.blocked,
        "action": str(getattr(verdict.action, "value", verdict.action)),
        "rule": verdict.rule,
        "reason": verdict.reason,
        "severity": str(getattr(verdict.severity, "value", verdict.severity)),
        "score": round(float(verdict.score or 0.0), 4),
        "text": verdict.text,
        "changed": verdict.text != text,
        "findings": [_jsonable(finding.to_dict()) for finding in verdict.findings],
    })


# -- observability ----------------------------------------------------------- #

def metrics(request: Request, context: Any) -> Response:
    """Return metrics as JSON, or as Prometheus text with ``?format=prometheus``."""
    registry = getattr(context, "metrics", None)
    if registry is None:
        return Response.json({"enabled": False, "server": context.stats()})
    if request.arg("format", "").lower() in {"prom", "prometheus", "text"}:
        return Response.text(registry.render(prometheus=True))
    return Response.json({"enabled": True, "server": context.stats(),
                          "metrics": _jsonable(registry.snapshot())})


def cortex_status(request: Request, context: Any) -> Response:
    """Report the upstream Cortex LLMHoster's reachability."""
    from ..llm.cortex import cortex_health

    base_url = str(getattr(getattr(context.backend, "settings", None), "base_url", "") or "")
    if not base_url:
        config = context.config
        base_url = str(getattr(getattr(config, "server", None), "base_url", "") or "")
    if not base_url:
        return Response.json({"configured": False, "backend":
                              str(getattr(context.backend, "name", "") or "")})
    payload: dict[str, Any] = {"configured": True, "base_url": base_url,
                               "backend": str(getattr(context.backend, "name", "") or "")}
    health_report = cortex_health(base_url, timeout=float(request.float_arg("timeout", 3.0)))
    payload["health"] = health_report
    payload["reachable"] = bool(health_report) and health_report.get("ok") is not False
    backend = context.backend
    if backend is not None:
        try:
            payload["backend_health"] = backend.health()
        except Exception as exc:  # noqa: BLE001 - report, do not fail
            payload["backend_error"] = str(exc)
    return Response.json(payload)


def cortex_models(request: Request, context: Any) -> Response:
    """List the models the configured backend reports."""
    backend = context.backend
    if backend is None:
        return Response.error("no backend is configured", 503, code="no_backend")
    try:
        models = backend.models()
    except Exception as exc:  # noqa: BLE001 - an unreachable server is a normal outcome
        return Response.json({"models": [], "count": 0,
                              "backend": str(getattr(backend, "name", "") or ""),
                              "error": f"{type(exc).__name__}: {exc}"})
    return Response.json({
        "count": len(models),
        "backend": str(getattr(backend, "name", "") or ""),
        "models": [_jsonable(item.to_dict()) for item in models],
    })


def improvements(request: Request, context: Any) -> Response:
    """Return the improvement manifest summary, optionally verified."""
    from ..improvements import collect, repo_root, summarize, verify

    limit = request.int_arg("limit", 0)
    try:
        root = repo_root()
        entries = collect(root)
    except Exception as exc:  # noqa: BLE001 - an installed wheel has no source tree
        return Response.json({"available": False, "reason": f"{type(exc).__name__}: {exc}"})
    payload: dict[str, Any] = {"available": True, "root": str(root), **summarize(entries)}
    if request.bool_arg("verify", False):
        payload["verification"] = _jsonable(verify(entries, root).to_dict())
    if limit > 0:
        payload["entries"] = [entry.to_dict() for entry in entries[:limit]]
    return Response.json(payload)


__all__ = [
    "EVENT_POLL_SECONDS",
    "LOOP_MODES",
    "MAX_EVENT_SECONDS",
    "build_routes",
    "config_view",
    "cortex_models",
    "cortex_status",
    "events",
    "guardrails_check",
    "health",
    "improvements",
    "info",
    "memory_add",
    "memory_list",
    "memory_search",
    "memory_stats",
    "metrics",
    "rag_ingest",
    "rag_search",
    "rag_stats",
    "ready",
    "routes",
    "run",
    "run_async",
    "run_option_names",
    "runs_cancel",
    "runs_list",
    "runs_show",
    "sessions_delete",
    "sessions_list",
    "sessions_show",
    "stats",
    "tools_call",
    "tools_list",
    "tools_schema",
    "tools_show",
]
