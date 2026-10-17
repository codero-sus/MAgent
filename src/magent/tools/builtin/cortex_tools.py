"""Tools for inspecting and controlling a running Cortex LLMHoster instance.

These wrap :class:`magent.llm.cortex_admin.CortexAdmin` so an agent can check
whether its own inference host is healthy, list the models it serves, read
server metrics, start or stop a runtime, and report the hardware underneath.
Reads are ``risk="read"``; anything that mutates server state is
``risk="write"`` and marked ``requires_approval`` so the approval layer gates
it even in a permissive configuration.

Cortex LLMHoster is a separate project by the same author
(``codero-sus/Cortex_LLMHoster``). MAgent talks to it over its documented HTTP
API and never imports, patches, or redistributes its code.
"""

from __future__ import annotations

import json
from typing import Any

from ...errors import ToolExecutionError
from ...llm.cortex import CortexBackend, CortexSettings
from ...llm.cortex_admin import CortexAdmin
from ...tools.base import ToolContext, tool
from ...utils.timeutil import format_duration


def _backend(context: ToolContext) -> CortexBackend:
    """Return the run's Cortex backend, building one from settings if needed.

    Raises :class:`ToolExecutionError` when the configured backend is not
    Cortex, because these tools are meaningless against the echo backend.
    """
    backend = context.backend
    if isinstance(backend, CortexBackend):
        return backend
    server = getattr(context.config, "server", None)
    if server is not None:
        settings = CortexSettings(
            base_url=server.base_url,
            api_key=server.api_key,
            api_prefix=server.api_prefix,
            timeout=server.timeout,
            connect_timeout=server.connect_timeout,
            max_retries=server.max_retries,
            verify_ssl=server.verify_ssl,
            max_upload_bytes=server.max_upload_bytes,
            organization=server.organization,
            default_model=getattr(getattr(context.config, "model", None), "model", "") or "",
        )
        return CortexBackend(settings=settings)
    if backend is not None:
        raise ToolExecutionError(
            f"The active backend is {getattr(backend, 'name', type(backend).__name__)!r}, "
            "not Cortex; these tools only apply to a Cortex LLMHoster instance"
        )
    raise ToolExecutionError("No Cortex server is configured")


def _admin(context: ToolContext) -> CortexAdmin:
    """Return a cached :class:`CortexAdmin` bound to the run's backend."""
    cached = context.extras.get("__cortex_admin__")
    if isinstance(cached, CortexAdmin):
        return cached
    admin = CortexAdmin(backend=_backend(context))
    context.extras["__cortex_admin__"] = admin
    return admin


@tool(category="cortex", risk="read", network=True, tags=["cortex", "health", "core"])
def cortex_health(include_models: bool = True, context: ToolContext | None = None) -> dict[str, Any]:
    """Check whether the Cortex LLMHoster instance is reachable and ready.

    Probes ``/health`` and ``/ready`` without invoking a model, so it is cheap
    enough to run before every task.

    Args:
        include_models: also list the served model identifiers
    """
    ctx = context or ToolContext()
    backend = _backend(ctx)
    probe = backend.health()
    readiness = probe.get("readiness") or {}
    body = readiness.get("body") or {}
    model_ids: list[str] = []
    if include_models and probe.get("ok"):
        try:
            model_ids = backend.model_ids()
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            probe["models_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "base_url": probe.get("base_url", ""),
        "backend": probe.get("backend", ""),
        "reachable": bool(probe.get("ok")),
        "liveness_status": (probe.get("liveness") or {}).get("status"),
        "readiness_status": readiness.get("status"),
        "ready": body.get("status") == "ready",
        "reason": body.get("reason", ""),
        "models": len(model_ids),
        "model_ids": model_ids[:50],
        "probe": probe,
    }


@tool(category="cortex", risk="read", network=True, tags=["cortex", "models"])
def cortex_models(capability: str = "", refresh: bool = False,
                  context: ToolContext | None = None) -> dict[str, Any]:
    """List the models a Cortex LLMHoster instance serves.

    Args:
        capability: only return models declaring this capability, e.g. "vision"
        refresh: bypass the backend's short lived model cache
    """
    ctx = context or ToolContext()
    backend = _backend(ctx)
    models = backend.models(refresh=refresh)
    entries: list[dict[str, Any]] = []
    for info in models:
        record = info.to_dict() if hasattr(info, "to_dict") else dict(info)
        capabilities = [str(item) for item in record.get("capabilities", [])]
        if capability and capability.lower() not in [item.lower() for item in capabilities]:
            continue
        entries.append({
            "id": record.get("id", ""),
            "owned_by": record.get("owned_by", ""),
            "runtime": record.get("runtime", ""),
            "capabilities": capabilities,
            "default": bool(record.get("default", False)),
            "context_length": record.get("context_length"),
            "embedding": bool(record.get("embedding", False)),
        })
    return {
        "base_url": backend.settings.base_url,
        "count": len(entries),
        "capability_filter": capability or None,
        "models": entries,
        "default_model": backend.default_model_id(),
        "ids": [entry["id"] for entry in entries],
    }


@tool(category="cortex", risk="read", network=True, tags=["cortex", "metrics"])
def cortex_metrics(reset: bool = False, context: ToolContext | None = None) -> dict[str, Any]:
    """Read Cortex request metrics: throughput, latency, and per-model usage.

    Args:
        reset: reset the counters after reading them
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    snapshot = admin.metrics()
    payload: dict[str, Any] = {
        "uptime_seconds": snapshot.uptime_seconds,
        "uptime_human": format_duration(snapshot.uptime_seconds),
        "in_flight": snapshot.in_flight,
        "total_requests": snapshot.total_requests,
        "completed_requests": snapshot.completed_requests,
        "failed_requests": snapshot.failed_requests,
        "average_latency_ms": snapshot.average_latency_ms,
        "error_rate": snapshot.error_rate,
        "status_codes": snapshot.status_codes,
        "requests_by_model": snapshot.requests_by_model,
    }
    if reset:
        payload["reset_response"] = admin.reset_metrics()
    return payload


@tool(category="cortex", risk="read", network=True, tags=["cortex", "state"])
def cortex_state(include_models: bool = True, context: ToolContext | None = None) -> dict[str, Any]:
    """Return the Cortex admin state: settings, API paths, runtime, and metrics.

    This is the richest single view of a Cortex instance and is what the
    ``magent cortex state`` CLI command prints.

    Args:
        include_models: include the full per-model configuration list
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    state = admin.state()
    payload = state.to_dict()
    if not include_models:
        payload.pop("models", None)
    payload["running_models"] = state.running_models
    return payload


@tool(category="cortex", risk="read", network=True, tags=["cortex", "hardware"])
def cortex_hardware(context: ToolContext | None = None) -> dict[str, Any]:
    """Report the hardware Cortex detected and the state of each runtime.

    Args:
        (none)
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    state = admin.state()
    runtimes = state.runtime_models
    return {
        "hardware": state.hardware,
        "runtime": state.runtime,
        "version": state.version,
        "api": state.api,
        "auth_enabled": state.auth_enabled,
        "env_overrides": state.env_overrides,
        "model_states": runtimes,
        "running": [name for name, value in runtimes.items()
                    if isinstance(value, dict) and value.get("status") == "running"],
        "stopped": [name for name, value in runtimes.items()
                    if isinstance(value, dict) and value.get("status") == "stopped"],
    }


@tool(category="cortex", risk="write", network=True, requires_approval=True,
      timeout=300.0, tags=["cortex", "runtime", "control"])
def cortex_model_control(model_id: str, action: str = "start", wait: bool = True,
                           timeout: float = 120.0, context: ToolContext | None = None) -> dict[str, Any]:
    """Start, stop, or restart a Cortex model runtime.

    Args:
        model_id: the model identifier reported by cortex_models
        action: start, stop, restart, or stop-all
        wait: poll until the runtime reaches the expected state
        timeout: seconds to wait for the state change
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    action = str(action).lower().strip()
    actions = {"start", "stop", "restart", "stop-all", "status"}
    if action not in actions:
        raise ToolExecutionError(f"action must be one of {sorted(actions)}")
    if action == "status":
        return {"model_id": model_id, "action": action,
                "status": admin.model_status(model_id)}
    if ctx.dry_run:
        return {"model_id": model_id, "action": action, "dry_run": True,
                "note": "no request would be sent"}
    if action == "stop-all":
        return {"model_id": None, "action": action, "response": admin.stop_all_models()}
    if action == "start":
        response = admin.start_model(model_id, wait=wait, timeout=timeout)
    elif action == "stop":
        response = admin.stop_model(model_id)
    else:
        response = admin.restart_model(model_id, wait=wait)
    return {
        "model_id": model_id,
        "action": action,
        "response": response,
        "status": admin.model_status(model_id),
    }


@tool(category="cortex", risk="read", network=True, tags=["cortex", "models", "files"])
def cortex_model_files(directory: str = "", context: ToolContext | None = None) -> dict[str, Any]:
    """List the model files Cortex has downloaded, with their sizes.

    Args:
        directory: restrict the listing to this models subdirectory
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    files = admin.model_files(directory=directory or None)
    total = sum(int(item.get("size_bytes") or item.get("size") or 0) for item in files if isinstance(item, dict))
    from ...utils.files import human_size

    return {
        "count": len(files),
        "total_bytes": total,
        "total_human": human_size(total),
        "files": files[:200],
    }


@tool(category="cortex", risk="read", network=True, tags=["cortex", "update"])
def cortex_update_check(context: ToolContext | None = None) -> dict[str, Any]:
    """Ask Cortex whether an update is available.

    Args:
        (none)
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    return admin.check_update()


@tool(category="cortex", risk="read", network=True, tags=["cortex", "auth"])
def cortex_auth_options(context: ToolContext | None = None) -> dict[str, Any]:
    """Report how the Cortex admin surface is authenticated.

    Useful for diagnosing 401 responses without exposing any credential.
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    backend = _backend(ctx)
    return {
        "options": admin.auth_options(),
        "key_configured": bool(backend.settings.api_key),
        "base_url": backend.settings.base_url,
    }


@tool(category="cortex", risk="read", network=True, timeout=180.0, tags=["cortex", "test"])
def cortex_test_model(model_id: str = "", prompt: str = "ping", timeout: float = 60.0,
                      context: ToolContext | None = None) -> dict[str, Any]:
    """Send a one-token probe through a model to verify end-to-end inference.

    Args:
        model_id: model to test, empty for the Cortex default
        prompt: the probe prompt
        timeout: seconds to wait for the completion
    """
    ctx = context or ToolContext()
    backend = _backend(ctx)
    admin = _admin(ctx)
    target = model_id or backend.default_model_id()
    if not target:
        raise ToolExecutionError("No model id supplied and Cortex reports no default model")
    response = admin.test_model(target, prompt=prompt, timeout=timeout)
    return {"model_id": target, "prompt": prompt, "response": response,
            "status": admin.model_status(target)}


@tool(category="cortex", risk="read", network=True, tags=["cortex", "cookbook"])
def cortex_cookbook(search: str = "", page: int = 1, limit: int = 20,
                    context: ToolContext | None = None) -> dict[str, Any]:
    """Read Cortex's cookbook of example prompts and recipes.

    Args:
        search: filter cookbook entries by substring
        page: page number
        limit: entries per page
    """
    ctx = context or ToolContext()
    admin = _admin(ctx)
    return admin.cookbook(page=max(1, page), limit=max(1, min(limit, 100)), search=search)


@tool(category="cortex", risk="read", network=False, tags=["cortex", "diagnose"])
def cortex_diagnose(context: ToolContext | None = None) -> dict[str, Any]:
    """Run every Cortex probe and return one consolidated, non-raising report.

    Designed to be the first thing an operator reads when MAgent cannot reach
    its inference host: it reports which probe failed and why, rather than
    propagating the first exception.
    """
    ctx = context or ToolContext()
    checks: list[dict[str, Any]] = []
    try:
        backend = _backend(ctx)
        checks.append({"name": "configured", "ok": True,
                       "detail": backend.settings.base_url})
    except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
        checks.append({"name": "configured", "ok": False, "detail": str(exc)})
        return {"ok": False, "checks": checks, "summary": "no Cortex server configured",
                "hint": "Set MAGENT_CORTEX_URL, or start Cortex LLMHoster on port 8624."}
    admin = CortexAdmin(backend=backend)
    probes: list[tuple[str, Any]] = [
        ("liveness", lambda: backend.ping()),
        ("health", lambda: backend.health().get("ok")),
        ("models", lambda: len(backend.model_ids())),
        ("admin_state", lambda: bool(admin.state().version)),
        ("metrics", lambda: admin.metrics().uptime_seconds >= 0),
    ]
    details: dict[str, Any] = {}
    for name, call in probes:
        try:
            value = call()
            ok = bool(value) if isinstance(value, bool) else True
            checks.append({"name": name, "ok": ok, "detail": _summarize(value)})
            details[name] = value if not isinstance(value, dict) else _summarize(value)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
            checks.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    failures = [check for check in checks if not check["ok"]]
    summary = "all checks passed" if not failures else "; ".join(
        f"{check['name']}: {check['detail']}" for check in failures
    )
    return {
        "ok": not failures,
        "base_url": backend.settings.base_url,
        "checks": checks,
        "details": details,
        "summary": summary[:2000],
        "hint": "" if not failures else _diagnose_hint(failures),
    }


def _diagnose_hint(failures: list[dict[str, Any]]) -> str:
    """Suggest a concrete next step based on which probes failed."""
    names = {failure["name"] for failure in failures}
    if "liveness" in names and "health" in names:
        return ("Cortex is not answering on the configured URL. Start it with "
                "'cortex-llmhoster' or set MAGENT_CORTEX_URL.")
    if "models" in names:
        return ("Cortex is up but serves no models. Configure one in Cortex and "
                "start it, then retry.")
    if "admin_state" in names or "metrics" in names:
        return ("The inference API answered but the admin API did not. Check "
                "MAGENT_CORTEX_API_KEY and CORTEX_API_AUTH_ENABLED.")
    return "Inspect the failing check above."


def _summarize(result: Any) -> str:
    """Render a Cortex response as a short human-readable detail line."""
    if isinstance(result, bool):
        return "ok" if result else "failed"
    if isinstance(result, dict):
        return ellipsize_json(result, 200)
    if isinstance(result, list):
        return f"{len(result)} item(s)"
    return str(result)[:200]


def ellipsize_json(value: Any, limit: int = 200) -> str:
    """Serialize a value to JSON and truncate it for display."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


TOOLS: list[Any] = [
    cortex_health, cortex_models, cortex_metrics, cortex_state, cortex_hardware,
    cortex_model_control, cortex_model_files, cortex_update_check, cortex_auth_options,
    cortex_test_model, cortex_cookbook, cortex_diagnose,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every Cortex tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "cortex_auth_options",
    "cortex_cookbook",
    "cortex_diagnose",
    "cortex_hardware",
    "cortex_health",
    "cortex_metrics",
    "cortex_model_control",
    "cortex_model_files",
    "cortex_models",
    "cortex_state",
    "cortex_test_model",
    "cortex_update_check",
    "ellipsize_json",
    "load",
]
