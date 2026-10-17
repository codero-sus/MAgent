"""The ``cortex`` command: drive Cortex LLMHoster from the CLI.

MAgent talks to `Cortex LLMHoster`_ over its HTTP API and never embeds or
patches it - the upstream license forbids modification and redistribution, and
keeping Cortex as a separate process is also what lets one host serve several
clients.

Everything here maps onto a real Cortex endpoint: ``/health`` and ``/ready`` for
liveness, ``/v1/models`` plus ``/admin/api/runtime`` for what is loaded, and the
``/admin/api`` control plane for starting models, reading metrics, inspecting
hardware and managing configuration. Destructive actions (stopping every model,
resetting configuration or metrics, installing an update) ask first unless
``--yes`` is given.

.. _Cortex LLMHoster: https://github.com/codero-sus/Cortex_LLMHoster
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...errors import BackendUnavailableError, MAgentError
from ...llm.cortex import CortexBackend, CortexSettings, cortex_health
from ...llm.cortex_admin import CortexAdmin
from ...utils.tables import render_key_value, render_markdown_table, render_table
from ...utils.timeutil import format_duration
from ..args import CliContext, EXIT_ERROR, EXIT_OK, confirm

STATUS_ICONS: dict[str, str] = {
    "running": "+",
    "stopped": "-",
    "removed": "x",
    "starting": "~",
    "unknown": "?",
}
"""Icon per model runtime state."""


def register(subparsers: Any) -> None:
    """Add the ``cortex`` subcommand and its actions."""
    parser = subparsers.add_parser("cortex", help="manage the Cortex LLMHoster server",
                                   description="Inspect and control Cortex LLMHoster.")
    parser.add_argument("--url", default="", metavar="URL",
                        help="Cortex base URL, overriding configuration")
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    action.add_parser("status", help="show health, readiness and version")

    models = action.add_parser("models", help="list models and their runtime state")
    models.add_argument("--running", action="store_true", help="only running models")

    start = action.add_parser("start", help="start (load) a model")
    start.add_argument("model", help="model id")
    start.add_argument("--wait-timeout", type=float, default=120.0,
                       help="seconds to wait for the model to load")
    start.add_argument("--no-wait", action="store_true", help="do not wait for it to load")

    stop = action.add_parser("stop", help="stop (unload) a model")
    stop.add_argument("model", help="model id")

    restart = action.add_parser("restart", help="restart a model")
    restart.add_argument("model", help="model id")

    action.add_parser("stop-all", help="stop every running model")

    test = action.add_parser("test", help="send one prompt to a model")
    test.add_argument("model", help="model id")
    test.add_argument("--prompt", default="Reply with the single word: pong",
                      help="prompt to send")

    metrics = action.add_parser("metrics", help="show request metrics")
    metrics.add_argument("--reset", action="store_true", help="reset counters after reading")

    action.add_parser("state", help="print the full server state as JSON")
    action.add_parser("runtime", help="show the runtime and loaded models")
    action.add_parser("hardware", help="show detected hardware")

    files = action.add_parser("files", help="list model files on the server")
    files.add_argument("--directory", default="", help="restrict to a directory")

    upload = action.add_parser("upload", help="upload a model file")
    upload.add_argument("path", help="local file to upload")
    upload.add_argument("--directory", default="", help="destination directory")

    config = action.add_parser("config", help="inspect the server configuration")
    config.add_argument("--reload", action="store_true", help="ask the server to reload it")
    config.add_argument("--reset", action="store_true", help="reset it to defaults")

    update = action.add_parser("update", help="check for a Cortex update")
    update.add_argument("--install", action="store_true", help="install an available update")

    action.add_parser("auth", help="show the server's authentication options")

    cookbook = action.add_parser("cookbook", help="list server cookbook entries")
    cookbook.add_argument("--search", default="", help="filter by text")
    cookbook.add_argument("--limit", type=int, default=20, help="maximum entries")

    parser.set_defaults(handler=cmd_cortex, action_default="status")


def _settings(context: CliContext, args: argparse.Namespace) -> CortexSettings:
    """Build Cortex settings from configuration plus the ``--url`` flag."""
    url = str(getattr(args, "url", "") or "").rstrip("/")
    server = context.config.server
    from ...secrets import is_secret_reference, resolve_secret

    raw_key = str(server.api_key or "")
    api_key = resolve_secret(raw_key) if is_secret_reference(raw_key) else raw_key
    settings = CortexSettings(
        base_url=url or str(server.base_url or "").rstrip("/") or CortexSettings().base_url,
        api_key=api_key,
        api_prefix=str(server.api_prefix or "/v1"),
        timeout=float(server.timeout or 120.0),
        verify_ssl=bool(server.verify_ssl),
        max_retries=int(server.max_retries or 3),
        default_model=str(context.config.model.model or ""),
        embedding_model=str(context.config.model.embedding_model or ""),
    )
    return settings


def _admin(context: CliContext, args: argparse.Namespace) -> CortexAdmin:
    """Build an admin client, raising a clear error when unreachable."""
    settings = _settings(context, args)
    health = cortex_health(settings.base_url, timeout=min(3.0, settings.timeout))
    if not health or health.get("status") != "ok":
        raise BackendUnavailableError(
            f"Cortex LLMHoster is not reachable at {settings.base_url}",
            details={"url": settings.base_url, "health": health,
                     "hint": "start Cortex, or pass --url"},
        )
    backend = CortexBackend(settings)
    return CortexAdmin(backend=backend)


def cmd_cortex(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``cortex`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "status")
    handlers: dict[str, Any] = {
        "status": _status,
        "models": _models,
        "start": _start,
        "stop": _stop,
        "restart": _restart,
        "stop-all": _stop_all,
        "test": _test,
        "metrics": _metrics,
        "state": _state,
        "runtime": _runtime,
        "hardware": _hardware,
        "files": _files,
        "upload": _upload,
        "config": _config,
        "update": _update,
        "auth": _auth,
        "cookbook": _cookbook,
    }
    handler = handlers.get(action, _status)
    return handler(args, context)


def _status(args: argparse.Namespace, context: CliContext) -> int:
    """Show liveness, readiness, and version."""
    settings = _settings(context, args)
    health = cortex_health(settings.base_url, timeout=3.0)
    reachable = bool(health) and health.get("status") == "ok"
    payload: dict[str, Any] = {"url": settings.base_url, "reachable": reachable, "health": health}
    admin: CortexAdmin | None = None
    if reachable:
        try:
            admin = _admin(context, args)
            state = admin.state()
            payload.update({
                "version": state.version,
                "ready": state.runtime_models() and True or False,
                "models_configured": len(state.models()),
                "running": state.running_models(),
                "auth_enabled": state.auth_enabled,
                "api": state.api(),
                "metrics": state.metrics(),
            })
        except MAgentError as exc:
            payload["error"] = str(exc)
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK if reachable else EXIT_ERROR
    printer = context.printer
    printer.write(printer.themed(settings.base_url, "brand"), min_level="quiet")
    if not reachable:
        printer.error("unreachable - is Cortex LLMHoster running?")
        printer.write(printer.themed(
            "hint: start it, or point MAgent elsewhere with --url or MAGENT_CORTEX_URL", "muted"),
            min_level="quiet")
        return EXIT_ERROR
    pairs: list[tuple[str, Any]] = [("health", health.get("status", "ok"))]
    if "version" in payload:
        pairs += [
            ("version", payload["version"]),
            ("models", payload["models_configured"]),
            ("running", ", ".join(payload["running"]) or "none"),
            ("auth", "enabled" if payload.get("auth_enabled") else "disabled"),
        ]
        api = payload.get("api") or {}
        if api:
            pairs.append(("api", f"{api.get('format', '')} {api.get('base_path', '')}"))
        metrics = payload.get("metrics") or {}
        if metrics:
            pairs.append(("requests", f"{metrics.get('total_requests', 0)} total, "
                                      f"{metrics.get('failed_requests', 0)} failed"))
            if metrics.get("uptime_seconds"):
                pairs.append(("uptime", format_duration(float(metrics["uptime_seconds"]))))
    printer.write(render_key_value(pairs), min_level="quiet")
    return EXIT_OK


def _models(args: argparse.Namespace, context: CliContext) -> int:
    """List models with their runtime state and capabilities."""
    admin = _admin(context, args)
    state = admin.state()
    runtime = state.runtime_models()
    configured = state.model_infos()
    rows: list[list[Any]] = []
    for info in configured:
        model_id = info.id
        status = str(runtime.get(model_id, {}).get("status", "") if isinstance(
            runtime.get(model_id), dict) else state.model_status(model_id))
        if getattr(args, "running", False) and status != "running":
            continue
        rows.append([
            STATUS_ICONS.get(status, "?"),
            model_id,
            status or "unknown",
            str(getattr(info, "runtime", "") or ""),
            ",".join(getattr(info, "capabilities", []) or [])[:34],
            "default" if getattr(info, "default", False) else "",
        ])
    if context.options.fmt == "json":
        context.emit({"url": admin.settings.base_url, "count": len(rows),
                      "models": [info.to_dict() if hasattr(info, "to_dict") else str(info)
                                 for info in configured],
                      "runtime": runtime})
        return EXIT_OK
    if not rows:
        context.printer.warn("no models matched"
                             + (" (none are running)" if getattr(args, "running", False) else ""))
        return EXIT_OK
    table = render_markdown_table(rows, headers=["", "id", "state", "runtime", "capabilities", ""]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["", "id", "state", "runtime", "capabilities", ""],
                          alignments=["left", "left", "left", "left", "left", "left"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    running = sum(1 for row in rows if row[2] == "running")
    context.printer.write(context.printer.themed(
        f"{len(rows)} model(s), {running} running", "muted"), min_level="normal")
    return EXIT_OK


def _start(args: argparse.Namespace, context: CliContext) -> int:
    """Start a model and, by default, wait until it is running."""
    admin = _admin(context, args)
    model = str(getattr(args, "model", "") or "")
    wait = not getattr(args, "no_wait", False)
    timeout = float(getattr(args, "wait_timeout", 120.0) or 120.0)
    if context.dry_run:
        context.printer.write(f"would start model {model!r} (wait={wait}, timeout={timeout}s)",
                              min_level="quiet")
        return EXIT_OK
    context.printer.write(context.printer.themed(f"starting {model}...", "info"), min_level="normal")
    outcome = admin.start_model(model, wait=wait, timeout=timeout)
    status = admin.model_status(model)
    if context.options.fmt == "json":
        context.emit({"model": model, "status": status, "response": outcome})
        return EXIT_OK if status == "running" or not wait else EXIT_ERROR
    if status == "running":
        context.printer.write(context.printer.themed(f"{model} is running", "success"),
                              min_level="quiet")
    else:
        context.printer.warn(f"{model} is {status or 'not running'} after the request")
        context.printer.write(json.dumps(outcome, indent=2, default=str), min_level="verbose")
    return EXIT_OK if status == "running" or not wait else EXIT_ERROR


def _stop(args: argparse.Namespace, context: CliContext) -> int:
    """Stop one model."""
    admin = _admin(context, args)
    model = str(getattr(args, "model", "") or "")
    if context.dry_run:
        context.printer.write(f"would stop model {model!r}", min_level="quiet")
        return EXIT_OK
    outcome = admin.stop_model(model)
    if context.options.fmt == "json":
        context.emit({"model": model, "response": outcome})
        return EXIT_OK
    context.printer.write(f"stopped {model}", min_level="quiet")
    return EXIT_OK


def _restart(args: argparse.Namespace, context: CliContext) -> int:
    """Restart one model."""
    admin = _admin(context, args)
    model = str(getattr(args, "model", "") or "")
    if context.dry_run:
        context.printer.write(f"would restart model {model!r}", min_level="quiet")
        return EXIT_OK
    outcome = admin.restart_model(model)
    status = admin.model_status(model)
    if context.options.fmt == "json":
        context.emit({"model": model, "status": status, "response": outcome})
        return EXIT_OK
    context.printer.write(f"restarted {model} ({status})", min_level="quiet")
    return EXIT_OK


def _stop_all(args: argparse.Namespace, context: CliContext) -> int:
    """Stop every running model, asking first."""
    admin = _admin(context, args)
    running = list(admin.iter_running_models())
    if not running:
        context.printer.write("no models are running", min_level="quiet")
        return EXIT_OK
    if context.dry_run:
        context.printer.write(f"would stop {len(running)} model(s): {', '.join(running)}",
                              min_level="quiet")
        return EXIT_OK
    if not confirm(f"stop {len(running)} running model(s): {', '.join(running)}?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    outcome = admin.stop_all_models()
    if context.options.fmt == "json":
        context.emit({"stopped": running, "response": outcome})
        return EXIT_OK
    context.printer.write(f"stopped {len(running)} model(s)", min_level="quiet")
    return EXIT_OK


def _test(args: argparse.Namespace, context: CliContext) -> int:
    """Send one prompt to a model and show the reply."""
    admin = _admin(context, args)
    model = str(getattr(args, "model", "") or "")
    prompt = str(getattr(args, "prompt", "") or "ping")
    timeout = float(getattr(args, "timeout", 60.0) or 60.0)
    if context.dry_run:
        context.printer.write(f"would send {prompt!r} to {model!r}", min_level="quiet")
        return EXIT_OK
    outcome = admin.test_model(model, prompt=prompt, timeout=timeout)
    if context.options.fmt == "json":
        context.emit({"model": model, "prompt": prompt, "response": outcome})
        return EXIT_OK
    ok = bool(outcome.get("ok", True)) if isinstance(outcome, dict) else True
    reply = ""
    if isinstance(outcome, dict):
        reply = str(outcome.get("reply") or outcome.get("content") or outcome.get("text") or "")
        if not reply:
            reply = json.dumps(outcome, ensure_ascii=False, default=str)
    if ok:
        context.printer.write(reply.strip() or "(empty reply)", min_level="quiet")
    else:
        context.printer.error(f"{model} did not answer: {reply}")
    return EXIT_OK if ok else EXIT_ERROR


def _metrics(args: argparse.Namespace, context: CliContext) -> int:
    """Show request metrics, optionally resetting them afterwards."""
    admin = _admin(context, args)
    metrics = admin.metrics()
    payload = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
    if getattr(args, "reset", False):
        if context.dry_run:
            payload["would_reset"] = True
        elif confirm("reset all Cortex metrics?", context=context,
                     assume_yes=context.assume_yes):
            payload["reset"] = admin.reset_metrics()
    if context.options.fmt == "json":
        context.emit(payload)
        return EXIT_OK
    pairs: list[tuple[str, Any]] = [
        ("uptime", format_duration(float(payload.get("uptime_seconds", 0.0) or 0.0))),
        ("in flight", payload.get("in_flight", 0)),
        ("requests", payload.get("total_requests", 0)),
        ("completed", payload.get("completed_requests", 0)),
        ("failed", payload.get("failed_requests", 0)),
        ("avg latency", f"{float(payload.get('average_latency_ms', 0.0) or 0.0):.1f} ms"),
    ]
    context.printer.write(render_key_value(pairs), min_level="quiet")
    for label, key in (("status codes", "status_codes"), ("by model", "requests_by_model")):
        block = payload.get(key) or {}
        if block:
            context.printer.write(context.printer.themed(label, "heading"), min_level="normal")
            rows = [[str(item), value] for item, value in sorted(block.items(),
                                                                  key=lambda kv: -int(kv[1] or 0))]
            context.printer.write(render_table(rows, headers=[label, "count"],
                                               alignments=["left", "right"],
                                               unicode_borders=context.options.unicode),
                                  min_level="normal")
    return EXIT_OK


def _state(args: argparse.Namespace, context: CliContext) -> int:
    """Print the whole server state."""
    admin = _admin(context, args)
    state = admin.state()
    context.printer.write(json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str),
                          min_level="quiet")
    return EXIT_OK


def _runtime(args: argparse.Namespace, context: CliContext) -> int:
    """Show the runtime block: hardware summary and per-model state."""
    admin = _admin(context, args)
    data = admin.runtime()
    if context.options.fmt == "json":
        context.emit(data)
        return EXIT_OK
    models = data.get("models") or {}
    rows = [[model_id, (value or {}).get("status", "?"), (value or {}).get("pid", ""),
             (value or {}).get("base_url", "")] for model_id, value in models.items()]
    if rows:
        context.printer.write(render_table(rows, headers=["model", "status", "pid", "base_url"],
                                           unicode_borders=context.options.unicode),
                              min_level="quiet")
    else:
        context.printer.write("no models in the runtime", min_level="quiet")
    hardware = data.get("hardware") or {}
    if hardware:
        context.printer.write(render_key_value([(str(key), value)
                                                for key, value in hardware.items()]),
                              min_level="normal")
    return EXIT_OK


def _hardware(args: argparse.Namespace, context: CliContext) -> int:
    """Show detected hardware."""
    admin = _admin(context, args)
    data = admin.hardware()
    if context.options.fmt == "json":
        context.emit(data)
        return EXIT_OK
    context.printer.write(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                          min_level="quiet")
    return EXIT_OK


def _files(args: argparse.Namespace, context: CliContext) -> int:
    """List model files known to the server."""
    admin = _admin(context, args)
    directory = str(getattr(args, "directory", "") or "") or None
    files = admin.model_files(directory=directory)
    if context.options.fmt == "json":
        context.emit({"count": len(files), "files": files})
        return EXIT_OK
    if not files:
        context.printer.write("no model files", min_level="quiet")
        return EXIT_OK
    rows = [[item.get("name", ""), item.get("path", ""),
             str(item.get("size", "")), item.get("type", "")] for item in files]
    context.printer.write(render_table(rows, headers=["name", "path", "size", "type"],
                                       unicode_borders=context.options.unicode),
                          min_level="quiet")
    return EXIT_OK


def _upload(args: argparse.Namespace, context: CliContext) -> int:
    """Upload a model file to the server."""
    from pathlib import Path

    source = Path(str(getattr(args, "path", "") or "")).expanduser()
    if not source.exists():
        context.printer.error(f"error: no such file: {source}")
        return EXIT_ERROR
    admin = _admin(context, args)
    directory = str(getattr(args, "directory", "") or "")
    if context.dry_run:
        context.printer.write(f"would upload {source} ({source.stat().st_size} bytes)"
                              + (f" to {directory}" if directory else ""), min_level="quiet")
        return EXIT_OK
    if not confirm(f"upload {source.name} ({source.stat().st_size} bytes) to {admin.settings.base_url}?",
                   context=context, assume_yes=context.assume_yes):
        context.printer.warn("cancelled")
        return EXIT_OK
    outcome = admin.upload_model_file(source, directory=directory)
    if context.options.fmt == "json":
        context.emit(outcome)
        return EXIT_OK
    context.printer.write(f"uploaded {source.name}", min_level="quiet")
    return EXIT_OK


def _config(args: argparse.Namespace, context: CliContext) -> int:
    """Export, reload, or reset the server configuration."""
    admin = _admin(context, args)
    if getattr(args, "reset", False):
        if context.dry_run:
            context.printer.write("would reset the server configuration to defaults",
                                  min_level="quiet")
            return EXIT_OK
        if not confirm("reset the Cortex server configuration to defaults?",
                       context=context, assume_yes=context.assume_yes):
            context.printer.warn("cancelled")
            return EXIT_OK
        context.emit(admin.reset_config())
        return EXIT_OK
    if getattr(args, "reload", False):
        if context.dry_run:
            context.printer.write("would ask the server to reload its configuration",
                                  min_level="quiet")
            return EXIT_OK
        context.emit(admin.reload_config())
        return EXIT_OK
    data = admin.export_config()
    context.printer.write(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                          min_level="quiet")
    return EXIT_OK


def _update(args: argparse.Namespace, context: CliContext) -> int:
    """Check for a Cortex update, installing one when asked."""
    admin = _admin(context, args)
    info = admin.check_update()
    available = bool(info.get("available", False)) if isinstance(info, dict) else False
    if getattr(args, "install", False):
        if not available:
            context.printer.write("already up to date", min_level="quiet")
            return EXIT_OK
        if context.dry_run:
            context.printer.write(f"would install {info.get('version', 'the update')}",
                                  min_level="quiet")
            return EXIT_OK
        if not confirm("install the Cortex update now? the server may restart",
                       context=context, assume_yes=context.assume_yes):
            context.printer.warn("cancelled")
            return EXIT_OK
        info = admin.install_update()
    if context.options.fmt == "json":
        context.emit(info)
        return EXIT_OK
    if isinstance(info, dict):
        context.printer.write(render_key_value([(str(key), value) for key, value in info.items()]),
                              min_level="quiet")
    else:
        context.printer.write(str(info), min_level="quiet")
    return EXIT_OK


def _auth(args: argparse.Namespace, context: CliContext) -> int:
    """Show the server's authentication options."""
    admin = _admin(context, args)
    data = admin.auth_options()
    if context.options.fmt == "json":
        context.emit(data)
        return EXIT_OK
    if isinstance(data, dict) and data:
        context.printer.write(render_key_value([(str(key), value) for key, value in data.items()]),
                              min_level="quiet")
    else:
        context.printer.write("the server did not report authentication options",
                              min_level="quiet")
    return EXIT_OK


def _cookbook(args: argparse.Namespace, context: CliContext) -> int:
    """List cookbook entries published by the server."""
    admin = _admin(context, args)
    data = admin.cookbook(search=str(getattr(args, "search", "") or ""),
                          limit=int(getattr(args, "limit", 20) or 20))
    if context.options.fmt == "json":
        context.emit(data)
        return EXIT_OK
    entries = data.get("items") or data.get("entries") or [] if isinstance(data, dict) else []
    if not entries:
        context.printer.write("no cookbook entries", min_level="quiet")
        return EXIT_OK
    rows = [[item.get("title", item.get("name", "")), item.get("id", ""),
             str(item.get("category", ""))] for item in entries if isinstance(item, dict)]
    context.printer.write(render_table(rows, headers=["title", "id", "category"],
                                       unicode_borders=context.options.unicode),
                          min_level="quiet")
    return EXIT_OK


__all__ = ["STATUS_ICONS", "cmd_cortex", "register"]
