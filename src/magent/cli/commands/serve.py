"""The ``serve`` command: expose MAgent over HTTP.

The same agent the CLI drives becomes a small REST API plus a web console, so a
notebook, a phone, or another service can use it without a Python interpreter.
Handlers delegate to the CLI's own objects, which is why this command is thin:
it resolves an address, builds a context, and stays out of the way.

Two details matter more than they look. The bind address defaults to loopback,
because this API can start runs that execute tools. And the server's own key is
``--server-key``, deliberately not ``--api-key`` - that global flag belongs to
the upstream LLM, and reusing it would silently point the wrong credential at
the wrong service.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ...errors import ServerError_
from ...secrets import generate_api_key
from ...server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAgentServer,
    ServerContext,
    build_context,
)
from ...utils.tables import render_key_value, render_table
from ..args import EXIT_ERROR, EXIT_INTERRUPT, EXIT_OK, CliContext

COMMAND_NAMES: tuple[str, ...] = ("serve",)
"""Subcommands this module registers."""

def register(subparsers: Any) -> None:
    """Add the ``serve`` subcommand."""
    parser = subparsers.add_parser(
        "serve",
        help="run the HTTP API and web console",
        description="Serve MAgent over HTTP: a REST API for runs, tools, memory "
                    "and retrieval, plus a bundled web console.",
        epilog="examples:\n"
               "  magent serve                     # loopback on port 8700\n"
               "  magent serve --port 0            # let the OS pick a free port\n"
               "  magent serve --generate-key      # print a key, then require it\n"
               "  magent serve --host 0.0.0.0 --server-key $KEY\n"
               "  magent serve --routes            # list the API without binding\n"
               "  magent serve --offline --no-ui   # an API with no LLM and no console\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="", metavar="ADDR",
                        help=f"address to bind (default {DEFAULT_HOST}, loopback only)")
    parser.add_argument("--port", type=int, default=0, metavar="PORT",
                        help=f"port to bind, 0 picks a free one (default {DEFAULT_PORT})")
    parser.add_argument("--server-key", default="", metavar="KEY",
                        help="API key clients must send as a Bearer token "
                             "(not --api-key, which is the upstream LLM key)")
    parser.add_argument("--generate-key", action="store_true",
                        help="generate a key, print it once, and require it")
    parser.add_argument("--no-auth", action="store_true",
                        help="serve without a key; refused unless the bind address is loopback")
    parser.add_argument("--no-ui", action="store_true", help="do not serve the web console")
    parser.add_argument("--threads", type=int, default=0, metavar="N",
                        help="worker thread ceiling (default from [http] threads)")
    parser.add_argument("--cors", action="append", default=[], metavar="ORIGIN",
                        help="allow this origin; repeatable (default: reflect any)")
    parser.add_argument("--max-body", type=int, default=0, metavar="BYTES",
                        help="largest accepted request body")
    parser.add_argument("--request-timeout", type=float, default=0.0, metavar="SECONDS",
                        help="how long one handler may run (not the global --timeout)")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="open the console in a browser once bound")
    parser.add_argument("--routes", action="store_true",
                        help="print the route table and exit without binding")
    parser.add_argument("--force", action="store_true",
                        help="serve even when [http] enabled=false")
    parser.set_defaults(handler=cmd_serve)


def cmd_serve(args: argparse.Namespace, context: CliContext) -> int:
    """Start the HTTP server and block until interrupted."""
    if args.routes:
        return _print_routes(context)

    http = context.config.http
    if not http.enabled and not args.force:
        context.printer.warn("[http] enabled=false; pass --force to serve anyway")
        return EXIT_OK

    server_key = _resolve_key(args, context)
    if server_key is None:
        return EXIT_ERROR

    host = args.host or http.host or DEFAULT_HOST
    port = int(args.port or http.port or DEFAULT_PORT)
    serve_ui = not args.no_ui and bool(http.serve_ui)

    if args.dry_run:
        return _report_plan(args, context, host=host, port=port, serve_ui=serve_ui,
                            key=bool(server_key))

    server_context = _server_context(context)
    try:
        server = MAgentServer(
            server_context, host=host, port=port, api_key=server_key,
            cors_origins=list(args.cors or http.cors_origins or []),
            serve_ui=serve_ui,
            max_body_bytes=int(args.max_body or http.max_body_bytes or 0),
            request_timeout=float(args.request_timeout or http.request_timeout or 0.0),
            threads=int(args.threads or http.threads or 0),
            allow_origins_any=bool(http.allow_origins_any),
            require_auth_off_loopback=not args.no_auth,
        )
    except ServerError_ as exc:
        context.printer.error(str(exc))
        for key, value in (exc.details or {}).items():
            context.printer.write(f"  {key}: {value}")
        context.printer.write("  hint: --server-key KEY, or bind 127.0.0.1",
                              min_level="verbose")
        return EXIT_ERROR

    try:
        server.start()
    except ServerError_ as exc:
        context.printer.error(str(exc))
        return EXIT_ERROR

    _announce(args, context, server, key=server_key)
    if args.open_browser:
        _open_browser(context, server.url)
    return _block(server, context)


def _resolve_key(args: argparse.Namespace, context: CliContext) -> str | None:
    """Work out which key, if any, the server should require.

    Returns:
        The key to use, an empty string for no authentication, or ``None`` when
        the combination of flags makes no sense and the command should stop.
    """
    if args.generate_key:
        key = generate_api_key()
        context.printer.heading("generated api key")
        context.printer.write(key, min_level="quiet")
        context.printer.warn("this key is printed once; it is not stored anywhere")
        return key
    if args.server_key:
        return args.server_key
    configured = str(context.config.http.api_key or "")
    if configured:
        from ...secrets import is_secret_reference, resolve_secret

        return resolve_secret(configured) if is_secret_reference(configured) else configured
    if args.no_auth:
        return ""
    return ""


def _server_context(context: CliContext) -> ServerContext:
    """Build the server context from everything the CLI already resolved."""
    agent = None
    try:
        agent = context.make_agent()
    except Exception as exc:  # noqa: BLE001 - the API still serves tools and stores
        context.printer.warn(f"no agent available: {exc}")
    built = build_context(context.config, agent=agent, registry=context.registry,
                          backend=context.backend)
    decision = getattr(context, "decision", None)
    if decision is not None and getattr(decision, "reason", ""):
        built.notes.append(f"backend: {decision.reason}")
    return built


def _announce(args: argparse.Namespace, context: CliContext, server: MAgentServer,
              *, key: str) -> None:
    """Print where the server is and what it can do."""
    info = server.info()
    printer = context.printer
    printer.heading("MAgent server")
    printer.write(render_key_value([
        ("url", server.url),
        ("console", f"{server.url}/" if server.serve_ui else "disabled"),
        ("backend", info.get("backend") or "(none)"),
        ("model", info.get("model") or "(backend default)"),
        ("tools", info.get("tools", 0)),
        ("routes", info.get("routes", 0)),
        ("auth", "required" if key else "off"),
        ("bound to", "loopback" if info.get("loopback") else "all interfaces"),
    ]), min_level="quiet")
    for note in getattr(server.context, "notes", []):
        printer.write(f"note: {note}", min_level="verbose")
    printer.blank()
    printer.write("try:", min_level="verbose")
    for line in (f"  curl {server.url}/health",
                 f"  curl {server.url}/api/tools?category=text",
                 f"  curl -X POST {server.url}/api/run -H 'Content-Type: application/json' "
                 f"-d '{{\"prompt\":\"hello\"}}'"):
        printer.write(line, min_level="verbose")
    printer.blank(min_level="verbose")
    printer.write("ctrl-c to stop", min_level="normal")


def _block(server: MAgentServer, context: CliContext) -> int:
    """Serve until interrupted, then report what happened.

    The socket is closed in the calling thread rather than left to
    ``start(block=True)``, so the CLI keeps control of the exit code and can
    print request statistics on the way out.
    """
    interrupted = False
    started = time.perf_counter()
    try:
        while server.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        interrupted = True
        context.printer.blank()
        context.printer.warn("interrupted; shutting down")
    finally:
        server.stop()
    stats = server.context.stats()
    stats["served_seconds"] = round(time.perf_counter() - started, 2)
    if context.options.fmt == "json":
        context.printer.write(json.dumps(stats, indent=2, default=str), min_level="quiet")
    else:
        context.printer.write(render_key_value(list(stats.items())), min_level="verbose")
    return EXIT_INTERRUPT if interrupted else EXIT_OK


def _open_browser(context: CliContext, url: str) -> None:
    """Open the console, reporting rather than raising when there is no browser."""
    import webbrowser

    try:
        if webbrowser.open(url):
            context.printer.write(f"opened {url}", min_level="verbose")
            return
    except Exception as exc:  # noqa: BLE001 - a headless box has no browser
        context.printer.warn(f"could not open a browser: {exc}")
    else:
        context.printer.warn("no browser available; open the URL yourself")


def _report_plan(args: argparse.Namespace, context: CliContext, *, host: str, port: int,
                 serve_ui: bool, key: bool) -> int:
    """Describe what ``serve`` would do, without binding a socket."""
    plan = {
        "command": "serve",
        "dry_run": True,
        "host": host,
        "port": port,
        "auth": "required" if key else "off",
        "console": serve_ui,
        "threads": int(args.threads or context.config.http.threads),
        "cors_origins": list(args.cors or context.config.http.cors_origins or []),
        "backend": str(getattr(context.backend, "name", "") or ""),
        "tools": context.registry.count() if context.registry is not None else 0,
        "routes": len(build_routes_preview()),
    }
    context.emit(plan, text=render_key_value(list(plan.items())))
    return EXIT_OK


def build_routes_preview() -> list[tuple[str, str, Any, dict[str, Any]]]:
    """Return the route table without constructing a server."""
    from ...server.routes import build_routes

    return build_routes()


def route_rows() -> list[dict[str, Any]]:
    """Return the route table as plain data, without binding a socket."""
    rows: list[dict[str, Any]] = []
    for method, pattern, handler, kwargs in build_routes_preview():
        rows.append({
            "method": method,
            "path": pattern,
            "name": str(kwargs.get("name") or getattr(handler, "__name__", "")),
            "auth": bool(kwargs.get("auth", True)),
            "body": bool(kwargs.get("body", False)),
        })
    rows.sort(key=lambda row: (row["path"], row["method"]))
    return rows


def _print_routes(context: CliContext) -> int:
    """Print the API surface and exit."""
    rows = route_rows()
    if context.options.fmt == "json":
        context.printer.write(json.dumps({"count": len(rows), "routes": rows}, indent=2),
                              min_level="quiet")
        return EXIT_OK
    context.printer.heading("HTTP API")
    context.printer.write(render_table(
        [(row["method"], row["path"], row["name"],
          "yes" if row["auth"] else "no", "yes" if row["body"] else "-") for row in rows],
        headers=("method", "path", "name", "auth", "body"),
    ), min_level="quiet")
    context.printer.blank()
    context.printer.write(
        f"{len(rows)} routes. Bodies and responses are JSON; errors are "
        f'{{"error", "status", "code", "details"}}.', min_level="normal")
    context.printer.write("/api/events streams server-sent events.", min_level="normal")
    return EXIT_OK


__all__ = [
    "COMMAND_NAMES",
    "build_routes_preview",
    "cmd_serve",
    "register",
    "route_rows",
]
