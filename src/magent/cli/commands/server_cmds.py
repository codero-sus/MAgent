"""The ``serve`` command: run the HTTP API and web UI.

Serving blocks the terminal, so the command prints the URL, the number of routes
and whether authentication is on before it does - and it prints the *bound* port,
which matters when ``--port 0`` was used to get a free one.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..commands import Command


def _configure_serve(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent serve``."""
    parser.add_argument("--host", default="", help="bind address (default from config)")
    parser.add_argument("--port", type=int, default=0, help="bind port (0 picks a free one)")
    parser.add_argument("--no-ui", dest="no_ui", action="store_true", help="disable the web UI")
    parser.add_argument("--background", action="store_true",
                        help="start detached and print the URL")


def serve_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent serve`` - start the HTTP server."""
    from ...server import MagentServer

    if args.no_ui:
        ctx.config.http.serve_ui = False
    if args.port:
        ctx.config.http.port = int(args.port)
    if args.host:
        ctx.config.http.host = args.host
    server = MagentServer(
        ctx.config,
        backend=ctx._cache.get("backend"),
        registry=ctx.registry,
        memory=ctx.memory,
        guardrails=ctx.guardrails,
        sessions=ctx.sessions,
        retriever=ctx._cache.get("retriever"),
        skills=ctx.skills,
        observability=ctx._cache.get("observability"),
    )
    if args.background:
        url = server.start_background()
        ctx.notice(f"serving on {url}", level="success")
        ctx.notice(f"{len(server.router)} routes; UI {'on' if server.serve_ui else 'off'}",
                   level="muted")
        return 0
    ctx.print(server.render_banner() if hasattr(server, "render_banner") else
              f"MAgent API on http://{server.host}:{server.port} ({len(server.router)} routes)")
    if server.api_key:
        ctx.notice("API key required (http.api_key is set)", level="muted")
    else:
        ctx.notice("no API key configured; bind to loopback unless you trust the network",
                   level="warning")
    ctx.notice("press Ctrl-C to stop", level="muted")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


COMMANDS: tuple[Command, ...] = (
    Command(
        name="serve",
        help="Run the HTTP API and web UI",
        run=serve_command,
        group="server",
        needs_backend=True,
        configure=_configure_serve,
        description=(
            "Starts a dependency-free HTTP server exposing the same tools, memory, "
            "retrieval, sessions and metrics the CLI uses, plus a single-page web UI. "
            "Use --background to start it from a script."
        ),
        examples=("magent serve", "magent serve --host 0.0.0.0 --port 8080",
                  "magent serve --port 0 --background"),
    ),
)


__all__ = [
    "COMMANDS",
    "serve_command",
]
