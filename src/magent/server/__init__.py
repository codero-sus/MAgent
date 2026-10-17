"""The embedded HTTP server and bundled web UI.

Public API:

:class:`MagentServer`
    binds a ``ThreadingHTTPServer``, exposes the REST API and serves the UI;
:class:`Router` / :class:`Route`
    the tiny dependency-free routing table behind it;
:class:`Request` / :class:`ServerStats`
    a parsed request and the request counters;
:func:`build_server` / :func:`free_port` / :func:`wait_for_health`
    helpers used by ``magent serve`` and the tests.

>>> from magent.server import MagentServer
>>> server = MagentServer()
>>> server.router.resolve("GET", "/api/health").found
True
"""

from __future__ import annotations

from .app import (DEFAULT_HOST, DEFAULT_PORT, INDEX_FILE, PUBLIC_ROUTES, SSE_HEADERS, STATIC_DIR,
                  MagentServer, Request, ServerStats, build_server, free_port, static_files,
                  wait_for_health)
from .routes import PARAM_RE, ROUTE_METHODS, Route, RouteMatch, Router, build_router

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "INDEX_FILE",
    "PARAM_RE",
    "PUBLIC_ROUTES",
    "ROUTE_METHODS",
    "SSE_HEADERS",
    "STATIC_DIR",
    "MagentServer",
    "Request",
    "Route",
    "RouteMatch",
    "Router",
    "ServerStats",
    "build_router",
    "build_server",
    "free_port",
    "static_files",
    "wait_for_health",
]
