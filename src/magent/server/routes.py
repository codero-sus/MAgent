"""Routing for the embedded HTTP server.

Small, explicit and dependency-free: a :class:`Route` is a method plus a path
template (``/api/sessions/{id}``), and :class:`Router` resolves a request to a
handler, returning the captured path parameters.  The router is strict about
methods (``405`` with an ``Allow`` header) and distinguishes "no such path"
from "path exists for another method", which is what makes the API pleasant to
debug with ``curl``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

ROUTE_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD")
"""HTTP methods the router knows about."""

PARAM_RE = re.compile(r"\{([A-Za-z_][\w]*)\}")
"""Path parameter matcher."""


@dataclass(slots=True)
class Route:
    """One method plus path template."""

    method: str
    path: str
    handler: Callable[..., Any]
    name: str = ""
    description: str = ""
    auth: bool = True
    stream: bool = False

    def __post_init__(self) -> None:
        """Normalise the method and default the route name."""
        self.method = self.method.upper()
        if not self.path.startswith("/"):
            self.path = "/" + self.path
        if not self.name:
            self.name = f"{self.method.lower()}{self.path.replace('/', '_').replace('{', '').replace('}', '')}"

    @property
    def parameters(self) -> list[str]:
        """Names of the path parameters."""
        return PARAM_RE.findall(self.path)

    @property
    def pattern(self) -> re.Pattern[str]:
        """Compiled regular expression matching this route."""
        escaped = PARAM_RE.sub(
            r"(?P<\1>[^/]+)", re.escape(self.path).replace(r"\{", "{").replace(r"\}", "}")
        )
        return re.compile("^" + escaped + "/?$")

    def match(self, method: str, path: str) -> dict[str, str] | None:
        """Return the path parameters when this route handles the request."""
        if self.method != method.upper() and not (
            self.method == "GET" and method.upper() == "HEAD"
        ):
            return None
        match = self.pattern.match(path or "/")
        return match.groupdict() if match else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the route for the API index."""
        return {
            "method": self.method,
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "auth": self.auth,
            "stream": self.stream,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class RouteMatch:
    """Outcome of resolving a request."""

    route: Route | None = None
    params: dict[str, str] = field(default_factory=dict)
    allowed: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        """True when a route matched."""
        return self.route is not None

    @property
    def method_not_allowed(self) -> bool:
        """True when the path exists but not for this method."""
        return self.route is None and bool(self.allowed)


class Router:
    """A method-and-path routing table."""

    def __init__(self) -> None:
        """Create an empty router."""
        self.routes: list[Route] = []

    def add(self, method: str, path: str, handler: Callable[..., Any], **kwargs: Any) -> Route:
        """Register a route."""
        route = Route(method=method, path=path, handler=handler, **kwargs)
        self.routes.append(route)
        return route

    def get(self, path: str, handler: Callable[..., Any], **kwargs: Any) -> Route:
        """Register a GET route."""
        return self.add("GET", path, handler, **kwargs)

    def post(self, path: str, handler: Callable[..., Any], **kwargs: Any) -> Route:
        """Register a POST route."""
        return self.add("POST", path, handler, **kwargs)

    def delete(self, path: str, handler: Callable[..., Any], **kwargs: Any) -> Route:
        """Register a DELETE route."""
        return self.add("DELETE", path, handler, **kwargs)

    def resolve(self, method: str, path: str) -> RouteMatch:
        """Find the route handling a request.

        Returns a :class:`RouteMatch`; when the path matches no route but other
        methods do, ``allowed`` lists them so the caller can answer ``405``.
        """
        allowed: list[str] = []
        for route in self.routes:
            params = route.match(method, path)
            if params is not None:
                return RouteMatch(route=route, params=params)
            if route.pattern.match(path or "/"):
                allowed.append(route.method)
        return RouteMatch(allowed=tuple(sorted(set(allowed))))

    def paths(self) -> list[str]:
        """Every registered path, sorted."""
        return sorted({route.path for route in self.routes})

    def describe(self) -> list[dict[str, Any]]:
        """Serialise every route, for ``GET /api``."""
        return [
            route.to_dict()
            for route in sorted(self.routes, key=lambda item: (item.path, item.method))
        ]

    def iter_routes(self) -> Iterable[Route]:
        """Iterate the registered routes."""
        return iter(self.routes)

    def __len__(self) -> int:
        """Number of routes."""
        return len(self.routes)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"Router(routes={len(self.routes)})"


def build_router(routes: Sequence[tuple[str, str, Callable[..., Any]]]) -> Router:
    """Build a router from ``(method, path, handler)`` triples."""
    router = Router()
    for method, path, handler in routes:
        router.add(method, path, handler)
    return router


__all__ = [
    "PARAM_RE",
    "ROUTE_METHODS",
    "Route",
    "RouteMatch",
    "Router",
    "build_router",
]
