"""Model and backend routing with failover.

A router sits between the agent and one or more backends and decides, per
request, which model on which backend should serve it. Decisions can be based
on:

* **capability** - embeddings go to an embedding model, vision to a VLM,
* **priority** - a primary host with ordered fallbacks,
* **health** - backends with an open circuit breaker are skipped,
* **cost/latency** - cheap models for simple turns, big models for planning,
* **load** - round-robin or least-recently-used across equivalent hosts.

The router is transparent: it implements :class:`~magent.llm.base.LLMBackend`,
so an agent does not know or care that failover is happening.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import (
    BackendError,
    BackendUnavailableError,
    CapabilityNotSupportedError,
    ModelNotFoundError,
)
from ..types import Capability, ChatResponse, ModelInfo
from .base import BaseBackend, CompletionRequest
from .retry import CircuitBreaker

STRATEGIES = ("priority", "round_robin", "least_used", "fastest", "cheapest", "random")
"""Supported route selection strategies."""


@dataclass(slots=True)
class Route:
    """One candidate backend/model pair with routing metadata."""

    backend: Any
    model: str = ""
    priority: int = 100
    weight: float = 1.0
    capabilities: list[str] = field(default_factory=list)
    max_tokens_per_minute: float = 0.0
    cost_weight: float = 1.0
    latency_ms: float = 1000.0
    enabled: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        """Derive a display label when none was given."""
        if not self.label:
            name = getattr(self.backend, "name", "backend")
            self.label = f"{name}:{self.model or 'default'}"

    def supports(self, capability: Capability | str) -> bool:
        """Report whether this route advertises a capability."""
        if not self.capabilities:
            return True
        wanted = capability.value if isinstance(capability, Capability) else str(capability)
        declared = {str(item).lower() for item in self.capabilities}
        if wanted.lower() in declared:
            return True
        parsed = Capability.parse(wanted)
        return parsed is not None and parsed.value in declared

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing metadata, never the backend object."""
        return {
            "label": self.label,
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "model": self.model,
            "priority": self.priority,
            "weight": self.weight,
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "cost_weight": self.cost_weight,
            "latency_ms": self.latency_ms,
        }


@dataclass(slots=True)
class RouteAttempt:
    """A record of one failover attempt, kept for diagnostics."""

    label: str
    ok: bool
    error: str = ""
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attempt."""
        return {
            "label": self.label,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


class ModelRouter(BaseBackend):
    """Route requests across backends with automatic failover."""

    name = "router"

    def __init__(
        self,
        routes: list[Route] | None = None,
        *,
        strategy: str = "priority",
        default_model: str = "",
        breaker_threshold: int = 4,
        breaker_reset: float = 30.0,
        max_failovers: int = 3,
        on_failover: Callable[[RouteAttempt], None] | None = None,
        stickiness: float = 0.0,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown routing strategy {strategy!r}; choose from {STRATEGIES}")
        super().__init__(default_model=default_model)
        self.routes: list[Route] = list(routes or [])
        self.strategy = strategy
        self.max_failovers = max(0, max_failovers)
        self.on_failover = on_failover
        self.stickiness = stickiness
        self._lock = threading.RLock()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_threshold = breaker_threshold
        self._breaker_reset = breaker_reset
        self._usage_counts: dict[str, int] = defaultdict(int)
        self._latency_ewma: dict[str, float] = {}
        self._round_robin_index = 0
        self._sticky_label: str | None = None
        self._sticky_until = 0.0
        self.failovers = 0
        self.selections = 0
        self.last_attempts: list[RouteAttempt] = []

    # -- route management ----------------------------------------------------

    def add_route(self, route: Route) -> ModelRouter:
        """Register a route and return ``self`` for chaining."""
        with self._lock:
            self.routes.append(route)
            self.routes.sort(key=lambda item: (item.priority, item.label))
        return self

    def add_backend(
        self,
        backend: Any,
        *,
        model: str = "",
        priority: int = 100,
        capabilities: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelRouter:
        """Convenience wrapper building a :class:`Route` from a backend."""
        return self.add_route(
            Route(
                backend=backend,
                model=model,
                priority=priority,
                capabilities=list(capabilities or []),
                **kwargs,
            )
        )

    def remove_route(self, label: str) -> bool:
        """Remove a route by label; report whether it existed."""
        with self._lock:
            before = len(self.routes)
            self.routes = [route for route in self.routes if route.label != label]
            return len(self.routes) < before

    def enable_route(self, label: str, enabled: bool = True) -> bool:
        """Enable or disable a route without removing it."""
        for route in self.routes:
            if route.label == label:
                route.enabled = enabled
                return True
        return False

    def route_labels(self) -> list[str]:
        """Return labels of all registered routes."""
        return [route.label for route in self.routes]

    def breaker_for(self, label: str) -> CircuitBreaker:
        """Return (creating if needed) the breaker guarding one route."""
        with self._lock:
            breaker = self._breakers.get(label)
            if breaker is None:
                breaker = CircuitBreaker(
                    failure_threshold=self._breaker_threshold,
                    reset_timeout=self._breaker_reset,
                    name=label,
                )
                self._breakers[label] = breaker
            return breaker

    # -- selection -----------------------------------------------------------

    def candidates(
        self, *, capability: Capability | str = Capability.CHAT, model: str = ""
    ) -> list[Route]:
        """Return enabled routes that can serve the request, best first."""
        with self._lock:
            pool = [route for route in self.routes if route.enabled]
        if model:
            exact = [route for route in pool if route.model == model]
            if exact:
                pool = exact
        filtered = [route for route in pool if route.supports(capability)]
        if not filtered:
            filtered = [route for route in pool if not route.capabilities]
        available = [
            route for route in filtered if self.breaker_for(route.label).state.allows_traffic()
        ]
        ordered = available or filtered
        return self._order(ordered)

    def _order(self, routes: list[Route]) -> list[Route]:
        """Sort candidate routes according to the active strategy."""
        if not routes:
            return []
        strategy = self.strategy
        if strategy == "priority":
            return sorted(routes, key=lambda route: (route.priority, -route.weight, route.label))
        if strategy == "round_robin":
            with self._lock:
                start = self._round_robin_index % len(routes)
                self._round_robin_index = (self._round_robin_index + 1) % max(1, len(routes))
            return routes[start:] + routes[:start]
        if strategy == "least_used":
            return sorted(
                routes, key=lambda route: (self._usage_counts.get(route.label, 0), route.priority)
            )
        if strategy == "fastest":
            return sorted(
                routes, key=lambda route: self._latency_ewma.get(route.label, route.latency_ms)
            )
        if strategy == "cheapest":
            return sorted(routes, key=lambda route: (route.cost_weight, route.priority))
        if strategy == "random":
            import random

            weighted = [route for route in routes for _ in range(max(1, int(route.weight * 10)))]
            return [random.choice(weighted)] if weighted else routes
        return routes

    def select(self, *, capability: Capability | str = Capability.CHAT, model: str = "") -> Route:
        """Return the single best route, raising when none can serve."""
        if self.stickiness > 0 and self._sticky_label and time.monotonic() < self._sticky_until:
            for route in self.routes:
                if (
                    route.label == self._sticky_label
                    and route.enabled
                    and route.supports(capability)
                ):
                    return route
        options = self.candidates(capability=capability, model=model)
        if not options:
            if model:
                raise ModelNotFoundError(
                    f"No route can serve model {model!r} with capability {capability!s}.",
                    details={"routes": self.route_labels()},
                )
            raise CapabilityNotSupportedError(
                f"No route advertises capability {capability!s}.",
                details={"routes": [route.to_dict() for route in self.routes]},
            )
        choice = options[0]
        with self._lock:
            self._usage_counts[choice.label] += 1
            self.selections += 1
            if self.stickiness > 0:
                self._sticky_label = choice.label
                self._sticky_until = time.monotonic() + self.stickiness
        return choice

    # -- inference -----------------------------------------------------------

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Run a completion, failing over across routes as needed."""
        attempts: list[RouteAttempt] = []
        routes = self.candidates(model=request.model)
        if not routes:
            routes = [self.select(model=request.model)]
        limit = min(len(routes), self.max_failovers + 1)
        last_error: BaseException | None = None
        for index, route in enumerate(routes[:limit]):
            started = time.perf_counter()
            breaker = self.breaker_for(route.label)
            if not breaker.allow_request():
                attempts.append(RouteAttempt(label=route.label, ok=False, error="circuit open"))
                continue
            try:
                response = route.backend.complete(request.clone(model=route.model or request.model))
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000.0
                breaker.record_failure()
                attempt = RouteAttempt(
                    label=route.label, ok=False, error=str(exc)[:300], latency_ms=latency
                )
                attempts.append(attempt)
                last_error = exc
                self.failovers += 1
                if self.on_failover is not None:
                    self.on_failover(attempt)
                continue
            latency = (time.perf_counter() - started) * 1000.0
            breaker.record_success()
            self._update_latency(route.label, latency)
            attempts.append(RouteAttempt(label=route.label, ok=True, latency_ms=latency))
            self.last_attempts = attempts
            response.model = response.model or route.model
            response.raw = {**(response.raw or {}), "route": route.label, "attempts": len(attempts)}
            self.record(response)
            return response
        self.last_attempts = attempts
        if last_error is not None:
            raise last_error
        raise BackendUnavailableError(
            "Every configured route failed or its circuit breaker is open.",
            details={"attempts": [attempt.to_dict() for attempt in attempts]},
        )

    def stream(self, request: CompletionRequest):
        """Stream from the best route; failover only before the first chunk."""
        routes = self.candidates(model=request.model)
        if not routes:
            routes = [self.select(model=request.model)]
        last_error: BaseException | None = None
        for index, route in enumerate(routes[: self.max_failovers + 1]):
            breaker = self.breaker_for(route.label)
            if not breaker.allow_request():
                continue
            try:
                iterator = route.backend.stream(request.clone(model=route.model or request.model))
                first = next(iterator, None)
            except Exception as exc:
                breaker.record_failure()
                last_error = exc
                self.failovers += 1
                continue
            breaker.record_success()
            self._usage_counts[route.label] += 1
            if first is not None:
                yield first
            for chunk in iterator:
                yield chunk
            return
        if last_error is not None:
            raise last_error
        raise BackendUnavailableError("No route was able to start a stream.")

    def _update_latency(self, label: str, latency_ms: float) -> None:
        """Fold a latency sample into the route's EWMA."""
        with self._lock:
            previous = self._latency_ewma.get(label)
            alpha = 0.3
            self._latency_ewma[label] = (
                latency_ms if previous is None else alpha * latency_ms + (1 - alpha) * previous
            )

    # -- discovery -----------------------------------------------------------

    def models(self) -> list[ModelInfo]:
        """Aggregate model lists from every enabled route."""
        seen: dict[str, ModelInfo] = {}
        for route in self.routes:
            if not route.enabled:
                continue
            try:
                discovered = route.backend.models()
            except BackendError:
                continue
            except Exception:
                continue
            for info in discovered:
                if route.model and info.id != route.model:
                    continue
                if info.id in seen:
                    existing = seen[info.id]
                    merged = set(existing.capabilities) | set(info.capabilities)
                    existing.capabilities = sorted(merged)
                else:
                    info.metadata["route"] = route.label
                    info.status = (
                        "running"
                        if self.breaker_for(route.label).state.allows_traffic()
                        else "degraded"
                    )
                    seen[info.id] = info
        return list(seen.values())

    def health(self) -> dict[str, Any]:
        """Return per-route health without failing on an unreachable backend."""
        results: dict[str, Any] = {}
        for route in self.routes:
            try:
                results[route.label] = route.backend.health()
            except Exception as exc:
                results[route.label] = {"ok": False, "error": str(exc)[:200]}
            results[route.label]["breaker"] = self.breaker_for(route.label).snapshot()
        healthy = sum(1 for value in results.values() if value.get("ok"))
        return {
            "backend": self.name,
            "strategy": self.strategy,
            "routes": len(self.routes),
            "healthy": healthy,
            "ok": healthy > 0,
            "details": results,
        }

    def stats(self) -> dict[str, Any]:
        """Return router telemetry."""
        base = super().stats()
        base.update(
            {
                "strategy": self.strategy,
                "routes": [route.to_dict() for route in self.routes],
                "selections": self.selections,
                "failovers": self.failovers,
                "usage_counts": dict(self._usage_counts),
                "latency_ewma_ms": {k: round(v, 2) for k, v in self._latency_ewma.items()},
                "breakers": {
                    label: breaker.snapshot() for label, breaker in self._breakers.items()
                },
                "last_attempts": [attempt.to_dict() for attempt in self.last_attempts],
            }
        )
        return base

    def close(self) -> None:
        """Close every distinct backend exactly once."""
        closed: set[int] = set()
        for route in self.routes:
            key = id(route.backend)
            if key in closed:
                continue
            closed.add(key)
            try:
                route.backend.close()
            except Exception:
                pass


def route_from_backend(
    backend: Any,
    *,
    model: str = "",
    priority: int = 100,
    capabilities: list[str] | None = None,
    **kwargs: Any,
) -> Route:
    """Build a :class:`Route` from a backend instance."""
    return Route(
        backend=backend,
        model=model,
        priority=priority,
        capabilities=list(capabilities or []),
        **kwargs,
    )


def single_backend_router(backend: Any, *, model: str = "") -> ModelRouter:
    """Wrap one backend in a router so callers always get failover semantics."""
    router = ModelRouter(default_model=model)
    router.add_backend(backend, model=model, priority=1)
    return router


__all__ = [
    "STRATEGIES",
    "ModelRouter",
    "Route",
    "RouteAttempt",
    "route_from_backend",
    "single_backend_router",
]
