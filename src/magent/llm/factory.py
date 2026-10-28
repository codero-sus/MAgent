"""Backend factory: turn configuration into a working LLM backend.

The factory is the only place that decides *which* backend MAgent uses, based on
explicit configuration, environment variables, and - when nothing is configured -
a live probe of the default Cortex LLMHoster address.

Resolution order:

1. ``backend`` explicitly set in config or on the CLI (``cortex`` | ``echo`` |
   ``scripted`` | ``router`` | ``openai``).
2. ``MAGENT_BACKEND`` environment variable.
3. Automatic: use Cortex when it answers ``/health``, otherwise fall back to the
   offline echo backend so MAgent still runs (and says why).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError
from .base import BaseBackend, CompletionRequest, LLMBackend, StreamChunk, collect_stream
from .cortex import DEFAULT_CORTEX_URL, CortexBackend, CortexSettings
from .echo import EchoBackend, LambdaBackend, ScriptedBackend, ToolScriptBackend
from .router import ModelRouter, Route

BACKEND_CHOICES = ("auto", "cortex", "echo", "scripted", "router", "lambda")
"""Valid values for the ``backend`` configuration key."""


@dataclass(slots=True)
class BackendRequest:
    """A declarative description of the backend the caller wants."""

    kind: str = "auto"
    model: str = ""
    embedding_model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: float = 120.0
    api_prefix: str = "/v1"
    verify_ssl: bool = True
    max_retries: int = 3
    fallback_to_echo: bool = True
    probe_timeout: float = 3.0
    cache_dir: str = ""
    cache_ttl: float = 3600.0
    cache_enabled: bool = True
    routes: list[dict[str, Any]] | None = None
    router_strategy: str = "priority"

    @classmethod
    def from_config(cls, config: Any) -> BackendRequest:
        """Build a request from a loaded :class:`~magent.config.Config`.

        The request is assembled from MAgent's real configuration sections -
        ``[model]``, ``[server]`` and ``[cache]`` - because a request derived from
        anywhere else would silently ignore the file the user actually edited.
        """
        llm = getattr(config, "llm", None)
        model = getattr(config, "model", llm)
        server = getattr(config, "server", llm)
        cache = getattr(config, "cache", llm)
        if model is None and server is None and cache is None:
            return cls()
        base_url = str(getattr(server, "base_url", "") or getattr(llm, "base_url", "") or "")
        return cls(
            kind=str(getattr(model, "backend", "auto") or "auto"),
            model=str(getattr(model, "model", "") or ""),
            embedding_model=str(getattr(model, "embedding_model", "") or ""),
            base_url=base_url,
            api_key=str(getattr(server, "api_key", "") or ""),
            timeout=float(getattr(server, "timeout", 120.0) or 120.0),
            api_prefix=str(getattr(server, "api_prefix", "/v1") or "/v1"),
            verify_ssl=bool(getattr(server, "verify_ssl", True)),
            max_retries=int(getattr(server, "max_retries", 3) or 3),
            fallback_to_echo=bool(getattr(model, "fallback_to_echo", True)),
            probe_timeout=float(getattr(server, "probe_timeout", 3.0) or 3.0),
            cache_dir=str(getattr(cache, "directory", "") or ""),
            cache_ttl=float(getattr(cache, "ttl", 3600.0) or 3600.0),
            cache_enabled=bool(getattr(cache, "enabled", True)),
            routes=list(getattr(model, "routes", []) or []),
            router_strategy=str(getattr(model, "router_strategy", "priority") or "priority"),
        )

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> BackendRequest:
        """Build a request purely from environment variables."""
        env = dict(os.environ if environ is None else environ)
        settings = CortexSettings.from_env(env)
        return cls(
            kind=env.get("MAGENT_BACKEND", "auto").strip().lower() or "auto",
            model=settings.default_model,
            embedding_model=settings.embedding_model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            api_prefix=settings.api_prefix,
            verify_ssl=settings.verify_ssl,
            max_retries=settings.max_retries,
            cache_dir=env.get("MAGENT_CACHE_DIR", ""),
            cache_enabled=env.get("MAGENT_CACHE_ENABLED", "true").lower()
            not in {"0", "false", "no"},
        )


@dataclass(slots=True)
class BackendDecision:
    """What the factory actually built, and why."""

    requested: str
    resolved: str
    reason: str
    probed: bool = False
    probe_ok: bool | None = None
    backend_name: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision for ``magent doctor`` output."""
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "reason": self.reason,
            "probed": self.probed,
            "probe_ok": self.probe_ok,
            "backend_name": self.backend_name,
            "model": self.model,
        }


class BackendFactory:
    """Builds and caches LLM backends from declarative requests."""

    def __init__(self, *, probe: bool = True) -> None:
        self.probe = probe
        self.last_decision: BackendDecision | None = None
        self.built = 0

    def build(self, request: BackendRequest | None = None, **overrides: Any) -> BaseBackend:
        """Construct the backend described by ``request``."""
        spec = request or BackendRequest.from_env()
        if overrides:
            for key, value in overrides.items():
                if value is not None and hasattr(spec, key):
                    setattr(spec, key, value)
        kind = (spec.kind or "auto").strip().lower()
        if kind not in BACKEND_CHOICES:
            raise ConfigError(
                f"Unknown backend {kind!r}; choose from {', '.join(BACKEND_CHOICES)}."
            )
        if kind == "auto":
            backend, decision = self._build_auto(spec)
        elif kind == "cortex":
            backend = self._build_cortex(spec)
            decision = BackendDecision(
                requested=kind,
                resolved="cortex",
                reason="explicitly requested",
                backend_name=backend.name,
                model=backend.default_model,
            )
        elif kind == "echo":
            backend = EchoBackend(models_list=[spec.model or "echo-small"])
            decision = BackendDecision(
                requested=kind,
                resolved="echo",
                reason="explicitly requested (offline)",
                backend_name=backend.name,
                model=backend.default_model,
            )
        elif kind == "scripted":
            backend = ScriptedBackend(models_list=[spec.model or "scripted-model"])
            decision = BackendDecision(
                requested=kind,
                resolved="scripted",
                reason="explicitly requested (offline)",
                backend_name=backend.name,
                model=backend.default_model,
            )
        elif kind == "lambda":
            backend = LambdaBackend(
                lambda req: "lambda backend reply", model=spec.model or "lambda-model"
            )
            decision = BackendDecision(
                requested=kind,
                resolved="lambda",
                reason="explicitly requested (offline)",
                backend_name=backend.name,
                model=backend.default_model,
            )
        else:
            backend, decision = self._build_router(spec)
        self.last_decision = decision
        self.built += 1
        return backend

    # -- internals -----------------------------------------------------------

    def _build_cortex(self, spec: BackendRequest) -> CortexBackend:
        """Construct the Cortex backend from a request spec."""
        settings = CortexSettings.from_env()
        if spec.base_url:
            settings.base_url = spec.base_url
        if spec.api_key:
            settings.api_key = spec.api_key
        if spec.model:
            settings.default_model = spec.model
        if spec.embedding_model:
            settings.embedding_model = spec.embedding_model
        if spec.timeout:
            settings.timeout = spec.timeout
        if spec.api_prefix:
            settings.api_prefix = spec.api_prefix
        settings.verify_ssl = spec.verify_ssl
        settings.max_retries = spec.max_retries
        return CortexBackend(settings)

    def _build_auto(self, spec: BackendRequest) -> tuple[BaseBackend, BackendDecision]:
        """Probe Cortex and fall back to echo when it is unreachable."""
        if spec.routes and len(spec.routes) > 1:
            return self._build_router(spec)
        url = spec.base_url or os.environ.get("MAGENT_CORTEX_URL") or DEFAULT_CORTEX_URL
        reachable = None
        if self.probe:
            reachable = _probe_cortex(url, timeout=spec.probe_timeout)
        if reachable is not False:
            backend = self._build_cortex(spec)
            backend.default_model = spec.model
            return backend, BackendDecision(
                requested="auto",
                resolved="cortex",
                reason=(
                    "Cortex LLMHoster answered its health probe"
                    if reachable
                    else "probing disabled; assuming Cortex LLMHoster"
                ),
                probed=self.probe,
                probe_ok=reachable,
                backend_name=backend.name,
                model=spec.model,
            )
        if not spec.fallback_to_echo:
            from ..errors import BackendUnavailableError

            raise BackendUnavailableError(
                f"Cortex LLMHoster is not reachable at {url}. Start it, or set "
                "MAGENT_BACKEND=echo for offline use."
            )
        backend = EchoBackend(models_list=[spec.model or "echo-small"])
        return backend, BackendDecision(
            requested="auto",
            resolved="echo",
            reason=f"Cortex LLMHoster did not answer at {url}; using the offline echo backend",
            probed=True,
            probe_ok=False,
            backend_name=backend.name,
            model=backend.default_model,
        )

    def _build_router(self, spec: BackendRequest) -> tuple[BaseBackend, BackendDecision]:
        """Construct a multi-route router from the request spec."""
        router = ModelRouter(strategy=spec.router_strategy, default_model=spec.model)
        for index, entry in enumerate(spec.routes or []):
            kind = str(entry.get("backend", "cortex")).lower()
            priority = int(entry.get("priority", 100 + index))
            if kind == "echo":
                backend: Any = EchoBackend(models_list=[str(entry.get("model") or "echo-small")])
            elif kind == "scripted":
                backend = ScriptedBackend(script=list(entry.get("script") or []))
            else:
                sub = BackendRequest(
                    kind="cortex",
                    model=str(entry.get("model") or ""),
                    base_url=str(entry.get("base_url") or spec.base_url or ""),
                    api_key=str(entry.get("api_key") or spec.api_key or ""),
                    timeout=float(entry.get("timeout") or spec.timeout),
                )
                backend = self._build_cortex(sub)
            router.add_route(
                Route(
                    backend=backend,
                    model=str(entry.get("model") or ""),
                    priority=priority,
                    capabilities=[str(item) for item in (entry.get("capabilities") or [])],
                    weight=float(entry.get("weight") or 1.0),
                    cost_weight=float(entry.get("cost_weight") or 1.0),
                    label=str(entry.get("label") or ""),
                )
            )
        if not router.routes:
            backend = self._build_cortex(spec)
            router.add_route(Route(backend=backend, model=spec.model, priority=1))
        return router, BackendDecision(
            requested=spec.kind,
            resolved="router",
            reason=f"{len(router.routes)} route(s) configured",
            backend_name=router.name,
            model=spec.model,
        )


def _probe_cortex(url: str, *, timeout: float = 3.0) -> bool:
    """Return whether Cortex answers ``GET /health`` at ``url``."""
    from ..http.client import HTTPClient
    from ..utils.urls import join_url, normalize_base_url

    client = HTTPClient(timeout=timeout, headers={"User-Agent": "MAgent-probe"})
    try:
        response = client.get(
            join_url(normalize_base_url(url), "/health"), raise_for_status=False, max_attempts=1
        )
        return response.status < 500
    except Exception:
        return False
    finally:
        client.close()


def build_backend(config: Any = None, *, offline: bool | None = None, **kwargs: Any) -> BaseBackend:
    """Build a backend from configuration plus keyword overrides.

    Precedence: an explicit ``config`` wins over the environment, because a user
    who edited ``magent.toml`` means it.  ``offline=True`` forces the echo
    backend, which is what ``--offline`` and the test suite use.

    Args:
        config: a loaded :class:`~magent.config.Config`
        offline: force the dependency-free echo backend
        **kwargs: :class:`BackendRequest` fields, e.g. ``kind`` or ``model``
    """
    if offline:
        return build_offline_backend(kwargs.get("model") or "echo-small")
    request = (
        BackendRequest.from_config(config) if config is not None else BackendRequest.from_env()
    )
    return BackendFactory().build(request, **kwargs)


def build_offline_backend(model: str = "echo-small") -> EchoBackend:
    """Build the offline echo backend, used by tests and ``--offline``."""
    return EchoBackend(models_list=[model])


__all__ = [
    "BACKEND_CHOICES",
    "BackendDecision",
    "BackendFactory",
    "BackendRequest",
    "BaseBackend",
    "CompletionRequest",
    "CortexBackend",
    "CortexSettings",
    "EchoBackend",
    "LLMBackend",
    "LambdaBackend",
    "ModelRouter",
    "Route",
    "ScriptedBackend",
    "StreamChunk",
    "ToolScriptBackend",
    "build_backend",
    "build_offline_backend",
    "collect_stream",
]
