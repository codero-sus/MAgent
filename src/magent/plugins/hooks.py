"""Plugin hooks: named extension points with ordered, isolated handlers.

A plugin rarely only *adds tools*; it wants to see the prompt before it is sent,
tag a tool result, or notice when a run failed.  :class:`HookRegistry` provides
those extension points.

Handlers are called in priority order and are isolated: one that raises is
logged and skipped, so a broken plugin cannot take down a run.  Handlers that
want to change data return a new value (or ``None`` to leave it alone), which
keeps the flow explicit - ``payload = hooks.apply("before_prompt", payload)``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..logging_ import get_logger
from .manifest import HOOK_POINTS

LOGGER = get_logger("plugins.hooks")
"""Logger for hook failures."""

DEFAULT_PRIORITY = 100
"""Priority for handlers that do not state one."""

MAX_HANDLERS_PER_HOOK = 64
"""Ceiling on handlers per hook point, so a loop cannot register forever."""


def _accepts_positional(fn: Callable[..., Any]) -> bool:
    """True when ``fn`` takes at least one positional parameter (the payload)."""
    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
    return False


@dataclass(slots=True)
class HookHandler:
    """One registered handler."""

    point: str
    fn: Callable[..., Any]
    priority: int = DEFAULT_PRIORITY
    name: str = ""
    plugin: str = ""
    calls: int = 0
    errors: int = 0
    total_seconds: float = 0.0
    once: bool = False
    accepts_payload: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Default the handler name and work out how to call it."""
        if not self.name:
            self.name = getattr(self.fn, "__name__", type(self.fn).__name__)
        self.priority = int(self.priority)
        self.accepts_payload = _accepts_positional(self.fn)

    @property
    def key(self) -> str:
        """Identifier used in logs and reports."""
        return f"{self.plugin or 'anonymous'}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the handler."""
        return {
            "point": self.point,
            "name": self.name,
            "plugin": self.plugin,
            "priority": self.priority,
            "calls": self.calls,
            "errors": self.errors,
            "seconds": round(self.total_seconds, 5),
            "once": self.once,
            "metadata": dict(self.metadata),
        }


class HookRegistry:
    """Registered handlers per hook point."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Create an empty registry.

        Args:
            enabled: when false every :meth:`apply` is a no-op
        """
        self.enabled = bool(enabled)
        self._handlers: dict[str, list[HookHandler]] = {}
        self.errors: list[str] = []

    def register(
        self,
        point: str,
        fn: Callable[..., Any],
        *,
        priority: int = DEFAULT_PRIORITY,
        name: str = "",
        plugin: str = "",
        once: bool = False,
        **metadata: Any,
    ) -> HookHandler:
        """Register a handler.

        Args:
            point: one of :data:`~magent.plugins.manifest.HOOK_POINTS`
            fn: callable receiving the payload (and any extra keyword arguments)
            priority: lower runs first
            name: handler name for logs
            plugin: owning plugin name
            once: remove the handler after its first successful call
            **metadata: extra values stored on the handler

        Raises:
            PluginError: when the hook point is unknown
        """
        from ..errors import PluginError

        if point not in HOOK_POINTS:
            raise PluginError(f"unknown hook point {point!r}", details={"known": list(HOOK_POINTS)})
        bucket = self._handlers.setdefault(point, [])
        if len(bucket) >= MAX_HANDLERS_PER_HOOK:
            raise PluginError(
                f"too many handlers for hook {point!r}", details={"limit": MAX_HANDLERS_PER_HOOK}
            )
        handler = HookHandler(
            point=point,
            fn=fn,
            priority=priority,
            name=name,
            plugin=plugin,
            once=once,
            metadata=dict(metadata),
        )
        bucket.append(handler)
        bucket.sort(key=lambda item: (item.priority, item.key))
        return handler

    def register_many(
        self, points: Iterable[str], fn: Callable[..., Any], **kwargs: Any
    ) -> list[HookHandler]:
        """Register the same handler on several hook points."""
        return [self.register(point, fn, **kwargs) for point in points]

    def handlers(self, point: str = "") -> list[HookHandler]:
        """Handlers for one point, or all of them when ``point`` is empty."""
        if point:
            return list(self._handlers.get(point, []))
        out: list[HookHandler] = []
        for bucket in self._handlers.values():
            out.extend(bucket)
        return out

    def points(self) -> list[str]:
        """Hook points that have at least one handler."""
        return sorted(point for point, bucket in self._handlers.items() if bucket)

    def apply(self, point: str, payload: Any = None, **context: Any) -> Any:
        """Run every handler for ``point``, threading the payload through them.

        Args:
            point: the hook point
            payload: value passed to each handler
            **context: extra keyword arguments (``agent``, ``run_id``, ...)

        Returns:
            the payload after every handler has had a chance to change it
        """
        if not self.enabled:
            return payload
        handlers = self._handlers.get(point)
        if not handlers:
            return payload
        current = payload
        for handler in list(handlers):
            started = time.perf_counter()
            try:
                # The calling convention is decided once, at registration, so a
                # TypeError raised *inside* a handler is not retried as a
                # signature mismatch.
                result = (
                    handler.fn(current, **context)
                    if handler.accepts_payload
                    else handler.fn(**context)
                )
            except Exception as exc:
                self._record_error(handler, exc, point)
                continue
            handler.calls += 1
            handler.total_seconds += time.perf_counter() - started
            if result is not None:
                current = result
            if handler.once:
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass
        return current

    def _record_error(self, handler: HookHandler, exc: BaseException, point: str) -> None:
        """Note a failing handler without interrupting the caller."""
        handler.errors += 1
        message = f"{point}:{handler.key} -> {type(exc).__name__}: {exc}"
        self.errors.append(message)
        if len(self.errors) > 200:
            del self.errors[:-200]
        LOGGER.warning("hook %s failed: %s", handler.key, exc)

    def collect(self, point: str, **context: Any) -> list[Any]:
        """Run handlers expecting each to return a value, collecting non-None results."""
        out: list[Any] = []
        for handler in list(self._handlers.get(point, [])):
            try:
                value = (
                    handler.fn(None, **context)
                    if handler.accepts_payload
                    else handler.fn(**context)
                )
            except Exception as exc:
                self._record_error(handler, exc, point)
                continue
            handler.calls += 1
            if value is not None:
                out.append(value)
        return out

    def unregister(self, point: str, name: str) -> bool:
        """Remove every handler with this name from a hook point."""
        bucket = self._handlers.get(point)
        if not bucket:
            return False
        before = len(bucket)
        self._handlers[point] = [item for item in bucket if item.name != name]
        return len(self._handlers[point]) != before

    def clear(self, point: str = "") -> int:
        """Remove handlers, from one point or all of them."""
        if point:
            return len(self._handlers.pop(point, []))
        removed = sum(len(bucket) for bucket in self._handlers.values())
        self._handlers.clear()
        return removed

    def stats(self) -> dict[str, Any]:
        """Per-point handler counts and error totals."""
        return {
            "enabled": self.enabled,
            "points": {point: len(bucket) for point, bucket in sorted(self._handlers.items())},
            "handlers": sum(len(bucket) for bucket in self._handlers.values()),
            "calls": sum(item.calls for item in self.handlers()),
            "errors": sum(item.errors for item in self.handlers()),
            "recent_errors": self.errors[-5:],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise every handler."""
        return {
            "points": {
                point: [item.to_dict() for item in bucket]
                for point, bucket in sorted(self._handlers.items())
            },
            "stats": self.stats(),
        }

    def __len__(self) -> int:
        """Number of registered handlers."""
        return sum(len(bucket) for bucket in self._handlers.values())

    def __contains__(self, point: object) -> bool:
        """True when a hook point has handlers."""
        return bool(self._handlers.get(str(point)))

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"HookRegistry(handlers={len(self)}, points={len(self.points())})"


def attach_to_agent(hooks: HookRegistry, agent: Any) -> Any:
    """Wire hook points that the agent exposes directly.

    Not every hook point has an agent-side counterpart; the ones that do
    (``before_run``/``after_run``/``on_error``) are subscribed to the event bus,
    which is how plugins observe runs without patching the agent.
    """
    bus = getattr(agent, "events", None)
    if bus is None or not hasattr(bus, "subscribe"):
        return None

    def on_event(event: Any) -> None:
        kind = str(getattr(getattr(event, "kind", ""), "value", getattr(event, "kind", "")))
        data = dict(getattr(event, "data", {}) or {})
        if kind == "run.started":
            hooks.apply(
                "before_run", {"agent": agent, **data}, agent=agent, run_id=data.get("run_id", "")
            )
        elif kind == "run.finished":
            hooks.apply(
                "after_run", {"agent": agent, **data}, agent=agent, run_id=data.get("run_id", "")
            )
        elif kind == "run.failed":
            hooks.apply(
                "on_error", {"agent": agent, **data}, agent=agent, run_id=data.get("run_id", "")
            )
        hooks.apply("on_event", event, agent=agent)

    return bus.subscribe(on_event, name="plugin-hooks", priority=300)


def chain_handlers(handlers: Sequence[HookHandler]) -> Callable[[Any], Any]:
    """Compose several handlers into one callable, in priority order."""
    ordered = sorted(handlers, key=lambda item: (item.priority, item.key))

    def composed(payload: Any = None, **context: Any) -> Any:
        current = payload
        for handler in ordered:
            result = (
                handler.fn(current, **context) if handler.accepts_payload else handler.fn(**context)
            )
            if result is not None:
                current = result
        return current

    composed.__name__ = "chained_hooks"
    composed.__doc__ = f"Chained handler over {len(ordered)} hook handlers."
    return composed


def describe_hooks(hooks: HookRegistry) -> list[tuple[str, str, str]]:
    """Rows of ``(point, handler, priority)`` for CLI tables."""
    rows: list[tuple[str, str, str]] = []
    for handler in hooks.handlers():
        rows.append((handler.point, handler.key, str(handler.priority)))
    return sorted(rows)


def load_hooks_from(mapping: Mapping[str, Any], *, plugin: str = "") -> HookRegistry:
    """Build a registry from a ``{hook_point: callable}`` mapping."""
    hooks = HookRegistry()
    for point, fn in mapping.items():
        hooks.register(str(point), fn, plugin=plugin)
    return hooks


__all__ = [
    "DEFAULT_PRIORITY",
    "MAX_HANDLERS_PER_HOOK",
    "HookHandler",
    "HookRegistry",
    "attach_to_agent",
    "chain_handlers",
    "describe_hooks",
    "load_hooks_from",
]
