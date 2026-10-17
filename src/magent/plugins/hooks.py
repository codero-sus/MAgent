"""Plugin hooks: the lifecycle points a plugin may attach to.

A hook registry is how a plugin extends behaviour without the framework knowing
its name. The design constraints come from having untrusted code on the other
end:

*hooks may not silently change control flow*
    a hook that raises is recorded and the run continues, because one broken
    plugin must not take down an agent that was working;

*transforms are explicit*
    ``before_tool`` and ``filter_output`` may return a replacement value, and
    that is the only way a hook changes anything - everything else observes;

*every call is metered*
    timings and failure counts are kept per hook, so a plugin that makes a run
    slow is identifiable rather than mysterious.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..errors import PluginError
from ..logging_ import get_logger
from .manifest import HOOK_NAMES

LOGGER = get_logger("plugins.hooks")
"""Logger for hook execution faults."""

HookFn = Callable[..., Any]
"""A hook callable. Its signature depends on the hook name."""

TRANSFORM_HOOKS: frozenset[str] = frozenset({"before_tool", "filter_output", "build_prompt"})
"""Hooks whose return value replaces the data passed in."""

VETO_HOOKS: frozenset[str] = frozenset({"before_tool", "before_run"})
"""Hooks that may block an action by returning a falsy verdict object."""


@dataclass(slots=True)
class HookRegistration:
    """One registered hook, with where it came from."""

    name: str
    fn: HookFn
    plugin: str = ""
    priority: int = 100
    enabled: bool = True
    once: bool = False
    calls: int = 0
    failures: int = 0
    total_ms: float = 0.0
    fired: bool = False

    @property
    def label(self) -> str:
        """A readable identifier for logs and reports."""
        owner = self.plugin or getattr(self.fn, "__module__", "?")
        return f"{self.name}:{owner}"

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "name": self.name,
            "plugin": self.plugin,
            "priority": self.priority,
            "enabled": self.enabled,
            "once": self.once,
            "fired": self.fired,
            "calls": self.calls,
            "failures": self.failures,
            "mean_ms": round(self.total_ms / self.calls, 3) if self.calls else 0.0,
        }


@dataclass(slots=True)
class HookResult:
    """What running one hook chain produced."""

    name: str
    value: Any = None
    transformed: bool = False
    blocked: bool = False
    reason: str = ""
    ran: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every hook ran without raising."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "name": self.name,
            "transformed": self.transformed,
            "blocked": self.blocked,
            "reason": self.reason,
            "ran": self.ran,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 3),
            "errors": self.errors[:10],
        }


class HookRegistry:
    """Registers and runs lifecycle hooks."""

    def __init__(self, *, strict: bool = False, timeout: float = 0.0) -> None:
        """Create a registry.

        Args:
            strict: re-raise hook exceptions instead of recording them
            timeout: unused ceiling kept for interface symmetry; hooks are
                expected to be fast and synchronous
        """
        self.strict = bool(strict)
        self.timeout = float(timeout)
        self._hooks: dict[str, list[HookRegistration]] = {name: [] for name in HOOK_NAMES}
        self._lock = threading.RLock()
        self.executed = 0
        self.failures = 0

    # -- registration --------------------------------------------------------

    def register(self, name: str, fn: HookFn, *, plugin: str = "", priority: int = 100,
                 once: bool = False) -> HookRegistration:
        """Attach a callable to one lifecycle point.

        Args:
            name: hook name, one of :data:`~magent.plugins.manifest.HOOK_NAMES`
            fn: callable to invoke
            plugin: owning plugin, for reporting and removal
            priority: lower runs first; equal priorities keep insertion order
            once: fire at most one time, then disable

        Raises:
            PluginError: the hook name is unknown or ``fn`` is not callable
        """
        if name not in HOOK_NAMES:
            raise PluginError(f"unknown hook {name!r}; known: {', '.join(HOOK_NAMES)}",
                              details={"hook": name, "plugin": plugin})
        if not callable(fn):
            raise PluginError(f"hook {name!r} needs a callable, got {type(fn).__name__}",
                              details={"hook": name, "plugin": plugin})
        registration = HookRegistration(name=name, fn=fn, plugin=plugin,
                                        priority=int(priority), once=bool(once))
        with self._lock:
            bucket = self._hooks[name]
            bucket.append(registration)
            # A stable sort keeps insertion order among equal priorities, so two
            # plugins at the default priority run in registration order.
            bucket.sort(key=lambda item: item.priority)
        return registration

    def unregister(self, name: str, *, plugin: str = "", fn: HookFn | None = None) -> int:
        """Remove hooks, by name and optionally by plugin or callable."""
        with self._lock:
            if name not in self._hooks:
                return 0
            before = len(self._hooks[name])
            self._hooks[name] = [
                item for item in self._hooks[name]
                if not ((not plugin or item.plugin == plugin) and (fn is None or item.fn is fn))
            ]
            return before - len(self._hooks[name])

    def unregister_plugin(self, plugin: str) -> int:
        """Remove every hook belonging to one plugin."""
        removed = 0
        for name in HOOK_NAMES:
            removed += self.unregister(name, plugin=plugin)
        return removed

    def disable(self, name: str, *, plugin: str = "", enabled: bool = False) -> int:
        """Enable or disable hooks without removing them."""
        count = 0
        with self._lock:
            for item in self._hooks.get(name, []):
                if not plugin or item.plugin == plugin:
                    item.enabled = bool(enabled)
                    count += 1
        return count

    def hooks(self, name: str = "") -> list[HookRegistration]:
        """Return registrations for one hook, or all of them."""
        with self._lock:
            if name:
                return list(self._hooks.get(name, []))
            return [item for bucket in self._hooks.values() for item in bucket]

    def names(self) -> list[str]:
        """Return hook names that have at least one registration."""
        with self._lock:
            return [name for name in HOOK_NAMES if self._hooks.get(name)]

    def count(self, name: str = "") -> int:
        """Return how many hooks are registered."""
        return len(self.hooks(name))

    # -- execution -----------------------------------------------------------

    def emit(self, name: str, payload: Any = None, **kwargs: Any) -> HookResult:
        """Run one hook chain and return what happened.

        For a transform hook the chain threads the value through each callable,
        so later hooks see earlier edits. For an observe hook the payload is
        passed unchanged and return values are ignored.

        A hook may block a veto hook by returning an object whose ``allow``
        attribute is false, or a ``False`` literal; the chain then stops.

        Args:
            name: hook to run
            payload: data passed to each hook
            kwargs: extra keyword arguments forwarded to each hook
        """
        started = time.perf_counter()
        result = HookResult(name=name)
        if name not in HOOK_NAMES:
            result.errors.append(f"unknown hook {name!r}")
            return result
        with self._lock:
            registrations = [item for item in self._hooks.get(name, [])
                             if item.enabled and not (item.once and item.fired)]
        if not registrations:
            result.duration_ms = (time.perf_counter() - started) * 1000.0
            return result

        value = payload
        is_transform = name in TRANSFORM_HOOKS
        is_veto = name in VETO_HOOKS
        for registration in registrations:
            call_started = time.perf_counter()
            try:
                returned = registration.fn(value, **kwargs) if kwargs else registration.fn(value)
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not stop a run
                elapsed = (time.perf_counter() - call_started) * 1000.0
                with self._lock:
                    registration.calls += 1
                    registration.failures += 1
                    registration.total_ms += elapsed
                    self.executed += 1
                    self.failures += 1
                result.failed += 1
                result.errors.append(f"{registration.label}: {type(exc).__name__}: {exc}")
                LOGGER.warning("hook %s raised: %s", registration.label, exc)
                if self.strict:
                    raise
                continue
            elapsed = (time.perf_counter() - call_started) * 1000.0
            with self._lock:
                registration.calls += 1
                registration.total_ms += elapsed
                if registration.once:
                    registration.fired = True
                self.executed += 1
            result.ran += 1

            if is_veto and _is_veto(returned):
                result.blocked = True
                result.reason = _veto_reason(returned)
                result.value = value
                break
            if is_transform and returned is not None:
                value = returned
                result.transformed = True

        result.value = value
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    def transform(self, name: str, value: Any, **kwargs: Any) -> Any:
        """Run a transform hook and return the possibly-modified value."""
        return self.emit(name, value, **kwargs).value

    # -- reporting -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return per-hook execution statistics."""
        with self._lock:
            by_hook = {
                name: {
                    "registered": len(bucket),
                    "enabled": sum(1 for item in bucket if item.enabled),
                    "calls": sum(item.calls for item in bucket),
                    "failures": sum(item.failures for item in bucket),
                    "total_ms": round(sum(item.total_ms for item in bucket), 3),
                }
                for name, bucket in self._hooks.items() if bucket
            }
        return {
            "hooks": by_hook,
            "executed": self.executed,
            "failures": self.failures,
            "registered": self.count(),
            "strict": self.strict,
        }

    def slowest(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the slowest hooks by mean duration."""
        items = [item.to_dict() | {"label": item.label} for item in self.hooks()]
        items.sort(key=lambda record: record.get("mean_ms", 0.0), reverse=True)
        return items[: max(1, int(limit))]

    def describe(self) -> str:
        """Return a one-line summary."""
        info = self.stats()
        return (f"{info['registered']} hooks registered, {info['executed']} executed, "
                f"{info['failures']} failed")

    def clear(self) -> None:
        """Remove every registration."""
        with self._lock:
            self._hooks = {name: [] for name in HOOK_NAMES}
            self.executed = 0
            self.failures = 0

    def __len__(self) -> int:
        """Number of registered hooks."""
        return self.count()

    def __contains__(self, name: str) -> bool:
        """Whether a hook name has registrations."""
        with self._lock:
            return bool(self._hooks.get(name))


def _is_veto(returned: Any) -> bool:
    """Whether a hook's return value blocks the action."""
    if returned is None:
        return False
    if isinstance(returned, bool):
        return not returned
    allow = getattr(returned, "allow", None)
    if allow is False:
        return True
    if isinstance(returned, dict) and returned.get("allow") is False:
        return True
    return False


def _veto_reason(returned: Any) -> str:
    """Extract a human-readable reason from a veto value."""
    if isinstance(returned, dict):
        return str(returned.get("reason") or returned.get("message") or "blocked by hook")
    reason = getattr(returned, "reason", "") or getattr(returned, "message", "")
    return str(reason) if reason else "blocked by hook"


def hook(name: str, *, plugin: str = "", priority: int = 100,
         once: bool = False) -> Callable[[HookFn], HookFn]:
    """Decorator marking a function as a hook, for use with a registry.

    The decorated function keeps working normally and gains a ``hook_names``
    attribute listing what it wants, so a plugin module can be introspected and
    registered in one pass::

        @hook("before_tool", priority=10)
        def audit(value):
            log(value)
    """
    if name not in HOOK_NAMES:
        raise PluginError(f"unknown hook {name!r}; known: {', '.join(HOOK_NAMES)}")

    def decorator(fn: HookFn) -> HookFn:
        existing = list(getattr(fn, "hook_names", []) or [])
        existing.append({
            "name": name, "plugin": plugin, "priority": priority, "once": once,
        })
        fn.hook_names = existing  # type: ignore[attr-defined]
        return fn

    return decorator


def register_decorated(module: Any, registry: HookRegistry, *, plugin: str = "") -> int:
    """Register every ``@hook``-decorated callable found in a module."""
    registered = 0
    for attribute in dir(module):
        if attribute.startswith("__"):
            continue
        candidate = getattr(module, attribute, None)
        declarations = getattr(candidate, "hook_names", None)
        if not callable(candidate) or not declarations:
            continue
        for declaration in declarations:
            registry.register(
                str(declaration.get("name", "")),
                candidate,
                plugin=str(declaration.get("plugin") or plugin),
                priority=int(declaration.get("priority", 100) or 100),
                once=bool(declaration.get("once", False)),
            )
            registered += 1
    return registered


def collect_hooks(modules: Iterable[Any], registry: HookRegistry, *,
                  plugin: str = "") -> int:
    """Register decorated hooks from several modules."""
    return sum(register_decorated(module, registry, plugin=plugin) for module in modules)


def known_hooks() -> Sequence[str]:
    """Return every hook name MAgent dispatches."""
    return HOOK_NAMES


__all__ = [
    "TRANSFORM_HOOKS",
    "VETO_HOOKS",
    "HookFn",
    "HookRegistration",
    "HookRegistry",
    "HookResult",
    "collect_hooks",
    "hook",
    "known_hooks",
    "register_decorated",
]
