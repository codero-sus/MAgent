"""The built-in tool catalogue.

Every module in this package exposes the same two names:

``TOOLS``
    The list of decorated handler functions defined in the module.
``load(registry)``
    Registers them all and returns how many were added.

:func:`load_all` walks :data:`MODULES` and calls each ``load``, so adding a new
tool module is one line here plus the module itself. Loading is tolerant by
default: a module that fails to import is recorded in
:func:`load_errors` rather than taking the whole agent down, which matters
because tool modules touch optional platform features.
"""

from __future__ import annotations

import importlib
import threading
import traceback
from types import ModuleType
from typing import Any, Iterable

from ...logging_ import get_logger

LOGGER = get_logger("tools.builtin")
"""Logger for tool module loading."""

MODULES: tuple[str, ...] = (
    "fs",
    "shell",
    "text_tools",
    "calc",
    "data_tools",
    "search",
    "web",
    "python_exec",
    "system",
    "media",
    "git_tools",
    "output_tools",
    "memory_tools",
    "agent_tools",
    "cortex_tools",
)
"""Built-in tool modules, in registration order."""

CATEGORY_ORDER: tuple[str, ...] = (
    "files", "shell", "text", "math", "datetime", "data", "search", "web",
    "python", "system", "media", "git", "output", "memory", "meta", "cortex",
)
"""Canonical tool category order, used for stable prompt rendering."""

_ERRORS: dict[str, str] = {}
"""Module name to error text for modules that failed to load."""


def load_module(registry: Any, name: str, *, strict: bool = False,
                only: Iterable[str] | None = None) -> int:
    """Import one built-in module and register its tools.

    Args:
        registry: the :class:`~magent.tools.base.ToolRegistry` to fill
        name: module name inside this package, e.g. ``"fs"``
        strict: re-raise import failures instead of recording them
        only: register just these tool names, ``None`` for the whole module
    """
    if name not in MODULES:
        raise KeyError(f"{name!r} is not a built-in tool module; known: {', '.join(MODULES)}")
    try:
        module: ModuleType = importlib.import_module(f"{__name__}.{name}")
        if only:
            wanted = {str(item) for item in only}
            registry.register_all([item.tool for item in getattr(module, "TOOLS", [])
                                   if item.tool.name in wanted])
            count = sum(1 for item in getattr(module, "TOOLS", [])
                        if item.tool.name in wanted)
        else:
            count = int(module.load(registry))
    except Exception as exc:  # noqa: BLE001 - one bad module must not kill the agent
        if strict:
            raise
        _ERRORS[name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        return 0
    _ERRORS.pop(name, None)
    return count


def load_all(registry: Any, *, include: Iterable[str] | None = None,
             exclude: Iterable[str] | None = None, strict: bool = False) -> int:
    """Register every built-in tool module; return the number of tools added.

    Args:
        registry: the registry to fill
        include: only load these modules, empty for all
        exclude: skip these modules
        strict: propagate import errors instead of recording them
    """
    blocked = {str(name) for name in (exclude or ())}
    if not include:
        return sum(load_module(registry, name, strict=strict)
                   for name in MODULES if name not in blocked)

    # ``include`` accepts module names *and* tool names. Silently registering
    # nothing when handed a tool name - the natural thing to write, since tool
    # names are what the agent and the docs talk about - produces an empty
    # registry and a confusing "unknown tool" failure much later, so both forms
    # resolve here and anything unrecognized is reported.
    index = tool_module_index()
    whole: set[str] = set()
    filters: dict[str, set[str]] = {}
    unknown: list[str] = []
    for raw in include:
        name = str(raw)
        if name in MODULES:
            whole.add(name)
        elif name in index:
            filters.setdefault(index[name], set()).add(name)
        else:
            unknown.append(name)
    if unknown:
        message = (f"unknown tool or module name(s): {', '.join(sorted(unknown))}")
        if strict:
            raise KeyError(message)
        LOGGER.warning("%s", message)
        _ERRORS["include"] = message
    total = 0
    for name in MODULES:
        if name in blocked:
            continue
        if name in whole:
            total += load_module(registry, name, strict=strict)
        elif name in filters:
            total += load_module(registry, name, strict=strict, only=filters[name])
    return total


_TOOL_MODULES: dict[str, str] = {}
_TOOL_INDEX_LOCK = threading.Lock()


def tool_module_index(*, refresh: bool = False) -> dict[str, str]:
    """Map every built-in tool name to the module that provides it.

    The mapping is built once by importing the modules, which is cheap compared
    with the alternative of failing to resolve a tool name at all.

    Construction is atomic. Building it in place would let a second thread see a
    half-populated mapping - enough for the ``if cached`` check to pass while
    some tool names were still missing, which then reported a valid tool as
    unknown. That is reachable from any parallel caller, including the
    evaluation harness running cases concurrently.

    Args:
        refresh: rebuild even if a cached mapping exists
    """
    with _TOOL_INDEX_LOCK:
        global _TOOL_MODULES
        if _TOOL_MODULES and not refresh:
            return dict(_TOOL_MODULES)
        built: dict[str, str] = {}
        for name in MODULES:
            try:
                for item in module_tools(name):
                    built[item.tool.name] = name
            except Exception:  # noqa: BLE001 - discovery is best effort
                continue
        _TOOL_MODULES = built
        return dict(built)


def load_errors() -> dict[str, str]:
    """Return the recorded load failures, keyed by module name."""
    return dict(_ERRORS)


def clear_errors() -> None:
    """Forget previously recorded load failures."""
    _ERRORS.clear()


def module_tools(name: str) -> list[Any]:
    """Return the ``TOOLS`` list of one module without registering anything."""
    module = importlib.import_module(f"{__name__}.{name}")
    return list(getattr(module, "TOOLS", []))


def builtin_tool_names() -> list[str]:
    """Return every built-in tool name, sorted, without building a registry."""
    names: list[str] = []
    for name in MODULES:
        try:
            names.extend(item.tool.name for item in module_tools(name))
        except Exception:  # noqa: BLE001 - discovery is best effort
            continue
    return sorted(names)


def builtin_tool_count() -> int:
    """Return how many tools the built-in catalogue provides."""
    return len(builtin_tool_names())


def tools_by_category() -> dict[str, list[str]]:
    """Group every built-in tool name by its declared category."""
    grouped: dict[str, list[str]] = {}
    for name in MODULES:
        try:
            items = module_tools(name)
        except Exception:  # noqa: BLE001 - discovery is best effort
            continue
        for item in items:
            grouped.setdefault(item.tool.category, []).append(item.tool.name)
    ordered: dict[str, list[str]] = {}
    for category in CATEGORY_ORDER:
        if category in grouped:
            ordered[category] = sorted(grouped[category])
    for category in sorted(grouped):
        if category not in ordered:
            ordered[category] = sorted(grouped[category])
    return ordered


__all__ = [
    "CATEGORY_ORDER",
    "MODULES",
    "builtin_tool_count",
    "builtin_tool_names",
    "clear_errors",
    "load_all",
    "load_errors",
    "load_module",
    "module_tools",
    "tool_module_index",
    "tools_by_category",
]
