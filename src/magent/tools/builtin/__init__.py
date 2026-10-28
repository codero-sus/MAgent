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
import traceback
from collections.abc import Iterable
from types import ModuleType
from typing import Any

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
    "desktop",
    "media",
    "git_tools",
    "output_tools",
    "memory_tools",
    "agent_tools",
    "cortex_tools",
)
"""Built-in tool modules, in registration order."""

CATEGORY_ORDER: tuple[str, ...] = (
    "files",
    "shell",
    "text",
    "math",
    "datetime",
    "data",
    "search",
    "web",
    "python",
    "system",
    "desktop",
    "media",
    "git",
    "output",
    "memory",
    "meta",
    "cortex",
)
"""Canonical tool category order, used for stable prompt rendering."""

_ERRORS: dict[str, str] = {}
"""Module name to error text for modules that failed to load."""


def load_module(registry: Any, name: str, *, strict: bool = False) -> int:
    """Import one built-in module and register its tools.

    Args:
        registry: the :class:`~magent.tools.base.ToolRegistry` to fill
        name: module name inside this package, e.g. ``"fs"``
        strict: re-raise import failures instead of recording them
    """
    if name not in MODULES:
        raise KeyError(f"{name!r} is not a built-in tool module; known: {', '.join(MODULES)}")
    try:
        module: ModuleType = importlib.import_module(f"{__name__}.{name}")
        count = int(module.load(registry))
    except Exception as exc:
        if strict:
            raise
        _ERRORS[name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        return 0
    _ERRORS.pop(name, None)
    return count


def load_all(
    registry: Any,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    strict: bool = False,
) -> int:
    """Register every built-in tool module; return the number of tools added.

    Args:
        registry: the registry to fill
        include: only load these modules, empty for all
        exclude: skip these modules
        strict: propagate import errors instead of recording them
    """
    wanted = {name for name in include} if include else set(MODULES)
    blocked = {name for name in (exclude or ())}
    total = 0
    for name in MODULES:
        if name not in wanted or name in blocked:
            continue
        total += load_module(registry, name, strict=strict)
    return total


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
        except Exception:
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
        except Exception:
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
    "tools_by_category",
]
