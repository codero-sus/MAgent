"""System and environment inspection tools.

Everything here is read-only about the host machine: it lets an agent reason
about the environment it is running in (CPU count, memory, platform, Python
version, environment variables) without being able to change it.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.files import human_size
from ...utils.timeutil import format_duration

ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TZ", "TERM", "EDITOR", "PWD",
    "PYTHONPATH", "PYTHON_VERSION", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
    "MAGENT_WORKSPACE", "MAGENT_CONFIG", "MAGENT_BACKEND", "MAGENT_CORTEX_URL",
    "MAGENT_MODEL", "MAGENT_MEMORY_DIR", "MAGENT_CACHE_DIR", "MAGENT_LOG_DIR",
    "MAGENT_TOOLS_ALLOW_NETWORK", "MAGENT_SANDBOX_ROOTS", "CORTEX_HOST", "CORTEX_PORT",
)
"""Environment variables the agent may read without an explicit allow flag."""

ENV_SECRET_HINTS: tuple[str, ...] = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
"""Substrings that mark an environment variable as sensitive."""


@tool(category="system", risk="read", tags=["system", "info", "core"])
def system_info(include_env: bool = False) -> dict[str, Any]:
    """Report platform, Python, CPU, and memory facts about this machine.

    Args:
        include_env: include allowlisted environment variables in the report
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "cpu_count": os.cpu_count() or 1,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timezone": time.tzname[0] if time.tzname else "UTC",
    }
    try:
        load1, load5, load15 = os.getloadavg()
        info["load_average"] = {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)}
    except (OSError, AttributeError):
        info["load_average"] = None
    memory = _memory_info()
    if memory:
        info["memory"] = memory
    if include_env:
        info["environment"] = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
    return info


def _memory_info() -> dict[str, Any] | None:
    """Read memory statistics from ``/proc/meminfo`` when available."""
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                multiplier = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
                values[name.strip()] = int(parts[0]) * multiplier
    except OSError:
        return None
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return {
        "total": total,
        "total_human": human_size(total),
        "available": available,
        "available_human": human_size(available),
        "used": total - available if total else 0,
        "percent_used": round(100.0 * (total - available) / total, 1) if total else 0.0,
    }


@tool(category="system", risk="read", tags=["disk", "usage"])
def disk_space(path: str = ".", context: ToolContext | None = None) -> dict[str, Any]:
    """Report free and total disk space for a path's filesystem.

    Args:
        path: a path on the filesystem to inspect, default is the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    probe = target if target.exists() else Path.cwd()
    usage = shutil.disk_usage(str(probe))
    return {
        "path": str(probe),
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "total_human": human_size(usage.total),
        "used_human": human_size(usage.used),
        "free_human": human_size(usage.free),
        "percent_used": round(100.0 * usage.used / usage.total, 1) if usage.total else 0.0,
    }


@tool(category="system", risk="read", tags=["env", "variables"])
def env_get(names: str, include_sensitive: bool = False,
            context: ToolContext | None = None) -> dict[str, Any]:
    """Read specific environment variables, masking sensitive values.

    Args:
        names: comma or space separated variable names
        include_sensitive: also allow variables whose name hints at a secret
    """
    ctx = context or ToolContext()
    from ...secrets import redact_text

    requested = [name.strip() for name in names.replace(",", " ").split() if name.strip()]
    if not requested:
        raise ToolExecutionError("Provide at least one variable name")
    out: dict[str, Any] = {}
    blocked: list[str] = []
    for name in requested:
        value = os.environ.get(name)
        if value is None:
            out[name] = None
            continue
        sensitive = any(hint in name.upper() for hint in ENV_SECRET_HINTS)
        if sensitive and not include_sensitive:
            out[name] = "<redacted: sensitive variable>"
            blocked.append(name)
        elif name not in ENV_ALLOWLIST and not include_sensitive:
            out[name] = "<not in allowlist: pass include_sensitive=true>"
            blocked.append(name)
        else:
            out[name] = redact_text(value, extra_secrets=list(ctx.secrets))
    return {
        "requested": requested,
        "found": sum(1 for value in out.values() if value is not None),
        "values": out,
        "masked": blocked,
    }


@tool(category="system", risk="read", tags=["env", "list"])
def env_list() -> dict[str, Any]:
    """List environment variable names, without values.

    Names alone are enough for an agent to detect misconfiguration; values are
    never emitted because they frequently hold credentials.
    """
    names = sorted(os.environ)
    allowlisted = [name for name in names if name in ENV_ALLOWLIST]
    sensitive = [name for name in names if any(hint in name.upper() for hint in ENV_SECRET_HINTS)]
    return {
        "count": len(names),
        "names": names,
        "allowlisted": allowlisted,
        "sensitive_names": sensitive,
    }


@tool(category="system", risk="read", tags=["python", "packages"])
def python_environment() -> dict[str, Any]:
    """Report the interpreter, virtualenv, and key stdlib versions in use.

    Args:
        (none)
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return {
        "version": sys.version,
        "version_info": list(sys.version_info),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_virtualenv": in_venv,
        "virtualenv": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_DEFAULT_ENV"),
        "path_entries": len(sys.path),
        "sys_path": sys.path[:20],
        "default_encoding": sys.getdefaultencoding(),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "recursion_limit": sys.getrecursionlimit(),
        "threads": threading.active_count(),
        "flags": {
            "optimize": sys.flags.optimize,
            "verbose": sys.flags.verbose,
            "no_site": sys.flags.no_site,
            "utf8_mode": bool(getattr(sys.flags, "utf8_mode", 0)),
        },
    }


@tool(category="system", risk="safe", tags=["uptime", "time"])
def uptime_and_time() -> dict[str, Any]:
    """Report process uptime, monotonic clock, and current wall-clock time.

    Args:
        (none)
    """
    from ..._version import __version__

    started = getattr(sys, "_magent_started", None)
    now = time.time()
    return {
        "version": __version__,
        "epoch": now,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "monotonic": time.monotonic(),
        "process_uptime_seconds": round(now - started, 3) if started else None,
        "process_uptime_human": format_duration(now - started) if started else "",
        "timezone": time.tzname[0] if time.tzname else "UTC",
        "tz_offset_seconds": -time.timezone if not time.daylight else -time.altzone,
    }


@tool(category="system", risk="safe", tags=["sleep", "wait"])
def wait(seconds: float, reason: str = "") -> dict[str, Any]:
    """Sleep for a bounded number of seconds, useful for rate-limit backoff.

    Args:
        seconds: how long to sleep, capped at 60
        reason: why the wait is needed, recorded in the tool result
    """
    bounded = max(0.0, min(float(seconds), 60.0))
    time.sleep(bounded)
    return {"slept_seconds": round(bounded, 3), "requested": seconds,
            "capped": bounded != float(seconds), "reason": reason}


TOOLS: list[Any] = [
    system_info, disk_space, env_get, env_list, python_environment, uptime_and_time, wait,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every system tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "ENV_ALLOWLIST",
    "ENV_SECRET_HINTS",
    "TOOLS",
    "disk_space",
    "env_get",
    "env_list",
    "load",
    "python_environment",
    "system_info",
    "uptime_and_time",
    "wait",
]
