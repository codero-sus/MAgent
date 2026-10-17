"""Shell and process tools.

Commands run through :class:`~magent.tools.sandbox.Sandbox`, which enforces the
allowlist, refuses the always-denied set, and scrubs secret-looking variables
from the child environment. Output is captured with a byte ceiling and a hard
timeout, and the exit code is always reported so the agent can react.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.text import ellipsize


def _run(
    command: list[str] | str,
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str],
    shell: bool = False,
    max_output: int = 200_000,
    stdin_text: str = "",
) -> dict[str, Any]:
    """Run one subprocess with capture, timeout, and output ceilings."""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=shell,
            cwd=str(cwd),
            env=env,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = (time.perf_counter() - started) * 1000.0
        raise ToolExecutionError(
            f"Command timed out after {timeout}s: {_as_text(command)}",
            details={
                "stdout": (exc.stdout or "")[:max_output] if isinstance(exc.stdout, str) else "",
                "duration_ms": duration,
            },
        ) from exc
    except OSError as exc:
        raise ToolExecutionError(f"Could not run {_as_text(command)}: {exc}") from exc
    duration = (time.perf_counter() - started) * 1000.0
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "command": _as_text(command),
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": ellipsize(stdout, max_output),
        "stderr": ellipsize(stderr, max_output),
        "stdout_lines": stdout.count("\n") + (1 if stdout and not stdout.endswith("\n") else 0),
        "stderr_lines": stderr.count("\n") + (1 if stderr and not stderr.endswith("\n") else 0),
        "stdout_truncated": len(stdout) > max_output,
        "stderr_truncated": len(stderr) > max_output,
        "duration_ms": round(duration, 2),
        "cwd": str(cwd),
    }


def _as_text(command: list[str] | str) -> str:
    """Render a command for display."""
    return command if isinstance(command, str) else " ".join(shlex.quote(part) for part in command)


@tool(category="shell", risk="execute", executes_code=True, timeout=120.0, tags=["exec", "core"])
def run_command(
    command: str,
    cwd: str = ".",
    timeout: float = 60.0,
    shell: bool = False,
    stdin: str = "",
    env: str = "",
    max_output: int = 200000,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Run a shell command inside the workspace and return its output.

    Args:
        command: the command line to execute
        cwd: working directory, relative to the workspace
        timeout: seconds before the command is killed
        shell: run through the platform shell instead of exec'ing directly
        stdin: text to feed to the process on stdin
        env: extra environment variables as KEY=VALUE pairs separated by commas
        max_output: maximum characters of stdout and stderr to return
    """
    ctx = context or ToolContext()
    sandbox = ctx.sandbox
    if sandbox is not None:
        sandbox.check_command(command)
    workdir = ctx.resolve_path(cwd)
    if not workdir.is_dir():
        raise ToolExecutionError(f"Working directory does not exist: {workdir}")
    base_env = sandbox.env_for_subprocess() if sandbox is not None else dict(os.environ)
    for pair in [item.strip() for item in env.split(",") if item.strip()]:
        key, _, value = pair.partition("=")
        if key:
            base_env[key.strip()] = value.strip()
    argv: list[str] | str = command if shell else shlex.split(command, posix=os.name != "nt")
    if not shell and not argv:
        raise ToolExecutionError("Empty command")
    return _run(
        argv,
        cwd=workdir,
        timeout=min(timeout, ctx.timeout or timeout),
        env=base_env,
        shell=shell,
        max_output=max_output,
        stdin_text=stdin,
    )


@tool(category="shell", risk="execute", executes_code=True, timeout=300.0, tags=["script"])
def run_script(
    script: str,
    interpreter: str = "",
    cwd: str = ".",
    timeout: float = 120.0,
    args: str = "",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Run a multi-line script through an interpreter without writing a file.

    Args:
        script: the script source to execute
        interpreter: program to run it with, defaults to python for .py-style source
        cwd: working directory relative to the workspace
        timeout: seconds before the script is killed
        args: extra arguments appended after the script
    """
    ctx = context or ToolContext()
    sandbox = ctx.sandbox
    chosen = interpreter or (sys.executable if "def " in script or "import " in script else "sh")
    if sandbox is not None:
        sandbox.check_command(chosen)
    workdir = ctx.resolve_path(cwd)
    env = sandbox.env_for_subprocess() if sandbox is not None else dict(os.environ)
    flags = ["-c"] if chosen.lower().endswith(("python", "python3", "sh", "bash", "zsh")) else []
    argv = [chosen, *flags, script, *(shlex.split(args) if args else [])]
    return _run(argv, cwd=workdir, timeout=timeout, env=env, shell=False)


@tool(category="shell", risk="read", executes_code=False, timeout=30.0, tags=["env", "system"])
def show_environment(
    include: str = "", redact: bool = True, context: ToolContext | None = None
) -> dict[str, Any]:
    """List environment variables visible to tools, with secrets redacted.

    Args:
        include: comma separated variable names or prefixes to show, empty for all
        redact: mask values whose names look like credentials
    """
    wanted = [item.strip().upper() for item in include.split(",") if item.strip()]
    out: dict[str, str] = {}
    secret_tokens = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "APIKEY", "CREDENTIAL", "PRIVATE")
    for key, value in sorted(os.environ.items()):
        if wanted and not any(key.upper().startswith(prefix) for prefix in wanted):
            continue
        if redact and any(token in key.upper() for token in secret_tokens):
            out[key] = "[REDACTED]"
        else:
            out[key] = value
    return {"count": len(out), "redacted": redact, "variables": out}


@tool(category="shell", risk="read", executes_code=False, tags=["which", "path"])
def find_executable(name: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Locate an executable on PATH and report whether it is usable.

    Args:
        name: program name to look up
    """
    resolved = shutil.which(name)
    return {
        "name": name,
        "found": bool(resolved),
        "path": resolved or "",
        "allowed": bool(context and context.sandbox and context.sandbox.check_command(name).allowed)
        if resolved
        else False,
    }


@tool(category="shell", risk="read", executes_code=False, tags=["processes"])
def list_processes(limit: int = 40, context: ToolContext | None = None) -> dict[str, Any]:
    """List running processes using a platform appropriate command.

    Args:
        limit: maximum number of processes to return
    """
    try:
        if sys.platform == "win32":
            completed = subprocess.run(
                ["tasklist"], capture_output=True, text=True, timeout=10, check=False
            )
            lines = (completed.stdout or "").splitlines()[3:]
            rows = [{"raw": line.strip()} for line in lines[:limit] if line.strip()]
        else:
            completed = subprocess.run(
                ["ps", "-eo", "pid,ppid,pcpu,pmem,etime,comm"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            lines = (completed.stdout or "").splitlines()
            rows = []
            for line in lines[1 : limit + 1]:
                parts = line.split(None, 5)
                if len(parts) == 6:
                    rows.append(
                        {
                            "pid": int(parts[0]),
                            "ppid": int(parts[1]),
                            "cpu": float(parts[2]),
                            "mem": float(parts[3]),
                            "elapsed": parts[4],
                            "command": parts[5],
                        }
                    )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolExecutionError(f"Could not list processes: {exc}") from exc
    return {"count": len(rows), "processes": rows}


TOOLS: list[Any] = [run_command, run_script, show_environment, find_executable, list_processes]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every shell tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "find_executable",
    "list_processes",
    "load",
    "run_command",
    "run_script",
    "show_environment",
]
