"""Restricted Python execution inside the agent workspace.

This tool is the most powerful and the most dangerous thing MAgent exposes, so
it is layered:

1. Source is statically audited before it runs: imports, attribute access to
   dunder names, and a blocklist of modules are rejected.
2. Execution happens with no builtins except a whitelisted subset.
3. A wall-clock timeout runs the code on a worker thread.
4. Output, printed text, and created files are captured with byte ceilings.
5. The tool is marked ``executes_code`` so the approval policy treats it as
   dangerous in the default configuration.

It is still not a security boundary against a determined attacker - a sandbox
that must run arbitrary Python cannot be. It exists to stop accidents and to
make every execution visible, logged, and approvable.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool

BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "ctypes",
        "multiprocessing",
        "signal",
        "shutil",
        "importlib",
        "pkgutil",
        "sys",
        "os",
        "pathlib",
        "tempfile",
        "pty",
        "resource",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "telnetlib",
        "webbrowser",
        "asyncio",
        "threading",
        "concurrent",
        "queue",
        "pickle",
        "shelve",
        "marshal",
    }
)
"""Modules that cannot be imported by executed code."""

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "statistics",
        "random",
        "re",
        "json",
        "itertools",
        "functools",
        "collections",
        "datetime",
        "decimal",
        "fractions",
        "string",
        "textwrap",
        "unicodedata",
        "heapq",
        "bisect",
        "copy",
        "csv",
        "base64",
        "hashlib",
        "uuid",
    }
)
"""Modules that are safe and useful for data manipulation."""

SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "ArithmeticError",
        "RuntimeError",
        "StopIteration",
    }
)
"""Builtins exposed to executed code."""

BLOCKED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__dict__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__reduce__",
        "__reduce_ex__",
    }
)
"""Attribute names that enable sandbox escapes."""

MAX_SOURCE_CHARS = 100_000
"""Hard ceiling on submitted source length."""


class SourceAuditError(ToolExecutionError):
    """Raised when static analysis rejects the submitted source."""

    code = "python_source_rejected"


@tool(
    category="python",
    risk="execute",
    executes_code=True,
    timeout=60.0,
    tags=["exec", "code", "core"],
)
def python_exec(
    code: str,
    timeout: float = 30.0,
    capture_files: bool = True,
    allow_imports: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Execute Python code in a restricted namespace and return its output.

    The code has no access to the filesystem, network, or process APIs unless
    the sandbox policy explicitly allows it. Assign to a variable named
    ``result`` to return structured data.

    Args:
        code: Python source to execute
        timeout: seconds before execution is abandoned
        capture_files: list files created in the workspace during execution
        allow_imports: permit imports from the safe module allowlist
    """
    ctx = context or ToolContext()
    source = str(code or "")
    if not source.strip():
        raise ToolExecutionError("Empty code")
    if len(source) > MAX_SOURCE_CHARS:
        raise ToolExecutionError(f"Code exceeds {MAX_SOURCE_CHARS} characters")
    audit_python_source(source, allow_imports=allow_imports)
    workspace = ctx.workspace
    before = _snapshot_files(workspace) if capture_files else set()

    namespace: dict[str, Any] = build_safe_namespace(workspace=workspace)
    stdout = io.StringIO()
    stderr = io.StringIO()
    started = time.perf_counter()
    error_text = ""
    error_type = ""
    traceback_text = ""
    completed = False
    try:
        compiled = compile(source, "<magent-python-exec>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result_container = _run_with_timeout(compiled, namespace, timeout)
        completed = True
        if isinstance(result_container, BaseException):
            completed = False
            error_type = type(result_container).__name__
            error_text = str(result_container)[:2000]
            import traceback

            traceback_text = "".join(
                traceback.format_exception(
                    type(result_container), result_container, result_container.__traceback__
                )
            )[-4000:]
    except SyntaxError as exc:
        error_type = "SyntaxError"
        error_text = f"line {exc.lineno}: {exc.msg}"
    except TimeoutError:
        error_type = "TimeoutError"
        error_text = f"Execution exceeded {timeout}s"
    duration = (time.perf_counter() - started) * 1000.0

    printed = stdout.getvalue()
    created = sorted(_snapshot_files(workspace) - before) if capture_files else []
    result_value = namespace.get("result", _MISSING)
    serialized: Any = None
    if result_value is not _MISSING:
        try:
            serialized = json.loads(json.dumps(result_value, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            serialized = repr(result_value)[:4000]
    payload: dict[str, Any] = {
        "ok": completed and not error_text,
        "stdout": printed[: ctx.output_limit],
        "stdout_chars": len(printed),
        "stderr": stderr.getvalue()[: ctx.output_limit],
        "result": serialized,
        "result_type": type(result_value).__name__ if result_value is not _MISSING else None,
        "duration_ms": round(duration, 2),
        "files_created": created[:100],
        "timed_out": error_type == "TimeoutError",
    }
    if error_text:
        payload["error"] = error_text
        payload["error_type"] = error_type
        payload["traceback"] = traceback_text
    return payload


_MISSING = object()
"""Sentinel distinguishing 'result not set' from 'result is None'."""


def _run_with_timeout(compiled: Any, namespace: dict[str, Any], timeout: float) -> Any:
    """Run compiled code on a worker thread with a wall-clock timeout."""
    import concurrent.futures

    if timeout <= 0:
        exec(compiled, namespace)
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(exec, compiled, namespace)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"execution exceeded {timeout}s") from None
        except BaseException as exc:
            return exc


def _snapshot_files(root: Path) -> set[str]:
    """Return the set of file paths currently under ``root``."""
    if not root.is_dir():
        return set()
    out: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_file():
            out.add(str(entry))
        if len(out) > 20000:
            break
    return out


def build_safe_namespace(*, workspace: Path | None = None) -> dict[str, Any]:
    """Build the restricted global namespace handed to executed code."""
    import builtins as _builtins
    import collections
    import datetime
    import decimal
    import fractions
    import itertools
    import statistics

    safe_builtins = {
        name: getattr(_builtins, name) for name in SAFE_BUILTINS if hasattr(_builtins, name)
    }

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        """Allow only modules on the safe allowlist."""
        root = name.split(".")[0]
        if root in BLOCKED_MODULES or root not in ALLOWED_MODULES:
            raise ImportError(f"import of {name!r} is not permitted in python_exec")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = guarded_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "__magent__"}
    namespace.update(
        {
            "math": math,
            "statistics": statistics,
            "itertools": itertools,
            "collections": collections,
            "datetime": datetime,
            "decimal": decimal,
            "fractions": fractions,
            "json": json,
            "re": re,
            "workspace": str(workspace) if workspace else ".",
        }
    )
    return namespace


def audit_python_source(source: str, *, allow_imports: bool = True) -> dict[str, Any]:
    """Statically audit Python source and reject unsafe constructs.

    Returns a report describing what was found, which is also useful on its own
    for code review tooling.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourceAuditError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc
    imports: list[str] = []
    from_imports: list[str] = []
    attributes: list[str] = []
    calls: list[str] = []
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    findings.append(f"blocked module import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            from_imports.append(module)
            if module.split(".")[0] in BLOCKED_MODULES:
                findings.append(f"blocked module import: {module}")
        elif isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            if node.attr in BLOCKED_ATTRIBUTES:
                findings.append(f"blocked attribute access: {node.attr}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.append(name)
            if name in {
                "exec",
                "eval",
                "compile",
                "open",
                "input",
                "breakpoint",
                "exit",
                "quit",
                "globals",
                "locals",
                "vars",
                "memoryview",
            }:
                findings.append(f"blocked call: {name}")
    if not allow_imports and (imports or from_imports):
        findings.append("imports are disabled for this execution")
    blocked_imports = [
        name for name in imports + from_imports if name.split(".")[0] in BLOCKED_MODULES
    ]
    report = {
        "ok": not findings,
        "findings": findings,
        "imports": sorted(set(imports)),
        "from_imports": sorted(set(from_imports)),
        "blocked_imports": sorted(set(blocked_imports)),
        "calls": sorted(set(calls)),
        "lines": source.count("\n") + 1,
        "chars": len(source),
    }
    if findings:
        raise SourceAuditError(
            "Code rejected by the static auditor: " + "; ".join(dict.fromkeys(findings))
        )
    return report


def _call_name(node: ast.AST) -> str:
    """Return the dotted name of a call target, or ``""``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


@tool(category="python", risk="read", tags=["audit", "security"])
def audit_python(code: str, allow_imports: bool = True) -> dict[str, Any]:
    """Statically audit Python source without running it.

    Args:
        code: Python source to inspect
        allow_imports: whether imports would be permitted at execution time
    """
    try:
        return audit_python_source(code, allow_imports=allow_imports)
    except SourceAuditError as exc:
        return {"ok": False, "error": str(exc), "findings": _findings_from_message(str(exc))}


def _findings_from_message(message: str) -> list[str]:
    """Recover the individual findings from an audit rejection message."""
    _, _, tail = message.partition(": ")
    return [item.strip() for item in tail.split(";") if item.strip()]


@tool(category="python", risk="read", tags=["inspect", "ast"])
def inspect_python(code: str) -> dict[str, Any]:
    """Report the functions, classes, imports, and docstrings in Python source.

    Args:
        code: Python source to inspect
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolExecutionError(f"Syntax error at line {exc.lineno}: {exc.msg}") from exc
    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[str] = []
    constants: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node) or ""
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "args": [arg.arg for arg in node.args.args],
                    "defaults": len(node.args.defaults),
                    "docstring": docstring.split("\n")[0][:200],
                    "returns": ast.unparse(node.returns) if node.returns else "",
                    "body_lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = [
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": methods,
                    "docstring": (ast.get_docstring(node) or "").split("\n")[0][:200],
                }
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            else:
                imports.append(node.module or "")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
    return {
        "module_docstring": (ast.get_docstring(tree) or "").split("\n")[0][:300],
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "constants": constants,
        "line_count": code.count("\n") + 1,
        "function_count": len(functions),
        "class_count": len(classes),
    }


TOOLS: list[Any] = [python_exec, audit_python, inspect_python]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every Python execution tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "ALLOWED_MODULES",
    "BLOCKED_ATTRIBUTES",
    "BLOCKED_MODULES",
    "MAX_SOURCE_CHARS",
    "SAFE_BUILTINS",
    "TOOLS",
    "SourceAuditError",
    "audit_python",
    "audit_python_source",
    "build_safe_namespace",
    "inspect_python",
    "load",
    "python_exec",
]
