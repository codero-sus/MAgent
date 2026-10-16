"""Tool definition, registration, and dispatch.

A tool is a typed function plus a JSON Schema description of its arguments.
MAgent generates the schema automatically from type hints and the docstring so
a tool author writes one thing, not three:

.. code-block:: python

    @tool(category="text")
    def shout(text: str, times: int = 1) -> str:
        \"\"\"Repeat TEXT in upper case.

        Args:
            text: the phrase to shout
            times: how many times to repeat it
        \"\"\"
        return (text.upper() + " ") * times

The docstring's first line becomes the tool description the model sees; the
``Args:`` block becomes per-parameter descriptions. Dispatch validates
arguments, enforces the sandbox and approval policy, times the call, truncates
oversized output, and returns a :class:`~magent.types.ToolResult`.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
import types
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from ..errors import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from ..logging_ import get_logger
from ..types import ToolCall, ToolResult, new_id
from ..utils.text import ellipsize, snake_case
from .schema import (
    SchemaValidator,
    array_schema,
    boolean_schema,
    enum_schema,
    integer_schema,
    number_schema,
    object_schema,
    string_schema,
)

LOGGER = get_logger("tools")

ToolHandler = Callable[..., Any]
"""Signature every tool implementation must satisfy."""

RISK_LEVELS = ("safe", "read", "write", "execute", "destructive", "network")
"""Ordered risk ladder used by approval policies."""

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    dict: "object",
    type(None): "null",
    Path: "string",
    Any: "string",
}
"""Mapping from Python annotations to JSON Schema type names."""

_ARGS_SECTION = re.compile(r"^\s*(?:Args|Arguments|Parameters|Params)\s*:\s*$", re.IGNORECASE)
_RETURNS_SECTION = re.compile(r"^\s*(?:Returns?|Yields?|Raises)\s*:\s*$", re.IGNORECASE)
_ARG_LINE = re.compile(r"^\s{2,}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ToolSpec:
    """Declarative metadata describing one tool."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: object_schema({}))
    returns: str = "any"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    risk: str = "safe"
    destructive: bool = False
    network: bool = False
    filesystem: bool = False
    executes_code: bool = False
    timeout: float = 60.0
    output_limit: int = 32000
    parallel_safe: bool = True
    deterministic: bool = True
    requires: list[str] = field(default_factory=list)
    requires_approval: bool = False
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    hidden: bool = False
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the name and derive risk flags from the category."""
        self.name = str(self.name).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", self.name):
            raise ValueError(f"Invalid tool name {self.name!r}")
        if self.risk not in RISK_LEVELS:
            self.risk = "safe"
        if self.network and self.risk == "safe":
            self.risk = "network"
        if self.executes_code and self.risk in {"safe", "read"}:
            self.risk = "execute"

    def to_openai(self) -> dict[str, Any]:
        """Render the OpenAI ``tools`` entry the model receives."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or self.name,
                "parameters": self.parameters or object_schema({}),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full specification."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "category": self.category,
            "tags": list(self.tags),
            "risk": self.risk,
            "destructive": self.destructive,
            "network": self.network,
            "filesystem": self.filesystem,
            "executes_code": self.executes_code,
            "timeout": self.timeout,
            "output_limit": self.output_limit,
            "parallel_safe": self.parallel_safe,
            "deterministic": self.deterministic,
            "requires": list(self.requires),
            "requires_approval": self.requires_approval,
            "examples": list(self.examples),
            "version": self.version,
            "hidden": self.hidden,
            "aliases": list(self.aliases),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolSpec:
        """Rebuild a spec from a serialized dictionary."""
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def required_parameters(self) -> list[str]:
        """Return the names of parameters the caller must supply."""
        return [str(item) for item in (self.parameters or {}).get("required", [])]

    def parameter_names(self) -> list[str]:
        """Return every declared parameter name."""
        return sorted((self.parameters or {}).get("properties", {}))


# --------------------------------------------------------------------------- #
# Execution context
# --------------------------------------------------------------------------- #


@dataclass
class ToolContext:
    """Everything a tool may need, injected at call time.

    Tools never reach into global state; they receive a context. This is what
    makes the sandbox enforceable and the tool set unit-testable in isolation.
    """

    workspace: Path = field(default_factory=Path.cwd)
    sandbox: Any = None
    approval: Any = None
    events: Any = None
    config: Any = None
    memory: Any = None
    agent: Any = None
    backend: Any = None
    session_id: str = ""
    run_id: str = ""
    step_index: int = 0
    call_id: str = field(default_factory=lambda: new_id("call"))
    tool_name: str = ""
    timeout: float = 60.0
    output_limit: int = 32000
    registry: Any = None
    secrets: list[str] = field(default_factory=list)
    dry_run: bool = False
    verbosity: str = "normal"
    logger: Any = field(default_factory=lambda: get_logger("tools"))
    extras: dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, candidate: str | Path) -> Path:
        """Resolve a possibly relative path against the workspace jail."""
        path = Path(str(candidate)).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        if self.sandbox is not None:
            return self.sandbox.resolve(path)
        return path

    def emit(self, kind: str, **data: Any) -> None:
        """Publish an event if an event bus is attached."""
        if self.events is not None:
            try:
                self.events.emit(kind, **data)
            except Exception:  # noqa: BLE001 - telemetry must not break tools
                pass

    def sub(self, **overrides: Any) -> ToolContext:
        """Return a copy of the context with selected fields replaced."""
        import dataclasses

        return dataclasses.replace(self, **overrides)

    # -- aliases -------------------------------------------------------------
    # Tools written against the shorter names keep working; the canonical
    # fields above are what the agent layer populates.

    @property
    def bus(self) -> Any:
        """Alias for :attr:`events`, the run's event bus."""
        return self.events

    @property
    def step(self) -> int:
        """Alias for :attr:`step_index`."""
        return self.step_index

    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for :attr:`extras`, the per-context scratch mapping."""
        return self.extras

    @property
    def workspace_path(self) -> Path:
        """Return the workspace as an absolute :class:`~pathlib.Path`."""
        return Path(self.workspace).expanduser().resolve()

    def get_backend_admin(self) -> Any:
        """Return a Cortex admin client for the configured backend, or ``None``.

        Built lazily so importing the tool layer never opens a socket, and so a
        run that never touches Cortex admin routes never constructs a client.
        """
        cached = self.extras.get("__cortex_admin__")
        if cached is not None:
            return cached
        backend = self.backend
        if backend is None:
            return None
        from ..llm.cortex import CortexBackend
        from ..llm.cortex_admin import CortexAdmin

        if not isinstance(backend, CortexBackend):
            return None
        admin = CortexAdmin(backend=backend)
        self.extras["__cortex_admin__"] = admin
        return admin


# --------------------------------------------------------------------------- #
# Tool object
# --------------------------------------------------------------------------- #


@dataclass
class Tool:
    """A callable plus its specification and execution policy."""

    spec: ToolSpec
    handler: ToolHandler
    source_module: str = ""
    validator: SchemaValidator = field(default_factory=SchemaValidator)
    call_count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    last_error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Record where the handler came from."""
        if not self.source_module:
            module = getattr(self.handler, "__module__", "")
            self.source_module = str(module or "builtin")

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self.spec.name

    @property
    def description(self) -> str:
        """Return the model-facing description."""
        return self.spec.description

    @property
    def category(self) -> str:
        """Return the tool category."""
        return self.spec.category

    @property
    def risk(self) -> str:
        """Return the risk level."""
        return self.spec.risk

    def to_openai(self) -> dict[str, Any]:
        """Render the OpenAI tool entry."""
        return self.spec.to_openai()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spec plus runtime statistics."""
        data = self.spec.to_dict()
        data["stats"] = self.stats()
        data["module"] = self.source_module
        return data

    def stats(self) -> dict[str, Any]:
        """Return per-tool telemetry."""
        with self._lock:
            return {
                "calls": self.call_count,
                "errors": self.error_count,
                "total_ms": round(self.total_ms, 2),
                "avg_ms": round(self.total_ms / self.call_count, 2) if self.call_count else 0.0,
                "last_error": self.last_error,
            }

    # -- validation and execution -------------------------------------------

    def validate_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and coerce arguments, raising :class:`ToolValidationError`."""
        payload = dict(arguments or {})
        result = self.validator.validate(payload, self.spec.parameters)
        if not result.valid:
            raise ToolValidationError(
                f"Invalid arguments for {self.name}: {result.error_text()}",
                details={"tool": self.name, "errors": result.to_dict()["errors"]},
            )
        coerced = result.coerced
        return dict(coerced) if isinstance(coerced, dict) else payload

    def run(self, arguments: Mapping[str, Any] | None = None, context: ToolContext | None = None) -> ToolResult:
        """Validate, execute, time, and package one tool call."""
        call_id = (context.call_id if context else "") or new_id("call")
        ctx = context or ToolContext(tool_name=self.name)
        started = time.perf_counter()
        try:
            parsed = self.validate_arguments(arguments or {})
        except ToolValidationError as exc:
            return self._record_failure(call_id, exc, started, code=exc.code)
        except Exception as exc:  # noqa: BLE001
            return self._record_failure(call_id, exc, started, code="tool_validation_error")
        ctx.tool_name = self.name
        if ctx.dry_run:
            return ToolResult(
                call_id=call_id,
                name=self.name,
                ok=True,
                content={"dry_run": True, "tool": self.name, "arguments": parsed},
                duration_ms=(time.perf_counter() - started) * 1000.0,
                metadata={"dry_run": True},
            )
        try:
            outcome = self._invoke(parsed, ctx)
        except ToolError as exc:
            return self._record_failure(call_id, exc, started, code=exc.code)
        except Exception as exc:  # noqa: BLE001 - tools may raise anything
            LOGGER.debug("tool %s raised %s", self.name, exc)
            return self._record_failure(call_id, exc, started, code="tool_execution_error")
        duration = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self.call_count += 1
            self.total_ms += duration
        content, truncated = _truncate_output(outcome, self.spec.output_limit)
        return ToolResult(
            call_id=call_id,
            name=self.name,
            ok=True,
            content=content,
            duration_ms=duration,
            truncated=truncated,
            metadata={"category": self.spec.category, "risk": self.spec.risk},
        )

    def _invoke(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        """Call the handler, injecting ``context`` when it accepts one."""
        signature = inspect.signature(self.handler)
        parameters = signature.parameters
        if "context" in parameters:
            return self.handler(context=context, **arguments)
        if "ctx" in parameters:
            return self.handler(ctx=context, **arguments)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if accepts_kwargs:
            return self.handler(context=context, **arguments)
        allowed = {key: value for key, value in arguments.items() if key in parameters}
        return self.handler(**allowed)

    def _record_failure(self, call_id: str, exc: BaseException, started: float, *, code: str) -> ToolResult:
        """Build a failure result and update error counters."""
        duration = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self.call_count += 1
            self.error_count += 1
            self.total_ms += duration
            self.last_error = str(exc)[:300]
        return ToolResult(
            call_id=call_id,
            name=self.name,
            ok=False,
            error=str(exc)[:2000],
            error_code=code,
            duration_ms=duration,
            metadata={"category": self.spec.category},
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling the tool like a plain function, bypassing validation."""
        return self.handler(*args, **kwargs)


def _truncate_output(value: Any, limit: int) -> tuple[Any, bool]:
    """Cap oversized tool output so it cannot blow the context window."""
    if limit <= 0:
        return value, False
    if isinstance(value, str):
        if len(value) <= limit:
            return value, False
        return ellipsize(value, limit, marker="\n...[truncated]"), True
    if isinstance(value, (bytes, bytearray)):
        if len(value) <= limit:
            return bytes(value), False
        return bytes(value[:limit]), True
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
        if len(text) <= limit:
            return value, False
        return {"truncated": True, "preview": text[: max(0, limit - 64)]}, True
    text = str(value)
    if len(text) <= limit:
        return value, False
    return ellipsize(text, limit), True


# --------------------------------------------------------------------------- #
# Schema inference from Python signatures
# --------------------------------------------------------------------------- #


def _json_type(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation into a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    origin = get_origin(annotation)
    if origin is Literal:
        values = list(get_args(annotation))
        return enum_schema(values)
    if origin in {list, set, tuple, frozenset, Iterable, Sequence}:
        args = get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return array_schema(item)
    if origin in {dict, Mapping}:
        return {"type": "object"}
    if origin is not None and type(None) in get_args(annotation):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        schema = _json_type(args[0]) if args else {"type": "string"}
        schema = dict(schema)
        schema["nullable"] = True
        return schema
    mapped = _PY_TO_JSON.get(annotation)
    if mapped:
        return {"type": mapped}
    if isinstance(annotation, type) and issubclass(annotation, Path):
        return {"type": "string", "format": "path"}
    return {"type": "string"}


def parse_docstring(doc: str) -> tuple[str, dict[str, str], str]:
    """Split a docstring into ``(summary, arg_descriptions, long_description)``.

    Understands the Google style ``Args:`` block, which is what MAgent's own
    tools use, and degrades gracefully for anything else.
    """
    if not doc:
        return "", {}, ""
    lines = inspect.cleandoc(doc).split("\n")
    summary_parts: list[str] = []
    arg_docs: dict[str, str] = {}
    body: list[str] = []
    mode = "summary"
    for line in lines:
        if _ARGS_SECTION.match(line):
            mode = "args"
            continue
        if _RETURNS_SECTION.match(line):
            mode = "body"
            continue
        if mode == "args":
            match = _ARG_LINE.match(line)
            if match:
                arg_docs[match.group(1)] = match.group(2).strip()
                continue
            if line.strip() and arg_docs and line.startswith(("    ", "\t")):
                last = list(arg_docs)[-1]
                arg_docs[last] = f"{arg_docs[last]} {line.strip()}".strip()
                continue
            if not line.strip():
                continue
            mode = "body"
        if mode == "summary":
            if line.strip():
                summary_parts.append(line.strip())
            elif summary_parts:
                mode = "body"
        elif mode == "body":
            body.append(line)
    summary = " ".join(summary_parts).strip()
    if len(summary) > 400:
        summary = summary[:397] + "..."
    return summary, arg_docs, "\n".join(body).strip()


def schema_from_function(fn: ToolHandler) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Derive a JSON Schema, description, and argument docs from a function."""
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 - forward refs that cannot resolve
        hints = {}
    summary, arg_docs, _ = parse_docstring(fn.__doc__ or "")
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls", "context", "ctx"}:
            continue
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        annotation = hints.get(name, parameter.annotation)
        schema = _json_type(annotation)
        if name in arg_docs:
            schema["description"] = arg_docs[name]
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            default = parameter.default
            if isinstance(default, (str, int, float, bool, type(None))):
                schema["default"] = default
            elif isinstance(default, (list, tuple, dict, set)):
                schema["default"] = json.loads(json.dumps(default, default=list))
        properties[name] = schema
    return object_schema(properties, required or None), summary, arg_docs


def tool_spec_from_function(fn: ToolHandler, **overrides: Any) -> ToolSpec:
    """Build a :class:`ToolSpec` from a decorated function."""
    parameters, summary, arg_docs = schema_from_function(fn)
    name = str(overrides.pop("name", "") or snake_case(fn.__name__))
    description = str(overrides.pop("description", "") or summary or name)
    returns = overrides.pop("returns", "")
    if not returns:
        hint = None
        try:
            hint = get_type_hints(fn).get("return")
        except Exception:  # noqa: BLE001
            hint = None
        returns = _json_type(hint).get("type", "any") if hint else "any"
    return ToolSpec(name=name, description=description, parameters=parameters,
                    returns=returns, metadata={"arg_docs": arg_docs, "module": fn.__module__ or ""},
                    **overrides)


def tool(
    _fn: ToolHandler | None = None,
    *,
    name: str = "",
    description: str = "",
    category: str = "general",
    risk: str = "safe",
    destructive: bool = False,
    network: bool = False,
    filesystem: bool = False,
    executes_code: bool = False,
    timeout: float = 60.0,
    output_limit: int = 32000,
    tags: list[str] | None = None,
    parallel_safe: bool = True,
    examples: list[dict[str, Any]] | None = None,
    aliases: list[str] | None = None,
    hidden: bool = False,
    requires: list[str] | None = None,
    requires_approval: bool = False,
    **extra_spec: Any,
) -> Any:
    """Decorator turning a function into a registered :class:`Tool`.

    The decorated name keeps working as a plain function; it gains a ``.tool``
    attribute holding the :class:`Tool` instance so registries can pick it up.
    """

    def decorate(fn: ToolHandler) -> ToolHandler:
        spec = tool_spec_from_function(
            fn,
            name=name,
            description=description,
            category=category,
            risk=risk,
            destructive=destructive,
            network=network,
            filesystem=filesystem,
            executes_code=executes_code,
            timeout=timeout,
            output_limit=output_limit,
            tags=list(tags or []),
            parallel_safe=parallel_safe,
            examples=list(examples or []),
            aliases=list(aliases or []),
            hidden=hidden,
            requires=list(requires or []),
            requires_approval=requires_approval,
            **extra_spec,
        )
        instance = Tool(spec=spec, handler=fn)
        fn.tool = instance  # type: ignore[attr-defined]
        fn.tool_spec = spec  # type: ignore[attr-defined]
        return fn

    if _fn is not None and callable(_fn):
        return decorate(_fn)
    return decorate


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ToolRegistry:
    """A named, categorized collection of tools with dispatch helpers."""

    def __init__(self, *, validator: SchemaValidator | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        self.validator = validator or SchemaValidator()
        self.dispatch_count = 0
        self.dispatch_errors = 0
        self.unknown_calls: Counter = Counter()

    # -- registration --------------------------------------------------------

    def register(self, item: Tool | ToolHandler, *, replace: bool = False) -> Tool:
        """Register a :class:`Tool` or a ``@tool`` decorated function."""
        instance = item if isinstance(item, Tool) else getattr(item, "tool", None)
        if instance is None:
            if callable(item):
                instance = Tool(spec=tool_spec_from_function(item), handler=item)
            else:
                raise ToolError(f"Cannot register {item!r} as a tool")
        instance.validator = self.validator
        with self._lock:
            if instance.name in self._tools and not replace:
                raise ToolError(f"Tool {instance.name!r} is already registered")
            self._tools[instance.name] = instance
            for alias in instance.spec.aliases:
                self._aliases[alias] = instance.name
        return instance

    def register_all(self, items: Iterable[Tool | ToolHandler], *, replace: bool = True) -> list[Tool]:
        """Register many tools at once."""
        return [self.register(item, replace=replace) for item in items]

    def register_module(self, module: types.ModuleType, *, replace: bool = True) -> list[Tool]:
        """Register every ``@tool`` decorated function found in a module."""
        registered: list[Tool] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            member = getattr(module, name)
            if callable(member) and hasattr(member, "tool"):
                registered.append(self.register(member.tool, replace=replace))
        return registered

    def unregister(self, name: str) -> bool:
        """Remove a tool by name or alias; report whether it existed."""
        with self._lock:
            resolved = self._resolve(name)
            if resolved is None:
                return False
            instance = self._tools.pop(resolved, None)
            if instance is None:
                return False
            for alias, target in list(self._aliases.items()):
                if target == resolved:
                    del self._aliases[alias]
            return True

    def clear(self) -> None:
        """Remove every registered tool."""
        with self._lock:
            self._tools.clear()
            self._aliases.clear()

    # -- lookup --------------------------------------------------------------

    def _resolve(self, name: str) -> str | None:
        """Return the canonical tool name for a name or alias."""
        if name in self._tools:
            return name
        canonical = self._aliases.get(name)
        return canonical if canonical in self._tools else None

    def get(self, name: str) -> Tool | None:
        """Return one tool by name or alias."""
        with self._lock:
            resolved = self._resolve(name)
            return self._tools.get(resolved) if resolved else None

    def require(self, name: str) -> Tool:
        """Return one tool or raise :class:`ToolNotFoundError` with suggestions."""
        instance = self.get(name)
        if instance is None:
            from ..utils.diff import closest_match

            suggestion = closest_match(name, self.names())
            hint = f" Did you mean {suggestion!r}?" if suggestion else ""
            self.unknown_calls[name] += 1
            raise ToolNotFoundError(
                f"Unknown tool {name!r}.{hint} Registered: {', '.join(sorted(self.names())) or 'none'}"
            )
        return instance

    def __contains__(self, name: object) -> bool:
        """Support ``"read_file" in registry``."""
        return isinstance(name, str) and self.get(name) is not None

    def __len__(self) -> int:
        """Return how many tools are registered."""
        with self._lock:
            return len(self._tools)

    def names(self) -> list[str]:
        """Return every registered tool name."""
        with self._lock:
            return sorted(self._tools)

    def tools(self) -> list[Tool]:
        """Return every registered tool."""
        with self._lock:
            return [self._tools[name] for name in sorted(self._tools)]

    def by_category(self) -> dict[str, list[Tool]]:
        """Group registered tools by category."""
        grouped: dict[str, list[Tool]] = defaultdict(list)
        for instance in self.tools():
            grouped[instance.category].append(instance)
        return dict(grouped)

    def categories(self) -> list[str]:
        """Return the distinct categories present."""
        return sorted({instance.category for instance in self.tools()})

    def filter(
        self,
        *,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        max_risk: str | None = None,
        network: bool | None = None,
        filesystem: bool | None = None,
        hidden: bool | None = False,
    ) -> list[Tool]:
        """Return the subset of tools matching all supplied constraints."""
        allowed = set(enabled) if enabled else None
        blocked = set(disabled or ())
        risk_ceiling = RISK_LEVELS.index(max_risk) if max_risk in RISK_LEVELS else None
        out: list[Tool] = []
        for instance in self.tools():
            spec = instance.spec
            if allowed is not None and instance.name not in allowed:
                continue
            if instance.name in blocked:
                continue
            if categories and spec.category not in categories:
                continue
            if risk_ceiling is not None and RISK_LEVELS.index(spec.risk) > risk_ceiling:
                continue
            if network is not None and spec.network is not network:
                continue
            if filesystem is not None and spec.filesystem is not filesystem:
                continue
            if hidden is not None and spec.hidden is not hidden:
                continue
            out.append(instance)
        return out

    def search(
        self,
        *,
        name: str = "",
        category: str = "",
        tag: str = "",
        text: str = "",
        max_risk: str | None = None,
        include_hidden: bool = False,
    ) -> list[ToolSpec]:
        """Return tool specifications matching a free-form discovery query.

        Unlike :meth:`filter`, which takes exact constraints, this is the
        discovery entry point used by the ``list_tools`` meta tool: it matches
        substrings of the name, the description, and the tag list.
        """
        needle = name.strip().lower()
        haystack = text.strip().lower()
        wanted_tag = tag.strip().lower()
        ceiling = RISK_LEVELS.index(max_risk) if max_risk in RISK_LEVELS else None
        out: list[ToolSpec] = []
        for instance in self.tools():
            spec = instance.spec
            if spec.hidden and not include_hidden:
                continue
            if category and spec.category != category:
                continue
            if wanted_tag and wanted_tag not in [item.lower() for item in spec.tags]:
                continue
            if ceiling is not None and RISK_LEVELS.index(spec.risk) > ceiling:
                continue
            if needle and needle not in spec.name.lower() and needle not in spec.description.lower():
                continue
            if haystack:
                corpus = " ".join([spec.name, spec.description, spec.category, *spec.tags]).lower()
                if not all(word in corpus for word in haystack.split()):
                    continue
            out.append(spec)
        return sorted(out, key=lambda item: item.name)

    def suggest(self, name: str, *, limit: int = 3) -> list[str]:
        """Return the closest existing tool names to a misspelled one."""
        from ..utils.diff import closest_match
        from ..utils.text import similarity

        pool = self.names()
        if not pool:
            return []
        exact = closest_match(name, pool)
        scored = sorted(
            ((similarity(name, candidate), candidate) for candidate in pool), reverse=True
        )
        ordered: list[str] = []
        if exact:
            ordered.append(exact)
        for _, candidate in scored:
            if candidate not in ordered:
                ordered.append(candidate)
            if len(ordered) >= max(1, limit):
                break
        return ordered

    def count(self) -> int:
        """Return how many tools are registered."""
        return len(self._tools)

    # -- model facing --------------------------------------------------------

    def schemas(self, *, include_hidden: bool = False, **filters: Any) -> list[dict[str, Any]]:
        """Return OpenAI ``tools`` entries for the filtered tool set."""
        if include_hidden:
            filters["hidden"] = None
        return [instance.to_openai() for instance in self.filter(**filters)]

    def prompt_block(self, *, include_hidden: bool = False, style: str = "compact") -> str:
        """Render the tool set as text for models without native tool calling."""
        lines: list[str] = []
        for instance in self.filter(hidden=None if include_hidden else False):
            spec = instance.spec
            if style == "verbose":
                lines.append(f"### {spec.name}")
                lines.append(spec.description)
                lines.append("```json")
                lines.append(json.dumps(spec.parameters, indent=2, sort_keys=True))
                lines.append("```")
            else:
                params = ", ".join(
                    f"{key}:{(value or {}).get('type', 'any')}"
                    + ("" if key not in spec.required_parameters() else "*")
                    for key, value in sorted((spec.parameters.get("properties") or {}).items())
                )
                lines.append(f"- {spec.name}({params}) - {spec.description}")
        return "\n".join(lines)

    def describe(self, name: str) -> dict[str, Any]:
        """Return the full specification dictionary for one tool."""
        return self.require(name).to_dict()

    # -- dispatch ------------------------------------------------------------

    def dispatch(self, call: ToolCall, context: ToolContext | None = None) -> ToolResult:
        """Execute one :class:`ToolCall` and return its result."""
        self.dispatch_count += 1
        instance = self.get(call.name)
        if instance is None:
            self.dispatch_errors += 1
            self.unknown_calls[call.name] += 1
            from ..utils.diff import closest_match

            suggestion = closest_match(call.name, self.names())
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error=(
                    f"Unknown tool {call.name!r}."
                    + (f" Did you mean {suggestion!r}?" if suggestion else "")
                    + f" Available: {', '.join(self.names()) or 'none'}"
                ),
                error_code="tool_not_found",
            )
        ctx = (context or ToolContext()).sub(call_id=call.id, tool_name=instance.name,
                                             timeout=instance.spec.timeout,
                                             output_limit=instance.spec.output_limit)
        result = instance.run(call.arguments, ctx)
        if not result.ok:
            self.dispatch_errors += 1
        return result

    def dispatch_many(
        self,
        calls: Sequence[ToolCall],
        context: ToolContext | None = None,
        *,
        parallel: bool = True,
        max_workers: int = 8,
        on_result: Callable[[ToolResult], None] | None = None,
    ) -> list[ToolResult]:
        """Execute several calls, in parallel when they are all parallel safe."""
        if not calls:
            return []
        base = context or ToolContext()
        use_parallel = (
            parallel
            and len(calls) > 1
            and max_workers > 1
            and all(
                (self.get(call.name) is not None and self.get(call.name).spec.parallel_safe)  # type: ignore[union-attr]
                for call in calls
            )
        )
        results: list[ToolResult] = []
        if use_parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
                futures = {
                    pool.submit(
                        self.dispatch,
                        call,
                        base.sub(call_id=call.id, tool_name=call.name, step_index=index),
                    ): call
                    for index, call in enumerate(calls)
                }
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
            order = {call.id: index for index, call in enumerate(calls)}
            results.sort(key=lambda item: order.get(item.call_id, len(order)))
            return results
        for index, call in enumerate(calls):
            result = self.dispatch(call, base.sub(call_id=call.id, tool_name=call.name, step_index=index))
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    # -- telemetry -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return registry wide telemetry."""
        return {
            "tools": len(self._tools),
            "aliases": len(self._aliases),
            "categories": self.categories(),
            "dispatch_count": self.dispatch_count,
            "dispatch_errors": self.dispatch_errors,
            "unknown_calls": dict(self.unknown_calls),
            "per_tool": {name: instance.stats() for name, instance in sorted(self._tools.items())},
        }

    def reset_stats(self) -> None:
        """Zero every counter."""
        self.dispatch_count = 0
        self.dispatch_errors = 0
        self.unknown_calls.clear()
        for instance in self.tools():
            with instance._lock:
                instance.call_count = 0
                instance.error_count = 0
                instance.total_ms = 0.0
                instance.last_error = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize every tool specification."""
        return {name: instance.to_dict() for name, instance in sorted(self._tools.items())}



def default_registry() -> ToolRegistry:
    """Return a registry pre-loaded with every built-in tool."""
    from .builtin import load_all

    registry = ToolRegistry()
    load_all(registry)
    return registry


__all__ = [
    "RISK_LEVELS",
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolRegistry",
    "ToolSpec",
    "default_registry",
    "object_schema",
    "parse_docstring",
    "schema_from_function",
    "string_schema",
    "tool",
    "tool_spec_from_function",
]
