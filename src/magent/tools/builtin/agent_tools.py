"""Meta tools: the agent inspecting and steering its own tool layer.

``list_tools`` and ``describe_tool`` let a model discover what it can do at run
time instead of relying on a static prompt. ``use_tool`` enables composition -
one tool call dispatching another - which is how multi-step operations get
expressed without a second LLM round trip. ``ask_llm`` gives a tool-using agent
a cheap way to consult the model for a judgment call in the middle of a run.
"""

from __future__ import annotations

import json
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import RISK_LEVELS, ToolContext, tool
from ...types import ToolCall
from ...utils.hashing import short_hash
from ...utils.json_utils import loads_lenient
from ...utils.text import chunk_text, ellipsize


def new_call_id() -> str:
    """Return a short unique identifier for a nested tool call."""
    import time

    return f"call_{short_hash(f'{time.time_ns()}')}"


@tool(category="meta", risk="safe", tags=["tools", "discovery", "core"])
def list_tools(
    category: str = "",
    tag: str = "",
    name_filter: str = "",
    include_schemas: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """List the tools available in this run.

    Args:
        category: only tools in this category
        tag: only tools carrying this tag
        name_filter: substring match on the tool name
        include_schemas: include the full JSON parameter schema
    """
    ctx = context or ToolContext()
    registry = ctx.registry
    if registry is None:
        raise ToolExecutionError("No tool registry available in this context")
    items = registry.search(name=name_filter, category=category, tag=tag)
    entries: list[dict[str, Any]] = []
    for spec in items:
        entry: dict[str, Any] = {
            "name": spec.name,
            "description": spec.description.split("\n")[0].strip(),
            "category": spec.category,
            "risk": spec.risk,
            "tags": list(spec.tags),
            "requires": list(spec.requires),
        }
        if include_schemas:
            entry["parameters"] = spec.parameters
        entries.append(entry)
    by_category: dict[str, int] = {}
    for entry in entries:
        by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1
    return {
        "count": len(entries),
        "total": registry.count(),
        "categories": by_category,
        "tools": entries,
        "text": "\n".join(
            f"- {entry['name']} ({entry['risk']}): {entry['description']}" for entry in entries
        ),
    }


@tool(category="meta", risk="safe", tags=["tools", "schema"])
def describe_tool(name: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Return the full specification and example usage for one tool.

    Args:
        name: the tool name
    """
    ctx = context or ToolContext()
    registry = ctx.registry
    if registry is None:
        raise ToolExecutionError("No tool registry available in this context")
    instance = registry.get(name)
    if instance is None:
        suggestions = registry.suggest(name, limit=5)
        raise ToolExecutionError(
            f"No tool named {name!r}"
            + (f"; did you mean {', '.join(suggestions)}?" if suggestions else "?")
        )
    spec = instance.spec
    properties = spec.parameters.get("properties", {})
    required = set(spec.parameters.get("required", []))
    arguments = ", ".join(
        f"{key}{'?' if key not in required else ''}: {value.get('type', 'any')}"
        for key, value in properties.items()
    )
    example = {key: _example_value(value) for key, value in properties.items() if key in required}
    return {
        "name": spec.name,
        "description": spec.description,
        "summary": spec.description.split("\n")[0].strip(),
        "category": spec.category,
        "risk": spec.risk,
        "risk_rank": RISK_LEVELS.index(spec.risk) if spec.risk in RISK_LEVELS else -1,
        "tags": list(spec.tags),
        "destructive": spec.destructive,
        "network": spec.network,
        "filesystem": spec.filesystem,
        "executes_code": spec.executes_code,
        "parallel_safe": spec.parallel_safe,
        "deterministic": spec.deterministic,
        "timeout": spec.timeout,
        "output_limit": spec.output_limit,
        "requires": list(spec.requires),
        "parameters": spec.parameters,
        "signature": f"{spec.name}({arguments})",
        "example": example,
        "example_json": json.dumps(example, ensure_ascii=False),
    }


def _example_value(schema: dict[str, Any]) -> Any:
    """Produce a plausible example value for a parameter schema."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "string":
        return schema.get("default", "example")
    if kind == "integer":
        return schema.get("default", 1)
    if kind == "number":
        return schema.get("default", 1.0)
    if kind == "boolean":
        return schema.get("default", False)
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return None


@tool(category="meta", risk="execute", tags=["tools", "compose"])
def use_tool(
    name: str, arguments: str = "{}", dry_run: bool = False, context: ToolContext | None = None
) -> dict[str, Any]:
    """Dispatch another tool by name, enabling composition inside one call.

    Args:
        name: the tool to invoke
        arguments: JSON object of arguments for that tool
        dry_run: validate and report without executing side effects
    """
    ctx = context or ToolContext()
    registry = ctx.registry
    if registry is None:
        raise ToolExecutionError("No tool registry available in this context")
    if name == "use_tool":
        raise ToolExecutionError("use_tool cannot call itself")
    instance = registry.get(name)
    if instance is None:
        suggestions = registry.suggest(name, limit=5)
        raise ToolExecutionError(
            f"No tool named {name!r}"
            + (f"; did you mean {', '.join(suggestions)}?" if suggestions else "?")
        )
    parsed = loads_lenient(arguments) if arguments else {}
    if not isinstance(parsed, dict):
        raise ToolExecutionError("arguments must be a JSON object")
    nested = ctx.sub(dry_run=dry_run or ctx.dry_run)
    nested.extras["parent_tool"] = "use_tool"
    result = registry.dispatch(ToolCall(id=new_call_id(), name=name, arguments=parsed), nested)
    return {
        "tool": name,
        "ok": result.ok,
        "dry_run": nested.dry_run,
        "duration_ms": result.duration_ms,
        "content": result.content,
        "error": result.error,
        "metadata": result.metadata,
        "truncated": result.truncated,
    }


@tool(category="meta", risk="safe", tags=["llm", "consult"])
def ask_llm(
    prompt: str,
    system: str = "",
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_output: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Ask the configured LLM a question in the middle of a tool run.

    Useful for classification, extraction, or a judgment call that is cheaper
    than a full agent turn.

    Args:
        prompt: the question or instruction
        system: optional system prompt override
        model: optional model override
        temperature: sampling temperature
        max_tokens: maximum tokens to generate
        json_output: request a JSON object response
    """
    ctx = context or ToolContext()
    backend = ctx.backend
    if backend is None:
        raise ToolExecutionError("No LLM backend available in this context")
    messages: list[dict[str, Any]] = []
    effective_system = system or (
        "You are a precise assistant embedded inside an agent tool call. Answer concisely."
        + (" Reply with a single JSON object." if json_output else "")
    )
    messages.append({"role": "system", "content": effective_system})
    messages.append({"role": "user", "content": prompt})
    response = backend.chat(
        messages,
        model=model or "",
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"} if json_output else None,
    )
    text = response.text or ""
    payload: dict[str, Any] = {
        "text": text,
        "model": response.model,
        "finish_reason": response.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "latency_ms": response.latency_ms,
    }
    if json_output:
        parsed = loads_lenient(text)
        payload["json"] = parsed
        payload["parsed_ok"] = parsed is not None
    return payload


@tool(category="meta", risk="safe", tags=["plan", "decompose"])
def make_plan(goal: str, steps: str = "", max_steps: int = 12) -> dict[str, Any]:
    """Record or refine an explicit plan for the current run.

    Args:
        goal: the objective the plan should achieve
        steps: JSON array of step descriptions, empty to derive from the goal
        max_steps: cap on the number of steps retained
    """
    parsed = loads_lenient(steps) if steps else []
    if isinstance(parsed, str):
        parsed = [line.strip() for line in parsed.splitlines() if line.strip()]
    if not isinstance(parsed, list):
        raise ToolExecutionError("steps must be a JSON array of strings")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(parsed[:max_steps], start=1):
        if isinstance(item, dict):
            entries.append(
                {
                    "id": index,
                    "title": str(item.get("title") or item.get("step") or f"step {index}"),
                    "detail": str(item.get("detail", "")),
                    "tool": str(item.get("tool", "")),
                    "status": str(item.get("status", "pending")),
                }
            )
        else:
            entries.append(
                {"id": index, "title": str(item), "detail": "", "tool": "", "status": "pending"}
            )
    text = "\n".join(
        f"{entry['id']}. {entry['title']}" + (f" - {entry['detail']}" if entry["detail"] else "")
        for entry in entries
    )
    return {
        "goal": goal,
        "steps": entries,
        "count": len(entries),
        "max_steps": max_steps,
        "truncated": len(parsed) > max_steps,
        "text": f"Goal: {goal}\n{text}" if entries else f"Goal: {goal}\n(no steps recorded)",
    }


@tool(category="meta", risk="safe", tags=["context", "summarize"])
def summarize_for_context(
    text: str, max_chars: int = 4000, strategy: str = "head_tail"
) -> dict[str, Any]:
    """Compress long text so it fits in a prompt without losing the ends.

    Args:
        text: the text to compress
        max_chars: character budget for the result
        strategy: head, tail, head_tail, or chunk
    """
    if strategy not in {"head", "tail", "head_tail", "chunk"}:
        raise ToolExecutionError("strategy must be head, tail, head_tail, or chunk")
    original = len(text)
    if original <= max_chars:
        return {
            "text": text,
            "chars": original,
            "original_chars": original,
            "compressed": False,
            "strategy": strategy,
            "chunks": 1,
        }
    if strategy == "head":
        result = ellipsize(text, max_chars)
    elif strategy == "tail":
        result = "..." + text[-(max_chars - 3) :]
    elif strategy == "head_tail":
        budget = max(200, (max_chars - 40) // 2)
        result = text[:budget] + "\n\n[... middle omitted ...]\n\n" + text[-budget:]
    else:
        pieces = chunk_text(text, max_chars=max_chars, overlap=0)
        result = pieces[0]
        return {
            "text": result,
            "chars": len(result),
            "original_chars": original,
            "compressed": True,
            "strategy": strategy,
            "chunks": len(pieces),
            "chunk_sizes": [len(piece) for piece in pieces],
        }
    return {
        "text": result,
        "chars": len(result),
        "original_chars": original,
        "compressed": True,
        "strategy": strategy,
        "chunks": 1,
        "reduction_percent": round(100 * (1 - len(result) / original), 1),
    }


TOOLS: list[Any] = [list_tools, describe_tool, use_tool, ask_llm, make_plan, summarize_for_context]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every meta tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "ask_llm",
    "describe_tool",
    "list_tools",
    "load",
    "make_plan",
    "new_call_id",
    "summarize_for_context",
    "use_tool",
]
