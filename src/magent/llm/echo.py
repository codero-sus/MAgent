"""Deterministic offline backends for tests, demos, CI, and dry runs.

These backends never touch the network. They exist so that the whole agent
stack - loop, tools, memory, guardrails, CLI, server - can be exercised on a
machine with no Cortex installed, which is exactly how MAgent's own test suite
runs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from ..types import Capability, ChatResponse, ModelInfo, ToolCall, Usage
from .base import BaseBackend, CompletionRequest, StreamChunk

ScriptEntry = tuple[str | re.Pattern[str], Any]
"""A ``(matcher, reply)`` pair in a scripted backend."""


def _tokens_for(text: str) -> int:
    """Estimate a token count for offline usage accounting."""
    return max(1, len(text) // 4)


@dataclass
class EchoBackend(BaseBackend):
    """A backend that reflects the prompt back with deterministic decoration.

    Useful as a smoke-test double: it proves the request pipeline (message
    serialization, tool schemas, streaming, retries, caching) works end to end
    without a model.
    """

    name: str = "echo"
    prefix: str = "echo:"
    delay: float = 0.0
    models_list: list[str] = field(default_factory=lambda: ["echo-small"])
    chunk_size: int = 24
    fail_times: int = 0
    latency_jitter: float = 0.0
    _failures: int = 0

    def __post_init__(self) -> None:
        """Initialize the shared backend counters."""
        super().__init__(default_model=self.models_list[0] if self.models_list else "echo-small")

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Return a deterministic echo of the last user message."""
        if self.delay:
            time.sleep(self.delay)
        if self._failures < self.fail_times:
            self._failures += 1
            from ..errors import ServerError

            raise ServerError("echo backend injected failure", status=503)
        model = self.resolve_model(request.model) or self.default_model
        last = ""
        for message in reversed(request.messages):
            if message.role.value == "user":
                last = message.text()
                break
        text = f"{self.prefix}{last}"
        if request.response_format and str(request.response_format.get("type")) == "json_object":
            text = json.dumps({"echo": last, "model": model})
        usage = Usage(
            prompt_tokens=sum(_tokens_for(message.text()) for message in request.messages),
            completion_tokens=_tokens_for(text),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        response = ChatResponse(
            text=text,
            model=model,
            usage=usage,
            finish_reason="stop",
            raw={"object": "chat.completion", "id": f"echo-{usage.total_tokens}"},
            latency_ms=self.delay * 1000.0,
        )
        self.record(response)
        return response

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Yield the echo reply in fixed-size chunks."""
        response = self.complete(request)
        text = response.text
        size = max(1, self.chunk_size)
        index = 0
        for start in range(0, len(text), size):
            index += 1
            yield StreamChunk(text=text[start : start + size], model=response.model, index=index)
        yield StreamChunk(
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model,
            done=True,
            index=index + 1,
        )

    def models(self) -> list[ModelInfo]:
        """Return the synthetic model list."""
        return [
            ModelInfo(
                id=model_id,
                runtime="echo",
                capabilities=[Capability.CHAT.value, Capability.TOOL_CALLING.value],
                default=index == 0,
                status="running",
            )
            for index, model_id in enumerate(self.models_list)
        ]

    def health(self) -> dict[str, Any]:
        """Report always-healthy status."""
        return {"backend": self.name, "ok": True, "models": len(self.models_list), "offline": True}


@dataclass
class ScriptedBackend(BaseBackend):
    """A backend that replays a fixed list of replies in order.

    Accepts plain strings, callables taking the request, or ``(matcher, reply)``
    pairs where the matcher is a substring or compiled regex tested against the
    last user message. Exhausting the script repeats the final entry, which
    keeps long agent loops from crashing tests.
    """

    name: str = "scripted"
    script: Sequence[Any] = ()
    models_list: list[str] = field(default_factory=lambda: ["scripted-model"])
    delay: float = 0.0
    _cursor: int = 0
    calls_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize shared counters and default model."""
        super().__init__(default_model=self.models_list[0] if self.models_list else "scripted-model")

    def _next_entry(self, request: CompletionRequest) -> Any:
        """Pick the reply for this call, honoring matchers when present."""
        last_user = ""
        for message in reversed(request.messages):
            if message.role.value == "user":
                last_user = message.text()
                break
        entries = list(self.script)
        if not entries:
            return ""
        for entry in entries:
            if isinstance(entry, tuple) and len(entry) == 2:
                matcher, reply = entry
                if isinstance(matcher, re.Pattern):
                    if matcher.search(last_user):
                        return reply
                elif isinstance(matcher, str) and matcher.lower() in last_user.lower():
                    return reply
        index = min(self._cursor, len(entries) - 1)
        self._cursor += 1
        entry = entries[index]
        if isinstance(entry, tuple) and len(entry) == 2:
            return entry[1]
        return entry

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Return the next scripted reply."""
        if self.delay:
            time.sleep(self.delay)
        entry = self._next_entry(request)
        model = self.resolve_model(request.model) or self.default_model
        tool_calls: list[ToolCall] = []
        if callable(entry):
            produced = entry(request)
        else:
            produced = entry
        if isinstance(produced, dict):
            text = str(produced.get("text", ""))
            raw_calls = produced.get("tool_calls") or []
            for index, raw in enumerate(raw_calls):
                if isinstance(raw, ToolCall):
                    tool_calls.append(raw)
                elif isinstance(raw, dict):
                    tool_calls.append(
                        ToolCall(
                            id=str(raw.get("id") or f"call_{index:02d}"),
                            name=str(raw.get("name", "")),
                            arguments=dict(raw.get("arguments") or {}),
                        )
                    )
            finish = str(produced.get("finish_reason") or ("tool_calls" if tool_calls else "stop"))
        else:
            text = str(produced)
            finish = "stop"
        self.calls_log.append({"model": model, "text": text[:200], "tools": [c.name for c in tool_calls]})
        usage = Usage(
            prompt_tokens=sum(_tokens_for(message.text()) for message in request.messages),
            completion_tokens=_tokens_for(text),
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        response = ChatResponse(
            text=text,
            model=model,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish,
            raw={"object": "chat.completion", "scripted": True},
        )
        self.record(response)
        return response

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream the scripted reply word by word."""
        response = self.complete(request)
        words = response.text.split(" ")
        for index, word in enumerate(words):
            piece = word if index == 0 else f" {word}"
            yield StreamChunk(text=piece, model=response.model, index=index + 1)
        yield StreamChunk(
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model,
            done=True,
            index=len(words) + 1,
        )

    def models(self) -> list[ModelInfo]:
        """Return the scripted model list."""
        return [
            ModelInfo(id=model_id, runtime="scripted",
                      capabilities=[Capability.CHAT.value], default=index == 0, status="running")
            for index, model_id in enumerate(self.models_list)
        ]

    def health(self) -> dict[str, Any]:
        """Report always-healthy status including script position."""
        return {
            "backend": self.name,
            "ok": True,
            "script_length": len(self.script),
            "cursor": self._cursor,
            "offline": True,
        }


@dataclass
class ToolScriptBackend(ScriptedBackend):
    """A scripted backend that emits a tool call then a final answer.

    This is the workhorse for agent-loop tests: step one asks for a tool, step
    two consumes the tool result and answers.
    """

    name: str = "tool-script"
    tool_name: str = "calculator"
    tool_arguments: dict[str, Any] = field(default_factory=lambda: {"expression": "2+2"})
    final_text: str = "The answer is 4."

    def __post_init__(self) -> None:
        """Build the two-step script from the configured tool call."""
        super().__post_init__()
        if not self.script:
            self.script = [
                {
                    "text": "",
                    "tool_calls": [
                        {"name": self.tool_name, "arguments": self.tool_arguments}
                    ],
                    "finish_reason": "tool_calls",
                },
                self.final_text,
            ]


def hash_backend_reply(text: str) -> str:
    """Return a stable digest of a reply, for golden-file tests."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


ReplyFn = Callable[[CompletionRequest], str]
"""Signature for programmable offline backends."""


class LambdaBackend(BaseBackend):
    """A backend whose reply is computed by a user supplied function."""

    name = "lambda"

    def __init__(self, fn: ReplyFn, *, model: str = "lambda-model") -> None:
        super().__init__(default_model=model)
        self.fn = fn

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Invoke the wrapped function to produce the reply."""
        text = self.fn(request)
        usage = Usage(prompt_tokens=_tokens_for(request.signature()), completion_tokens=_tokens_for(text))
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        response = ChatResponse(text=text, model=self.default_model, usage=usage, finish_reason="stop")
        self.record(response)
        return response

    def models(self) -> list[ModelInfo]:
        """Return the single lambda model."""
        return [ModelInfo(id=self.default_model, runtime="lambda", capabilities=[Capability.CHAT.value])]


__all__ = [
    "EchoBackend",
    "LambdaBackend",
    "ReplyFn",
    "ScriptEntry",
    "ScriptedBackend",
    "ToolScriptBackend",
    "hash_backend_reply",
]
