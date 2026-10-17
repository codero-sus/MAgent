"""LLM backend abstraction.

MAgent never talks to a model directly; it talks to an :class:`LLMBackend`.
Cortex LLMHoster is the first-class implementation, but the same interface is
served by an offline echo backend (tests, demos, CI) and by a routing backend
that fails over between hosts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import Capability, ChatResponse, Message, ModelInfo, ToolCall, Usage

GenerationParams = dict[str, Any]
"""Loose typing for provider specific sampling parameters."""


@dataclass(slots=True)
class StreamChunk:
    """One incremental piece of a streamed completion."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str = ""
    done: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    @property
    def is_final(self) -> bool:
        """Report whether the stream has ended with this chunk."""
        return self.done or self.finish_reason is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk for event logs."""
        return {
            "index": self.index,
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "done": self.done,
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
        }


@dataclass(slots=True)
class CompletionRequest:
    """Everything needed for one backend call, provider agnostic."""

    messages: list[Message] = field(default_factory=list)
    model: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    timeout: float | None = None
    extra: GenerationParams = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, *, include_tools: bool = True) -> dict[str, Any]:
        """Serialize into an OpenAI-compatible request body."""
        payload: dict[str, Any] = {
            "messages": [message.to_openai() for message in self.messages],
            "stream": self.stream,
        }
        if self.model:
            payload["model"] = self.model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.stop:
            payload["stop"] = self.stop
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.user:
            payload["user"] = self.user
        if self.response_format:
            payload["response_format"] = self.response_format
        if include_tools and self.tools:
            payload["tools"] = self.tools
            if self.tool_choice is not None:
                payload["tool_choice"] = self.tool_choice
        payload.update(self.extra)
        return payload

    def signature(self) -> str:
        """Return a cache key covering every semantically relevant field."""
        import json

        relevant = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "response_format": self.response_format,
            "extra": self.extra,
            "messages": [message.to_openai() for message in self.messages],
        }
        return json.dumps(relevant, sort_keys=True, default=str, ensure_ascii=False)

    def clone(self, **overrides: Any) -> CompletionRequest:
        """Return a copy with selected fields replaced."""
        import dataclasses

        return dataclasses.replace(self, **overrides)


@runtime_checkable
class LLMBackend(Protocol):
    """The contract every model backend must satisfy."""

    name: str
    """Short backend identifier used in logs and events."""

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Run a non-streaming completion."""
        ...

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Run a streaming completion, yielding chunks as they arrive."""
        ...

    def models(self) -> list[ModelInfo]:
        """List the models this backend can serve."""
        ...

    def health(self) -> dict[str, Any]:
        """Return a cheap liveness/readiness probe result."""
        ...

    def close(self) -> None:
        """Release sockets, threads, and file handles."""
        ...


class BaseBackend:
    """Convenience base class implementing the parts every backend shares."""

    name = "base"
    """Overridden by subclasses."""

    def __init__(self, *, default_model: str = "", default_timeout: float = 60.0) -> None:
        self.default_model = default_model
        self.default_timeout = default_timeout
        self.calls = 0
        self.errors = 0
        self.total_usage = Usage()
        self.started_at = time.time()

    # -- lifecycle -----------------------------------------------------------

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """Run a completion; subclasses must override."""
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion; the default collects :meth:`complete` output."""
        response = self.complete(request)
        yield StreamChunk(
            text=response.text,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            model=response.model,
            done=True,
            raw=response.raw,
        )

    def models(self) -> list[ModelInfo]:
        """List available models; backends without discovery return one entry."""
        if not self.default_model:
            return []
        return [ModelInfo(id=self.default_model)]

    def health(self) -> dict[str, Any]:
        """Report backend liveness without touching the network."""
        return {"backend": self.name, "ok": True, "models": len(self.models())}

    def close(self) -> None:
        """Release resources; the base implementation is a no-op."""

    # -- bookkeeping ---------------------------------------------------------

    def resolve_model(self, model: str | None) -> str:
        """Return the model to use, falling back to the backend default."""
        return model or self.default_model

    def record(self, response: ChatResponse) -> None:
        """Update call counters and cumulative usage."""
        self.calls += 1
        self.total_usage.add(response.usage)

    def record_error(self, error: BaseException) -> None:
        """Record a failed call."""
        self.errors += 1

    def stats(self) -> dict[str, Any]:
        """Return backend telemetry."""
        return {
            "backend": self.name,
            "calls": self.calls,
            "errors": self.errors,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "usage": self.total_usage.to_dict(),
            "default_model": self.default_model,
        }

    def __enter__(self) -> BaseBackend:
        """Support context manager usage."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the backend when leaving a ``with`` block."""
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} model={self.default_model!r}>"


@dataclass(slots=True)
class CapabilityRequest:
    """A request for a backend that can serve a given capability."""

    capability: Capability = Capability.CHAT
    model: str = ""
    prefer_running: bool = True

    def matches(self, info: ModelInfo) -> bool:
        """Report whether a model satisfies this capability request."""
        if self.model and info.id != self.model:
            return False
        return info.supports(self.capability)


StreamHandler = Callable[[StreamChunk], None]
"""Callback signature for incremental stream consumers."""


def collect_stream(
    chunks: Iterator[StreamChunk], on_chunk: StreamHandler | None = None
) -> ChatResponse:
    """Drain a stream into a single :class:`ChatResponse`.

    Used by callers that asked for streaming telemetry but want a final string,
    and by backends whose upstream does not support streaming at all.
    """
    text_parts: list[str] = []
    tool_fragments: list[dict[str, Any]] = []
    finish_reason = "stop"
    usage = Usage()
    model = ""
    last_raw: dict[str, Any] = {}
    started = time.perf_counter()
    for chunk in chunks:
        if on_chunk is not None:
            on_chunk(chunk)
        if chunk.text:
            text_parts.append(chunk.text)
        if chunk.model:
            model = chunk.model
        if chunk.usage:
            usage = chunk.usage
        if chunk.raw:
            last_raw = chunk.raw
            tool_fragments.extend(call.to_dict() for call in chunk.tool_calls)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
    from ..http.sse import merge_tool_call_deltas
    from ..types import ToolCall as _ToolCall

    calls = [_ToolCall.from_dict(raw) for raw in tool_fragments] if tool_fragments else []
    if not calls and last_raw:
        merge = merge_tool_call_deltas(_extract_streamed_calls(last_raw))
        calls = [_ToolCall.from_dict(raw) for raw in merge]
    return ChatResponse(
        text="".join(text_parts),
        model=model,
        usage=usage,
        tool_calls=calls,
        finish_reason=finish_reason,
        raw=last_raw,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _extract_streamed_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull raw streamed tool call fragments out of a stored payload."""
    raw_calls = payload.get("__streamed_tool_calls__")
    if isinstance(raw_calls, list):
        return [call for call in raw_calls if isinstance(call, dict)]
    return []


def estimate_message_chars(messages: Sequence[Message]) -> int:
    """Return the total character count of a message list."""
    return sum(message.char_count() for message in messages)


__all__ = [
    "BaseBackend",
    "CapabilityRequest",
    "CompletionRequest",
    "GenerationParams",
    "LLMBackend",
    "StreamChunk",
    "StreamHandler",
    "collect_stream",
    "estimate_message_chars",
]
