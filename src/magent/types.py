"""Core immutable data types shared across MAgent.

Everything the framework passes around is defined here so that no module has to
invent its own ad-hoc dictionaries. Types are ``@dataclass(slots=True)`` for
speed and memory, frozen where mutation would be a bug, and all of them support
lossless ``to_dict()`` / ``from_dict()`` round trips for persistence.

Wire compatibility
------------------
The dict shapes produced by :meth:`Message.to_openai` match the OpenAI chat
schema that Cortex LLMHoster exposes at ``/v1/chat/completions``, including
multimodal ``content`` arrays and ``tool_calls`` blocks.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #


def new_id(prefix: str = "id") -> str:
    """Return a short, sortable, collision-resistant identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def now_ts() -> float:
    """Return the current wall clock time in seconds since the epoch."""
    return time.time()


def iso_ts(ts: float | None = None) -> str:
    """Format a timestamp as an ISO-8601 UTC string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts() if ts is None else ts))


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Role(StrEnum):
    """Chat message roles understood by the OpenAI-compatible wire format."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"

    @classmethod
    def parse(cls, value: str) -> Role:
        """Coerce a loose string into a :class:`Role`, defaulting to USER."""
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.USER

    def is_input(self) -> bool:
        """Report whether this role counts as caller supplied context."""
        return self in (Role.SYSTEM, Role.USER, Role.DEVELOPER, Role.TOOL)


class Capability(StrEnum):
    """Model capability tags mirrored from Cortex ``capabilities`` lists."""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VISION = "vision"
    OCR = "ocr"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    AUDIO_UNDERSTANDING = "audio_understanding"
    VIDEO = "video"
    MODERATION = "moderation"
    TOOL_CALLING = "tool_calling"

    @classmethod
    def aliases(cls) -> dict[str, Capability]:
        """Return the loose-name to capability mapping Cortex also accepts."""
        return {
            "llm": cls.CHAT,
            "t2t": cls.CHAT,
            "text": cls.CHAT,
            "text-to-text": cls.CHAT,
            "vlm": cls.VISION,
            "image-to-text": cls.VISION,
            "embed": cls.EMBEDDING,
            "embeddings": cls.EMBEDDING,
            "stt": cls.AUDIO_TRANSCRIPTION,
            "asr": cls.AUDIO_TRANSCRIPTION,
            "transcribe": cls.AUDIO_TRANSCRIPTION,
            "tts": cls.AUDIO_GENERATION,
            "speech": cls.AUDIO_GENERATION,
            "t2i": cls.IMAGE_GENERATION,
            "i2i": cls.IMAGE_GENERATION,
            "img2img": cls.IMAGE_GENERATION,
            "txt2img": cls.IMAGE_GENERATION,
            "t2v": cls.VIDEO,
            "i2v": cls.VIDEO,
            "v2v": cls.VIDEO,
            "reranker": cls.RERANK,
            "mod": cls.MODERATION,
        }

    @classmethod
    def parse(cls, value: str) -> Capability | None:
        """Resolve a capability name or alias, or ``None`` when unknown."""
        key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(key)
        except ValueError:
            return cls.aliases().get(key)


class EventKind(StrEnum):
    """Kinds of event emitted by the agent event bus."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    THINKING = "agent.thinking"
    TEXT_DELTA = "agent.text_delta"
    TEXT_DONE = "agent.text_done"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    APPROVAL_REQUESTED = "guard.approval_requested"
    APPROVAL_DECIDED = "guard.approval_decided"
    GUARDRAIL_BLOCKED = "guard.blocked"
    MEMORY_WRITE = "memory.write"
    MEMORY_READ = "memory.read"
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_RETRY = "llm.retry"
    LLM_CACHE_HIT = "llm.cache_hit"
    BUDGET_WARNING = "budget.warning"
    MODEL_SWITCHED = "router.model_switched"
    PLAN_CREATED = "plan.created"
    PLAN_STEP = "plan.step"
    REFLECTION = "agent.reflection"
    LOG = "log"
    CUSTOM = "custom"


class StopReason(StrEnum):
    """Why an agent run terminated."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    BUDGET = "budget_exceeded"
    CANCELLED = "cancelled"
    ERROR = "error"
    GUARDRAIL = "guardrail"
    NEEDS_INPUT = "needs_input"
    TOOL_LIMIT = "tool_limit"

    def is_success(self) -> bool:
        """Report whether the run finished in an acceptable state."""
        return self is StopReason.COMPLETED


class ApprovalDecision(StrEnum):
    """Possible human answers to an approval prompt."""

    APPROVE = "approve"
    DENY = "deny"
    APPROVE_SESSION = "approve_session"
    EDIT = "edit"


ApprovalMode = Literal["never", "ask", "always", "dangerous"]
"""When the agent must pause for human approval before running a tool."""


Verbosity = Literal["quiet", "normal", "verbose", "debug"]
"""CLI output verbosity ladder."""


# --------------------------------------------------------------------------- #
# Content parts (multimodal)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TextPart:
    """A plain text span inside a multimodal message."""

    text: str
    type: Literal["text"] = "text"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI ``content`` part shape."""
        return {"type": "text", "text": self.text}

    def char_count(self) -> int:
        """Return the number of characters in the span."""
        return len(self.text)


@dataclass(slots=True)
class ImagePart:
    """An image reference, either a data URL or an http(s) URL."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"
    type: Literal["image_url"] = "image_url"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI ``image_url`` part shape."""
        return {"type": "image_url", "image_url": {"url": self.url, "detail": self.detail}}

    def is_inline(self) -> bool:
        """Report whether the image is embedded as a data URL."""
        return self.url.startswith("data:")

    def char_count(self) -> int:
        """Return the URL length as a rough payload proxy."""
        return len(self.url)


@dataclass(slots=True)
class AudioPart:
    """An audio reference for transcription or understanding models."""

    url: str
    format: str = "wav"
    type: Literal["input_audio"] = "input_audio"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI ``input_audio`` part shape."""
        return {"type": "input_audio", "input_audio": {"data": self.url, "format": self.format}}

    def char_count(self) -> int:
        """Return the payload length as a rough size proxy."""
        return len(self.url)


@dataclass(slots=True)
class FilePart:
    """A document reference handed to models that accept file inputs."""

    filename: str
    data: str
    mime_type: str = "application/octet-stream"
    type: Literal["file"] = "file"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI ``file`` part shape."""
        return {
            "type": "file",
            "file": {
                "filename": self.filename,
                "file_data": self.data,
                "mime_type": self.mime_type,
            },
        }

    def char_count(self) -> int:
        """Return the encoded payload length."""
        return len(self.data)


ContentPart = TextPart | ImagePart | AudioPart | FilePart
"""Union of multimodal content parts a message may hold."""


def part_from_dict(raw: dict[str, Any]) -> ContentPart:
    """Rebuild a concrete content part from its wire dictionary."""
    kind = str(raw.get("type", "text"))
    if kind == "image_url":
        payload = raw.get("image_url") or {}
        url = payload.get("url") if isinstance(payload, dict) else str(payload)
        detail = payload.get("detail", "auto") if isinstance(payload, dict) else "auto"
        return ImagePart(url=str(url or ""), detail=detail)
    if kind == "input_audio":
        payload = raw.get("input_audio") or {}
        return AudioPart(
            url=str(payload.get("data", "")),
            format=str(payload.get("format", "wav")),
        )
    if kind == "file":
        payload = raw.get("file") or {}
        return FilePart(
            filename=str(payload.get("filename", "file")),
            data=str(payload.get("file_data", "")),
            mime_type=str(payload.get("mime_type", "application/octet-stream")),
        )
    return TextPart(text=str(raw.get("text", "")))


# --------------------------------------------------------------------------- #
# Tool calls and results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the OpenAI ``tool_calls`` element shape."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments, sort_keys=True),
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolCall:
        """Parse one ``tool_calls`` element, tolerating malformed arguments."""
        function = raw.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        text = str(function.get("arguments") or "")
        parsed: dict[str, Any] = {}
        if text.strip():
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    parsed = loaded
            except json.JSONDecodeError:
                parsed = {}
        return cls(
            id=str(raw.get("id") or new_id("call")),
            name=str(function.get("name") or ""),
            arguments=parsed,
            raw_arguments=text,
        )

    def signature(self) -> str:
        """Return a stable string identifying this exact call."""
        return f"{self.name}({json.dumps(self.arguments, sort_keys=True, default=str)})"


@dataclass(slots=True)
class ToolResult:
    """The outcome of executing a :class:`ToolCall`."""

    call_id: str
    name: str
    ok: bool = True
    content: Any = None
    error: str | None = None
    error_code: str | None = None
    duration_ms: float = 0.0
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result including timing metadata."""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "error_code": self.error_code,
            "duration_ms": round(self.duration_ms, 3),
            "truncated": self.truncated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ToolResult:
        """Rebuild a result from a persisted dictionary."""
        return cls(
            call_id=str(raw.get("call_id", "")),
            name=str(raw.get("name", "")),
            ok=bool(raw.get("ok", True)),
            content=raw.get("content"),
            error=raw.get("error"),
            error_code=raw.get("error_code"),
            duration_ms=float(raw.get("duration_ms") or 0.0),
            truncated=bool(raw.get("truncated", False)),
            metadata=dict(raw.get("metadata") or {}),
        )

    def payload_text(self) -> str:
        """Render the result as the text payload of a ``tool`` role message.

        Failures are serialized rather than raised, because the model has to see
        what went wrong in order to recover from it.
        """
        if not self.ok:
            return json.dumps({"error": self.error, "code": self.error_code}, default=str)
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, default=str, ensure_ascii=False)

    def to_message(self) -> "Message":
        """Return this result as a ``tool`` role :class:`Message`."""
        return Message.tool(self)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Message:
    """One chat message, optionally multimodal and optionally tool-bearing."""

    role: Role
    content: str | list[ContentPart] = ""
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    created_at: float = field(default_factory=now_ts)
    id: str = field(default_factory=lambda: new_id("msg"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the role into a :class:`Role` member."""
        if not isinstance(self.role, Role):
            self.role = Role.parse(str(self.role))

    # -- construction helpers ------------------------------------------------

    @classmethod
    def system(cls, text: str, **meta: Any) -> Message:
        """Build a system prompt message."""
        return cls(role=Role.SYSTEM, content=text, metadata=dict(meta))

    @classmethod
    def user(cls, text: str | list[ContentPart], **meta: Any) -> Message:
        """Build a user message from text or multimodal parts."""
        return cls(role=Role.USER, content=text, metadata=dict(meta))

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None, **meta: Any) -> Message:
        """Build an assistant message, optionally carrying tool calls."""
        return cls(
            role=Role.ASSISTANT,
            content=text,
            tool_calls=list(tool_calls or []),
            metadata=dict(meta),
        )

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        """Build the ``tool`` role reply for an executed tool call."""
        return cls(
            role=Role.TOOL,
            content=result.payload_text(),
            tool_call_id=result.call_id,
            name=result.name,
            metadata={"ok": result.ok, "duration_ms": result.duration_ms},
        )

    # -- inspection ----------------------------------------------------------

    def text(self) -> str:
        """Return the concatenated plain text of the message."""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    def parts(self) -> list[ContentPart]:
        """Return content as a list of parts, wrapping bare strings."""
        if isinstance(self.content, str):
            return [TextPart(text=self.content)] if self.content else []
        return list(self.content)

    def has_media(self) -> bool:
        """Report whether the message carries non-text content."""
        return any(not isinstance(part, TextPart) for part in self.parts())

    def char_count(self) -> int:
        """Return the total character count across all parts."""
        return sum(part.char_count() for part in self.parts())

    def wants_tools(self) -> bool:
        """Report whether the assistant message requested tool execution."""
        return bool(self.tool_calls)

    # -- serialization -------------------------------------------------------

    def to_openai(self) -> dict[str, Any]:
        """Serialize into an OpenAI chat message object."""
        payload: dict[str, Any] = {"role": self.role.value}
        if isinstance(self.content, str):
            payload["content"] = self.content
        else:
            payload["content"] = [part.to_dict() for part in self.content]
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Serialize into MAgent's richer persistence shape."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": (
                self.content
                if isinstance(self.content, str)
                else [part.to_dict() for part in self.content]
            ),
            "name": self.name,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Message:
        """Rebuild a message from either persistence or OpenAI shape."""
        content = raw.get("content", "")
        if isinstance(content, list):
            parsed: str | list[ContentPart] = [
                part_from_dict(part) for part in content if isinstance(part, dict)
            ]
        else:
            parsed = "" if content is None else str(content)
        calls = [
            ToolCall.from_dict(call)
            for call in (raw.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        return cls(
            role=Role.parse(str(raw.get("role", "user"))),
            content=parsed,
            name=raw.get("name"),
            tool_calls=calls,
            tool_call_id=raw.get("tool_call_id"),
            created_at=float(raw.get("created_at") or now_ts()),
            id=str(raw.get("id") or new_id("msg")),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_json(self) -> str:
        """Serialize the message to a compact JSON string."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


def messages_from_dicts(raws: list[dict[str, Any]]) -> list[Message]:
    """Parse a list of raw dictionaries into :class:`Message` objects."""
    return [Message.from_dict(raw) for raw in raws if isinstance(raw, dict)]


# --------------------------------------------------------------------------- #
# Usage and cost accounting
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Usage:
    """Token usage counters for a single backend call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    image_count: int = 0

    def __post_init__(self) -> None:
        """Derive ``total_tokens`` when the caller supplied only the halves.

        ``from_dict`` has always filled a missing total from ``prompt +
        completion``; the constructor did not, so ``Usage(prompt_tokens=10,
        completion_tokens=5).total_tokens`` was ``0`` while the same numbers
        parsed from JSON gave ``15``. Budgets and cost estimates read this
        field, so the two paths have to agree.
        """
        if not self.total_tokens and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Usage:
        """Parse an OpenAI-style ``usage`` object, tolerating missing keys."""
        raw = raw or {}
        prompt = int(raw.get("prompt_tokens") or 0)
        completion = int(raw.get("completion_tokens") or 0)
        total = int(raw.get("total_tokens") or (prompt + completion))
        details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_tokens=int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0,
            reasoning_tokens=(
                int(completion_details.get("reasoning_tokens") or 0)
                if isinstance(completion_details, dict)
                else 0
            ),
            audio_tokens=int(raw.get("audio_tokens") or 0),
            image_count=int(raw.get("image_count") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the counters."""
        return asdict(self)

    def merge(self, other: Usage) -> Usage:
        """Return a new :class:`Usage` that is the sum of both."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            audio_tokens=self.audio_tokens + other.audio_tokens,
            image_count=self.image_count + other.image_count,
        )

    def add(self, other: Usage) -> None:
        """Mutate in place by adding another usage record."""
        merged = self.merge(other)
        self.prompt_tokens = merged.prompt_tokens
        self.completion_tokens = merged.completion_tokens
        self.total_tokens = merged.total_tokens
        self.cached_tokens = merged.cached_tokens
        self.reasoning_tokens = merged.reasoning_tokens
        self.audio_tokens = merged.audio_tokens
        self.image_count = merged.image_count

    def is_empty(self) -> bool:
        """Report whether no tokens were accounted at all."""
        return self.total_tokens == 0 and self.prompt_tokens == 0


@dataclass(slots=True)
class CostEstimate:
    """A local cost estimate; Cortex hosting has no per-token cloud charge."""

    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    currency: str = "USD"
    source: str = "local"

    @property
    def total(self) -> float:
        """Return the summed cost."""
        return round(self.prompt_cost + self.completion_cost, 8)

    def to_dict(self) -> dict[str, Any]:
        """Serialize including the computed total."""
        return {**asdict(self), "total": self.total}


# --------------------------------------------------------------------------- #
# Model information
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ModelInfo:
    """A model as advertised by Cortex ``GET /v1/models``."""

    id: str
    runtime: str = "llama.cpp"
    owned_by: str = "cortex-llmhoster"
    capabilities: list[str] = field(default_factory=list)
    created: int = 0
    context_size: int | None = None
    default: bool = False
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelInfo:
        """Parse one element of the Cortex models list."""
        return cls(
            id=str(raw.get("id", "")),
            runtime=str(raw.get("runtime") or raw.get("owned_by") or "llama.cpp"),
            owned_by=str(raw.get("owned_by", "cortex-llmhoster")),
            capabilities=[str(c) for c in (raw.get("capabilities") or [])],
            created=int(raw.get("created") or 0),
            context_size=raw.get("context_size"),
            default=bool(raw.get("default", False)),
            status=str(raw.get("status", "unknown")),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model record."""
        return asdict(self)

    def supports(self, capability: Capability | str) -> bool:
        """Report whether the model declares a capability (or its aliases)."""
        wanted = (
            capability
            if isinstance(capability, Capability)
            else Capability.parse(capability) or capability
        )
        key = wanted.value if isinstance(wanted, Capability) else str(wanted)
        declared = {str(item).lower().replace("-", "_") for item in self.capabilities}
        if key in declared:
            return True
        return any(
            (Capability.parse(item) or item) == key or str(Capability.parse(item) or item) == key
            for item in declared
        )

    def is_chat(self) -> bool:
        """Report whether the model can serve chat completions."""
        return self.supports(Capability.CHAT) or not self.capabilities

    def is_embedding(self) -> bool:
        """Report whether the model can serve embeddings."""
        return self.supports(Capability.EMBEDDING)


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AgentEvent:
    """A typed notification emitted while an agent runs."""

    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=now_ts)
    agent: str = "main"
    run_id: str = ""
    id: str = field(default_factory=lambda: new_id("ev"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the event log or the SSE stream."""
        return {
            "id": self.id,
            "kind": self.kind.value if isinstance(self.kind, EventKind) else str(self.kind),
            "agent": self.agent,
            "run_id": self.run_id,
            "ts": self.ts,
            "iso": iso_ts(self.ts),
            "data": self.data,
        }

    def to_json(self) -> str:
        """Serialize to a compact JSON line."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

    @classmethod
    def of(cls, kind: EventKind | str, **data: Any) -> AgentEvent:
        """Convenience constructor accepting a loose kind string."""
        parsed = kind if isinstance(kind, EventKind) else EventKind(str(kind))
        return cls(kind=parsed, data=data)


# --------------------------------------------------------------------------- #
# Steps, runs, results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class StepRecord:
    """One iteration of the agent loop, recorded for replay and evaluation."""

    index: int
    thought: str = ""
    message: Message | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=now_ts)
    duration_ms: float = 0.0
    model: str = ""
    stop_reason: str = ""
    cache_hit: bool = False
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the step for transcript storage."""
        return {
            "index": self.index,
            "thought": self.thought,
            "message": self.message.to_dict() if self.message else None,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_results": [result.to_dict() for result in self.tool_results],
            "usage": self.usage.to_dict(),
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "model": self.model,
            "stop_reason": self.stop_reason,
            "cache_hit": self.cache_hit,
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StepRecord:
        """Rebuild a step record from persisted JSON."""
        message = raw.get("message")
        return cls(
            index=int(raw.get("index", 0)),
            thought=str(raw.get("thought") or ""),
            message=Message.from_dict(message) if isinstance(message, dict) else None,
            tool_calls=[
                ToolCall.from_dict(call)
                for call in (raw.get("tool_calls") or [])
                if isinstance(call, dict)
            ],
            tool_results=[
                ToolResult.from_dict(result)
                for result in (raw.get("tool_results") or [])
                if isinstance(result, dict)
            ],
            usage=Usage.from_dict(raw.get("usage")),
            started_at=float(raw.get("started_at") or now_ts()),
            duration_ms=float(raw.get("duration_ms") or 0.0),
            model=str(raw.get("model") or ""),
            stop_reason=str(raw.get("stop_reason") or ""),
            cache_hit=bool(raw.get("cache_hit", False)),
            retries=int(raw.get("retries") or 0),
        )

    def tool_names(self) -> list[str]:
        """Return the names of tools invoked during this step."""
        return [call.name for call in self.tool_calls]

    def all_ok(self) -> bool:
        """Report whether every tool call in the step succeeded."""
        return all(result.ok for result in self.tool_results)


@dataclass(slots=True)
class RunResult:
    """The final outcome of an agent run."""

    run_id: str
    output: str = ""
    stop_reason: StopReason = StopReason.COMPLETED
    steps: list[StepRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: CostEstimate = field(default_factory=CostEstimate)
    started_at: float = field(default_factory=now_ts)
    finished_at: float = field(default_factory=now_ts)
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    structured: Any = None
    events: list[AgentEvent] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Return the total wall clock duration in milliseconds."""
        return max(0.0, (self.finished_at - self.started_at) * 1000.0)

    @property
    def iterations(self) -> int:
        """Return how many agent loop iterations were executed."""
        return len(self.steps)

    @property
    def tool_call_count(self) -> int:
        """Return the total number of tool invocations across all steps."""
        return sum(len(step.tool_calls) for step in self.steps)

    @property
    def tool_error_count(self) -> int:
        """Return how many tool invocations failed."""
        return sum(
            1
            for step in self.steps
            for result in step.tool_results
            if not result.ok
        )

    @property
    def succeeded(self) -> bool:
        """Report whether the run completed successfully."""
        return self.stop_reason.is_success() and self.error is None

    def tokens_per_second(self) -> float:
        """Return completion throughput, or 0.0 when nothing was generated."""
        seconds = self.duration_ms / 1000.0
        if seconds <= 0 or self.usage.completion_tokens <= 0:
            return 0.0
        return round(self.usage.completion_tokens / seconds, 2)

    def cache_hit_rate(self) -> float:
        """Return the fraction of steps served from cache."""
        if not self.steps:
            return 0.0
        hits = sum(1 for step in self.steps if step.cache_hit)
        return round(hits / len(self.steps), 4)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole run for transcripts and the REST API."""
        return {
            "run_id": self.run_id,
            "output": self.output,
            "stop_reason": self.stop_reason.value,
            "succeeded": self.succeeded,
            "iterations": self.iterations,
            "tool_calls": self.tool_call_count,
            "tool_errors": self.tool_error_count,
            "usage": self.usage.to_dict(),
            "cost": self.cost.to_dict(),
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 3),
            "tokens_per_second": self.tokens_per_second(),
            "cache_hit_rate": self.cache_hit_rate(),
            "steps": [step.to_dict() for step in self.steps],
            "messages": [message.to_dict() for message in self.messages],
            "structured": self.structured,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the run to JSON, optionally pretty printed."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunResult:
        """Rebuild a run result from persisted JSON."""
        return cls(
            run_id=str(raw.get("run_id") or new_id("run")),
            output=str(raw.get("output") or ""),
            stop_reason=StopReason(str(raw.get("stop_reason") or "completed")),
            steps=[
                StepRecord.from_dict(step)
                for step in (raw.get("steps") or [])
                if isinstance(step, dict)
            ],
            usage=Usage.from_dict(raw.get("usage")),
            cost=CostEstimate(**{
                k: v
                for k, v in (raw.get("cost") or {}).items()
                if k in {"prompt_cost", "completion_cost", "currency", "source"}
            }),
            started_at=float(raw.get("started_at") or now_ts()),
            finished_at=float(raw.get("finished_at") or now_ts()),
            model=str(raw.get("model") or ""),
            messages=messages_from_dicts(raw.get("messages") or []),
            structured=raw.get("structured"),
            error=raw.get("error"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def summary_line(self) -> str:
        """Return a one line human summary suitable for CLI output."""
        return (
            f"run={self.run_id} status={self.stop_reason.value} "
            f"steps={self.iterations} tools={self.tool_call_count} "
            f"tokens={self.usage.total_tokens} time={self.duration_ms:.0f}ms"
        )


@dataclass(slots=True)
class ChatResponse:
    """A single non-agentic completion from the LLM backend."""

    text: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cache_hit: bool = False
    id: str = field(default_factory=lambda: new_id("resp"))

    @property
    def has_tool_calls(self) -> bool:
        """Report whether the model asked to call tools."""
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response, dropping the raw payload."""
        return {
            "id": self.id,
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 3),
            "cache_hit": self.cache_hit,
        }

    def to_message(self) -> Message:
        """Convert into an assistant :class:`Message`."""
        return Message.assistant(self.text, tool_calls=self.tool_calls)


@dataclass(slots=True)
class Embedding:
    """A single embedding vector with provenance."""

    vector: list[float]
    text: str = ""
    model: str = ""
    token_count: int = 0
    id: str = field(default_factory=lambda: new_id("emb"))

    @property
    def dim(self) -> int:
        """Return the vector dimensionality."""
        return len(self.vector)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the embedding record."""
        return {
            "id": self.id,
            "model": self.model,
            "dim": self.dim,
            "text": self.text,
            "token_count": self.token_count,
            "vector": self.vector,
        }


__all__ = [
    "AgentEvent",
    "ApprovalDecision",
    "ApprovalMode",
    "AudioPart",
    "Capability",
    "ChatResponse",
    "ContentPart",
    "CostEstimate",
    "Embedding",
    "EventKind",
    "FilePart",
    "ImagePart",
    "Message",
    "ModelInfo",
    "Role",
    "RunResult",
    "StepRecord",
    "StopReason",
    "TextPart",
    "ToolCall",
    "ToolResult",
    "Usage",
    "Verbosity",
    "iso_ts",
    "messages_from_dicts",
    "new_id",
    "now_ts",
    "part_from_dict",
]
