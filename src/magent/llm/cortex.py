"""First-class Cortex LLMHoster backend.

`Cortex LLMHoster <https://github.com/codero-sus/Cortex_LLMHoster>`_ hosts local
models (llama.cpp GGUF plus any OpenAI-compatible local engine) and exposes them
behind an OpenAI dialect at ``/v1`` with an admin control plane at
``/admin/api``.

This module is the only place in MAgent that knows Cortex's URL layout, auth
scheme, error envelope, and capability vocabulary, so the rest of the framework
stays hoster-agnostic.

Environment variables read (MAgent's own name wins when both are set):

============================  =========================================
``MAGENT_CORTEX_URL``         Base URL, default ``http://127.0.0.1:8624``
``MAGENT_CORTEX_API_KEY``     Bearer key (falls back to ``CORTEX_API_KEY``)
``MAGENT_CORTEX_MODEL``       Default model id
``MAGENT_CORTEX_TIMEOUT``     Read timeout in seconds
``MAGENT_CORTEX_API_PREFIX``  API prefix override, default ``/v1``
============================  =========================================
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..errors import (
    AuthenticationError,
    BackendError,
    CapabilityNotSupportedError,
    ModelNotFoundError,
    ModelNotRunningError,
    StreamingError,
)
from ..http.client import HTTPClient, RetryPolicy, StdlibTransport
from ..http.multipart import encode_multipart
from ..http.sse import (
    SSEParser,
    extract_delta_text,
    extract_delta_tool_calls,
    merge_tool_call_deltas,
)
from ..types import Capability, ChatResponse, Message, ModelInfo, ToolCall, Usage
from ..utils.urls import join_url, normalize_base_url
from .base import BaseBackend, CompletionRequest, StreamChunk

DEFAULT_CORTEX_URL = "http://127.0.0.1:8624"
"""Where Cortex LLMHoster listens unless configured otherwise."""

DEFAULT_API_PREFIX = "/v1"
"""Cortex's OpenAI-compatible route prefix."""

CHAT_COMPLETIONS_PATH = "/chat/completions"
"""Path for chat inference."""

COMPLETIONS_PATH = "/completions"
"""Path for legacy text completion inference."""

EMBEDDINGS_PATH = "/embeddings"
"""Path for embedding generation."""

RESPONSES_PATH = "/responses"
"""Path for the Responses API dialect."""

MODELS_PATH = "/models"
"""Path listing configured models."""

TRANSCRIPTION_PATH = "/audio/transcriptions"
"""Path for speech-to-text."""

SPEECH_PATH = "/audio/speech"
"""Path for text-to-speech."""

IMAGE_GENERATION_PATH = "/images/generations"
"""Path for text-to-image generation."""

IMAGE_EDIT_PATH = "/images/edits"
"""Path for image-to-image editing."""

VIDEO_GENERATION_PATH = "/videos/generations"
"""Path for video generation."""

RERANK_PATH = "/rerank"
"""Path for reranking candidate documents."""

MODERATIONS_PATH = "/moderations"
"""Path for content moderation."""

HEALTH_PATH = "/health"
"""Cortex liveness probe; never touches the model runtime."""

READY_PATH = "/ready"
"""Cortex readiness probe; checks the default model is running."""


@dataclass(slots=True)
class CortexSettings:
    """Connection settings for a Cortex LLMHoster instance."""

    base_url: str = DEFAULT_CORTEX_URL
    api_key: str = ""
    api_prefix: str = DEFAULT_API_PREFIX
    default_model: str = ""
    embedding_model: str = ""
    timeout: float = 120.0
    connect_timeout: float = 10.0
    max_retries: int = 3
    verify_ssl: bool = True
    require_auth: bool = True
    max_upload_bytes: int = 512 * 1024 * 1024
    organization: str = ""
    extra_headers: dict[str, str] | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> CortexSettings:
        """Build settings from environment variables, MAgent names first."""
        env = dict(os.environ if environ is None else environ)

        def pick(*names: str, default: str = "") -> str:
            """Return the first non-empty variable among ``names``."""
            for name in names:
                value = env.get(name)
                if value:
                    return value.strip()
            return default

        url = normalize_base_url(
            pick("MAGENT_CORTEX_URL", "CORTEX_URL", default=DEFAULT_CORTEX_URL)
        )
        timeout = pick("MAGENT_CORTEX_TIMEOUT", "CORTEX_TIMEOUT", default="120")
        try:
            timeout_value = float(timeout)
        except ValueError:
            timeout_value = 120.0
        return cls(
            base_url=url,
            api_key=pick("MAGENT_CORTEX_API_KEY", "CORTEX_API_KEY"),
            api_prefix=pick("MAGENT_CORTEX_API_PREFIX", default=DEFAULT_API_PREFIX),
            default_model=pick("MAGENT_CORTEX_MODEL", "CORTEX_DEFAULT_MODEL"),
            embedding_model=pick("MAGENT_EMBEDDING_MODEL", "MAGENT_CORTEX_EMBEDDING_MODEL"),
            timeout=timeout_value,
            max_retries=int(pick("MAGENT_CORTEX_RETRIES", default="3") or 3),
            verify_ssl=pick("MAGENT_CORTEX_VERIFY_SSL", default="true").lower()
            not in {"0", "false", "no"},
            require_auth=pick("MAGENT_CORTEX_REQUIRE_AUTH", default="true").lower()
            not in {"0", "false", "no"},
        )

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Serialize settings, masking the API key by default."""
        from ..utils.text import mask_secret

        data = {
            "base_url": self.base_url,
            "api_prefix": self.api_prefix,
            "api_key": mask_secret(self.api_key) if redact else self.api_key,
            "default_model": self.default_model,
            "embedding_model": self.embedding_model,
            "timeout": self.timeout,
            "connect_timeout": self.connect_timeout,
            "max_retries": self.max_retries,
            "verify_ssl": self.verify_ssl,
            "require_auth": self.require_auth,
            "organization": self.organization,
        }
        return data


class CortexBackend(BaseBackend):
    """Talk to a running Cortex LLMHoster over its OpenAI-compatible API."""

    name = "cortex"

    def __init__(
        self,
        settings: CortexSettings | None = None,
        *,
        client: HTTPClient | None = None,
        on_event: Any = None,
    ) -> None:
        self.settings = settings or CortexSettings.from_env()
        super().__init__(
            default_model=self.settings.default_model,
            default_timeout=self.settings.timeout,
        )
        self.on_event = on_event
        self.client = client or self._build_client()
        self._models_cache: list[ModelInfo] = []
        self._models_cached_at = 0.0
        self._models_ttl = 15.0
        self.streams_opened = 0
        self.stream_bytes = 0
        self.first_token_ms: list[float] = []

    # -- construction --------------------------------------------------------

    def _build_client(self) -> HTTPClient:
        """Create the pooled HTTP client used for every Cortex call."""
        headers = {"User-Agent": _user_agent()}
        if self.settings.organization:
            headers["OpenAI-Organization"] = self.settings.organization
        if self.settings.extra_headers:
            headers.update(self.settings.extra_headers)
        return HTTPClient(
            base_url=self.settings.base_url,
            transport=StdlibTransport(
                max_body_bytes=max(1 << 20, self.settings.max_upload_bytes * 4),
                verify_ssl=self.settings.verify_ssl,
            ),
            timeout=self.settings.timeout,
            connect_timeout=self.settings.connect_timeout,
            read_timeout=self.settings.timeout,
            headers=headers,
            api_key=self.settings.api_key or None,
            retry=RetryPolicy(
                max_attempts=max(1, self.settings.max_retries),
                backoff_base=0.35,
                backoff_factor=2.0,
                backoff_cap=15.0,
            ),
            on_retry=self._handle_retry,
        )

    def _handle_retry(self, attempt: int, delay: float, error: BaseException | None) -> None:
        """Emit an event whenever the client retries a Cortex call."""
        self._emit(
            "llm.retry",
            {
                "backend": self.name,
                "attempt": attempt,
                "delay_seconds": round(delay, 3),
                "error": str(error)[:300] if error else None,
            },
        )

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        """Forward an event to the optional callback."""
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:
                pass

    # -- URL helpers ---------------------------------------------------------

    def api_url(self, path: str) -> str:
        """Return the absolute URL for an API-prefixed Cortex route."""
        return join_url(self.settings.base_url, f"{self.settings.api_prefix}{path}")

    def admin_url(self, path: str) -> str:
        """Return the absolute URL for an ``/admin/api`` Cortex route."""
        return join_url(self.settings.base_url, f"/admin/api{path}")

    def root_url(self, path: str) -> str:
        """Return the absolute URL for an unprefixed Cortex route."""
        return join_url(self.settings.base_url, path)

    # -- health and discovery ------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Probe ``/health`` and ``/ready`` without invoking a model."""
        result: dict[str, Any] = {
            "backend": self.name,
            "base_url": self.settings.base_url,
            "liveness": None,
            "readiness": None,
            "ok": False,
        }
        try:
            live = self.client.get(
                self.root_url(HEALTH_PATH),
                timeout=self.settings.connect_timeout,
                raise_for_status=False,
                max_attempts=1,
            )
            result["liveness"] = {"status": live.status, "body": live.json({})}
        except Exception as exc:
            result["liveness"] = {"status": None, "error": str(exc)[:300]}
            result["ok"] = False
            return result
        try:
            ready = self.client.get(
                self.root_url(READY_PATH),
                timeout=self.settings.connect_timeout,
                raise_for_status=False,
                max_attempts=1,
            )
            result["readiness"] = {"status": ready.status, "body": ready.json({})}
            result["ok"] = live.status < 500
        except Exception as exc:
            result["readiness"] = {"status": None, "error": str(exc)[:300]}
            result["ok"] = live.status < 500
        return result

    def ping(self, *, timeout: float | None = None) -> bool:
        """Report whether Cortex answers its liveness probe."""
        try:
            response = self.client.get(
                self.root_url(HEALTH_PATH),
                timeout=timeout or self.settings.connect_timeout,
                raise_for_status=False,
                max_attempts=1,
            )
            return response.status < 500
        except Exception:
            return False

    def models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """List models from ``GET /v1/models`` with a short TTL cache."""
        now = time.monotonic()
        if not refresh and self._models_cache and now - self._models_cached_at < self._models_ttl:
            return list(self._models_cache)
        response = self.client.get(self.api_url(MODELS_PATH), raise_for_status=False)
        if response.status == 401 or response.status == 403:
            raise AuthenticationError(
                f"Cortex rejected the API key at {self.settings.base_url}. "
                "Set MAGENT_CORTEX_API_KEY, or start Cortex with --no-auth."
            )
        if response.status >= 400:
            raise BackendError(
                f"Cortex returned HTTP {response.status} listing models",
                details={"body": response.text()[:1000]},
            )
        payload = response.json({}) or {}
        entries = payload.get("data") if isinstance(payload, dict) else payload
        models = [
            ModelInfo.from_dict(entry) for entry in (entries or []) if isinstance(entry, dict)
        ]
        self._models_cache = models
        self._models_cached_at = now
        self._emit("llm.models", {"count": len(models), "ids": [model.id for model in models]})
        return list(models)

    def model_ids(self, *, refresh: bool = False) -> list[str]:
        """Return just the configured model identifiers."""
        return [model.id for model in self.models(refresh=refresh)]

    def get_model(self, model_id: str, *, refresh: bool = False) -> ModelInfo | None:
        """Look up one model record by id."""
        for model in self.models(refresh=refresh):
            if model.id == model_id:
                return model
        return None

    def default_model_id(self) -> str:
        """Return the model MAgent should use when the caller does not pick one."""
        if self.default_model:
            return self.default_model
        models = self.models()
        if not models:
            return ""
        for model in models:
            if model.default:
                return model.id
        for model in models:
            if model.is_chat():
                return model.id
        return models[0].id

    def assert_capability(self, model_id: str, capability: Capability) -> None:
        """Raise when a model does not declare the required capability."""
        info = self.get_model(model_id, refresh=True)
        if info is None:
            raise ModelNotFoundError(
                f"Model {model_id!r} is not configured on Cortex at {self.settings.base_url}.",
                details={"available": self.model_ids()},
            )
        if info.capabilities and not info.supports(capability):
            raise CapabilityNotSupportedError(
                f"Model {model_id!r} does not declare capability {capability.value!r}.",
                details={"declared": info.capabilities},
            )

    # -- inference -----------------------------------------------------------

    def complete(self, request: CompletionRequest) -> ChatResponse:
        """POST ``/v1/chat/completions`` and parse the reply."""
        model = self.resolve_model(request.model) or self.default_model_id()
        if not model:
            raise ModelNotFoundError(
                "No model was specified and Cortex reports none configured. "
                "Set MAGENT_CORTEX_MODEL or add a model in the Cortex dashboard."
            )
        payload = request.to_payload()
        payload["model"] = model
        payload["stream"] = False
        started = time.perf_counter()
        self._emit(
            "llm.request",
            {
                "backend": self.name,
                "model": model,
                "path": CHAT_COMPLETIONS_PATH,
                "messages": len(payload["messages"]),
            },
        )
        response = self.client.post(
            self.api_url(CHAT_COMPLETIONS_PATH),
            json_body=payload,
            timeout=request.timeout or self.settings.timeout,
            raise_for_status=False,
        )
        latency = (time.perf_counter() - started) * 1000.0
        self._raise_for_inference(response, model)
        data = response.json({}) or {}
        result = self._parse_completion(data, model=model, latency_ms=latency)
        self.record(result)
        self._emit(
            "llm.response",
            {
                "backend": self.name,
                "model": model,
                "latency_ms": round(latency, 2),
                "usage": result.usage.to_dict(),
                "finish_reason": result.finish_reason,
            },
        )
        return result

    def completion(self, prompt: str, *, model: str = "", **params: Any) -> ChatResponse:
        """Call the legacy ``/v1/completions`` endpoint with a raw prompt."""
        target = self.resolve_model(model) or self.default_model_id()
        payload: dict[str, Any] = {"model": target, "prompt": prompt, "stream": False, **params}
        response = self.client.post(
            self.api_url(COMPLETIONS_PATH), json_body=payload, raise_for_status=False
        )
        self._raise_for_inference(response, target)
        data = response.json({}) or {}
        choices = data.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            text = str(choices[0].get("text") or "")
        return ChatResponse(
            text=text,
            model=str(data.get("model") or target),
            usage=Usage.from_dict(data.get("usage")),
            finish_reason=str((choices[0] or {}).get("finish_reason") or "stop")
            if choices
            else "stop",
            raw=data,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream ``/v1/chat/completions`` with ``stream: true``."""
        model = self.resolve_model(request.model) or self.default_model_id()
        if not model:
            raise ModelNotFoundError("No model available to stream from Cortex.")
        payload = request.to_payload()
        payload["model"] = model
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        self.streams_opened += 1
        started = time.perf_counter()
        first_token_recorded = False
        parser = SSEParser()
        fragments: list[dict[str, Any]] = []
        finish_reason = "stop"
        usage: Usage | None = None
        index = 0
        try:
            for raw_chunk in self.client.stream(
                "POST",
                self.api_url(CHAT_COMPLETIONS_PATH),
                json_body=payload,
                timeout=request.timeout or self.settings.timeout,
            ):
                self.stream_bytes += len(raw_chunk)
                for event in parser.feed(raw_chunk):
                    if event.is_done:
                        final_calls = [
                            ToolCall.from_dict(raw) for raw in merge_tool_call_deltas(fragments)
                        ]
                        yield StreamChunk(
                            tool_calls=final_calls,
                            finish_reason=finish_reason,
                            usage=usage,
                            model=model,
                            done=True,
                            index=index + 1,
                            raw={"__streamed_tool_calls__": fragments},
                        )
                        return
                    data = event.json(None)
                    if not isinstance(data, dict):
                        continue
                    if data.get("error"):
                        raise StreamingError(
                            str(data["error"].get("message", "stream error"))
                            if isinstance(data["error"], dict)
                            else str(data["error"]),
                            details={"model": model},
                        )
                    if data.get("usage"):
                        usage = Usage.from_dict(data["usage"])
                    fragments.extend(extract_delta_tool_calls(data))
                    choices = data.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        if choices[0].get("finish_reason"):
                            finish_reason = str(choices[0]["finish_reason"])
                    text = extract_delta_text(data)
                    if text and not first_token_recorded:
                        first_token_recorded = True
                        self.first_token_ms.append(
                            round((time.perf_counter() - started) * 1000.0, 2)
                        )
                        self.first_token_ms = self.first_token_ms[-50:]
                    if text:
                        index += 1
                        yield StreamChunk(text=text, model=model, index=index, raw=data)
            tail = parser.close()
            if tail is not None and not tail.is_done:
                data = tail.json(None)
                if isinstance(data, dict):
                    text = extract_delta_text(data)
                    if text:
                        yield StreamChunk(text=text, model=model, index=index + 1, raw=data)
        except StreamingError:
            raise
        except Exception as exc:
            if isinstance(exc, BackendError):
                raise
            raise StreamingError(f"Cortex stream failed: {exc}", details={"model": model}) from exc
        calls = [ToolCall.from_dict(raw) for raw in merge_tool_call_deltas(fragments)]
        yield StreamChunk(
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            done=True,
            index=index + 1,
            raw={"__streamed_tool_calls__": fragments},
        )

    def chat(self, messages: list[Message] | list[dict[str, Any]], **params: Any) -> ChatResponse:
        """Convenience wrapper accepting either :class:`Message` or raw dicts."""
        parsed = [
            message if isinstance(message, Message) else Message.from_dict(message)
            for message in messages
        ]
        return self.complete(CompletionRequest(messages=parsed, **params))

    def ask(self, prompt: str, *, system: str = "", **params: Any) -> str:
        """Send one prompt and return the assistant text."""
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(prompt))
        return self.complete(CompletionRequest(messages=messages, **params)).text

    # -- embeddings and other modalities -------------------------------------

    def embed(
        self,
        texts: list[str],
        *,
        model: str = "",
        dimensions: int | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """POST ``/v1/embeddings`` and return one vector per input text."""
        target = model or self.settings.embedding_model or self.default_model_id()
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = texts[start : start + max(1, batch_size)]
            payload: dict[str, Any] = {"model": target, "input": batch}
            if dimensions:
                payload["dimensions"] = dimensions
            response = self.client.post(
                self.api_url(EMBEDDINGS_PATH), json_body=payload, raise_for_status=False
            )
            self._raise_for_inference(response, target)
            data = response.json({}) or {}
            items = data.get("data") or []
            ordered = sorted(
                (item for item in items if isinstance(item, dict)),
                key=lambda item: int(item.get("index") or 0),
            )
            vectors.extend([list(map(float, item.get("embedding") or [])) for item in ordered])
        return vectors

    def embed_one(self, text: str, *, model: str = "") -> list[float]:
        """Embed a single string."""
        result = self.embed([text], model=model)
        return result[0] if result else []

    def responses_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the Responses API dialect and return the parsed body."""
        response = self.client.post(
            self.api_url(RESPONSES_PATH), json_body=payload, raise_for_status=False
        )
        model = str(payload.get("model") or self.default_model)
        self._raise_for_inference(response, model)
        return response.json({}) or {}

    def transcribe(
        self,
        audio_path: str,
        *,
        model: str = "",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Upload audio to ``/v1/audio/transcriptions`` as multipart."""
        target = model or self.default_model_id()
        fields: dict[str, Any] = {"model": target, "response_format": response_format}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt
        if temperature is not None:
            fields["temperature"] = temperature
        body, content_type = encode_multipart(
            fields, {"file": audio_path}, max_bytes=self.settings.max_upload_bytes
        )
        response = self.client.post(
            self.api_url(TRANSCRIPTION_PATH),
            body=body,
            headers={"Content-Type": content_type},
            raise_for_status=False,
        )
        self._raise_for_inference(response, target)
        if response_format in {"json", "verbose_json"}:
            return response.json({}) or {}
        return {"text": response.text()}

    def speech(
        self,
        text: str,
        *,
        model: str = "",
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        """POST ``/v1/audio/speech`` and return the raw audio bytes."""
        target = model or self.default_model_id()
        payload = {
            "model": target,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        response = self.client.post(
            self.api_url(SPEECH_PATH), json_body=payload, raise_for_status=False
        )
        self._raise_for_inference(response, target)
        return response.body

    def generate_image(
        self,
        prompt: str,
        *,
        model: str = "",
        size: str = "1024x1024",
        n: int = 1,
        response_format: str = "b64_json",
        negative_prompt: str | None = None,
    ) -> dict[str, Any]:
        """POST ``/v1/images/generations`` and return the parsed body."""
        target = model or self.default_model_id()
        payload: dict[str, Any] = {
            "model": target,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": response_format,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        response = self.client.post(
            self.api_url(IMAGE_GENERATION_PATH), json_body=payload, raise_for_status=False
        )
        self._raise_for_inference(response, target)
        return response.json({}) or {}

    def edit_image(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        model: str = "",
        mask_path: str | None = None,
        size: str = "1024x1024",
    ) -> dict[str, Any]:
        """POST ``/v1/images/edits`` with one or more source images."""
        target = model or self.default_model_id()
        files: dict[str, Any] = {}
        if len(image_paths) == 1:
            files["image"] = image_paths[0]
        else:
            files["image[]"] = image_paths
        if mask_path:
            files["mask"] = mask_path
        body, content_type = encode_multipart(
            {"model": target, "prompt": prompt, "size": size},
            files,
            max_bytes=self.settings.max_upload_bytes,
        )
        response = self.client.post(
            self.api_url(IMAGE_EDIT_PATH),
            body=body,
            headers={"Content-Type": content_type},
            raise_for_status=False,
        )
        self._raise_for_inference(response, target)
        return response.json({}) or {}

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str = "",
        top_n: int | None = None,
    ) -> dict[str, Any]:
        """POST ``/v1/rerank`` and return the ranked results."""
        target = model or self.default_model_id()
        payload: dict[str, Any] = {"model": target, "query": query, "documents": documents}
        if top_n:
            payload["top_n"] = top_n
        response = self.client.post(
            self.api_url(RERANK_PATH), json_body=payload, raise_for_status=False
        )
        self._raise_for_inference(response, target)
        return response.json({}) or {}

    def moderate(self, input_text: str | list[str], *, model: str = "") -> dict[str, Any]:
        """POST ``/v1/moderations`` and return the verdict."""
        target = model or self.default_model_id()
        response = self.client.post(
            self.api_url(MODERATIONS_PATH),
            json_body={"model": target, "input": input_text},
            raise_for_status=False,
        )
        self._raise_for_inference(response, target)
        return response.json({}) or {}

    def generate_video(self, prompt: str, *, model: str = "", **params: Any) -> dict[str, Any]:
        """POST ``/v1/videos/generations`` and return the parsed body."""
        target = model or self.default_model_id()
        payload = {"model": target, "prompt": prompt, **params}
        response = self.client.post(
            self.api_url(VIDEO_GENERATION_PATH), json_body=payload, raise_for_status=False
        )
        self._raise_for_inference(response, target)
        return response.json({}) or {}

    # -- parsing and errors --------------------------------------------------

    def _parse_completion(
        self, data: dict[str, Any], *, model: str, latency_ms: float
    ) -> ChatResponse:
        """Turn an OpenAI-shaped completion body into a :class:`ChatResponse`."""
        if not isinstance(data, dict):
            raise BackendError(f"Cortex returned a non-object completion body for {model!r}")
        if data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise BackendError(
                f"Cortex reported an inference error: {message}", details={"model": model}
            )
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise BackendError(
                f"Cortex returned no choices for {model!r}; is the model running?",
                details={"model": model, "body_keys": sorted(data)},
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        raw_calls = message.get("tool_calls") if isinstance(message, dict) else None
        calls = [ToolCall.from_dict(call) for call in (raw_calls or []) if isinstance(call, dict)]
        return ChatResponse(
            text=text,
            model=str(data.get("model") or model),
            usage=Usage.from_dict(data.get("usage")),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            raw=data,
            latency_ms=latency_ms,
        )

    def _raise_for_inference(self, response: Any, model: str) -> None:
        """Convert a non-2xx Cortex response into a precise MAgent error."""
        if response.ok:
            return
        body_text = response.text()[:1500]
        parsed: Any = None
        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError:
            parsed = None
        detail = ""
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            if not detail:
                detail = str(parsed.get("message") or "")
        lowered = detail.lower() + body_text.lower()
        if response.status in (401, 403):
            raise AuthenticationError(
                f"Cortex refused the request ({response.status}). "
                "Check MAGENT_CORTEX_API_KEY, or disable auth on Cortex with --no-auth.",
                status=response.status,
                url=response.url,
                body=body_text,
            )
        if response.status == 404 and ("model" in lowered or model in lowered):
            known = ", ".join(self.model_ids()) or "none"
            raise ModelNotFoundError(
                f"Cortex does not know model {model!r}. Configured models: {known}.",
                status=response.status,
                url=response.url,
            )
        if "not running" in lowered or "not_running" in lowered or "no local model" in lowered:
            raise ModelNotRunningError(
                f"Model {model!r} is configured but its runtime is not running. "
                "Start it from the Cortex dashboard or with `magent cortex start`.",
                status=response.status,
                url=response.url,
            )
        if response.status == 429:
            from ..errors import RateLimitedError

            raise RateLimitedError(
                f"Cortex is at capacity ({response.status}); retry shortly.",
                retry_after=response.retry_after(),
                url=response.url,
            )
        response.raise_for_status()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP pool."""
        self.client.close()

    def stats(self) -> dict[str, Any]:
        """Return backend telemetry including streaming latency stats."""
        base = super().stats()
        base.update(
            {
                "base_url": self.settings.base_url,
                "api_prefix": self.settings.api_prefix,
                "authenticated": bool(self.settings.api_key),
                "streams_opened": self.streams_opened,
                "stream_bytes": self.stream_bytes,
                "first_token_ms_avg": (
                    round(sum(self.first_token_ms) / len(self.first_token_ms), 2)
                    if self.first_token_ms
                    else 0.0
                ),
                "http": self.client.stats(),
            }
        )
        return base

    def describe(self) -> dict[str, Any]:
        """Return a redacted description suitable for ``magent doctor``."""
        return {
            "backend": self.name,
            "settings": self.settings.to_dict(),
            "models": [model.to_dict() for model in self._models_cache],
            "stats": self.stats(),
        }


def _user_agent() -> str:
    """Return the User-Agent MAgent sends to Cortex."""
    from .._version import USER_AGENT

    return USER_AGENT


def build_cortex_backend(**overrides: Any) -> CortexBackend:
    """Build a :class:`CortexBackend` from environment plus explicit overrides."""
    settings = CortexSettings.from_env()
    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return CortexBackend(settings)


def cortex_health(url: str = DEFAULT_CORTEX_URL, *, timeout: float = 3.0) -> dict[str, Any]:
    """One-shot health probe that never raises, for scripts and ``doctor``."""
    backend = CortexBackend(
        CortexSettings(base_url=normalize_base_url(url), timeout=timeout, require_auth=False)
    )
    try:
        return backend.health()
    finally:
        backend.close()


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "COMPLETIONS_PATH",
    "DEFAULT_API_PREFIX",
    "DEFAULT_CORTEX_URL",
    "EMBEDDINGS_PATH",
    "HEALTH_PATH",
    "IMAGE_EDIT_PATH",
    "IMAGE_GENERATION_PATH",
    "MODELS_PATH",
    "MODERATIONS_PATH",
    "READY_PATH",
    "RERANK_PATH",
    "RESPONSES_PATH",
    "SPEECH_PATH",
    "TRANSCRIPTION_PATH",
    "VIDEO_GENERATION_PATH",
    "CortexBackend",
    "CortexSettings",
    "build_cortex_backend",
    "cortex_health",
]
