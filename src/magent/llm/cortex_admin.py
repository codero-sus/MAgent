"""Cortex LLMHoster admin control plane client.

The ``/admin/api`` surface lets MAgent inspect and drive the hoster: read
hardware and runtime state, start and stop local models, run model smoke tests,
read inference metrics, list local model files, browse the model cookbook, and
export or reload configuration.

Admin routes are protected by the same bearer key as inference routes, so a
:class:`CortexAdmin` shares the :class:`~magent.llm.cortex.CortexSettings`
credential and never logs it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import BackendError, ConfigError
from ..types import ModelInfo
from .cortex import CortexBackend, CortexSettings

STATE_PATH = "/state"
"""Full dashboard state: settings, metrics, runtime, hardware."""

METRICS_PATH = "/metrics"
"""Inference counters for the current worker."""

METRICS_RESET_PATH = "/metrics/reset"
"""Zero the inference counters."""

RUNTIME_PATH = "/runtime"
"""Runtime manager status."""

MODEL_START_PATH = "/models/start"
"""Start a model's local runtime."""

MODEL_STOP_PATH = "/models/stop"
"""Stop a model's local runtime."""

MODEL_STOP_ALL_PATH = "/models/stop-all"
"""Stop every running model runtime."""

MODEL_RESTART_PATH = "/models/restart"
"""Restart a model's local runtime."""

MODEL_TEST_PATH = "/models/test"
"""Run a smoke test against a model."""

MODEL_FILES_PATH = "/models/files"
"""List local GGUF/weight files discovered on disk."""

MODEL_UPLOAD_PATH = "/models/files/upload"
"""Upload a model file."""

DETECT_CAPABILITIES_PATH = "/models/detect-capabilities"
"""Infer capabilities from a model file."""

COOKBOOK_PATH = "/cookbook"
"""Paginated Hugging Face model cookbook."""

CONFIG_PATH = "/config"
"""Update server configuration."""

CONFIG_RELOAD_PATH = "/config/reload"
"""Reload configuration from disk."""

CONFIG_RESET_PATH = "/config/reset"
"""Reset configuration to defaults."""

CONFIG_EXPORT_PATH = "/config/export"
"""Download a secret-free configuration export."""

UPDATE_CHECK_PATH = "/update/check"
"""Check the pinned source branch for updates."""

UPDATE_INSTALL_PATH = "/update/install"
"""Install an available update."""

AUTH_OPTIONS_PATH = "/auth/options"
"""Report how the dashboard accepts sign-in."""


@dataclass(slots=True)
class CortexState:
    """A typed view of ``GET /admin/api/state``."""

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CortexState:
        """Wrap a raw state payload."""
        return cls(raw=data if isinstance(data, dict) else {})

    @property
    def version(self) -> str:
        """Return the Cortex version string."""
        return str(self.raw.get("version") or "unknown")

    @property
    def server(self) -> dict[str, Any]:
        """Return the server settings section."""
        value = self.raw.get("server")
        return value if isinstance(value, dict) else {}

    @property
    def models(self) -> list[dict[str, Any]]:
        """Return the configured model records."""
        value = self.raw.get("models")
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    @property
    def model_infos(self) -> list[ModelInfo]:
        """Return configured models as :class:`ModelInfo` objects."""
        return [ModelInfo.from_dict(item) for item in self.models]

    @property
    def runtime(self) -> dict[str, Any]:
        """Return the runtime section (hardware plus per-model state)."""
        value = self.raw.get("runtime")
        return value if isinstance(value, dict) else {}

    @property
    def hardware(self) -> dict[str, Any]:
        """Return detected hardware information."""
        value = self.runtime.get("hardware")
        return value if isinstance(value, dict) else {}

    @property
    def runtime_models(self) -> dict[str, Any]:
        """Return the per-model runtime state mapping."""
        value = self.runtime.get("models")
        return value if isinstance(value, dict) else {}

    @property
    def metrics(self) -> dict[str, Any]:
        """Return the inference metrics snapshot."""
        value = self.raw.get("metrics")
        return value if isinstance(value, dict) else {}

    @property
    def api(self) -> dict[str, Any]:
        """Return the public API format description."""
        value = self.raw.get("api")
        return value if isinstance(value, dict) else {}

    @property
    def auth_enabled(self) -> bool:
        """Report whether Cortex requires a bearer key."""
        return bool(self.server.get("api_auth_enabled"))

    @property
    def env_overrides(self) -> list[str]:
        """Return the environment variables currently overriding config."""
        value = self.raw.get("env_overrides")
        return [str(item) for item in value] if isinstance(value, list) else []

    def model_status(self, model_id: str) -> str:
        """Return the runtime status string for one model."""
        record = self.runtime_models.get(model_id)
        if isinstance(record, dict):
            return str(record.get("status") or "unknown")
        return "unknown"

    def running_models(self) -> list[str]:
        """Return ids of models whose runtime is currently running."""
        return [
            model_id
            for model_id, record in self.runtime_models.items()
            if isinstance(record, dict) and str(record.get("status")) == "running"
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying raw payload."""
        return dict(self.raw)


@dataclass(slots=True)
class CortexMetrics:
    """A typed view of ``GET /admin/api/metrics``."""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_requests(self) -> int:
        """Return the number of inference requests seen by this worker."""
        return int(self.raw.get("total_requests") or 0)

    @property
    def completed_requests(self) -> int:
        """Return the number of completed requests."""
        return int(self.raw.get("completed_requests") or 0)

    @property
    def failed_requests(self) -> int:
        """Return the number of requests that ended in a 5xx."""
        return int(self.raw.get("failed_requests") or 0)

    @property
    def in_flight(self) -> int:
        """Return how many requests are currently executing."""
        return int(self.raw.get("in_flight") or 0)

    @property
    def average_latency_ms(self) -> float:
        """Return mean inference latency in milliseconds."""
        return float(self.raw.get("average_latency_ms") or 0.0)

    @property
    def uptime_seconds(self) -> float:
        """Return how long the worker has been up."""
        return float(self.raw.get("uptime_seconds") or 0.0)

    @property
    def status_codes(self) -> dict[str, int]:
        """Return the response status histogram."""
        value = self.raw.get("status_codes")
        return {str(k): int(v) for k, v in value.items()} if isinstance(value, dict) else {}

    @property
    def requests_by_model(self) -> dict[str, int]:
        """Return per-model request counts."""
        value = self.raw.get("requests_by_model")
        return {str(k): int(v) for k, v in value.items()} if isinstance(value, dict) else {}

    @property
    def error_rate(self) -> float:
        """Return the fraction of requests that failed."""
        if not self.total_requests:
            return 0.0
        return round(self.failed_requests / self.total_requests, 4)

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying raw payload."""
        return dict(self.raw)


class CortexAdmin:
    """Client for the Cortex ``/admin/api`` control plane."""

    def __init__(self, backend: CortexBackend | None = None, settings: CortexSettings | None = None) -> None:
        self.backend = backend or CortexBackend(settings or CortexSettings.from_env())
        self.settings = self.backend.settings

    # -- transport -----------------------------------------------------------

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Perform an authenticated admin GET and decode JSON."""
        response = self.backend.client.get(
            self.backend.admin_url(path), params=params, raise_for_status=False
        )
        return self._unwrap(response, path)

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        """Perform an authenticated admin POST and decode JSON."""
        response = self.backend.client.post(
            self.backend.admin_url(path), json_body=payload or {}, raise_for_status=False
        )
        return self._unwrap(response, path)

    def _put(self, path: str, payload: dict[str, Any]) -> Any:
        """Perform an authenticated admin PUT and decode JSON."""
        response = self.backend.client.put(
            self.backend.admin_url(path), json_body=payload, raise_for_status=False
        )
        return self._unwrap(response, path)

    def _unwrap(self, response: Any, path: str) -> Any:
        """Raise a helpful error for admin failures, else return the body."""
        if response.status in (401, 403):
            from ..errors import AuthenticationError

            raise AuthenticationError(
                f"Cortex admin API refused access ({response.status}) at {path}. "
                "The admin plane always requires a valid bearer key.",
                status=response.status,
            )
        if response.status >= 400:
            detail = response.text()[:800]
            raise BackendError(
                f"Cortex admin call {path} failed with HTTP {response.status}: {detail}",
                details={"status": response.status, "path": path},
            )
        return response.json({})

    # -- read operations -----------------------------------------------------

    def state(self) -> CortexState:
        """Fetch the full dashboard state."""
        return CortexState.from_dict(self._get(STATE_PATH) or {})

    def metrics(self) -> CortexMetrics:
        """Fetch inference metrics."""
        return CortexMetrics(raw=self._get(METRICS_PATH) or {})

    def runtime(self) -> dict[str, Any]:
        """Fetch runtime manager status."""
        return self._get(RUNTIME_PATH) or {}

    def hardware(self) -> dict[str, Any]:
        """Fetch detected hardware information."""
        return self.state().hardware

    def auth_options(self) -> dict[str, Any]:
        """Report how the dashboard accepts sign-in."""
        response = self.backend.client.get(
            self.backend.admin_url(AUTH_OPTIONS_PATH), raise_for_status=False
        )
        return response.json({}) or {}

    def model_files(self, *, directory: str | None = None) -> list[dict[str, Any]]:
        """List local model files Cortex can see."""
        params = {"dir": directory} if directory else None
        data = self._get(MODEL_FILES_PATH, params=params) or {}
        if isinstance(data, dict):
            files = data.get("files") or data.get("data") or []
        else:
            files = data
        return [item for item in files if isinstance(item, dict)]

    def cookbook(self, *, page: int = 1, limit: int = 20, search: str = "") -> dict[str, Any]:
        """Fetch a page of the Hugging Face model cookbook."""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if search:
            params["search"] = search
        return self._get(COOKBOOK_PATH, params=params) or {}

    def export_config(self) -> dict[str, Any]:
        """Download the secret-free configuration export."""
        return self._get(CONFIG_EXPORT_PATH) or {}

    def check_update(self) -> dict[str, Any]:
        """Check the pinned source branch for a newer Cortex."""
        return self._get(UPDATE_CHECK_PATH) or {}

    # -- write operations ----------------------------------------------------

    def start_model(self, model_id: str, *, wait: bool = True, timeout: float = 120.0) -> dict[str, Any]:
        """Start a model's local runtime, optionally waiting for readiness."""
        result = self._post(MODEL_START_PATH, {"id": model_id}) or {}
        if wait:
            self.wait_for_model(model_id, "running", timeout=timeout)
        return result if isinstance(result, dict) else {"ok": True}

    def stop_model(self, model_id: str) -> dict[str, Any]:
        """Stop a model's local runtime."""
        result = self._post(MODEL_STOP_PATH, {"id": model_id}) or {}
        return result if isinstance(result, dict) else {"ok": True}

    def restart_model(self, model_id: str, *, wait: bool = True) -> dict[str, Any]:
        """Restart a model's local runtime."""
        result = self._post(MODEL_RESTART_PATH, {"id": model_id}) or {}
        if wait:
            self.wait_for_model(model_id, "running", timeout=120.0)
        return result if isinstance(result, dict) else {"ok": True}

    def stop_all_models(self) -> dict[str, Any]:
        """Stop every running model runtime to free memory."""
        result = self._post(MODEL_STOP_ALL_PATH, {}) or {}
        return result if isinstance(result, dict) else {"ok": True}

    def test_model(self, model_id: str, *, prompt: str = "ping", timeout: float = 60.0) -> dict[str, Any]:
        """Run Cortex's own smoke test against a model."""
        return self._post(MODEL_TEST_PATH, {"id": model_id, "prompt": prompt, "timeout": timeout}) or {}

    def detect_capabilities(self, model_path: str) -> dict[str, Any]:
        """Ask Cortex to infer capabilities from a model file on disk."""
        return self._post(DETECT_CAPABILITIES_PATH, {"model_path": model_path}) or {}

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply a configuration change through the admin API."""
        if not isinstance(payload, dict) or not payload:
            raise ConfigError("Configuration payload must be a non-empty object.")
        return self._put(CONFIG_PATH, payload) or {}

    def reload_config(self) -> dict[str, Any]:
        """Ask Cortex to re-read its configuration from disk."""
        return self._post(CONFIG_RELOAD_PATH, {}) or {}

    def reset_config(self) -> dict[str, Any]:
        """Reset Cortex configuration to defaults."""
        return self._post(CONFIG_RESET_PATH, {}) or {}

    def reset_metrics(self) -> dict[str, Any]:
        """Zero the inference counters."""
        return self._post(METRICS_RESET_PATH, {}) or {}

    def install_update(self) -> dict[str, Any]:
        """Install an available Cortex update (requires a single worker)."""
        return self._post(UPDATE_INSTALL_PATH, {}) or {}

    def upload_model_file(self, path: str | Path, *, directory: str = "") -> dict[str, Any]:
        """Upload a model file into Cortex's models directory."""
        from ..http.multipart import encode_multipart

        source = Path(path).expanduser()
        if not source.is_file():
            raise ConfigError(f"No such model file to upload: {source}")
        fields = {"dir": directory} if directory else {}
        body, content_type = encode_multipart(
            fields, {"file": source}, max_bytes=self.settings.max_upload_bytes
        )
        response = self.backend.client.post(
            self.backend.admin_url(MODEL_UPLOAD_PATH),
            body=body,
            headers={"Content-Type": content_type},
            raise_for_status=False,
        )
        return self._unwrap(response, MODEL_UPLOAD_PATH) or {}

    # -- polling -------------------------------------------------------------

    def wait_for_model(self, model_id: str, status: str = "running", *, timeout: float = 120.0,
                       interval: float = 1.0) -> bool:
        """Poll runtime state until a model reaches ``status`` or times out."""
        import time

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            current = self.model_status(model_id)
            if current == status:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(interval, max(0.05, deadline - time.monotonic())))

    def model_status(self, model_id: str) -> str:
        """Return the current runtime status of one model."""
        try:
            return self.state().model_status(model_id)
        except BackendError:
            return "unknown"

    def iter_running_models(self) -> Iterator[str]:
        """Yield ids of models that are currently running."""
        yield from self.state().running_models()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the shared backend connection pool."""
        self.backend.close()

    def __enter__(self) -> CortexAdmin:
        """Support context manager usage."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on context exit."""
        self.close()


__all__ = [
    "AUTH_OPTIONS_PATH",
    "CONFIG_EXPORT_PATH",
    "CONFIG_PATH",
    "CONFIG_RELOAD_PATH",
    "CONFIG_RESET_PATH",
    "COOKBOOK_PATH",
    "CortexAdmin",
    "CortexMetrics",
    "CortexState",
    "DETECT_CAPABILITIES_PATH",
    "METRICS_PATH",
    "METRICS_RESET_PATH",
    "MODEL_FILES_PATH",
    "MODEL_RESTART_PATH",
    "MODEL_START_PATH",
    "MODEL_STOP_ALL_PATH",
    "MODEL_STOP_PATH",
    "MODEL_TEST_PATH",
    "MODEL_UPLOAD_PATH",
    "RUNTIME_PATH",
    "STATE_PATH",
    "UPDATE_CHECK_PATH",
    "UPDATE_INSTALL_PATH",
]
