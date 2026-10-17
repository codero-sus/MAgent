"""Layered configuration: TOML files, environment variables, and CLI flags.

Precedence, lowest to highest:

1. Built-in defaults (:func:`default_config`).
2. ``magent.toml`` in the system config directory.
3. ``magent.toml`` in the user home directory.
4. ``magent.toml`` in the current working directory.
5. ``$MAGENT_CONFIG`` if it points at a file.
6. Environment variables (``MAGENT_*``).
7. Explicit keyword overrides from the CLI or the Python API.

Every section is a frozen-ish dataclass with validation, so a bad value fails at
load time with a precise message instead of deep inside a request. Secrets are
resolved through :mod:`magent.secrets` and are never written back to disk.
"""

from __future__ import annotations

import copy
import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError, ConfigFileNotFound, ConfigValidationError
from .utils.json_utils import deep_merge

CONFIG_FILENAME = "magent.toml"
"""Name of the project configuration file."""

ENV_PREFIX = "MAGENT_"
"""Prefix for every MAgent environment variable."""

CONFIG_ENV_VAR = "MAGENT_CONFIG"
"""Environment variable pointing at an explicit config file."""

PROFILE_ENV_VAR = "MAGENT_PROFILE"
"""Environment variable selecting a named profile."""

SEARCH_ORDER = ("system", "user", "project", "explicit")
"""Order in which configuration sources are layered."""


def system_config_dir() -> Path:
    """Return the system wide configuration directory."""
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA", r"C:\\ProgramData")
        return Path(base) / "magent"
    return Path(os.environ.get("XDG_CONFIG_HOME", "/etc/xdg")) / "magent"


def user_config_dir() -> Path:
    """Return the per-user configuration directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(base) / "magent"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "magent"


def user_data_dir() -> Path:
    """Return the per-user data directory for caches, sessions, and stores."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(base) / "magent"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "magent"


def user_cache_dir() -> Path:
    """Return the per-user cache directory."""
    if sys.platform == "win32":
        return user_data_dir() / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "magent"


def default_config_paths() -> list[Path]:
    """Return the configuration files MAgent looks for, in layering order."""
    paths = [
        system_config_dir() / CONFIG_FILENAME,
        user_config_dir() / CONFIG_FILENAME,
        Path.cwd() / CONFIG_FILENAME,
    ]
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        paths.append(Path(explicit).expanduser())
    return paths


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ServerConfig:
    """Configuration for connecting to Cortex LLMHoster."""

    base_url: str = "http://127.0.0.1:8624"
    api_key: str = ""
    api_key_env: str = "MAGENT_CORTEX_API_KEY"
    api_prefix: str = "/v1"
    timeout: float = 120.0
    connect_timeout: float = 10.0
    max_retries: int = 3
    verify_ssl: bool = True
    probe_timeout: float = 3.0
    max_upload_bytes: int = 512 * 1024 * 1024
    organization: str = ""

    def validate(self) -> None:
        """Reject malformed connection settings."""
        if self.timeout <= 0:
            raise ConfigValidationError("[server] timeout must be positive")
        if self.connect_timeout <= 0:
            raise ConfigValidationError("[server] connect_timeout must be positive")
        if self.max_retries < 0:
            raise ConfigValidationError("[server] max_retries cannot be negative")
        if not self.api_prefix.startswith("/"):
            raise ConfigValidationError("[server] api_prefix must start with '/'")
        if self.base_url and "://" in self.base_url:
            scheme = self.base_url.split("://", 1)[0].lower()
            if scheme not in {"http", "https"}:
                raise ConfigValidationError(f"[server] unsupported URL scheme {scheme!r}")


@dataclass(slots=True)
class ModelConfig:
    """Which models MAgent uses for each job."""

    backend: str = "auto"
    model: str = ""
    embedding_model: str = ""
    vision_model: str = ""
    audio_model: str = ""
    image_model: str = ""
    rerank_model: str = ""
    planner_model: str = ""
    summarizer_model: str = ""
    fallback_model: str = ""
    router_strategy: str = "priority"
    fallback_to_echo: bool = True
    routes: list[dict[str, Any]] = field(default_factory=list)
    context_size: int = 8192
    max_output_tokens: int = 2048
    auto_start_models: bool = True

    def validate(self) -> None:
        """Reject unknown backends and nonsensical sizes."""
        from .llm.factory import BACKEND_CHOICES

        if self.backend not in BACKEND_CHOICES:
            raise ConfigValidationError(
                f"[model] backend must be one of {', '.join(BACKEND_CHOICES)}"
            )
        if self.context_size <= 0:
            raise ConfigValidationError("[model] context_size must be positive")
        if self.max_output_tokens < 0:
            raise ConfigValidationError("[model] max_output_tokens cannot be negative")
        if self.router_strategy not in ("priority", "round_robin", "least_used", "fastest", "cheapest", "random"):
            raise ConfigValidationError(f"[model] unknown router_strategy {self.router_strategy!r}")

    def model_for(self, role: str) -> str:
        """Return the model id configured for a specific job role."""
        mapping = {
            "default": self.model,
            "embedding": self.embedding_model,
            "vision": self.vision_model,
            "audio": self.audio_model,
            "image": self.image_model,
            "rerank": self.rerank_model,
            "planner": self.planner_model or self.model,
            "summarizer": self.summarizer_model or self.model,
            "fallback": self.fallback_model or self.model,
        }
        return mapping.get(role, self.model)


@dataclass(slots=True)
class GenerationConfig:
    """Sampling parameters sent with every completion."""

    temperature: float = 0.4
    top_p: float = 1.0
    max_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = True
    request_format: str = "chat"
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject out-of-range sampling values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigValidationError("[generation] temperature must be within 0.0..2.0")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigValidationError("[generation] top_p must be within (0.0, 1.0]")
        if self.max_tokens < 1:
            raise ConfigValidationError("[generation] max_tokens must be at least 1")
        if self.request_format not in {"chat", "responses", "completion"}:
            raise ConfigValidationError("[generation] request_format is chat, responses, or completion")

    def to_params(self) -> dict[str, Any]:
        """Return the subset of parameters that is non-default."""
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty != 1.0:
            params["extra"] = {"repeat_penalty": self.repetition_penalty}
        if self.seed is not None:
            params["seed"] = self.seed
        if self.stop:
            params["stop"] = list(self.stop)
        params.update(self.extra)
        return params


@dataclass(slots=True)
class AgentConfig:
    """Behaviour of the agent loop."""

    max_iterations: int = 12
    max_tool_calls_per_step: int = 8
    parallel_tools: bool = True
    reflection: bool = False
    reflection_every: int = 3
    planning: bool = False
    replan_on_failure: bool = True
    loop_mode: str = "react"
    stop_on_error: bool = False
    max_consecutive_failures: int = 3
    include_reasoning: bool = True
    confirm_destructive: bool = True
    persona: str = "assistant"
    system_prompt: str = ""
    extra_instructions: str = ""
    verbose: bool = False
    tool_timeout: float = 60.0
    max_tool_output_chars: int = 32000

    def validate(self) -> None:
        """Reject loop settings that could hang or spin."""
        if self.max_iterations < 1:
            raise ConfigValidationError("[agent] max_iterations must be at least 1")
        if self.max_tool_calls_per_step < 1:
            raise ConfigValidationError("[agent] max_tool_calls_per_step must be at least 1")
        if self.loop_mode not in {"react", "plan", "reflect", "single"}:
            raise ConfigValidationError("[agent] loop_mode is react, plan, reflect, or single")
        if self.max_consecutive_failures < 1:
            raise ConfigValidationError("[agent] max_consecutive_failures must be at least 1")
        if self.tool_timeout <= 0:
            raise ConfigValidationError("[agent] tool_timeout must be positive")


@dataclass(slots=True)
class MemoryConfig:
    """Short and long term memory settings."""

    enabled: bool = True
    max_messages: int = 60
    max_context_tokens: int = 6000
    trim_strategy: str = "middle"
    summarize: bool = True
    summary_threshold: int = 24
    summary_model: str = ""
    store_path: str = ""
    persistence: bool = True
    vector_enabled: bool = True
    vector_dim: int = 0
    retrieval_top_k: int = 5
    retrieval_mode: str = "hybrid"
    embedding_batch_size: int = 16
    entity_tracking: bool = True
    cache_embeddings: bool = True
    dedupe: bool = True

    def validate(self) -> None:
        """Reject memory settings that would grow without bound."""
        if self.max_messages < 1:
            raise ConfigValidationError("[memory] max_messages must be at least 1")
        if self.max_context_tokens < 256:
            raise ConfigValidationError("[memory] max_context_tokens must be at least 256")
        if self.trim_strategy not in {"middle", "oldest", "newest"}:
            raise ConfigValidationError("[memory] trim_strategy is middle, oldest, or newest")
        if self.retrieval_mode not in {"vector", "bm25", "hybrid", "none"}:
            raise ConfigValidationError("[memory] retrieval_mode is vector, bm25, hybrid, or none")
        if self.retrieval_top_k < 0:
            raise ConfigValidationError("[memory] retrieval_top_k cannot be negative")


@dataclass(slots=True)
class ToolsConfig:
    """Tool registry, sandbox, and approval settings."""

    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    workspace: str = ""
    allow_paths: list[str] = field(default_factory=list)
    deny_paths: list[str] = field(default_factory=list)
    allow_commands: list[str] = field(default_factory=list)
    deny_commands: list[str] = field(default_factory=list)
    approval_mode: str = "dangerous"
    allow_network: bool = False
    allow_domains: list[str] = field(default_factory=list)
    deny_domains: list[str] = field(default_factory=list)
    max_file_read_bytes: int = 2 * 1024 * 1024
    max_file_write_bytes: int = 8 * 1024 * 1024
    max_output_chars: int = 32000
    command_timeout: float = 60.0
    python_timeout: float = 30.0
    shell: str = ""
    env_passthrough: list[str] = field(default_factory=list)
    plugins_dir: str = ""
    autoload_plugins: bool = True

    def validate(self) -> None:
        """Reject approval modes and limits that do not make sense."""
        if self.approval_mode not in {"never", "ask", "always", "dangerous"}:
            raise ConfigValidationError(
                "[tools] approval_mode is never, ask, always, or dangerous"
            )
        if self.max_file_read_bytes < 0 or self.max_file_write_bytes < 0:
            raise ConfigValidationError("[tools] file byte limits cannot be negative")
        if self.command_timeout <= 0:
            raise ConfigValidationError("[tools] command_timeout must be positive")


@dataclass(slots=True)
class GuardrailsConfig:
    """Safety checks applied before and after model calls."""

    enabled: bool = True
    injection_detection: bool = True
    injection_action: str = "warn"
    injection_threshold: float = 0.55
    pii_redaction: bool = True
    pii_action: str = "redact"
    max_prompt_chars: int = 200000
    max_output_chars: int = 400000
    block_patterns: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)
    budget_max_tokens: int = 0
    budget_max_cost: float = 0.0
    budget_max_seconds: float = 0.0
    budget_max_steps: int = 0
    budget_max_tool_calls: int = 0
    budget_warn_at: float = 0.8
    rate_limit_per_minute: int = 0
    audit_log: str = ""
    audit_enabled: bool = True
    url_allowlist_only: bool = False

    def validate(self) -> None:
        """Reject guardrail settings that would silently disable safety."""
        if self.injection_action not in {"warn", "block", "strip", "off"}:
            raise ConfigValidationError("[guardrails] injection_action is warn, block, strip, or off")
        if self.pii_action not in {"redact", "mask", "block", "off"}:
            raise ConfigValidationError("[guardrails] pii_action is redact, mask, block, or off")
        if not 0.0 <= self.injection_threshold <= 1.0:
            raise ConfigValidationError("[guardrails] injection_threshold must be within 0..1")
        if not 0.0 < self.budget_warn_at <= 1.0:
            raise ConfigValidationError("[guardrails] budget_warn_at must be within (0, 1]")


@dataclass(slots=True)
class CacheConfig:
    """Response and embedding cache settings."""

    enabled: bool = True
    directory: str = ""
    ttl: float = 3600.0
    max_bytes: int = 256 * 1024 * 1024
    embed_cache: bool = True
    cache_streamed: bool = False
    namespace: str = "chat"

    def validate(self) -> None:
        """Reject cache settings that could grow without bound."""
        if self.ttl < 0:
            raise ConfigValidationError("[cache] ttl cannot be negative")
        if self.max_bytes < 1024:
            raise ConfigValidationError("[cache] max_bytes must be at least 1024")


@dataclass(slots=True)
class OutputConfig:
    """How MAgent renders results."""

    format: str = "text"
    color: bool = True
    unicode: bool = True
    width: int = 0
    show_usage: bool = True
    show_timing: bool = True
    show_tool_calls: bool = True
    stream_to_terminal: bool = True
    markdown: bool = True
    quiet: bool = False
    verbosity: str = "normal"
    spinner: bool = True
    progress: bool = True

    def validate(self) -> None:
        """Reject unknown output formats and verbosity levels."""
        if self.format not in {"text", "json", "markdown", "html", "csv", "jsonl"}:
            raise ConfigValidationError("[output] format is text, json, markdown, html, csv, or jsonl")
        if self.verbosity not in {"quiet", "normal", "verbose", "debug"}:
            raise ConfigValidationError("[output] verbosity is quiet, normal, verbose, or debug")


@dataclass(slots=True)
class SessionConfig:
    """Session persistence and history settings."""

    enabled: bool = True
    directory: str = ""
    autosave: bool = True
    keep_last: int = 50
    resume_last: bool = False
    transcript_format: str = "jsonl"
    redact_secrets: bool = True
    max_session_bytes: int = 64 * 1024 * 1024

    def validate(self) -> None:
        """Reject session settings that would lose history silently."""
        if self.keep_last < 1:
            raise ConfigValidationError("[session] keep_last must be at least 1")
        if self.transcript_format not in {"jsonl", "json", "markdown"}:
            raise ConfigValidationError("[session] transcript_format is jsonl, json, or markdown")


@dataclass(slots=True)
class ObservabilityConfig:
    """Logging, tracing, and metrics settings."""

    log_level: str = "INFO"
    log_format: str = "text"
    log_file: str = ""
    log_requests: bool = False
    log_responses: bool = False
    redact_secrets: bool = True
    trace_enabled: bool = True
    trace_file: str = ""
    metrics_enabled: bool = True
    metrics_file: str = ""
    max_log_bytes: int = 32 * 1024 * 1024
    rotate_logs: bool = True
    keep_rotated: int = 5

    def validate(self) -> None:
        """Reject unknown log levels and formats."""
        if self.log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ConfigValidationError("[observability] log_level is one of the standard levels")
        if self.log_format not in {"text", "json"}:
            raise ConfigValidationError("[observability] log_format is text or json")
        if self.keep_rotated < 0:
            raise ConfigValidationError("[observability] keep_rotated cannot be negative")


@dataclass(slots=True)
class HttpServerConfig:
    """Settings for the embedded MAgent HTTP server and web UI."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8700
    api_key: str = ""
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    serve_ui: bool = True
    max_body_bytes: int = 32 * 1024 * 1024
    request_timeout: float = 300.0
    threads: int = 16
    allow_origins_any: bool = True

    def validate(self) -> None:
        """Reject bind settings that would fail or leak."""
        if not 0 <= self.port <= 65535:
            raise ConfigValidationError("[http] port must be within 0..65535")
        if self.threads < 1:
            raise ConfigValidationError("[http] threads must be at least 1")
        if self.max_body_bytes < 1024:
            raise ConfigValidationError("[http] max_body_bytes must be at least 1024")


@dataclass(slots=True)
class EvalConfig:
    """Settings for the offline evaluation harness."""

    dataset: str = "builtin"
    dataset_path: str = ""
    metrics: list[str] = field(default_factory=lambda: ["exact_match", "contains", "tool_used", "latency"])
    repeats: int = 1
    concurrency: int = 4
    fail_below: float = 0.0
    report_format: str = "markdown"
    report_path: str = ""

    def validate(self) -> None:
        """Reject evaluation settings that cannot run."""
        if self.repeats < 1:
            raise ConfigValidationError("[eval] repeats must be at least 1")
        if self.concurrency < 1:
            raise ConfigValidationError("[eval] concurrency must be at least 1")


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #

SECTION_TYPES: dict[str, type] = {
    "server": ServerConfig,
    "model": ModelConfig,
    "generation": GenerationConfig,
    "agent": AgentConfig,
    "memory": MemoryConfig,
    "tools": ToolsConfig,
    "guardrails": GuardrailsConfig,
    "cache": CacheConfig,
    "output": OutputConfig,
    "session": SessionConfig,
    "observability": ObservabilityConfig,
    "http": HttpServerConfig,
    "eval": EvalConfig,
}
"""Mapping of TOML table name to section dataclass."""

ENV_MAP: dict[str, tuple[str, str, type]] = {
    "MAGENT_BASE_URL": ("server", "base_url", str),
    "MAGENT_CORTEX_URL": ("server", "base_url", str),
    "MAGENT_API_KEY": ("server", "api_key", str),
    "MAGENT_CORTEX_API_KEY": ("server", "api_key", str),
    "MAGENT_API_PREFIX": ("server", "api_prefix", str),
    "MAGENT_TIMEOUT": ("server", "timeout", float),
    "MAGENT_BACKEND": ("model", "backend", str),
    "MAGENT_MODEL": ("model", "model", str),
    "MAGENT_EMBEDDING_MODEL": ("model", "embedding_model", str),
    "MAGENT_PLANNER_MODEL": ("model", "planner_model", str),
    "MAGENT_TEMPERATURE": ("generation", "temperature", float),
    "MAGENT_TOP_P": ("generation", "top_p", float),
    "MAGENT_MAX_TOKENS": ("generation", "max_tokens", int),
    "MAGENT_STREAM": ("generation", "stream", bool),
    "MAGENT_SEED": ("generation", "seed", int),
    "MAGENT_MAX_ITERATIONS": ("agent", "max_iterations", int),
    "MAGENT_LOOP_MODE": ("agent", "loop_mode", str),
    "MAGENT_PERSONA": ("agent", "persona", str),
    "MAGENT_SYSTEM_PROMPT": ("agent", "system_prompt", str),
    "MAGENT_PLANNING": ("agent", "planning", bool),
    "MAGENT_REFLECTION": ("agent", "reflection", bool),
    "MAGENT_PARALLEL_TOOLS": ("agent", "parallel_tools", bool),
    "MAGENT_TOOL_TIMEOUT": ("agent", "tool_timeout", float),
    "MAGENT_WORKSPACE": ("tools", "workspace", str),
    "MAGENT_APPROVAL_MODE": ("tools", "approval_mode", str),
    "MAGENT_ALLOW_NETWORK": ("tools", "allow_network", bool),
    "MAGENT_TOOLS_ENABLED": ("tools", "enabled", list),
    "MAGENT_TOOLS_DISABLED": ("tools", "disabled", list),
    "MAGENT_MEMORY_ENABLED": ("memory", "enabled", bool),
    "MAGENT_MEMORY_MAX_MESSAGES": ("memory", "max_messages", int),
    "MAGENT_RETRIEVAL_MODE": ("memory", "retrieval_mode", str),
    "MAGENT_GUARDRAILS": ("guardrails", "enabled", bool),
    "MAGENT_INJECTION_ACTION": ("guardrails", "injection_action", str),
    "MAGENT_BUDGET_MAX_TOKENS": ("guardrails", "budget_max_tokens", int),
    "MAGENT_BUDGET_MAX_SECONDS": ("guardrails", "budget_max_seconds", float),
    "MAGENT_CACHE_ENABLED": ("cache", "enabled", bool),
    "MAGENT_CACHE_DIR": ("cache", "directory", str),
    "MAGENT_CACHE_TTL": ("cache", "ttl", float),
    "MAGENT_OUTPUT_FORMAT": ("output", "format", str),
    "MAGENT_COLOR": ("output", "color", bool),
    "MAGENT_VERBOSITY": ("output", "verbosity", str),
    "MAGENT_SESSION_DIR": ("session", "directory", str),
    "MAGENT_LOG_LEVEL": ("observability", "log_level", str),
    "MAGENT_LOG_FORMAT": ("observability", "log_format", str),
    "MAGENT_LOG_FILE": ("observability", "log_file", str),
    "MAGENT_HTTP_HOST": ("http", "host", str),
    "MAGENT_HTTP_PORT": ("http", "port", int),
    "MAGENT_HTTP_API_KEY": ("http", "api_key", str),
}
"""Environment variable to ``(section, field, type)`` mapping."""


@dataclass(slots=True)
class Config:
    """The fully resolved MAgent configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    http: HttpServerConfig = field(default_factory=HttpServerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    profile: str = "default"
    sources: list[str] = field(default_factory=list)
    config_path: Path | None = None
    data_dir: Path = field(default_factory=user_data_dir)
    cache_dir: Path = field(default_factory=user_cache_dir)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- validation and derivation ------------------------------------------

    def validate(self) -> Config:
        """Validate every section and return ``self`` for chaining."""
        for section_name, section_type in SECTION_TYPES.items():
            section = getattr(self, section_name, None)
            if section is None:
                raise ConfigValidationError(f"Missing configuration section [{section_name}]")
            if not isinstance(section, section_type):
                raise ConfigValidationError(
                    f"Section [{section_name}] must be a {section_type.__name__}"
                )
            if hasattr(section, "validate"):
                section.validate()
        self._apply_paths()
        return self

    def _apply_paths(self) -> None:
        """Fill in default storage paths under the user data directory."""
        self.data_dir = Path(self.data_dir).expanduser()
        self.cache_dir = Path(self.cache_dir).expanduser()
        if not self.cache.directory:
            self.cache.directory = str(self.cache_dir / "responses")
        if not self.session.directory:
            self.session.directory = str(self.data_dir / "sessions")
        if not self.memory.store_path:
            self.memory.store_path = str(self.data_dir / "memory.db")
        if not self.observability.log_file:
            self.observability.log_file = str(self.data_dir / "logs" / "magent.log")
        if not self.observability.trace_file and self.observability.trace_enabled:
            self.observability.trace_file = str(self.data_dir / "logs" / "traces.jsonl")
        if not self.tools.workspace:
            self.tools.workspace = str(Path.cwd())
        if not self.guardrails.audit_log and self.guardrails.audit_enabled:
            self.guardrails.audit_log = str(self.data_dir / "logs" / "audit.jsonl")
        if not self.tools.plugins_dir:
            self.tools.plugins_dir = str(self.data_dir / "plugins")

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Serialize the configuration, masking secrets by default."""
        from .utils.text import mask_secret

        data: dict[str, Any] = {}
        for name in SECTION_TYPES:
            section = getattr(self, name)
            data[name] = asdict(section)
        data["profile"] = self.profile
        data["sources"] = list(self.sources)
        data["config_path"] = str(self.config_path) if self.config_path else None
        data["data_dir"] = str(self.data_dir)
        data["cache_dir"] = str(self.cache_dir)
        data["extra"] = dict(self.extra)
        if redact:
            for path in (("server", "api_key"), ("http", "api_key")):
                section, key = path
                value = data[section].get(key)
                if value:
                    data[section][key] = mask_secret(str(value))
        return data

    def to_toml(self) -> str:
        """Render the configuration as a TOML document (secrets redacted)."""
        lines: list[str] = [
            "# MAgent configuration - generated by `magent config init`",
            f'profile = "{self.profile}"',
            "",
        ]
        for name in SECTION_TYPES:
            lines.append(f"[{name}]")
            section = getattr(self, name)
            for item in fields(section):
                value = getattr(section, item.name)
                if item.name in {"api_key"} and value:
                    value = "env:MAGENT_API_KEY"
                lines.append(f"{item.name} = {_toml_value(value)}")
            lines.append("")
        return "\n".join(lines)

    def clone(self, **overrides: Any) -> Config:
        """Return an independent copy with selected top-level keys replaced.

        A shallow dataclass ``replace`` is not sufficient here: section objects
        contain mutable lists and dictionaries, and callers clone precisely so
        they can alter a policy without mutating the process-wide configuration.
        """
        cloned = copy.deepcopy(self)
        for name, value in overrides.items():
            if not hasattr(cloned, name):
                raise ConfigError(f"Unknown top-level configuration key {name!r}")
            setattr(cloned, name, value)
        return cloned

    def section(self, name: str) -> Any:
        """Return one configuration section by name."""
        if name not in SECTION_TYPES:
            raise ConfigError(f"Unknown configuration section {name!r}")
        return getattr(self, name)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value by dotted path, e.g. ``agent.max_iterations``."""
        section_name, _, key = dotted.partition(".")
        section = getattr(self, section_name, None)
        if section is None:
            return default
        if not key:
            return section
        return getattr(section, key, default)

    def set(self, dotted: str, value: Any) -> None:
        """Write a value by dotted path, coercing to the field's type."""
        section_name, _, key = dotted.partition(".")
        if section_name not in SECTION_TYPES:
            raise ConfigError(f"Unknown configuration section {section_name!r}")
        section = getattr(self, section_name)
        if not hasattr(section, key):
            raise ConfigError(f"Unknown configuration key {dotted!r}")
        current = getattr(section, key)
        setattr(section, key, _coerce(value, type(current), current))

    def describe(self) -> list[dict[str, Any]]:
        """Return a flat list of every setting, used by ``magent config list``."""
        rows: list[dict[str, Any]] = []
        for name in SECTION_TYPES:
            section = getattr(self, name)
            for item in fields(section):
                value = getattr(section, item.name)
                rows.append({
                    "key": f"{name}.{item.name}",
                    "value": value,
                    "type": item.type if isinstance(item.type, str) else type(value).__name__,
                })
        return rows


def _toml_value(value: Any) -> str:
    """Render one Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return '""'
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
        return "{" + inner + "}"
    return f'"{value}"'


def _coerce(value: Any, target: Any, current: Any = None) -> Any:
    """Coerce a raw config value to the type of the existing field."""
    if value is None:
        return current
    reference = current if current is not None else value
    if isinstance(reference, bool) or target is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}
    if isinstance(reference, int) and not isinstance(reference, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f"Expected an integer, got {value!r}") from exc
    if isinstance(reference, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f"Expected a number, got {value!r}") from exc
    if isinstance(reference, list):
        if isinstance(value, list):
            return value
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(reference, dict):
        return value if isinstance(value, dict) else {}
    return str(value)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_toml(path: str | Path, *, required: bool = False) -> dict[str, Any]:
    """Load one TOML file into a plain dictionary."""
    target = Path(path).expanduser()
    if not target.is_file():
        if required:
            raise ConfigFileNotFound(f"Configuration file not found: {target}")
        return {}
    try:
        with target.open("rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{target} must contain a TOML table")
    return data


def _apply_profiles(data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Merge a ``[profiles.<name>]`` table over the base configuration."""
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        return data
    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        if profile != "default":
            raise ConfigError(
                f"Profile {profile!r} is not defined; available: {', '.join(sorted(profiles))}"
            )
        return data
    merged = deep_merge(
        {key: value for key, value in data.items() if key != "profiles"}, selected
    )
    return merged


def apply_env(config: Config, environ: dict[str, str] | None = None) -> list[str]:
    """Overlay environment variables onto ``config``; return applied names."""
    env = dict(os.environ if environ is None else environ)
    applied: list[str] = []
    for name, (section_name, key, caster) in ENV_MAP.items():
        if name not in env:
            continue
        raw = env[name]
        if raw == "":
            continue
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, key):
            continue
        current = getattr(section, key)
        try:
            if caster is bool:
                value: Any = str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}
            elif caster is list:
                value = [item.strip() for item in raw.split(",") if item.strip()]
            elif caster is int:
                value = int(raw)
            elif caster is float:
                value = float(raw)
            else:
                value = raw
        except ValueError as exc:
            raise ConfigValidationError(f"{name}={raw!r} is not a valid {caster.__name__}") from exc
        setattr(section, key, value)
        applied.append(name)
    if not config.server.api_key:
        from .secrets import resolve_secret

        resolved = resolve_secret(f"env:{config.server.api_key_env}", default="")
        if resolved:
            config.server.api_key = resolved
            applied.append(config.server.api_key_env)
    return applied


def apply_overrides(config: Config, overrides: dict[str, Any]) -> list[str]:
    """Apply explicit dotted or section-level overrides."""
    applied: list[str] = []
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if "." in key:
            config.set(key, value)
            applied.append(key)
            continue
        if key in SECTION_TYPES and isinstance(value, dict):
            section = getattr(config, key)
            for field_name, field_value in value.items():
                if hasattr(section, field_name):
                    current = getattr(section, field_name)
                    setattr(section, field_name, _coerce(field_value, None, current))
                    applied.append(f"{key}.{field_name}")
        elif hasattr(config, key):
            setattr(config, key, value)
            applied.append(key)
    return applied


def build_config_data(
    *,
    paths: list[Path] | None = None,
    profile: str = "",
    required: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Merge every TOML layer and return ``(data, source_labels)``."""
    candidates = paths if paths is not None else default_config_paths()
    merged: dict[str, Any] = {}
    sources: list[str] = []
    explicit_requested = bool(os.environ.get(CONFIG_ENV_VAR))
    for index, path in enumerate(candidates):
        is_explicit = index == len(candidates) - 1 and explicit_requested
        if is_explicit and not path.is_file() and required:
            raise ConfigFileNotFound(f"{CONFIG_ENV_VAR} points at a missing file: {path}")
        data = load_toml(path, required=required and is_explicit)
        if data:
            merged = deep_merge(merged, data)
            sources.append(str(path))
    active_profile = profile or merged.get("profile") or os.environ.get(PROFILE_ENV_VAR) or "default"
    merged = _apply_profiles(merged, str(active_profile))
    merged.setdefault("profile", str(active_profile))
    return merged, sources


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build a :class:`Config` while preserving extension-owned settings.

    Unknown fields inside a known section are grouped under
    ``extra['unknown_keys']``. Entirely unknown top-level tables are copied into
    :attr:`Config.extra` under their original name, which lets plugins keep
    configuration beside MAgent's own tables without the core interpreting it.
    """
    config = Config()
    unknown: dict[str, Any] = {}
    for name, section_type in SECTION_TYPES.items():
        raw = data.get(name)
        if not isinstance(raw, dict):
            continue
        section = getattr(config, name)
        for key, value in raw.items():
            if not hasattr(section, key):
                unknown.setdefault(name, {})[key] = copy.deepcopy(value)
                continue
            current = getattr(section, key)
            setattr(section, key, _coerce(value, None, current))
    config.profile = str(data.get("profile", "default"))
    supplied_extra = data.get("extra")
    if isinstance(supplied_extra, dict):
        config.extra.update(copy.deepcopy(supplied_extra))
    metadata = {"profile", "profiles", "sources", "config_path", "data_dir", "cache_dir",
                "extra"}
    for name, value in data.items():
        if name not in SECTION_TYPES and name not in metadata:
            config.extra[name] = copy.deepcopy(value)
    if unknown:
        config.extra["unknown_keys"] = unknown
    return config


def load_config(
    *,
    path: str | Path | None = None,
    profile: str = "",
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
    required: bool = False,
) -> Config:
    """Load, layer, validate, and return the effective configuration."""
    paths = [Path(path).expanduser()] if path else None
    if path and not Path(path).expanduser().is_file():
        if required:
            raise ConfigFileNotFound(f"Configuration file not found: {path}")
        paths = None
    data, sources = build_config_data(paths=paths, profile=profile, required=required)
    config = config_from_dict(data)
    config.sources = sources
    config.config_path = Path(path).expanduser() if path else (Path(sources[-1]) if sources else None)
    if use_env:
        applied = apply_env(config)
        if applied:
            config.sources.append(f"env:{len(applied)} vars")
    if overrides:
        applied_overrides = apply_overrides(config, overrides)
        if applied_overrides:
            config.sources.append(f"overrides:{len(applied_overrides)}")
    config.validate()
    return config


def default_config() -> Config:
    """Return a validated configuration built purely from defaults."""
    return load_config(use_env=False)


def write_example_config(path: str | Path, *, force: bool = False) -> Path:
    """Write a fully commented example ``magent.toml``."""
    from .utils.files import atomic_write_text

    target = Path(path).expanduser()
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists; pass force=True to overwrite")
    atomic_write_text(target, EXAMPLE_CONFIG)
    return target


EXAMPLE_CONFIG = '''# MAgent configuration
# Every key is optional; uncomment and edit what you need.
# Run `magent config list` to see effective values and where they came from.

profile = "default"

[server]
# Where Cortex LLMHoster is listening.
base_url = "http://127.0.0.1:8624"
# Reference a secret instead of writing it here: env:NAME or file:/path
api_key = "env:MAGENT_CORTEX_API_KEY"
api_prefix = "/v1"
timeout = 120.0
connect_timeout = 10.0
max_retries = 3
verify_ssl = true

[model]
# auto | cortex | echo | scripted | router | lambda
backend = "auto"
model = ""
embedding_model = ""
planner_model = ""
context_size = 8192
max_output_tokens = 2048
fallback_to_echo = true
auto_start_models = true

[generation]
temperature = 0.4
top_p = 1.0
max_tokens = 2048
stream = true
# seed = 1234

[agent]
max_iterations = 12
loop_mode = "react"       # react | plan | reflect | single
parallel_tools = true
planning = false
reflection = false
persona = "assistant"
tool_timeout = 60.0
confirm_destructive = true

[memory]
enabled = true
max_messages = 60
max_context_tokens = 6000
trim_strategy = "middle"  # middle | oldest | newest
summarize = true
retrieval_mode = "hybrid" # vector | bm25 | hybrid | none
retrieval_top_k = 5

[tools]
# Empty enabled list means "all built-in tools".
enabled = []
disabled = []
approval_mode = "dangerous"   # never | ask | always | dangerous
allow_network = false
max_file_read_bytes = 2097152
command_timeout = 60.0

[guardrails]
enabled = true
injection_detection = true
injection_action = "warn"     # warn | block | strip | off
pii_redaction = true
pii_action = "redact"         # redact | mask | block | off
budget_max_tokens = 0         # 0 = unlimited
budget_max_seconds = 0.0
audit_enabled = true

[cache]
enabled = true
ttl = 3600.0
max_bytes = 268435456
embed_cache = true

[output]
format = "text"   # text | json | markdown | html | csv | jsonl
color = true
unicode = true
show_usage = true
show_tool_calls = true
markdown = true
verbosity = "normal"

[session]
enabled = true
autosave = true
keep_last = 50
resume_last = false
transcript_format = "jsonl"

[observability]
log_level = "INFO"
log_format = "text"   # text | json
trace_enabled = true
metrics_enabled = true
redact_secrets = true

[http]
host = "127.0.0.1"
port = 8700
api_key = ""
serve_ui = true
threads = 16

[eval]
dataset = "builtin"
repeats = 1
concurrency = 4
report_format = "markdown"

# Named profiles override the tables above.
[profiles.fast]
generation = { temperature = 0.2, max_tokens = 1024 }
agent = { max_iterations = 6, parallel_tools = true }

[profiles.careful]
tools = { approval_mode = "always" }
guardrails = { injection_action = "block", pii_action = "block" }

[profiles.offline]
model = { backend = "echo", fallback_to_echo = true }
'''
"""Commented example configuration written by ``magent config init``."""


__all__ = [
    "AgentConfig",
    "CacheConfig",
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAME",
    "Config",
    "ConfigError",
    "ENV_MAP",
    "ENV_PREFIX",
    "EvalConfig",
    "EXAMPLE_CONFIG",
    "GenerationConfig",
    "GuardrailsConfig",
    "HttpServerConfig",
    "MemoryConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "OutputConfig",
    "PROFILE_ENV_VAR",
    "SEARCH_ORDER",
    "SECTION_TYPES",
    "ServerConfig",
    "SessionConfig",
    "ToolsConfig",
    "apply_env",
    "apply_overrides",
    "build_config_data",
    "config_from_dict",
    "default_config",
    "default_config_paths",
    "load_config",
    "load_toml",
    "system_config_dir",
    "user_cache_dir",
    "user_config_dir",
    "user_data_dir",
    "write_example_config",
]
