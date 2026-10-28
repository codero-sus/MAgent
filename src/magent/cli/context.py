"""The CLI context: everything a command needs, built lazily.

Starting ``magent`` must not open the network, load a model list or scan the
workspace - ``magent --help`` and ``magent version`` have to be instant.  So
:class:`CliContext` holds the loaded configuration, the renderer and the output
sink, and builds the backend, tool registry, memory store, session manager,
guardrails, retriever and observability only when a command asks for them
(``ctx.backend``, ``ctx.registry``, ...).  Each accessor caches, so a command
that needs three of them still builds each once.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..config import Config, default_config, load_config
from ..logging_ import get_logger
from ..output.format import normalize_format
from ..output.render import RenderOptions, ResultRenderer
from ..output.sink import ConsoleSink, OutputSink, TeeSink

LOGGER = get_logger("cli")
"""Logger for CLI commands."""


@dataclass(slots=True)
class CliContext:
    """Shared state for one CLI invocation."""

    config: Config
    args: Any = None
    sink: OutputSink | None = None
    renderer: ResultRenderer | None = None
    started_at: float = field(default_factory=time.time)
    verbose: bool = False
    quiet: bool = False
    color: bool | None = None
    _cache: dict[str, Any] = field(default_factory=dict)

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls, args: Any = None, *, config: Config | None = None, sink: OutputSink | None = None
    ) -> CliContext:
        """Create a context from parsed arguments.

        Args:
            args: the argparse namespace
            config: pre-loaded configuration, loaded from disk when omitted
            sink: output sink, built from the configuration when omitted
        """
        if config is None:
            overrides = getattr(args, "set", None) or []
            profile = getattr(args, "profile", "") or ""
            path = getattr(args, "config", "") or ""
            config = load_config(path=path, profile=profile, overrides=overrides)
        context = cls(config=config, args=args)
        context.quiet = bool(getattr(args, "quiet", False) or config.output.quiet)
        context.verbose = bool(getattr(args, "verbose", False))
        context.color = getattr(args, "color", None)
        context.apply_output_overrides()
        context.sink = sink if sink is not None else context.build_sink()
        context.renderer = ResultRenderer(
            RenderOptions.from_config(config),
            sink=context.sink,
            width=int(config.output.width or 0),
        )
        return context

    def apply_output_overrides(self) -> None:
        """Fold CLI flags into the output configuration."""
        args = self.args
        if args is None:
            return
        if self.quiet:
            self.config.output.quiet = True
            self.config.output.verbosity = "quiet"
            self.config.output.show_usage = False
        fmt = getattr(args, "format", "") or ""
        if fmt:
            self.config.output.format = normalize_format(fmt)
        if getattr(args, "no_color", False) or self.color is False:
            self.config.output.color = False
        if getattr(args, "color", None) is True:
            self.config.output.color = True
        if getattr(args, "ascii", False):
            self.config.output.unicode = False
        if getattr(args, "width", 0):
            self.config.output.width = int(args.width)
        if getattr(args, "json", False):
            self.config.output.format = "json"
        if getattr(args, "jsonl", False):
            self.config.output.format = "jsonl"
        if getattr(args, "yaml", False):
            self.config.output.format = "yaml"
        if self.verbose:
            self.config.output.verbosity = "verbose"

    def build_sink(self) -> OutputSink:
        """Create the output sink, teeing to a file when configured."""
        from ..output.sink import FileSink, NullSink

        path = (getattr(self.args, "output_file", "") or "") if self.args else ""
        quiet = self.config.output.quiet
        sinks: list[OutputSink] = []
        if not quiet:
            sinks.append(ConsoleSink(color=self.config.output.color))
        if path:
            sinks.append(FileSink(path))
        if not sinks:
            return NullSink()
        if len(sinks) == 1:
            return sinks[0]
        return TeeSink(*sinks)

    # -- lazy dependencies ---------------------------------------------------

    def cached(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return a cached value, building it on first use.

        A factory that raises stores nothing, so a transient failure (a backend
        that is not running yet) does not poison later attempts.
        """
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def backend(self) -> Any:
        """The LLM backend, chosen by the factory and cached."""
        return self.cached("backend", lambda: self._build_backend())

    def _build_backend(self) -> Any:
        """Construct the backend, honouring ``--backend`` and ``--model``."""
        from ..llm.factory import build_backend

        args = self.args
        overrides: dict[str, Any] = {}
        name = getattr(args, "backend", "") or ""
        model = getattr(args, "model", "") or ""
        settings = self.config
        if name:
            settings.model.backend = name
            overrides["kind"] = name
        if model:
            settings.model.model = model
            overrides["model"] = model
        offline = True if getattr(args, "offline", False) else None
        return build_backend(config=settings, offline=offline, **overrides)

    @property
    def registry(self) -> Any:
        """The tool registry, with plugins loaded once."""
        return self.cached("registry", self._build_registry)

    def _build_registry(self) -> Any:
        """Build the default registry and load configured plugins into it."""
        from ..plugins import load_plugins
        from ..tools import default_registry

        registry = default_registry()
        hooks = self.hooks
        loader = load_plugins(
            self.config, registry=registry, hooks=hooks, workspace=self.workspace, strict=False
        )
        self._cache["plugin_loader"] = loader
        return registry

    @property
    def hooks(self) -> Any:
        """The plugin hook registry."""
        return self.cached("hooks", self._build_hooks)

    def _build_hooks(self) -> Any:
        """Create an empty hook registry for plugins to populate."""
        from ..plugins import HookRegistry

        return HookRegistry()

    @property
    def plugin_loader(self) -> Any:
        """The plugin loader, after the registry has been built."""
        _ = self.registry
        return self._cache.get("plugin_loader")

    @property
    def workspace(self) -> Path:
        """Workspace root for this invocation."""
        args = self.args
        explicit = getattr(args, "workspace", "") if args else ""
        root = explicit or self.config.tools.workspace or Path.cwd()
        return Path(root).expanduser().resolve()

    @property
    def memory(self) -> Any:
        """Long-term memory store (``None`` when disabled)."""
        if not self.config.memory.enabled:
            return None
        return self.cached("memory", lambda: self._build_memory())

    def _build_memory(self) -> Any:
        """Open the memory store, using the backend embedder when available."""
        from ..memory.store import build_store

        backend = None
        try:
            backend = self.backend
        except Exception as exc:
            LOGGER.debug("memory embedder backend unavailable: %s", exc)
        return build_store(self.config, backend=backend, root=self.config.data_dir)

    @property
    def sessions(self) -> Any:
        """Session manager."""
        return self.cached("sessions", lambda: self._build_sessions())

    def _build_sessions(self) -> Any:
        """Build the session manager, sharing the memory store's history window."""
        from ..session import build_session_manager

        return build_session_manager(self.config)

    @property
    def guardrails(self) -> Any:
        """Guardrail manager (``None`` when disabled)."""
        if not self.config.guardrails.enabled:
            return None
        return self.cached("guardrails", lambda: self._build_guardrails())

    def _build_guardrails(self) -> Any:
        """Build guardrails bound to a fresh event bus."""
        from ..guardrails import build_guardrails

        return build_guardrails(self.config, events=self.events, data_dir=self.config.data_dir)

    @property
    def events(self) -> Any:
        """Event bus shared by agents built from this context."""
        return self.cached("events", self._build_events)

    def _build_events(self) -> Any:
        """Create the shared event bus."""
        from ..events import EventBus

        return EventBus()

    @property
    def observability(self) -> Any:
        """Observability façade, attached to the shared event bus."""
        return self.cached("observability", self._build_observability)

    def _build_observability(self) -> Any:
        """Build observability and attach it to the shared bus."""
        from ..observability import build_observability

        instance = build_observability(self.config, attach_to=None)
        instance.attach(bus=self.events)
        return instance

    @property
    def retriever(self) -> Any:
        """RAG retriever, indexed lazily."""
        return self.cached("retriever", lambda: self._build_retriever())

    def _build_retriever(self) -> Any:
        """Build a retriever bound to the workspace index under the cache dir."""
        from ..rag import Retriever

        index_path = Path(self.config.cache_dir or Path.cwd()) / "rag-index.json"
        backend = None
        try:
            backend = self.backend
        except Exception as exc:
            LOGGER.debug("retriever backend unavailable: %s", exc)
        retriever = Retriever(
            backend=backend, max_chars=1_200, limit=self.config.memory.retrieval_top_k or 5
        )
        if index_path.is_file():
            try:
                retriever = Retriever.load(index_path, backend=backend)
            except Exception as exc:
                LOGGER.debug("could not load the RAG index: %s", exc)
        self._cache["rag_index_path"] = index_path
        return retriever

    @property
    def skills(self) -> Any:
        """Skill library for this workspace."""
        return self.cached("skills", lambda: self._build_skills())

    def _build_skills(self) -> Any:
        """Build the skill library, honouring ``--no-skills``."""
        from ..skills import build_library

        if getattr(self.args, "no_skills", False):
            from ..skills import SkillLibrary

            return SkillLibrary(include_builtins=True)
        return build_library(self.config, workspace=self.workspace)

    # -- agent construction --------------------------------------------------

    def build_agent(self, **overrides: Any) -> Any:
        """Create an agent wired to every component this context provides.

        Args:
            **overrides: passed straight to :class:`~magent.agent.Agent`
        """
        from ..agent import Agent

        args = self.args
        persona = getattr(args, "persona", "") or ""
        loop_mode = getattr(args, "loop_mode", "") or ""
        config = self.config.agent
        if persona:
            config.persona = persona
        if loop_mode:
            config.loop_mode = loop_mode
        session = self.sessions if self.config.session.enabled else None
        if "approval" not in overrides:
            import sys

            from ..tools.approval import ApprovalManager, ApprovalPolicy, prompter_for_mode
            from ..types import ApprovalMode

            mode = cast(ApprovalMode, self.config.tools.approval_mode)
            overrides["approval"] = ApprovalManager(
                ApprovalPolicy(mode=mode),
                prompt=prompter_for_mode(mode, interactive=sys.stdin.isatty()),
                events=self.events,
            )
        agent = Agent(
            backend=self.backend,
            settings=self.config,
            config=config,
            registry=self.registry,
            memory=self.memory,
            guardrails=self.guardrails,
            session=session,
            events=self.events,
            workspace=self.workspace,
            name=overrides.pop("name", "MAgent"),
            **overrides,
        )
        return agent

    def build_preset_agent(self, preset: str, **overrides: Any) -> Any:
        """Create an agent from a preset, keeping this context's components."""
        from ..agents.presets import build_preset_agent

        return build_preset_agent(
            preset,
            backend=self.backend,
            registry=self.registry,
            config=self.config,
            workspace=self.workspace,
            memory=self.memory,
            session=self.sessions,
            guardrails=self.guardrails,
            events=self.events,
            **overrides,
        )

    def run_agent(self, prompt: str, *, preset: str = "", **options: Any) -> Any:
        """Run one prompt through an agent (preset or default)."""
        agent = self.build_preset_agent(preset) if preset else self.build_agent()
        return agent.run(prompt, **options)

    # -- output helpers ------------------------------------------------------

    def print(self, text: str = "") -> None:
        """Write one line to the sink, unless quiet."""
        if self.sink is not None and not self.quiet:
            self.sink.line(text)

    def write(self, text: str) -> None:
        """Write text without a trailing newline."""
        if self.sink is not None and not self.quiet:
            self.sink.write(text)

    def emit(self, payload: Any, *, fmt: str = "", width: int = 0) -> None:
        """Render a value in the configured output format."""
        from ..output.format import render_value

        active = fmt or self.config.output.format
        text = render_value(payload, active, width=width or self.config.output.width or 0)
        if not text:
            return
        if self.renderer is not None and active == "text":
            self.print(text)
        else:
            self.write(text if text.endswith("\n") else text + "\n")

    def table(
        self, rows: Sequence[Sequence[Any]], headers: Sequence[str], *, title: str = ""
    ) -> None:
        """Render a table, or the raw rows when a machine format is selected."""
        fmt = self.config.output.format
        if fmt != "text":
            self.emit(
                [dict(zip([str(item) for item in headers], row, strict=False)) for row in rows],
                fmt=fmt,
            )
            return
        if self.renderer is None:
            return
        self.print(self.renderer.table(rows, headers, title=title))

    def key_values(self, pairs: Sequence[tuple[str, Any]], *, title: str = "") -> None:
        """Render aligned key/value output."""
        if self.config.output.format != "text":
            self.emit({key: value for key, value in pairs}, fmt=self.config.output.format)
            return
        if self.renderer is None:
            return
        self.print(self.renderer.key_value(pairs, title=title))

    def notice(self, text: str, *, level: str = "info") -> None:
        """Print a status line, honouring verbosity."""
        if self.renderer is None:
            return
        if level == "muted" and not self.verbose:
            return
        if level == "info" and self.config.output.verbosity in {"quiet"}:
            return
        self.renderer.notice(text, level=level)

    def error(self, text: str) -> None:
        """Print an error line."""
        if self.renderer is not None:
            self.renderer.print(self.renderer.render_error(text))
        elif self.sink is not None:
            self.sink.line(f"error: {text}")

    def progress(self, items: Sequence[Any], *, label: str = "") -> Any:
        """Iterate with a progress bar when the terminal allows it."""
        if self.renderer is None:
            return iter(items)
        return self.renderer.progress(items, label=label)

    def save_index(self) -> Path | None:
        """Persist the RAG index after an ingest, returning the path written."""
        retriever = self._cache.get("retriever")
        path = self._cache.get("rag_index_path")
        if retriever is None or path is None:
            return None
        retriever.save(path)
        return path

    @property
    def elapsed(self) -> float:
        """Seconds since the context was created."""
        return time.time() - self.started_at

    def close(self) -> None:
        """Release everything the context opened."""
        observability = self._cache.get("observability")
        if observability is not None:
            observability.close()
        memory = self._cache.get("memory")
        if memory is not None and hasattr(memory, "close"):
            try:
                memory.close()
            except Exception as exc:
                LOGGER.debug("memory close failed: %s", exc)
        guardrails = self._cache.get("guardrails")
        if guardrails is not None and hasattr(guardrails, "close"):
            guardrails.close()
        if self.sink is not None:
            self.sink.flush()

    def describe(self) -> dict[str, Any]:
        """Describe the context, for ``magent doctor``."""
        return {
            "workspace": str(self.workspace),
            "profile": self.config.profile,
            "config_path": str(self.config.config_path or ""),
            "data_dir": str(self.config.data_dir or ""),
            "cache_dir": str(self.config.cache_dir or ""),
            "format": self.config.output.format,
            "quiet": self.quiet,
            "verbose": self.verbose,
            "built": sorted(self._cache),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"CliContext(workspace={str(self.workspace)!r}, built={sorted(self._cache)})"


def build_context(args: Any = None, **kwargs: Any) -> CliContext:
    """Build a :class:`CliContext` from parsed arguments."""
    return CliContext.build(args, **kwargs)


def offline_context(**kwargs: Any) -> CliContext:
    """Build a context that never touches the network (used by tests)."""
    return CliContext.build(config=default_config(), **kwargs)


__all__ = [
    "CliContext",
    "build_context",
    "offline_context",
]
