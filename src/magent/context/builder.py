"""The context builder: one place that decides what the model gets to see.

An agent run has many sources of context - the workspace, retrieved documents,
long-term memory, the running session summary, user-supplied notes, the tools
themselves.  Each has a different value per token, and together they routinely
exceed the window.  :class:`ContextBuilder` collects them as prioritised
:class:`~magent.context.block.ContextBlock` objects, asks the
:class:`~magent.context.compression.ContextCompressor` to fit them into a
budget, and returns a :class:`ContextAssembly` that can be dropped straight into
the system prompt.

The builder never calls the network on its own: retrieval and summarisation use
whatever objects were injected, so a run stays reproducible and testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..llm.tokens import TokenEstimator, estimate_tokens
from ..logging_ import get_logger
from ..prompts.builder import describe_environment
from .block import (ContextBlock, custom_block, environment_block, memory_block, rag_block,
                    render_blocks, session_block, task_block, total_tokens)
from .compression import CompressionReport, ContextCompressor
from .files import describe_workspace, file_context_blocks

LOGGER = get_logger("context.builder")
"""Logger for context assembly."""

DEFAULT_BUDGET = 4_000
"""Default token budget for injected context."""

RESERVE_FOR_PROMPT = 1_200
"""Tokens kept free for the identity, rules and tool sections of the prompt."""


@dataclass(slots=True)
class ContextAssembly:
    """The assembled context for one run."""

    blocks: list[ContextBlock] = field(default_factory=list)
    text: str = ""
    tokens: int = 0
    budget: int = 0
    report: CompressionReport = field(default_factory=CompressionReport)
    stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def dropped(self) -> list[str]:
        """Titles of blocks that did not fit."""
        return list(self.report.dropped)

    @property
    def truncated(self) -> list[str]:
        """Titles of blocks that were shortened."""
        return list(self.report.truncated)

    @property
    def fits(self) -> bool:
        """True when the assembly is inside its budget."""
        return bool(self.report.fits)

    @property
    def empty(self) -> bool:
        """True when nothing was assembled."""
        return not self.text.strip()

    def kinds(self) -> dict[str, int]:
        """Token count per block kind."""
        out: dict[str, int] = {}
        for item in self.blocks:
            out[item.kind] = out.get(item.kind, 0) + item.tokens
        return out

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the assembly for events, logs and session records."""
        payload: dict[str, Any] = {
            "tokens": self.tokens,
            "budget": self.budget,
            "seconds": round(self.seconds, 4),
            "blocks": [item.to_dict() for item in self.blocks],
            "report": self.report.to_dict(),
            "stats": dict(self.stats),
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def summary(self) -> str:
        """One-line description, suitable for the ``context`` event."""
        kinds = ", ".join(f"{kind}={tokens}" for kind, tokens in sorted(self.kinds().items()))
        return f"{self.tokens}/{self.budget} tokens [{kinds or 'empty'}] {self.report.summary()}"


class ContextBuilder:
    """Collect, prioritise and compress context blocks."""

    def __init__(self, *, budget: int = DEFAULT_BUDGET, workspace: str | Path | None = None,
                 sandbox: Any = None, retriever: Any = None, memory: Any = None,
                 short_term: Any = None, summarizer: Callable[[str], str] | None = None,
                 estimator: TokenEstimator | None = None, style: str = "heading",
                 include_files: bool = False, include_environment: bool = True,
                 top_k: int = 4, memory_top_k: int = 4, reserve: int = RESERVE_FOR_PROMPT,
                 strip_boilerplate: bool = True) -> None:
        """Configure a builder.

        Args:
            budget: token ceiling for injected context
            workspace: directory scanned for files and environment facts
            sandbox: optional sandbox authorizing every file read
            retriever: object with ``retrieve(query, limit=)`` for RAG context
            memory: object with ``search(query, limit=)`` for remembered facts
            short_term: object with ``window.summary`` / ``transcript()``
            summarizer: callable used to summarize oversized blocks
            estimator: token estimator shared with the rest of the run
            style: block rendering style passed to :func:`render_blocks`
            include_files: attach query-ranked workspace files by default
            include_environment: attach the environment description
            top_k: retrieved chunks to request
            memory_top_k: memories to request
            reserve: tokens kept for the fixed prompt sections
            strip_boilerplate: let the compressor clean block text
        """
        self.budget = max(0, int(budget))
        self.workspace = Path(workspace).expanduser() if workspace else Path.cwd()
        self.sandbox = sandbox
        self.retriever = retriever
        self.memory = memory
        self.short_term = short_term
        self.summarizer = summarizer
        self.estimator = estimator
        self.style = style
        self.include_files = bool(include_files)
        self.include_environment = bool(include_environment)
        self.top_k = max(1, int(top_k))
        self.memory_top_k = max(1, int(memory_top_k))
        self.reserve = max(0, int(reserve))
        self.stats_files: dict[str, Any] = {}
        self.compressor = ContextCompressor(budget=self.budget, summarizer=summarizer,
                                            estimator=estimator, style=style,
                                            strip_boilerplate=strip_boilerplate)
        self._blocks: list[ContextBlock] = []

    @classmethod
    def from_config(cls, config: Any = None, *, backend: Any = None, workspace: str | Path | None = None,
                    memory: Any = None, retriever: Any = None, short_term: Any = None,
                    sandbox: Any = None, summarizer: Callable[[str], str] | None = None,
                    include_files: bool = False) -> ContextBuilder:
        """Build a context builder from a :class:`~magent.config.Config`.

        Args:
            config: loaded configuration; ``None`` uses defaults
            backend: LLM backend, used to size the budget from the context window
            workspace: override for the configured tool workspace
            memory: long-term memory store
            retriever: RAG retriever
            short_term: short-term memory
            sandbox: tool sandbox
            summarizer: callable used for oversized blocks
            include_files: attach query-ranked workspace files by default
        """
        memory_config = getattr(config, "memory", None)
        tools_config = getattr(config, "tools", None)
        model_config = getattr(config, "model", None)
        budget = DEFAULT_BUDGET
        if memory_config is not None:
            budget = int(getattr(memory_config, "max_context_tokens", 0) or 0) // 2 or DEFAULT_BUDGET
        context_size = int(getattr(model_config, "context_size", 0) or 0)
        if context_size:
            budget = min(budget, max(1_000, context_size // 3))
        top_k = int(getattr(memory_config, "retrieval_top_k", 4) or 4)
        root = workspace or getattr(tools_config, "workspace", None) or Path.cwd()
        return cls(budget=budget, workspace=root, sandbox=sandbox, retriever=retriever,
                   memory=memory, short_term=short_term, summarizer=summarizer,
                   include_files=include_files, top_k=top_k, memory_top_k=top_k)

    # -- collecting ----------------------------------------------------------

    def add(self, block: ContextBlock) -> ContextBlock:
        """Append a pre-built block."""
        self._blocks.append(block)
        return block

    def add_text(self, title: str, body: str, **kwargs: Any) -> ContextBlock:
        """Append a block built from a title and body."""
        return self.add(custom_block(title, body, **kwargs))

    def add_blocks(self, blocks: Iterable[ContextBlock]) -> int:
        """Append many blocks; returns how many were added."""
        added = 0
        for item in blocks:
            if item is not None and not item.empty:
                self.add(item)
                added += 1
        return added

    def clear(self) -> None:
        """Drop all collected blocks."""
        self._blocks.clear()

    @property
    def blocks(self) -> list[ContextBlock]:
        """Blocks collected so far."""
        return list(self._blocks)

    def tokens(self) -> int:
        """Tokens currently collected, before compression."""
        return total_tokens(self._blocks)

    # -- gathering from sources ---------------------------------------------

    def gather_environment(self, *, extra: str = "", model: str = "") -> ContextBlock | None:
        """Describe the runtime environment and workspace."""
        if not self.include_environment:
            return None
        parts = [describe_environment(self.workspace, include_files=False, include_git=True)]
        try:
            parts.append(describe_workspace(self.workspace, limit=12, sandbox=self.sandbox))
        except Exception as exc:  # noqa: BLE001 - workspace scanning is best effort
            LOGGER.debug("workspace description failed: %s", exc)
        if extra.strip():
            parts.append(extra.strip())
        block = environment_block("\n\n".join(part for part in parts if part))
        return self.add(block)

    def gather_rag(self, query: str, *, limit: int | None = None) -> ContextBlock | None:
        """Retrieve documents and add them as a high-priority block."""
        if self.retriever is None or not query.strip():
            return None
        try:
            result = self.retriever.retrieve(query, limit=limit or self.top_k)
        except Exception as exc:  # noqa: BLE001 - retrieval failure must not abort a run
            LOGGER.warning("retrieval failed: %s", exc)
            return None
        chunks = list(getattr(result, "chunks", []) or [])
        if not chunks:
            return None
        block = rag_block(chunks)
        block.metadata["query"] = query
        block.metadata["chunks"] = len(chunks)
        return self.add(block)

    def gather_memory(self, query: str, *, limit: int | None = None) -> ContextBlock | None:
        """Recall stored memories relevant to the query."""
        if self.memory is None or not query.strip():
            return None
        try:
            entries = self.memory.search(query, limit=limit or self.memory_top_k)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("memory recall failed: %s", exc)
            return None
        entries = [entry for entry in entries or [] if getattr(entry, "content", "")]
        if not entries:
            return None
        block = memory_block(entries)
        block.metadata["entries"] = len(entries)
        return self.add(block)

    def gather_session(self) -> ContextBlock | None:
        """Add the rolling summary of the current session."""
        if self.short_term is None:
            return None
        window = getattr(self.short_term, "window", None)
        summary = str(getattr(window, "summary", "") or "")
        transcript = ""
        if callable(getattr(self.short_term, "transcript", None)):
            try:
                transcript = self.short_term.transcript(limit=1_600)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("transcript failed: %s", exc)
        if not summary.strip() and not transcript.strip():
            return None
        return self.add(session_block(summary, transcript=transcript))

    def gather_files(self, query: str, *, named: Sequence[str] = ()) -> list[ContextBlock]:
        """Add workspace files that are named by the user or match the query."""
        blocks, stats = file_context_blocks(
            self.workspace, query=query, named=named, include_conventions=True,
            limit=max(2, self.top_k), sandbox=self.sandbox,
        )
        self.stats_files = stats
        added = [block for block in blocks if not block.empty]
        self.add_blocks(added)
        return added

    def gather_task(self, task: str) -> ContextBlock | None:
        """Add the user's task as a required block."""
        if not task.strip():
            return None
        return self.add(task_block(task))

    # -- assembling ----------------------------------------------------------

    def build(self, query: str = "", *, task: str = "", named_files: Sequence[str] = (),
              budget: int | None = None, extra_blocks: Iterable[ContextBlock] = (),
              include_rag: bool = True, include_memory: bool = True,
              include_session: bool = True, include_files: bool | None = None) -> ContextAssembly:
        """Assemble and compress every context source.

        Args:
            query: text used for retrieval, memory recall and file ranking
            task: the user's task, added as a required block
            named_files: files the user mentioned explicitly
            budget: override the configured token budget
            extra_blocks: caller-supplied blocks
            include_rag: allow the retrieval block
            include_memory: allow the memory block
            include_session: allow the session summary block
            include_files: override the configured file-inclusion default

        Returns:
            a :class:`ContextAssembly` whose ``text`` is prompt-ready
        """
        started = time.perf_counter()
        self.stats_files = {}
        candidates = list(self._blocks)
        if include_rag and query.strip():
            rag = self.gather_rag(query)
            if rag is not None:
                candidates.append(rag)
        if include_memory and query.strip():
            recalled = self.gather_memory(query)
            if recalled is not None:
                candidates.append(recalled)
        if include_session:
            session = self.gather_session()
            if session is not None:
                candidates.append(session)
        want_files = self.include_files if include_files is None else include_files
        if want_files and (query.strip() or named_files):
            candidates.extend(self.gather_files(query, named=named_files))
        if task.strip():
            candidates.append(task_block(task))
        candidates.extend(block for block in extra_blocks if block is not None and not block.empty)
        limit = int(budget if budget is not None else self.budget)
        kept, report = self.compressor.fit(candidates, budget=limit)
        text = render_blocks(kept, style=self.style)
        assembly = ContextAssembly(
            blocks=kept, text=text, tokens=estimate_tokens(text) if text else 0,
            budget=limit, report=report,
            stats={"files": self.stats_files, "sources": {item.kind for item in kept}},
            seconds=time.perf_counter() - started,
        )
        assembly.stats["sources"] = sorted(assembly.stats["sources"])
        LOGGER.debug("context assembled: %s", assembly.summary())
        return assembly

    def render(self, query: str = "", **kwargs: Any) -> str:
        """Build and return just the prompt text."""
        return self.build(query, **kwargs).text

    def describe(self) -> dict[str, Any]:
        """Describe the builder configuration, for ``magent doctor``."""
        return {
            "budget": self.budget,
            "reserve": self.reserve,
            "workspace": str(self.workspace),
            "style": self.style,
            "include_files": self.include_files,
            "include_environment": self.include_environment,
            "top_k": self.top_k,
            "memory_top_k": self.memory_top_k,
            "retriever": type(self.retriever).__name__ if self.retriever else None,
            "memory": type(self.memory).__name__ if self.memory else None,
            "short_term": type(self.short_term).__name__ if self.short_term else None,
            "summarizer": bool(self.summarizer),
            "pending_blocks": len(self._blocks),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"ContextBuilder(budget={self.budget}, blocks={len(self._blocks)})"


def build_context(query: str = "", *, budget: int = DEFAULT_BUDGET,
                  workspace: str | Path | None = None, retriever: Any = None,
                  memory: Any = None, short_term: Any = None, sandbox: Any = None,
                  summarizer: Callable[[str], str] | None = None,
                  blocks: Iterable[ContextBlock] = (), **kwargs: Any) -> ContextAssembly:
    """One-shot helper that builds a :class:`ContextBuilder` and assembles it."""
    builder = ContextBuilder(budget=budget, workspace=workspace, retriever=retriever,
                             memory=memory, short_term=short_term, sandbox=sandbox,
                             summarizer=summarizer)
    builder.add_blocks(blocks)
    return builder.build(query, **kwargs)


__all__ = [
    "DEFAULT_BUDGET",
    "RESERVE_FOR_PROMPT",
    "ContextAssembly",
    "ContextBuilder",
    "build_context",
]
