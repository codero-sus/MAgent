"""Ready-made fixtures: sample corpora, offline agents, temp workspaces."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import CompletionRequest
from ..llm.echo import EchoBackend, ScriptedBackend, ToolScriptBackend
from ..types import Message

SAMPLE_DOCUMENTS: dict[str, str] = {
    "readme.md": (
        "# MAgent\n\n"
        "Mr. Agent, Powerful. A zero-dependency Python agent framework built for\n"
        "Cortex LLMHoster. It plans, calls tools, remembers, and explains itself.\n"
    ),
    "quickstart.md": (
        "# Quickstart\n\n"
        "1. Start Cortex LLMHoster.\n"
        "2. Export MAGENT_CORTEX_API_KEY.\n"
        "3. Run `magent chat`.\n\n"
        "MAgent probes http://127.0.0.1:8624/health automatically.\n"
    ),
    "tools.md": (
        "# Tools\n\n"
        "MAgent ships file, shell, web, python, search, data, git, memory and\n"
        "Cortex control tools. Every tool declares a JSON schema and runs inside\n"
        "a sandbox with an approval policy.\n"
    ),
    "memory.md": (
        "# Memory\n\n"
        "Short term memory is a token-aware sliding window. Long term memory is a\n"
        "SQLite store with BM25 plus vector hybrid retrieval and recursive\n"
        "summarization.\n"
    ),
    "cortex.md": (
        "# Cortex LLMHoster integration\n\n"
        "Cortex hosts local models behind an OpenAI-compatible API at /v1 and an\n"
        "admin control plane at /admin/api. By default Cortex listens on port\n"
        "8624 at 127.0.0.1, and MAgent probes /health before choosing a backend.\n"
        "MAgent can list, start, stop and restart models, read metrics, and\n"
        "inspect hardware.\n"
    ),
    "guardrails.md": (
        "# Guardrails\n\n"
        "Prompt-injection heuristics, PII redaction, path jails, command\n"
        "allowlists, token budgets, rate limits and an append-only audit log.\n"
    ),
    "faq.md": (
        "# FAQ\n\n"
        "Q: Does MAgent need a cloud API key? A: No. It runs against local\n"
        "models hosted by Cortex LLMHoster.\n\n"
        "Q: Does it need dependencies? A: No, the standard library is enough.\n"
    ),
    "changelog.md": (
        "# Changelog\n\n"
        "## 1.0.0\n\n"
        "- Initial public release with 777+ verified improvements.\n"
        "- Cortex LLMHoster backend, tool sandbox, hybrid memory, web UI.\n"
    ),
}
"""Small document corpus used by RAG tests and demos."""

SAMPLE_CODE: dict[str, str] = {
    "app.py": (
        "def add(a, b):\n"
        "    \"\"\"Return the sum of two numbers.\"\"\"\n"
        "    return a + b\n\n\n"
        "def divide(a, b):\n"
        "    \"\"\"Return a divided by b, or None when b is zero.\"\"\"\n"
        "    if b == 0:\n"
        "        return None\n"
        "    return a / b\n"
    ),
    "notes.txt": (
        "TODO: add retries\n"
        "TODO: cache embeddings\n"
        "DONE: streaming support\n"
        "FIXME: token estimate drifts on CJK\n"
    ),
}
"""Sample source files written into temp workspaces."""


@dataclass(slots=True)
class SampleCorpus:
    """A corpus of sample documents plus convenience accessors."""

    documents: dict[str, str] = field(default_factory=lambda: dict(SAMPLE_DOCUMENTS))

    def texts(self) -> list[str]:
        """Return the document bodies in insertion order."""
        return list(self.documents.values())

    def pairs(self) -> list[tuple[str, str]]:
        """Return ``(name, text)`` pairs."""
        return list(self.documents.items())

    def write_to(self, directory: str | Path) -> list[Path]:
        """Materialize the corpus as real files and return their paths."""
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, text in self.documents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            written.append(target)
        return written

    def query_answers(self) -> list[dict[str, str]]:
        """Return question/answer pairs for retrieval evaluation."""
        return [
            {
                "question": "Does MAgent need a cloud API key?",
                "answer": "No. It runs against local models hosted by Cortex LLMHoster.",
                "source": "faq.md",
            },
            {
                "question": "How does MAgent store long term memory?",
                "answer": "SQLite store with BM25 plus vector hybrid retrieval.",
                "source": "memory.md",
            },
            {
                "question": "What port does Cortex listen on by default?",
                "answer": "8624",
                "source": "cortex.md",
            },
            {
                "question": "What guardrails does MAgent include?",
                "answer": "Prompt-injection heuristics, PII redaction, path jails, budgets.",
                "source": "guardrails.md",
            },
        ]


def build_sample_corpus(include_code: bool = False) -> SampleCorpus:
    """Return the sample corpus, optionally including code fixtures."""
    documents = dict(SAMPLE_DOCUMENTS)
    if include_code:
        documents.update(SAMPLE_CODE)
    return SampleCorpus(documents=documents)


@contextlib.contextmanager
def temp_workspace(corpus: SampleCorpus | None = None) -> Iterator[Path]:
    """Yield a temporary directory pre-populated with sample files."""
    root = Path(tempfile.mkdtemp(prefix="magent-ws-"))
    try:
        payload = corpus or build_sample_corpus(include_code=True)
        payload.write_to(root)
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def offline_backend(model: str = "echo-small") -> EchoBackend:
    """Return an offline echo backend for demos and tests."""
    return EchoBackend(models_list=[model])


def make_backend(**kwargs: Any) -> Any:
    """Return an offline backend, choosing the variant by keyword."""
    if kwargs.get("script"):
        return ScriptedBackend(script=list(kwargs.pop("script")), **kwargs)
    if kwargs.get("tool_name"):
        return ToolScriptBackend(
            tool_name=kwargs.pop("tool_name"),
            tool_arguments=dict(kwargs.pop("tool_arguments", {})),
            final_text=kwargs.pop("final_text", "done"),
        )
    return offline_backend(**kwargs)


def make_request(text: str, *, system: str = "", **kwargs: Any) -> CompletionRequest:
    """Build a :class:`CompletionRequest` from a single prompt."""
    messages: list[Message] = []
    if system:
        messages.append(Message.system(system))
    messages.append(Message.user(text))
    return CompletionRequest(messages=messages, **kwargs)


def make_agent(**kwargs: Any) -> Any:
    """Build a fully wired offline agent, importing lazily to avoid cycles.

    Loop settings (``max_iterations``, ``loop_mode``, ``reflection``, ...) become
    the :class:`~magent.config.AgentConfig`; everything else - ``workspace``,
    ``registry``, ``sandbox``, ``memory`` - is passed to the agent constructor.
    """
    from ..agent import make_agent as build_agent

    backend = kwargs.pop("backend", None) or offline_backend()
    return build_agent(backend, **kwargs)


def scripted_tool_agent(
    tool_name: str = "calculator",
    tool_arguments: dict[str, Any] | None = None,
    final_text: str = "The answer is 4.",
    **kwargs: Any,
) -> Any:
    """Build an agent whose backend asks for one tool call then answers."""
    backend = ToolScriptBackend(
        tool_name=tool_name,
        tool_arguments=dict(tool_arguments or {"expression": "2+2"}),
        final_text=final_text,
    )
    return make_agent(backend=backend, **kwargs)


def sample_messages(count: int = 6) -> list[Message]:
    """Return a small synthetic conversation."""
    messages: list[Message] = [Message.system("You are MAgent, a helpful local assistant.")]
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        if role == "user":
            messages.append(Message.user(f"Question number {index}: how does this work?"))
        else:
            messages.append(Message.assistant(f"Answer number {index}: it works step by step."))
    return messages


__all__ = [
    "SAMPLE_CODE",
    "SAMPLE_DOCUMENTS",
    "SampleCorpus",
    "build_sample_corpus",
    "make_agent",
    "make_backend",
    "make_request",
    "offline_backend",
    "sample_messages",
    "scripted_tool_agent",
    "temp_workspace",
]
