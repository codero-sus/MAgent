#!/usr/bin/env python3
"""Run one offline end-to-end MAgent scenario.

This is deliberately broader than a unit test and faster than the full suite. It
executes a model/tool/model loop, speaks Cortex's protocol over a real loopback
socket (including SSE and embeddings), retrieves cited local documents, loads a
manifest-declared plugin, and resolves the generated improvement inventory.

No external network, model weights, GPU, package installation, or writes to the
user's actual home are required.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEMP = Path(tempfile.mkdtemp(prefix="magent-smoke-"))
os.environ["XDG_DATA_HOME"] = str(TEMP / "data")
os.environ["XDG_CONFIG_HOME"] = str(TEMP / "config")
os.environ["XDG_CACHE_HOME"] = str(TEMP / "cache")
os.environ.pop("MAGENT_CONFIG", None)

from magent import Agent, CortexBackend, CortexSettings, Message, default_registry, load_config  # noqa: E402
from magent.improvements import collect, verify  # noqa: E402
from magent.llm import CompletionRequest, collect_stream  # noqa: E402
from magent.plugins import HookRegistry, PluginLoader, scaffold  # noqa: E402
from magent.rag import Retriever  # noqa: E402
from magent.testing import (  # noqa: E402
    EMBEDDING_MODEL,
    build_sample_corpus,
    make_backend,
    start_fake_cortex,
)
from magent.tools import ToolContext, ToolRegistry  # noqa: E402
from magent.types import StopReason, ToolCall  # noqa: E402


class SmokeFailure(AssertionError):
    """Raised when one smoke-stage contract is not met."""


def require(condition: object, message: str) -> None:
    """Raise :class:`SmokeFailure` when ``condition`` is false."""
    if not condition:
        raise SmokeFailure(message)


def stage(name: str) -> None:
    """Print a stable stage marker for CI logs."""
    print(f"[smoke] {name}")


def smoke_agent() -> None:
    """Execute a real two-step tool loop and inspect the result."""
    stage("agent + calculate tool")
    config = load_config(use_env=False)
    backend = make_backend(
        tool_name="calculate",
        tool_arguments={"expression": "6*7"},
        final_text="The answer is 42.",
    )
    agent = Agent(backend=backend, config=config.agent, settings=config,
                  registry=default_registry(), workspace=ROOT)
    result = agent.run("calculate six times seven")
    require(result.stop_reason is StopReason.COMPLETED, result.summary_line())
    require(result.output == "The answer is 42.", result.output)
    require(result.tool_call_count == 1 and result.tool_error_count == 0,
            "expected one successful tool call")
    require(result.steps[0].tool_results[0].content["result"] == 42,
            "calculate did not return 42")
    print("  ", result.summary_line())


def smoke_cortex() -> None:
    """Exercise Cortex liveness, models, chat, streaming, and embeddings."""
    stage("Cortex HTTP contract")
    server = start_fake_cortex()
    backend = CortexBackend(CortexSettings(
        base_url=server.url,
        require_auth=False,
        max_retries=0,
        embedding_model=EMBEDDING_MODEL,
    ))
    try:
        require(backend.health()["ok"], "fake Cortex not healthy")
        require(len(backend.models()) >= 4, "model list did not parse")
        request = CompletionRequest(messages=[Message.user("hello")])
        require("hello" in backend.complete(request).text, "chat response did not parse")
        streamed = collect_stream(backend.stream(
            CompletionRequest(messages=[Message.user("stream me")], stream=True)))
        require(streamed.text.endswith("stream me"), "SSE text did not reassemble")
        vector = backend.embed_one("MAgent", model=EMBEDDING_MODEL)
        require(len(vector) == 32, "embedding shape mismatch")
        print(f"   {len(backend.models())} models · {streamed.usage.total_tokens} stream tokens")
    finally:
        backend.close()
        server.stop()


def smoke_rag() -> None:
    """Ingest the sample corpus and require the correct top source."""
    stage("hybrid retrieval + citations")
    corpus_root = TEMP / "corpus"
    build_sample_corpus().write_to(corpus_root)
    retriever = Retriever(mode="hybrid", limit=3, use_mmr=True)
    report = retriever.ingest([corpus_root])
    result = retriever.retrieve("What port does Cortex listen on by default?")
    require(report.files >= 8 and report.chunks >= 8, "sample corpus did not ingest")
    require(bool(result.chunks), "retrieval returned no chunks")
    require(Path(result.chunks[0].source).name == "cortex.md",
            f"wrong top source: {result.chunks[0].source}")
    check = result.verify("Cortex listens on port 8624 [1].")
    require(check["grounded"], str(check))
    print(f"   {report.files} files · top={Path(result.chunks[0].source).name}")


def smoke_plugin() -> None:
    """Scaffold, load, and dispatch one manifest-declared plugin tool."""
    stage("plugin manifest + tool + hook")
    parent = TEMP / "plugins"
    scaffold(parent, "weather")
    registry = ToolRegistry()
    hooks = HookRegistry()
    loader = PluginLoader([parent], registry=registry, hooks=hooks)
    report = loader.load_all()
    require(len(report.loaded) == 1 and report.tool_count == 1, report.render())
    result = registry.dispatch(
        ToolCall(id="smoke", name="weather_hello", arguments={"name": "Gurugram"}),
        ToolContext(workspace=ROOT),
    )
    require(result.ok and result.content == "hello, Gurugram", str(result.to_dict()))
    require(hooks.count("before_run") == 1, "plugin hook was not registered")
    loader.unload_all()
    print("  ", report.render().splitlines()[0])


def smoke_improvements() -> None:
    """Collect and resolve every current improvement reference."""
    stage("verified improvements")
    entries = collect(ROOT)
    report = verify(entries, ROOT)
    require(report.ok, report.render())
    require(report.total >= 777, f"only {report.total} improvements")
    print(f"   {report.passed}/{report.total} references resolve (target {report.target})")


def main() -> int:
    """Run every stage and return a shell-friendly status."""
    try:
        smoke_agent()
        smoke_cortex()
        smoke_rag()
        smoke_plugin()
        smoke_improvements()
    except Exception as exc:  # noqa: BLE001 - smoke should print one concise failure
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(TEMP, ignore_errors=True)
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
