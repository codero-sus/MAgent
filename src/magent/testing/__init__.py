"""Test doubles and offline fixtures for MAgent.

Everything here is importable from production code on purpose: ``magent demo``,
``magent doctor --offline`` and the evaluation harness all reuse the same fake
Cortex hoster that the test suite runs against, so there is exactly one
implementation of the fake and it is exercised by real usage.
"""

from __future__ import annotations

from ..llm.echo import EchoBackend, LambdaBackend, ScriptedBackend, ToolScriptBackend
from .fake_cortex import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FakeCortexServer,
    FakeCortexState,
    FakeModel,
    fake_embedding,
    start_fake_cortex,
)
from .fixtures import (
    SAMPLE_CODE,
    SAMPLE_DOCUMENTS,
    SampleCorpus,
    build_sample_corpus,
    make_agent,
    make_backend,
    make_request,
    offline_backend,
    sample_messages,
    scripted_tool_agent,
    temp_workspace,
)

__all__ = [
    "DEFAULT_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "SAMPLE_CODE",
    "SAMPLE_DOCUMENTS",
    "EchoBackend",
    "FakeCortexServer",
    "FakeCortexState",
    "FakeModel",
    "LambdaBackend",
    "SampleCorpus",
    "ScriptedBackend",
    "ToolScriptBackend",
    "build_sample_corpus",
    "fake_embedding",
    "make_agent",
    "make_backend",
    "make_request",
    "offline_backend",
    "sample_messages",
    "scripted_tool_agent",
    "start_fake_cortex",
    "temp_workspace",
]
