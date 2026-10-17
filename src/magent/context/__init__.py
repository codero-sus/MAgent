"""Context window management: budgeting, assembly, and compression.

The public surface is small on purpose:

- :class:`ContextPolicy` says how a window is divided.
- :class:`ContextBuilder` collects sections and fits them, plus history.
- :func:`compress_messages` shrinks a conversation by trimming, truncating,
  windowing, or summarizing.
- :func:`extractive_summary` summarizes without a model, so compression works
  (and can be tested) offline.

Typical use inside an agent loop::

    builder = ContextBuilder(config=settings)
    builder.set("system", system_prompt)
    builder.set("retrieved", rag_context)
    builder.set("memory", memory_block)
    build = builder.build(history=messages)
    response = backend.complete(CompletionRequest(messages=build.messages))
"""

from __future__ import annotations

from .builder import ContextBuild, ContextBuilder, ContextSection, build_context
from .compressor import (
    MIN_TRUNCATE_TOKENS,
    STOPWORDS,
    SUMMARY_PREFIX,
    CompressionResult,
    ModelSummarizer,
    Summarizer,
    compress_messages,
    extractive_summary,
    summarize_messages,
)
from .policy import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_OUTPUT_RESERVE,
    DEFAULT_SHARES,
    REQUIRED_SECTIONS,
    SECTION_NAMES,
    STRATEGIES,
    ContextPolicy,
    SectionPolicy,
    default_sections,
    policy_from_config,
)

__all__ = [
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_OUTPUT_RESERVE",
    "DEFAULT_SHARES",
    "MIN_TRUNCATE_TOKENS",
    "REQUIRED_SECTIONS",
    "SECTION_NAMES",
    "STOPWORDS",
    "STRATEGIES",
    "SUMMARY_PREFIX",
    "CompressionResult",
    "ContextBuild",
    "ContextBuilder",
    "ContextPolicy",
    "ContextSection",
    "ModelSummarizer",
    "SectionPolicy",
    "Summarizer",
    "build_context",
    "compress_messages",
    "default_sections",
    "extractive_summary",
    "policy_from_config",
    "summarize_messages",
]
