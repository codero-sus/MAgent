"""Context assembly: what the model sees besides the conversation.

Public API:

:class:`ContextBlock`
    one titled, prioritised, token-counted slice of context;
:class:`ContextBuilder` / :func:`build_context`
    collect blocks from the workspace, retrieval, memory and the session, then
    compress them into a token budget;
:class:`ContextCompressor` / :func:`compress_text`
    the compression machinery, usable on its own for tool output or transcripts;
:func:`scan_workspace` / :func:`file_context_blocks`
    workspace scanning that respects the tool sandbox.

>>> from magent.context import ContextBuilder, ContextBlock
>>> builder = ContextBuilder(budget=400)
>>> _ = builder.add(ContextBlock(title="Task", body="Fix the parser", kind="task", priority=9))
>>> assembly = builder.build()
>>> assembly.tokens > 0 and "Fix the parser" in assembly.text
True
"""

from __future__ import annotations

from .block import (
    BLOCK_KINDS,
    DEFAULT_PRIORITY,
    MAX_PRIORITY,
    ContextBlock,
    block,
    custom_block,
    environment_block,
    files_block,
    memory_block,
    rag_block,
    render_blocks,
    session_block,
    sort_blocks,
    task_block,
    total_tokens,
)
from .builder import (
    DEFAULT_BUDGET,
    RESERVE_FOR_PROMPT,
    ContextAssembly,
    ContextBuilder,
    build_context,
)
from .compression import (
    MAX_LINE_REPEATS,
    NOISE_PATTERNS,
    CompressionReport,
    ContextCompressor,
    compress_messages,
    compress_text,
    dedupe_lines,
    strip_noise,
    summarize_with,
)
from .files import (
    CONVENTION_FILES,
    DEFAULT_PATTERNS,
    MAX_FILE_CHARS,
    MAX_FILES_SCANNED,
    SKIP_DIRS,
    FileRef,
    WorkspaceSnapshot,
    describe_workspace,
    file_context_blocks,
    files_listing_block,
    open_editor_files,
    rank_files,
    read_file_ref,
    scan_workspace,
    touched_recently,
)

__all__ = [
    "BLOCK_KINDS",
    "CONVENTION_FILES",
    "DEFAULT_BUDGET",
    "DEFAULT_PATTERNS",
    "DEFAULT_PRIORITY",
    "MAX_FILES_SCANNED",
    "MAX_FILE_CHARS",
    "MAX_LINE_REPEATS",
    "MAX_PRIORITY",
    "NOISE_PATTERNS",
    "RESERVE_FOR_PROMPT",
    "SKIP_DIRS",
    "CompressionReport",
    "ContextAssembly",
    "ContextBlock",
    "ContextBuilder",
    "ContextCompressor",
    "FileRef",
    "WorkspaceSnapshot",
    "block",
    "build_context",
    "compress_messages",
    "compress_text",
    "custom_block",
    "dedupe_lines",
    "describe_workspace",
    "environment_block",
    "file_context_blocks",
    "files_block",
    "files_listing_block",
    "memory_block",
    "open_editor_files",
    "rag_block",
    "rank_files",
    "read_file_ref",
    "render_blocks",
    "scan_workspace",
    "session_block",
    "sort_blocks",
    "strip_noise",
    "summarize_with",
    "task_block",
    "total_tokens",
    "touched_recently",
]
