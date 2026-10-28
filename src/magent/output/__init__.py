"""Rendering agent output for humans and machines.

Public API:

:class:`ResultRenderer` / :class:`RenderOptions`
    turn a :class:`~magent.types.RunResult` into terminal text, JSON, YAML, CSV
    or Markdown, honouring ``OutputConfig``;
:class:`Spinner`
    animated progress for long runs, auto-disabled on non-tty streams;
:func:`render_value` / :func:`to_yaml` / :func:`to_csv` / :func:`to_markdown`
    format-agnostic payload rendering, including a dependency-free YAML emitter;
:func:`render_markdown`
    Markdown to ANSI conversion for model answers;
:class:`OutputSink` and friends
    where text goes: console, buffer, file, tee, timestamp, null.

>>> from magent.output import render_value, render_markdown, to_yaml
>>> render_value({"a": 1}, "yaml")
'a: 1'
>>> "b:" in to_yaml({"a": 1, "b": [1, 2]})
True
>>> "bold" in render_markdown("**bold**", color=False)
True
"""

from __future__ import annotations

from .format import (
    DEFAULT_FORMAT,
    FORMATS,
    YAML_INDENT,
    dump_formats,
    iter_records,
    normalize_format,
    parse_format_choice,
    render_jsonl,
    render_value,
    to_csv,
    to_markdown,
    to_plain,
    to_text,
    to_yaml,
    write_formatted,
)
from .markdown import (
    MarkdownOptions,
    bullet_list,
    extract_code_blocks,
    inline,
    looks_like_markdown,
    plain_text,
    render_code_block,
    render_diff,
    render_heading,
    render_list,
    render_markdown,
    render_table,
    summarize_markdown,
)
from .render import (
    LEVEL_STYLES,
    MAX_THOUGHT_CHARS,
    MAX_TOOL_OUTPUT,
    RenderOptions,
    ResultRenderer,
    Spinner,
    print_run,
    render_run,
)
from .sink import (
    BufferSink,
    ConsoleSink,
    FileSink,
    NullSink,
    OutputSink,
    TeeSink,
    TimestampSink,
    build_sink,
    capture_lines,
    redirect_stdout,
    write_capture,
)

__all__ = [
    "DEFAULT_FORMAT",
    "FORMATS",
    "LEVEL_STYLES",
    "MAX_THOUGHT_CHARS",
    "MAX_TOOL_OUTPUT",
    "YAML_INDENT",
    "BufferSink",
    "ConsoleSink",
    "FileSink",
    "MarkdownOptions",
    "NullSink",
    "OutputSink",
    "RenderOptions",
    "ResultRenderer",
    "Spinner",
    "TeeSink",
    "TimestampSink",
    "build_sink",
    "bullet_list",
    "capture_lines",
    "dump_formats",
    "extract_code_blocks",
    "inline",
    "iter_records",
    "looks_like_markdown",
    "normalize_format",
    "parse_format_choice",
    "plain_text",
    "print_run",
    "redirect_stdout",
    "render_code_block",
    "render_diff",
    "render_heading",
    "render_jsonl",
    "render_list",
    "render_markdown",
    "render_run",
    "render_table",
    "render_value",
    "summarize_markdown",
    "to_csv",
    "to_markdown",
    "to_plain",
    "to_text",
    "to_yaml",
    "write_capture",
    "write_formatted",
]
