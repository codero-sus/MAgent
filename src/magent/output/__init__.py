"""Output rendering: results to text, JSON, or Markdown, plus live feedback.

Two halves that share one set of options:

- :mod:`magent.output.renderer` formats finished work - a :class:`RunResult`,
  tool calls, usage, errors, statistics.
- :mod:`magent.output.terminal` shows work in progress - a :class:`Printer`,
  a :class:`Spinner`, a :class:`ProgressBar`.

Build options once from the resolved config and pass them everywhere, so a
piped JSON run and an interactive terminal run cannot disagree about colour,
verbosity, or whether animation is allowed::

    opts = OutputOptions.from_config(settings)
    printer = Printer(opts)
    with Spinner("thinking", printer=printer):
        result = agent.run(task)
    printer.write(render_result(result, options=opts))
"""

from __future__ import annotations

from .renderer import (
    DEFAULT_WIDTH,
    FORMATS,
    VERBOSITY_LEVELS,
    OutputOptions,
    plain,
    render_error,
    render_messages,
    render_progress,
    render_result,
    render_result_json,
    render_result_markdown,
    render_result_text,
    render_session_info,
    render_stats,
    render_tool_call,
    render_tool_result,
    render_usage,
)
from .terminal import (
    DEFAULT_SPINNER_INTERVAL,
    Printer,
    ProgressBar,
    Spinner,
    is_interactive,
    progress,
    spinner,
)

__all__ = [
    "DEFAULT_SPINNER_INTERVAL",
    "DEFAULT_WIDTH",
    "FORMATS",
    "VERBOSITY_LEVELS",
    "OutputOptions",
    "Printer",
    "ProgressBar",
    "Spinner",
    "is_interactive",
    "plain",
    "progress",
    "render_error",
    "render_messages",
    "render_progress",
    "render_result",
    "render_result_json",
    "render_result_markdown",
    "render_result_text",
    "render_session_info",
    "render_stats",
    "render_tool_call",
    "render_tool_result",
    "render_usage",
    "spinner",
]
