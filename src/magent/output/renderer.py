"""Turning run results into something a human or a pipe can consume.

Three formats share one code path so they cannot drift apart: ``text`` for
terminals, ``json`` for scripts, ``markdown`` for reports and issue bodies.
Everything honours :class:`~magent.config.OutputConfig`, including ``quiet``
and the ``verbosity`` ladder, so a caller never has to hand-trim output before
piping it somewhere.

Colour is decided once, from the config and the terminal's actual capability,
and every renderer receives that decision rather than probing ``isatty``
itself - which is what keeps output identical when it is redirected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..types import Message, RunResult, StepRecord
from ..utils.colors import strip_ansi, supports_color, supports_unicode, terminal_size, themed
from ..utils.tables import (
    pluralize,
    render_key_value,
    render_markdown_table,
    render_table,
)
from ..utils.tables import render_progress as _progress_bar
from ..utils.timeutil import format_duration, format_timestamp

FORMATS: tuple[str, ...] = ("text", "json", "markdown")
"""Output formats every renderer understands."""

VERBOSITY_LEVELS: dict[str, int] = {"quiet": 0, "minimal": 1, "normal": 2, "verbose": 3, "debug": 4}
"""Numeric rank of each verbosity name, for threshold comparisons."""

DEFAULT_WIDTH = 100
"""Column width used when the terminal does not report one."""


@dataclass(slots=True)
class OutputOptions:
    """Resolved rendering choices, independent of the config object."""

    fmt: str = "text"
    color: bool = False
    unicode: bool = True
    width: int = DEFAULT_WIDTH
    show_usage: bool = True
    show_timing: bool = True
    show_tool_calls: bool = True
    markdown: bool = True
    quiet: bool = False
    verbosity: str = "normal"
    stream_to_terminal: bool = True
    spinner: bool = True
    progress: bool = True

    def __post_init__(self) -> None:
        """Normalize values and resolve ``auto`` settings against the terminal."""
        if self.fmt not in FORMATS:
            self.fmt = "text"
        if self.verbosity not in VERBOSITY_LEVELS:
            self.verbosity = "normal"
        if self.quiet:
            self.verbosity = "quiet"
        if not self.width or self.width <= 0:
            self.width = terminal_size((DEFAULT_WIDTH, 24))[0]
        self.width = max(40, min(400, int(self.width)))
        self.unicode = bool(self.unicode) and supports_unicode()
        self.color = bool(self.color) and supports_color()

    @property
    def level(self) -> int:
        """Numeric verbosity rank."""
        return VERBOSITY_LEVELS.get(self.verbosity, 2)

    def at_least(self, name: str) -> bool:
        """Return whether the current verbosity reaches ``name``."""
        return self.level >= VERBOSITY_LEVELS.get(name, 2)

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> OutputOptions:
        """Build options from a resolved config's ``output`` section.

        Args:
            config: configuration object, may be ``None`` for defaults
            overrides: explicit values that win over the config
        """
        section = getattr(config, "output", None)
        values: dict[str, Any] = {}
        if section is not None:
            for name in ("format", "color", "unicode", "width", "show_usage", "show_timing",
                         "show_tool_calls", "stream_to_terminal", "markdown", "quiet",
                         "verbosity", "spinner", "progress"):
                if hasattr(section, name):
                    values[name] = getattr(section, name)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def themed(self, text: str, name: str) -> str:
        """Apply a semantic colour using these options' colour decision."""
        return themed(text, name, enabled=self.color)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for diagnostics."""
        return {
            "format": self.fmt, "color": self.color, "unicode": self.unicode,
            "width": self.width, "show_usage": self.show_usage,
            "show_timing": self.show_timing, "show_tool_calls": self.show_tool_calls,
            "markdown": self.markdown, "quiet": self.quiet, "verbosity": self.verbosity,
            "spinner": self.spinner, "progress": self.progress, "level": self.level,
        }


def render_result(result: RunResult, *, options: OutputOptions | None = None,
                  config: Any = None) -> str:
    """Render a finished run in the configured format.

    Args:
        result: the run to render
        options: explicit options; built from ``config`` when omitted
        config: configuration used to derive options
    """
    active = options or OutputOptions.from_config(config)
    if active.fmt == "json":
        return render_result_json(result, options=active)
    if active.fmt == "markdown":
        return render_result_markdown(result, options=active)
    return render_result_text(result, options=active)


def render_result_text(result: RunResult, *, options: OutputOptions | None = None) -> str:
    """Render a run for a terminal, honouring the verbosity ladder."""
    opts = options or OutputOptions()
    if opts.quiet:
        return result.output or ""
    lines: list[str] = []
    if result.output:
        lines.append(result.output.rstrip())
    if opts.at_least("normal") and (opts.show_timing or opts.show_usage):
        lines.append("")
        lines.append(_stats_line(result, opts))
    if opts.at_least("verbose") and result.steps:
        lines.append("")
        lines.append(_steps_text(result.steps, opts))
    if opts.at_least("debug") and result.messages:
        lines.append("")
        lines.append(render_messages(result.messages, options=opts))
    return "\n".join(lines).strip() + "\n"


def render_result_json(result: RunResult, *, options: OutputOptions | None = None,
                       indent: int = 2) -> str:
    """Render a run as stable JSON for scripts and machines."""
    opts = options or OutputOptions()
    payload: dict[str, Any] = {
        "run_id": result.run_id,
        "output": result.output,
        "stop_reason": str(getattr(result.stop_reason, "value", result.stop_reason)),
        "model": result.model,
        "iterations": result.iterations,
        "tool_calls": result.tool_call_count,
        "duration_ms": round(result.duration_ms, 2),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
    }
    usage = getattr(result, "usage", None)
    if usage is not None and opts.show_usage:
        payload["usage"] = usage.to_dict() if hasattr(usage, "to_dict") else dict(usage)
    cost = getattr(result, "cost", None)
    if cost is not None and opts.show_usage:
        payload["cost"] = cost.to_dict() if hasattr(cost, "to_dict") else dict(cost)
    if result.structured is not None:
        payload["structured"] = result.structured
    if result.error:
        payload["error"] = result.error
    if opts.at_least("verbose"):
        payload["steps"] = [_step_dict(step) for step in result.steps]
    if opts.at_least("debug"):
        payload["messages"] = [message.to_dict() for message in result.messages]
    if result.metadata:
        payload["metadata"] = result.metadata
    return json.dumps(payload, indent=indent or None, ensure_ascii=False, default=str)


def render_result_markdown(result: RunResult, *, options: OutputOptions | None = None) -> str:
    """Render a run as a Markdown report."""
    opts = options or OutputOptions()
    lines = ["## Run summary", ""]
    pairs: list[tuple[str, Any]] = [
        ("Run", f"`{result.run_id}`"),
        ("Model", f"`{result.model or 'default'}`"),
        ("Stop reason", str(getattr(result.stop_reason, "value", result.stop_reason))),
        ("Iterations", result.iterations),
        ("Tool calls", result.tool_call_count),
    ]
    if opts.show_timing:
        pairs.append(("Duration", format_duration(result.duration_ms / 1000.0)))
    if opts.show_usage:
        usage = getattr(result, "usage", None)
        if usage is not None:
            pairs.append(("Tokens", _usage_summary(usage)))
        cost = getattr(result, "cost", None)
        if cost is not None and getattr(cost, "total", 0.0):
            pairs.append(("Cost", f"${cost.total:.4f}"))
    lines.append(render_key_value(pairs))
    lines.append("")
    lines.append("### Answer")
    lines.append("")
    lines.append(result.output.rstrip() or "_(no output)_")
    if opts.at_least("verbose") and result.steps:
        lines.append("")
        lines.append("### Steps")
        lines.append("")
        lines.append(_steps_markdown(result.steps, opts))
    if result.error:
        lines.append("")
        lines.append(f"> **Error:** {result.error}")
    return "\n".join(lines).strip() + "\n"


def render_messages(messages: Sequence[Message], *, options: OutputOptions | None = None,
                    limit: int = 0) -> str:
    """Render a message list as a readable transcript."""
    opts = options or OutputOptions()
    items = list(messages)[-limit:] if limit else list(messages)
    if opts.fmt == "json":
        return json.dumps([message.to_dict() for message in items], indent=2,
                          ensure_ascii=False, default=str)
    lines: list[str] = []
    for message in items:
        role = str(getattr(message.role, "value", message.role))
        label = {"user": "You", "assistant": "MAgent", "system": "System", "tool": "Tool"}.get(role, role)
        heading = opts.themed(label, role if role in {"user", "assistant", "system"} else "tool")
        body = (message.text() if callable(getattr(message, "text", None))
                else str(message.content or "")).strip()
        lines.append(f"{heading}: {body}" if body else f"{heading}:")
        for call in message.tool_calls or []:
            payload = call.to_dict() if hasattr(call, "to_dict") else dict(call)
            function = payload.get("function", payload)
            lines.append(f"  -> {function.get('name', 'tool')} "
                         f"{_compact(function.get('arguments', ''), opts.width - 6)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def render_tool_call(name: str, arguments: Any, *, options: OutputOptions | None = None) -> str:
    """Render one tool invocation as a single compact line."""
    opts = options or OutputOptions()
    label = opts.themed(name, "tool")
    return f"{label} {_compact(arguments, max(20, opts.width - len(name) - 4))}"


def render_tool_result(name: str, result: Any, *, options: OutputOptions | None = None,
                       max_chars: int = 600) -> str:
    """Render a tool result, marking failures and truncating long output."""
    opts = options or OutputOptions()
    content = getattr(result, "content", result)
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    failed = not bool(getattr(result, "ok", True))
    code = getattr(result, "error_code", "") or ""
    icon = opts.themed("x" if failed else "+", "error" if failed else "success")
    label = opts.themed(name, "tool")
    duration = getattr(result, "duration_ms", 0.0) or 0.0
    timing = f" {opts.themed(format_duration(duration / 1000.0), 'muted')}" \
        if opts.show_timing and duration else ""
    suffix = f" [{code}]" if failed and code else ""
    body = " ".join(content.split())
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "…"
    return f"{icon} {label}{timing}{suffix} {body}".rstrip()


def render_usage(result: RunResult, *, options: OutputOptions | None = None) -> str:
    """Render just the usage and cost line for a finished run."""
    opts = options or OutputOptions()
    return _stats_line(result, opts)


def render_error(error: Any, *, options: OutputOptions | None = None) -> str:
    """Render an exception for display, keeping type and detail on one line."""
    opts = options or OutputOptions()
    name = type(error).__name__
    message = str(error) or "(no message)"
    details = getattr(error, "details", None)
    line = f"{opts.themed(name, 'error')}: {message}"
    if details and opts.at_least("verbose"):
        line += "\n" + render_key_value(
            [(str(key), value) for key, value in dict(details).items()], key_width=12
        )
    return line


def render_progress(value: float, total: float, *, label: str = "", width: int = 0,
                    options: OutputOptions | None = None) -> str:
    """Render a progress bar respecting the colour and width options.

    Args:
        value: completed units
        total: units that make up 100%
        label: text placed before the bar
        width: bar width, derived from the terminal when 0
        options: resolved output options
    """
    opts = options or OutputOptions()
    active = width or max(10, min(40, opts.width // 3))
    return _progress_bar(value, total, width=active, label=label, color=opts.color)


# -- internals --------------------------------------------------------------- #

def _stats_line(result: RunResult, opts: OutputOptions) -> str:
    """Build the muted statistics line shown under an answer."""
    parts: list[str] = []
    if opts.show_timing:
        parts.append(format_duration(result.duration_ms / 1000.0))
    parts.append(pluralize(result.iterations, "step"))
    if result.tool_call_count:
        parts.append(pluralize(result.tool_call_count, "tool call"))
    if opts.show_usage:
        usage = getattr(result, "usage", None)
        if usage is not None:
            parts.append(_usage_summary(usage))
        cost = getattr(result, "cost", None)
        if cost is not None and getattr(cost, "total", 0.0):
            parts.append(f"${cost.total:.4f}")
    reason = str(getattr(result.stop_reason, "value", result.stop_reason))
    if reason not in {"completed", "stop"}:
        parts.append(f"stop: {reason}")
    return opts.themed(" · ".join(parts), "muted")


def _usage_summary(usage: Any) -> str:
    """Render token counts compactly."""
    total = getattr(usage, "total_tokens", 0) or 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    if not total and not prompt and not completion:
        return "0 tokens"
    return f"{total} tokens ({prompt}+{completion})"


def _step_dict(step: StepRecord) -> dict[str, Any]:
    """Serialize one step for JSON output."""
    payload: dict[str, Any] = {"index": getattr(step, "index", 0)}
    for name in ("model", "stop_reason", "error"):
        value = getattr(step, name, None)
        if value:
            payload[name] = str(getattr(value, "value", value))
    for name in ("duration_ms",):
        value = getattr(step, name, 0.0)
        if value:
            payload[name] = round(float(value), 2)
    usage = getattr(step, "usage", None)
    if usage is not None:
        payload["usage"] = usage.to_dict() if hasattr(usage, "to_dict") else dict(usage)
    calls = getattr(step, "tool_calls", None) or []
    if calls:
        payload["tool_calls"] = [
            {
                "name": getattr(call, "name", "") or (call.get("function", {}) or {}).get("name", ""),
                "arguments": getattr(call, "arguments", None)
                if not isinstance(call, dict) else (call.get("function", {}) or {}).get("arguments"),
            }
            for call in calls
        ]
    results = getattr(step, "tool_results", None) or []
    if results:
        payload["tool_results"] = [
            {
                "name": getattr(item, "name", ""),
                "ok": bool(getattr(item, "ok", True)),
                "duration_ms": round(float(getattr(item, "duration_ms", 0.0) or 0.0), 2),
                "chars": len(str(getattr(item, "content", "") or "")),
            }
            for item in results
        ]
    text = getattr(step, "text", "") or ""
    if text:
        payload["text"] = text
    return payload


def _steps_text(steps: Sequence[StepRecord], opts: OutputOptions) -> str:
    """Render steps as an aligned table for verbose terminal output."""
    rows: list[list[Any]] = []
    for step in steps:
        calls = getattr(step, "tool_calls", None) or []
        names = ", ".join(
            getattr(call, "name", "") or (call.get("function", {}) or {}).get("name", "")
            for call in calls
        ) or "-"
        failed = sum(1 for item in (getattr(step, "tool_results", None) or [])
                     if not getattr(item, "ok", True))
        rows.append([
            getattr(step, "index", len(rows)),
            names[: max(8, opts.width // 4)],
            f"{len(calls)}" + (f" ({failed} failed)" if failed else ""),
            format_duration(float(getattr(step, "duration_ms", 0.0) or 0.0) / 1000.0),
        ])
    return render_table(rows, headers=["#", "tools", "calls", "time"],
                        alignments=["right", "left", "left", "right"],
                        unicode_borders=opts.unicode)


def _steps_markdown(steps: Sequence[StepRecord], opts: OutputOptions) -> str:
    """Render steps as a Markdown table."""
    rows: list[list[Any]] = []
    for step in steps:
        calls = getattr(step, "tool_calls", None) or []
        names = ", ".join(
            getattr(call, "name", "") or (call.get("function", {}) or {}).get("name", "")
            for call in calls
        ) or "-"
        rows.append([
            getattr(step, "index", len(rows)),
            f"`{names}`",
            len(calls),
            format_duration(float(getattr(step, "duration_ms", 0.0) or 0.0) / 1000.0),
        ])
    return render_markdown_table(rows, headers=["#", "tools", "calls", "time"])


def _compact(value: Any, width: int) -> str:
    """Flatten a value to one line no wider than ``width``."""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    flat = " ".join(value.split())
    limit = max(12, int(width))
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def render_session_info(infos: Iterable[Any], *, options: OutputOptions | None = None) -> str:
    """Render a list of :class:`~magent.session.SessionInfo` records."""
    opts = options or OutputOptions()
    items = list(infos)
    if not items:
        return "no sessions stored"
    if opts.fmt == "json":
        return json.dumps([info.to_dict() for info in items], indent=2, ensure_ascii=False,
                          default=str)
    rows = [[
        str(info.session_id)[:12],
        format_timestamp(info.updated_at or info.created_at, fmt="%Y-%m-%d %H:%M"),
        info.entries,
        f"{info.bytes / 1024:.1f}K",
        (info.title or "")[: max(12, opts.width - 46)],
    ] for info in items]
    if opts.fmt == "markdown":
        return render_markdown_table(rows, headers=["id", "updated", "turns", "size", "title"])
    return render_table(rows, headers=["id", "updated", "turns", "size", "title"],
                        alignments=["left", "left", "right", "right", "left"],
                        unicode_borders=opts.unicode)


def render_stats(stats: dict[str, Any], *, options: OutputOptions | None = None,
                 title: str = "Statistics") -> str:
    """Render an arbitrary statistics mapping in the chosen format."""
    opts = options or OutputOptions()
    if opts.fmt == "json":
        return json.dumps(stats, indent=2, ensure_ascii=False, default=str)
    pairs = [(str(key), value) for key, value in stats.items()]
    if opts.fmt == "markdown":
        return f"### {title}\n\n" + render_markdown_table(
            [[key, _compact(value, 60)] for key, value in pairs], headers=["key", "value"]
        )
    return f"{opts.themed(title, 'heading')}\n" + render_key_value(pairs)


def plain(text: str) -> str:
    """Strip ANSI escapes, for logs, files, and diffing."""
    return strip_ansi(text or "")


__all__ = [
    "DEFAULT_WIDTH",
    "FORMATS",
    "VERBOSITY_LEVELS",
    "OutputOptions",
    "plain",
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
]
