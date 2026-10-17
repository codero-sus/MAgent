"""Terminal rendering of agent runs: answers, steps, tools, usage, spinners.

One renderer turns the objects an agent produces - :class:`~magent.types.RunResult`,
:class:`~magent.types.StepRecord`, :class:`~magent.types.ToolResult`,
:class:`~magent.types.AgentEvent` - into lines for a human.  It is driven by
:class:`~magent.config.OutputConfig`, so ``--quiet``, ``--no-color``,
``--format json``, ``--show-timing`` and the verbosity levels all change output
without any command having to know about them.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import AgentEvent, EventKind, RunResult, StepRecord, ToolCall, ToolResult
from ..utils.colors import ASCII_SPINNER_FRAMES, SPINNER_FRAMES, colorize, strip_ansi, themed
from ..utils.tables import render_key_value, render_progress, render_spinner_frame
from ..utils.text import ellipsize
from ..utils.timeutil import format_duration
from .format import normalize_format, render_value
from .markdown import looks_like_markdown, render_markdown
from .sink import BufferSink, OutputSink

MAX_TOOL_OUTPUT = 400
"""Characters of tool output shown inline before eliding."""

MAX_THOUGHT_CHARS = 300
"""Characters of a step's reasoning shown in normal verbosity."""

LEVEL_STYLES = {
    "info": "info",
    "success": "success",
    "warning": "warning",
    "error": "error",
    "muted": "muted",
    "tool": "tool",
}
"""Theme name used per message level."""


@dataclass(slots=True)
class RenderOptions:
    """Everything that shapes rendered output."""

    fmt: str = "text"
    color: bool = True
    unicode: bool = True
    width: int = 80
    markdown: bool = True
    show_usage: bool = True
    show_timing: bool = True
    show_tool_calls: bool = True
    stream_to_terminal: bool = True
    spinner: bool = True
    progress: bool = True
    verbosity: str = "normal"
    quiet: bool = False
    max_tool_output: int = MAX_TOOL_OUTPUT
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any = None) -> RenderOptions:
        """Read options from an :class:`~magent.config.OutputConfig`.

        Args:
            config: an output config, a full config, or ``None``
        """
        output = getattr(config, "output", config)
        if output is None:
            return cls()
        return cls(
            fmt=normalize_format(getattr(output, "format", "text")),
            color=bool(getattr(output, "color", True)),
            unicode=bool(getattr(output, "unicode", True)),
            width=int(getattr(output, "width", 0) or 0),
            markdown=bool(getattr(output, "markdown", True)),
            show_usage=bool(getattr(output, "show_usage", True)),
            show_timing=bool(getattr(output, "show_timing", True)),
            show_tool_calls=bool(getattr(output, "show_tool_calls", True)),
            stream_to_terminal=bool(getattr(output, "stream_to_terminal", True)),
            spinner=bool(getattr(output, "spinner", True)),
            progress=bool(getattr(output, "progress", True)),
            verbosity=str(getattr(output, "verbosity", "normal")),
            quiet=bool(getattr(output, "quiet", False)),
        )

    def style(self, text: str, name: str) -> str:
        """Apply a theme colour when color is enabled."""
        return themed(text, name, enabled=self.color)

    @property
    def verbose(self) -> bool:
        """True for ``verbose`` and ``debug`` verbosity."""
        return self.verbosity in {"verbose", "debug"}

    @property
    def debug(self) -> bool:
        """True only for ``debug`` verbosity."""
        return self.verbosity == "debug"

    @property
    def silent(self) -> bool:
        """True when nothing should be printed."""
        return self.quiet or self.fmt == "quiet"


class Spinner:
    """A terminal spinner that keeps working while a run is in flight."""

    def __init__(
        self,
        sink: OutputSink | None = None,
        *,
        message: str = "thinking",
        enabled: bool = True,
        unicode: bool = True,
        color: bool = True,
        interval: float = 0.1,
    ) -> None:
        """Prepare the spinner.

        Args:
            sink: where frames are drawn; defaults to a console sink on stderr
            message: text shown next to the frame
            enabled: allow animation (auto-disabled for non-tty streams)
            unicode: use braille frames instead of ASCII
            color: colour the frame
            interval: seconds between frames
        """
        from .sink import ConsoleSink

        self.sink = sink or ConsoleSink(error=True)
        self.message = message
        self.color = color
        self.interval = max(0.02, interval)
        self.frames = SPINNER_FRAMES if unicode else ASCII_SPINNER_FRAMES
        self.enabled = bool(enabled) and self._is_tty()
        self._index = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._drawn = False

    def _is_tty(self) -> bool:
        """True when the sink writes to an interactive terminal."""
        stream = getattr(self.sink, "stream", None)
        if stream is None:
            stream = sys.stderr
        try:
            return bool(stream.isatty())
        except Exception:
            return False

    def start(self, message: str = "") -> Spinner:
        """Begin animating; returns self so it can be used in a ``with``."""
        if message:
            self.message = message
        if not self.enabled or self._thread is not None:
            return self
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._spin, name="magent-spinner", daemon=True)
        self._thread.start()
        return self

    def update(self, message: str) -> None:
        """Change the spinner label."""
        with self._lock:
            self.message = message

    def _spin(self) -> None:
        """Animation loop, run on a daemon thread."""
        while not self._stop.is_set():
            self._draw()
            self._stop.wait(self.interval)
        self._clear()

    def _draw(self) -> None:
        """Draw one frame."""
        with self._lock:
            frame = render_spinner_frame(self._index, self.frames)
            elapsed = time.monotonic() - self._started_at
            label = f"{frame} {self.message}"
            if elapsed > 2:
                label += f" {format_duration(elapsed)}"
            self._index += 1
        text = colorize(label, "spinner", enabled=self.color) if self.color else label
        try:
            self.sink.write("\r\033[K" + text)
            self._drawn = True
        except Exception:
            self._stop.set()

    def _clear(self) -> None:
        """Erase the current frame."""
        if not self._drawn:
            return
        try:
            self.sink.write("\r\033[K")
        except Exception:
            pass
        self._drawn = False

    def stop(self, final: str = "") -> float:
        """Stop animating and optionally print a final line; returns seconds spun."""
        elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._clear()
        if final:
            self.sink.line(final)
        return elapsed

    @property
    def elapsed(self) -> float:
        """Seconds since the spinner started."""
        return time.monotonic() - self._started_at if self._started_at else 0.0

    def __enter__(self) -> Spinner:
        """Start on entry."""
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        """Stop on exit."""
        self.stop()


class ResultRenderer:
    """Render agent runs for humans or machines."""

    def __init__(
        self,
        options: RenderOptions | None = None,
        *,
        sink: OutputSink | None = None,
        width: int = 0,
    ) -> None:
        """Configure the renderer.

        Args:
            options: render options; defaults to :class:`RenderOptions`
            sink: destination for printed output
            width: column budget, 0 to auto-detect
        """
        self.options = options or RenderOptions()
        self.sink = sink or BufferSink()
        self._width = width

    @classmethod
    def from_config(
        cls, config: Any = None, *, sink: OutputSink | None = None, width: int = 0
    ) -> ResultRenderer:
        """Build a renderer from configuration."""
        return cls(RenderOptions.from_config(config), sink=sink, width=width)

    @property
    def width(self) -> int:
        """Column budget, auto-detected from the sink when unset."""
        if self._width:
            return self._width
        if self.options.width:
            return self.options.width
        sink_width = getattr(self.sink, "width", 0)
        return int(sink_width or 80)

    # -- printing ------------------------------------------------------------

    def print(self, text: str = "") -> None:
        """Write one line unless output is silenced."""
        if self.options.silent:
            return
        self.sink.line(text)

    def write(self, text: str) -> None:
        """Write text without a newline."""
        if self.options.silent:
            return
        self.sink.write(text)

    def notice(self, text: str, *, level: str = "info") -> None:
        """Print a prefixed status line (info/success/warning/error/muted)."""
        style = LEVEL_STYLES.get(level, "muted")
        marker = {
            "info": "•",
            "success": "✓",
            "warning": "!",
            "error": "✗",
            "muted": "·",
            "tool": "→",
        }.get(level, "•")
        if not self.options.unicode:
            marker = {"✓": "+", "✗": "x", "→": "->", "•": "*"}.get(marker, marker)
        self.print(self.options.style(f"{marker} {text}", style))

    # -- results -------------------------------------------------------------

    def render_result(self, result: RunResult) -> str:
        """Render a complete run in the configured format."""
        if self.options.fmt != "text":
            return render_value(result, self.options.fmt, width=self.width)
        parts: list[str] = []
        if self.options.show_tool_calls and self.options.verbose:
            steps = self.render_steps(result)
            if steps:
                parts.append(steps)
        answer = self.render_answer(result.output)
        if answer:
            parts.append(answer)
        if result.error:
            parts.append(self.render_error(result.error))
        tail = self.render_footer(result)
        if tail:
            parts.append(tail)
        return "\n\n".join(part for part in parts if part)

    def render_answer(self, text: str) -> str:
        """Render the model's answer, as Markdown when it looks like Markdown."""
        if not text:
            return ""
        if self.options.markdown and looks_like_markdown(text):
            return render_markdown(
                text,
                color=self.options.color,
                unicode=self.options.unicode,
                width=self.width,
                show_link_urls=self.options.verbose,
            )
        if self.options.color:
            return text
        return strip_ansi(text)

    def render_steps(self, result: RunResult) -> str:
        """Render every step of a run."""
        lines: list[str] = []
        for step in result.steps:
            rendered = self.render_step(step)
            if rendered:
                lines.append(rendered)
        return "\n".join(lines)

    def render_step(self, step: StepRecord) -> str:
        """Render one reasoning step with its tool traffic."""
        lines: list[str] = []
        head = f"step {step.index + 1}"
        if step.model:
            head += f" [{step.model}]"
        if self.options.show_timing and step.duration_ms:
            head += f" {format_duration(step.duration_ms / 1000)}"
        if step.retries:
            head += f" retries={step.retries}"
        if step.cache_hit:
            head += " (cache)"
        lines.append(self.options.style(head, "muted"))
        if step.thought:
            limit = MAX_THOUGHT_CHARS if not self.options.debug else 2_000
            lines.append("  " + self.options.style(ellipsize(step.thought, limit), "system"))
        for call in step.tool_calls:
            lines.append(self.render_tool_call(call))
        for item in step.tool_results:
            lines.append(self.render_tool_result(item))
        return "\n".join(lines)

    def render_tool_call(self, call: ToolCall) -> str:
        """Render one tool invocation."""
        args = ", ".join(
            f"{key}={_short(value)}" for key, value in list(call.arguments.items())[:6]
        )
        more = len(call.arguments) - 6
        if more > 0:
            args += f", +{more}"
        text = f"→ {call.name}({ellipsize(args, 160)})"
        return "  " + self.options.style(text, "tool")

    def render_tool_result(self, result: ToolResult) -> str:
        """Render one tool result, marking failures clearly."""
        if result.ok:
            body = _short(result.content, self.options.max_tool_output)
            text = f"← {result.name}: {body}"
            if self.options.show_timing and result.duration_ms:
                text += f" ({format_duration(result.duration_ms / 1000)})"
            style = "success"
        else:
            text = f"✗ {result.name}: {result.error or 'failed'}"
            if result.error_code:
                text += f" [{result.error_code}]"
            style = "error"
        return "  " + self.options.style(ellipsize(text, self.options.max_tool_output + 60), style)

    def render_usage(self, result: RunResult) -> str:
        """Render the token and cost line for a run."""
        usage = result.usage
        pairs: list[tuple[str, str]] = []
        if usage and not usage.is_empty():
            pairs.append(
                (
                    "tokens",
                    f"{usage.total_tokens} "
                    f"(prompt {usage.prompt_tokens}, "
                    f"completion {usage.completion_tokens})",
                )
            )
            if usage.cached_tokens:
                pairs.append(("cached", f"{usage.cached_tokens} ({result.cache_hit_rate():.0%})"))
            if usage.reasoning_tokens:
                pairs.append(("reasoning", str(usage.reasoning_tokens)))
        if result.cost and result.cost.total > 0:
            pairs.append(("cost", f"{result.cost.total:.4f} {result.cost.currency}"))
        if self.options.show_timing:
            pairs.append(("time", format_duration(result.duration_ms / 1000)))
            if result.tokens_per_second():
                pairs.append(("speed", f"{result.tokens_per_second():.1f} tok/s"))
        pairs.append(("steps", str(result.iterations)))
        if result.tool_call_count:
            pairs.append(("tools", f"{result.tool_call_count}"))
            if result.tool_error_count:
                pairs.append(("tool errors", str(result.tool_error_count)))
        if not pairs:
            return ""
        return render_key_value(pairs, key_width=12)

    def render_footer(self, result: RunResult) -> str:
        """Render the trailing status line plus usage."""
        lines: list[str] = []
        reason = str(getattr(result.stop_reason, "value", result.stop_reason))
        if result.succeeded:
            head = self.options.style(f"done ({reason})", "success")
        else:
            head = self.options.style(f"stopped: {reason}", "warning")
        lines.append(head)
        if self.options.show_usage:
            usage = self.render_usage(result)
            if usage:
                lines.append(usage)
        return "\n".join(lines)

    def render_error(self, error: Any) -> str:
        """Render an error message or exception."""
        text = str(getattr(error, "message", error) or error)
        details = getattr(error, "details", None)
        if details:
            text += f" ({details})"
        return self.options.style(f"✗ {text}", "error")

    def render_event(self, event: AgentEvent) -> str:
        """Render one streaming event as a single line (empty to skip)."""
        kind = event.kind
        data = event.data or {}
        if kind == EventKind.TEXT_DELTA:
            return ""
        if kind in {
            EventKind.STEP_STARTED,
        }:
            return self.options.style(f"· step {int(data.get('step', 0)) + 1}", "muted")
        if kind == EventKind.TOOL_CALL:
            call = data.get("call")
            if isinstance(call, ToolCall):
                return self.render_tool_call(call)
            return self.options.style(f"→ {data.get('name', 'tool')}(...)", "tool")
        if kind == EventKind.TOOL_RESULT:
            item = data.get("result")
            if isinstance(item, ToolResult):
                return self.render_tool_result(item)
            return self.options.style(f"← {data.get('name', 'tool')}", "success")
        if kind == EventKind.TOOL_ERROR:
            return self.render_error(data.get("error") or data.get("message") or "tool failed")
        if kind == EventKind.APPROVAL_REQUESTED:
            return self.options.style(
                f"? approval needed for {data.get('name', 'tool')}", "warning"
            )
        if kind == EventKind.GUARDRAIL_BLOCKED:
            return self.options.style(f"! guardrail: {data.get('reason', 'blocked')}", "warning")
        if kind == EventKind.PLAN_CREATED:
            steps = data.get("steps") or []
            return self.options.style(f"plan: {len(steps)} steps", "info")
        if kind == EventKind.REFLECTION:
            return self.options.style(
                f"reflection: {ellipsize(str(data.get('note', '')), 120)}", "accent"
            )
        if kind == EventKind.LLM_RETRY:
            return self.options.style(
                f"retry {data.get('attempt', '?')}: {ellipsize(str(data.get('error', '')), 100)}",
                "warning",
            )
        if kind == EventKind.MODEL_SWITCHED:
            return self.options.style(f"model -> {data.get('model', '?')}", "info")
        if kind in {EventKind.RUN_FINISHED, EventKind.RUN_FAILED}:
            return self.options.style(
                f"run {'failed' if kind == EventKind.RUN_FAILED else 'finished'}",
                "error" if kind == EventKind.RUN_FAILED else "success",
            )
        if self.options.debug:
            return self.options.style(f"· {getattr(kind, 'value', kind)} {data}", "muted")
        return ""

    def print_event(self, event: AgentEvent) -> None:
        """Render and print one event, honouring verbosity."""
        line = self.render_event(event)
        if line:
            self.print(line)

    def print_result(self, result: RunResult) -> None:
        """Render and print a complete run."""
        text = self.render_result(result)
        if text:
            self.print(text)

    def stream(
        self, events: Iterable[AgentEvent], *, show_text: bool = True
    ) -> Iterator[AgentEvent]:
        """Pass events through, printing deltas and notable lines as they arrive."""
        for event in events:
            if show_text and self.options.stream_to_terminal and event.kind == EventKind.TEXT_DELTA:
                self.write(str(event.data.get("text", "")))
            else:
                self.print_event(event)
            yield event
        if show_text and self.options.stream_to_terminal:
            self.write("\n")

    def spinner(self, message: str = "thinking") -> Spinner:
        """Create a spinner bound to this renderer's settings."""
        return Spinner(
            self.sink,
            message=message,
            enabled=self.options.spinner and not self.options.silent,
            unicode=self.options.unicode,
            color=self.options.color,
        )

    def progress(
        self, items: Sequence[Any], *, total: int = 0, label: str = "", show: bool | None = None
    ) -> Iterator[Any]:
        """Iterate items while drawing a progress bar."""
        enabled = self.options.progress if show is None else show
        count = total or len(items)
        if not enabled or not count or self.options.silent:
            yield from items
            return
        for index, item in enumerate(items, start=1):
            yield item
            bar = render_progress(
                index,
                count,
                width=min(30, max(10, self.width // 3)),
                label=label,
                color=self.options.color,
            )
            self.sink.write("\r\033[K" + bar)
        self.sink.write("\n")

    def key_value(self, pairs: Iterable[tuple[str, Any]], *, title: str = "") -> str:
        """Render aligned key/value output."""
        rendered = [(str(key), _short(value, 200)) for key, value in pairs]
        if title:
            return self.options.style(title, "heading") + "\n" + render_key_value(rendered)
        return render_key_value(rendered)

    def table(
        self, rows: Sequence[Sequence[Any]], headers: Sequence[str], *, title: str = ""
    ) -> str:
        """Render a bordered table."""
        from ..utils.tables import render_table

        body = render_table(
            [[_short(cell, 60) for cell in row] for row in rows],
            headers=list(headers),
            unicode_borders=self.options.unicode,
            color=self.options.color,
        )
        if title:
            return self.options.style(title, "heading") + "\n" + body
        return body


def _short(value: Any, limit: int = 120) -> str:
    """Flatten a value to one short line."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return ellipsize(str(value).replace("\n", " "), limit)
    if isinstance(value, dict):
        import json

        try:
            return ellipsize(json.dumps(value, default=str, ensure_ascii=False), limit)
        except (TypeError, ValueError):
            return ellipsize(str(value), limit)
    if isinstance(value, (list, tuple)):
        return ellipsize(", ".join(_short(item, 40) for item in value), limit)
    return ellipsize(str(value).replace("\n", " "), limit)


def render_run(result: RunResult, *, config: Any = None, sink: OutputSink | None = None) -> str:
    """Convenience one-shot renderer used by CLI commands."""
    renderer = ResultRenderer.from_config(config, sink=sink)
    return renderer.render_result(result)


def print_run(result: RunResult, *, config: Any = None, sink: OutputSink | None = None) -> None:
    """Render a run and write it to the sink (stdout by default)."""
    from .sink import ConsoleSink

    renderer = ResultRenderer.from_config(config, sink=sink or ConsoleSink())
    renderer.print_result(result)


__all__ = [
    "LEVEL_STYLES",
    "MAX_THOUGHT_CHARS",
    "MAX_TOOL_OUTPUT",
    "RenderOptions",
    "ResultRenderer",
    "Spinner",
    "print_run",
    "render_run",
]
