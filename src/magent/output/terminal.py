"""Live terminal feedback: a spinner, a progress bar, and a config-aware printer.

Everything here is optional and degrades to silence. When output is not a
terminal, when ``quiet`` is set, or when the format is JSON, the spinner and
progress bar draw nothing - a stray animation frame in a JSON pipe breaks
whatever is parsing it, which is a worse failure than a missing spinner.

The printer is the single place that decides whether something should appear,
so the rest of the CLI never re-implements that logic and never disagrees with
itself about verbosity.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Iterable, Iterator, TextIO

from ..utils.colors import (
    ASCII_SPINNER_FRAMES,
    SPINNER_FRAMES,
    strip_ansi,
    supports_unicode,
)
from ..utils.tables import render_progress
from ..utils.timeutil import format_duration
from .renderer import OutputOptions

DEFAULT_SPINNER_INTERVAL = 0.09
"""Seconds between spinner frames."""


def is_interactive(stream: TextIO | None = None) -> bool:
    """Return whether the stream is a terminal that can accept animation."""
    target = stream or sys.stderr
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):
        return False


class Printer:
    """Writes output that respects one resolved :class:`OutputOptions`."""

    def __init__(self, options: OutputOptions | None = None, *, config: Any = None,
                 out: TextIO | None = None, err: TextIO | None = None) -> None:
        """Create a printer.

        Args:
            options: resolved options; built from ``config`` when omitted
            config: configuration used to derive options
            out: stream for normal output, defaults to ``sys.stdout``
            err: stream for diagnostics, defaults to ``sys.stderr``
        """
        self.options = options or OutputOptions.from_config(config)
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.lines_written = 0
        self.chars_written = 0
        self._lock = threading.RLock()
        self._live = ""

    # -- capability ----------------------------------------------------------

    @property
    def quiet(self) -> bool:
        """Whether all informational output is suppressed."""
        return bool(self.options.quiet)

    @property
    def color(self) -> bool:
        """Whether colour is enabled for this printer's stream."""
        return self.options.color

    def interactive(self) -> bool:
        """Whether the output stream is a terminal."""
        return is_interactive(self.out)

    def at_least(self, level: str) -> bool:
        """Return whether the verbosity reaches ``level``."""
        return self.options.at_least(level)

    def themed(self, text: str, name: str) -> str:
        """Apply a semantic colour according to these options."""
        return self.options.themed(text, name)

    # -- writing -------------------------------------------------------------

    def write(self, text: str = "", *, stream: TextIO | None = None,
              end: str = "\n", min_level: str = "normal") -> bool:
        """Write text unless suppressed by quiet or verbosity.

        Args:
            text: the payload
            stream: destination, defaults to ``out``
            end: terminator appended to the text
            min_level: verbosity required to emit this line

        Returns:
            Whether anything was written.
        """
        if self.quiet or not self.at_least(min_level):
            return False
        target = stream or self.out
        with self._lock:
            self._clear_live(target)
            try:
                target.write(f"{text}{end}")
                target.flush()
            except (ValueError, OSError):
                return False
            self.lines_written += 1 if end == "\n" else 0
            self.chars_written += len(text) + len(end)
        return True

    def error(self, text: str) -> bool:
        """Write to stderr, bypassing the verbosity ladder but not ``quiet``."""
        if self.quiet:
            return False
        return self.write(self.themed(text, "error"), stream=self.err, min_level="quiet")

    def warn(self, text: str) -> bool:
        """Write a warning to stderr at ``minimal`` verbosity or above."""
        return self.write(self.themed(f"warning: {text}", "warning"), stream=self.err,
                          min_level="minimal")

    def debug(self, text: str) -> bool:
        """Write a diagnostic line, shown only at ``debug`` verbosity."""
        return self.write(self.themed(text, "muted"), stream=self.err, min_level="debug")

    def heading(self, text: str) -> bool:
        """Write a section heading."""
        return self.write(self.themed(text, "heading"), min_level="normal")

    def blank(self, *, min_level: str = "normal") -> bool:
        """Write an empty line to separate blocks."""
        return self.write("", min_level=min_level)

    def lines(self, items: Iterable[str], *, min_level: str = "normal") -> int:
        """Write many lines, returning how many were emitted."""
        return sum(1 for item in items if self.write(item, min_level=min_level))

    # -- streaming -----------------------------------------------------------

    def stream(self, chunks: Iterable[str], *, prefix: str = "", suffix: str = "") -> str:
        """Echo streamed tokens as they arrive, returning the full text.

        Streaming is suppressed entirely when the format is not text or the
        output is not interactive, because partial lines are useless in a pipe
        and harmful in JSON.
        """
        enabled = (self.options.stream_to_terminal and not self.quiet
                   and self.options.fmt == "text" and self.at_least("normal"))
        collected: list[str] = []
        started = False
        for chunk in chunks:
            collected.append(chunk)
            if enabled and chunk:
                target = self.out
                with self._lock:
                    if not started and prefix:
                        target.write(prefix)
                        started = True
                    try:
                        target.write(chunk)
                        target.flush()
                        self.chars_written += len(chunk)
                    except (ValueError, OSError):
                        enabled = False
        if enabled and started and suffix:
            with self._lock:
                try:
                    self.out.write(suffix)
                    self.out.flush()
                except (ValueError, OSError):
                    pass
        if enabled:
            with self._lock:
                try:
                    self.out.write("\n")
                    self.out.flush()
                except (ValueError, OSError):
                    pass
        return "".join(collected)

    # -- live line (spinner / progress) --------------------------------------

    def show_live(self, text: str, *, stream: TextIO | None = None) -> None:
        """Draw a transient line that the next real write will erase."""
        if self.quiet or not is_interactive(stream or self.err):
            return
        target = stream or self.err
        with self._lock:
            self._clear_live(target)
            try:
                target.write(text)
                target.flush()
            except (ValueError, OSError):
                return
            self._live = text

    def clear_live(self, *, stream: TextIO | None = None) -> None:
        """Erase any transient line."""
        with self._lock:
            self._clear_live(stream or self.err)

    def _clear_live(self, target: TextIO) -> None:
        """Overwrite the transient line with spaces, if one was drawn."""
        if not self._live:
            return
        width = len(strip_ansi(self._live))
        try:
            target.write("\r" + " " * width + "\r")
            target.flush()
        except (ValueError, OSError):
            pass
        self._live = ""

    def stats(self) -> dict[str, Any]:
        """Return counters for diagnostics."""
        return {
            "lines": self.lines_written,
            "chars": self.chars_written,
            "color": self.options.color,
            "quiet": self.options.quiet,
            "verbosity": self.options.verbosity,
            "format": self.options.fmt,
            "interactive": self.interactive(),
        }


class Spinner:
    """A background animation showing that work is happening.

    Usable as a context manager or driven manually. It never writes when the
    target is not a terminal, so wrapping a call site in a spinner is always
    safe.
    """

    def __init__(self, text: str = "", *, printer: Printer | None = None,
                 interval: float = DEFAULT_SPINNER_INTERVAL,
                 stream: TextIO | None = None, frames: Iterable[str] | None = None) -> None:
        """Configure a spinner.

        Args:
            text: label drawn after the animation frame
            printer: printer supplying colour and quiet settings
            interval: seconds between frames
            stream: destination, defaults to the printer's stderr
            frames: custom frame characters
        """
        self.printer = printer or Printer()
        self.text = text
        self.interval = max(0.02, float(interval))
        self.stream = stream or self.printer.err
        unicode_ok = self.printer.options.unicode and supports_unicode(self.stream)
        default = SPINNER_FRAMES if unicode_ok else ASCII_SPINNER_FRAMES
        self.frames = list(frames) if frames else list(default)
        self.enabled = (not self.printer.quiet and self.printer.options.spinner
                        and is_interactive(self.stream))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._lock = threading.RLock()
        self.started_at = 0.0
        self.elapsed = 0.0

    @property
    def running(self) -> bool:
        """Whether the animation thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> Spinner:
        """Begin animating; a no-op when disabled."""
        if not self.enabled or self.running:
            return self
        self._stop.clear()
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._spin, name="magent-spinner", daemon=True)
        self._thread.start()
        return self

    def stop(self, *, final_text: str = "") -> float:
        """Stop animating and return the elapsed seconds."""
        self.elapsed = (time.monotonic() - self.started_at) if self.started_at else 0.0
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.interval * 4)
        self._thread = None
        if self.enabled:
            with self._lock:
                self.printer.clear_live(stream=self.stream)
                if final_text:
                    self.printer.write(final_text, stream=self.stream, min_level="quiet")
        return self.elapsed

    def update(self, text: str) -> None:
        """Change the label while spinning."""
        self.text = text

    def _spin(self) -> None:
        """Animation loop, run on a daemon thread."""
        while not self._stop.is_set():
            frame = self.frames[self._index % len(self.frames)]
            self._index += 1
            elapsed = time.monotonic() - self.started_at
            colour = self.printer.themed(frame, "spinner")
            label = f"{colour} {self.text}" if self.text else colour
            if self.printer.at_least("verbose"):
                label += self.printer.themed(f" {format_duration(elapsed)}", "muted")
            self.printer.show_live(label, stream=self.stream)
            self._stop.wait(self.interval)

    def __enter__(self) -> Spinner:
        """Start the spinner."""
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        """Stop the spinner."""
        self.stop()


class ProgressBar:
    """A determinate progress bar for multi-step work."""

    def __init__(self, total: int, *, label: str = "", printer: Printer | None = None,
                 stream: TextIO | None = None, width: int = 0) -> None:
        """Configure a bar.

        Args:
            total: number of units representing 100%
            label: text drawn before the bar
            printer: printer supplying colour and quiet settings
            stream: destination, defaults to the printer's stderr
            width: bar width, derived from the terminal when 0
        """
        self.total = max(1, int(total))
        self.label = label
        self.printer = printer or Printer()
        self.stream = stream or self.printer.err
        self.width = width or max(10, min(40, self.printer.options.width // 3))
        self.value = 0
        self.enabled = (not self.printer.quiet and self.printer.options.progress
                        and is_interactive(self.stream))
        self.started_at = time.monotonic()
        self.elapsed = 0.0
        self._lock = threading.RLock()

    @property
    def ratio(self) -> float:
        """Completion fraction in ``0..1``."""
        return min(1.0, max(0.0, self.value / self.total))

    @property
    def done(self) -> bool:
        """Whether every unit is complete."""
        return self.value >= self.total

    def update(self, value: int | None = None, *, label: str = "") -> None:
        """Set the current value and redraw."""
        with self._lock:
            if value is not None:
                self.value = max(0, min(self.total, int(value)))
            if label:
                self.label = label
        self.draw()

    def advance(self, amount: int = 1, *, label: str = "") -> None:
        """Move forward by ``amount`` units."""
        with self._lock:
            self.value = max(0, min(self.total, self.value + int(amount)))
        self.draw()

    def draw(self) -> None:
        """Render the bar, or nothing when disabled."""
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started_at
        remaining = (elapsed / max(1, self.value)) * (self.total - self.value) if self.value else 0.0
        text = render_progress(self.value, self.total, width=self.width,
                               label=self.label, color=self.printer.color)
        if self.printer.at_least("verbose"):
            text += self.printer.themed(f" {self.value}/{self.total}", "muted")
            text += self.printer.themed(f" eta {format_duration(remaining)}", "muted")
        self.printer.show_live(text, stream=self.stream)

    def finish(self, *, message: str = "") -> float:
        """Complete the bar and return the elapsed seconds."""
        self.value = self.total
        self.draw()
        self.elapsed = time.monotonic() - self.started_at
        if self.enabled:
            with self._lock:
                self.printer.clear_live(stream=self.stream)
        if message:
            self.printer.write(message, stream=self.stream, min_level="quiet")
        return self.elapsed

    def wrap(self, items: Iterable[Any], *, label: str = "") -> Iterator[Any]:
        """Iterate ``items``, advancing the bar once per item."""
        if label:
            self.label = label
        for item in items:
            yield item
            self.advance()
        self.finish()

    def __enter__(self) -> ProgressBar:
        """Draw the empty bar."""
        self.draw()
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Finish the bar."""
        self.finish()

    def stats(self) -> dict[str, Any]:
        """Return progress counters."""
        return {"value": self.value, "total": self.total, "ratio": round(self.ratio, 4),
                "elapsed": round(self.elapsed, 3), "enabled": self.enabled}


def spinner(text: str = "", **kwargs: Any) -> Spinner:
    """Return a spinner, ready to use as a context manager."""
    return Spinner(text, **kwargs)


def progress(total: int, **kwargs: Any) -> ProgressBar:
    """Return a progress bar, ready to use as a context manager."""
    return ProgressBar(total, **kwargs)


__all__ = [
    "DEFAULT_SPINNER_INTERVAL",
    "Printer",
    "ProgressBar",
    "Spinner",
    "is_interactive",
    "progress",
    "spinner",
]
