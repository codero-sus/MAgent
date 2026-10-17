"""Output sinks: where rendered text goes.

Every renderer writes through a sink rather than printing directly.  That keeps
the CLI testable (a :class:`BufferSink` captures everything), lets one run write
to the terminal *and* a transcript file (:class:`TeeSink`), and makes the quiet
flag a single switch (:class:`NullSink`) instead of a condition scattered over
the renderers.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, TextIO

from ..utils.files import atomic_write_text, ensure_dir


class OutputSink:
    """Base sink: accepts text and flushes it somewhere."""

    name = "sink"
    """Identifier used in logs and ``doctor`` output."""

    def write(self, text: str) -> None:
        """Write text to the sink."""
        raise NotImplementedError

    def line(self, text: str = "") -> None:
        """Write text followed by a newline."""
        self.write(text + "\n")

    def lines(self, items: Iterable[str]) -> None:
        """Write many lines."""
        for item in items:
            self.line(item)

    def flush(self) -> None:
        """Flush buffered output."""

    def close(self) -> None:
        """Release any underlying resource."""

    def tell(self) -> int:
        """Number of characters written so far."""
        return 0

    def __enter__(self) -> OutputSink:
        """Support ``with`` usage."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Flush and close on exit."""
        self.flush()
        self.close()

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"{type(self).__name__}(name={self.name!r})"


class ConsoleSink(OutputSink):
    """Write to a terminal stream, optionally without color."""

    name = "console"

    def __init__(self, *, stream: TextIO | None = None, error: bool = False,
                 color: bool | None = None) -> None:
        """Bind the sink to a stream.

        Args:
            stream: explicit stream, defaulting to stdout/stderr
            error: use stderr when no stream is given
            color: force color on/off; ``None`` detects the terminal
        """
        self.error = error
        self._explicit = stream
        self._stream = stream
        self._color = color
        self._written = 0
        self._lock = threading.Lock()

    @property
    def stream(self) -> TextIO:
        """The live stream, re-resolved so redirected stdout is honoured."""
        if self._explicit is not None:
            return self._explicit
        return sys.stderr if self.error else sys.stdout

    @property
    def color(self) -> bool:
        """Whether ANSI color should be emitted."""
        if self._color is not None:
            return bool(self._color)
        from ..utils.colors import supports_color

        return supports_color(self.stream)

    @property
    def width(self) -> int:
        """Terminal width in columns."""
        from ..utils.colors import terminal_size

        return terminal_size(self.stream)[0]

    def write(self, text: str) -> None:
        """Write text, stripping color when the stream is not a terminal."""
        if not text:
            return
        if not self.color:
            from ..utils.colors import strip_ansi

            text = strip_ansi(text)
        with self._lock:
            stream = self.stream
            stream.write(text)
            stream.flush()
            self._written += len(text)

    def flush(self) -> None:
        """Flush the underlying stream."""
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass

    def tell(self) -> int:
        """Characters written."""
        return self._written


class NullSink(OutputSink):
    """Discard everything; used by ``--quiet`` and headless runs."""

    name = "null"

    def __init__(self) -> None:
        """Create the sink."""
        self._written = 0

    def write(self, text: str) -> None:
        """Count and discard."""
        self._written += len(text)

    def tell(self) -> int:
        """Characters swallowed."""
        return self._written


class BufferSink(OutputSink):
    """Collect output in memory; the workhorse of the test suite."""

    name = "buffer"

    def __init__(self, *, keep: int = 0) -> None:
        """Create the buffer.

        Args:
            keep: maximum characters retained, 0 for unlimited
        """
        self._parts: list[str] = []
        self._keep = max(0, int(keep))
        self._written = 0

    def write(self, text: str) -> None:
        """Append text to the buffer."""
        if not text:
            return
        self._parts.append(text)
        self._written += len(text)
        if self._keep and sum(len(part) for part in self._parts) > self._keep * 2:
            joined = "".join(self._parts)[-self._keep:]
            self._parts = [joined]

    @property
    def text(self) -> str:
        """Everything written so far."""
        return "".join(self._parts)

    def value(self) -> str:
        """Alias of :attr:`text`, matching ``io.StringIO``."""
        return self.text

    def clear(self) -> None:
        """Empty the buffer without resetting the character count."""
        self._parts.clear()

    def contains(self, needle: str) -> bool:
        """True when ``needle`` appears in the captured text."""
        return needle in self.text

    def getvalue(self) -> str:
        """``io.StringIO``-compatible accessor."""
        return self.text

    def tell(self) -> int:
        """Characters written."""
        return self._written

    def __str__(self) -> str:
        """String form is the captured text."""
        return self.text


class FileSink(OutputSink):
    """Append output to a file, creating parent directories as needed."""

    name = "file"

    def __init__(self, path: str | Path, *, append: bool = True, encoding: str = "utf-8") -> None:
        """Open the target file.

        Args:
            path: destination file
            append: append when the file already exists
            encoding: text encoding
        """
        self.path = Path(path).expanduser()
        self.encoding = encoding
        ensure_dir(self.path.parent)
        self._handle = self.path.open("a" if append else "w", encoding=encoding)
        self._written = 0

    def write(self, text: str) -> None:
        """Write text to the file."""
        if not text:
            return
        from ..utils.colors import strip_ansi

        self._handle.write(strip_ansi(text))
        self._written += len(text)

    def flush(self) -> None:
        """Flush the file handle."""
        try:
            self._handle.flush()
        except (ValueError, OSError):
            pass

    def close(self) -> None:
        """Close the file handle."""
        try:
            self._handle.close()
        except (ValueError, OSError):
            pass

    def tell(self) -> int:
        """Characters written."""
        return self._written


class TeeSink(OutputSink):
    """Fan output out to several sinks."""

    name = "tee"

    def __init__(self, *sinks: OutputSink) -> None:
        """Wrap the given sinks."""
        self.sinks = [sink for sink in sinks if sink is not None]

    def write(self, text: str) -> None:
        """Write to every sink, tolerating individual failures."""
        for sink in self.sinks:
            try:
                sink.write(text)
            except Exception:  # noqa: BLE001 - one dead sink must not stop the others
                continue

    def flush(self) -> None:
        """Flush every sink."""
        for sink in self.sinks:
            sink.flush()

    def close(self) -> None:
        """Close every sink."""
        for sink in self.sinks:
            sink.close()

    def tell(self) -> int:
        """Largest character count among the sinks."""
        return max((sink.tell() for sink in self.sinks), default=0)


class TimestampSink(OutputSink):
    """Prefix every line with a timestamp, for transcripts and logs."""

    name = "timestamp"

    def __init__(self, inner: OutputSink, *, fmt: str = "%H:%M:%S") -> None:
        """Wrap ``inner``.

        Args:
            inner: sink receiving the decorated text
            fmt: ``strftime`` pattern for the prefix
        """
        self.inner = inner
        self.fmt = fmt
        self._at_line_start = True

    def write(self, text: str) -> None:
        """Write text, prefixing each new line with the time."""
        if not text:
            return
        out: list[str] = []
        for char in text:
            if self._at_line_start and char != "\n":
                out.append(time.strftime(self.fmt) + " ")
                self._at_line_start = False
            out.append(char)
            if char == "\n":
                self._at_line_start = True
        self.inner.write("".join(out))

    def flush(self) -> None:
        """Flush the wrapped sink."""
        self.inner.flush()

    def close(self) -> None:
        """Close the wrapped sink."""
        self.inner.close()

    def tell(self) -> int:
        """Characters written to the wrapped sink."""
        return self.inner.tell()


def build_sink(config: Any = None, *, path: str | Path | None = None,
               stream: TextIO | None = None, quiet: bool | None = None) -> OutputSink:
    """Choose a sink from configuration.

    Args:
        config: an :class:`~magent.config.OutputConfig` or full config
        path: transcript file to tee into
        stream: explicit stream for the console sink
        quiet: override the configured quiet flag
    """
    output = getattr(config, "output", config)
    is_quiet = bool(quiet if quiet is not None else getattr(output, "quiet", False))
    if is_quiet and not path:
        return NullSink()
    color = None
    if getattr(output, "color", None) is False:
        color = False
    sinks: list[OutputSink] = []
    if not is_quiet:
        sinks.append(ConsoleSink(stream=stream, color=color))
    if path:
        sinks.append(FileSink(path))
    if not sinks:
        return NullSink()
    if len(sinks) == 1:
        return sinks[0]
    return TeeSink(*sinks)


def capture_lines(text: str) -> list[str]:
    """Split rendered output into lines, dropping the trailing empty one."""
    return text.splitlines()


def write_capture(path: str | Path, text: str) -> Path:
    """Atomically write captured output to a file."""
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    return atomic_write_text(target, text)


def redirect_stdout(sink: OutputSink) -> Any:
    """Context manager routing ``print`` through a sink."""
    import contextlib

    class _Writer(io.TextIOBase):
        def write(self, text: str) -> int:
            sink.write(text)
            return len(text)

        def flush(self) -> None:
            sink.flush()

    return contextlib.redirect_stdout(_Writer())


__all__ = [
    "BufferSink",
    "ConsoleSink",
    "FileSink",
    "NullSink",
    "OutputSink",
    "TeeSink",
    "TimestampSink",
    "build_sink",
    "capture_lines",
    "redirect_stdout",
    "write_capture",
]
