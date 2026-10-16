"""Structured logging with automatic secret redaction and rotation.

MAgent uses the standard :mod:`logging` module so it composes with whatever the
host application already does, but adds three things the stdlib lacks:

* a :class:`RedactingFilter` that scrubs credentials from every record,
* a JSON formatter producing one object per line for machine consumption,
* size based rotation with a bounded number of retained files.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .secrets import redact_text
from .utils.colors import ANSI, colorize, supports_color
from .utils.files import ensure_dir
from .utils.timeutil import utc_iso

LOGGER_NAME = "magent"
"""Name of the root MAgent logger."""

LEVEL_COLORS = {
    "DEBUG": ANSI.GRAY,
    "INFO": ANSI.CYAN,
    "WARNING": ANSI.YELLOW,
    "ERROR": ANSI.RED,
    "CRITICAL": ANSI.BRIGHT_RED + ANSI.BOLD,
}
"""ANSI colour per log level."""

VALID_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
"""Levels accepted by :func:`setup_logging`."""

_configured = False
_lock = threading.RLock()


class RedactingFilter(logging.Filter):
    """Strip credentials from every log record before it is emitted."""

    def __init__(self, extra_secrets: list[str] | None = None) -> None:
        super().__init__()
        self.extra_secrets = [secret for secret in (extra_secrets or []) if secret]
        self.redactions = 0

    def add_secret(self, secret: str) -> None:
        """Register an additional literal secret to scrub."""
        if secret and secret not in self.extra_secrets:
            self.extra_secrets.append(secret)

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record message and args in place, then allow it."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging break the program
            return True
        cleaned = redact_text(message, extra_secrets=self.extra_secrets)
        if cleaned != message:
            self.redactions += 1
        record.msg = cleaned
        record.args = ()
        for key, value in list(getattr(record, "__dict__").items()):
            if key in {"msg", "args"} or not isinstance(value, str):
                continue
            scrubbed = redact_text(value, extra_secrets=self.extra_secrets)
            if scrubbed != value:
                setattr(record, key, scrubbed)
        return True


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON objects."""

    def __init__(self, *, include_stack: bool = True, static_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.include_stack = include_stack
        self.static_fields = dict(static_fields or {})

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON line."""
        payload: dict[str, Any] = {
            "ts": utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(self.static_fields)
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context
        if record.exc_info and self.include_stack:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human readable formatter with optional colour and relative timestamps."""

    def __init__(self, *, color: bool | None = None, show_logger: bool = True,
                 relative_time: bool = False) -> None:
        super().__init__()
        self.color = supports_color(sys.stderr) if color is None else color
        self.show_logger = show_logger
        self.relative_time = relative_time
        self._start = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a coloured single line."""
        if self.relative_time:
            stamp = f"+{time.monotonic() - self._start:8.3f}s"
        else:
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname[:4].ljust(4)
        if self.color:
            level = colorize(level, LEVEL_COLORS.get(record.levelname, ""), enabled=True)
        parts = [stamp, level]
        if self.show_logger and record.name != LOGGER_NAME:
            parts.append(colorize(record.name.replace("magent.", ""), ANSI.GRAY, enabled=self.color))
        parts.append(record.getMessage())
        line = " ".join(part for part in parts if part)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def setup_logging(
    *,
    level: str = "INFO",
    fmt: str = "text",
    log_file: str | Path | None = None,
    color: bool | None = None,
    force: bool = False,
    quiet: bool = False,
    max_bytes: int = 32 * 1024 * 1024,
    backup_count: int = 5,
    static_fields: dict[str, Any] | None = None,
    stream: Any = None,
) -> logging.Logger:
    """Configure the MAgent logger and return it.

    Idempotent: repeated calls return the same configured logger unless
    ``force`` is set, which is what ``magent --log-level debug`` uses.
    """
    global _configured
    with _lock:
        logger = logging.getLogger(LOGGER_NAME)
        if _configured and not force:
            return logger
        normalized = str(level).upper()
        if normalized not in VALID_LEVELS:
            normalized = "INFO"
        logger.setLevel(getattr(logging, normalized))
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        if quiet:
            logger.addHandler(logging.NullHandler())
            _configured = True
            return logger
        redactor = RedactingFilter()
        target = stream if stream is not None else sys.stderr
        console = logging.StreamHandler(target)
        console.setLevel(getattr(logging, normalized))
        if fmt == "json":
            console.setFormatter(JsonFormatter(static_fields=static_fields))
        else:
            console.setFormatter(TextFormatter(color=color))
        console.addFilter(redactor)
        logger.addHandler(console)
        if log_file:
            path = Path(log_file).expanduser()
            try:
                ensure_dir(path.parent)
                file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                    path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
            except OSError:
                file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(getattr(logging, normalized))
            file_handler.setFormatter(JsonFormatter(static_fields=static_fields))
            file_handler.addFilter(redactor)
            logger.addHandler(file_handler)
        logger.magent_redactor = redactor  # type: ignore[attr-defined]
        _configured = True
        return logger


def get_logger(name: str = "") -> logging.Logger:
    """Return a child logger under the MAgent namespace."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def register_secret(secret: str) -> None:
    """Add a literal secret to every configured redaction filter."""
    logger = logging.getLogger(LOGGER_NAME)
    redactor = getattr(logger, "magent_redactor", None)
    if isinstance(redactor, RedactingFilter):
        redactor.add_secret(secret)


def log_event(level: str, message: str, **context: Any) -> None:
    """Emit one structured log record with an attached context mapping."""
    logger = get_logger()
    record_level = getattr(logging, str(level).upper(), logging.INFO)
    extra = {"context": context} if context else {}
    logger.log(record_level, message, extra=extra)


def debug(message: str, **context: Any) -> None:
    """Log at DEBUG level."""
    log_event("DEBUG", message, **context)


def info(message: str, **context: Any) -> None:
    """Log at INFO level."""
    log_event("INFO", message, **context)


def warning(message: str, **context: Any) -> None:
    """Log at WARNING level."""
    log_event("WARNING", message, **context)


def error(message: str, **context: Any) -> None:
    """Log at ERROR level."""
    log_event("ERROR", message, **context)


def exception(message: str, **context: Any) -> None:
    """Log at ERROR level including the current traceback."""
    logger = get_logger()
    logger.exception(message, extra={"context": context} if context else {})


class LogCapture:
    """In-memory log handler used by tests and ``magent doctor``."""

    def __init__(self, level: str = "DEBUG") -> None:
        self.handler = logging.StreamHandler(_NullStream())
        self.handler.setLevel(getattr(logging, level))
        self.handler.setFormatter(JsonFormatter())
        self.records: list[logging.LogRecord] = []
        self._collector = logging.Handler()
        self._collector.emit = self.records.append  # type: ignore[method-assign]
        self._collector.setLevel(getattr(logging, level))
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.addHandler(self._collector)
        self.logger.setLevel(getattr(logging, level))

    def messages(self, level: str | None = None) -> list[str]:
        """Return captured messages, optionally filtered by level."""
        out = []
        for record in self.records:
            if level and record.levelname != level.upper():
                continue
            try:
                out.append(record.getMessage())
            except Exception:  # noqa: BLE001
                continue
        return out

    def contains(self, needle: str) -> bool:
        """Report whether any captured message contains ``needle``."""
        return any(needle in message for message in self.messages())

    def stop(self) -> None:
        """Detach the capture handler."""
        self.logger.removeHandler(self._collector)
        self.logger.removeHandler(self.handler)

    def __enter__(self) -> LogCapture:
        """Start capturing."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop capturing."""
        self.stop()


class _NullStream:
    """A write-only sink used to keep :class:`LogCapture` handler valid."""

    def write(self, _data: str) -> int:
        """Discard the data and report zero bytes written."""
        return 0

    def flush(self) -> None:
        """No-op flush to satisfy the handler interface."""


def log_file_size(path: str | Path) -> int:
    """Return the size of a log file, or 0 when it does not exist."""
    try:
        return Path(path).expanduser().stat().st_size
    except OSError:
        return 0


def tail_log(path: str | Path, lines: int = 50) -> list[str]:
    """Return the last ``lines`` lines of a log file."""
    target = Path(path).expanduser()
    if not target.is_file():
        return []
    try:
        with target.open("r", encoding="utf-8", errors="replace") as stream:
            from collections import deque

            return list(deque(stream, maxlen=max(1, lines)))
    except OSError:
        return []


def env_log_level(default: str = "INFO") -> str:
    """Return the log level from ``MAGENT_LOG_LEVEL`` or ``default``."""
    value = os.environ.get("MAGENT_LOG_LEVEL", "").strip().upper()
    return value if value in VALID_LEVELS else default


__all__ = [
    "JsonFormatter",
    "LEVEL_COLORS",
    "LOGGER_NAME",
    "LogCapture",
    "RedactingFilter",
    "TextFormatter",
    "VALID_LEVELS",
    "debug",
    "env_log_level",
    "error",
    "exception",
    "get_logger",
    "info",
    "log_event",
    "log_file_size",
    "register_secret",
    "setup_logging",
    "tail_log",
    "warning",
]
