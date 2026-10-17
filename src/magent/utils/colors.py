"""ANSI terminal styling with automatic capability detection."""

from __future__ import annotations

import os
import re
import sys

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
"""Regex that matches every CSI escape sequence."""


class ANSI:
    """Namespace of ANSI escape codes used by the MAgent renderer."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"
    BLINK = "\x1b[5m"
    REVERSE = "\x1b[7m"
    HIDDEN = "\x1b[8m"
    STRIKETHROUGH = "\x1b[9m"

    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    GRAY = "\x1b[90m"

    BRIGHT_RED = "\x1b[91m"
    BRIGHT_GREEN = "\x1b[92m"
    BRIGHT_YELLOW = "\x1b[93m"
    BRIGHT_BLUE = "\x1b[94m"
    BRIGHT_MAGENTA = "\x1b[95m"
    BRIGHT_CYAN = "\x1b[96m"
    BRIGHT_WHITE = "\x1b[97m"

    BG_BLACK = "\x1b[40m"
    BG_RED = "\x1b[41m"
    BG_GREEN = "\x1b[42m"
    BG_YELLOW = "\x1b[43m"
    BG_BLUE = "\x1b[44m"

    CLEAR_LINE = "\x1b[2K"
    CURSOR_UP = "\x1b[1A"
    CURSOR_DOWN = "\x1b[1B"
    LINE_START = "\x1b[0G"
    HIDE_CURSOR = "\x1b[?25l"
    SHOW_CURSOR = "\x1b[?25h"

    @classmethod
    def rgb(cls, red: int, green: int, blue: int, *, background: bool = False) -> str:
        """Return a 24-bit truecolor escape sequence."""
        code = 48 if background else 38
        return f"\x1b[{code};2;{red & 255};{green & 255};{blue & 255}m"

    @classmethod
    def color256(cls, index: int, *, background: bool = False) -> str:
        """Return an 8-bit 256-color escape sequence."""
        code = 48 if background else 38
        return f"\x1b[{code};5;{index & 255}m"


THEME: dict[str, str] = {
    "brand": ANSI.BRIGHT_CYAN,
    "accent": ANSI.MAGENTA,
    "success": ANSI.GREEN,
    "warning": ANSI.YELLOW,
    "error": ANSI.RED,
    "info": ANSI.BLUE,
    "muted": ANSI.GRAY,
    "heading": ANSI.BOLD + ANSI.BRIGHT_WHITE,
    "code": ANSI.CYAN,
    "tool": ANSI.BRIGHT_MAGENTA,
    "user": ANSI.BRIGHT_GREEN,
    "assistant": ANSI.BRIGHT_CYAN,
    "system": ANSI.GRAY,
    "spinner": ANSI.BRIGHT_CYAN,
}
"""Semantic colour names so the whole CLI can be re-themed at once."""

SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
"""Braille spinner frames."""

ASCII_SPINNER_FRAMES: tuple[str, ...] = ("|", "/", "-", "\\")
"""Fallback spinner for terminals without Unicode support."""

PROGRESS_BLOCKS = (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")
"""Eighth-block characters for smooth progress bars."""

BOX_DRAWING = {
    "tl": "╭",
    "tr": "╮",
    "bl": "╰",
    "br": "╯",
    "h": "─",
    "v": "│",
    "lt": "├",
    "rt": "┤",
    "tt": "┬",
    "bt": "┴",
    "cross": "┼",
}
"""Unicode box drawing characters."""

ASCII_BOX = {
    "tl": "+",
    "tr": "+",
    "bl": "+",
    "br": "+",
    "h": "-",
    "v": "|",
    "lt": "+",
    "rt": "+",
    "tt": "+",
    "bt": "+",
    "cross": "+",
}
"""ASCII fallback box drawing characters."""

_cache: dict[str, bool] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with a sensible default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def supports_color(stream: object | None = None) -> bool:
    """Report whether the target stream should receive ANSI colour codes.

    Respects ``NO_COLOR``, ``FORCE_COLOR``, ``TERM=dumb``, non-TTY streams and
    Windows terminals without VT processing enabled.
    """
    key = f"color:{id(stream) if stream is not None else 'stdout'}"
    if key in _cache:
        return _cache[key]
    if _env_flag("NO_COLOR"):
        result = False
    elif _env_flag("FORCE_COLOR"):
        result = True
    else:
        target = stream if stream is not None else sys.stdout
        isatty = bool(getattr(target, "isatty", lambda: False)())
        term = os.environ.get("TERM", "")
        result = (
            isatty
            and term != "dumb"
            and (
                os.name != "nt"
                or term
                or _env_flag("WT_SESSION")
                or os.environ.get("ANSICON") is not None
            )
        )
    _cache[key] = result
    return result


def supports_unicode(stream: object | None = None) -> bool:
    """Report whether the stream can render Unicode box/spinner characters."""
    target = stream if stream is not None else sys.stdout
    encoding = str(getattr(target, "encoding", "") or "").lower()
    if "utf" in encoding:
        return True
    return os.environ.get("LANG", "").lower().endswith(("utf-8", "utf8"))


def terminal_size(default: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """Return ``(columns, rows)`` for the controlling terminal."""
    try:
        size = os.get_terminal_size()
        return size.columns or default[0], size.lines or default[1]
    except OSError:
        columns = os.environ.get("COLUMNS")
        rows = os.environ.get("LINES")
        return (
            int(columns) if columns and columns.isdigit() else default[0],
            int(rows) if rows and rows.isdigit() else default[1],
        )


def strip_ansi(text: str) -> str:
    """Remove every ANSI escape sequence from ``text``."""
    return _ANSI_ESCAPE.sub("", text)


def visible_width(text: str) -> int:
    """Return the on-screen width of ``text`` ignoring escape codes."""
    return len(strip_ansi(text))


def colorize(
    text: str,
    *styles: str,
    enabled: bool | None = None,
) -> str:
    """Wrap ``text`` in ANSI styles, or return it plain when disabled.

    A style may be a raw escape sequence (``ANSI.RED``) or a :data:`THEME` name
    such as ``"error"``.  Anything else is dropped instead of being emitted: a
    typo used to render the style name itself into the output, which looked like
    a formatting bug in the text rather than in the call.
    """
    if enabled is None:
        enabled = supports_color()
    if not enabled or not styles or not text:
        return text
    resolved: list[str] = []
    for style in styles:
        if not style:
            continue
        if style.startswith("\x1b"):
            resolved.append(style)
        elif style in THEME:
            resolved.append(THEME[style])
    if not resolved:
        return text
    return f"{''.join(resolved)}{text}{ANSI.RESET}"


def themed(text: str, name: str, *, enabled: bool | None = None) -> str:
    """Apply a :data:`THEME` entry by semantic name."""
    style = THEME.get(name, "")
    return colorize(text, style, enabled=enabled) if style else text


def reset_color_cache() -> None:
    """Clear the memoized capability checks (used after changing env vars)."""
    _cache.clear()


__all__ = [
    "ANSI",
    "ASCII_BOX",
    "ASCII_SPINNER_FRAMES",
    "BOX_DRAWING",
    "PROGRESS_BLOCKS",
    "SPINNER_FRAMES",
    "THEME",
    "colorize",
    "reset_color_cache",
    "strip_ansi",
    "supports_color",
    "supports_unicode",
    "terminal_size",
    "themed",
    "visible_width",
]
