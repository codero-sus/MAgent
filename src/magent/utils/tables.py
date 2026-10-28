"""Dependency-free table, box, and progress rendering for the terminal."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .colors import (
    ANSI,
    ASCII_BOX,
    BOX_DRAWING,
    PROGRESS_BLOCKS,
    colorize,
    strip_ansi,
    supports_unicode,
)
from .text import ellipsize


def truncate_cell(value: Any, width: int, *, marker: str = "...") -> str:
    """Truncate a rendered cell to ``width`` columns, respecting ANSI codes."""
    text = str(value)
    if width <= 0:
        return ""
    plain = strip_ansi(text)
    if len(plain) <= width:
        return text
    if "\x1b" in text:
        return f"{ellipsize(plain, width, marker=marker)}"
    return ellipsize(text, width, marker=marker)


def _cell_width(value: Any) -> int:
    """Return the display width of a cell value."""
    return len(strip_ansi(str(value)))


def _align(value: Any, width: int, alignment: str) -> str:
    """Pad ``value`` to ``width`` according to ``alignment``."""
    text = str(value)
    padding = max(0, width - _cell_width(text))
    if alignment == "right":
        return " " * padding + text
    if alignment == "center":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding


def render_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str] | None = None,
    *,
    alignments: Sequence[str] | None = None,
    max_widths: Sequence[int] | None = None,
    default_max_width: int = 60,
    borders: bool = True,
    unicode_borders: bool | None = None,
    color: bool | None = None,
    header_style: str = ANSI.BOLD,
    zebra: bool = False,
) -> str:
    """Render a list of rows as an aligned ASCII/Unicode table.

    Column widths are computed from content, capped by ``max_widths``. Cells
    are truncated with an ellipsis rather than wrapping, which keeps tables
    readable in narrow terminals.
    """
    use_unicode = supports_unicode() if unicode_borders is None else unicode_borders
    box = BOX_DRAWING if use_unicode else ASCII_BOX
    body = [list(row) for row in rows]
    column_count = max([len(headers or [])] + [len(row) for row in body] or [0])
    if column_count == 0:
        return ""
    for row in body:
        while len(row) < column_count:
            row.append("")

    widths = [0] * column_count
    limits = list(max_widths or []) + [default_max_width] * column_count
    if headers:
        for index, header in enumerate(headers[:column_count]):
            widths[index] = min(_cell_width(header), limits[index] or default_max_width)
    for row in body:
        for index, cell in enumerate(row[:column_count]):
            limit = limits[index] or default_max_width
            widths[index] = max(widths[index], min(_cell_width(cell), limit))

    align_list = list(alignments or []) + ["left"] * column_count

    def line(left: str, middle: str, right: str, fill: str) -> str:
        return left + middle.join(fill * (width + 2) for width in widths) + right

    out: list[str] = []
    if borders:
        out.append(line(box["tl"], box["tt"], box["tr"], box["h"]))
    if headers:
        header_cells = [
            headers[index] if index < len(headers) else "" for index in range(column_count)
        ]
        cells = [
            _align(
                colorize(truncate_cell(header, widths[index]), header_style, enabled=color),
                widths[index],
                align_list[index],
            )
            for index, header in enumerate(header_cells)
        ]
        row_text = f" {box['v']} ".join(cells) if borders else "  ".join(cells)
        if borders:
            row_text = f"{box['v']} {row_text} {box['v']}"
        out.append(row_text)
        if borders:
            out.append(line(box["lt"], box["cross"], box["rt"], box["h"]))
    for row_index, row in enumerate(body):
        cells = [
            _align(
                truncate_cell(cell, widths[index]),
                widths[index],
                align_list[index],
            )
            for index, cell in enumerate(row)
        ]
        row_text = f" {box['v']} ".join(cells) if borders else "  ".join(cells)
        if borders:
            row_text = f"{box['v']} {row_text} {box['v']}"
        if zebra and row_index % 2 == 1:
            row_text = colorize(row_text, ANSI.DIM, enabled=color)
        out.append(row_text)
    if borders:
        out.append(line(box["bl"], box["bt"], box["br"], box["h"]))
    return "\n".join(out)


def render_markdown_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    *,
    alignments: Sequence[str] | None = None,
) -> str:
    """Render rows as a GitHub flavoured Markdown table."""
    if not headers:
        return ""
    column_count = len(headers)
    align_list = list(alignments or []) + ["left"] * column_count
    widths = [_cell_width(header) for header in headers]
    body: list[list[str]] = []
    for row in rows:
        cells = [str(cell) for cell in list(row)[:column_count]]
        cells += [""] * (column_count - len(cells))
        body.append(cells)
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], _cell_width(cell))

    def separator(alignment: str, width: int) -> str:
        if alignment == "right":
            return "-" * (width - 1) + ":"
        if alignment == "center":
            return ":" + "-" * max(1, width - 2) + ":"
        return "-" * width

    lines = ["| " + " | ".join(_align(h, widths[i], "left") for i, h in enumerate(headers)) + " |"]
    lines.append(
        "| " + " | ".join(separator(align_list[i], widths[i]) for i in range(column_count)) + " |"
    )
    for cells in body:
        lines.append(
            "| " + " | ".join(_align(c, widths[i], "left") for i, c in enumerate(cells)) + " |"
        )
    return "\n".join(lines)


def render_key_value(pairs: Iterable[tuple[str, Any]], *, key_width: int | None = None) -> str:
    """Render ``key: value`` lines with aligned colons."""
    items = [(str(key), value) for key, value in pairs]
    if not items:
        return ""
    width = key_width or max(len(key) for key, _ in items)
    return "\n".join(f"{key.ljust(width)} : {value}" for key, value in items)


def render_box(
    content: str,
    *,
    title: str | None = None,
    width: int | None = None,
    padding: int = 1,
    unicode_borders: bool | None = None,
    style: str = "",
    color: bool | None = None,
) -> str:
    """Render ``content`` inside a bordered box with an optional title."""
    use_unicode = supports_unicode() if unicode_borders is None else unicode_borders
    box = BOX_DRAWING if use_unicode else ASCII_BOX
    lines = content.split("\n") if content else [""]
    inner = width or (max(len(strip_ansi(line)) for line in lines) + padding * 2)
    inner = max(inner, len(title or "") + 4)
    body_width = inner - padding * 2

    def frame(left: str, middle: str, right: str) -> str:
        if title and left == box["tl"]:
            label = f" {title} "
            fill = max(0, inner - len(label) - 2)
            return colorize(left + box["h"] + label + box["h"] * fill + right, style, enabled=color)
        return colorize(left + middle * inner + right, style, enabled=color)

    out = [frame(box["tl"], box["h"], box["tr"])]
    for line in lines:
        rendered = line if len(line) <= body_width else ellipsize(line, body_width)
        pad = " " * padding
        out.append(
            colorize(box["v"], style, enabled=color)
            + pad
            + rendered.ljust(body_width)
            + pad
            + colorize(box["v"], style, enabled=color)
        )
    out.append(frame(box["bl"], box["h"], box["br"]))
    return "\n".join(out)


def render_progress(
    value: float,
    total: float,
    *,
    width: int = 30,
    label: str = "",
    show_percent: bool = True,
    color: bool | None = None,
) -> str:
    """Render a fractional progress bar with eighth-block precision."""
    ratio = 0.0 if total <= 0 else max(0.0, min(1.0, value / total))
    filled = ratio * width
    whole = int(filled)
    remainder = filled - whole
    bar = "█" * whole
    if whole < width and remainder > 0:
        index = int(remainder * (len(PROGRESS_BLOCKS) - 1))
        bar += PROGRESS_BLOCKS[index]
    bar = bar.ljust(width)
    text = f"[{bar}]"
    if show_percent:
        text += f" {ratio * 100:5.1f}%"
    if label:
        text = f"{label} {text}"
    return colorize(text, ANSI.CYAN, enabled=color)


def render_bar_chart(
    data: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    width: int = 30,
    label_width: int | None = None,
    color: bool | None = None,
) -> str:
    """Render a horizontal bar chart from a mapping or pair sequence."""
    items = (
        list(data.items()) if isinstance(data, Mapping) else [(str(k), float(v)) for k, v in data]
    )
    if not items:
        return ""
    peak = max((abs(float(value)) for _, value in items), default=0.0) or 1.0
    key_width = label_width or max(len(str(key)) for key, _ in items)
    lines: list[str] = []
    for key, value in items:
        blocks = int(round(abs(float(value)) / peak * width))
        bar = colorize("█" * blocks, ANSI.BRIGHT_CYAN, enabled=color)
        lines.append(f"{str(key).ljust(key_width)} │ {bar} {value:g}")
    return "\n".join(lines)


def render_spinner_frame(index: int, frames: Sequence[str] | None = None) -> str:
    """Return one animation frame for a spinner."""
    from .colors import SPINNER_FRAMES

    options = frames or SPINNER_FRAMES
    return options[index % len(options)]


def indent(text: str, prefix: str = "  ") -> str:
    """Prefix every line of ``text`` with ``prefix``."""
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Return ``"1 file"`` or ``"3 files"`` style phrasing."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


__all__ = [
    "indent",
    "pluralize",
    "render_bar_chart",
    "render_box",
    "render_key_value",
    "render_markdown_table",
    "render_progress",
    "render_spinner_frame",
    "render_table",
    "truncate_cell",
]
