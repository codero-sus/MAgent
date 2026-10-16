"""Markdown rendering for the terminal.

Model answers arrive as Markdown.  Dumping raw ``**asterisks**`` and fenced code
blocks into a terminal is readable but noisy, and piping them through a browser
is not an option for a CLI.  This module converts Markdown into ANSI-styled
text: headings are bold, inline code is cyan, code blocks are boxed and dimmed,
lists are re-bulleted, tables are aligned, and links keep their URL in a muted
suffix.  Everything degrades to plain text when the stream is not a terminal, so
redirected output stays diff-friendly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..utils.colors import colorize, strip_ansi, themed, visible_width
from ..utils.tables import render_box
from ..utils.text import ellipsize, word_wrap

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
"""ATX heading matcher."""

FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)\s*$")
"""Fenced code block delimiter matcher."""

BULLET_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
"""Unordered list item matcher."""

ORDERED_RE = re.compile(r"^(\s*)(\d{1,9})[.)]\s+(.*)$")
"""Ordered list item matcher."""

QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
"""Blockquote matcher."""

RULE_RE = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$")
"""Horizontal rule matcher."""

TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
"""Markdown table row matcher."""

TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
"""Markdown table separator matcher."""

BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
"""Bold inline matcher."""

ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)|(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)")
"""Italic inline matcher."""

CODE_RE = re.compile(r"`([^`]+)`")
"""Inline code matcher."""

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
"""Markdown link matcher."""

STRIKE_RE = re.compile(r"~~(.+?)~~")
"""Strikethrough matcher."""

MAX_CODE_LINES = 400
"""Code blocks longer than this are elided in the middle."""


@dataclass(slots=True)
class MarkdownOptions:
    """Knobs for :func:`render_markdown`."""

    color: bool = True
    unicode: bool = True
    width: int = 80
    box_code: bool = True
    show_link_urls: bool = True
    max_code_lines: int = MAX_CODE_LINES
    indent: int = 0
    heading_prefix: str = ""

    def style(self, text: str, name: str) -> str:
        """Apply a theme colour when color is enabled."""
        return themed(text, name, enabled=self.color)


def inline(text: str, options: MarkdownOptions | None = None) -> str:
    """Render inline Markdown (bold, italic, code, links, strikethrough)."""
    opts = options or MarkdownOptions()
    out = text

    def _link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if not opts.show_link_urls or label.strip() == url.strip():
            return opts.style(label, "info")
        return f"{opts.style(label, 'info')} {opts.style('(' + url + ')', 'muted')}"

    out = LINK_RE.sub(_link, out)
    out = STRIKE_RE.sub(lambda m: opts.style(m.group(1), "muted"), out)
    out = CODE_RE.sub(lambda m: opts.style(m.group(1), "code"), out)
    out = BOLD_RE.sub(lambda m: opts.style(m.group(1) or m.group(2) or "", "heading"), out)
    out = ITALIC_RE.sub(lambda m: opts.style(m.group(1) or m.group(2) or "", "accent"), out)
    return out


def render_heading(level: int, text: str, options: MarkdownOptions | None = None) -> str:
    """Render one heading line."""
    opts = options or MarkdownOptions()
    marker = "#" * max(1, min(6, level))
    body = inline(text, opts)
    if level == 1:
        prefix = opts.heading_prefix or ""
        return prefix + opts.style(f"{marker} {text}", "heading")
    return opts.style(f"{marker} ", "muted") + body


def render_code_block(code: str, language: str = "", options: MarkdownOptions | None = None) -> str:
    """Render a fenced code block, boxed when the terminal is wide enough."""
    opts = options or MarkdownOptions()
    lines = code.rstrip("\n").split("\n")
    if len(lines) > opts.max_code_lines:
        head = lines[: opts.max_code_lines // 2]
        tail = lines[-opts.max_code_lines // 4 :]
        lines = head + [opts.style(f"... {len(lines) - len(head) - len(tail)} lines elided ...",
                                   "muted")] + tail
    body = "\n".join(opts.style(line, "code") if line.strip() else line for line in lines)
    if opts.box_code and opts.width >= 50:
        title = language or None
        return render_box(body, title=title, width=max(40, min(opts.width - 2, 100)),
                          unicode_borders=opts.unicode, color=opts.color)
    if language:
        return opts.style(f"[{language}]", "muted") + "\n" + body
    return body


def render_list(items: Sequence[tuple[int, str, str]], ordered: bool,
                options: MarkdownOptions | None = None) -> str:
    """Render collected list items.

    Args:
        items: ``(indent, marker, text)`` triples
        ordered: renumber sequentially
        options: rendering options
    """
    opts = options or MarkdownOptions()
    out: list[str] = []
    counters: dict[int, int] = {}
    for index, (indent, marker, text) in enumerate(items):
        depth = min(4, len(indent) // 2)
        if ordered:
            counters[depth] = counters.get(depth, 0) + 1
            bullet = opts.style(f"{counters[depth]}.", "brand")
        else:
            bullet = opts.style("•" if opts.unicode else "*", "brand") if depth == 0 else \
                opts.style("◦" if opts.unicode else "-", "muted")
        pad = "  " * (depth + opts.indent)
        body = inline(text, opts)
        if visible_width(body) > max(20, opts.width - len(pad) - 4):
            body = word_wrap(strip_ansi(body), max(20, opts.width - len(pad) - 4))
        out.append(f"{pad}{bullet} {body}")
        del index, marker
    return "\n".join(out)


def render_table(rows: Sequence[Sequence[str]], options: MarkdownOptions | None = None) -> str:
    """Render a Markdown table with aligned columns."""
    opts = options or MarkdownOptions()
    if not rows:
        return ""
    from ..utils.tables import render_table

    header, *body = rows
    return render_table([list(row) for row in body], headers=list(header), borders=True,
                        unicode_borders=opts.unicode, color=opts.color)


def _split_row(line: str) -> list[str]:
    """Split one Markdown table row into trimmed cells."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def render_markdown(text: str, *, color: bool = True, unicode: bool = True, width: int = 80,
                    box_code: bool = True, show_link_urls: bool = True, indent: int = 0) -> str:
    """Convert a Markdown document into terminal-friendly text.

    Args:
        text: Markdown source, typically a model answer
        color: emit ANSI styles
        unicode: use box drawing and typographic bullets
        width: target column count
        box_code: draw a border around code blocks
        show_link_urls: append the URL after a link label
        indent: extra indentation applied to list items
    """
    opts = MarkdownOptions(color=color, unicode=unicode, width=max(30, width),
                           box_code=box_code, show_link_urls=show_link_urls, indent=indent)
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    index = 0
    pending_list: list[tuple[int, str, str]] = []
    list_ordered = False
    pending_table: list[list[str]] = []

    def flush_list() -> None:
        if pending_list:
            out.append(render_list(pending_list, list_ordered, opts))
            pending_list.clear()

    def flush_table() -> None:
        if pending_table:
            out.append(render_table(pending_table, opts))
            pending_table.clear()

    while index < len(lines):
        line = lines[index]
        fence = FENCE_RE.match(line)
        if fence:
            flush_list()
            flush_table()
            delimiter = fence.group(1)
            language = fence.group(2)
            collected: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(delimiter):
                collected.append(lines[index])
                index += 1
            index += 1
            out.append(render_code_block("\n".join(collected), language, opts))
            continue
        if TABLE_ROW_RE.match(line):
            flush_list()
            if TABLE_SEP_RE.match(line) and pending_table:
                index += 1
                continue
            pending_table.append(_split_row(line))
            index += 1
            continue
        flush_table()
        heading = HEADING_RE.match(line)
        if heading:
            flush_list()
            out.append(render_heading(len(heading.group(1)), heading.group(2), opts))
            index += 1
            continue
        if RULE_RE.match(line):
            flush_list()
            out.append(opts.style("─" * min(width, 60) if opts.unicode else "-" * min(width, 60),
                                  "muted"))
            index += 1
            continue
        quote = QUOTE_RE.match(line)
        if quote:
            flush_list()
            body = inline(quote.group(1), opts)
            bar = opts.style("│ " if opts.unicode else "| ", "muted")
            out.append("\n".join(bar + wrapped for wrapped in
                                 word_wrap(strip_ansi(body), max(20, width - 4)).split("\n")))
            index += 1
            continue
        bullet = BULLET_RE.match(line)
        ordered = ORDERED_RE.match(line)
        if bullet or ordered:
            match = bullet or ordered
            assert match is not None
            if pending_list and bool(ordered) != list_ordered:
                flush_list()
            list_ordered = bool(ordered)
            pending_list.append((match.group(1), match.group(2), match.group(3)))
            index += 1
            continue
        flush_list()
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            index += 1
            continue
        wrapped = word_wrap(line, max(20, width))
        out.append("\n".join(inline(part, opts) for part in wrapped.split("\n")))
        index += 1
    flush_list()
    flush_table()
    rendered = "\n".join(out)
    while "\n\n\n" in rendered:
        rendered = rendered.replace("\n\n\n", "\n\n")
    return rendered.strip("\n")


def plain_text(text: str) -> str:
    """Strip Markdown syntax without adding color, for logs and files."""
    return strip_ansi(render_markdown(text, color=False, unicode=False, box_code=False,
                                      show_link_urls=False))


def looks_like_markdown(text: str) -> bool:
    """Heuristic: does this text contain Markdown worth rendering?"""
    if not text:
        return False
    markers = (HEADING_RE, FENCE_RE, BULLET_RE, ORDERED_RE, BOLD_RE, CODE_RE, LINK_RE, TABLE_ROW_RE)
    hits = sum(1 for pattern in markers if pattern.search(text))
    return hits >= 1


def summarize_markdown(text: str, *, limit: int = 200) -> str:
    """Return a one-line plain-text summary of a Markdown document."""
    return ellipsize(plain_text(text).replace("\n", " "), limit)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(language, code)`` pairs for every fenced block."""
    blocks: list[tuple[str, str]] = []
    lines = (text or "").split("\n")
    index = 0
    while index < len(lines):
        fence = FENCE_RE.match(lines[index])
        if not fence:
            index += 1
            continue
        delimiter, language = fence.group(1), fence.group(2)
        collected: list[str] = []
        index += 1
        while index < len(lines) and not lines[index].strip().startswith(delimiter):
            collected.append(lines[index])
            index += 1
        index += 1
        blocks.append((language, "\n".join(collected)))
    return blocks


def render_diff(diff: str, *, color: bool = True) -> str:
    """Colour a unified diff green/red for the terminal."""
    out: list[str] = []
    for line in (diff or "").split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            out.append(colorize(line, "success", enabled=color))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(colorize(line, "error", enabled=color))
        elif line.startswith("@@"):
            out.append(colorize(line, "info", enabled=color))
        elif line.startswith(("diff ", "index ", "---", "+++")):
            out.append(colorize(line, "muted", enabled=color))
        else:
            out.append(line)
    return "\n".join(out)


def bullet_list(items: Iterable[str], *, unicode: bool = True, color: bool = True) -> str:
    """Render a simple bullet list with a themed marker."""
    marker = themed("•" if unicode else "*", "brand", enabled=color)
    return "\n".join(f"{marker} {item}" for item in items)


__all__ = [
    "BOLD_RE",
    "BULLET_RE",
    "CODE_RE",
    "FENCE_RE",
    "HEADING_RE",
    "ITALIC_RE",
    "LINK_RE",
    "MAX_CODE_LINES",
    "ORDERED_RE",
    "QUOTE_RE",
    "RULE_RE",
    "STRIKE_RE",
    "TABLE_ROW_RE",
    "TABLE_SEP_RE",
    "MarkdownOptions",
    "bullet_list",
    "extract_code_blocks",
    "inline",
    "looks_like_markdown",
    "plain_text",
    "render_code_block",
    "render_diff",
    "render_heading",
    "render_list",
    "render_markdown",
    "render_table",
    "summarize_markdown",
]
