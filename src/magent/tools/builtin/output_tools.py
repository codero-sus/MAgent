"""Output, formatting, and reporting tools.

These let an agent produce structured deliverables - Markdown reports, tables,
charts rendered in ASCII, and files written to the workspace - instead of
returning prose that has to be post-processed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.files import atomic_write_text, ensure_dir, human_size
from ...utils.json_utils import loads_lenient
from ...utils.tables import (
    render_bar_chart,
    render_box,
    render_key_value,
    render_markdown_table,
    render_progress,
    render_table,
)


@tool(category="output", risk="safe", tags=["format", "table", "core"])
def format_table(data: str, columns: str = "", style: str = "markdown",
                 title: str = "", align: str = "") -> dict[str, Any]:
    """Render records as a table in one of several styles.

    Args:
        data: a JSON array of objects, a JSON array of arrays, or CSV text
        columns: column order for object input, empty to infer
        style: markdown, box, plain, or csv
        title: optional caption rendered above the table
        align: per-column alignment letters such as "lrr"
    """
    rows, header = _coerce_rows(data, columns)
    alignments = list(align) if align else None
    styles = {"markdown", "box", "plain", "csv", "ascii"}
    if style not in styles:
        raise ToolExecutionError(f"style must be one of {sorted(styles)}")
    if style == "markdown":
        text = render_markdown_table(rows, header)
    elif style == "box":
        text = render_table(rows, header, alignments=alignments, borders=True, unicode_borders=True)
    elif style == "csv":
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        if header:
            writer.writerow(header)
        writer.writerows(rows)
        text = buffer.getvalue()
    elif style == "ascii":
        text = render_table(rows, header, alignments=alignments, borders=True, unicode_borders=False)
    else:
        text = render_table(rows, header, alignments=alignments, borders=False)
    if title:
        text = f"{title}\n{text}"
    return {"style": style, "rows": len(rows), "columns": header, "text": text,
            "chars": len(text), "lines": text.count("\n") + 1}


def _coerce_rows(data: str, columns: str) -> tuple[list[list[str]], list[str]]:
    """Turn JSON or CSV text into ``(rows, header)``."""
    parsed = loads_lenient(data)
    if isinstance(parsed, dict):
        for key in ("data", "items", "results", "records", "rows"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]
    header = [name.strip() for name in columns.split(",") if name.strip()]
    if isinstance(parsed, list) and parsed:
        if isinstance(parsed[0], dict):
            header = header or list(dict.fromkeys(key for item in parsed if isinstance(item, dict) for key in item))
            rows = [[_cell(item.get(name, "")) for name in header] for item in parsed if isinstance(item, dict)]
            return rows, header
        rows = [[_cell(value) for value in item] if isinstance(item, list) else [_cell(item)] for item in parsed]
        width = max((len(row) for row in rows), default=0)
        header = header or [f"col_{index}" for index in range(width)]
        rows = [row + [""] * (len(header) - len(row)) for row in rows]
        return rows, header
    if isinstance(parsed, str) or parsed is None:
        import csv
        import io

        reader = list(csv.reader(io.StringIO(data)))
        if not reader:
            raise ToolExecutionError("No tabular data supplied")
        header = header or [name.strip() for name in reader[0]]
        rows = [[_cell(value) for value in row] for row in reader[1:]]
        return rows, header
    raise ToolExecutionError("data must be a JSON array, object, or CSV text")


def _cell(value: Any) -> str:
    """Render one table cell as text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


@tool(category="output", risk="safe", tags=["chart", "visualize"])
def render_chart(values: str, labels: str = "", style: str = "bar", width: int = 40,
                 title: str = "") -> dict[str, Any]:
    """Render an ASCII bar chart from numeric values.

    Args:
        values: comma separated numbers or a JSON array
        labels: comma separated labels matching the values
        style: bar or progress
        width: character width of the longest bar
        title: caption rendered above the chart
    """
    numbers: list[float] = []
    for chunk in str(values).replace("[", " ").replace("]", " ").replace(",", " ").split():
        chunk = chunk.strip().strip('"')
        if not chunk:
            continue
        try:
            numbers.append(float(chunk))
        except ValueError:
            raise ToolExecutionError(f"Not a number: {chunk!r}") from None
    if not numbers:
        raise ToolExecutionError("No numeric values supplied")
    names = [item.strip().strip('"') for item in str(labels).replace("[", " ").replace("]", " ").split(",")] if labels else []
    names = names or [f"#{index + 1}" for index in range(len(numbers))]
    names = (names + [f"#{index + 1}" for index in range(len(numbers))])[: len(numbers)]
    if style == "progress":
        total = max(numbers) or 1.0
        rows = [[name, render_progress(value, total, width=width)]
                for name, value in zip(names, numbers, strict=False)]
        text = render_table(rows, ["label", f"progress (max {max(numbers):g})"], borders=False)
    else:
        text = render_bar_chart(dict(zip(names, numbers, strict=False)), width=width)
    if title:
        text = f"{title}\n{text}"
    return {"style": style, "points": len(numbers), "max": max(numbers), "min": min(numbers),
            "sum": round(sum(numbers), 6), "text": text}


@tool(category="output", risk="safe", tags=["markdown", "report", "core"])
def build_report(title: str, sections: str, summary: str = "", footer: str = "",
                 include_toc: bool = True) -> dict[str, Any]:
    """Assemble a Markdown report from titled sections.

    Args:
        title: report title, rendered as an H1
        sections: JSON array of {"heading": ..., "body": ...} objects
        summary: optional executive summary rendered under the title
        footer: optional closing text
        include_toc: add a table of contents with anchor links
    """
    parsed = loads_lenient(sections)
    if not isinstance(parsed, list):
        raise ToolExecutionError("sections must be a JSON array")
    from ...utils.text import slugify

    lines: list[str] = [f"# {title}", ""]
    if summary:
        lines += [summary, ""]
    entries: list[tuple[str, str]] = []
    for section in parsed:
        if isinstance(section, str):
            heading, body = "", section
        elif isinstance(section, dict):
            heading = str(section.get("heading") or section.get("title") or "")
            body = str(section.get("body") or section.get("content") or "")
        else:
            raise ToolExecutionError("Each section must be a string or an object")
        entries.append((heading, body))
    if include_toc and any(heading for heading, _ in entries):
        lines += ["## Contents", ""]
        for heading, _ in entries:
            if heading:
                lines.append(f"- [{heading}](#{slugify(heading)})")
        lines.append("")
    for heading, body in entries:
        if heading:
            lines += [f"## {heading}", ""]
        lines += [body.rstrip(), ""]
    if footer:
        lines += ["---", "", footer, ""]
    markdown = "\n".join(lines).rstrip() + "\n"
    return {
        "markdown": markdown,
        "chars": len(markdown),
        "words": len(markdown.split()),
        "sections": len(entries),
        "headings": [heading for heading, _ in entries if heading],
    }


@tool(category="output", risk="write", filesystem=True, tags=["write", "artifact"])
def write_artifact(path: str, content: str, mode: str = "text", append: bool = False,
                   create_dirs: bool = True, context: ToolContext | None = None) -> dict[str, Any]:
    """Write a deliverable file into the workspace.

    Args:
        path: destination path, relative to the workspace
        content: text content, or JSON when mode is "json"
        mode: text, json, or markdown
        append: append instead of replacing the file
        create_dirs: create missing parent directories
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_write(target)
    if mode == "json":
        parsed = loads_lenient(content)
        if parsed is None:
            raise ToolExecutionError("mode=json requires valid JSON content")
        text = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
    else:
        text = content
    if ctx.dry_run:
        return {"path": str(target), "dry_run": True, "bytes": len(text.encode()), "mode": mode}
    if create_dirs:
        ensure_dir(target.parent)
    if append and target.is_file():
        with target.open("a", encoding="utf-8") as stream:
            stream.write(text)
        written = target.stat().st_size
        action = "appended"
    else:
        atomic_write_text(target, text)
        written = len(text.encode())
        action = "wrote"
    return {
        "action": action,
        "path": str(target),
        "relative": ctx.sandbox.relative(target) if ctx.sandbox else str(target),
        "bytes": written,
        "size_human": human_size(written),
        "mode": mode,
        "lines": text.count("\n") + (1 if text else 0),
    }


@tool(category="output", risk="safe", tags=["box", "format"])
def render_box_tool(text: str, title: str = "", style: str = "rounded", width: int = 60) -> dict[str, Any]:
    """Draw text inside a Unicode box, useful for highlighting a finding.

    Args:
        text: the text to frame
        title: optional title shown in the box border
        style: rounded, square, double, or bold
        width: box width in characters
    """
    framed = render_box(text, title=title, style=style, width=max(10, min(width, 200)))
    return {"text": framed, "lines": framed.count("\n") + 1, "style": style, "width": width}


@tool(category="output", risk="safe", tags=["keyvalue", "format"])
def render_key_value_tool(pairs: str, title: str = "", sort: bool = False) -> dict[str, Any]:
    """Render a two-column key/value listing with aligned colons.

    Args:
        pairs: JSON object, or "key: value" lines
        title: caption rendered above the listing
        sort: sort keys alphabetically
    """
    parsed = loads_lenient(pairs)
    if isinstance(parsed, dict):
        items = list(parsed.items())
    else:
        items = []
        for line in str(pairs).splitlines():
            if not line.strip():
                continue
            key, _, value = line.partition(":")
            items.append((key.strip(), value.strip()))
    if sort:
        items.sort(key=lambda pair: str(pair[0]).lower())
    text = render_key_value([(str(key), value) for key, value in items])
    if title:
        text = f"{title}\n{text}"
    return {"text": text, "pairs": len(items), "lines": text.count("\n") + 1}


TOOLS: list[Any] = [
    format_table, render_chart, build_report, write_artifact, render_box_tool, render_key_value_tool,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every output tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "build_report",
    "format_table",
    "load",
    "render_box_tool",
    "render_chart",
    "render_key_value_tool",
    "write_artifact",
]
