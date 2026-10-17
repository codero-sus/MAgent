"""Structured output formats for CLI and API payloads.

``magent`` commands all accept ``--format``.  Rather than each command growing
its own ``if fmt == "json"`` ladder, values are funnelled through
:func:`render_value`, which knows how to turn dicts, dataclasses, lists and
plain strings into ``text``, ``json``, ``jsonl``, ``yaml``, ``csv``, ``md`` or
``quiet`` output.  The YAML writer is a small emitter of our own - the package has
no dependencies, and pulling PyYAML in for a debug view is not worth it.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.json_utils import to_json

FORMATS: tuple[str, ...] = ("text", "json", "jsonl", "yaml", "csv", "md", "quiet", "raw")
"""Output formats every command accepts."""

DEFAULT_FORMAT = "text"
"""Format used when the caller does not pick one."""

YAML_INDENT = "  "
"""Indentation used by the built-in YAML emitter."""


def normalize_format(name: str | None) -> str:
    """Map user input onto a known format, tolerating aliases."""
    value = (name or DEFAULT_FORMAT).strip().lower()
    aliases = {
        "markdown": "md",
        "table": "text",
        "plain": "text",
        "yml": "yaml",
        "ndjson": "jsonl",
        "lines": "jsonl",
        "none": "quiet",
        "silent": "quiet",
        "string": "raw",
    }
    value = aliases.get(value, value)
    return value if value in FORMATS else DEFAULT_FORMAT


CYCLE_MARKER = "<recursive>"
"""Placeholder emitted when a value refers back to itself."""

MAX_PLAIN_DEPTH = 32
"""Depth ceiling for :func:`to_plain`, so a deep structure cannot blow the stack."""


def to_plain(value: Any, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Convert dataclasses, enums and paths into JSON-friendly primitives.

    Cycles are cut rather than followed: a run's final event carries the run
    itself, so the naive walk recursed until Python raised ``RecursionError``.
    Objects that define ``to_dict`` are asked for their own representation
    first, because that is the shape their author intended to serialise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if _depth >= MAX_PLAIN_DEPTH:
        return CYCLE_MARKER
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return CYCLE_MARKER
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return {str(key): to_plain(item, seen, _depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [to_plain(item, seen, _depth + 1) for item in value]
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            try:
                return to_plain(converter(), seen, _depth + 1)
            except TypeError:
                pass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                key: to_plain(getattr(value, key), seen, _depth + 1)
                for key in value.__dataclass_fields__
            }
        if hasattr(value, "__dict__"):
            return {
                key: to_plain(item, seen, _depth + 1)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return str(value)
    finally:
        seen.discard(identity)


def to_yaml(value: Any, *, indent: int = 0) -> str:
    """Emit a small, valid YAML document for nested primitives.

    Handles the shapes MAgent actually produces - dicts of scalars, lists of
    dicts, nested dicts - and quotes strings that would otherwise be ambiguous.
    """
    plain = to_plain(value)
    pad = YAML_INDENT * indent
    lines: list[str] = []
    if isinstance(plain, Mapping):
        if not plain:
            return pad + "{}"
        for key, item in plain.items():
            if isinstance(item, Mapping):
                if item:
                    lines.append(f"{pad}{_yaml_key(key)}:")
                    lines.append(to_yaml(item, indent=indent + 1))
                else:
                    lines.append(f"{pad}{_yaml_key(key)}: {{}}")
            elif isinstance(item, (list, tuple)):
                if item:
                    lines.append(f"{pad}{_yaml_key(key)}:")
                    lines.extend(_yaml_list(list(item), indent + 1))
                else:
                    lines.append(f"{pad}{_yaml_key(key)}: []")
            else:
                lines.append(f"{pad}{_yaml_key(key)}: {_yaml_scalar(item)}")
        return "\n".join(line for line in lines if line)
    if isinstance(plain, (list, tuple)):
        return "\n".join(_yaml_list(list(plain), indent))
    return pad + _yaml_scalar(plain)


def _yaml_key(key: str) -> str:
    """Quote a mapping key when it is not a plain identifier."""
    if key and all(char.isalnum() or char in "_-." for char in key) and not key[0].isdigit():
        return key
    return json.dumps(str(key))


def _yaml_scalar(value: Any) -> str:
    """Render one scalar in YAML syntax."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    if text == "":
        return '""'
    if "\n" in text:
        body = "\n".join(YAML_INDENT + line for line in text.split("\n"))
        return "|-\n" + body
    needs_quotes = (
        text[0] in "!&*?|>%@`#-[]{},\"'"
        or text.strip() != text
        or ": " in text
        or " #" in text
        or text.lower() in {"true", "false", "null", "yes", "no", "on", "off"}
        or _is_number_like(text)
    )
    return json.dumps(text) if needs_quotes else text


def _is_number_like(text: str) -> bool:
    """True when a string would be parsed as a number by a YAML reader."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _yaml_list(items: Sequence[Any], indent: int) -> list[str]:
    """Render a YAML sequence, inlining scalars and nesting mappings."""
    pad = YAML_INDENT * indent
    out: list[str] = []
    for item in items:
        plain = to_plain(item)
        if isinstance(plain, Mapping) and plain:
            nested = to_yaml(plain, indent=indent + 1).split("\n")
            first = nested[0].lstrip()
            out.append(f"{pad}- {first}")
            out.extend(nested[1:])
        elif isinstance(plain, (list, tuple)) and plain:
            out.append(f"{pad}-")
            out.extend(_yaml_list(list(plain), indent + 1))
        else:
            out.append(f"{pad}- {_yaml_scalar(plain)}")
    return out


def to_csv(value: Any, *, delimiter: str = ",") -> str:
    """Render a list of mappings (or a single mapping) as CSV."""
    plain = to_plain(value)
    rows: list[dict[str, Any]] = []
    if isinstance(plain, Mapping):
        rows = [plain]
    elif isinstance(plain, list):
        for item in plain:
            if isinstance(item, Mapping):
                rows.append(dict(item))
            else:
                rows.append({"value": item})
    if not rows:
        return ""
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_csv_cell(row.get(header, "")) for header in headers])
    return buffer.getvalue().rstrip("\n")


def _csv_cell(value: Any) -> str:
    """Flatten one CSV cell to a string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def to_text(value: Any, *, width: int = 0) -> str:
    """Render a value as human-readable text.

    Mappings become ``key: value`` lines (nested mappings are indented), lists
    become bullet lines, and scalars are returned as-is.
    """
    plain = to_plain(value)
    if isinstance(plain, str):
        return plain
    if isinstance(plain, Mapping):
        return _text_mapping(plain, 0, width)
    if isinstance(plain, (list, tuple)):
        return "\n".join(_text_item(item, 0, width) for item in plain)
    if plain is None:
        return ""
    return str(plain)


def _text_mapping(mapping: Mapping[str, Any], depth: int, width: int) -> str:
    """Render a mapping as aligned ``key: value`` lines."""
    pad = YAML_INDENT * depth
    if not mapping:
        return pad + "(empty)"
    key_width = max(len(str(key)) for key in mapping)
    if width:
        key_width = min(key_width, max(8, width // 3))
    lines: list[str] = []
    for key, item in mapping.items():
        label = str(key).ljust(key_width)
        if isinstance(item, Mapping) and item:
            lines.append(f"{pad}{label}:")
            lines.append(_text_mapping(item, depth + 1, width))
        elif isinstance(item, (list, tuple)) and item:
            lines.append(f"{pad}{label}:")
            lines.extend(_text_list(list(item), depth + 1, width))
        else:
            lines.append(f"{pad}{label}: {_inline_scalar(item, width)}")
    return "\n".join(lines)


def _text_list(items: Sequence[Any], depth: int, width: int) -> list[str]:
    """Render a list as bullet lines."""
    pad = YAML_INDENT * depth
    out: list[str] = []
    for item in items:
        out.append(_text_item(item, depth, width, bullet=True, pad_override=pad))
    return out


def _text_item(
    item: Any, depth: int, width: int, *, bullet: bool = False, pad_override: str = ""
) -> str:
    """Render one list element."""
    pad = pad_override or YAML_INDENT * depth
    prefix = f"{pad}- " if bullet else pad
    plain = to_plain(item)
    if isinstance(plain, Mapping):
        body = _text_mapping(plain, depth + (1 if bullet else 0), width)
        first, _, rest = body.partition("\n")
        return f"{prefix}{first}" + (f"\n{rest}" if rest else "")
    if isinstance(plain, (list, tuple)):
        body = "\n".join(_text_list(list(plain), depth + 1, width))
        return f"{prefix.rstrip()}\n{body}" if body else prefix
    return f"{prefix}{_inline_scalar(plain, width)}"


def _inline_scalar(value: Any, width: int) -> str:
    """Render a scalar on one line, eliding over-long strings."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    if width and len(text) > width:
        return text[: max(1, width - 1)] + "…"
    return text.replace("\n", " ")


def render_value(
    value: Any,
    fmt: str = DEFAULT_FORMAT,
    *,
    width: int = 0,
    indent: int = 2,
    sort_keys: bool = False,
) -> str:
    """Render any value in one of :data:`FORMATS`.

    Args:
        value: the payload
        fmt: output format name, aliases accepted
        width: text column budget
        indent: JSON/YAML indentation
        sort_keys: sort mapping keys in JSON output
    """
    active = normalize_format(fmt)
    if active == "quiet":
        return ""
    if active == "raw":
        return value if isinstance(value, str) else str(value)
    if active == "json":
        return to_json(to_plain(value), indent=indent, sort_keys=sort_keys)
    if active == "jsonl":
        return render_jsonl(value, sort_keys=sort_keys)
    if active == "yaml":
        return to_yaml(value)
    if active == "csv":
        return to_csv(value)
    if active == "md":
        return to_markdown(value)
    return to_text(value, width=width)


def render_jsonl(value: Any, *, sort_keys: bool = False) -> str:
    """Render a sequence of records as one JSON object per line."""
    plain = to_plain(value)
    items = plain if isinstance(plain, list) else [plain]
    return "\n".join(json.dumps(item, default=str, sort_keys=sort_keys) for item in items)


def to_markdown(value: Any, *, title: str = "") -> str:
    """Render a value as Markdown: a table for records, bullets otherwise."""
    plain = to_plain(value)
    lines: list[str] = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    if isinstance(plain, Mapping):
        rows = [item for item in plain.values() if isinstance(item, Mapping)]
        if rows and all(set(rows[0]) == set(row) for row in rows):
            for key, item in plain.items():
                if isinstance(item, list) and item and all(isinstance(x, Mapping) for x in item):
                    lines.append(f"### {key}")
                    lines.append("")
                    lines.append(_markdown_table([to_plain(x) for x in item]))
                    lines.append("")
                elif isinstance(item, Mapping):
                    lines.append(f"### {key}")
                    lines.append("")
                    lines.append(_markdown_table([item], title=str(key)))
                    lines.append("")
                else:
                    lines.append(f"- **{key}**: {_inline_scalar(to_plain(item), 0)}")
            return "\n".join(lines).rstrip()
        lines.append(_markdown_table([plain]))
        return "\n".join(lines)
    if isinstance(plain, list) and plain and all(isinstance(item, Mapping) for item in plain):
        lines.append(_markdown_table(plain))
        return "\n".join(lines)
    if isinstance(plain, list):
        lines.extend(f"- {_inline_scalar(item, 0)}" for item in plain)
        return "\n".join(lines)
    lines.append(str(plain))
    return "\n".join(lines)


def _markdown_table(records: Sequence[Mapping[str, Any]], *, title: str = "") -> str:
    """Render records as a GitHub-flavoured Markdown table."""
    if not records:
        return title or ""
    headers: list[str] = []
    for record in records:
        for key in record:
            if key not in headers:
                headers.append(str(key))
    out: list[str] = []
    if title:
        out.append(f"### {title}")
        out.append("")
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for record in records:
        cells = [_markdown_cell(record.get(header, "")) for header in headers]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _markdown_cell(value: Any) -> str:
    """Escape one table cell."""
    text = _inline_scalar(to_plain(value), 0)
    return text.replace("|", "\\|").replace("\n", " ")


def parse_format_choice(text: str) -> str:
    """Validate a ``--format`` argument, raising a helpful error on typos."""
    from ..errors import ConfigError

    value = normalize_format(text)
    if value == DEFAULT_FORMAT and text and text.strip().lower() not in FORMATS:
        lowered = text.strip().lower()
        if lowered not in {"text", "table", "plain"}:
            raise ConfigError(f"unknown format {text!r}", details={"allowed": list(FORMATS)})
    return value


def write_formatted(value: Any, sink: Any, fmt: str = DEFAULT_FORMAT, **kwargs: Any) -> int:
    """Render a value and write it to a sink; returns characters written."""
    text = render_value(value, fmt, **kwargs)
    if not text:
        return 0
    sink.write(text if text.endswith("\n") else text + "\n")
    return len(text)


def dump_formats(value: Any) -> dict[str, str]:
    """Render a value in every format; used by tests and ``magent config show``."""
    return {name: render_value(value, name) for name in FORMATS if name != "quiet"}


def iter_records(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield mapping records from a payload, wrapping scalars."""
    plain = to_plain(value)
    if isinstance(plain, Mapping):
        yield plain
        return
    if isinstance(plain, list):
        for item in plain:
            if isinstance(item, Mapping):
                yield item
            else:
                yield {"value": item}


__all__ = [
    "CYCLE_MARKER",
    "DEFAULT_FORMAT",
    "FORMATS",
    "MAX_PLAIN_DEPTH",
    "YAML_INDENT",
    "dump_formats",
    "iter_records",
    "normalize_format",
    "parse_format_choice",
    "render_jsonl",
    "render_value",
    "to_csv",
    "to_markdown",
    "to_plain",
    "to_text",
    "to_yaml",
    "write_formatted",
]
