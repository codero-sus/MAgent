"""JSON parsing that survives what LLMs actually emit.

Language models wrap JSON in prose, use trailing commas, single quotes, Python
literals, unquoted keys, and comments. :func:`loads_lenient` and
:func:`repair_json` recover structured data from all of those without a
third-party dependency.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .text import find_balanced

_FENCE = re.compile(r"```(?:json|json5|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_PY_LITERALS = re.compile(r"\b(True|False|None)\b")
_SINGLE_QUOTES = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")
_NAN_INF = re.compile(r"\b(NaN|Infinity|-Infinity)\b")


def to_json(value: Any, *, indent: int | None = None, sort_keys: bool = False) -> str:
    """Serialize ``value`` to JSON, coercing unknown types via ``str()``."""
    return json.dumps(value, indent=indent, sort_keys=sort_keys, default=str, ensure_ascii=False)


def extract_json(text: str) -> str | None:
    """Pull the first plausible JSON document out of free-form model output.

    Handles fenced code blocks, leading prose, and balanced-brace scanning.
    """
    if not text:
        return None
    stripped = text.strip()
    fence = _FENCE.search(stripped)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()
    candidates: list[tuple[int, int]] = []
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = stripped.find(open_char)
        if start < 0:
            continue
        opened, closed = find_balanced(stripped, open_char, close_char, start)
        if opened >= 0 and closed > opened:
            candidates.append((opened, closed))
    if not candidates:
        return None
    # Whichever bracket opens first owns the document: scanning "{" before "["
    # would otherwise return the first object inside a JSON array and silently
    # drop the rest of the list.
    opened, closed = min(candidates)
    return stripped[opened : closed + 1]


def parse_json_block(text: str) -> Any:
    """Extract then parse the first JSON document found in ``text``."""
    candidate = extract_json(text)
    if candidate is None:
        raise ValueError("no JSON document found in text")
    return json.loads(candidate)


def repair_json(text: str) -> str:
    """Best-effort cleanup of malformed JSON emitted by a model.

    Applied fixes: strip comments, convert Python literals, quote bare keys,
    drop trailing commas, replace NaN/Infinity, and balance unclosed brackets.
    """
    cleaned = text.strip()
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    cleaned = _BLOCK_COMMENT.sub("", cleaned)
    cleaned = _LINE_COMMENT.sub("", cleaned)
    cleaned = _PY_LITERALS.sub(lambda m: {"True": "true", "False": "false", "None": "null"}[m.group(1)], cleaned)
    cleaned = _NAN_INF.sub(lambda m: {"NaN": "null", "Infinity": "1e999", "-Infinity": "-1e999"}[m.group(1)], cleaned)
    cleaned = _UNQUOTED_KEY.sub(r'\1"\2":', cleaned)
    cleaned = _TRAILING_COMMA.sub(r"\1", cleaned)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    open_braces = cleaned.count("{") - cleaned.count("}")
    open_brackets = cleaned.count("[") - cleaned.count("]")
    if cleaned.count('"') % 2 == 1:
        cleaned += '"'
    cleaned += "}" * max(0, open_braces)
    cleaned += "]" * max(0, open_brackets)
    return cleaned


def loads_lenient(text: str, *, default: Any = None) -> Any:
    """Parse JSON tolerantly; return ``default`` when recovery fails.

    Order of attempts: strict parse -> fenced/balanced extraction -> repair ->
    Python literal evaluation. Each step is cheap and short-circuits.
    """
    if text is None:
        return default
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return default
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    candidate = extract_json(stripped)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(candidate))
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(repair_json(stripped))
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return default
    return value if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))) else default


class JsonCodec:
    """Reusable encoder/decoder with a repair budget and statistics.

    Tracks how often strict parsing failed so you can see whether a model is
    reliably emitting valid JSON.
    """

    def __init__(self, *, allow_repair: bool = True, allow_python_literals: bool = True) -> None:
        self.allow_repair = allow_repair
        self.allow_python_literals = allow_python_literals
        self.attempts = 0
        self.strict_hits = 0
        self.repaired = 0
        self.failures = 0

    def decode(self, text: str, *, default: Any = None) -> Any:
        """Decode ``text``, updating the codec statistics."""
        self.attempts += 1
        try:
            value = json.loads(text)
            self.strict_hits += 1
            return value
        except (json.JSONDecodeError, TypeError):
            pass
        value = loads_lenient(text, default=None)
        if value is None and default is not None:
            self.failures += 1
            return default
        if value is None:
            self.failures += 1
            return default
        self.repaired += 1
        return value

    def encode(self, value: Any, *, indent: int | None = None) -> str:
        """Encode ``value`` to JSON text."""
        return to_json(value, indent=indent)

    @property
    def strict_rate(self) -> float:
        """Return the fraction of decodes that needed no repair."""
        return round(self.strict_hits / self.attempts, 4) if self.attempts else 0.0

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of codec statistics."""
        return {
            "attempts": self.attempts,
            "strict_hits": self.strict_hits,
            "repaired": self.repaired,
            "failures": self.failures,
            "strict_rate": self.strict_rate,
        }

    def reset(self) -> None:
        """Clear all statistics."""
        self.attempts = self.strict_hits = self.repaired = self.failures = 0


def flatten(data: Any, *, separator: str = ".") -> dict[str, Any]:
    """Flatten nested mappings into dotted-path keys."""
    flat: dict[str, Any] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}{separator}{key}" if prefix else str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{prefix}{separator}{index}" if prefix else str(index))
        else:
            flat[prefix] = value

    walk(data, "")
    return flat


def unflatten(data: dict[str, Any], *, separator: str = ".") -> dict[str, Any]:
    """Rebuild a nested mapping from dotted-path keys."""
    root: dict[str, Any] = {}
    for key, value in data.items():
        parts = str(key).split(separator)
        cursor: Any = root
        for part in parts[:-1]:
            if part.isdigit():
                part = str(int(part))
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                break
        if isinstance(cursor, dict):
            cursor[parts[-1]] = value
    return root


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def pick(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Return only the requested keys that are present."""
    return {key: data[key] for key in keys if key in data}


def omit(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Return a copy without the listed keys."""
    excluded = set(keys)
    return {key: value for key, value in data.items() if key not in excluded}


def jsonl_lines(items: list[Any]) -> str:
    """Serialize items as newline delimited JSON."""
    return "\n".join(to_json(item) for item in items) + ("\n" if items else "")


def parse_jsonl(text: str) -> list[Any]:
    """Parse newline delimited JSON, skipping blank and malformed lines."""
    out: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


__all__ = [
    "JsonCodec",
    "deep_merge",
    "extract_json",
    "flatten",
    "jsonl_lines",
    "loads_lenient",
    "omit",
    "parse_json_block",
    "parse_jsonl",
    "pick",
    "repair_json",
    "to_json",
    "unflatten",
]
