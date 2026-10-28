"""Text processing tools the model can call directly.

These give a local model real string manipulation without asking it to do
arithmetic or regex work in its head, which is where small models fail most
often. Every tool is pure, deterministic, and sandbox-free because it never
touches the filesystem or the network.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import tool
from ...utils.json_utils import extract_json, loads_lenient
from ...utils.text import (
    camel_case,
    chunk_text,
    collapse_whitespace,
    ellipsize,
    levenshtein,
    normalize_newlines,
    normalize_unicode,
    pascal_case,
    sentence_split,
    similarity,
    snake_case,
    squish,
    strip_accents,
    strip_markdown,
    title_case,
    word_wrap,
)


@tool(category="text", risk="safe", tags=["regex", "search"])
def regex_find(
    pattern: str,
    text: str,
    ignore_case: bool = False,
    multiline: bool = False,
    max_results: int = 100,
    group: int = 0,
) -> dict[str, Any]:
    """Find every match of a regular expression in a string.

    Args:
        pattern: regular expression to compile
        text: the text to search
        ignore_case: match case-insensitively
        multiline: make ^ and $ match line boundaries
        max_results: stop after this many matches
        group: capture group index to return, 0 for the whole match
    """
    flags = 0
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE | re.DOTALL
    try:
        matcher = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolExecutionError(f"Invalid regular expression: {exc}") from exc
    matches: list[dict[str, Any]] = []
    for index, match in enumerate(matcher.finditer(text)):
        if index >= max_results:
            break
        matches.append(
            {
                "index": index,
                "start": match.start(),
                "end": match.end(),
                "match": ellipsize(match.group(0), 500),
                "groups": [ellipsize(item or "", 200) for item in match.groups()],
                "named": {
                    key: ellipsize(value or "", 200) for key, value in match.groupdict().items()
                },
                "selected": ellipsize(
                    match.group(group) if group <= len(match.groups()) else match.group(0), 500
                ),
            }
        )
    return {
        "pattern": pattern,
        "count": len(matches),
        "matches": matches,
        "truncated": len(matches) >= max_results,
    }


@tool(category="text", risk="safe", tags=["regex", "replace"])
def regex_replace(
    pattern: str, replacement: str, text: str, count: int = 0, ignore_case: bool = False
) -> dict[str, Any]:
    """Replace regular expression matches in a string.

    Args:
        pattern: regular expression to find
        replacement: replacement text, may use \\1 group references
        text: the text to modify
        count: maximum replacements, 0 for all
        ignore_case: match case-insensitively
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        updated, replaced = re.subn(pattern, replacement, text, count=count, flags=flags)
    except re.error as exc:
        raise ToolExecutionError(f"Invalid regular expression: {exc}") from exc
    return {
        "replacements": replaced,
        "text": updated,
        "changed": replaced > 0,
        "before_chars": len(text),
        "after_chars": len(updated),
    }


@tool(category="text", risk="safe", tags=["stats", "count"])
def text_stats(text: str) -> dict[str, Any]:
    """Report character, word, line, and sentence counts for a string.

    Args:
        text: the text to analyze
    """
    normalized = normalize_newlines(text)
    words = normalized.split()
    sentences = sentence_split(normalized)
    unique = set(word.lower().strip(".,;:!?\"'()[]{}") for word in words)
    return {
        "chars": len(text),
        "chars_no_whitespace": len(re.sub(r"\s", "", text)),
        "words": len(words),
        "unique_words": len(unique),
        "lines": normalized.count("\n")
        + (1 if normalized and not normalized.endswith("\n") else 0),
        "sentences": len(sentences),
        "paragraphs": len([part for part in normalized.split("\n\n") if part.strip()]),
        "avg_word_length": round(sum(len(word) for word in words) / len(words), 2)
        if words
        else 0.0,
        "avg_sentence_length": round(len(words) / len(sentences), 2) if sentences else 0.0,
        "estimated_tokens": max(1, len(text) // 4),
        "longest_word": max(words, key=len) if words else "",
        "top_words": _top_words(words, 10),
    }


def _top_words(words: list[str], limit: int) -> list[dict[str, Any]]:
    """Return the most frequent non-stopword tokens."""
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "doing",
        "of",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "as",
        "not",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
    }
    counts: dict[str, int] = {}
    for word in words:
        cleaned = word.lower().strip(".,;:!?\"'()[]{}")
        if not cleaned or cleaned in stopwords or cleaned.isdigit():
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"word": word, "count": count} for word, count in ranked]


@tool(category="text", risk="safe", tags=["transform"])
def transform_text(text: str, operation: str = "strip", width: int = 80) -> dict[str, Any]:
    """Apply one named transformation to a string.

    Args:
        text: input text
        operation: one of strip, lower, upper, title, snake, camel, pascal, squish,
            wrap, unwrap, dedent, unaccent, normalize, markdown_strip, reverse, sort_lines,
            dedupe_lines, number_lines
        width: column width for the wrap operation
    """
    operations = {
        "strip": lambda value: value.strip(),
        "lower": str.lower,
        "upper": str.upper,
        "title": title_case,
        "snake": snake_case,
        "camel": camel_case,
        "pascal": pascal_case,
        "squish": squish,
        "collapse": collapse_whitespace,
        "wrap": lambda value: word_wrap(value, width),
        "unwrap": lambda value: re.sub(r"\s*\n\s*", " ", value).strip(),
        "unaccent": strip_accents,
        "normalize": lambda value: normalize_unicode(value),
        "markdown_strip": strip_markdown,
        "reverse": lambda value: value[::-1],
        "sort_lines": lambda value: "\n".join(sorted(value.split("\n"))),
        "dedupe_lines": lambda value: "\n".join(dict.fromkeys(value.split("\n"))),
        "number_lines": lambda value: "\n".join(
            f"{index:>{len(str(len(value.split(chr(10)))))}}│{line}"
            for index, line in enumerate(value.split("\n"), start=1)
        ),
        "sentences": lambda value: "\n".join(sentence_split(value)),
        "chunks": lambda value: "\n\n---\n\n".join(chunk_text(value, max_chars=max(64, width * 8))),
    }
    if operation not in operations:
        raise ToolExecutionError(
            f"Unknown operation {operation!r}; choose from {', '.join(sorted(operations))}"
        )
    result = operations[operation](text)
    if isinstance(result, list):
        result = "\n".join(result)
    return {
        "operation": operation,
        "before_chars": len(text),
        "after_chars": len(result),
        "text": result,
    }


@tool(category="text", risk="safe", tags=["compare", "similarity"])
def compare_text(left: str, right: str) -> dict[str, Any]:
    """Compare two strings and report similarity and edit distance.

    Args:
        left: first string
        right: second string
    """
    distance = levenshtein(left, right)
    return {
        "identical": left == right,
        "edit_distance": distance,
        "similarity": similarity(left, right),
        "left_chars": len(left),
        "right_chars": len(right),
        "char_delta": len(right) - len(left),
        "left_words": len(left.split()),
        "right_words": len(right.split()),
        "shared_words": len(set(left.lower().split()) & set(right.lower().split())),
    }


@tool(category="text", risk="safe", tags=["truncate", "extract"])
def extract_lines(
    text: str,
    start: int = 1,
    end: int = 0,
    grep: str = "",
    ignore_case: bool = True,
    max_lines: int = 500,
) -> dict[str, Any]:
    """Slice or filter lines out of a larger string.

    Args:
        text: the source text
        start: first line to keep, 1-based
        end: last line to keep, 0 for the end of the text
        grep: only keep lines containing this substring
        ignore_case: make the grep filter case-insensitive
        max_lines: maximum number of lines to return
    """
    lines = normalize_newlines(text).split("\n")
    first = max(0, start - 1)
    last = end or len(lines)
    selected = lines[first:last]
    if grep:
        needle = grep.lower() if ignore_case else grep
        selected = [line for line in selected if needle in (line.lower() if ignore_case else line)]
    truncated = len(selected) > max_lines
    return {
        "total_lines": len(lines),
        "selected": len(selected[:max_lines]),
        "truncated": truncated,
        "text": "\n".join(selected[:max_lines]),
    }


@tool(category="text", risk="safe", tags=["encode", "hash"])
def encode_text(text: str, operation: str = "base64_encode") -> dict[str, Any]:
    """Encode, decode, or hash a string.

    Args:
        text: input text, or encoded input for the decode operations
        operation: base64_encode, base64_decode, url_encode, url_decode, sha256, md5, hex, unhex
    """
    from urllib.parse import quote, unquote

    try:
        if operation == "base64_encode":
            result = base64.b64encode(text.encode("utf-8")).decode("ascii")
        elif operation == "base64_decode":
            result = base64.b64decode(text.encode("ascii")).decode("utf-8", "replace")
        elif operation == "url_encode":
            result = quote(text, safe="")
        elif operation == "url_decode":
            result = unquote(text)
        elif operation == "sha256":
            result = hashlib.sha256(text.encode("utf-8")).hexdigest()
        elif operation == "md5":
            result = hashlib.md5(text.encode("utf-8")).hexdigest()
        elif operation == "hex":
            result = text.encode("utf-8").hex()
        elif operation == "unhex":
            result = bytes.fromhex(text.strip()).decode("utf-8", "replace")
        else:
            raise ToolExecutionError(f"Unknown operation {operation!r}")
    except (ValueError, TypeError) as exc:
        raise ToolExecutionError(f"{operation} failed: {exc}") from exc
    return {
        "operation": operation,
        "before_chars": len(text),
        "after_chars": len(result),
        "text": result,
    }


@tool(category="text", risk="safe", tags=["join", "split"])
def split_text(
    text: str,
    separator: str = "\n",
    max_parts: int = 100,
    strip: bool = True,
    drop_empty: bool = True,
) -> dict[str, Any]:
    """Split a string into parts.

    Args:
        text: input text
        separator: delimiter to split on
        max_parts: maximum number of parts to return
        strip: trim whitespace from each part
        drop_empty: discard parts that are empty after stripping
    """
    parts = text.split(separator)
    if strip:
        parts = [part.strip() for part in parts]
    if drop_empty:
        parts = [part for part in parts if part]
    total = len(parts)
    return {
        "parts": parts[:max_parts],
        "count": total,
        "returned": min(total, max_parts),
        "truncated": total > max_parts,
        "separator": separator or "\\n",
    }


@tool(category="text", risk="safe", tags=["json", "extract"])
def extract_json_from_text(text: str, repair: bool = True) -> dict[str, Any]:
    """Pull the first JSON document out of prose and parse it.

    Args:
        text: model output or any text containing JSON
        repair: attempt to fix trailing commas, bare keys, and Python literals
    """
    candidate = extract_json(text)
    if candidate is None:
        return {"found": False, "value": None, "reason": "no JSON document found"}
    value: Any = None
    if repair:
        value = loads_lenient(candidate)
    else:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = None
    if value is None:
        return {"found": True, "parsed": False, "raw": ellipsize(candidate, 2000)}
    return {
        "found": True,
        "parsed": True,
        "value": value,
        "type": type(value).__name__,
        "keys": sorted(value) if isinstance(value, dict) else None,
        "items": len(value) if isinstance(value, (dict, list)) else None,
    }


@tool(category="text", risk="safe", tags=["summarize", "extractive"])
def extractive_summary(text: str, sentences: int = 3) -> dict[str, Any]:
    """Build a short extractive summary by scoring sentence word frequency.

    This is a deterministic, offline summarizer. It is useful for previews and
    for cases where the model is unavailable; for real summaries use the agent.

    Args:
        text: the document to summarize
        sentences: how many sentences to keep
    """
    parts = sentence_split(text)
    if not parts:
        return {"summary": "", "sentences": 0}
    words = [word.lower().strip(".,;:!?\"'()[]{}") for word in re.findall(r"[A-Za-z']+", text)]
    counts = _top_words(words, 40)
    weights = {item["word"]: item["count"] for item in counts}
    scored = []
    for index, sentence in enumerate(parts):
        tokens = [word.lower().strip(".,;:!?") for word in sentence.split()]
        score = sum(weights.get(token, 0) for token in tokens) / max(1, len(tokens))
        position_bonus = 1.2 if index < 2 else 1.0
        scored.append((score * position_bonus, index, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[: max(1, sentences)], key=lambda item: item[1])
    summary = " ".join(item[2] for item in chosen)
    return {
        "summary": summary,
        "sentences": len(chosen),
        "source_sentences": len(parts),
        "compression": round(len(summary) / max(1, len(text)), 3),
        "keywords": [item["word"] for item in counts[:8]],
    }


TOOLS: list[Any] = [
    regex_find,
    regex_replace,
    text_stats,
    transform_text,
    compare_text,
    extract_lines,
    encode_text,
    split_text,
    extract_json_from_text,
    extractive_summary,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every text tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "compare_text",
    "encode_text",
    "extract_json_from_text",
    "extract_lines",
    "extractive_summary",
    "load",
    "regex_find",
    "regex_replace",
    "split_text",
    "text_stats",
    "transform_text",
]
