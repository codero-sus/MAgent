"""Text processing helpers: normalization, chunking, similarity, wrapping."""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections.abc import Iterable, Iterator

_WHITESPACE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_WORD_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z0-9\"'(\[])")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_MARKDOWN_NOISE = re.compile(
    r"(\*\*|__|\*|_|`{1,3}|~~|>\s?|#{1,6}\s|\[([^\]]*)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\))"
)


def normalize_newlines(text: str) -> str:
    """Convert CRLF and CR line endings to plain LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace(text: str) -> str:
    """Squeeze runs of horizontal whitespace into single spaces."""
    return _WHITESPACE.sub(" ", text)


def squish(text: str) -> str:
    """Normalize newlines, collapse whitespace, and trim the result."""
    return collapse_whitespace(normalize_newlines(text)).strip()


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax, keeping the readable prose."""
    cleaned = _MARKDOWN_NOISE.sub(lambda m: m.group(2) or "", text)
    return collapse_whitespace(cleaned).strip()


def dedent_block(text: str) -> str:
    """Remove common leading indentation from a multi-line block."""
    return textwrap.dedent(normalize_newlines(text)).strip("\n")


def indent_block(text: str, prefix: str = "  ", *, skip_blank: bool = True) -> str:
    """Indent every line of ``text`` with ``prefix``."""
    lines = normalize_newlines(text).split("\n")
    return "\n".join(
        line if (skip_blank and not line.strip()) else f"{prefix}{line}" for line in lines
    )


def ellipsize(text: str, limit: int, *, marker: str = "...", middle: bool = False) -> str:
    """Truncate ``text`` to ``limit`` characters, inserting ``marker``."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if middle:
        keep = max(0, limit - len(marker))
        head = keep // 2
        tail = keep - head
        return f"{text[:head]}{marker}{text[len(text) - tail :] if tail else ''}"
    return f"{text[: max(0, limit - len(marker))]}{marker}"


def word_wrap(text: str, width: int = 80, *, indent: str = "", hang: str = "") -> str:
    """Hard wrap prose to ``width`` columns with optional hanging indent."""
    if width <= 10:
        return text
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=hang or indent,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
    )
    out: list[str] = []
    for paragraph in normalize_newlines(text).split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        out.extend(wrapper.wrap(paragraph))
    return "\n".join(out)


def sentence_split(text: str) -> list[str]:
    """Split prose into sentences using punctuation and capitalization cues."""
    normalized = squish(text)
    if not normalized:
        return []
    parts = [piece.strip() for piece in _SENTENCE_END.split(normalized)]
    return [part for part in parts if part]


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 120,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Recursively split ``text`` into overlapping chunks under ``max_chars``.

    The splitter tries the strongest separator first (paragraph breaks) and
    falls back down the list, so chunks stay semantically coherent instead of
    being sliced mid-word.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    normalized = normalize_newlines(text)
    if len(normalized) <= max_chars:
        return [normalized] if normalized.strip() else []
    if overlap >= max_chars:
        overlap = max(0, max_chars // 8)

    chunks: list[str] = []

    def split_with(piece: str, level: int) -> None:
        if len(piece) <= max_chars:
            if piece.strip():
                chunks.append(piece.strip())
            return
        sep = separators[min(level, len(separators) - 1)]
        if not sep:
            for start in range(0, len(piece), max_chars - overlap):
                window = piece[start : start + max_chars]
                if window.strip():
                    chunks.append(window.strip())
                if start + max_chars >= len(piece):
                    break
            return
        segments = [segment for segment in piece.split(sep) if segment != ""]
        buffer = ""
        for segment in segments:
            candidate = segment if not buffer else f"{buffer}{sep}{segment}"
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(buffer.strip())
                tail = buffer[-overlap:] if overlap else ""
                buffer = f"{tail}{sep}{segment}" if tail else segment
            else:
                buffer = segment
            if len(buffer) > max_chars:
                split_with(buffer, level + 1)
                buffer = ""
        if buffer.strip():
            chunks.append(buffer.strip())

    split_with(normalized, 0)
    return chunks


def chunk_by_lines(text: str, size: int = 200) -> Iterator[tuple[int, str]]:
    """Yield ``(start_line_number, block)`` pairs of ``size`` lines each."""
    lines = normalize_newlines(text).split("\n")
    for start in range(0, len(lines), max(1, size)):
        block = "\n".join(lines[start : start + max(1, size)])
        if block.strip():
            yield start + 1, block


def snake_case(value: str) -> str:
    """Convert any identifier-ish string to ``snake_case``."""
    spaced = _CAMEL.sub(" ", value)
    spaced = re.sub(r"[\s\-./]+", " ", spaced)
    return re.sub(r"_+", "_", spaced.strip().lower().replace(" ", "_"))


def camel_case(value: str) -> str:
    """Convert a ``snake_case`` string to ``camelCase``."""
    head, *rest = snake_case(value).split("_")
    return head + "".join(part.title() for part in rest)


def pascal_case(value: str) -> str:
    """Convert a ``snake_case`` string to ``PascalCase``."""
    return "".join(part.title() for part in snake_case(value).split("_") if part)


def title_case(value: str) -> str:
    """Title case a phrase while keeping small words lowercase."""
    small = {"a", "an", "the", "and", "or", "but", "for", "nor", "on", "at", "to", "of", "in", "by"}
    words = value.split()
    return " ".join(
        word if (index and word.lower() in small) else (word[:1].upper() + word[1:])
        for index, word in enumerate(words)
    )


def split_camel(value: str) -> str:
    """Turn ``camelCase`` into space separated words."""
    return _WORD_BOUNDARY.sub(" ", value).strip()


def slugify(value: str, *, separator: str = "-", max_length: int = 60, lowercase: bool = True) -> str:
    """Turn arbitrary text into a URL and anchor safe slug.

    Accents are stripped, non-alphanumeric runs collapse into the separator,
    and the result is trimmed to ``max_length`` without leaving a dangling
    separator. Markdown heading anchors and file names are both built with it.
    """
    text = strip_accents(normalize_unicode(value or ""))
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^a-z0-9]+", separator, text, flags=re.IGNORECASE)
    text = text.strip(separator)
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip(separator)
    return text


def levenshtein(left: str, right: str, *, max_distance: int | None = None) -> int:
    """Return the edit distance between two strings.

    Uses a single rolling row, so memory is ``O(min(len))`` rather than
    quadratic. ``max_distance`` enables early exit for fuzzy matching.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        best = i
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if max_distance is not None and best > max_distance:
            return best
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    """Return a 0..1 normalized string similarity based on edit distance."""
    if not left and not right:
        return 1.0
    longest = max(len(left), len(right))
    if longest == 0:
        return 0.0
    return round(1.0 - levenshtein(left.lower(), right.lower()) / longest, 6)


def longest_common_substring(left: str, right: str) -> str:
    """Return the longest substring shared by both inputs."""
    if not left or not right:
        return ""
    best_end = 0
    best_len = 0
    previous = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len = current[j]
                    best_end = i
        previous = current
    return left[best_end - best_len : best_end]


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply Unicode normalization so comparisons are stable."""
    return unicodedata.normalize(form, text)


def strip_accents(text: str) -> str:
    """Remove diacritics, useful for fuzzy ASCII search."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def count_words(text: str) -> int:
    """Return the number of whitespace separated tokens."""
    return len(text.split())


def count_lines(text: str) -> int:
    """Return the number of lines, counting a trailing newline as one line."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def common_prefix(values: Iterable[str]) -> str:
    """Return the longest shared prefix of all given strings."""
    items = list(values)
    if not items:
        return ""
    shortest = min(items, key=len)
    for index, char in enumerate(shortest):
        if any(item[index] != char for item in items):
            return shortest[:index]
    return shortest


def common_suffix(values: Iterable[str]) -> str:
    """Return the longest shared suffix of all given strings."""
    reversed_prefix = common_prefix([value[::-1] for value in values])
    return reversed_prefix[::-1]


def find_balanced(text: str, open_char: str = "{", close_char: str = "}", start: int = 0) -> tuple[int, int]:
    """Locate the first balanced ``open_char..close_char`` span in ``text``.

    Returns ``(open_index, close_index)`` or ``(-1, -1)`` when unbalanced.
    Naive quoting is respected so braces inside strings are skipped.
    """
    depth = 0
    opened = -1
    in_string = False
    escape = False
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in "\"'":
            in_string = True
            quote = char
            continue
        if char == open_char:
            if depth == 0:
                opened = index
            depth += 1
        elif char == close_char:
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return opened, index
    return -1, -1


def mask_secret(value: str, *, keep: int = 4) -> str:
    """Mask a secret, keeping only the last ``keep`` characters visible."""
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return f"{'*' * (len(value) - keep)}{value[-keep:]}"


__all__ = [
    "camel_case",
    "chunk_by_lines",
    "chunk_text",
    "collapse_whitespace",
    "common_prefix",
    "common_suffix",
    "count_lines",
    "count_words",
    "dedent_block",
    "ellipsize",
    "find_balanced",
    "indent_block",
    "levenshtein",
    "longest_common_substring",
    "mask_secret",
    "normalize_newlines",
    "normalize_unicode",
    "pascal_case",
    "sentence_split",
    "similarity",
    "slugify",
    "snake_case",
    "split_camel",
    "squish",
    "strip_accents",
    "strip_markdown",
    "title_case",
    "word_wrap",
]
