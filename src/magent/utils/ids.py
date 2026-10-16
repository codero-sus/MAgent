"""Identifier generation and normalization."""

from __future__ import annotations

import re
import unicodedata

from .hashing import random_hex, random_token, snowflake, ulid_like, uuid_hex

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, separator: str = "-", max_length: int = 64, lowercase: bool = True) -> str:
    """Convert arbitrary text into a URL/filename safe slug.

    Accented characters are transliterated to ASCII first so ``"Café"``
    becomes ``"cafe"`` rather than being dropped.
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    if lowercase:
        text = text.lower()
    text = _NON_SLUG.sub(separator, text).strip(separator)
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip(separator)
    return text or "item"


def run_id(prefix: str = "run") -> str:
    """Return a new sortable run identifier."""
    return ulid_like(prefix=f"{prefix}_")


def session_id() -> str:
    """Return a new sortable session identifier."""
    return ulid_like(prefix="ses_")


def message_id() -> str:
    """Return a new message identifier."""
    return f"msg_{uuid_hex()[:20]}"


def tool_call_id(index: int = 0) -> str:
    """Return a new tool call identifier, optionally namespaced by ``index``."""
    return f"call_{index:02d}_{random_hex(6)}"


def trace_id() -> str:
    """Return a new 32-hex distributed trace identifier."""
    return uuid_hex()


def span_id() -> str:
    """Return a new 16-hex span identifier."""
    return random_hex(8)


def api_key(prefix: str = "mag") -> str:
    """Generate a new random MAgent API key."""
    return f"{prefix}_{random_token(24)}"


def next_snowflake(worker_id: int = 1) -> int:
    """Return the next snowflake id for a given worker."""
    return snowflake(worker_id=worker_id)


__all__ = [
    "api_key",
    "message_id",
    "next_snowflake",
    "random_hex",
    "random_token",
    "run_id",
    "session_id",
    "slugify",
    "snowflake",
    "span_id",
    "tool_call_id",
    "trace_id",
    "ulid_like",
    "uuid_hex",
]
