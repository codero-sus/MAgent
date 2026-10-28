"""Deterministic hashing and stable identifier helpers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_COUNTER_LOCK = threading.Lock()
_COUNTER = 0
_BOOT_MS = int(time.time() * 1000) & 0xFFFFFFFFFF


def sha256_hex(data: str | bytes) -> str:
    """Return the hex SHA-256 digest of text or bytes."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def blake2b_hex(data: str | bytes, *, digest_size: int = 16) -> str:
    """Return a hex BLAKE2b digest, faster than SHA-256 for cache keys."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1048576) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Re-exported from :mod:`magent.utils.files` so hashing callers do not have to
    know which helper module owns the streaming implementation.
    """
    from .files import file_sha256 as _stream_digest

    return _stream_digest(path, chunk_size=chunk_size)


def short_hash(data: str | bytes, *, length: int = 10) -> str:
    """Return a short hex digest suitable for display and filenames."""
    return sha256_hex(data)[: max(4, length)]


def content_hash(*parts: Any) -> str:
    """Hash a sequence of values with type tags to avoid ambiguity.

    ``content_hash("ab", "c")`` and ``content_hash("a", "bc")`` produce
    different digests because each part is length prefixed.
    """
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, bytes):
            encoded = part
        elif isinstance(part, str):
            encoded = part.encode("utf-8")
        else:
            encoded = json.dumps(part, sort_keys=True, default=str, ensure_ascii=False).encode()
        digest.update(b"%d:" % len(encoded))
        digest.update(encoded)
    return digest.hexdigest()


def stable_hash(value: Any) -> int:
    """Return a deterministic 64-bit integer hash (unlike builtin ``hash``).

    Python randomizes ``hash(str)`` per process; caches persisted to disk need
    a value that survives restarts.
    """
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def hmac_sign(key: str, message: str) -> str:
    """Return a hex HMAC-SHA256 signature of ``message`` under ``key``."""
    import hmac

    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def hmac_verify(key: str, message: str, signature: str) -> bool:
    """Constant-time verify an HMAC signature produced by :func:`hmac_sign`."""
    import hmac

    expected = hmac_sign(key, message)
    return hmac.compare_digest(expected, str(signature))


def constant_time_equals(left: str, right: str) -> bool:
    """Compare two strings without leaking timing information."""
    import hmac

    return hmac.compare_digest(str(left), str(right))


def uuid_hex() -> str:
    """Return a 32 character hex UUID4 without dashes."""
    return uuid.uuid4().hex


def ulid_like(*, prefix: str = "") -> str:
    """Return a lexicographically sortable, time-ordered unique identifier.

    Format: 46-bit millisecond timestamp + 16-bit counter + 80 bits of
    randomness, base32-ish encoded. Sorts correctly as a plain string.
    """
    global _COUNTER
    with _COUNTER_LOCK:
        _COUNTER = (_COUNTER + 1) & 0xFFFF
        counter = _COUNTER
    millis = int(time.time() * 1000) & 0xFFFFFFFFFF
    entropy = int.from_bytes(os.urandom(10), "big")
    value = (millis << 96) | (counter << 80) | entropy
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    chars: list[str] = []
    for _ in range(24):
        chars.append(alphabet[value & 0x1F])
        value >>= 5
    body = "".join(reversed(chars))
    return f"{prefix}{body}" if prefix else body


def snowflake(*, worker_id: int = 1) -> int:
    """Return a 64-bit Twitter-style snowflake id unique per process."""
    global _COUNTER
    with _COUNTER_LOCK:
        _COUNTER = (_COUNTER + 1) & 0xFFF
        sequence = _COUNTER
    timestamp = int(time.time() * 1000) - _BOOT_MS
    return ((timestamp & 0x1FFFFFFFFFF) << 22) | ((worker_id & 0x3FF) << 12) | sequence


def random_token(nbytes: int = 32) -> str:
    """Return a URL-safe random token for API keys and session ids."""
    import base64

    return base64.urlsafe_b64encode(os.urandom(nbytes)).decode("ascii").rstrip("=")


def random_hex(nbytes: int = 8) -> str:
    """Return ``nbytes`` of cryptographically random hex."""
    return os.urandom(nbytes).hex()


__all__ = [
    "blake2b_hex",
    "constant_time_equals",
    "content_hash",
    "file_sha256",
    "hmac_sign",
    "hmac_verify",
    "random_hex",
    "random_token",
    "sha256_hex",
    "short_hash",
    "snowflake",
    "stable_hash",
    "ulid_like",
    "uuid_hex",
]
