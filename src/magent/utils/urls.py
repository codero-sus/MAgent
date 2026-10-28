"""URL building, normalization, and secret redaction."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
"""Hostnames considered loopback/local."""

_PRIVATE_NET_RE = re.compile(
    r"^(10\.|127\.|169\.254\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|100\.(6[4-9]|[7-9]\d|1\d{2}|2[0-6])\.)"
)
"""IPv4 ranges that are private, loopback, link-local or CGNAT."""

_CREDENTIAL_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<user>[^/@]+)@")
"""Regex matching ``scheme://user:pass@host`` credentials in a URL."""

DEFAULT_CORTEX_PORT = 8624
"""Port Cortex LLMHoster binds by default."""


def normalize_base_url(
    url: str, *, default_scheme: str = "http", default_port: int | None = None
) -> str:
    """Normalize a base URL: add scheme, lowercase host, strip trailing slash.

    Bare ``host:port`` and bare ``localhost`` inputs are accepted so users can
    type ``--cortex-url 127.0.0.1:8624`` or even just ``localhost``.
    """
    text = str(url or "").strip()
    if not text:
        text = f"127.0.0.1:{default_port or DEFAULT_CORTEX_PORT}"
    if "://" not in text:
        text = f"{default_scheme}://{text}"
    parts = urlsplit(text)
    scheme = (parts.scheme or default_scheme).lower()
    host = (parts.hostname or "127.0.0.1").lower()
    port = parts.port
    if port is None and default_port and host in _LOCAL_HOSTS - {"0.0.0.0"}:
        port = default_port
    netloc = host
    if port:
        netloc = f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def join_url(base: str, path: str) -> str:
    """Join a base URL and a path without losing or duplicating slashes."""
    normalized_base = str(base).rstrip("/")
    normalized_path = str(path)
    if not normalized_path:
        return normalized_base
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    if normalized_base.endswith(normalized_path):
        return normalized_base
    return f"{normalized_base}{normalized_path}"


def parse_url(url: str) -> dict[str, object]:
    """Split a URL into a dictionary of its components."""
    parts = urlsplit(str(url))
    return {
        "scheme": parts.scheme,
        "host": parts.hostname or "",
        "port": parts.port,
        "netloc": parts.netloc,
        "path": parts.path,
        "query": dict(parse_qsl(parts.query, keep_blank_values=True)),
        "fragment": parts.fragment,
        "username": parts.username,
        "has_password": bool(parts.password),
    }


def url_origin(url: str) -> str:
    """Return the ``scheme://host[:port]`` origin of a URL."""
    parts = urlsplit(str(url))
    origin = f"{parts.scheme}://{parts.hostname or ''}"
    if parts.port:
        origin += f":{parts.port}"
    return origin


def build_query(params: dict[str, object]) -> str:
    """Encode a mapping into a URL query string, skipping ``None`` values."""
    items = [(str(key), str(value)) for key, value in params.items() if value is not None]
    return urlencode(items, doseq=False)


def redact_url(url: str) -> str:
    """Remove any embedded credentials from a URL before logging it."""
    return _CREDENTIAL_RE.sub(lambda m: f"{m.group('scheme')}[REDACTED]@", str(url))


def is_local_url(url: str) -> bool:
    """Report whether a URL points at loopback or a private network address."""
    host = (urlsplit(str(url)).hostname or "").lower().strip("[]")
    if host in _LOCAL_HOSTS or host.endswith(".local") or host.endswith(".localhost"):
        return True
    return bool(_PRIVATE_NET_RE.match(host))


def is_private_host(host: str) -> bool:
    """Report whether a bare hostname is loopback/private/link-local."""
    return is_local_url(f"http://{host}")


def quote_path(path: str) -> str:
    """Percent-encode a URL path while preserving separators."""
    return quote(str(path), safe="/-_.~")


def unquote_path(path: str) -> str:
    """Decode a percent-encoded URL path."""
    return unquote(str(path))


def data_url(mime_type: str, payload: bytes, *, base64: bool = True) -> str:
    """Build a ``data:`` URL from raw bytes."""
    import base64 as _base64

    if base64:
        encoded = _base64.b64encode(payload).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    return f"data:{mime_type},{quote(payload.decode('latin-1'))}"


def parse_data_url(url: str) -> tuple[str, bytes] | None:
    """Decode a ``data:`` URL into ``(mime_type, payload)``."""
    import base64 as _base64

    if not str(url).startswith("data:"):
        return None
    header, _, body = str(url)[5:].partition(",")
    parts = header.split(";")
    mime_type = parts[0] or "application/octet-stream"
    try:
        if "base64" in parts[1:]:
            return mime_type, _base64.b64decode(body, validate=False)
        return mime_type, unquote(body).encode("utf-8")
    except (ValueError, TypeError):
        return None


def strip_query(url: str) -> str:
    """Return a URL without its query string or fragment."""
    parts = urlsplit(str(url))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def with_port(url: str, port: int) -> str:
    """Return a copy of ``url`` bound to a different port."""
    parts = urlsplit(str(url))
    netloc = f"{parts.hostname or '127.0.0.1'}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def ensure_scheme(url: str, scheme: str = "http") -> str:
    """Prepend ``scheme://`` when the URL has no scheme."""
    text = str(url).strip()
    return text if "://" in text else f"{scheme}://{text}"


__all__ = [
    "DEFAULT_CORTEX_PORT",
    "build_query",
    "data_url",
    "ensure_scheme",
    "is_local_url",
    "is_private_host",
    "join_url",
    "normalize_base_url",
    "parse_data_url",
    "parse_url",
    "quote_path",
    "redact_url",
    "strip_query",
    "unquote_path",
    "url_origin",
    "with_port",
]
