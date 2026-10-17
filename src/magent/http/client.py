"""Pooled, retrying HTTP client built only on the standard library."""

from __future__ import annotations

import dataclasses
import gzip
import http.client
import io
import json
import socket
import ssl
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ..errors import (
    AuthenticationError,
    ConnectionError_,
    HTTPError,
    PayloadTooLargeError,
    RateLimitedError,
    ServerError,
    TimeoutError_,
    classify_status,
)
from ..utils.urls import redact_url

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/event-stream;q=0.9, */*;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
"""Headers sent on every request unless overridden."""

DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024
"""Hard ceiling on a decoded response body (64 MiB)."""

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 521, 522, 524})
"""HTTP statuses that are worth retrying."""


@dataclass(slots=True)
class HTTPRequest:
    """A fully specified outbound HTTP request."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = 30.0
    read_timeout: float | None = None
    json_body: Any = None
    stream: bool = False

    def __post_init__(self) -> None:
        """Serialize a ``json_body`` into bytes when one was supplied."""
        self.method = str(self.method).upper()
        if self.json_body is not None and self.body is None:
            self.body = json.dumps(self.json_body, default=str, ensure_ascii=False).encode("utf-8")
            self.headers.setdefault("Content-Type", "application/json")

    def safe_url(self) -> str:
        """Return the URL with any embedded credentials removed."""
        return redact_url(self.url)

    def content_length(self) -> int:
        """Return the byte length of the request body."""
        return len(self.body) if self.body else 0


@dataclass(slots=True)
class HTTPResponse:
    """The result of an HTTP request."""

    status: int
    headers: dict[str, str]
    body: bytes = b""
    url: str = ""
    elapsed_ms: float = 0.0
    reason: str = ""
    truncated: bool = False
    from_pool: bool = False

    @property
    def ok(self) -> bool:
        """Report whether the status is in the 2xx range."""
        return 200 <= self.status < 300

    @property
    def content_type(self) -> str:
        """Return the ``Content-Type`` header value."""
        return self.headers.get("content-type", "")

    def is_json(self) -> bool:
        """Report whether the response advertises JSON."""
        return "json" in self.content_type.lower()

    def is_event_stream(self) -> bool:
        """Report whether the response advertises SSE."""
        return "text/event-stream" in self.content_type.lower()

    def text(self, encoding: str = "utf-8") -> str:
        """Decode the body as text."""
        return self.body.decode(encoding, errors="replace")

    def json(self, default: Any = None) -> Any:
        """Decode the body as JSON, returning ``default`` on failure."""
        if not self.body:
            return default
        try:
            return json.loads(self.body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return default

    def header(self, name: str, default: str = "") -> str:
        """Look up a response header case-insensitively."""
        return self.headers.get(name.lower(), default)

    def safe_url(self) -> str:
        """Return the response URL with embedded credentials removed.

        Responses need the same protection as requests because error messages
        and structured ``HTTPError.details`` include their URL. The status path
        previously called this method even though only :class:`HTTPRequest`
        implemented it, masking every non-2xx response with ``AttributeError``.
        """
        return redact_url(self.url)

    def retry_after(self) -> float | None:
        """Parse a ``Retry-After`` header into seconds, if present."""
        raw = self.header("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime

            delta = parsedate_to_datetime(raw).timestamp() - time.time()
            return max(0.0, delta)
        except (ValueError, TypeError):
            return None

    def raise_for_status(self) -> None:
        """Raise a specific :class:`HTTPError` subclass for non-2xx statuses."""
        if self.ok:
            return
        message = f"HTTP {self.status} {self.reason} for {self.safe_url()}"
        detail = self.text()[:2000]
        if detail:
            message = f"{message}: {detail.strip()[:400]}"
        error = classify_status(self.status, message, url=self.safe_url(), body=detail)
        if isinstance(error, RateLimitedError):
            raise RateLimitedError(message, retry_after=self.retry_after(), url=self.safe_url())
        raise error

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata about the response (never the body)."""
        return {
            "status": self.status,
            "reason": self.reason,
            "url": self.url,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "bytes": len(self.body),
            "content_type": self.content_type,
            "from_pool": self.from_pool,
        }


class HTTPTransport:
    """Abstract transport interface so ``httpx`` can be swapped in."""

    def send(self, request: HTTPRequest) -> HTTPResponse:
        """Perform one request and return the response."""
        raise NotImplementedError

    def stream(self, request: HTTPRequest) -> Iterator[bytes]:
        """Yield raw response body chunks for a streaming request."""
        raise NotImplementedError

    def close(self) -> None:
        """Release every held resource."""

    @property
    def name(self) -> str:
        """Return the transport implementation name."""
        return type(self).__name__


class HTTPPool:
    """Thread-safe keep-alive connection pool keyed by origin.

    Idle connections older than ``max_idle_seconds`` are discarded on checkout,
    which avoids handing out sockets a server has already closed.
    """

    def __init__(
        self,
        *,
        max_connections: int = 16,
        max_keepalive: int = 8,
        max_idle_seconds: float = 60.0,
    ) -> None:
        self.max_connections = max(1, max_connections)
        self.max_keepalive = max(1, max_keepalive)
        self.max_idle_seconds = max_idle_seconds
        self._lock = threading.Lock()
        self._idle: dict[str, deque[tuple[http.client.HTTPConnection, float]]] = {}
        self._active = 0
        self.hits = 0
        self.misses = 0
        self.created = 0
        self.evicted = 0

    @staticmethod
    def _origin_key(url: str) -> str:
        """Return a pool key identifying scheme, host, and port."""
        parts = urlsplit(url)
        scheme = (parts.scheme or "http").lower()
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if scheme == "https" else 80)
        return f"{scheme}://{host}:{port}"

    def acquire(self, url: str, *, timeout: float) -> tuple[http.client.HTTPConnection, bool]:
        """Return a connection for ``url`` and whether it came from the pool."""
        key = self._origin_key(url)
        with self._lock:
            bucket = self._idle.get(key)
            now = time.monotonic()
            while bucket:
                connection, stored_at = bucket.popleft()
                if now - stored_at > self.max_idle_seconds:
                    self.evicted += 1
                    with_suppressed_close(connection)
                    continue
                if not bucket:
                    self._idle.pop(key, None)
                self._active += 1
                self.hits += 1
                return connection, True
            if self._active >= self.max_connections:
                self.misses += 1
                raise ConnectionError_(
                    f"HTTP connection pool exhausted ({self.max_connections} connections)"
                )
            self._active += 1
            self.misses += 1
        connection = _build_connection(url, timeout=timeout)
        self.created += 1
        return connection, False

    def release(self, url: str, connection: http.client.HTTPConnection, *, reusable: bool) -> None:
        """Return a connection to the pool or discard it."""
        key = self._origin_key(url)
        with self._lock:
            self._active = max(0, self._active - 1)
            if not reusable:
                with_suppressed_close(connection)
                return
            bucket = self._idle.setdefault(key, deque())
            if len(bucket) >= self.max_keepalive:
                self.evicted += 1
                with_suppressed_close(connection)
                return
            bucket.append((connection, time.monotonic()))

    def discard(self, url: str, connection: http.client.HTTPConnection) -> None:
        """Drop a connection without returning it to the pool."""
        self.release(url, connection, reusable=False)

    def clear(self) -> None:
        """Close every idle connection."""
        with self._lock:
            for bucket in self._idle.values():
                for connection, _ in bucket:
                    with_suppressed_close(connection)
            self._idle.clear()

    def stats(self) -> dict[str, Any]:
        """Return pool utilization counters."""
        with self._lock:
            idle = sum(len(bucket) for bucket in self._idle.values())
            return {
                "active": self._active,
                "idle": idle,
                "created": self.created,
                "hits": self.hits,
                "misses": self.misses,
                "evicted": self.evicted,
                "max_connections": self.max_connections,
                "max_keepalive": self.max_keepalive,
            }


def with_suppressed_close(connection: http.client.HTTPConnection) -> None:
    """Close a connection, ignoring errors from an already-dead socket."""
    try:
        connection.close()
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass


def _build_connection(url: str, *, timeout: float, context: ssl.SSLContext | None = None) -> http.client.HTTPConnection:
    """Create a fresh :mod:`http.client` connection for ``url``."""
    parts = urlsplit(url)
    scheme = (parts.scheme or "http").lower()
    host = parts.hostname or "127.0.0.1"
    if scheme == "https":
        ssl_context = context or ssl.create_default_context()
        return http.client.HTTPSConnection(host, parts.port or 443, timeout=timeout, context=ssl_context)
    return http.client.HTTPConnection(host, parts.port or 80, timeout=timeout)


def _request_target(url: str) -> str:
    """Return the path+query portion to send in the request line."""
    parts = urlsplit(url)
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    return target


def _decode_body(raw: bytes, encoding: str) -> bytes:
    """Decode gzip or deflate response bodies."""
    if not raw:
        return raw
    lowered = encoding.lower()
    if "gzip" in lowered:
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError, zlib.error):
            return raw
    if "deflate" in lowered:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            try:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
            except zlib.error:
                return raw
    return raw


class StdlibTransport(HTTPTransport):
    """HTTP transport implemented with :mod:`http.client` and a pool."""

    def __init__(
        self,
        *,
        pool: HTTPPool | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.pool = pool or HTTPPool()
        self.max_body_bytes = max_body_bytes
        self.verify_ssl = verify_ssl
        self.ssl_context = ssl_context
        if ssl_context is None and not verify_ssl:
            insecure = ssl.create_default_context()
            insecure.check_hostname = False
            insecure.verify_mode = ssl.CERT_NONE
            self.ssl_context = insecure
        self.requests_sent = 0
        self.bytes_received = 0
        self.errors = 0

    # -- public API ----------------------------------------------------------

    def send(self, request: HTTPRequest) -> HTTPResponse:
        """Perform one request, retrying once on a stale pooled connection."""
        started = time.perf_counter()
        for attempt in (0, 1):
            connection, reused = self.pool.acquire(request.url, timeout=request.timeout)
            if attempt == 1:
                connection = _build_connection(
                    request.url, timeout=request.timeout, context=self.ssl_context
                )
                reused = False
            try:
                headers = {**DEFAULT_HEADERS, **request.headers}
                if request.body:
                    headers["Content-Length"] = str(len(request.body))
                if reused:
                    headers.pop("Connection", None)
                connection.request(
                    request.method,
                    _request_target(request.url),
                    body=request.body,
                    headers=headers,
                )
                if request.timeout != connection.timeout:
                    connection.timeout = request.timeout
                response = connection.getresponse()
                raw = self._read_bounded(response)
                body = _decode_body(raw, response.getheader("Content-Encoding", ""))
                elapsed = (time.perf_counter() - started) * 1000.0
                self.requests_sent += 1
                self.bytes_received += len(body)
                keep_alive = (
                    response.will_close is False
                    and response.getheader("Connection", "").lower() != "close"
                    and not response.isclosed()
                )
                result = HTTPResponse(
                    status=response.status,
                    reason=response.reason or "",
                    headers={key.lower(): value for key, value in response.getheaders()},
                    body=body,
                    url=request.safe_url(),
                    elapsed_ms=elapsed,
                    truncated=len(raw) >= self.max_body_bytes,
                    from_pool=reused,
                )
                self.pool.release(request.url, connection, reusable=keep_alive)
                return result
            except (http.client.BadStatusLine, http.client.RemoteDisconnected, BrokenPipeError) as exc:
                self.pool.discard(request.url, connection)
                self.errors += 1
                if attempt == 0 and reused:
                    continue
                raise ConnectionError_(
                    f"Connection to {request.safe_url()} was dropped: {exc}", url=request.safe_url()
                ) from exc
            except socket.timeout as exc:
                self.pool.discard(request.url, connection)
                self.errors += 1
                raise TimeoutError_(
                    f"Timed out after {request.timeout}s talking to {request.safe_url()}",
                    url=request.safe_url(),
                ) from exc
            except OSError as exc:
                self.pool.discard(request.url, connection)
                self.errors += 1
                raise ConnectionError_(
                    f"Could not reach {request.safe_url()}: {exc}", url=request.safe_url()
                ) from exc
        raise ConnectionError_(f"Unreachable retry path for {request.safe_url()}")

    def stream(self, request: HTTPRequest) -> Iterator[bytes]:
        """Yield decoded body chunks without buffering the whole response."""
        connection, reused = self.pool.acquire(request.url, timeout=request.timeout)
        reusable = False
        try:
            headers = {**DEFAULT_HEADERS, **request.headers}
            if request.body:
                headers["Content-Length"] = str(len(request.body))
            connection.request(
                request.method,
                _request_target(request.url),
                body=request.body,
                headers=headers,
            )
            response = connection.getresponse()
            if response.status >= 400:
                raw = self._read_bounded(response)
                yield raw
                return
            encoding = (response.getheader("Content-Encoding") or "").lower()
            decoder: Any = None
            if "gzip" in encoding:
                decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
            elif "deflate" in encoding:
                decoder = zlib.decompressobj()
            seen = 0
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                seen += len(chunk)
                if seen > self.max_body_bytes:
                    raise PayloadTooLargeError(
                        f"Streamed response from {request.safe_url()} exceeded "
                        f"{self.max_body_bytes} bytes"
                    )
                if decoder is not None:
                    decoded = decoder.decompress(chunk)
                    if decoded:
                        yield decoded
                else:
                    yield chunk
            if decoder is not None:
                tail = decoder.flush()
                if tail:
                    yield tail
            reusable = response.will_close is False
        except socket.timeout as exc:
            raise TimeoutError_(f"Stream timed out from {request.safe_url()}") from exc
        except OSError as exc:
            if not isinstance(exc, socket.timeout):
                raise ConnectionError_(f"Stream failed for {request.safe_url()}: {exc}") from exc
            raise
        finally:
            if reusable:
                self.pool.release(request.url, connection, reusable=True)
            else:
                self.pool.discard(request.url, connection)

    def close(self) -> None:
        """Close every pooled connection."""
        self.pool.clear()

    # -- internals -----------------------------------------------------------

    def _read_bounded(self, response: http.client.HTTPResponse) -> bytes:
        """Read a response body without exceeding ``max_body_bytes``."""
        buffer = io.BytesIO()
        remaining = self.max_body_bytes
        while True:
            chunk = response.read(min(65536, remaining))
            if not chunk:
                break
            buffer.write(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                break
        return buffer.getvalue()

    def stats(self) -> dict[str, Any]:
        """Return transport level counters."""
        return {
            "requests_sent": self.requests_sent,
            "bytes_received": self.bytes_received,
            "errors": self.errors,
            "pool": self.pool.stats(),
        }


@dataclass(slots=True)
class RetryPolicy:
    """Configuration for automatic request retries."""

    max_attempts: int = 3
    backoff_base: float = 0.4
    backoff_factor: float = 2.0
    backoff_cap: float = 20.0
    jitter: float = 0.25
    retry_statuses: frozenset[int] = RETRYABLE_STATUS
    retry_methods: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "POST", "PUT"})
    honor_retry_after: bool = True
    seed: float | None = 1234.0

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Return how long to wait before the given retry attempt."""
        import random

        if retry_after is not None and self.honor_retry_after:
            return min(self.backoff_cap, max(0.0, retry_after))
        generator = random.Random(self.seed if self.seed is not None else None)
        delay = min(self.backoff_cap, self.backoff_base * (self.backoff_factor ** max(0, attempt - 1)))
        if self.jitter > 0:
            delay *= 1.0 + generator.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)

    def should_retry(self, attempt: int, error: BaseException | None = None, status: int | None = None) -> bool:
        """Decide whether another attempt is allowed."""
        if attempt >= self.max_attempts:
            return False
        if status is not None:
            return status in self.retry_statuses
        if error is None:
            return False
        from ..errors import is_retryable

        return is_retryable(error)


class HTTPClient:
    """High level client: pooling, retries, timeouts, auth, and JSON helpers."""

    def __init__(
        self,
        *,
        base_url: str = "",
        transport: HTTPTransport | None = None,
        timeout: float = 30.0,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        retry: RetryPolicy | None = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        verify_ssl: bool = True,
        on_retry: Callable[[int, float, BaseException | None], None] | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.transport = transport or StdlibTransport(
            max_body_bytes=max_body_bytes, verify_ssl=verify_ssl
        )
        self.timeout = timeout
        self.connect_timeout = connect_timeout or timeout
        self.read_timeout = read_timeout or timeout
        self.headers = dict(headers or {})
        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme
        self.retry = retry or RetryPolicy()
        self.on_retry = on_retry
        self.attempts_total = 0
        self.retries_total = 0
        self.last_error: str | None = None

    # -- header management ---------------------------------------------------

    def auth_headers(self) -> dict[str, str]:
        """Build the authentication headers for the current API key."""
        if not self.api_key:
            return {}
        value = (
            f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else str(self.api_key)
        )
        return {self.auth_header: value}

    def merged_headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        """Merge client, auth, and per-request headers with correct precedence."""
        merged: dict[str, str] = {}
        merged.update(self.headers)
        merged.update(self.auth_headers())
        if extra:
            merged.update(extra)
        return merged

    def resolve_url(self, url: str) -> str:
        """Join a relative path against ``base_url``."""
        if "://" in url:
            return url
        if not self.base_url:
            return url
        return f"{self.base_url}/{str(url).lstrip('/')}"

    # -- verbs ---------------------------------------------------------------

    def get(self, url: str, *, params: dict[str, Any] | None = None, **kwargs: Any) -> HTTPResponse:
        """Issue a GET request, optionally adding query parameters."""
        from ..utils.urls import build_query

        target = self.resolve_url(url)
        if params:
            query = build_query(params)
            if query:
                target = f"{target}{'&' if '?' in target else '?'}{query}"
        return self.request("GET", target, **kwargs)

    def post(self, url: str, *, json_body: Any = None, body: bytes | None = None, **kwargs: Any) -> HTTPResponse:
        """Issue a POST request with a JSON or raw body."""
        return self.request("POST", self.resolve_url(url), json_body=json_body, body=body, **kwargs)

    def put(self, url: str, *, json_body: Any = None, **kwargs: Any) -> HTTPResponse:
        """Issue a PUT request with a JSON body."""
        return self.request("PUT", self.resolve_url(url), json_body=json_body, **kwargs)

    def patch(self, url: str, *, json_body: Any = None, **kwargs: Any) -> HTTPResponse:
        """Issue a PATCH request with a JSON body."""
        return self.request("PATCH", self.resolve_url(url), json_body=json_body, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> HTTPResponse:
        """Issue a DELETE request."""
        return self.request("DELETE", self.resolve_url(url), **kwargs)

    def head(self, url: str, **kwargs: Any) -> HTTPResponse:
        """Issue a HEAD request."""
        return self.request("HEAD", self.resolve_url(url), **kwargs)

    # -- core ----------------------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        body: bytes | None = None,
        timeout: float | None = None,
        raise_for_status: bool = True,
        max_attempts: int | None = None,
    ) -> HTTPResponse:
        """Perform a request with retries and raise on error statuses.

        ``raise_for_status=False`` returns the raw response so callers can
        implement their own error handling (the CLI uses this for ``doctor``).
        """
        policy = (
            dataclasses.replace(self.retry, max_attempts=max_attempts)
            if max_attempts is not None
            else self.retry
        )
        attempt = 0
        last_exception: BaseException | None = None
        while True:
            attempt += 1
            self.attempts_total += 1
            payload = HTTPRequest(
                method=method,
                url=url,
                headers=self.merged_headers(headers),
                body=body,
                json_body=json_body,
                timeout=timeout if timeout is not None else self.read_timeout,
            )
            try:
                response = self.transport.send(payload)
            except (ConnectionError_, TimeoutError_, ServerError) as exc:
                last_exception = exc
                self.last_error = str(exc)
                if policy.should_retry(attempt, error=exc):
                    delay = policy.delay_for(attempt)
                    self.retries_total += 1
                    if self.on_retry:
                        self.on_retry(attempt, delay, exc)
                    time.sleep(delay)
                    continue
                raise
            if policy.should_retry(attempt, status=response.status) and attempt < policy.max_attempts:
                delay = policy.delay_for(attempt, response.retry_after())
                self.retries_total += 1
                self.last_error = f"HTTP {response.status}"
                if self.on_retry:
                    self.on_retry(attempt, delay, None)
                time.sleep(delay)
                continue
            if raise_for_status:
                response.raise_for_status()
            return response

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        timeout: float | None = None,
    ) -> Iterator[bytes]:
        """Open a streaming request and yield body chunks."""
        payload = HTTPRequest(
            method=method,
            url=self.resolve_url(url),
            headers=self.merged_headers(headers),
            json_body=json_body,
            timeout=timeout if timeout is not None else self.read_timeout,
            stream=True,
        )
        yield from self.transport.stream(payload)

    def get_json(self, url: str, *, default: Any = None, **kwargs: Any) -> Any:
        """GET a URL and decode the JSON body."""
        return self.get(url, **kwargs).json(default)

    def post_json(self, url: str, json_body: Any, *, default: Any = None, **kwargs: Any) -> Any:
        """POST JSON and decode the JSON response body."""
        return self.post(url, json_body=json_body, **kwargs).json(default)

    def close(self) -> None:
        """Close the underlying transport."""
        self.transport.close()

    def __enter__(self) -> HTTPClient:
        """Support ``with HTTPClient(...) as client:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client when leaving a ``with`` block."""
        self.close()

    def stats(self) -> dict[str, Any]:
        """Return client level counters including transport stats."""
        base = self.transport.stats() if isinstance(self.transport, StdlibTransport) else {}
        return {
            "attempts_total": self.attempts_total,
            "retries_total": self.retries_total,
            "last_error": self.last_error,
            "base_url": self.base_url,
            **base,
        }


def request(method: str, url: str, **kwargs: Any) -> HTTPResponse:
    """One-shot convenience wrapper around a temporary :class:`HTTPClient`."""
    with HTTPClient() as client:
        return client.request(method, url, **kwargs)


def ping(url: str, *, timeout: float = 2.0) -> tuple[bool, str]:
    """Cheap reachability probe returning ``(ok, detail)``."""
    try:
        response = HTTPClient(timeout=timeout).get(url, raise_for_status=False)
        return response.status < 500, f"HTTP {response.status}"
    except (AuthenticationError, HTTPError) as exc:
        return False, str(exc)[:200]
    except OSError as exc:
        return False, f"unreachable: {exc}"


__all__ = [
    "DEFAULT_HEADERS",
    "DEFAULT_MAX_BODY_BYTES",
    "HTTPClient",
    "HTTPPool",
    "HTTPRequest",
    "HTTPResponse",
    "HTTPTransport",
    "RETRYABLE_STATUS",
    "RetryPolicy",
    "StdlibTransport",
    "ping",
    "request",
    "with_suppressed_close",
]
