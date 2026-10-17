"""Stdlib-only HTTP transport for MAgent.

MAgent has zero mandatory dependencies, so it implements its own pooled HTTP
client on top of :mod:`http.client`. The public surface here is deliberately
small and mirrors what a higher level library would offer.

Features implemented here that a naive ``urllib`` call would not give you:

* keep-alive connection pooling per origin with idle expiry,
* connect / read / total timeouts as separate knobs,
* bounded response bodies so a hostile server cannot exhaust memory,
* gzip and deflate decoding,
* Server-Sent Events streaming, including reassembly of split tool calls,
* bounded multipart/form-data encoding for media uploads,
* retry with exponential backoff, full jitter, and ``Retry-After`` support,
* automatic credential redaction in every logged URL.
"""

from __future__ import annotations

from .client import (
    DEFAULT_HEADERS,
    DEFAULT_MAX_BODY_BYTES,
    RETRYABLE_STATUS,
    HTTPClient,
    HTTPPool,
    HTTPRequest,
    HTTPResponse,
    HTTPTransport,
    RetryPolicy,
    StdlibTransport,
    ping,
    request,
    with_suppressed_close,
)
from .multipart import (
    CRLF,
    DEFAULT_MAX_UPLOAD_BYTES,
    MultipartEncoder,
    MultipartField,
    build_image_body,
    build_transcription_body,
    decode_multipart,
    encode_multipart,
    guess_field_name,
)
from .sse import (
    DONE_SENTINEL,
    SSEEvent,
    SSEParser,
    extract_delta_text,
    extract_delta_tool_calls,
    iter_sse,
    merge_tool_call_deltas,
    parse_sse_line,
)

__all__ = [
    "CRLF",
    "DEFAULT_HEADERS",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DONE_SENTINEL",
    "RETRYABLE_STATUS",
    "HTTPClient",
    "HTTPPool",
    "HTTPRequest",
    "HTTPResponse",
    "HTTPTransport",
    "MultipartEncoder",
    "MultipartField",
    "RetryPolicy",
    "SSEEvent",
    "SSEParser",
    "StdlibTransport",
    "build_image_body",
    "build_transcription_body",
    "decode_multipart",
    "encode_multipart",
    "extract_delta_text",
    "extract_delta_tool_calls",
    "guess_field_name",
    "iter_sse",
    "merge_tool_call_deltas",
    "parse_sse_line",
    "ping",
    "request",
    "with_suppressed_close",
]
