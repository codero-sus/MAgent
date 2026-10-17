"""HTTP and web content tools, gated by the sandbox network policy.

Network access is off by default (``tools.allow_network = false``). When it is
enabled, an allowlist or denylist of domains can further constrain it, and every
decision is recorded by the sandbox so the audit log shows exactly what the
agent fetched.

The page-to-markdown converter is deliberately small and dependency-free: it
strips scripts, styles, and navigation, then preserves headings, lists, links,
tables, and code blocks, which is what an LLM actually needs.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any
from urllib.parse import urljoin

from ..._version import __version__
from ...errors import ToolExecutionError
from ...http.client import HTTPClient, HTTPResponse
from ...tools.base import ToolContext, tool
from ...utils.files import atomic_write_bytes, ensure_dir, human_size
from ...utils.mime import guess_mime, is_text_mime
from ...utils.text import collapse_whitespace, ellipsize, normalize_newlines

_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|template|svg|iframe)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEAD_BLOCK = re.compile(r"<head\b[^>]*>.*?</head>", re.DOTALL | re.IGNORECASE)
_NAV_BLOCK = re.compile(
    r"<(nav|footer|aside|header|form)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_TAG = re.compile(r"<[^>]+>")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _client(context: ToolContext, timeout: float) -> HTTPClient:
    """Build an HTTP client honoring the sandbox policy."""
    settings = getattr(getattr(context, "config", None), "server", None)
    verify = bool(getattr(settings, "verify_ssl", True))
    agent = getattr(getattr(context, "config", None), "output", None)
    identifier = getattr(agent, "user_agent", "") if agent is not None else ""
    return HTTPClient(
        timeout=timeout,
        verify_ssl=verify,
        headers={"User-Agent": identifier or f"MAgent/{__version__}"},
    )


def _check_network(context: ToolContext, url: str) -> None:
    """Authorize an outbound URL through the sandbox."""
    if context.sandbox is not None:
        context.sandbox.check_url(url)


@tool(category="web", risk="network", network=True, timeout=60.0, tags=["http", "get"])
def http_get(
    url: str,
    headers: str = "",
    timeout: float = 30.0,
    max_bytes: int = 200000,
    follow_redirects: bool = True,
    as_text: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Fetch a URL with GET and return the response.

    Args:
        url: absolute URL to fetch
        headers: extra request headers as "Name: value" lines separated by newlines
        timeout: seconds to wait for the response
        max_bytes: maximum response body characters to return
        follow_redirects: follow 3xx redirects
        as_text: decode the body as text instead of returning hex
    """
    ctx = context or ToolContext()
    _check_network(ctx, url)
    parsed_headers = _parse_headers(headers)
    client = _client(ctx, timeout)
    try:
        response = _request_with_redirects(
            client, "GET", url, parsed_headers, timeout, follow_redirects
        )
    finally:
        client.close()
    return _format_response(response, max_bytes, as_text)


@tool(category="web", risk="network", network=True, timeout=60.0, tags=["http", "post"])
def http_post(
    url: str,
    body: str = "",
    headers: str = "",
    content_type: str = "application/json",
    timeout: float = 30.0,
    max_bytes: int = 200000,
    method: str = "POST",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Send a request with a body and return the response.

    Args:
        url: absolute URL to send to
        body: request body text, or a JSON document
        headers: extra request headers as "Name: value" lines
        content_type: value for the Content-Type header
        timeout: seconds to wait for the response
        max_bytes: maximum response body characters to return
        method: HTTP method, POST by default
    """
    ctx = context or ToolContext()
    _check_network(ctx, url)
    allowed = {"POST", "PUT", "PATCH", "DELETE"}
    if method.upper() not in allowed:
        raise ToolExecutionError(f"method must be one of {sorted(allowed)}")
    parsed_headers = _parse_headers(headers)
    parsed_headers.setdefault("Content-Type", content_type)
    client = _client(ctx, timeout)
    try:
        response = client.request(
            method.upper(),
            url,
            headers=parsed_headers,
            body=body.encode("utf-8"),
            timeout=timeout,
            raise_for_status=False,
        )
    finally:
        client.close()
    return _format_response(response, max_bytes, True)


def _parse_headers(text: str) -> dict[str, str]:
    """Parse ``Name: value`` lines into a header mapping."""
    out: dict[str, str] = {}
    for line in normalize_newlines(text or "").split("\n"):
        if not line.strip():
            continue
        name, sep, value = line.partition(":")
        if sep:
            out[name.strip()] = value.strip()
    return out


def _request_with_redirects(
    client: HTTPClient,
    method: str,
    url: str,
    headers: dict[str, str],
    timeout: float,
    follow: bool,
    max_hops: int = 5,
) -> HTTPResponse:
    """Follow redirects manually so each hop goes through the network policy."""
    current = url
    response = client.request(
        method, current, headers=headers, timeout=timeout, raise_for_status=False
    )
    hops = 0
    while follow and response.status in {301, 302, 303, 307, 308} and hops < max_hops:
        location = response.header("location")
        if not location:
            break
        current = urljoin(current, location)
        hops += 1
        response = client.request(
            method if response.status in {307, 308} else "GET",
            current,
            headers=headers,
            timeout=timeout,
            raise_for_status=False,
        )
    return response


def _format_response(response: HTTPResponse, max_bytes: int, as_text: bool) -> dict[str, Any]:
    """Package an HTTP response into a tool result payload."""
    content_type = response.content_type
    body_bytes = response.body
    payload: dict[str, Any] = {
        "url": response.url,
        "status": response.status,
        "ok": response.ok,
        "headers": dict(response.headers),
        "content_type": content_type,
        "bytes": len(body_bytes),
        "size_human": human_size(len(body_bytes)),
        "elapsed_ms": round(response.elapsed_ms, 2),
        "is_text": is_text_mime(content_type),
    }
    if as_text and (is_text_mime(content_type) or not content_type):
        text = body_bytes.decode("utf-8", errors="replace")
        truncated = len(text) > max_bytes
        payload["body"] = ellipsize(text, max_bytes)
        payload["truncated"] = truncated
        if "json" in content_type:
            try:
                payload["json"] = json.loads(text)
            except json.JSONDecodeError:
                payload["json"] = None
    else:
        payload["body_hex"] = body_bytes[: min(len(body_bytes), max_bytes // 2)].hex()
        payload["truncated"] = len(body_bytes) > max_bytes // 2
    return payload


@tool(category="web", risk="network", network=True, timeout=60.0, tags=["scrape", "markdown"])
def fetch_page(
    url: str,
    timeout: float = 30.0,
    max_chars: int = 40000,
    keep_links: bool = True,
    extract_main: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Fetch an HTML page and convert it to clean Markdown.

    Args:
        url: absolute page URL
        timeout: seconds to wait
        max_chars: maximum Markdown characters to return
        keep_links: preserve links as Markdown anchors
        extract_main: drop navigation, footer, and form blocks
    """
    ctx = context or ToolContext()
    _check_network(ctx, url)
    client = _client(ctx, timeout)
    try:
        response = _request_with_redirects(client, "GET", url, {}, timeout, True)
    finally:
        client.close()
    if not response.ok:
        raise ToolExecutionError(f"HTTP {response.status} fetching {url}")
    raw = response.text()
    markdown, meta = html_to_markdown(
        raw, base_url=response.url or url, keep_links=keep_links, extract_main=extract_main
    )
    return {
        "url": response.url,
        "status": response.status,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "links": meta.get("links", [])[:40],
        "markdown": ellipsize(markdown, max_chars),
        "chars": len(markdown),
        "truncated": len(markdown) > max_chars,
        "word_count": len(markdown.split()),
    }


def html_to_markdown(
    markup: str, *, base_url: str = "", keep_links: bool = True, extract_main: bool = True
) -> tuple[str, dict[str, Any]]:
    """Convert an HTML document into Markdown plus extracted metadata.

    Returns ``(markdown, meta)`` where ``meta`` carries the title, description,
    and absolute link targets found in the document.
    """
    text = normalize_newlines(markup or "")
    meta: dict[str, Any] = {"title": "", "description": "", "links": []}
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.DOTALL | re.IGNORECASE)
    if title_match:
        meta["title"] = html.unescape(_TAG.sub("", title_match.group(1))).strip()
    description = re.search(
        r"<meta[^>]+name=[\"']description[\"'][^>]+content=[\"']([^\"']*)[\"']", text, re.IGNORECASE
    ) or re.search(
        r"<meta[^>]+content=[\"']([^\"']*)[\"'][^>]+name=[\"']description[\"']", text, re.IGNORECASE
    )
    if description:
        meta["description"] = html.unescape(description.group(1)).strip()

    links: list[dict[str, str]] = []
    for match in re.finditer(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", text, re.DOTALL | re.IGNORECASE
    ):
        href, label = match.group(1), _TAG.sub("", match.group(2))
        label = html.unescape(collapse_whitespace(label)).strip()
        absolute = urljoin(base_url, href) if base_url else href
        if label and not href.startswith(("#", "javascript:", "mailto:")):
            links.append({"text": ellipsize(label, 120), "href": absolute})
    meta["links"] = links[:200]

    cleaned = _SCRIPT_STYLE.sub(" ", text)
    cleaned = _COMMENT.sub(" ", cleaned)
    cleaned = _HEAD_BLOCK.sub(" ", cleaned)
    if extract_main:
        cleaned = _NAV_BLOCK.sub(" ", cleaned)

    replacements: list[tuple[str, str]] = [
        (r"(?is)<br\s*/?>", "\n"),
        (r"(?is)</p>", "\n\n"),
        (r"(?is)</div>", "\n"),
        (r"(?is)</li>", "\n"),
        (r"(?is)<li[^>]*>", "- "),
        (r"(?is)<h1[^>]*>", "\n\n# "),
        (r"(?is)<h2[^>]*>", "\n\n## "),
        (r"(?is)<h3[^>]*>", "\n\n### "),
        (r"(?is)<h4[^>]*>", "\n\n#### "),
        (r"(?is)<h5[^>]*>", "\n\n##### "),
        (r"(?is)<h6[^>]*>", "\n\n###### "),
        (r"(?is)</h[1-6]>", "\n\n"),
        (r"(?is)<pre[^>]*>", "\n\n```\n"),
        (r"(?is)</pre>", "\n```\n\n"),
        (r"(?is)<code[^>]*>", "`"),
        (r"(?is)</code>", "`"),
        (r"(?is)<(strong|b)[^>]*>", "**"),
        (r"(?is)</(strong|b)>", "**"),
        (r"(?is)<(em|i)[^>]*>", "*"),
        (r"(?is)</(em|i)>", "*"),
        (r"(?is)<blockquote[^>]*>", "\n\n> "),
        (r"(?is)</blockquote>", "\n\n"),
        (r"(?is)<hr\s*/?>", "\n\n---\n\n"),
        (r"(?is)</tr>", "\n"),
        (r"(?is)</t[dh]>", " | "),
        (r"(?is)<tr[^>]*>", "| "),
    ]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    if keep_links:
        cleaned = re.sub(
            r"(?is)<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
            lambda match: (
                f"[{collapse_whitespace(_TAG.sub('', match.group(2))).strip()}]"
                f"({urljoin(base_url, match.group(1)) if base_url else match.group(1)})"
            ),
            cleaned,
        )
    else:
        cleaned = re.sub(r"(?is)<a\b[^>]*>(.*?)</a>", lambda m: _TAG.sub("", m.group(1)), cleaned)
    cleaned = _TAG.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = _MULTI_BLANK.sub("\n\n", cleaned).strip()
    return cleaned, meta


@tool(category="web", risk="network", network=True, timeout=120.0, tags=["download"])
def download_file(
    url: str,
    destination: str,
    timeout: float = 60.0,
    max_bytes: int = 104857600,
    overwrite: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Download a URL to a file inside the workspace.

    Args:
        url: absolute URL to download
        destination: file path to write, relative to the workspace
        timeout: seconds to wait
        max_bytes: refuse downloads larger than this
        overwrite: replace an existing file
    """
    ctx = context or ToolContext()
    _check_network(ctx, url)
    target = ctx.resolve_path(destination)
    if ctx.sandbox is not None:
        ctx.sandbox.check_write(target)
    if target.exists() and not overwrite:
        raise ToolExecutionError(f"{target} already exists; pass overwrite=true to replace it")
    client = _client(ctx, timeout)
    try:
        response = _request_with_redirects(client, "GET", url, {}, timeout, True)
    finally:
        client.close()
    if not response.ok:
        raise ToolExecutionError(f"HTTP {response.status} downloading {url}")
    if len(response.body) > max_bytes:
        raise ToolExecutionError(
            f"Download is {len(response.body)} bytes, above the {max_bytes} limit"
        )
    if ctx.dry_run:
        return {
            "url": url,
            "destination": str(target),
            "dry_run": True,
            "bytes": len(response.body),
        }
    ensure_dir(target.parent)
    atomic_write_bytes(target, response.body)
    return {
        "url": response.url,
        "destination": str(target),
        "bytes": len(response.body),
        "size_human": human_size(len(response.body)),
        "content_type": response.content_type,
        "mime": guess_mime(target),
    }


@tool(category="web", risk="read", network=False, tags=["parse", "url"])
def parse_url_tool(url: str) -> dict[str, Any]:
    """Break a URL into its components without making any request.

    Args:
        url: the URL to parse
    """
    from ...utils.urls import is_local_url, parse_url, redact_url

    parts = parse_url(url)
    return {
        **parts,
        "redacted": redact_url(url),
        "is_local": is_local_url(url),
        "origin": f"{parts['scheme']}://{parts['netloc']}",
    }


TOOLS: list[Any] = [http_get, http_post, fetch_page, download_file, parse_url_tool]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every web tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "download_file",
    "fetch_page",
    "html_to_markdown",
    "http_get",
    "http_post",
    "load",
    "parse_url_tool",
]
