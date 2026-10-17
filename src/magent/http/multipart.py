"""Bounded ``multipart/form-data`` encoding for media uploads.

Cortex LLMHoster accepts bounded multipart uploads for image, audio, video and
OCR endpoints, mirroring OpenAI's ``multipart`` transcription format. This
module builds those bodies without buffering more than necessary and refuses to
grow past a configured ceiling.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import PayloadTooLargeError
from ..utils.hashing import random_hex
from ..utils.mime import guess_mime

DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
"""Default ceiling for an encoded multipart body (512 MiB)."""

CRLF = b"\r\n"
"""Line separator mandated by RFC 2046."""


@dataclass(slots=True)
class MultipartField:
    """One part of a multipart body: either a value or a file."""

    name: str
    value: str | bytes | None = None
    filename: str | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def encoded_size(self) -> int:
        """Return the byte length this field contributes to the body."""
        payload = self.value
        if payload is None:
            return 0
        return len(payload) if isinstance(payload, bytes) else len(payload.encode("utf-8"))

    def as_bytes(self) -> bytes:
        """Return the field payload encoded as bytes."""
        if self.value is None:
            return b""
        return self.value if isinstance(self.value, bytes) else str(self.value).encode("utf-8")


class MultipartEncoder:
    """Streaming multipart encoder with a hard size ceiling.

    Parts are encoded lazily by :meth:`iter_body` so a large file never needs to
    live in memory twice. :attr:`content_type` carries the generated boundary.
    """

    def __init__(
        self,
        fields: Sequence[MultipartField] | None = None,
        *,
        boundary: str | None = None,
        max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        self.fields: list[MultipartField] = list(fields or [])
        self.boundary = boundary or f"----MAgentBoundary{random_hex(12)}"
        self.max_bytes = max_bytes

    # -- building ------------------------------------------------------------

    def add_field(
        self, name: str, value: Any, *, content_type: str | None = None
    ) -> MultipartEncoder:
        """Append a simple named value and return ``self`` for chaining."""
        self.fields.append(MultipartField(name=name, value=str(value), content_type=content_type))
        return self

    def add_file(
        self,
        name: str,
        path: str | Path,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> MultipartEncoder:
        """Append a file from disk and return ``self`` for chaining."""
        target = Path(path).expanduser()
        if not target.is_file():
            raise FileNotFoundError(f"No such file for upload: {target}")
        self.fields.append(
            MultipartField(
                name=name,
                value=None,
                filename=filename or target.name,
                content_type=content_type or guess_mime(target),
                headers={"__path__": str(target)},
            )
        )
        return self

    def add_bytes(
        self,
        name: str,
        data: bytes,
        *,
        filename: str = "upload.bin",
        content_type: str = "application/octet-stream",
    ) -> MultipartEncoder:
        """Append raw bytes as a file part and return ``self`` for chaining."""
        self.fields.append(
            MultipartField(name=name, value=data, filename=filename, content_type=content_type)
        )
        return self

    def add_json(self, name: str, payload: Any) -> MultipartEncoder:
        """Append a JSON encoded value part."""
        import json

        return self.add_field(
            name,
            json.dumps(payload, default=str, ensure_ascii=False),
            content_type="application/json",
        )

    # -- encoding ------------------------------------------------------------

    @property
    def content_type(self) -> str:
        """Return the ``Content-Type`` header value including the boundary."""
        return f"multipart/form-data; boundary={self.boundary}"

    def _part_header(self, part: MultipartField) -> bytes:
        """Build the header block that precedes one part body."""
        disposition = f'form-data; name="{_quote(part.name)}"'
        if part.filename:
            disposition += f'; filename="{_quote(part.filename)}"'
        lines = [f"Content-Disposition: {disposition}"]
        if part.content_type:
            lines.append(f"Content-Type: {part.content_type}")
        for key, value in part.headers.items():
            if key.startswith("__"):
                continue
            lines.append(f"{key}: {value}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")

    def iter_body(self, *, chunk_size: int = 1 << 16) -> Iterable[bytes]:
        """Yield the encoded body in bounded chunks."""
        boundary = f"--{self.boundary}".encode()
        total = 0
        for part in self.fields:
            block = boundary + CRLF + self._part_header(part)
            total += len(block)
            if total > self.max_bytes:
                raise PayloadTooLargeError(
                    f"Multipart body exceeds the {self.max_bytes} byte upload limit"
                )
            yield block
            source_path = part.headers.get("__path__")
            if source_path:
                with Path(source_path).open("rb") as stream:
                    while True:
                        chunk = stream.read(chunk_size)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > self.max_bytes:
                            raise PayloadTooLargeError(
                                f"Multipart body exceeds the {self.max_bytes} byte upload limit"
                            )
                        yield chunk
            elif part.value is not None:
                payload = part.as_bytes()
                total += len(payload)
                if total > self.max_bytes:
                    raise PayloadTooLargeError(
                        f"Multipart body exceeds the {self.max_bytes} byte upload limit"
                    )
                yield payload
            yield CRLF
            total += 2
        closing = boundary + b"--" + CRLF
        total += len(closing)
        if total > self.max_bytes:
            raise PayloadTooLargeError(
                f"Multipart body exceeds the {self.max_bytes} byte upload limit"
            )
        yield closing

    def encode(self) -> bytes:
        """Return the whole encoded body as a single bytes object."""
        return b"".join(self.iter_body())

    def content_length(self) -> int:
        """Compute the encoded body length without building it."""
        total = len(f"--{self.boundary}".encode()) + 2
        for part in self.fields:
            total += len(self._part_header(part))
            source_path = part.headers.get("__path__")
            if source_path:
                total += os.path.getsize(source_path)
            else:
                total += part.encoded_size()
            total += 2
            total += len(f"--{self.boundary}".encode()) + 2
        total += len(f"--{self.boundary}--".encode()) + 2
        return total

    def part_count(self) -> int:
        """Return how many parts the body contains."""
        return len(self.fields)


def _quote(value: str) -> str:
    """Escape a value for safe inclusion in a Content-Disposition header."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def guess_field_name(path: str | Path) -> str:
    """Guess the OpenAI-style form field name for a media file.

    ``audio.wav`` -> ``file``, ``photo.png`` -> ``image``; anything else falls
    back to ``file`` which is what OpenAI-compatible servers expect.
    """
    mime = guess_mime(path)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "file"
    if mime.startswith("video/"):
        return "video"
    return "file"


def encode_multipart(
    data: Mapping[str, Any] | None = None,
    files: Mapping[str, str | Path | bytes] | None = None,
    *,
    boundary: str | None = None,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> tuple[bytes, str]:
    """Encode simple mappings into ``(body, content_type)``.

    ``data`` values may be scalars or lists (repeated fields). ``files`` maps a
    field name to a path or raw bytes.
    """
    encoder = MultipartEncoder(boundary=boundary, max_bytes=max_bytes)
    for key, value in (data or {}).items():
        if isinstance(value, (list, tuple)):
            for item in value:
                encoder.add_field(key, item)
        else:
            encoder.add_field(key, value)
    for key, payload in (files or {}).items():
        if isinstance(payload, bytes):
            encoder.add_bytes(key, payload, filename=f"{key}.bin")
        else:
            encoder.add_file(key, payload)
    return encoder.encode(), encoder.content_type


def decode_multipart(body: bytes, content_type: str) -> dict[str, list[dict[str, Any]]]:
    """Parse a multipart body back into parts, for tests and the local server."""
    boundary = ""
    for parameter in content_type.split(";")[1:]:
        key, _, value = parameter.strip().partition("=")
        if key.lower() == "boundary":
            boundary = value.strip('"')
    if not boundary:
        return {"fields": [], "files": []}
    delimiter = f"--{boundary}".encode()
    fields: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for chunk in body.split(delimiter):
        chunk = chunk.strip(CRLF)
        if not chunk or chunk == b"--":
            continue
        header_blob, _, payload = chunk.partition(CRLF + CRLF)
        headers = _parse_part_headers(header_blob.decode("utf-8", "replace"))
        disposition = headers.get("content-disposition", "")
        name = _disposition_param(disposition, "name")
        filename = _disposition_param(disposition, "filename")
        record = {
            "name": name,
            "content_type": headers.get("content-type", ""),
            "size": len(payload),
        }
        if filename:
            record["filename"] = filename
            record["data"] = payload
            files.append(record)
        else:
            record["value"] = payload.decode("utf-8", "replace")
            fields.append(record)
    return {"fields": fields, "files": files}


def _parse_part_headers(blob: str) -> dict[str, str]:
    """Parse raw part headers into a lowercase keyed mapping."""
    headers: dict[str, str] = {}
    for line in blob.split("\r\n"):
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return headers


def _disposition_param(disposition: str, key: str) -> str:
    """Extract one parameter from a Content-Disposition header."""
    for piece in disposition.split(";")[1:]:
        name, _, value = piece.strip().partition("=")
        if name.strip().lower() == key:
            return value.strip().strip('"')
    return ""


def build_transcription_body(
    audio_path: str | Path,
    model: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Build an OpenAI-style ``/v1/audio/transcriptions`` multipart body."""
    data: dict[str, Any] = {"model": model}
    data.update(dict(extra or {}))
    return encode_multipart(data=data, files={"file": audio_path})


def build_image_body(
    prompt: str,
    model: str,
    *,
    image_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Build a text-to-image or image-to-image multipart body."""
    data: dict[str, Any] = {"prompt": prompt, "model": model}
    data.update(dict(extra or {}))
    files = {"image": image_path} if image_path else None
    return encode_multipart(data=data, files=files)


__all__ = [
    "CRLF",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "MultipartEncoder",
    "MultipartField",
    "build_image_body",
    "build_transcription_body",
    "decode_multipart",
    "encode_multipart",
    "guess_field_name",
]
