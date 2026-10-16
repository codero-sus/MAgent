"""Media inspection tools implemented with the standard library only.

MAgent does not decode images or transcode audio - it reads container headers
to report dimensions, duration, bit depth, and codec. That is enough for an
agent to decide whether a file is the right asset, whether it needs resizing,
or whether a transcription step is worth its cost, without pulling in Pillow or
ffmpeg as hard dependencies.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.files import file_sha256, human_size
from ...utils.mime import guess_mime, media_kind

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
"""First eight bytes of every PNG file."""

GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
"""Magic bytes for GIF files."""

WEBP_RIFF = b"RIFF"
"""Container magic for WebP, WAV, and AVI."""


@tool(category="media", risk="read", filesystem=True, tags=["image", "inspect", "core"])
def image_info(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Read an image's dimensions, format, and bit depth from its header.

    Supports PNG, JPEG, GIF, BMP, WebP, and ICO without any third-party
    decoder.

    Args:
        path: image file path inside the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such image: {target}")
    size = target.stat().st_size
    with target.open("rb") as stream:
        head = stream.read(64)
    info: dict[str, Any] = {
        "path": str(target),
        "relative": ctx.sandbox.relative(target) if ctx.sandbox else str(target),
        "bytes": size,
        "size_human": human_size(size),
        "mime": guess_mime(target),
        "format": "",
        "width": None,
        "height": None,
    }
    if head.startswith(PNG_SIGNATURE):
        info["format"] = "PNG"
        if len(head) >= 24:
            width, height = struct.unpack(">II", head[16:24])
            info["width"], info["height"] = width, height
            info["bit_depth"] = head[24] if len(head) > 24 else None
            info["color_type"] = head[25] if len(head) > 25 else None
            info["interlaced"] = bool(head[28]) if len(head) > 28 else False
    elif head.startswith(GIF_SIGNATURES):
        info["format"] = "GIF"
        width, height = struct.unpack("<HH", head[6:10])
        info["width"], info["height"] = width, height
        info["global_color_table"] = bool(head[10] & 0x80)
    elif head.startswith(b"\xff\xd8\xff"):
        info["format"] = "JPEG"
        dimensions = _jpeg_dimensions(target)
        if dimensions:
            info["width"], info["height"] = dimensions
        info["progressive"] = _jpeg_is_progressive(target)
    elif head.startswith(b"BM"):
        info["format"] = "BMP"
        if len(head) >= 26:
            width, height = struct.unpack("<ii", head[18:26])
            info["width"], info["height"] = width, abs(height)
            info["bit_depth"] = struct.unpack("<H", head[28:30])[0]
    elif head[:4] == WEBP_RIFF and head[8:12] == b"WEBP":
        info["format"] = "WebP"
        width, height = _webp_dimensions(target)
        info["width"], info["height"] = width, height
    elif head[:4] == b"\x00\x00\x01\x00":
        info["format"] = "ICO"
        count = struct.unpack("<H", head[4:6])[0]
        info["images"] = count
        if len(head) >= 10:
            info["width"] = head[6] or 256
            info["height"] = head[7] or 256
    else:
        info["format"] = "unknown"
    if info["width"] and info["height"]:
        info["megapixels"] = round(info["width"] * info["height"] / 1_000_000, 3)
        info["aspect_ratio"] = round(info["width"] / info["height"], 4)
        info["orientation"] = _orientation(info["width"], info["height"])
    return info


def _orientation(width: int, height: int) -> str:
    """Classify an image as landscape, portrait, or square."""
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Scan JPEG SOF markers for the encoded dimensions."""
    try:
        with path.open("rb") as stream:
            data = stream.read(4 * 1024 * 1024)
    except OSError:
        return None
    index = 2
    limit = len(data)
    while index < limit - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        index += 2 + length
    return None


def _jpeg_is_progressive(path: Path) -> bool:
    """Report whether a JPEG uses progressive encoding (SOF2)."""
    try:
        with path.open("rb") as stream:
            data = stream.read(4 * 1024 * 1024)
    except OSError:
        return False
    index = 2
    while index < len(data) - 4:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xC2:
            return True
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if 0xC0 <= marker <= 0xCF:
            return False
        index += 2 + length
    return False


def _webp_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Read dimensions from a WebP VP8, VP8L, or VP8X chunk."""
    try:
        with path.open("rb") as stream:
            data = stream.read(4096)
    except OSError:
        return None, None
    if len(data) < 30:
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    return None, None


@tool(category="media", risk="read", filesystem=True, tags=["audio", "inspect"])
def audio_info(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Read a WAV file's duration, sample rate, and channel count.

    Non-WAV formats are reported by container detection only, since decoding
    them would require a third-party library.

    Args:
        path: audio file path inside the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such audio file: {target}")
    size = target.stat().st_size
    info: dict[str, Any] = {
        "path": str(target),
        "bytes": size,
        "size_human": human_size(size),
        "mime": guess_mime(target),
        "kind": media_kind(target),
        "format": "",
        "duration_seconds": None,
    }
    with target.open("rb") as stream:
        head = stream.read(16)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        info["format"] = "WAV"
        try:
            with wave.open(str(target), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                info.update({
                    "channels": wav.getnchannels(),
                    "sample_width_bytes": wav.getsampwidth(),
                    "sample_rate": rate,
                    "frames": frames,
                    "duration_seconds": round(frames / rate, 4) if rate else None,
                    "duration_human": _duration(frames / rate) if rate else "",
                })
        except (wave.Error, EOFError) as exc:
            info["warning"] = f"could not parse WAV headers: {exc}"
    elif head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        info["format"] = "MP3"
        info["note"] = "duration requires decoding; use duration_estimate_bytes"
        info["duration_estimate_seconds"] = _estimate_duration(size, 128_000)
    elif head[4:8] == b"ftyp":
        info["format"] = "MP4/M4A"
        info["brand"] = head[8:12].decode("ascii", "replace")
    elif head[:4] == b"OggS":
        info["format"] = "OGG"
    elif head[:4] == b"fLaC":
        info["format"] = "FLAC"
    else:
        info["format"] = "unknown"
    return info


def _estimate_duration(size: int, bitrate: int) -> float:
    """Estimate audio duration from file size and an assumed bitrate."""
    if bitrate <= 0:
        return 0.0
    return round(size * 8 / bitrate, 2)


def _duration(seconds: float) -> str:
    """Format a duration as ``M:SS``."""
    total = int(max(0.0, seconds))
    return f"{total // 60}:{total % 60:02d}"


@tool(category="media", risk="read", filesystem=True, tags=["video", "inspect"])
def video_info(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Detect a video container and report what can be read without decoding.

    Args:
        path: video file path inside the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such video file: {target}")
    size = target.stat().st_size
    with target.open("rb") as stream:
        head = stream.read(32)
    info: dict[str, Any] = {
        "path": str(target),
        "bytes": size,
        "size_human": human_size(size),
        "mime": guess_mime(target),
        "format": "unknown",
    }
    if head[4:8] == b"ftyp":
        info["format"] = "MP4"
        info["brand"] = head[8:12].decode("ascii", "replace")
    elif head[:4] == b"\x1aE\xdf\xa3":
        info["format"] = "Matroska/WebM"
    elif head[:4] == WEBP_RIFF and head[8:12] == b"AVI ":
        info["format"] = "AVI"
    elif head[:4] == b"FLV\x01":
        info["format"] = "FLV"
    elif head[4:8] == b"moov":
        info["format"] = "MOV"
    return info


@tool(category="media", risk="read", filesystem=True, tags=["hash", "verify"])
def media_hash(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Hash a media file so duplicates and re-downloads can be detected.

    Args:
        path: file path inside the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such file: {target}")
    digest = file_sha256(target)
    return {
        "path": str(target),
        "sha256": digest,
        "short": digest[:16],
        "bytes": target.stat().st_size,
        "mime": guess_mime(target),
        "kind": media_kind(target),
    }


TOOLS: list[Any] = [image_info, audio_info, video_info, media_hash]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every media tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "GIF_SIGNATURES",
    "PNG_SIGNATURE",
    "TOOLS",
    "WEBP_RIFF",
    "audio_info",
    "image_info",
    "load",
    "media_hash",
    "video_info",
]
