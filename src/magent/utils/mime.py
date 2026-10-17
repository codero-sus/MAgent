"""MIME type detection and text/binary classification."""

from __future__ import annotations

import mimetypes
from pathlib import Path

_EXTRA_MIMES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".toml": "application/toml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".ts": "text/plain",
    ".tsx": "text/plain",
    ".jsx": "text/plain",
    ".vue": "text/plain",
    ".svelte": "text/plain",
    ".gguf": "application/octet-stream",
    ".safetensors": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".onnx": "application/octet-stream",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".heic": "image/heic",
    ".m4a": "audio/mp4",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".ipynb": "application/x-ipynb+json",
    ".lock": "text/plain",
    ".gitignore": "text/plain",
    ".env": "text/plain",
}
"""Extension overrides where the stdlib table is missing or wrong."""

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".jsx", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".svg", ".css", ".scss", ".less", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java",
    ".kt", ".kts", ".scala", ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".sql", ".graphql",
    ".gql", ".proto", ".tex", ".bib", ".diff", ".patch", ".makefile", ".mk", ".cmake", ".dockerfile",
    ".env", ".gitignore", ".gitattributes", ".editorconfig", ".lock", ".log", ".vue", ".svelte",
}
"""Extensions always treated as text."""

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tif", ".tiff", ".webp", ".avif", ".heic",
    ".mp3", ".wav", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".wma",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".zst",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".jar", ".war",
    ".gguf", ".safetensors", ".onnx", ".pt", ".pth", ".ckpt", ".h5", ".npz", ".npy",
    ".db", ".sqlite", ".sqlite3", ".mdb",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
}
"""Extensions always treated as binary."""

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif", ".heic"}
"""Extensions accepted as vision inputs."""

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".webm"}
"""Extensions accepted as audio inputs."""

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
"""Extensions accepted as video inputs."""

MAX_SNIFF_BYTES = 8192
"""How many bytes :func:`sniff_text` inspects before deciding."""

mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("application/toml", ".toml")
mimetypes.add_type("application/yaml", ".yaml")


def guess_mime(path: str | Path, *, default: str = "application/octet-stream") -> str:
    """Return the MIME type for ``path``, preferring the override table."""
    name = Path(path).name.lower()
    if name in {"makefile", "dockerfile", "license", "readme"}:
        return "text/plain"
    suffix = Path(path).suffix.lower()
    if suffix in _EXTRA_MIMES:
        return _EXTRA_MIMES[suffix]
    guessed, _ = mimetypes.guess_type(name)
    return guessed or default


def ext_to_mime(suffix: str) -> str:
    """Return the MIME type for a bare extension such as ``".md"``."""
    normalized = suffix if suffix.startswith(".") else f".{suffix}"
    return guess_mime(f"file{normalized.lower()}")


def mime_to_ext(mime_type: str, *, default: str = ".bin") -> str:
    """Return a filename extension for a MIME type."""
    lowered = str(mime_type).split(";")[0].strip().lower()
    for suffix, candidate in _EXTRA_MIMES.items():
        if candidate == lowered:
            return suffix
    guessed = mimetypes.guess_extension(lowered)
    return guessed or default


def is_binary(path: str | Path) -> bool:
    """Report whether ``path`` should be treated as binary by extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _BINARY_EXTENSIONS:
        return True
    if suffix in _TEXT_EXTENSIONS:
        return False
    return not guess_mime(path).startswith(("text/", "application/json", "application/xml"))


def is_text_mime(mime_type: str) -> bool:
    """Report whether a MIME type carries human readable text."""
    lowered = str(mime_type).split(";")[0].strip().lower()
    return lowered.startswith(("text/", "application/json", "application/xml", "application/yaml",
                               "application/toml", "application/javascript", "application/x-ndjson"))


def is_image(path: str | Path) -> bool:
    """Report whether ``path`` looks like a supported image file."""
    return Path(path).suffix.lower() in _IMAGE_EXTENSIONS


def is_audio(path: str | Path) -> bool:
    """Report whether ``path`` looks like a supported audio file."""
    return Path(path).suffix.lower() in _AUDIO_EXTENSIONS


def is_video(path: str | Path) -> bool:
    """Report whether ``path`` looks like a supported video file."""
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS


def media_kind(path: str | Path) -> str:
    """Classify ``path`` as ``image``, ``audio``, ``video``, ``text`` or ``binary``."""
    if is_image(path):
        return "image"
    if is_audio(path):
        return "audio"
    if is_video(path):
        return "video"
    if is_binary(path):
        return "binary"
    return "text"


def sniff_text(data: bytes, limit: int = MAX_SNIFF_BYTES) -> bool:
    """Heuristically decide whether raw ``data`` is text by byte inspection."""
    sample = data[:limit]
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    try:
        sample.decode("latin-1")
    except UnicodeDecodeError:
        return False
    control = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return control / len(sample) < 0.02


def charset_from_content_type(content_type: str, default: str = "utf-8") -> str:
    """Extract the charset parameter from a ``Content-Type`` header."""
    for part in str(content_type).split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"' ") or default
    return default


__all__ = [
    "MAX_SNIFF_BYTES",
    "charset_from_content_type",
    "ext_to_mime",
    "guess_mime",
    "is_audio",
    "is_binary",
    "is_image",
    "is_text_mime",
    "is_video",
    "media_kind",
    "mime_to_ext",
    "sniff_text",
]
