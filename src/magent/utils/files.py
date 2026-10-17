"""Filesystem helpers with atomic writes, safe traversal, and size parsing."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..errors import StoreError

_SIZE_RE = re.compile(r"^([0-9]*\.?[0-9]+)([a-z]*)$")
"""Regex for :func:`parse_size`."""

_SIZE_UNITS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}
"""Multiplier lookup for :func:`parse_size`."""


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a :class:`Path`."""
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically via a sibling temp file + rename.

    A crash mid-write can never leave a half-written file at the destination.
    """
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except OSError as exc:  # pragma: no cover - platform specific
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise StoreError(f"Could not write {target}: {exc}") from exc
    with contextlib.suppress(OSError):
        os.chmod(target, 0o644)
    return target


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write raw ``data`` to ``path`` atomically."""
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except OSError as exc:  # pragma: no cover
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise StoreError(f"Could not write {target}: {exc}") from exc
    return target


def read_text_safe(path: str | Path, *, encoding: str = "utf-8", limit: int | None = None) -> str:
    """Read a text file, returning ``""`` when it is missing or unreadable."""
    target = Path(path).expanduser()
    try:
        with target.open("r", encoding=encoding, errors="replace", newline="") as stream:
            return stream.read(limit) if limit else stream.read()
    except OSError:
        return ""


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` on any failure."""
    text = read_text_safe(path)
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def write_json(path: str | Path, data: Any, *, indent: int | None = 2, sort_keys: bool = False) -> Path:
    """Atomically write ``data`` as JSON."""
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, default=str, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_size(value: str | int | float, *, default: int = 0) -> int:
    """Parse a human size such as ``"12 MB"`` or ``"512k"`` into bytes."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return default
    match = _SIZE_RE.match(text)
    if not match:
        return default
    number = float(match.group(1))
    unit = match.group(2) or "b"
    return int(number * _SIZE_UNITS.get(unit, 1))


def human_size(num_bytes: int | float, *, precision: int = 1) -> str:
    """Format a byte count as a human readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(size) < 1024.0 or unit == "PB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:.{precision}f} PB"


def safe_filename(name: str, *, max_length: int = 128, replacement: str = "_") -> str:
    """Sanitize ``name`` so it is safe to use as a single path component."""
    cleaned = "".join(
        char if (char.isalnum() or char in "._-") else replacement for char in str(name).strip()
    )
    cleaned = cleaned.strip("._")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    if not cleaned:
        cleaned = "file"
    return cleaned[:max_length] or "file"


def iter_files(
    root: str | Path,
    *,
    patterns: Iterable[str] = ("*",),
    exclude: Iterable[str] = (".git", "__pycache__", "node_modules", ".venv", "dist", "build"),
    follow_symlinks: bool = False,
    max_depth: int | None = None,
    limit: int | None = None,
) -> Iterator[Path]:
    """Walk ``root`` yielding files matching any glob in ``patterns``.

    Directories named in ``exclude`` are pruned before descent, so walking a
    repository never pays for ``node_modules`` or ``.git``.
    """
    base = Path(root).expanduser()
    if not base.exists():
        return
    if base.is_file():
        if any(fnmatch.fnmatch(base.name, pattern) for pattern in patterns):
            yield base
        return
    pattern_list = list(patterns)
    exclude_set = set(exclude)
    emitted = 0
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        if max_depth is not None and depth > max_depth:
            continue
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        directories: list[Path] = []
        for entry in entries:
            try:
                is_dir = entry.is_dir() and (follow_symlinks or not entry.is_symlink())
            except OSError:
                continue
            if is_dir:
                if entry.name not in exclude_set and not entry.name.startswith("."):
                    directories.append(entry)
                elif entry.name in exclude_set:
                    continue
                continue
            if entry.is_symlink() and not follow_symlinks:
                continue
            if any(fnmatch.fnmatch(entry.name, pattern) for pattern in pattern_list):
                yield entry
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
        stack.extend((item, depth + 1) for item in directories)


def count_files(root: str | Path, **kwargs: Any) -> int:
    """Return how many files :func:`iter_files` would yield."""
    return sum(1 for _ in iter_files(root, **kwargs))


def dir_size(root: str | Path) -> int:
    """Return the total size in bytes of every file under ``root``."""
    total = 0
    for path in iter_files(root, patterns=("*",), exclude=()):
        with contextlib.suppress(OSError):
            total += path.stat().st_size
    return total


def copy_tree(src: str | Path, dst: str | Path) -> Path:
    """Recursively copy ``src`` into ``dst``, merging with existing content."""
    target = Path(dst).expanduser()
    shutil.copytree(Path(src).expanduser(), target, dirs_exist_ok=True)
    return target


def remove_tree(path: str | Path, *, missing_ok: bool = True) -> bool:
    """Delete a directory tree; report whether anything was removed."""
    target = Path(path).expanduser()
    if not target.exists():
        if missing_ok:
            return False
        raise StoreError(f"No such directory: {target}")
    shutil.rmtree(target, ignore_errors=True)
    return True


@contextlib.contextmanager
def temp_dir(prefix: str = "magent-") -> Iterator[Path]:
    """Context manager yielding a temporary directory that is cleaned up."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def temp_file(suffix: str = "", prefix: str = "magent-") -> Iterator[Path]:
    """Context manager yielding a temporary file that is cleaned up."""
    handle, name = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(handle)
    path = Path(name)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def touch(path: str | Path) -> Path:
    """Create an empty file (and parents) or refresh its mtime."""
    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.touch(exist_ok=True)
    return target


def is_within(child: str | Path, parent: str | Path) -> bool:
    """Report whether ``child`` resolves to a location inside ``parent``.

    This is the core primitive behind the tool sandbox: symlinks and ``..``
    segments are resolved before comparison, so they cannot escape the jail.
    """
    try:
        resolved_child = Path(child).expanduser().resolve()
        resolved_parent = Path(parent).expanduser().resolve()
    except OSError:
        return False
    return resolved_child == resolved_parent or resolved_parent in resolved_child.parents


def free_space(path: str | Path = ".") -> int:
    """Return free bytes on the filesystem holding ``path``."""
    try:
        return shutil.disk_usage(Path(path).expanduser()).free
    except OSError:
        return 0


def backup_file(path: str | Path, *, suffix: str = ".bak") -> Path | None:
    """Copy ``path`` to ``path + suffix``; return the backup or ``None``."""
    target = Path(path).expanduser()
    if not target.is_file():
        return None
    destination = target.with_name(target.name + suffix)
    shutil.copy2(target, destination)
    return destination


__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "backup_file",
    "copy_tree",
    "count_files",
    "dir_size",
    "ensure_dir",
    "file_sha256",
    "free_space",
    "human_size",
    "is_within",
    "iter_files",
    "parse_size",
    "read_json",
    "read_text_safe",
    "remove_tree",
    "safe_filename",
    "temp_dir",
    "temp_file",
    "touch",
    "write_json",
]
