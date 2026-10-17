"""Filesystem tools: read, write, edit, search, inspect.

Every tool here routes through :class:`~magent.tools.sandbox.Sandbox`, so a path
outside the workspace jail, a symlink escape, or an oversized read is refused
before any I/O happens. Writes are atomic and diffs are returned so the operator
can see exactly what changed.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
from pathlib import Path
from typing import Any

from ...errors import PermissionDeniedError, ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.diff import diff_summary, unified_diff
from ...utils.files import atomic_write_text, human_size
from ...utils.hashing import file_sha256, short_hash
from ...utils.mime import guess_mime, is_binary, media_kind
from ...utils.text import chunk_text, ellipsize, normalize_newlines


def _sandbox_check(context: ToolContext, path: str | Path, action: str, size: int = 0) -> Path:
    """Authorize one filesystem action and return the resolved path."""
    sandbox = context.sandbox
    if sandbox is None:
        resolved = context.resolve_path(path)
        return resolved
    if action == "read":
        sandbox.check_read(path if Path(str(path)).is_absolute() else context.workspace / str(path))
    elif action in {"write", "append"}:
        sandbox.check_write(
            path if Path(str(path)).is_absolute() else context.workspace / str(path), size
        )
    else:
        sandbox.check_path(
            path if Path(str(path)).is_absolute() else context.workspace / str(path), action=action
        )
    return context.resolve_path(path)


def _number_lines(text: str, start: int = 1) -> str:
    """Prefix every line with a right-aligned line number."""
    lines = text.split("\n")
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{index + start:>{width}}│{line}" for index, line in enumerate(lines))


@tool(category="files", risk="read", filesystem=True, tags=["read", "core"])
def read_file(
    path: str,
    encoding: str = "utf-8",
    max_bytes: int = 0,
    start_line: int = 0,
    end_line: int = 0,
    number_lines: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Read a text file from the workspace.

    Args:
        path: file path, relative to the workspace or absolute inside it
        encoding: text encoding to decode with
        max_bytes: stop after this many bytes, 0 for the configured limit
        start_line: first line to return, 1-based, 0 to start at the top
        end_line: last line to return, 1-based, 0 to read to the end
        number_lines: prefix each line with its number
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    if not target.exists():
        raise ToolExecutionError(
            f"No such file: {ctx.sandbox.relative(target) if ctx.sandbox else target}"
        )
    if target.is_dir():
        raise ToolExecutionError(f"{target} is a directory, use list_dir instead")
    if is_binary(target):
        stat = target.stat()
        return {
            "path": str(target),
            "binary": True,
            "kind": media_kind(target),
            "mime": guess_mime(target),
            "size": stat.st_size,
            "size_human": human_size(stat.st_size),
            "sha256": file_sha256(target),
        }
    limit = max_bytes or (ctx.sandbox.policy.max_read_bytes if ctx.sandbox else 2 * 1024 * 1024)
    with target.open("r", encoding=encoding, errors="replace", newline="") as stream:
        text = stream.read(limit + 1)
    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
    text = normalize_newlines(text)
    lines = text.split("\n")
    total = len(lines)
    if start_line or end_line:
        first = max(0, (start_line or 1) - 1)
        last = end_line or total
        lines = lines[first:last]
        text = "\n".join(lines)
    body = _number_lines(text, start_line or 1) if number_lines else text
    return {
        "path": str(target),
        "content": ellipsize(body, ctx.output_limit) if body else "",
        "lines": total,
        "returned_lines": len(lines),
        "chars": len(text),
        "truncated": truncated or len(body) > ctx.output_limit,
        "encoding": encoding,
        "mime": guess_mime(target),
        "sha256_short": short_hash(text),
    }


@tool(category="files", risk="write", filesystem=True, destructive=False, tags=["write", "core"])
def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    append: bool = False,
    create_parents: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Write or append to a file inside the workspace.

    Args:
        path: destination file path
        content: text to write
        encoding: text encoding to use
        append: append instead of replacing the file
        create_parents: create missing parent directories
    """
    ctx = context or ToolContext()
    payload = normalize_newlines(content or "")
    target = _sandbox_check(
        ctx, path, "append" if append else "write", len(payload.encode(encoding))
    )
    if ctx.dry_run:
        return {"path": str(target), "dry_run": True, "bytes": len(payload)}
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    previous = ""
    if append and target.is_file():
        previous = target.read_text(encoding=encoding, errors="replace")
    elif target.is_file():
        try:
            previous = target.read_text(encoding=encoding, errors="replace")
        except OSError:
            previous = ""
    if append:
        with target.open("a", encoding=encoding, newline="\n") as stream:
            stream.write(payload)
        final = previous + payload
    else:
        atomic_write_text(target, payload, encoding=encoding)
        final = payload
    return {
        "path": str(target),
        "bytes_written": len(payload.encode(encoding)),
        "appended": append,
        "lines": final.count("\n") + (0 if final.endswith("\n") or not final else 1),
        "diff": diff_summary(previous, final)
        if not append
        else f"+{payload.count(chr(10)) + 1} lines",
        "sha256_short": short_hash(final),
    }


@tool(category="files", risk="write", filesystem=True, tags=["edit", "patch", "core"])
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    fuzzy: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Replace an exact snippet of text inside a file.

    Safer than rewriting a whole file: the change is surgical, the diff is
    returned, and nothing is written unless the snippet is found.

    Args:
        path: file to edit
        old_text: text to find
        new_text: replacement text
        replace_all: replace every occurrence instead of just the first
        fuzzy: tolerate indentation and whitespace differences when matching
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    if not target.is_file():
        raise ToolExecutionError(f"No such file: {target}")
    original = target.read_text(encoding="utf-8", errors="replace")
    if not old_text:
        raise ToolExecutionError("old_text must not be empty")
    updated, count = _replace_snippet(
        original, old_text, new_text, replace_all=replace_all, fuzzy=fuzzy
    )
    if count == 0:
        from ...utils.diff import closest_match

        candidates = [line for line in original.split("\n") if line.strip()][:500]
        hint = closest_match(old_text.split("\n")[0], candidates, cutoff=0.5)
        raise ToolExecutionError(
            f"Snippet not found in {target.name}."
            + (f" Closest line: {hint.strip()[:120]!r}" if hint else "")
        )
    _sandbox_check(ctx, path, "write", len(updated.encode("utf-8")))
    if not ctx.dry_run:
        atomic_write_text(target, updated)
    return {
        "path": str(target),
        "replacements": count,
        "diff": unified_diff(
            original,
            updated,
            from_label=f"{target.name} (before)",
            to_label=f"{target.name} (after)",
        ),
        "summary": diff_summary(original, updated),
        "dry_run": ctx.dry_run,
    }


def _replace_snippet(
    text: str, old: str, new: str, *, replace_all: bool, fuzzy: bool
) -> tuple[str, int]:
    """Replace ``old`` with ``new``, optionally tolerating whitespace drift."""
    if old in text:
        if replace_all:
            return text.replace(old, new), text.count(old)
        return text.replace(old, new, 1), 1
    if not fuzzy:
        return text, 0
    normalized_old = re.sub(r"[ \t]+", " ", old).strip()
    pattern_parts = [re.escape(part.strip()) for part in normalized_old.split("\n") if part.strip()]
    if not pattern_parts:
        return text, 0
    pattern = re.compile(r"[ \t]*" + r"\n[ \t]*".join(pattern_parts))
    matches = list(pattern.finditer(text))
    if not matches:
        return text, 0
    if not replace_all:
        matches = matches[:1]
    result = text
    for match in reversed(matches):
        result = result[: match.start()] + new + result[match.end() :]
    return result, len(matches)


@tool(category="files", risk="read", filesystem=True, tags=["list", "core"])
def list_dir(
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    max_entries: int = 500,
    show_hidden: bool = False,
    sort_by: str = "name",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """List directory contents with optional recursion and filtering.

    Args:
        path: directory to list
        pattern: glob pattern for entry names
        recursive: descend into subdirectories
        max_entries: stop after this many entries
        show_hidden: include dotfiles
        sort_by: name, size, or modified
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    if not target.exists():
        raise ToolExecutionError(f"No such directory: {target}")
    if not target.is_dir():
        raise ToolExecutionError(f"Not a directory: {target}")
    entries: list[dict[str, Any]] = []
    if recursive:
        walker: Any = target.rglob(pattern)
    else:
        walker = target.glob(pattern)
    for entry in walker:
        if len(entries) >= max_entries:
            break
        if not show_hidden and any(
            part.startswith(".") for part in entry.relative_to(target).parts
        ):
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": entry.name,
                "path": str(entry),
                "relative": str(entry.relative_to(target)),
                "is_dir": entry.is_dir(),
                "size": stat.st_size,
                "size_human": human_size(stat.st_size),
                "modified": stat.st_mtime,
                "mime": "" if entry.is_dir() else guess_mime(entry),
            }
        )
    key = {"size": lambda item: -item["size"], "modified": lambda item: -item["modified"]}.get(
        sort_by, lambda item: item["relative"].lower()
    )
    entries.sort(key=key)
    directories = sum(1 for item in entries if item["is_dir"])
    return {
        "path": str(target),
        "count": len(entries),
        "files": len(entries) - directories,
        "directories": directories,
        "total_bytes": sum(item["size"] for item in entries if not item["is_dir"]),
        "truncated": len(entries) >= max_entries,
        "entries": entries,
    }


@tool(category="files", risk="read", filesystem=True, tags=["search", "glob"])
def glob_files(
    pattern: str, root: str = ".", max_results: int = 200, context: ToolContext | None = None
) -> dict[str, Any]:
    """Find files matching a glob pattern.

    Args:
        pattern: glob pattern such as "**/*.py"
        root: directory to search under
        max_results: stop after this many matches
    """
    ctx = context or ToolContext()
    base = _sandbox_check(ctx, root, "read")
    matches: list[str] = []
    for entry in sorted(base.rglob("*")):
        if len(matches) >= max_results:
            break
        if not entry.is_file():
            continue
        relative = str(entry.relative_to(base))
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(entry.name, pattern):
            matches.append(relative)
    return {
        "root": str(base),
        "pattern": pattern,
        "count": len(matches),
        "matches": matches,
        "truncated": len(matches) >= max_results,
    }


@tool(category="files", risk="read", filesystem=True, tags=["grep", "search", "core"])
def grep(
    pattern: str,
    path: str = ".",
    include: str = "*",
    ignore_case: bool = False,
    max_results: int = 100,
    context_lines: int = 0,
    regex: bool = True,
    exclude_dirs: str = ".git,__pycache__,node_modules,.venv,dist,build",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Search file contents for a pattern, ripgrep style.

    Args:
        pattern: text or regular expression to find
        path: file or directory to search
        include: glob for which filenames to consider
        ignore_case: match case-insensitively
        max_results: stop after this many matching lines
        context_lines: lines of surrounding context to include
        regex: treat the pattern as a regular expression
        exclude_dirs: comma separated directory names to skip
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        matcher = re.compile(pattern, flags) if regex else None
    except re.error as exc:
        raise ToolExecutionError(f"Invalid regex {pattern!r}: {exc}") from exc
    excluded = {item.strip() for item in exclude_dirs.split(",") if item.strip()}
    matches: list[dict[str, Any]] = []
    files_scanned = 0

    def scan_file(file_path: Path) -> None:
        """Search one file and append its matching lines."""
        nonlocal files_scanned
        files_scanned += 1
        if is_binary(file_path):
            return
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lines = text.split("\n")
        for index, line in enumerate(lines):
            hit = (
                matcher.search(line)
                if matcher
                else (pattern.lower() in line.lower() if ignore_case else pattern in line)
            )
            if not hit:
                continue
            before = lines[max(0, index - context_lines) : index] if context_lines else []
            after = lines[index + 1 : index + 1 + context_lines] if context_lines else []
            matches.append(
                {
                    "path": str(file_path),
                    "line": index + 1,
                    "text": ellipsize(line.rstrip(), 400),
                    "before": [item.rstrip() for item in before],
                    "after": [item.rstrip() for item in after],
                }
            )
            if len(matches) >= max_results:
                return

    if target.is_file():
        scan_file(target)
    else:
        for entry in sorted(target.rglob("*")):
            if len(matches) >= max_results:
                break
            if not entry.is_file():
                continue
            if any(part in excluded for part in entry.parts):
                continue
            if not fnmatch.fnmatch(entry.name, include):
                continue
            scan_file(entry)
    return {
        "pattern": pattern,
        "root": str(target),
        "matches": len(matches),
        "files_scanned": files_scanned,
        "truncated": len(matches) >= max_results,
        "results": matches[:max_results],
    }


@tool(category="files", risk="read", filesystem=True, tags=["stat", "inspect"])
def file_info(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Report metadata about a file or directory.

    Args:
        path: file or directory to inspect
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    if not target.exists():
        raise ToolExecutionError(f"No such path: {target}")
    stat = target.stat()
    info: dict[str, Any] = {
        "path": str(target),
        "name": target.name,
        "exists": True,
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "is_symlink": target.is_symlink(),
        "size": stat.st_size,
        "size_human": human_size(stat.st_size),
        "modified": stat.st_mtime,
        "accessed": stat.st_atime,
        "mode": oct(stat.st_mode)[-4:],
        "mime": guess_mime(target) if target.is_file() else "inode/directory",
        "kind": media_kind(target) if target.is_file() else "directory",
        "suffix": target.suffix,
    }
    if target.is_file():
        info["sha256"] = file_sha256(target)
        info["lines"] = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
    if target.is_dir():
        children = list(target.iterdir())
        info["entries"] = len(children)
        info["directories"] = sum(1 for item in children if item.is_dir())
    return info


@tool(category="files", risk="destructive", filesystem=True, destructive=True, tags=["delete"])
def delete_path(
    path: str, recursive: bool = False, context: ToolContext | None = None
) -> dict[str, Any]:
    """Delete a file or empty directory inside the workspace.

    Args:
        path: file or directory to delete
        recursive: also delete non-empty directories
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "delete")
    if not target.exists():
        return {"path": str(target), "deleted": False, "reason": "did not exist"}
    if target == ctx.workspace or (ctx.sandbox and target in ctx.sandbox.roots()):
        raise PermissionDeniedError("Refusing to delete the workspace root")
    if ctx.dry_run:
        return {"path": str(target), "deleted": False, "dry_run": True}
    if target.is_dir():
        if recursive:
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.rmdir()
            except OSError as exc:
                raise ToolExecutionError(
                    f"Directory not empty: {target}. Pass recursive=true to delete it."
                ) from exc
    else:
        target.unlink()
    return {"path": str(target), "deleted": True, "recursive": recursive}


@tool(category="files", risk="write", filesystem=True, tags=["copy", "move"])
def copy_path(
    source: str, destination: str, overwrite: bool = False, context: ToolContext | None = None
) -> dict[str, Any]:
    """Copy a file or directory within the workspace.

    Args:
        source: existing path to copy
        destination: new path
        overwrite: replace the destination if it exists
    """
    ctx = context or ToolContext()
    src = _sandbox_check(ctx, source, "read")
    dst = _sandbox_check(ctx, destination, "write")
    if not src.exists():
        raise ToolExecutionError(f"Source does not exist: {src}")
    if dst.exists() and not overwrite:
        raise ToolExecutionError(f"Destination exists: {dst}. Pass overwrite=true to replace it.")
    if ctx.dry_run:
        return {"source": str(src), "destination": str(dst), "dry_run": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "copied": True,
        "size": src.stat().st_size if src.is_file() else 0,
    }


@tool(category="files", risk="write", filesystem=True, tags=["move", "rename"])
def move_path(
    source: str, destination: str, overwrite: bool = False, context: ToolContext | None = None
) -> dict[str, Any]:
    """Move or rename a file or directory within the workspace.

    Args:
        source: existing path
        destination: new path
        overwrite: replace the destination if it exists
    """
    ctx = context or ToolContext()
    src = _sandbox_check(ctx, source, "read")
    dst = _sandbox_check(ctx, destination, "write")
    if not src.exists():
        raise ToolExecutionError(f"Source does not exist: {src}")
    if dst.exists() and not overwrite:
        raise ToolExecutionError(f"Destination exists: {dst}. Pass overwrite=true to replace it.")
    if ctx.dry_run:
        return {"source": str(src), "destination": str(dst), "dry_run": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"source": str(src), "destination": str(dst), "moved": True}


@tool(category="files", risk="write", filesystem=True, tags=["mkdir"])
def make_dir(path: str, parents: bool = True, context: ToolContext | None = None) -> dict[str, Any]:
    """Create a directory inside the workspace.

    Args:
        path: directory to create
        parents: create missing parent directories
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "write")
    if ctx.dry_run:
        return {"path": str(target), "dry_run": True}
    target.mkdir(parents=parents, exist_ok=True)
    return {"path": str(target), "created": True, "exists": target.is_dir()}


@tool(category="files", risk="read", filesystem=True, tags=["diff", "compare"])
def diff_files(
    left: str,
    right: str,
    context_lines: int = 3,
    color: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Show a unified diff between two files.

    Args:
        left: first file path
        right: second file path
        context_lines: unchanged lines to show around each change
        color: emit ANSI coloured diff output
    """
    ctx = context or ToolContext()
    first = _sandbox_check(ctx, left, "read")
    second = _sandbox_check(ctx, right, "read")
    left_text = first.read_text(encoding="utf-8", errors="replace") if first.is_file() else ""
    right_text = second.read_text(encoding="utf-8", errors="replace") if second.is_file() else ""
    return {
        "left": str(first),
        "right": str(second),
        "identical": left_text == right_text,
        "diff": unified_diff(
            left_text,
            right_text,
            from_label=first.name,
            to_label=second.name,
            context=context_lines,
            color=color,
        ),
        "summary": diff_summary(left_text, right_text),
    }


@tool(category="files", risk="read", filesystem=True, tags=["tree"])
def directory_tree(
    path: str = ".",
    max_depth: int = 3,
    max_entries: int = 400,
    exclude: str = ".git,__pycache__,node_modules,.venv,dist,build",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Render a directory tree as indented text.

    Args:
        path: root directory
        max_depth: how many levels to descend
        max_entries: stop after this many entries
        exclude: comma separated directory names to skip
    """
    ctx = context or ToolContext()
    root = _sandbox_check(ctx, path, "read")
    excluded = {item.strip() for item in exclude.split(",") if item.strip()}
    lines: list[str] = [root.name + "/"]
    count = 0

    def walk(directory: Path, depth: int, prefix: str) -> None:
        """Recursively render one directory level."""
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            entries = sorted(
                directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower())
            )
        except OSError:
            return
        for index, entry in enumerate(entries):
            if count >= max_entries:
                return
            if entry.name in excluded or (entry.is_dir() and entry.name.startswith(".")):
                continue
            last = index == len(entries) - 1
            connector = "└── " if last else "├── "
            count += 1
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                walk(entry, depth + 1, prefix + ("    " if last else "│   "))
            else:
                size = entry.stat().st_size
                lines.append(f"{prefix}{connector}{entry.name} ({human_size(size)})")

    walk(root, 1, "")
    return {
        "root": str(root),
        "entries": count,
        "truncated": count >= max_entries,
        "tree": "\n".join(lines),
    }


@tool(category="files", risk="read", filesystem=True, tags=["hash", "integrity"])
def hash_file(
    path: str, algorithm: str = "sha256", context: ToolContext | None = None
) -> dict[str, Any]:
    """Compute a checksum of a file.

    Args:
        path: file to hash
        algorithm: sha256, sha1, md5, or blake2b
    """
    import hashlib

    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    if not target.is_file():
        raise ToolExecutionError(f"Not a file: {target}")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ToolExecutionError(f"Unsupported algorithm {algorithm!r}") from exc
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(target),
        "algorithm": algorithm,
        "digest": digest.hexdigest(),
        "size": target.stat().st_size,
    }


@tool(category="files", risk="read", filesystem=True, tags=["chunk", "rag"])
def chunk_file(
    path: str, max_chars: int = 1200, overlap: int = 120, context: ToolContext | None = None
) -> dict[str, Any]:
    """Split a text file into overlapping chunks suitable for embedding.

    Args:
        path: file to chunk
        max_chars: maximum characters per chunk
        overlap: characters of overlap between consecutive chunks
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    text = target.read_text(encoding="utf-8", errors="replace")
    pieces = chunk_text(text, max_chars=max_chars, overlap=overlap)
    return {
        "path": str(target),
        "chunks": len(pieces),
        "chars": len(text),
        "max_chars": max_chars,
        "overlap": overlap,
        "pieces": [
            {"index": index, "chars": len(piece), "text": ellipsize(piece, 500)}
            for index, piece in enumerate(pieces[:50])
        ],
    }


@tool(category="files", risk="read", filesystem=True, tags=["search", "content"])
def find_in_files(
    query: str,
    paths: str = ".",
    file_pattern: str = "*",
    max_results: int = 50,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Find files whose content mentions a literal query, ranked by frequency.

    Args:
        query: literal text to look for
        paths: comma separated files or directories to search
        file_pattern: glob for filenames
        max_results: maximum files to report
    """
    ctx = context or ToolContext()
    results: list[dict[str, Any]] = []
    needle = query.lower()
    for raw in [item.strip() for item in paths.split(",") if item.strip()]:
        target = _sandbox_check(ctx, raw, "read")
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        for entry in candidates:
            if len(results) >= max_results:
                break
            if not entry.is_file() or not fnmatch.fnmatch(entry.name, file_pattern):
                continue
            if is_binary(entry):
                continue
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            occurrences = text.lower().count(needle)
            if occurrences:
                first = text.lower().find(needle)
                results.append(
                    {
                        "path": str(entry),
                        "occurrences": occurrences,
                        "snippet": ellipsize(text[max(0, first - 60) : first + 160].strip(), 220),
                    }
                )
    results.sort(key=lambda item: -item["occurrences"])
    return {"query": query, "files": len(results), "results": results[:max_results]}


@tool(category="files", risk="read", filesystem=True, tags=["size", "usage"])
def disk_usage(path: str = ".", context: ToolContext | None = None) -> dict[str, Any]:
    """Report the size of a directory tree and the free space on its volume.

    Args:
        path: directory to measure
    """
    ctx = context or ToolContext()
    target = _sandbox_check(ctx, path, "read")
    total = 0
    files = 0
    directories = 0
    by_extension: dict[str, int] = {}
    if target.is_file():
        total = target.stat().st_size
        files = 1
    else:
        for entry in target.rglob("*"):
            if entry.is_dir():
                directories += 1
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            files += 1
            total += size
            key = entry.suffix.lower() or "(none)"
            by_extension[key] = by_extension.get(key, 0) + size
    usage = shutil.disk_usage(target)
    top = sorted(by_extension.items(), key=lambda item: -item[1])[:10]
    return {
        "path": str(target),
        "files": files,
        "directories": directories,
        "bytes": total,
        "size_human": human_size(total),
        "by_extension": {key: human_size(value) for key, value in top},
        "volume_total": human_size(usage.total),
        "volume_free": human_size(usage.free),
        "volume_used_percent": round(usage.used / usage.total * 100, 2) if usage.total else 0.0,
    }


TOOLS: list[Any] = [
    read_file,
    write_file,
    edit_file,
    list_dir,
    glob_files,
    grep,
    file_info,
    delete_path,
    copy_path,
    move_path,
    make_dir,
    diff_files,
    directory_tree,
    hash_file,
    chunk_file,
    find_in_files,
    disk_usage,
]
"""Every tool defined in this module, for the registry loader."""


def load(registry: Any) -> int:
    """Register every filesystem tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "TOOLS",
    "chunk_file",
    "copy_path",
    "delete_path",
    "diff_files",
    "directory_tree",
    "disk_usage",
    "edit_file",
    "file_info",
    "find_in_files",
    "glob_files",
    "grep",
    "hash_file",
    "list_dir",
    "load",
    "make_dir",
    "move_path",
    "read_file",
    "write_file",
]
