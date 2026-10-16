"""Git repository tools, implemented with the ``git`` CLI through the sandbox.

Every command goes through :meth:`Sandbox.check_command`, so an allowlist or
denylist policy applies, and mutating git operations are marked ``risk="write"``
so the approval layer can gate them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool

MAX_GIT_OUTPUT = 200_000
"""Ceiling on captured git output."""

DANGEROUS_GIT_ARGS = re.compile(
    r"(--exec|--upload-pack|--receive-pack|--git-dir\s*=\s*/|-c\s+core\.sshCommand|"
    r"--force\s+.*--delete|clean\s+-[a-z]*f[a-z]*d)",
    re.IGNORECASE,
)
"""Argument patterns that could escape the repository or delete work."""


def _git(args: list[str], context: ToolContext, *, cwd: Path | None = None,
         timeout: float = 60.0) -> dict[str, Any]:
    """Run a git command inside the workspace and return a structured result."""
    if not args:
        raise ToolExecutionError("No git arguments supplied")
    command = ["git", *args]
    joined = " ".join(command)
    if DANGEROUS_GIT_ARGS.search(joined):
        raise ToolExecutionError("Refusing git arguments that could escape the repository")
    workspace = cwd or context.workspace
    if context.sandbox is not None:
        context.sandbox.check_command(joined)
    env = context.sandbox.env_for_subprocess() if context.sandbox is not None else None
    if context.dry_run:
        return {"command": joined, "dry_run": True, "stdout": "", "stderr": "", "returncode": 0}
    try:
        completed = subprocess.run(  # noqa: S603 - sandbox-checked, no shell
            command, cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(f"git command timed out after {timeout}s") from exc
    stdout = completed.stdout[:MAX_GIT_OUTPUT]
    stderr = completed.stderr[:MAX_GIT_OUTPUT]
    return {
        "command": joined,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_lines": stdout.count("\n") + (1 if stdout else 0),
        "truncated": len(completed.stdout) > MAX_GIT_OUTPUT,
    }


@tool(category="git", risk="read", timeout=60.0, tags=["git", "core"])
def git_status(context: ToolContext | None = None) -> dict[str, Any]:
    """Show the working tree status, branch, and ahead/behind counts.

    Args:
        (none)
    """
    ctx = context or ToolContext()
    result = _git(["status", "--porcelain=v2", "--branch"], ctx)
    parsed = parse_status_v2(result.get("stdout", ""))
    result.update(parsed)
    return result


def parse_status_v2(output: str) -> dict[str, Any]:
    """Parse ``git status --porcelain=v2 --branch`` into structured data."""
    branch = ""
    upstream = ""
    ahead = behind = 0
    staged: list[str] = []
    modified: list[str] = []
    untracked: list[str] = []
    conflicts: list[str] = []
    for line in output.splitlines():
        if line.startswith("# branch.head "):
            branch = line.split(" ", 2)[2].strip()
        elif line.startswith("# branch.upstream "):
            upstream = line.split(" ", 2)[2].strip()
        elif line.startswith("# branch.ab "):
            parts = line.split()
            for part in parts[2:]:
                if part.startswith("+"):
                    ahead = int(part[1:])
                elif part.startswith("-"):
                    behind = int(part[1:])
        elif line.startswith("1 ") or line.startswith("2 "):
            fields = line.split(" ")
            xy = fields[1] if len(fields) > 1 else "??"
            path = fields[-1]
            if xy == "..":
                continue
            (staged if xy[0] != "." else modified).append(f"{xy} {path}")
        elif line.startswith("u "):
            conflicts.append(line.split(" ")[-1])
        elif line.startswith("? "):
            untracked.append(line.split(" ", 1)[1])
    return {
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
        "conflicts": conflicts,
        "clean": not (staged or modified or untracked or conflicts),
        "changed_files": len(staged) + len(modified) + len(untracked),
    }


@tool(category="git", risk="read", timeout=60.0, tags=["git", "history"])
def git_log(limit: int = 20, path: str = "", author: str = "", grep: str = "",
            context: ToolContext | None = None) -> dict[str, Any]:
    """Show commit history as structured records.

    Args:
        limit: maximum number of commits to return
        path: restrict to commits touching this path
        author: filter by author name or email substring
        grep: filter by commit message substring
    """
    ctx = context or ToolContext()
    args = ["log", f"-{max(1, min(limit, 500))}", "--pretty=format:%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s%x1f%D"]
    if path:
        args += ["--", path]
    if author:
        args.insert(2, f"--author={author}")
    if grep:
        args.insert(2, f"--grep={grep}")
    result = _git(args, ctx)
    commits: list[dict[str, Any]] = []
    for line in result.get("stdout", "").splitlines():
        if not line.strip():
            continue
        parts = line.split("\x1f")
        if len(parts) < 6:
            continue
        commits.append({
            "hash": parts[0],
            "short_hash": parts[1],
            "author": parts[2],
            "email": parts[3],
            "date": parts[4],
            "subject": parts[5],
            "refs": parts[6] if len(parts) > 6 else "",
        })
    result["commits"] = commits
    result["count"] = len(commits)
    result.pop("stdout", None)
    return result


@tool(category="git", risk="read", timeout=60.0, tags=["git", "diff"])
def git_diff(staged: bool = False, path: str = "", ref: str = "",
             context: ToolContext | None = None) -> dict[str, Any]:
    """Show a diff of the working tree, index, or a specific ref.

    Args:
        staged: diff the index against HEAD instead of the working tree
        path: restrict the diff to this path
        ref: diff against this commit, branch, or range such as "main..HEAD"
    """
    ctx = context or ToolContext()
    args = ["diff", "--stat", "--patch"]
    if staged:
        args.append("--staged")
    if ref:
        args.append(ref)
    if path:
        args += ["--", path]
    result = _git(args, ctx)
    text = result.get("stdout", "")
    result.update({
        "files_changed": len(re.findall(r"^diff --git ", text, re.MULTILINE)),
        "additions": sum(int(match) for match in re.findall(r"^\+(\+\+)?", text, re.MULTILINE) if not match),
        "lines": text.count("\n"),
    })
    result["additions"] = len(re.findall(r"^\+(?!\+\+)", text, re.MULTILINE))
    result["deletions"] = len(re.findall(r"^-(?!--)", text, re.MULTILINE))
    return result


@tool(category="git", risk="write", timeout=60.0, tags=["git", "commit"])
def git_commit(message: str, add_all: bool = False, allow_empty: bool = False,
               context: ToolContext | None = None) -> dict[str, Any]:
    """Stage tracked changes and create a commit.

    Args:
        message: commit message
        add_all: run "git add -A" first
        allow_empty: permit a commit with no changes
    """
    ctx = context or ToolContext()
    if not message.strip():
        raise ToolExecutionError("Commit message must not be empty")
    if add_all:
        _git(["add", "-A"], ctx)
    args = ["commit", "-m", message]
    if allow_empty:
        args.append("--allow-empty")
    result = _git(args, ctx)
    head = _git(["rev-parse", "HEAD"], ctx)
    result["commit"] = head.get("stdout", "").strip()
    return result


@tool(category="git", risk="write", timeout=60.0, tags=["git", "stage"])
def git_add(paths: str, all_changes: bool = False, context: ToolContext | None = None) -> dict[str, Any]:
    """Stage files in the index.

    Args:
        paths: space or comma separated paths to stage
        all_changes: stage everything with "git add -A"
    """
    ctx = context or ToolContext()
    if all_changes:
        return _git(["add", "-A"], ctx)
    items = [item.strip() for item in paths.replace(",", " ").split() if item.strip()]
    if not items:
        raise ToolExecutionError("Provide at least one path, or set all_changes=true")
    return _git(["add", "--", *items], ctx)


@tool(category="git", risk="read", timeout=60.0, tags=["git", "branches"])
def git_branches(all_remotes: bool = True, context: ToolContext | None = None) -> dict[str, Any]:
    """List local and remote branches with their latest commit.

    Args:
        all_remotes: include remote-tracking branches
    """
    ctx = context or ToolContext()
    args = ["branch", "--all", "--format=%(refname:short)%09%(objectname:short)%09%(HEAD)%09%(upstream:short)"]
    if not all_remotes:
        args[1] = "--no-color"
        args = ["branch", "--format=%(refname:short)%09%(objectname:short)%09%(HEAD)%09%(upstream:short)"]
    result = _git(args, ctx)
    branches: list[dict[str, Any]] = []
    current = ""
    for line in result.get("stdout", "").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name = parts[0].strip()
        is_head = len(parts) > 2 and parts[2].strip() == "*"
        if is_head:
            current = name
        branches.append({
            "name": name,
            "commit": parts[1] if len(parts) > 1 else "",
            "current": is_head,
            "upstream": parts[3] if len(parts) > 3 else "",
            "remote": name.startswith("origin/") or "/" in name,
        })
    result["branches"] = branches
    result["current"] = current
    result["count"] = len(branches)
    result.pop("stdout", None)
    return result


@tool(category="git", risk="read", timeout=120.0, tags=["git", "blame", "annotate"])
def git_blame(path: str, line_start: int = 0, line_count: int = 50,
              context: ToolContext | None = None) -> dict[str, Any]:
    """Show which commit last touched each line of a file.

    Args:
        path: file to annotate, relative to the workspace
        line_start: first line to annotate, 0 for the whole file
        line_count: how many lines to annotate
    """
    ctx = context or ToolContext()
    args = ["blame", "--porcelain"]
    if line_start > 0:
        args += ["-L", f"{line_start},+{max(1, line_count)}"]
    args += ["--", path]
    result = _git(args, ctx)
    records: list[dict[str, Any]] = []
    current_commit = ""
    author = ""
    summary = ""
    for line in result.get("stdout", "").splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            current_commit = line.split()[0][:12]
        elif line.startswith("author "):
            author = line[7:]
        elif line.startswith("summary "):
            summary = line[8:]
        elif line.startswith("\t"):
            records.append({"commit": current_commit, "author": author,
                            "summary": summary, "line": line[1:]})
    result["blame"] = records[:max(1, line_count) * 2]
    result.pop("stdout", None)
    return result


@tool(category="git", risk="read", timeout=60.0, tags=["git", "remote"])
def git_remote_info(context: ToolContext | None = None) -> dict[str, Any]:
    """List configured remotes and their fetch URLs, with credentials redacted.

    Args:
        (none)
    """
    ctx = context or ToolContext()
    from ...utils.urls import redact_url

    result = _git(["remote", "-v"], ctx)
    remotes: list[dict[str, Any]] = []
    for line in result.get("stdout", "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remotes.append({"name": parts[0], "url": redact_url(parts[1]),
                            "direction": parts[2].strip("()") if len(parts) > 2 else ""})
    result["remotes"] = remotes
    result.pop("stdout", None)
    return result


@tool(category="git", risk="write", timeout=180.0, requires_approval=True,
      tags=["git", "push", "network"])
def git_push(remote: str = "origin", branch: str = "", force: bool = False,
             set_upstream: bool = False, context: ToolContext | None = None) -> dict[str, Any]:
    """Push commits to a remote repository.

    Args:
        remote: remote name, default "origin"
        branch: branch to push, default the current branch
        force: use --force-with-lease, never a bare --force
        set_upstream: also set the upstream tracking branch
    """
    ctx = context or ToolContext()
    args = ["push"]
    if set_upstream:
        args.append("-u")
    if force:
        args.append("--force-with-lease")
    args.append(remote)
    if branch:
        args.append(branch)
    return _git(args, ctx, timeout=180.0)


@tool(category="git", risk="read", timeout=60.0, tags=["git", "show"])
def git_show(ref: str = "HEAD", path: str = "", stat_only: bool = False,
             context: ToolContext | None = None) -> dict[str, Any]:
    """Show a commit, tag, or file at a revision.

    Args:
        ref: commit, tag, or branch reference
        path: restrict to this file within the revision
        stat_only: show the diffstat instead of the full patch
    """
    ctx = context or ToolContext()
    args = ["show", "--stat" if stat_only else "--patch", "--format=fuller", ref]
    if path:
        args += ["--", path]
    return _git(args, ctx)


TOOLS: list[Any] = [
    git_status, git_log, git_diff, git_commit, git_add, git_branches, git_blame,
    git_remote_info, git_push, git_show,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every git tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "DANGEROUS_GIT_ARGS",
    "MAX_GIT_OUTPUT",
    "TOOLS",
    "git_add",
    "git_blame",
    "git_branches",
    "git_commit",
    "git_diff",
    "git_log",
    "git_push",
    "git_remote_info",
    "git_show",
    "git_status",
    "load",
    "parse_status_v2",
]
