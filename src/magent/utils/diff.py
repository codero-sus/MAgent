"""Text differencing used by the file-edit tool and the review renderer."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .colors import ANSI, colorize
from .text import normalize_newlines

_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
"""Regex for a unified diff hunk header line."""


@dataclass(slots=True)
class DiffHunk:
    """One contiguous change region between two texts."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)

    @property
    def additions(self) -> int:
        """Return how many lines were added in this hunk."""
        return sum(1 for line in self.lines if line.startswith("+"))

    @property
    def deletions(self) -> int:
        """Return how many lines were removed in this hunk."""
        return sum(1 for line in self.lines if line.startswith("-"))

    def to_dict(self) -> dict[str, object]:
        """Serialize the hunk for tool output payloads."""
        return {
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "additions": self.additions,
            "deletions": self.deletions,
            "lines": self.lines,
        }


@dataclass(slots=True)
class DiffResult:
    """The full comparison of two texts."""

    hunks: list[DiffHunk] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    changed: bool = False
    similarity: float = 1.0

    def to_dict(self) -> dict[str, object]:
        """Serialize the diff result."""
        return {
            "changed": self.changed,
            "additions": self.additions,
            "deletions": self.deletions,
            "similarity": round(self.similarity, 4),
            "hunks": [hunk.to_dict() for hunk in self.hunks],
        }

    def summary_line(self) -> str:
        """Return a one line ``+3 -1`` style summary."""
        return f"+{self.additions} -{self.deletions}"


def unified_diff(
    old: str,
    new: str,
    *,
    from_label: str = "before",
    to_label: str = "after",
    context: int = 3,
    color: bool = False,
) -> str:
    """Return a unified diff of two texts, optionally ANSI coloured."""
    old_lines = normalize_newlines(old).splitlines(keepends=True)
    new_lines = normalize_newlines(new).splitlines(keepends=True)
    lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_label,
            tofile=to_label,
            n=context,
        )
    )
    if not color:
        return "".join(lines)
    out: list[str] = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(colorize(line, ANSI.GREEN))
        elif line.startswith("-") and not line.startswith("---"):
            out.append(colorize(line, ANSI.RED))
        elif line.startswith("@@"):
            out.append(colorize(line, ANSI.CYAN))
        elif line.startswith(("---", "+++")):
            out.append(colorize(line, ANSI.BOLD))
        else:
            out.append(line)
    return "".join(out)


def diff_texts(old: str, new: str, *, context: int = 3) -> DiffResult:
    """Compute structured hunks plus addition/deletion counts."""
    old_lines = normalize_newlines(old).split("\n")
    new_lines = normalize_newlines(new).split("\n")
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    hunks: list[DiffHunk] = []
    additions = deletions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        lines: list[str] = []
        for line in old_lines[i1:i2]:
            lines.append(f"-{line}")
            deletions += 1
        for line in new_lines[j1:j2]:
            lines.append(f"+{line}")
            additions += 1
        for offset in range(1, context + 1):
            if i1 - offset >= 0:
                lines.insert(0, f" {old_lines[i1 - offset]}")
        for offset in range(context):
            if i2 + offset < len(old_lines):
                lines.append(f" {old_lines[i2 + offset]}")
        hunks.append(
            DiffHunk(
                old_start=i1 + 1,
                old_count=i2 - i1,
                new_start=j1 + 1,
                new_count=j2 - j1,
                lines=lines,
            )
        )
    return DiffResult(
        hunks=hunks,
        additions=additions,
        deletions=deletions,
        changed=bool(hunks),
        similarity=matcher.ratio(),
    )


def diff_summary(old: str, new: str) -> str:
    """Return a compact human readable summary of the change."""
    result = diff_texts(old, new)
    if not result.changed:
        return "no changes"
    parts = [f"{len(result.hunks)} hunk{'s' if len(result.hunks) != 1 else ''}"]
    parts.append(result.summary_line())
    parts.append(f"similarity {result.similarity * 100:.0f}%")
    return ", ".join(parts)


def side_by_side(old: str, new: str, *, width: int = 40, color: bool = False) -> str:
    """Render two texts as aligned side-by-side columns."""
    old_lines = normalize_newlines(old).split("\n")
    new_lines = normalize_newlines(new).split("\n")
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    rows: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                rows.append((old_lines[i1 + offset], " ", new_lines[j1 + offset]))
            continue
        left = old_lines[i1:i2]
        right = new_lines[j1:j2]
        for offset in range(max(len(left), len(right))):
            rows.append(
                (
                    left[offset] if offset < len(left) else "",
                    "|" if tag == "replace" else ("-" if offset < len(left) else "+"),
                    right[offset] if offset < len(right) else "",
                )
            )
    header = f"{'OLD':<{width}}   {'NEW':<{width}}"
    out = [header, "-" * (width * 2 + 3)]
    for left, marker, right in rows:
        line = f"{left[:width]:<{width}} {marker} {right[:width]:<{width}}"
        if color and marker != " ":
            line = colorize(line, ANSI.RED if marker == "-" else ANSI.GREEN)
        out.append(line)
    return "\n".join(out)


def word_diff(old: str, new: str) -> list[tuple[str, str]]:
    """Return a word level diff as ``(tag, text)`` pairs."""
    matcher = difflib.SequenceMatcher(a=old.split(), b=new.split(), autojunk=False)
    out: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.append(("equal", " ".join(old.split()[i1:i2])))
        else:
            if i2 > i1:
                out.append(("delete", " ".join(old.split()[i1:i2])))
            if j2 > j1:
                out.append(("insert", " ".join(new.split()[j1:j2])))
    return out


def similarity_ratio(left: str, right: str) -> float:
    """Return a 0..1 line-based similarity ratio for two texts."""
    return round(
        difflib.SequenceMatcher(
            a=normalize_newlines(left).split("\n"),
            b=normalize_newlines(right).split("\n"),
            autojunk=False,
        ).ratio(),
        6,
    )


def closest_match(target: str, candidates: Sequence[str], *, cutoff: float = 0.6) -> str | None:
    """Return the candidate closest to ``target`` for typo suggestions."""
    matches = difflib.get_close_matches(target, list(candidates), n=1, cutoff=cutoff)
    return matches[0] if matches else None


def apply_patch(text: str, patch: str) -> str:
    """Apply a unified diff patch to ``text``, raising on mismatch."""
    old_lines = normalize_newlines(text).split("\n")
    result: list[str] = []
    cursor = 0
    lines = normalize_newlines(patch).split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@"):
            index += 1
            continue
        try:
            match = _HUNK_HEADER.match(line)
            if not match:
                raise ValueError(f"malformed hunk header: {line}")
            start = int(match.group("old_start"))
        except ValueError as exc:
            raise ValueError(f"malformed hunk header: {line}") from exc
        target = start - 1
        if target < cursor:
            raise ValueError("hunks must be applied in ascending order")
        result.extend(old_lines[cursor:target])
        cursor = target
        index += 1
        while index < len(lines):
            current = lines[index]
            if current.startswith("@@"):
                break
            if current.startswith("-"):
                if cursor >= len(old_lines) or old_lines[cursor] != current[1:]:
                    raise ValueError(f"patch context mismatch at line {cursor + 1}")
                cursor += 1
            elif current.startswith("+"):
                result.append(current[1:])
            elif current.startswith(" "):
                if cursor >= len(old_lines):
                    raise ValueError("patch ran past end of file")
                result.append(current[1:])
                cursor += 1
            elif current.startswith("\\"):
                pass
            else:
                break
            index += 1
    result.extend(old_lines[cursor:])
    return "\n".join(result)


__all__ = [
    "DiffHunk",
    "DiffResult",
    "apply_patch",
    "closest_match",
    "diff_summary",
    "diff_texts",
    "side_by_side",
    "similarity_ratio",
    "unified_diff",
    "word_diff",
]
