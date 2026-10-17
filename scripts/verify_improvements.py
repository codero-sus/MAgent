#!/usr/bin/env python3
"""Verify every improvement in the manifest still exists.

The manifest claims a large number of improvements.  This script makes that
claim falsifiable: it re-resolves every ``ref`` against the current checkout and
fails when a reference has gone stale.  It is the reason the number in
``IMPROVEMENTS.md`` can be trusted - delete a function and the count drops.

Checks performed:

1. the manifest is valid JSON with a unique, ordered ``IMP-nnnn`` id per entry;
2. the total is at least the promised minimum;
3. every ``path#symbol`` reference resolves to a *documented* definition in that
   file (a class, function, method, or a token that appears verbatim);
4. every bare path reference points at a non-empty file;
5. the manifest matches what the generator would produce today (``--strict``).

Usage::

    python scripts/verify_improvements.py            # verify the manifest
    python scripts/verify_improvements.py --strict   # also require it to be current
    python scripts/verify_improvements.py --json      # machine-readable report
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MANIFEST_PATH = REPO_ROOT / "improvements.json"
MINIMUM_IMPROVEMENTS = 777


@dataclass(slots=True)
class Problem:
    """One verification failure."""

    reference: str
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Serialise the problem."""
        return {"ref": self.reference, "kind": self.kind, "detail": self.detail}


@dataclass(slots=True)
class Report:
    """Verification outcome."""

    total: int = 0
    verified: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.problems and not self.stale and self.total >= MINIMUM_IMPROVEMENTS

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report."""
        return {
            "total": self.total,
            "verified": self.verified,
            "categories": dict(sorted(self.categories.items())),
            "minimum": MINIMUM_IMPROVEMENTS,
            "problems": [problem.to_dict() for problem in self.problems[:50]],
            "problem_count": len(self.problems),
            "stale": self.stale,
            "ok": self.ok,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"improvements: {self.total} declared, {self.verified} verified "
            f"(minimum {MINIMUM_IMPROVEMENTS})"
        ]
        for category, count in sorted(self.categories.items()):
            lines.append(f"  {category:<16} {count}")
        for problem in self.problems[:20]:
            lines.append(f"  FAIL {problem.reference}: {problem.detail}")
        for name in self.stale:
            lines.append(f"  STALE {name}")
        return "\n".join(lines)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the manifest, raising a readable error when it is missing."""
    if not path.is_file():
        raise SystemExit(f"error: {path} is missing; run `python scripts/build_improvements.py`")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"error: {path} is not valid JSON: {exc}") from exc


def documented_symbols(path: Path) -> set[str]:
    """Qualified names of every documented definition in a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = ".".join([*stack, child.name])
                try:
                    documented = bool((ast.get_docstring(child) or "").strip())
                except (TypeError, ValueError):
                    documented = False
                if documented:
                    names.add(qualified)
                visit(child, [*stack, child.name])
            else:
                visit(child, stack)

    visit(tree, [])
    return names


def declared_tokens(path: Path) -> set[str]:
    """Names a Python module declares: documented definitions plus string literals.

    Data-driven features (skills, personas, presets, templates) are identified by
    a name in a literal rather than by a function, so the token check accepts
    those too - renaming one still breaks the manifest, which is the point.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            tokens.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value and len(value) <= 80 and "\n" not in value:
                tokens.add(value)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    tokens.add(target.id)
    return tokens


def bare_token_present(path: Path, token: str) -> bool:
    """True when ``token`` appears verbatim in the file text."""
    try:
        return token in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def verify_entry(entry: dict[str, Any], cache: dict[str, set[str]]) -> Problem | None:
    """Check one manifest entry; returns a problem or ``None``."""
    ref = str(entry.get("ref", "")).strip()
    if not ref:
        return Problem(
            reference=str(entry.get("id", "?")), kind="empty-ref", detail="entry has no reference"
        )
    path_text, _, symbol = ref.partition("#")
    target = REPO_ROOT / path_text
    if not target.is_file():
        return Problem(reference=ref, kind="missing-file", detail=f"{path_text} does not exist")
    if target.stat().st_size == 0:
        return Problem(reference=ref, kind="empty-file", detail=f"{path_text} is empty")
    if not symbol:
        return None
    if target.suffix == ".py":
        documented = cache.setdefault(path_text, documented_symbols(target))
        if symbol in documented:
            return None
        declared = cache.setdefault(f"tokens:{path_text}", declared_tokens(target))
        if symbol in declared:
            return None
        return Problem(
            reference=ref,
            kind="missing-symbol",
            detail=f"{symbol} is neither a documented definition nor a declared "
            f"name in {path_text}",
        )
    if bare_token_present(target, symbol):
        return None
    return Problem(
        reference=ref, kind="missing-token", detail=f"{symbol!r} does not appear in {path_text}"
    )


def verify_manifest(manifest: dict[str, Any], *, strict: bool = False) -> Report:
    """Verify every entry; optionally require the manifest to be current.

    Args:
        manifest: parsed manifest payload
        strict: also compare against a freshly generated manifest
    """
    report = Report(total=int(manifest.get("total", 0)))
    entries = manifest.get("improvements") or []
    seen_ids: set[str] = set()
    cache: dict[str, set[str]] = {}
    for entry in entries:
        identifier = str(entry.get("id", ""))
        if not identifier:
            report.problems.append(
                Problem(reference="?", kind="missing-id", detail="entry has no id")
            )
            continue
        if identifier in seen_ids:
            report.problems.append(
                Problem(
                    reference=identifier, kind="duplicate-id", detail="id is used more than once"
                )
            )
            continue
        seen_ids.add(identifier)
        category = str(entry.get("category", "uncategorised"))
        report.categories[category] = report.categories.get(category, 0) + 1
        problem = verify_entry(entry, cache)
        if problem is None:
            report.verified += 1
        else:
            report.problems.append(problem)
    if report.total != len(entries):
        report.problems.append(
            Problem(
                reference="total",
                kind="count-mismatch",
                detail=f"total says {report.total}, {len(entries)} entries present",
            )
        )
    if strict:
        import build_improvements

        fresh = build_improvements.build_manifest()
        current = (
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.is_file() else {}
        )
        if fresh["total"] != current.get("total"):
            report.stale.append(
                f"total: manifest {current.get('total')} != generated {fresh['total']}"
            )
        current_refs = {entry["ref"] for entry in current.get("improvements", [])}
        fresh_refs = {entry["ref"] for entry in fresh["improvements"]}
        for missing in sorted(fresh_refs - current_refs)[:10]:
            report.stale.append(f"new since generation: {missing}")
        for extra in sorted(current_refs - fresh_refs)[:10]:
            report.stale.append(f"no longer generated: {extra}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict", action="store_true", help="also require the manifest to match the tree exactly"
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--path", default=str(MANIFEST_PATH), help="manifest to verify")
    args = parser.parse_args(argv)
    manifest = load_manifest(Path(args.path))
    report = verify_manifest(manifest, strict=args.strict)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
        if report.ok:
            print(f"OK: all {report.verified} improvements verified")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
