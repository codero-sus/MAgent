"""The ``improvements`` command: report and re-verify the manifest.

MAgent publishes a numbered list of everything it adds over an empty repository,
and the list is generated from the source tree rather than written by hand. This
command is how that claim stays honest: ``--verify`` re-resolves every
``path#symbol`` reference against the working tree and exits non-zero when any
entry points at something that no longer exists, or when the total falls below
:data:`~magent._version.IMPROVEMENT_COUNT_TARGET`.

That makes the count falsifiable. Deleting a function removes its entries and
fails verification; padding the list without writing code is not possible,
because an entry with no resolvable symbol is a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...improvements import (
    CATEGORIES,
    IMPROVEMENT_COUNT_TARGET,
    collect,
    parse_manifest,
    render_markdown,
    repo_root,
    summarize,
    verify,
    write_json_manifest,
    write_manifest,
)
from ...utils.tables import render_markdown_table, render_table
from ..args import CliContext, EXIT_ERROR, EXIT_OK


def register(subparsers: Any) -> None:
    """Add the ``improvements`` subcommand."""
    parser = subparsers.add_parser(
        "improvements",
        help="report and verify the improvement manifest",
        description="Show the numbered improvements MAgent adds, and verify every reference.",
    )
    parser.add_argument("--verify", action="store_true",
                        help="resolve every reference and exit non-zero on failure")
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite IMPROVEMENTS.md from the source tree")
    parser.add_argument("--json-manifest", default="", metavar="PATH",
                        help="also write the manifest as JSON")
    parser.add_argument("--category", default="", choices=["", *CATEGORIES],
                        help="only this category")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="print at most N entries")
    parser.add_argument("--target", type=int, default=0,
                        help=f"required total (default {IMPROVEMENT_COUNT_TARGET})")
    parser.add_argument("--root", default="", metavar="DIR",
                        help="repository root, detected automatically by default")
    parser.add_argument("--problems-only", action="store_true",
                        help="with --verify, print only the failures")
    parser.set_defaults(handler=cmd_improvements)


def _root(args: argparse.Namespace) -> Path:
    """Resolve the repository root to inspect."""
    wanted = str(getattr(args, "root", "") or "")
    return Path(wanted).expanduser().resolve() if wanted else repo_root()


def cmd_improvements(args: argparse.Namespace, context: CliContext) -> int:
    """Report the manifest, verifying it when asked."""
    root = _root(args)
    target = int(getattr(args, "target", 0) or 0) or IMPROVEMENT_COUNT_TARGET
    category = str(getattr(args, "category", "") or "")
    limit = int(getattr(args, "limit", 0) or 0)

    entries = collect(root)
    if category:
        entries = [entry for entry in entries if entry.category == category]
    report = verify(entries, root) if (getattr(args, "verify", False)
                                       or getattr(args, "regenerate", False)) else None
    if report is not None:
        report.target = target

    json_path = str(getattr(args, "json_manifest", "") or "")
    if json_path and not context.dry_run:
        write_json_manifest(json_path, root)

    if getattr(args, "regenerate", False):
        if context.dry_run:
            context.printer.write(f"would regenerate {root / 'IMPROVEMENTS.md'} "
                                  f"with {len(entries)} entries", min_level="quiet")
        else:
            written, regenerated = write_manifest(root / "IMPROVEMENTS.md", root)
            regenerated.target = target
            report = regenerated
            context.printer.write(f"wrote {written} ({written.stat().st_size} bytes, "
                                  f"{len(entries)} entries)", min_level="quiet")

    if context.options.fmt == "json":
        payload: dict[str, Any] = summarize(entries)
        payload["root"] = str(root)
        if json_path:
            payload["json_manifest"] = json_path
        if report is not None:
            payload["verification"] = report.to_dict()
        if limit:
            payload["entries"] = [entry.to_dict() for entry in entries[:limit]]
        context.emit(payload)
        return _exit_code(report, len(entries), target)

    printer = context.printer
    if report is not None and getattr(args, "problems_only", False):
        if report.problems:
            for problem in report.problems:
                printer.error(str(problem))
        else:
            printer.write(printer.themed(
                f"all {report.checked} references resolve", "success"), min_level="quiet")
        return _exit_code(report, len(entries), target)

    info = summarize(entries)
    rows = [[name, count] for name, count in info["categories"].items()]
    table = render_markdown_table(rows, headers=["category", "count"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["category", "count"], alignments=["left", "right"],
                          unicode_borders=context.options.unicode)
    printer.write(printer.themed("MAgent improvements by category", "heading"), min_level="quiet")
    printer.write(table, min_level="quiet")
    total = info["total"]
    verdict = "meets" if total >= target else "BELOW"
    theme = "success" if total >= target else "error"
    printer.write(printer.themed(
        f"total {total} - {verdict} the target of {target}", theme), min_level="quiet")

    if report is not None:
        printer.blank(min_level="quiet")
        printer.write(printer.themed("verification", "heading"), min_level="quiet")
        printer.write(printer.themed(
            f"{report.passed}/{report.checked} references resolve, {report.failed} failures -> "
            f"{'PASS' if report.ok else 'FAIL'}",
            "success" if report.ok else "error"), min_level="quiet")
        for problem in report.problems[:15]:
            printer.error(str(problem))
        if len(report.problems) > 15:
            printer.write(printer.themed(
                f"... and {len(report.problems) - 15} more", "muted"), min_level="quiet")

    if limit:
        printer.blank(min_level="normal")
        printer.write(printer.themed(f"first {min(limit, len(entries))} entries", "heading"),
                      min_level="normal")
        for entry in entries[:limit]:
            printer.write(f"  {entry.identifier:>4}. [{entry.category}] {entry.title} "
                          f"({entry.ref})", min_level="normal")
    return _exit_code(report, total, target)


def _exit_code(report: Any | None, total: int, target: int) -> int:
    """Return the exit code implied by verification and the target count."""
    if report is not None and not report.ok:
        return EXIT_ERROR
    if total < target:
        return EXIT_ERROR
    return EXIT_OK


def register_manifest(subparsers: Any) -> None:
    """Add the hidden ``manifest`` alias used by the generator scripts."""
    parser = subparsers.add_parser("manifest", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", metavar="PATH", help="destination file")
    parser.add_argument("--json", action="store_true", help="write JSON instead of Markdown")
    parser.set_defaults(handler=cmd_manifest)


def cmd_manifest(args: argparse.Namespace, context: CliContext) -> int:
    """Write the manifest to a path, used by ``scripts/`` and CI."""
    root = _root(args)
    out = str(getattr(args, "out", "") or "")
    if context.dry_run:
        entries = collect(root)
        context.printer.write(f"would write {len(entries)} entries to "
                              f"{out or root / 'IMPROVEMENTS.md'}", min_level="quiet")
        return EXIT_OK
    if getattr(args, "json", False):
        target = Path(out) if out else root / "improvements.json"
        write_json_manifest(target, root)
    else:
        target, report = write_manifest(Path(out) if out else None, root)
        context.printer.write(f"{target}: {report.total} entries, "
                              f"{report.passed}/{report.checked} verified", min_level="quiet")
    context.printer.write(f"wrote {target}", min_level="quiet")
    return EXIT_OK


def cmd_show_manifest(args: argparse.Namespace, context: CliContext) -> int:
    """Print entries from an already-generated manifest file."""
    root = _root(args)
    path = Path(str(getattr(args, "path", "") or "") or (root / "IMPROVEMENTS.md"))
    if not path.exists():
        context.printer.error(f"error: no manifest at {path}; run 'magent improvements "
                              f"--regenerate'")
        return EXIT_ERROR
    records = parse_manifest(path)
    category = str(getattr(args, "category", "") or "")
    if category:
        records = [item for item in records if item["category"] == category]
    if context.options.fmt == "json":
        context.emit({"path": str(path), "count": len(records), "entries": records})
        return EXIT_OK
    for record in records:
        context.printer.write(
            f"{record['id']:>4}. [{record['category']}] {record['title']} ({record['ref']})",
            min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(records)} entries read from {path.name}", "muted"), min_level="normal")
    return EXIT_OK


__all__ = [
    "cmd_improvements",
    "cmd_manifest",
    "cmd_show_manifest",
    "register",
    "register_manifest",
]
