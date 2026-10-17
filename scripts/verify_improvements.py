#!/usr/bin/env python3
"""Verify every improvement listed in the MAgent manifest.

The manifest is generated from the source tree, so this script is what makes the
count trustworthy rather than decorative. It re-resolves every ``path#symbol``
reference against the working tree and fails when:

* a referenced file or symbol no longer exists,
* a code symbol has no docstring,
* an identifier or key is duplicated,
* the total falls below ``IMPROVEMENT_COUNT_TARGET``.

Usage::

    python scripts/verify_improvements.py
    python scripts/verify_improvements.py --manifest IMPROVEMENTS.md
    python scripts/verify_improvements.py --json

Exit code is 0 when everything resolves and the target is met, 1 otherwise, so
CI can gate on it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
"""Repository root, derived from this script's location."""

sys.path.insert(0, str(ROOT / "src"))

from magent._version import IMPROVEMENT_COUNT_TARGET  # noqa: E402
from magent.improvements import collect, parse_manifest, repo_root, verify  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Verify the manifest and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="", metavar="PATH",
                        help="verify an already-written manifest file instead of the tree")
    parser.add_argument("--root", default="", metavar="DIR", help="repository root")
    parser.add_argument("--target", type=int, default=IMPROVEMENT_COUNT_TARGET,
                        help=f"required minimum total (default {IMPROVEMENT_COUNT_TARGET})")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    parser.add_argument("--max-problems", type=int, default=25,
                        help="how many failures to print")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else repo_root(ROOT)

    if args.manifest:
        path = Path(args.manifest).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            print(f"error: no manifest at {path}", file=sys.stderr)
            return 1
        records = parse_manifest(path)
        if not records:
            print(f"error: {path} contains no manifest rows", file=sys.stderr)
            return 1
        entries = collect(root)
        by_key = {entry.key: entry for entry in entries}
        # The manifest is the claim; the tree is the evidence. Every row must
        # still exist in a freshly collected manifest.
        missing = [record for record in records
                   if not any(item["ref"] == record["ref"] for item in
                              (entry.to_dict() for entry in by_key.values()))]
        report = verify(entries, root)
        report.target = args.target
        payload = report.to_dict()
        payload["manifest_rows"] = len(records)
        payload["manifest_rows_missing_from_tree"] = len(missing)
        ok = report.ok and not missing
    else:
        entries = collect(root)
        report = verify(entries, root)
        report.target = args.target
        payload = report.to_dict()
        ok = report.ok

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if ok else 1

    if not args.quiet:
        print(report.render())
    else:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {report.total} improvements, {report.passed}/{report.checked} "
              f"references resolve, target {report.target}")
    if args.manifest and not ok:
        print("manifest rows no longer present in the tree:", file=sys.stderr)
        for record in payload.get("manifest_rows_missing_from_tree", []) or []:
            print(f"  {record}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
