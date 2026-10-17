#!/usr/bin/env python3
"""Count the improvements MAgent adds, and regenerate the manifest.

Counting is derived, never asserted: the total is whatever an AST walk of the
source tree produces, so the number moves only when the code moves.

Usage::

    python scripts/count_improvements.py                # print the total
    python scripts/count_improvements.py --by-category  # per-category breakdown
    python scripts/count_improvements.py --write        # regenerate IMPROVEMENTS.md
    python scripts/count_improvements.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
"""Repository root, derived from this script's location."""

sys.path.insert(0, str(ROOT / "src"))

from magent._version import IMPROVEMENT_COUNT_TARGET, __version__  # noqa: E402
from magent.improvements import (  # noqa: E402
    collect,
    repo_root,
    summarize,
    verify,
    write_json_manifest,
    write_manifest,
)


def main(argv: list[str] | None = None) -> int:
    """Count improvements and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="", metavar="DIR", help="repository root")
    parser.add_argument("--by-category", action="store_true", help="print a breakdown")
    parser.add_argument("--write", action="store_true", help="regenerate IMPROVEMENTS.md")
    parser.add_argument("--out", default="", metavar="PATH",
                        help="manifest destination (default IMPROVEMENTS.md)")
    parser.add_argument("--json-out", default="", metavar="PATH",
                        help="also write the manifest as JSON")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument("--target", type=int, default=IMPROVEMENT_COUNT_TARGET,
                        help=f"required minimum (default {IMPROVEMENT_COUNT_TARGET})")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when the target is not met")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else repo_root(ROOT)
    entries = collect(root)
    info = summarize(entries)
    info["target"] = args.target
    info["meets_target"] = info["total"] >= args.target

    written = ""
    if args.write:
        target_path = Path(args.out).expanduser() if args.out else root / "IMPROVEMENTS.md"
        if not target_path.is_absolute():
            target_path = root / target_path
        path, report = write_manifest(target_path, root)
        report.target = args.target
        written = str(path)
        info["written"] = written
        info["bytes"] = path.stat().st_size
        info["verification"] = report.to_dict()
        info["meets_target"] = info["meets_target"] and report.ok

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        if not json_path.is_absolute():
            json_path = root / json_path
        write_json_manifest(json_path, root)
        info["json_manifest"] = str(json_path)

    if args.json:
        info["version"] = __version__
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print(f"MAgent {__version__}: {info['total']} improvements "
              f"(target {args.target}, {'met' if info['meets_target'] else 'NOT met'})")
        if args.by_category:
            width = max((len(name) for name in info["categories"]), default=10)
            for name, count in info["categories"].items():
                print(f"  {name.ljust(width)}  {count:>5}")
        if written:
            print(f"manifest: {written}")
            verification = info.get("verification") or {}
            if verification:
                print(f"verified: {verification.get('passed')}/{verification.get('checked')} "
                      f"references resolve, {verification.get('failed')} failures")

    if args.strict and not info["meets_target"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
