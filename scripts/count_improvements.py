#!/usr/bin/env python3
"""Print the improvement count, with optional per-category detail.

Small and dependency-free: it reads ``improvements.json`` so it can run in CI
without importing MAgent, and it never writes anything.

Usage::

    python scripts/count_improvements.py            # total only
    python scripts/count_improvements.py --detail    # per-category and per-kind
    python scripts/count_improvements.py --minimum 777
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "improvements.json"
DEFAULT_MINIMUM = 777


def load(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Read the manifest, exiting with a clear message when it is missing."""
    if not path.is_file():
        raise SystemExit(f"error: {path} is missing; run `python scripts/build_improvements.py`")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detail", action="store_true", help="show category and kind breakdown")
    parser.add_argument(
        "--minimum", type=int, default=DEFAULT_MINIMUM, help="fail when the total is below this"
    )
    parser.add_argument("--path", default=str(MANIFEST_PATH), help="manifest to read")
    args = parser.parse_args(argv)
    manifest = load(Path(args.path))
    total = int(manifest.get("total", len(manifest.get("improvements", []))))
    if args.detail:
        print(f"total: {total}")
        print("by category:")
        for category, count in sorted((manifest.get("categories") or {}).items()):
            print(f"  {category:<16} {count}")
        print("by kind:")
        for kind, count in sorted((manifest.get("kinds") or {}).items()):
            print(f"  {kind:<16} {count}")
    else:
        print(total)
    if total < args.minimum:
        print(f"error: {total} improvements is below the required {args.minimum}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
