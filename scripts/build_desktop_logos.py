#!/usr/bin/env python3
"""Generate MAgent's eight deterministic desktop logo pairs and gallery."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "src" / "magent" / "desktop" / "assets" / "logos"
sys.path.insert(0, str(REPO_ROOT / "src"))

from magent.desktop.branding import write_logo_assets  # noqa: E402


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when generated assets differ")
    args = parser.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="magent-logos-") as directory:
            candidate = Path(directory)
            write_logo_assets(candidate)
            actual = _snapshot(ASSET_DIR) if ASSET_DIR.exists() else {}
            expected = _snapshot(candidate)
            if actual != expected:
                missing = sorted(set(expected) - set(actual))
                stale = sorted(set(actual) - set(expected))
                changed = sorted(
                    name for name in set(actual) & set(expected) if actual[name] != expected[name]
                )
                print(
                    f"desktop logo assets are stale: missing={missing} extra={stale} changed={changed}"
                )
                return 1
        print("desktop logo assets are up to date (8 icon/lockup pairs)")
        return 0
    written = write_logo_assets(ASSET_DIR)
    print(f"wrote {len(written) - 1} logo assets and {written[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
