"""Store and retrieve local long-term memory.

Usage:
    python3 examples/04_memory.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from magent import MemoryStore


def main() -> None:
    """Add facts to a temporary SQLite store and run hybrid retrieval."""
    with tempfile.TemporaryDirectory(prefix="magent-memory-example-") as temporary:
        store = MemoryStore(path=Path(temporary) / "memory.db")
        store.add("Production runs in ap-south-1", kind="fact",
                  tags=["deployment"], importance=0.9, source="runbook")
        store.add("The design system uses cobalt blue", kind="preference",
                  tags=["design"], importance=0.6)

        for entry in store.search("production deployment region", limit=3):
            print(f"{entry.score:.3f}  {entry.content}  [{entry.source or 'local'}]")
        print("stats:", store.stats())


if __name__ == "__main__":
    main()
