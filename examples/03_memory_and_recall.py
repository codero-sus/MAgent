#!/usr/bin/env python3
"""Example 03: long-term memory and recall.

Run: ``python examples/03_memory_and_recall.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import agent, heading, main_guard
from magent.memory import build_store


def body() -> None:
    """Store facts, search them, and watch an agent recall them."""
    store = build_store()
    heading("storing facts")
    for content, kind in (
        ("the deploy key lives in ~/.config/magent/secrets.toml", "fact"),
        ("we deploy on fridays at 15:00 UTC", "fact"),
        ("the parser does not handle nested arrays yet", "issue"),
    ):
        entry = store.add(content, kind=kind, tags=["demo"])
        print(f"  {entry.id[:12]}  {entry.content}")
    store.save()

    heading("searching")
    for entry in store.search("when do we deploy"):
        print(f"  {entry.score:.2f}  {entry.content}")

    heading("the agent recalling it in a prompt")
    instance = agent(memory=store)
    block = instance.system_prompt(
        instance.run_options
        if hasattr(instance, "run_options")
        else __import__("magent.agent", fromlist=["RunOptions"]).RunOptions()
    )
    marker = "Relevant memories"
    print("recall block present:", marker in block)

    heading("stats")
    for key, value in store.stats().items():
        if not isinstance(value, (dict, list)):
            print(f"  {key:<14} {value}")


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
