#!/usr/bin/env python3
"""Example 01: build an agent and run it.

Run: ``python examples/01_hello_agent.py``
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _bootstrap import agent, backend_name, heading, main_guard  # noqa: E402


def body() -> None:
    """Run one prompt and show what the run recorded."""
    instance = agent()
    heading(f"backend: {backend_name(instance.backend)}")

    result = instance.run("Explain in one sentence what you are.")
    print(result.output)

    heading("what the run recorded")
    print(f"stop reason : {result.stop_reason}")
    print(f"succeeded   : {result.succeeded}")
    print(f"steps       : {len(result.steps)}")
    print(f"tokens      : {result.usage.total_tokens}")
    print(f"duration    : {result.duration_ms:.0f} ms")
    print(f"model       : {result.model}")

    heading("events")
    for event in result.events[:6]:
        print(f"  {str(event.kind):<22} {list(event.data)[:4]}")


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
