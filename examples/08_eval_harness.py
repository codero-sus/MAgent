#!/usr/bin/env python3
"""Example 08: evaluate an agent and gate on the result.

Run: ``python examples/08_eval_harness.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import agent, heading, main_guard  # noqa: E402

from magent.eval import EvalRunner, build_dataset  # noqa: E402


def body() -> None:
    """Run a small dataset, print the report and fail below a threshold."""
    dataset = build_dataset([
        {"id": "hello", "prompt": "Say hello.", "not_contains": ["traceback"], "max_steps": 3},
        {"id": "math", "prompt": "What is 6*7?", "contains": ["42"], "tools": ["calculate"]},
        {"id": "json", "prompt": "Reply with the JSON object {\"ok\": true}.", "expect_json": True},
    ], name="example")

    runner = EvalRunner(lambda: agent(), repeats=2, concurrency=2, threshold=0.5)
    report = runner.run(dataset)

    heading("report")
    print(report.render("text"))

    heading("markdown")
    print(report.to_markdown())

    heading("gate")
    try:
        report.check(0.5)
        print("pass rate meets the 50% threshold")
    except Exception as exc:  # noqa: BLE001 - the example shows the failure path too
        print("gate failed:", exc)


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
