#!/usr/bin/env python3
"""Example 09: preview and test a desktop workflow without touching the desktop.

Run: ``python examples/09_desktop_automation.py``

The recording driver exercises the exact controller and workflow code used by
the live desktop tools, but only records actions in memory. For real automation,
install ``MAgent[desktop]`` and use ``magent desktop`` so changing actions remain
behind the interactive approval prompt and emergency stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import heading, main_guard
from magent.config import DesktopConfig
from magent.desktop import DesktopController, DesktopWorkflow, DesktopWorkflowRunner
from magent.desktop.driver import RecordingDesktopDriver


def body() -> None:
    """Load a declarative workflow, preview it, and run it against a fake desktop."""
    workflow_path = Path(__file__).with_name("desktop_workflow.json")
    workflow = DesktopWorkflow.parse(workflow_path.read_text(encoding="utf-8"))
    driver = RecordingDesktopDriver(width=1440, height=900)
    controller = DesktopController(
        DesktopConfig(backend="recording", fail_safe=False, action_pause=0),
        driver=driver,
    )
    runner = DesktopWorkflowRunner(controller)

    heading("dry-run preview")
    preview = runner.run(workflow, dry_run=True)
    for step in preview.steps:
        print(f"  {step.index + 1}. {step.action:<16} {step.output['params']}")
    print("recorded actions:", len(driver.actions))

    heading("deterministic recording run")
    result = runner.run(workflow)
    print("completed:", result.ok, "steps:", len(result.steps))
    for action in driver.actions:
        print(" ", action)

    heading("live equivalent")
    print("  pip install 'MAgent[desktop]'")
    print("  magent desktop")
    print("  magent tools inspect desktop_run_workflow")


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
