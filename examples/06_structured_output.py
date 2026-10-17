"""Validate a structured answer and repair an invalid first attempt.

Usage:
    python3 examples/06_structured_output.py
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from magent import Agent
from magent.testing import ScriptedBackend

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "risk"],
    "additionalProperties": False,
}
"""Schema expected from the model."""


def main() -> None:
    """Make the first answer invalid, then accept the repair turn."""
    backend = ScriptedBackend(script=[
        '{"summary":"Missing the required risk"}',
        '{"summary":"No external side effects","risk":"low"}',
    ])
    agent = Agent(backend=backend, workspace=ROOT)
    result = agent.run("Assess the change as JSON", structured=SCHEMA)
    print(result.output)
    print("parsed:", result.structured)
    print("backend calls:", len(backend.calls_log))


if __name__ == "__main__":
    main()
