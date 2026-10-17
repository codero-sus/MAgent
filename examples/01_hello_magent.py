"""Run the smallest MAgent task with the offline backend.

Usage:
    python3 examples/01_hello_magent.py
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401 - installs src on sys.path

from magent import Agent, EchoBackend


def main() -> None:
    """Run one deterministic turn and print its recorded outcome."""
    agent = Agent(backend=EchoBackend(), workspace=ROOT)
    result = agent.run("hello from MAgent")
    print(result.output)
    print(result.summary_line())


if __name__ == "__main__":
    main()
