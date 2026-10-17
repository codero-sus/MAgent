"""Define a typed tool and run a complete model/tool/model loop.

Usage:
    python3 examples/03_custom_tool.py
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from magent import Agent, tool
from magent.testing import make_backend
from magent.tools import ToolRegistry


@tool(category="math", risk="safe")
def multiply(left: int, right: int) -> int:
    """Multiply two integers.

    Args:
        left: first factor
        right: second factor
    """
    return left * right


def main() -> None:
    """Let a scripted model request ``multiply`` and consume its result."""
    registry = ToolRegistry()
    registered = registry.register(multiply)
    print("schema:", registered.to_openai())

    backend = make_backend(
        tool_name="multiply",
        tool_arguments={"left": 6, "right": 7},
        final_text="The tool returned 42.",
    )
    agent = Agent(backend=backend, registry=registry, workspace=ROOT)
    result = agent.run("Multiply 6 by 7")
    print(result.output)
    print("tool result:", result.steps[0].tool_results[0].content)


if __name__ == "__main__":
    main()
