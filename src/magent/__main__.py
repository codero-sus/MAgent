"""Allow ``python -m magent`` to run the CLI.

This exists so the framework is usable straight from a source checkout, before
``pip install`` has created the ``magent`` console script.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit code.

    Args:
        argv: argument list, defaulting to ``sys.argv[1:]``
    """
    from .cli.main import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
