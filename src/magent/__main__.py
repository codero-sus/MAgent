"""``python -m magent`` entry point.

The console scripts call :func:`magent.cli.main.run`; this module makes the same
CLI available without an install step, which is what CI and the test suite use.
"""

from __future__ import annotations

import sys

from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
