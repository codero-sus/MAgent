"""Make examples runnable from a source checkout without installation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
"""Repository root containing ``src``."""

SRC = ROOT / "src"
"""Source-layout package directory."""

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
