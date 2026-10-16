"""Shared setup for the examples.

Every example runs offline: the backend is the echo backend unless Cortex
LLMHoster is reachable at ``MAGENT_CORTEX_URL``, and all state (memory,
sessions, cache) is written into a temporary ``MAGENT_HOME`` that is removed
afterwards.  Run them with ``python examples/01_hello_agent.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_TEMP_HOME = Path(tempfile.mkdtemp(prefix="magent-example-"))
os.environ.setdefault("MAGENT_HOME", str(_TEMP_HOME))
os.environ.setdefault("MAGENT_LOG_LEVEL", "WARNING")


def backend() -> Any:
    """Return a backend: Cortex when it is reachable, the echo backend otherwise."""
    from magent.llm.factory import build_backend

    return build_backend()


def backend_name(instance: Any) -> str:
    """Human-readable backend name."""
    return getattr(instance, "name", type(instance).__name__)


def agent(**kwargs: Any) -> Any:
    """Build an agent with the example workspace."""
    from magent.agent import Agent

    return Agent(backend=backend(), workspace=kwargs.pop("workspace", workspace()), **kwargs)


def workspace(*files: str) -> Path:
    """Create (or return) a small scratch workspace for the examples."""
    root = _TEMP_HOME / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    defaults = {
        "README.md": "# Demo project\n\nA tiny project used by the MAgent examples.\n",
        "notes.txt": "deploy on fridays\nrun the tests first\n",
        "src/app.py": "def main():\n    print('hello')\n",
    }
    for name, text in defaults.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(text, encoding="utf-8")
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return root


def heading(text: str) -> None:
    """Print a section heading."""
    print(f"\n\033[1m{text}\033[0m\n" + "-" * len(text))


@contextmanager
def cleanup() -> Iterator[None]:
    """Remove the temporary home when the example exits."""
    try:
        yield
    finally:
        shutil.rmtree(_TEMP_HOME, ignore_errors=True)


def main_guard(fn: Any) -> Any:
    """Wrap an example body so the temporary state is always cleaned up."""
    def runner() -> int:
        try:
            fn()
        finally:
            shutil.rmtree(_TEMP_HOME, ignore_errors=True)
        return 0

    return runner
