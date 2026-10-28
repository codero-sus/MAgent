#!/usr/bin/env python3
"""Example 04: index documents, retrieve, and verify citations.

Run: ``python examples/04_rag_citations.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import heading, main_guard, workspace
from magent.rag import Retriever


def body() -> None:
    """Ingest a corpus, answer a question from it and check the citation."""
    root = workspace()
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "cortex.md").write_text(
        "# Cortex integration\n\nCortex listens on port 8624 by default and exposes "
        "an OpenAI-compatible API at /v1.\n",
        encoding="utf-8",
    )
    (root / "docs" / "sandbox.md").write_text(
        "# Sandbox\n\nEvery path is resolved and jailed to the workspace; denied "
        "patterns win over allowed ones.\n",
        encoding="utf-8",
    )

    retriever = Retriever()
    report = retriever.ingest_directory(root / "docs")
    heading(
        f"indexed {report.chunks} chunks from {report.files} files in "
        f"{report.seconds * 1000:.0f} ms"
    )

    question = "which port does Cortex listen on"
    result = retriever.retrieve(question, limit=2)
    heading(f"question: {question}")
    print(result.context)

    heading("citation check")
    answer = "Cortex listens on port 8624 by default [1]."
    print(result.verify(answer))

    heading("sources")
    for source in result.citations.sources():
        print(" ", source)


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
