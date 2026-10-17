"""Ingest sample documents, retrieve context, and verify citations.

Usage:
    python3 examples/05_rag_citations.py
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from magent.rag import Retriever
from magent.testing import build_sample_corpus, temp_workspace


def main() -> None:
    """Run deterministic hybrid retrieval over a temporary corpus."""
    with temp_workspace(build_sample_corpus()) as workspace:
        retriever = Retriever(mode="hybrid", limit=3, use_mmr=True)
        report = retriever.ingest([workspace])
        print(f"ingested {report.files} files into {report.chunks} chunks")

        result = retriever.retrieve("What port does Cortex use by default?")
        print(result.context)
        print("sources:", result.sources)
        print("citation check:", result.verify("Cortex listens on port 8624 [1]."))


if __name__ == "__main__":
    main()
