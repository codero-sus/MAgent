"""Exercise MAgent's Cortex client over real loopback HTTP.

This example uses MAgent's protocol-compatible fake hoster so it runs everywhere.
Replace ``server.url`` with the URL from ``CortexSettings.from_env()`` to target
an external Cortex LLMHoster process.

Usage:
    python3 examples/02_cortex_http.py
"""

from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from magent import CortexBackend, CortexSettings, Message
from magent.llm import CompletionRequest, collect_stream
from magent.testing import EMBEDDING_MODEL, start_fake_cortex


def main() -> None:
    """Call health, models, chat, streaming, and embeddings."""
    server = start_fake_cortex()
    backend = CortexBackend(CortexSettings(
        base_url=server.url,
        require_auth=False,
        max_retries=0,
        embedding_model=EMBEDDING_MODEL,
    ))
    try:
        print("health:", backend.health()["ok"])
        print("models:", ", ".join(backend.model_ids(refresh=True)))

        request = CompletionRequest(messages=[Message.user("hello over Cortex HTTP")])
        print("chat:", backend.complete(request).text)

        print("stream: ", end="")
        response = collect_stream(
            backend.stream(CompletionRequest(messages=[Message.user("stream this")], stream=True)),
            on_chunk=lambda chunk: print(chunk.text, end="", flush=True),
        )
        print(f"\nstream usage: {response.usage.total_tokens} tokens")

        vector = backend.embed_one("MAgent uses Cortex", model=EMBEDDING_MODEL)
        print("embedding dimensions:", len(vector))
    finally:
        backend.close()
        server.stop()


if __name__ == "__main__":
    main()
