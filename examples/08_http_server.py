"""Start MAgent's embedded server, call it, and stop cleanly.

Usage:
    python3 examples/08_http_server.py
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request

from _bootstrap import ROOT  # noqa: F401

from magent import Agent, EchoBackend, load_config
from magent.server import MAgentServer, build_context, free_port


def main() -> None:
    """Serve on an ephemeral loopback port and issue one JSON run request."""
    with tempfile.TemporaryDirectory(prefix="magent-server-example-") as temporary:
        saved = {name: os.environ.get(name) for name in ("XDG_DATA_HOME", "XDG_CONFIG_HOME")}
        os.environ["XDG_DATA_HOME"] = temporary
        os.environ["XDG_CONFIG_HOME"] = temporary
        server: MAgentServer | None = None
        backend: EchoBackend | None = None
        try:
            config = load_config(use_env=False)
            backend = EchoBackend()
            agent = Agent(backend=backend, config=config.agent, settings=config,
                          workspace=ROOT)
            context = build_context(config, agent=agent, backend=backend)
            server = MAgentServer(context, host="127.0.0.1", port=free_port())
            server.start()
            payload = json.dumps({"prompt": "hello over the MAgent API"}).encode("utf-8")
            outgoing = urllib.request.Request(
                server.url + "/api/run",
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(outgoing, timeout=10) as response:
                result = json.load(response)
            print("url:", server.url)
            print("output:", result["output"])
            print("routes:", len(server.router.routes))
        finally:
            if server is not None:
                server.stop()
            if backend is not None:
                backend.close()
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    main()
