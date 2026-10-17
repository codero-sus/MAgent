#!/usr/bin/env python3
"""Example 07: start the HTTP API and talk to it.

Run: ``python examples/07_server_and_client.py``
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import backend, heading, main_guard, workspace  # noqa: E402

from magent.config import default_config  # noqa: E402
from magent.server import MagentServer, free_port, wait_for_health  # noqa: E402


def get(url: str, path: str, *, method: str = "GET", body: dict | None = None) -> dict:
    """Send one JSON request and decode the reply."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def body() -> None:
    """Serve, then exercise the endpoints the UI uses."""
    config = default_config()
    config.tools.workspace = str(workspace())
    server = MagentServer(config, backend=backend())
    url = server.start_background(host="127.0.0.1", port=free_port())
    try:
        heading(f"serving on {url}")
        print("health:", get(url, "/api/health"))
        print("routes:", len(get(url, "/api")["routes"]))
        print("tools :", get(url, "/api/tools")["count"])
        print("chat  :", get(url, "/api/chat", method="POST",
                             body={"prompt": "say hello over HTTP"})["output"])
        print("tool  :", get(url, "/api/tools/calculate", method="POST",
                             body={"expression": "6*7"})["content"])
        print("skills:", len(get(url, "/api/skills")["skills"]))
    finally:
        server.stop()
    heading("server stopped")


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
