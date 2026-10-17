"""Tests for the embedded HTTP server and its API surface."""

from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

from magent.server import MAgentServer, Router, free_port, wait_for_health
from magent.testing import EchoBackend
from tests.support import MAgentTestCase


class RouterTest(unittest.TestCase):
    """Routing matches paths, parameters and methods."""

    def setUp(self) -> None:
        self.router = Router()
        self.router.get("/api/health", lambda request: (200, {}))
        self.router.get("/api/sessions/{id}", lambda request: (200, request.params))
        self.router.post("/api/chat", lambda request: (200, {}))

    def test_exact_path(self) -> None:
        match = self.router.resolve("GET", "/api/health")
        self.assertTrue(match.found)
        self.assertEqual(match.params, {})

    def test_path_parameters_are_captured(self) -> None:
        match = self.router.resolve("GET", "/api/sessions/ses_123")
        self.assertTrue(match.found)
        self.assertEqual(match.params["id"], "ses_123")

    def test_trailing_slash_is_tolerated(self) -> None:
        self.assertTrue(self.router.resolve("GET", "/api/health/").found)

    def test_method_mismatch_reports_allowed_methods(self) -> None:
        match = self.router.resolve("DELETE", "/api/chat")
        self.assertFalse(match.found)
        self.assertTrue(match.method_not_allowed)
        self.assertEqual(match.allowed, ("POST",))

    def test_unknown_path(self) -> None:
        match = self.router.resolve("GET", "/nope")
        self.assertFalse(match.found)
        self.assertFalse(match.method_not_allowed)

    def test_describe_lists_routes(self) -> None:
        described = self.router.describe()
        self.assertEqual(len(described), 3)
        self.assertIn("path", described[0])


class ServerTest(MAgentTestCase):
    """End-to-end requests against a live server on a loopback port."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create the server that every test in this class shares."""
        super().setUpClass()

    def setUp(self) -> None:
        super().setUp()
        config = self.config()
        config.tools.workspace = str(self.workspace)
        config.session.directory = str(self.home.path / "sessions")
        config.http.api_key = ""
        self.server = MAgentServer(config, backend=EchoBackend())
        self.port = free_port()
        self.url = self.server.start_background(host="127.0.0.1", port=self.port)
        self.assertTrue(wait_for_health(self.url, timeout=5.0))

    def tearDown(self) -> None:
        self.server.stop()
        super().tearDown()

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        """Send one request and decode the JSON reply."""
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return response.status, (
                    json.loads(raw) if raw[:1] in (b"{", b"[") else {"raw": raw.decode()}
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw or b"{}")

    def test_health_is_public(self) -> None:
        status, payload = self._request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_api_index_lists_routes(self) -> None:
        status, payload = self._request("/api")
        self.assertEqual(status, 200)
        self.assertTrue(payload["routes"])

    def test_info_reports_counts(self) -> None:
        status, payload = self._request("/api/info")
        self.assertEqual(status, 200)
        self.assertGreater(payload["tools"], 100)

    def test_tools_listing_and_dispatch(self) -> None:
        status, payload = self._request("/api/tools")
        self.assertEqual(status, 200)
        self.assertGreater(payload["count"], 100)
        status, result = self._request(
            "/api/tools/calculate", method="POST", body={"expression": "6*7"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["content"]["result"], 42)

    def test_unknown_tool_returns_404_with_suggestions(self) -> None:
        status, payload = self._request("/api/tools/read_fil", method="POST", body={})
        self.assertEqual(status, 404)
        self.assertIn("suggestions", payload)

    def test_chat_returns_a_run(self) -> None:
        status, payload = self._request("/api/chat", method="POST", body={"prompt": "hello api"})
        self.assertEqual(status, 200)
        self.assertIn("hello api", payload["output"])
        self.assertEqual(payload["stop_reason"], "completed")

    def test_chat_without_a_prompt_is_rejected(self) -> None:
        status, payload = self._request("/api/chat", method="POST", body={})
        self.assertEqual(status, 400)
        self.assertIn("prompt", payload["error"])

    def test_presets_skills_and_metrics(self) -> None:
        self.assertEqual(self._request("/api/presets")[0], 200)
        self.assertGreaterEqual(len(self._request("/api/skills")[1]["skills"]), 10)
        self.assertEqual(self._request("/api/metrics")[0], 200)

    def test_memory_endpoints(self) -> None:
        status, payload = self._request("/api/memory")
        self.assertEqual(status, 200)
        self.assertIn("entries", payload)

    def test_sessions_endpoint(self) -> None:
        self.assertEqual(self._request("/api/sessions")[0], 200)

    def test_ui_is_served(self) -> None:
        request = urllib.request.Request(self.url + "/", method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            self.assertIn("<!doctype html", body.lower())
            self.assertIn("MAgent", body)
            content_type = response.headers.get("Content-Type", "")
        self.assertIn("text/html", content_type)

    def test_unknown_route_is_404(self) -> None:
        status, payload = self._request("/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("no route", payload["error"])

    def test_method_mismatch_is_405(self) -> None:
        status, payload = self._request("/api", method="DELETE")
        self.assertEqual(status, 405)
        self.assertTrue(payload["allowed"])

    def test_cors_preflight_is_answered(self) -> None:
        request = urllib.request.Request(self.url + "/api/chat", method="OPTIONS")
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual(response.status, 204)
            self.assertIn("Access-Control-Allow-Methods", response.headers)

    def test_sse_stream_emits_events_and_closes(self) -> None:
        request = urllib.request.Request(self.url + "/api/chat/stream?prompt=stream+me")
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
        self.assertIn("event:", body)
        self.assertIn("done", body)

    def test_stats_and_describe(self) -> None:
        self._request("/api/health")
        stats = self.server.stats.to_dict()
        self.assertGreaterEqual(stats["requests"], 1)
        self.assertIn("url", self.server.describe())


class AuthTest(MAgentTestCase):
    """A configured API key guards the non-public routes."""

    def setUp(self) -> None:
        super().setUp()
        config = self.config()
        config.http.api_key = "test-token"
        self.server = MAgentServer(config, backend=EchoBackend())
        self.url = self.server.start_background(host="127.0.0.1", port=free_port())

    def tearDown(self) -> None:
        self.server.stop()
        super().tearDown()

    def _status(self, path: str, token: str = "") -> int:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(self.url + path, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_health_stays_public(self) -> None:
        self.assertEqual(self._status("/api/health"), 200)

    def test_protected_route_requires_a_token(self) -> None:
        self.assertEqual(self._status("/api/info"), 401)
        self.assertEqual(self._status("/api/info", "wrong"), 401)
        self.assertEqual(self._status("/api/info", "test-token"), 200)


if __name__ == "__main__":
    unittest.main()
