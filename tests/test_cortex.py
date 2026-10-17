"""End-to-end tests of MAgent against Cortex LLMHoster's HTTP contract.

The server is :class:`magent.testing.FakeCortexServer`: a real threaded HTTP
server that implements Cortex 0.2's liveness, readiness, OpenAI-compatible
``/v1`` routes, SSE framing, bearer authentication, embeddings, and supported
``/admin/api`` routes. Therefore these tests exercise the same stdlib transport,
JSON parsing, auth headers, retry classification, and streamed tool-call merger
used with an external Cortex process; no source from Cortex is vendored and no
model weights or network access are needed.
"""

from __future__ import annotations

import math
import unittest

from magent.errors import AuthenticationError, CapabilityNotSupportedError, ServerError
from magent.llm import (
    CortexAdmin,
    CortexBackend,
    CortexSettings,
    CompletionRequest,
    collect_stream,
)
from magent.testing import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FakeCortexServer,
    FakeCortexState,
    fake_embedding,
)
from magent.types import Capability, Message


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


class CortexHTTPTestCase(unittest.TestCase):
    """One shared fake process with fresh telemetry before every test."""

    server: FakeCortexServer
    backend: CortexBackend

    @classmethod
    def setUpClass(cls) -> None:
        """Bind an ephemeral loopback port once for the class."""
        cls.server = FakeCortexServer().start()
        cls.backend = CortexBackend(CortexSettings(
            base_url=cls.server.url,
            require_auth=False,
            timeout=5.0,
            connect_timeout=2.0,
            max_retries=0,
            embedding_model=EMBEDDING_MODEL,
        ))

    @classmethod
    def tearDownClass(cls) -> None:
        """Close pooled connections and release the socket."""
        cls.backend.close()
        cls.server.stop()

    def setUp(self) -> None:
        """Clear request counters and restore the default model to running."""
        self.server.reset()
        for model in self.server.state.models.values():
            model.running = model.id != "stopped-model-q4"

    def request(self, text: str, **kwargs: object) -> CompletionRequest:
        """Build one user request."""
        return CompletionRequest(messages=[Message.user(text)], **kwargs)


class HealthTests(CortexHTTPTestCase):
    """Liveness, readiness, URL construction, and model discovery."""

    def test_health_hits_root_routes_not_v1(self) -> None:
        """Cortex serves /health and /ready at the root."""
        health = self.backend.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["liveness"]["body"], {"status": "ok"})
        self.assertEqual(health["readiness"]["body"]["status"], "ready")

    def test_ping_reports_reachable(self) -> None:
        """ping is a cheap boolean liveness check."""
        self.assertTrue(self.backend.ping())

    def test_url_helpers_match_cortex_layout(self) -> None:
        """OpenAI routes live under /v1 and admin routes under /admin/api."""
        self.assertEqual(self.backend.root_url("/health"), self.server.url + "/health")
        self.assertEqual(self.backend.api_url("/models"), self.server.url + "/v1/models")
        self.assertEqual(self.backend.admin_url("/state"),
                         self.server.url + "/admin/api/state")

    def test_models_parse_the_real_list_shape(self) -> None:
        """GET /v1/models parses object/list/data records into ModelInfo."""
        models = self.backend.models()
        self.assertGreaterEqual(len(models), 4)
        by_id = {model.id: model for model in models}
        self.assertTrue(by_id[DEFAULT_MODEL].default)
        self.assertEqual(by_id[DEFAULT_MODEL].runtime, "llama.cpp")
        self.assertTrue(by_id[DEFAULT_MODEL].supports("tool_calling"))
        self.assertTrue(by_id[EMBEDDING_MODEL].supports("embedding"))

    def test_model_helpers(self) -> None:
        """The default id and point lookup are derived from the model list."""
        self.assertEqual(self.backend.default_model_id(), DEFAULT_MODEL)
        self.assertIn(DEFAULT_MODEL, self.backend.model_ids(refresh=True))
        self.assertEqual(self.backend.get_model(DEFAULT_MODEL).id, DEFAULT_MODEL)  # type: ignore[union-attr]
        self.assertIsNone(self.backend.get_model("does-not-exist"))

    def test_describe_redacts_connection_secrets(self) -> None:
        """Diagnostics report the endpoints but never the API key."""
        backend = CortexBackend(CortexSettings(base_url=self.server.url,
                                               api_key="secret-token", require_auth=False))
        described = backend.describe()
        self.assertNotIn("secret-token", str(described))
        self.assertIn(self.server.url, str(described))
        backend.close()


class CompletionTests(CortexHTTPTestCase):
    """Chat, legacy completion, helpers, and errors use Cortex's wire shapes."""

    def test_chat_completion(self) -> None:
        """A full /v1/chat/completions response becomes ChatResponse."""
        response = self.backend.complete(self.request("hello cortex"))
        self.assertEqual(response.text, "Fake Cortex reply to: hello cortex")
        self.assertEqual(response.model, DEFAULT_MODEL)
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertEqual(response.usage.total_tokens,
                         response.usage.prompt_tokens + response.usage.completion_tokens)

    def test_chat_helper_accepts_messages(self) -> None:
        """chat is a typed convenience around complete."""
        response = self.backend.chat([Message.system("brief"), Message.user("hi")])
        self.assertEqual(response.text, "Fake Cortex reply to: hi")

    def test_ask_returns_only_text(self) -> None:
        """ask is the smallest synchronous convenience API."""
        self.assertEqual(self.backend.ask("what now?"),
                         "Fake Cortex reply to: what now?")

    def test_legacy_completion(self) -> None:
        """POST /v1/completions is implemented separately from chat."""
        response = self.backend.completion("complete me")
        self.assertIn("complete me", response.text)
        self.assertGreater(response.usage.total_tokens, 0)

    def test_requested_model_is_sent(self) -> None:
        """An explicit model survives request serialization and response parsing."""
        response = self.backend.complete(self.request("vision question",
                                                      model="vision-model-q4"))
        self.assertEqual(response.model, "vision-model-q4")
        self.assertEqual(self.server.state.transcripts[-1]["model"], "vision-model-q4")

    def test_tool_call_parsing(self) -> None:
        """Cortex's function.arguments JSON string becomes a dict."""
        response = self.backend.complete(self.request(
            "please calculate 6 times 7", tools=[CALCULATOR_SCHEMA]))
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].name, "calculator")
        self.assertIn("expression", response.tool_calls[0].arguments)

    def test_tool_result_continues_the_conversation(self) -> None:
        """A tool role message is accepted on the next Cortex turn."""
        messages = [
            Message.user("calculate"),
            Message.assistant("", tool_calls=[]),
            Message(role="tool", content='{"result": 42}', name="calculator",
                    tool_call_id="call_1"),
        ]
        response = self.backend.complete(CompletionRequest(messages=messages))
        self.assertIn("42", response.text)
        self.assertIn("tool result", response.text)

    def test_server_error_is_classified(self) -> None:
        """A Cortex 503 becomes a retryable ServerError."""
        self.server.fail_next(DEFAULT_MODEL)
        with self.assertRaises(ServerError) as caught:
            self.backend.complete(self.request("fail once"))
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.status, 503)

    def test_failed_request_does_not_poison_next_call(self) -> None:
        """The transport remains reusable after a JSON error response."""
        self.server.fail_next(DEFAULT_MODEL)
        with self.assertRaises(ServerError):
            self.backend.complete(self.request("first"))
        self.assertIn("second", self.backend.complete(self.request("second")).text)


class StreamingTests(CortexHTTPTestCase):
    """SSE text and tool-call deltas are reassembled losslessly."""

    def test_text_stream(self) -> None:
        """Several data frames and [DONE] become one ChatResponse."""
        response = collect_stream(self.backend.stream(self.request("stream this response")))
        self.assertEqual(response.text, "Fake Cortex reply to: stream this response")
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreater(response.usage.total_tokens, 0)

    def test_stream_has_incrementing_indices_and_terminal_chunk(self) -> None:
        """Every parsed frame is ordered and exactly one is terminal."""
        chunks = list(self.backend.stream(self.request("stream me")))
        self.assertEqual([chunk.index for chunk in chunks], list(range(1, len(chunks) + 1)))
        self.assertTrue(chunks[-1].done)
        self.assertFalse(any(chunk.done for chunk in chunks[:-1]))

    def test_streamed_tool_arguments_are_merged(self) -> None:
        """Split tool-call argument fragments parse into one complete call."""
        response = collect_stream(self.backend.stream(self.request(
            "calculate 8 plus 9", tools=[CALCULATOR_SCHEMA])))
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual(call.name, "calculator")
        self.assertIsInstance(call.arguments, dict)
        self.assertIn("expression", call.arguments)

    def test_stream_callback_matches_collected_text(self) -> None:
        """Rendering callbacks receive every non-empty delta exactly once."""
        deltas: list[str] = []
        response = collect_stream(self.backend.stream(self.request("callback")),
                                  on_chunk=lambda chunk: deltas.append(chunk.text))
        self.assertEqual("".join(deltas), response.text)


class EmbeddingTests(CortexHTTPTestCase):
    """The embeddings route supports batches, dimensions, and deterministic use."""

    def test_batch_embeddings(self) -> None:
        """One vector is returned per input in the same order."""
        vectors = self.backend.embed(["alpha", "beta", "gamma"])
        self.assertEqual(len(vectors), 3)
        self.assertTrue(all(len(vector) == EMBEDDING_DIM for vector in vectors))
        self.assertNotEqual(vectors[0], vectors[1])

    def test_embedding_is_normalized(self) -> None:
        """Cosine similarity is meaningful because vectors have unit norm."""
        vector = self.backend.embed_one("normalized vector")
        norm = math.sqrt(sum(value * value for value in vector))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_embedding_matches_the_fake_contract(self) -> None:
        """The HTTP result is exactly what the reference fake generated."""
        self.assertEqual(self.backend.embed_one("same text"), fake_embedding("same text"))

    def test_repeated_input_is_deterministic(self) -> None:
        """Identical input and model produce identical vectors."""
        self.assertEqual(self.backend.embed_one("repeat"),
                         self.backend.embed_one("repeat"))

    def test_dimension_override(self) -> None:
        """The optional dimensions field is sent and honored."""
        vector = self.backend.embed(["small"], dimensions=8)[0]
        self.assertEqual(len(vector), 8)


class CapabilityTests(CortexHTTPTestCase):
    """Capability checks use model metadata before expensive requests."""

    def test_supported_capability(self) -> None:
        """A declared capability returns normally."""
        self.assertIsNone(self.backend.assert_capability(DEFAULT_MODEL, Capability.CHAT))

    def test_unsupported_capability_raises_specific_error(self) -> None:
        """A text-only model cannot silently accept vision work."""
        with self.assertRaises(CapabilityNotSupportedError):
            self.backend.assert_capability(DEFAULT_MODEL, Capability.VISION)


class AuthenticationTests(unittest.TestCase):
    """Bearer authentication is verified over a real HTTP exchange."""

    def setUp(self) -> None:
        self.server = FakeCortexServer(FakeCortexState(api_key="top-secret")).start()

    def tearDown(self) -> None:
        self.server.stop()

    def backend(self, key: str) -> CortexBackend:
        return CortexBackend(CortexSettings(base_url=self.server.url, api_key=key,
                                            timeout=3.0, max_retries=0))

    def test_correct_bearer_token(self) -> None:
        """The configured key is sent as Authorization: Bearer."""
        backend = self.backend("top-secret")
        self.assertIn(DEFAULT_MODEL, backend.model_ids(refresh=True))
        backend.close()

    def test_missing_token_is_rejected(self) -> None:
        """401 becomes AuthenticationError with actionable guidance."""
        backend = self.backend("")
        with self.assertRaises(AuthenticationError) as caught:
            backend.models()
        self.assertIn("MAGENT_CORTEX_API_KEY", str(caught.exception))
        backend.close()

    def test_wrong_token_is_rejected(self) -> None:
        """A non-empty but wrong token is not treated as anonymous access."""
        backend = self.backend("wrong")
        with self.assertRaises(AuthenticationError):
            backend.complete(CompletionRequest(messages=[Message.user("hi")]))
        backend.close()

    def test_settings_redact_key_by_default(self) -> None:
        """Diagnostics do not expose the Bearer credential."""
        settings = CortexSettings(api_key="top-secret")
        self.assertNotIn("top-secret", str(settings.to_dict()))
        self.assertEqual(settings.to_dict(redact=False)["api_key"], "top-secret")


class SettingsTests(unittest.TestCase):
    """MAgent-prefixed variables take precedence over Cortex fallbacks."""

    def test_defaults_match_cortex(self) -> None:
        """The integration targets Cortex's actual default bind."""
        settings = CortexSettings.from_env({})
        self.assertEqual(settings.base_url, "http://127.0.0.1:8624")
        self.assertEqual(settings.api_prefix, "/v1")

    def test_magent_variables_win(self) -> None:
        """Application-specific settings override generic Cortex variables."""
        settings = CortexSettings.from_env({
            "MAGENT_CORTEX_URL": "http://127.0.0.1:9000/",
            "CORTEX_URL": "http://127.0.0.1:8000",
            "MAGENT_CORTEX_API_KEY": "magent-key",
            "CORTEX_API_KEY": "cortex-key",
        })
        self.assertEqual(settings.base_url, "http://127.0.0.1:9000")
        self.assertEqual(settings.api_key, "magent-key")

    def test_boolean_and_timeout_parsing(self) -> None:
        """Common environment spellings become typed settings."""
        settings = CortexSettings.from_env({
            "MAGENT_CORTEX_TIMEOUT": "4.5",
            "MAGENT_CORTEX_VERIFY_SSL": "false",
            "MAGENT_CORTEX_REQUIRE_AUTH": "0",
        })
        self.assertEqual(settings.timeout, 4.5)
        self.assertFalse(settings.verify_ssl)
        self.assertFalse(settings.require_auth)


class AdminTests(CortexHTTPTestCase):
    """The supported /admin/api routes are parsed into typed snapshots."""

    def setUp(self) -> None:
        super().setUp()
        self.admin = CortexAdmin(backend=self.backend)

    def test_state(self) -> None:
        """Dashboard state exposes version, auth, models, hardware, and runtime."""
        state = self.admin.state()
        self.assertEqual(state.version, "0.2.0-fake")
        self.assertFalse(state.auth_enabled)
        self.assertEqual(state.model_status(DEFAULT_MODEL), "running")
        self.assertIn(DEFAULT_MODEL, state.running_models())
        self.assertGreaterEqual(len(state.model_infos), 4)

    def test_metrics_after_completion(self) -> None:
        """Inference counters reflect actual requests to the fake hoster."""
        self.backend.complete(self.request("metric event"))
        metrics = self.admin.metrics()
        self.assertGreaterEqual(metrics.total_requests, 1)
        self.assertGreaterEqual(metrics.completed_requests, 1)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertIn(DEFAULT_MODEL, metrics.requests_by_model)

    def test_runtime_and_hardware(self) -> None:
        """Runtime status and hardware use Cortex's nested state shape."""
        runtime = self.admin.runtime()
        hardware = self.admin.hardware()
        self.assertEqual(runtime["models"][DEFAULT_MODEL]["status"], "running")
        self.assertEqual(hardware["cpu_count"], 8)
        self.assertEqual(hardware["platform"], "fake")

    def test_stop_then_start_model(self) -> None:
        """Control routes mutate the runtime state and return it."""
        stopped = self.admin.stop_model(DEFAULT_MODEL)
        self.assertTrue(stopped["ok"])
        self.assertEqual(self.admin.model_status(DEFAULT_MODEL), "stopped")
        started = self.admin.start_model(DEFAULT_MODEL, wait=False)
        self.assertTrue(started["ok"])
        self.assertEqual(self.admin.model_status(DEFAULT_MODEL), "running")

    def test_restart_model(self) -> None:
        """Restart returns a running model record."""
        response = self.admin.restart_model(DEFAULT_MODEL, wait=False)
        self.assertTrue(response["ok"])
        self.assertEqual(response["model"]["status"], "running")

    def test_model_smoke_test(self) -> None:
        """The admin test route returns a model-specific pong."""
        response = self.admin.test_model(DEFAULT_MODEL, prompt="ping")
        self.assertTrue(response["ok"])
        self.assertEqual(response["model"], DEFAULT_MODEL)
        self.assertEqual(response["reply"], "pong")

    def test_iter_running_models(self) -> None:
        """The convenience iterator excludes stopped runtimes."""
        running = set(self.admin.iter_running_models())
        self.assertIn(DEFAULT_MODEL, running)
        self.assertNotIn("stopped-model-q4", running)


if __name__ == "__main__":
    unittest.main()
