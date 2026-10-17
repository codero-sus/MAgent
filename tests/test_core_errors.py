"""Tests for the error hierarchy in :mod:`magent.errors`.

Every error MAgent raises derives from :class:`~magent.errors.MAgentError`, which
carries a stable ``code``, an ``exit_code`` the CLI turns into a process status,
and optional ``details``. Callers branch on the code rather than parsing prose,
so the registry that maps codes to classes has to stay in step with the classes.
"""

from __future__ import annotations

import unittest

from magent import errors
from magent.errors import (
    ERROR_REGISTRY,
    MAgentError,
    classify_status,
    error_from_code,
    is_retryable,
    scrub,
)


class HierarchyTests(unittest.TestCase):
    """One base class, so a caller can catch everything MAgent raises."""

    def test_every_registered_error_derives_from_the_base(self) -> None:
        """The registry holds no foreign exception types."""
        for code, kind in ERROR_REGISTRY.items():
            self.assertTrue(issubclass(kind, MAgentError), f"{code} -> {kind}")

    def test_registry_codes_are_unique_and_snake_case(self) -> None:
        """Codes are machine identifiers, not sentences."""
        self.assertEqual(len(ERROR_REGISTRY), len(set(ERROR_REGISTRY)))
        for code in ERROR_REGISTRY:
            self.assertRegex(code, r"^[a-z0-9_]+$")

    def test_registry_is_substantial(self) -> None:
        """The catalogue covers the failure modes the framework distinguishes."""
        self.assertGreaterEqual(len(ERROR_REGISTRY), 60)

    def test_base_is_an_exception(self) -> None:
        """MAgentError can be raised and caught as an Exception."""
        with self.assertRaises(MAgentError):
            raise MAgentError("boom")
        self.assertTrue(issubclass(MAgentError, Exception))


class DetailTests(unittest.TestCase):
    """Details survive construction and serialization."""

    def test_details_are_attached(self) -> None:
        """A caller can read structured context off the exception."""
        exc = errors.ToolError("failed", details={"tool": "read_file", "path": "/x"})
        self.assertEqual(exc.details["tool"], "read_file")
        self.assertEqual(str(exc), "failed")

    def test_details_default_to_a_mapping(self) -> None:
        """Omitting details yields an empty dict, never None."""
        self.assertEqual(errors.ToolError("failed").details, {})

    def test_to_dict_is_json_shaped(self) -> None:
        """to_dict carries the code, message, and details."""
        payload = errors.ToolError("failed", details={"tool": "t"}).to_dict()["error"]
        self.assertEqual(payload["code"], "tool_error")
        self.assertEqual(payload["message"], "failed")
        self.assertEqual(payload["details"]["tool"], "t")

    def test_subclasses_declare_specific_codes(self) -> None:
        """Related error subclasses still expose distinct stable codes."""
        self.assertEqual(errors.ToolNotFoundError("x").code, "tool_not_found")
        self.assertEqual(errors.ConfigValidationError("x").code, "config_validation_error")
        self.assertNotEqual(errors.ToolError("x").code,
                            errors.ToolNotFoundError("x").code)


class ExitCodeTests(unittest.TestCase):
    """Exit codes are part of the CLI contract."""

    def test_generic_errors_exit_nonzero(self) -> None:
        """A raised error never maps to success."""
        self.assertNotEqual(errors.MAgentError("x").exit_code, 0)

    def test_distinct_stop_conditions_have_distinct_cli_codes(self) -> None:
        """The run command maps stop reasons, not exception classes, to codes."""
        from magent.cli.commands.run import STOP_EXIT_CODES
        from magent.types import StopReason

        codes = {
            STOP_EXIT_CODES[StopReason.MAX_ITERATIONS.value],
            STOP_EXIT_CODES[StopReason.BUDGET.value],
            STOP_EXIT_CODES[StopReason.GUARDRAIL.value],
        }
        self.assertEqual(codes, {3, 4, 5})


class RetryTests(unittest.TestCase):
    """Retryability is declared, not guessed from the message."""

    def test_rate_limits_are_retryable(self) -> None:
        """A 429 is worth another attempt after a backoff."""
        self.assertTrue(is_retryable(errors.RateLimitExceededError("slow down")))

    def test_validation_errors_are_not_retryable(self) -> None:
        """Retrying identical bad arguments cannot succeed."""
        self.assertFalse(is_retryable(errors.ToolValidationError("bad args")))

    def test_non_magent_exceptions_are_classifiable(self) -> None:
        """is_retryable accepts any exception, not just ours."""
        self.assertIsInstance(is_retryable(ValueError("x")), bool)


class StatusClassificationTests(unittest.TestCase):
    """HTTP statuses become the error type a caller should handle."""

    def test_status_maps_to_a_specific_class(self) -> None:
        """Common statuses have dedicated classes."""
        self.assertIsInstance(classify_status(429, "slow down"),
                              errors.RateLimitedError)
        self.assertIsInstance(classify_status(404, "gone"), errors.NotFoundError)
        self.assertIsInstance(classify_status(401, "nope"), errors.AuthenticationError)

    def test_server_errors_use_the_http_server_error(self) -> None:
        """A 5xx status becomes the retryable HTTP ServerError."""
        classified = classify_status(503, "down")
        self.assertIsInstance(classified, errors.ServerError)
        self.assertTrue(classified.retryable)

    def test_every_classification_is_an_http_error(self) -> None:
        """Callers can catch HTTPError for anything status-derived."""
        for status in (400, 401, 403, 404, 408, 413, 422, 429, 500, 502, 503, 504):
            self.assertIsInstance(classify_status(status, "x"), errors.HTTPError)

    def test_unknown_status_still_produces_an_error(self) -> None:
        """An unmapped status falls back rather than raising."""
        self.assertIsInstance(classify_status(418, "teapot"), errors.HTTPError)


class RoundTripTests(unittest.TestCase):
    """A code can be turned back into the class that produced it."""

    def test_error_from_code_recovers_the_class(self) -> None:
        """error_from_code is the inverse of the ``code`` property."""
        exc = error_from_code("tool_not_found", "missing")
        self.assertIsInstance(exc, errors.ToolNotFoundError)
        self.assertEqual(str(exc), "missing")

    def test_every_registered_code_round_trips(self) -> None:
        """No code in the registry fails to resolve."""
        for code, kind in ERROR_REGISTRY.items():
            self.assertIsInstance(error_from_code(code, "x"), kind)

    def test_unknown_code_falls_back_to_the_base(self) -> None:
        """An unrecognized code still yields a catchable MAgentError."""
        self.assertIsInstance(error_from_code("not_a_real_code", "x"), MAgentError)

    def test_details_pass_through(self) -> None:
        """Extra keywords land in details."""
        self.assertEqual(error_from_code("tool_error", "x", tool="t").details["tool"], "t")


class ScrubTests(unittest.TestCase):
    """Secret material must not ride along inside an exception."""

    def test_scrub_masks_secret_shaped_values(self) -> None:
        """A bearer token in details is redacted, not echoed."""
        cleaned = scrub({"authorization": "Bearer sk-supersecretvalue"})
        self.assertNotIn("sk-supersecretvalue", str(cleaned))

    def test_scrub_leaves_ordinary_values_alone(self) -> None:
        """Non-secret context survives, or errors become useless."""
        cleaned = scrub({"tool": "calculate", "count": 3})
        self.assertEqual(cleaned["tool"], "calculate")
        self.assertEqual(cleaned["count"], 3)

    def test_scrub_handles_nested_structures(self) -> None:
        """A secret buried in a list of dicts is still caught."""
        cleaned = scrub({"headers": [{"api_key": "abcdefgh12345678"}]})
        self.assertNotIn("abcdefgh12345678", str(cleaned))


if __name__ == "__main__":
    unittest.main()
