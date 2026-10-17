"""Tests for the shared value types in :mod:`magent.types`.

These are the objects every other module passes around, so their invariants are
worth pinning down: a message round-trips through JSON without losing its tool
calls, usage arithmetic is exact, and the enums keep the wire values the CLI and
the HTTP API both depend on.
"""

from __future__ import annotations

import unittest

from magent.types import (
    EventKind,
    Message,
    ModelInfo,
    Role,
    StopReason,
    TextPart,
    ToolCall,
    ToolResult,
    Usage,
)


class MessageTests(unittest.TestCase):
    """Message construction, serialization, and the OpenAI wire form."""

    def test_role_constructors_set_the_right_role(self) -> None:
        """Each factory produces a message with its own role."""
        self.assertEqual(Message.system("sys").role, Role.SYSTEM)
        self.assertEqual(Message.user("hi").role, Role.USER)
        self.assertEqual(Message.assistant("yo").role, Role.ASSISTANT)

    def test_text_returns_plain_content(self) -> None:
        """text() yields the string body for a simple message."""
        self.assertEqual(Message.user("hello").text(), "hello")

    def test_text_joins_multipart_content(self) -> None:
        """text() concatenates the text parts of a multipart message."""
        message = Message.user([TextPart(text="a"), TextPart(text="b")])
        self.assertEqual(message.text(), "ab")

    def test_multipart_from_the_wire_shape(self) -> None:
        """A JSON content array parses into parts, not raw dicts."""
        message = Message.from_dict({"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        self.assertEqual(message.text(), "ab")
        self.assertFalse(message.has_media())

    def test_round_trip_preserves_content_and_role(self) -> None:
        """to_dict/from_dict is lossless for a plain message."""
        original = Message.user("round trip")
        restored = Message.from_dict(original.to_dict())
        self.assertEqual(restored.role, original.role)
        self.assertEqual(restored.text(), original.text())

    def test_round_trip_preserves_tool_calls(self) -> None:
        """A tool call survives serialization, or the loop cannot continue."""
        call = ToolCall(id="call_1", name="calculate", arguments={"expression": "1+1"})
        original = Message.assistant("", tool_calls=[call])
        restored = Message.from_dict(original.to_dict())
        self.assertEqual(len(restored.tool_calls), 1)
        self.assertEqual(restored.tool_calls[0].name, "calculate")
        self.assertEqual(restored.tool_calls[0].arguments, {"expression": "1+1"})

    def test_wants_tools_only_for_assistant_with_calls(self) -> None:
        """wants_tools() distinguishes a tool turn from a final answer."""
        self.assertTrue(Message.assistant("", tool_calls=[
            ToolCall(id="c", name="t")]).wants_tools())
        self.assertFalse(Message.assistant("done").wants_tools())
        self.assertFalse(Message.user("hi").wants_tools())

    def test_to_openai_uses_the_wire_role_names(self) -> None:
        """The OpenAI payload carries role and content under their wire keys."""
        payload = Message.user("hi").to_openai()
        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["content"], "hi")

    def test_ids_are_unique(self) -> None:
        """Two messages built in the same instant still differ by id."""
        self.assertNotEqual(Message.user("a").id, Message.user("a").id)

    def test_char_count_counts_the_body(self) -> None:
        """char_count() reports the length of the text content."""
        self.assertEqual(Message.user("abcd").char_count(), 4)


class ToolResultTests(unittest.TestCase):
    """Tool results carry success, payload, and failure detail separately."""

    def test_success_defaults(self) -> None:
        """A result is successful unless told otherwise."""
        result = ToolResult(call_id="c1", name="calculate", content={"value": 42})
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)
        self.assertEqual(result.content, {"value": 42})

    def test_failure_carries_code_and_message(self) -> None:
        """A failure keeps both the machine code and the human message."""
        result = ToolResult(call_id="c1", name="read_file", ok=False,
                            error="no such file", error_code="not_found")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "not_found")

    def test_round_trip(self) -> None:
        """to_dict/from_dict preserves every field a caller reads back."""
        original = ToolResult(call_id="c1", name="t", ok=False, error="boom",
                              error_code="tool_error", duration_ms=1.5, truncated=True)
        restored = ToolResult.from_dict(original.to_dict())
        self.assertEqual(restored.call_id, "c1")
        self.assertEqual(restored.error_code, "tool_error")
        self.assertTrue(restored.truncated)
        self.assertAlmostEqual(restored.duration_ms, 1.5)

    def test_to_message_links_the_call(self) -> None:
        """The tool message keeps the call id, or the model cannot match them."""
        message = ToolResult(call_id="call_9", name="t", content="ok").to_message()
        self.assertEqual(message.role, Role.TOOL)
        self.assertEqual(message.tool_call_id, "call_9")

    def test_payload_text_serializes_structured_content(self) -> None:
        """A dict result becomes JSON text the model can read."""
        text = ToolResult(call_id="c", name="t", content={"v": 1}).payload_text()
        self.assertEqual(text, '{"v": 1}')

    def test_payload_text_reports_failures_instead_of_raising(self) -> None:
        """The model has to see what went wrong to recover from it."""
        text = ToolResult(call_id="c", name="t", ok=False, error="boom",
                          error_code="tool_error").payload_text()
        self.assertIn("boom", text)
        self.assertIn("tool_error", text)


class UsageTests(unittest.TestCase):
    """Token accounting has to be exact: it drives budgets and cost."""

    def test_total_tokens_sums_the_two_halves(self) -> None:
        """total_tokens falls back to prompt + completion."""
        usage = Usage(prompt_tokens=10, completion_tokens=5)
        self.assertEqual(usage.total_tokens, 15)

    def test_constructor_and_from_dict_agree(self) -> None:
        """Both construction paths derive the same total.

        A regression guard: the constructor used to leave total_tokens at 0
        while from_dict filled it in, so budgets silently under-counted any
        usage built directly.
        """
        built = Usage(prompt_tokens=10, completion_tokens=5)
        parsed = Usage.from_dict({"prompt_tokens": 10, "completion_tokens": 5})
        self.assertEqual(built.total_tokens, parsed.total_tokens)
        self.assertEqual(built.total_tokens, 15)

    def test_total_tokens_prefers_an_explicit_value(self) -> None:
        """A provider-reported total wins over the derived sum."""
        self.assertEqual(Usage(prompt_tokens=10, completion_tokens=5,
                               total_tokens=99).total_tokens, 99)

    def test_add_mutates_in_place(self) -> None:
        """add() accumulates into the receiver."""
        usage = Usage(prompt_tokens=1, completion_tokens=2)
        usage.add(Usage(prompt_tokens=3, completion_tokens=4))
        self.assertEqual(usage.prompt_tokens, 4)
        self.assertEqual(usage.completion_tokens, 6)

    def test_merge_totals_are_consistent(self) -> None:
        """Merging two derived usages sums the totals, not zero them."""
        merged = Usage(prompt_tokens=10, completion_tokens=5).merge(
            Usage(prompt_tokens=1, completion_tokens=1))
        self.assertEqual(merged.total_tokens, 17)

    def test_merge_returns_a_new_object(self) -> None:
        """merge() leaves both operands untouched."""
        left = Usage(prompt_tokens=1)
        right = Usage(prompt_tokens=2)
        merged = left.merge(right)
        self.assertEqual(merged.prompt_tokens, 3)
        self.assertEqual(left.prompt_tokens, 1)
        self.assertEqual(right.prompt_tokens, 2)

    def test_is_empty(self) -> None:
        """A fresh Usage is empty; any token makes it not."""
        self.assertTrue(Usage().is_empty())
        self.assertFalse(Usage(prompt_tokens=1).is_empty())

    def test_from_dict_tolerates_none(self) -> None:
        """from_dict(None) yields a zeroed usage rather than raising."""
        self.assertTrue(Usage.from_dict(None).is_empty())


class EnumTests(unittest.TestCase):
    """The enum wire values are part of the CLI and HTTP contract."""

    def test_stop_reason_values(self) -> None:
        """Stop reasons keep the values the exit codes are derived from."""
        self.assertEqual(StopReason.COMPLETED.value, "completed")
        self.assertEqual(StopReason.BUDGET.value, "budget_exceeded")
        self.assertEqual(StopReason.MAX_ITERATIONS.value, "max_iterations")

    def test_event_kinds_are_dotted(self) -> None:
        """Event kinds are dotted names, which the SSE console listens for."""
        self.assertEqual(EventKind.RUN_STARTED.value, "run.started")
        self.assertEqual(EventKind.TOOL_CALL.value, "tool.call")
        self.assertEqual(EventKind.TEXT_DELTA.value, "agent.text_delta")

    def test_every_event_kind_is_unique(self) -> None:
        """No two kinds share a wire value."""
        values = [kind.value for kind in EventKind]
        self.assertEqual(len(values), len(set(values)))

    def test_roles_include_tool(self) -> None:
        """The tool role exists, since tool results are messages."""
        self.assertIn("tool", [role.value for role in Role])


class ModelInfoTests(unittest.TestCase):
    """Model metadata drives routing and capability checks."""

    def test_capability_helpers(self) -> None:
        """supports() answers from the declared capability list."""
        model = ModelInfo(id="m", capabilities=["chat", "embeddings"])
        self.assertTrue(model.supports("chat"))
        self.assertFalse(model.supports("vision"))

    def test_round_trip(self) -> None:
        """to_dict/from_dict preserves the identity and capabilities."""
        original = ModelInfo(id="m", runtime="local", capabilities=["chat"], default=True)
        restored = ModelInfo.from_dict(original.to_dict())
        self.assertEqual(restored.id, "m")
        self.assertTrue(restored.default)
        self.assertIn("chat", restored.capabilities)


class ToolCallTests(unittest.TestCase):
    """Tool calls are what the model emits and the registry dispatches."""

    def test_arguments_default_to_an_empty_mapping(self) -> None:
        """A call with no arguments still has a dict, not None."""
        self.assertEqual(ToolCall(id="c", name="t").arguments, {})

    def test_round_trip(self) -> None:
        """Serialization keeps the id, name, and arguments."""
        original = ToolCall(id="c1", name="calculate", arguments={"expression": "2*3"})
        restored = ToolCall.from_dict(original.to_dict())
        self.assertEqual(restored.id, "c1")
        self.assertEqual(restored.arguments["expression"], "2*3")

    def test_signature_is_stable(self) -> None:
        """Two identical calls share a signature, which caching relies on."""
        left = ToolCall(id="a", name="t", arguments={"x": 1, "y": 2})
        right = ToolCall(id="b", name="t", arguments={"y": 2, "x": 1})
        self.assertEqual(left.signature(), right.signature())


if __name__ == "__main__":
    unittest.main()
