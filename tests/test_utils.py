"""Tests for the utility layer: text, files, ids, hashing, tables, numbers, time."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from magent.utils import (
    colors,
    diff,
    files,
    hashing,
    ids,
    json_utils,
    mime,
    numbers,
    tables,
    text,
    timeutil,
    urls,
)


class TextHelpersTest(unittest.TestCase):
    """String helpers behave as documented."""

    def test_slugify_collapses_punctuation(self) -> None:
        self.assertEqual(text.slugify("  Hello, World! -- again  "), "hello-world-again")

    def test_ellipsize_never_exceeds_the_limit(self) -> None:
        self.assertLessEqual(len(text.ellipsize("x" * 100, 20)), 20)

    def test_chunk_text_is_keyword_only(self) -> None:
        chunks = text.chunk_text("a" * 50, max_chars=20, overlap=0)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))

    def test_similarity_is_symmetric_and_bounded(self) -> None:
        self.assertAlmostEqual(text.similarity("abc", "abd"), text.similarity("abd", "abc"))
        self.assertEqual(text.similarity("same", "same"), 1.0)
        self.assertEqual(text.similarity("a", ""), 0.0)

    def test_word_wrap_respects_width(self) -> None:
        wrapped = text.word_wrap("one two three four five six seven", 14)
        self.assertTrue(all(len(line) <= 14 for line in wrapped.splitlines()))

    def test_word_wrap_leaves_narrow_widths_alone(self) -> None:
        self.assertEqual(text.word_wrap("some text", 8), "some text")

    def test_strip_markdown_removes_syntax(self) -> None:
        self.assertEqual(text.strip_markdown("**bold** and `code`").strip(), "bold and code")

    def test_case_helpers(self) -> None:
        self.assertEqual(text.snake_case("HTTPServerError"), "http_server_error")
        self.assertEqual(text.camel_case("hello world"), "helloWorld")
        self.assertEqual(text.pascal_case("hello world"), "HelloWorld")


class FileHelpersTest(unittest.TestCase):
    """File helpers stay inside their roots and report failures."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.txt").write_text("hello", encoding="utf-8")
        (self.tmp / "sub").mkdir()
        (self.tmp / "sub" / "b.py").write_text("print(1)\n", encoding="utf-8")

    def test_iter_files_prunes_excluded_directories(self) -> None:
        (self.tmp / "node_modules").mkdir()
        (self.tmp / "node_modules" / "x.js").write_text("1", encoding="utf-8")
        found = {path.name for path in files.iter_files(self.tmp)}
        self.assertIn("a.txt", found)
        self.assertNotIn("x.js", found)

    def test_is_within_rejects_traversal(self) -> None:
        self.assertTrue(files.is_within(self.tmp / "sub" / "b.py", self.tmp))
        self.assertFalse(files.is_within(self.tmp / ".." / "elsewhere", self.tmp))

    def test_atomic_write_and_read_round_trip(self) -> None:
        target = self.tmp / "out.txt"
        files.atomic_write_text(target, "payload")
        self.assertEqual(files.read_text_safe(target), "payload")

    def test_read_text_safe_honours_limit(self) -> None:
        big = self.tmp / "big.txt"
        big.write_text("z" * 100, encoding="utf-8")
        self.assertEqual(len(files.read_text_safe(big, limit=10)), 10)

    def test_json_round_trip(self) -> None:
        target = self.tmp / "data.json"
        files.write_json(target, {"a": [1, 2]})
        self.assertEqual(files.read_json(target)["a"], [1, 2])

    def test_human_size_and_parse_size_agree(self) -> None:
        self.assertEqual(files.parse_size(files.human_size(4096)), 4096)

    def test_file_sha256_matches_hashing_module(self) -> None:
        path = self.tmp / "a.txt"
        self.assertEqual(files.file_sha256(path), hashing.file_sha256(path))


class IdsAndHashingTest(unittest.TestCase):
    """Identifiers are unique, sortable and prefixed."""

    def test_ids_are_unique_and_prefixed(self) -> None:
        self.assertTrue(ids.run_id().startswith("run_"))
        self.assertTrue(ids.session_id().startswith("ses_"))
        self.assertEqual(len(ids.trace_id()), 32)
        self.assertNotEqual(ids.tool_call_id(), ids.tool_call_id())

    def test_snowflake_increments(self) -> None:
        self.assertLess(ids.snowflake(), ids.snowflake())

    def test_stable_hash_is_stable(self) -> None:
        self.assertEqual(hashing.stable_hash("value"), hashing.stable_hash("value"))
        self.assertNotEqual(hashing.stable_hash("a"), hashing.stable_hash("b"))

    def test_ulid_like_prefix_is_keyword_only(self) -> None:
        self.assertTrue(hashing.ulid_like(prefix="sub_").startswith("sub_"))

    def test_random_token_length(self) -> None:
        # 16 random bytes, base64url encoded without padding
        self.assertEqual(len(ids.random_token(16)), 22)
        self.assertNotEqual(ids.random_token(8), ids.random_token(8))


class JsonHelpersTest(unittest.TestCase):
    """Lenient parsing handles the shapes models actually emit."""

    def test_extract_json_finds_object_in_prose(self) -> None:
        raw = json_utils.extract_json('here you go:\n{"a": 1}\nthanks')
        self.assertEqual(json.loads(raw), {"a": 1})

    def test_extract_json_keeps_arrays(self) -> None:
        self.assertEqual(json.loads(json_utils.extract_json("[1, 2, 3]")), [1, 2, 3])

    def test_repair_json_fixes_trailing_comma(self) -> None:
        self.assertEqual(json.loads(json_utils.repair_json('{"a": 1,}')), {"a": 1})

    def test_deep_merge_nests(self) -> None:
        merged = json_utils.deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
        self.assertEqual(merged, {"a": {"b": 1, "c": 2}})

    def test_flatten_and_unflatten_round_trip(self) -> None:
        flat = json_utils.flatten({"a": {"b": 1}})
        self.assertEqual(json_utils.unflatten(flat), {"a": {"b": 1}})

    def test_parse_jsonl_skips_blank_lines(self) -> None:
        self.assertEqual(json_utils.parse_jsonl('{"a":1}\n\n{"a":2}'), [{"a": 1}, {"a": 2}])


class NumbersTest(unittest.TestCase):
    """Numeric helpers are stable at the boundaries."""

    def test_clamp_and_clamp_int(self) -> None:
        self.assertEqual(numbers.clamp(5, 0, 1), 1)
        self.assertEqual(numbers.clamp_int(-3, 0, 10), 0)

    def test_cosine_similarity_identity(self) -> None:
        self.assertAlmostEqual(numbers.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(numbers.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_safe_div_handles_zero(self) -> None:
        self.assertEqual(numbers.safe_div(1, 0, default=-1), -1)

    def test_percentile_bounds(self) -> None:
        values = list(range(100))
        self.assertEqual(numbers.percentile(values, 0), 0)
        self.assertEqual(numbers.percentile(values, 100), 99)

    def test_softmax_sums_to_one(self) -> None:
        self.assertAlmostEqual(sum(numbers.softmax([1.0, 2.0, 3.0])), 1.0, places=6)


class TablesAndColorsTest(unittest.TestCase):
    """Terminal rendering degrades cleanly."""

    def test_render_table_plain_when_color_disabled(self) -> None:
        rendered = tables.render_table([[1, 2]], headers=["a", "b"], color=False)
        self.assertIn("a", rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_render_key_value_aligns(self) -> None:
        rendered = tables.render_key_value([("a", 1), ("bb", 2)])
        self.assertIn("a ", rendered)

    def test_colorize_accepts_theme_names_and_codes(self) -> None:
        self.assertEqual(colors.strip_ansi(colors.colorize("x", "error", enabled=True)), "x")
        self.assertEqual(
            colors.strip_ansi(colors.colorize("x", colors.ANSI.RED, enabled=True)), "x"
        )

    def test_colorize_drops_unknown_styles(self) -> None:
        self.assertEqual(colors.colorize("x", "not-a-style", enabled=True), "x")

    def test_visible_width_ignores_ansi(self) -> None:
        self.assertEqual(colors.visible_width(colors.colorize("abc", "brand", enabled=True)), 3)
        self.assertEqual(colors.visible_width(colors.colorize("abc", "brand", enabled=False)), 3)

    def test_time_formatting_round_trip(self) -> None:
        self.assertEqual(timeutil.format_duration(timeutil.parse_duration("90s")), "1m 30s")


class DiffAndMiscTest(unittest.TestCase):
    """Diffing, MIME detection and URL helpers."""

    def test_unified_diff_detects_change(self) -> None:
        patch = diff.unified_diff("a\nb\n", "a\nc\n", from_label="before", to_label="after")
        self.assertIn("-b", patch)
        self.assertIn("+c", patch)

    def test_apply_patch_round_trip(self) -> None:
        before = "one\ntwo\nthree\n"
        after = "one\nTWO\nthree\n"
        patch = diff.unified_diff(before, after)
        self.assertEqual(diff.apply_patch(before, patch), after)

    def test_word_diff_and_summary(self) -> None:
        summary = diff.diff_summary("one two", "one three")
        self.assertIn("hunk", summary.lower())
        self.assertIn("+1", summary)
        self.assertIn(("insert", "three"), diff.word_diff("one two", "one three"))

    def test_mime_detection(self) -> None:
        self.assertTrue(mime.is_image(Path("photo.png")))
        self.assertTrue(mime.is_binary(Path("archive.zip")))
        self.assertTrue(mime.is_text_mime("text/plain"))

    def test_url_helpers(self) -> None:
        self.assertEqual(urls.url_origin("https://example.com/a/b?q=1"), "https://example.com")
        self.assertEqual(urls.strip_query("https://example.com/a?x=1"), "https://example.com/a")
        self.assertTrue(urls.is_local_url("http://127.0.0.1:8624"))
        self.assertFalse(urls.is_local_url("https://example.com"))
        self.assertEqual(urls.with_port("http://127.0.0.1", 8624), "http://127.0.0.1:8624")
        self.assertEqual(
            urls.join_url("http://127.0.0.1:8624", "/v1/models"), "http://127.0.0.1:8624/v1/models"
        )


if __name__ == "__main__":
    unittest.main()
