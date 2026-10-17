"""Tests for sessions, the output layer and evaluation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from magent.eval import (Dataset, EvalCase, EvalRunner, METRICS, aggregate, build_dataset,
                         load_dataset, metric_names, parse_markdown_dataset, run_eval, sample_dataset,
                         save_dataset, score_case)
from magent.output import (BufferSink, FileSink, NullSink, RenderOptions, ResultRenderer, TeeSink,
                           render_markdown, render_value, to_csv, to_yaml)
from magent.session import (SessionManager, SessionRecord, SessionStore, derive_title,
                            export_transcript, merge_records, parse_transcript, render_transcript,
                            transcript_stats)
from magent.testing import EchoBackend, ToolScriptBackend
from magent.types import Message

from tests.support import MagentTestCase


class SessionRecordTest(MagentTestCase):
    """Session records capture a conversation and its metadata."""

    def test_record_assigns_identity_and_title(self) -> None:
        record = SessionRecord()
        record.add_message(Message.user("please refactor the parser module"))
        self.assertTrue(record.id)
        self.assertIn("refactor", record.title)

    def test_turns_and_usage_accumulate(self) -> None:
        from magent.types import Usage

        record = SessionRecord(title="t")
        record.add_turn("question", "answer", tokens=10)
        record.add_usage(Usage(prompt_tokens=10, completion_tokens=5), cost=0.01)
        self.assertEqual(record.turn_count, 1)
        self.assertEqual(record.tokens, 15)
        self.assertAlmostEqual(record.cost, 0.01)

    def test_round_trip_through_json(self) -> None:
        record = SessionRecord(title="round trip")
        record.add_message(Message.user("hello"))
        restored = SessionRecord.from_json(record.to_json())
        self.assertEqual(restored.id, record.id)
        self.assertEqual(restored.messages[0].text(), "hello")

    def test_title_derivation_truncates(self) -> None:
        title = derive_title("one two three four five six seven eight nine ten")
        self.assertLessEqual(len(title.split()), 8)

    def test_merge_combines_records(self) -> None:
        first = SessionRecord(title="a")
        first.add_message(Message.user("one"))
        second = SessionRecord(title="b")
        second.add_message(Message.user("two"))
        merged = merge_records([first, second])
        self.assertEqual(len(merged.messages), 2)


class SessionStoreTest(MagentTestCase):
    """Stored sessions can be listed, loaded, searched and exported."""

    def setUp(self) -> None:
        super().setUp()
        self.store = SessionStore(self.home.path / "sessions", redact=True, keep_last=3)

    def test_save_and_load(self) -> None:
        record = SessionRecord(title="saved")
        record.add_turn("prompt", "answer")
        path = self.store.save(record)
        self.assertTrue(path.exists())
        loaded = self.store.load(record.id)
        self.assertEqual(loaded.title, "saved")

    def test_secrets_are_redacted_on_write(self) -> None:
        record = SessionRecord(title="keys")
        record.add_turn("here is sk-abcdefghijklmnopqrstuvwxyz", "noted")
        path = self.store.save(record)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", path.read_text(encoding="utf-8"))

    def test_listing_and_search(self) -> None:
        for index in range(3):
            record = SessionRecord(title=f"session {index}")
            record.add_turn(f"question {index}", "answer")
            self.store.save(record)
        self.assertEqual(len(self.store.list()), 3)
        hits = self.store.search("question 1")
        self.assertEqual(len(hits), 1)

    def test_prune_keeps_the_newest(self) -> None:
        store = SessionStore(self.home.path / "sessions")
        for index in range(6):
            store.save(SessionRecord(title=f"old {index}"))
        removed = store.prune(3)
        self.assertEqual(removed, 3)
        self.assertEqual(len(store.list(limit=100)), 3)

    def test_keep_last_prunes_on_save(self) -> None:
        for index in range(6):
            self.store.save(SessionRecord(title=f"capped {index}"))
        self.assertLessEqual(len(self.store.list(limit=100)), 3)

    def test_export_writes_a_transcript(self) -> None:
        record = SessionRecord(title="exported")
        record.add_turn("hello", "hi there")
        self.store.save(record)
        target = export_transcript(record, self.home.path / "out.md", fmt="markdown")
        text = Path(target).read_text(encoding="utf-8")
        self.assertIn("hi there", text)

    def test_stats_describe_the_directory(self) -> None:
        self.store.save(SessionRecord(title="one"))
        stats = self.store.stats()
        self.assertEqual(stats["sessions"], 1)
        self.assertTrue(stats["directory"])


class TranscriptTest(unittest.TestCase):
    """Transcripts render and parse back."""

    def test_markdown_and_html_renderings(self) -> None:
        record = SessionRecord(title="demo")
        record.add_message(Message.user("hello"))
        record.add_message(Message.assistant("hi"))
        self.assertIn("hello", render_transcript(record, fmt="markdown"))
        self.assertIn("<html", render_transcript(record, fmt="html"))
        self.assertIn("user:", render_transcript(record, fmt="text"))
        self.assertIn("\"messages\"", render_transcript(record, fmt="json"))

    def test_parse_transcript_round_trips(self) -> None:
        messages = parse_transcript("user: hello there\nassistant: general kenobi\n")
        self.assertEqual([message.role for message in messages], ["user", "assistant"])

    def test_stats_count_words_and_tools(self) -> None:
        record = SessionRecord(title="stats")
        record.add_turn("q", "a", tools=["calculate"])
        stats = transcript_stats(record)
        self.assertEqual(stats["turns"], 1)
        self.assertIn("calculate", stats["tools"])


class SessionManagerTest(MagentTestCase):
    """The manager ties the agent to a stored session."""

    def test_record_run_records_the_prompt(self) -> None:
        from magent.agent import Agent

        manager = SessionManager(store=SessionStore(self.home.path / "sessions"))
        agent = Agent(backend=EchoBackend(), settings=self.config(), workspace=self.workspace,
                      session=manager)
        agent.run("explain the memory subsystem")
        record = manager.current
        self.assertIsNotNone(record)
        self.assertEqual(record.turns[0].prompt, "explain the memory subsystem")
        self.assertGreater(record.tokens, 0)

    def test_resume_finds_the_newest_session(self) -> None:
        manager = SessionManager(store=SessionStore(self.home.path / "sessions"))
        manager.start(title="first")
        manager.close()
        manager.start(title="second")
        manager.close()
        resumed = manager.resume()
        self.assertEqual(resumed.title, "second")

    def test_handoff_block_summarises_the_session(self) -> None:
        manager = SessionManager(store=SessionStore(self.home.path / "sessions"))
        record = manager.start(title="handoff")
        record.add_turn("do the thing", "done")
        manager.save()
        block = manager.handoff_block()
        self.assertIn("handoff", block)
        self.assertIn("do the thing", block)


class OutputTest(MagentTestCase):
    """Rendering and format conversion."""

    def test_formats_render_consistently(self) -> None:
        payload = {"a": 1, "b": [1, 2]}
        self.assertIn("a", render_value(payload, "text"))
        self.assertEqual(json.loads(render_value(payload, "json"))["a"], 1)
        self.assertIn("a: 1", render_value(payload, "yaml"))
        self.assertIn("a", render_value(payload, "csv"))
        self.assertIn("|", render_value(payload, "md"))

    def test_yaml_emitter_handles_nesting(self) -> None:
        text = to_yaml({"outer": {"inner": [1, 2]}, "flag": True})
        self.assertIn("outer:", text)
        self.assertIn("inner:", text)
        self.assertIn("true", text)

    def test_csv_quotes_commas(self) -> None:
        self.assertIn('"a,b"', to_csv([{"value": "a,b"}]))

    def test_markdown_renders_code_and_lists(self) -> None:
        rendered = render_markdown("# Title\n\n- one\n- two\n\n```python\nx = 1\n```", color=False)
        self.assertIn("Title", rendered)
        self.assertIn("x = 1", rendered)
        self.assertNotIn("```", rendered)

    def test_renderer_prints_a_result(self) -> None:
        from magent.agent import make_agent

        result = make_agent(backend=EchoBackend()).run("hello")
        sink = BufferSink()
        renderer = ResultRenderer(RenderOptions(color=False, unicode=False), sink=sink)
        renderer.print_result(result)
        self.assertIn("hello", sink.text)

    def test_renderer_emits_json(self) -> None:
        from magent.agent import make_agent

        result = make_agent(backend=EchoBackend()).run("hello")
        payload = ResultRenderer(RenderOptions(fmt="json"), sink=BufferSink()).render_result(result)
        self.assertEqual(json.loads(payload)["output"], result.output)

    def test_sinks_compose(self) -> None:
        first, second = BufferSink(), BufferSink()
        tee = TeeSink(first, second)
        tee.line("hello")
        self.assertEqual(first.text, second.text)
        NullSink().line("discarded")

    def test_file_sink_strips_color(self) -> None:
        path = self.home.path / "out.txt"
        with FileSink(path) as sink:
            sink.line("\x1b[31mred\x1b[0m")
        self.assertEqual(path.read_text(encoding="utf-8").strip(), "red")


class EvalTest(MagentTestCase):
    """Datasets, metrics and the runner."""

    def test_sample_dataset_is_available(self) -> None:
        dataset = sample_dataset("demo")
        self.assertGreaterEqual(len(dataset), 4)
        self.assertTrue(dataset.stats()["assertions"])

    def test_dataset_loads_from_json(self) -> None:
        path = self.home.path / "cases.json"
        path.write_text(json.dumps([{"id": "a", "prompt": "hi", "contains": ["hi"]}]),
                        encoding="utf-8")
        dataset = load_dataset(path)
        self.assertEqual(dataset.cases[0].id, "a")

    def test_dataset_loads_from_markdown_table(self) -> None:
        dataset = parse_markdown_dataset(
            "| prompt | expected | contains |\n| --- | --- | --- |\n| What is 2+2? | | 4 |\n")
        self.assertEqual(dataset.cases[0].contains, ("4",))

    def test_dataset_round_trips(self) -> None:
        dataset = sample_dataset("round")
        path = save_dataset(dataset, self.home.path / "dataset.json")
        reloaded = load_dataset(path)
        self.assertEqual(len(reloaded), len(dataset))

    def test_dataset_filter_and_split(self) -> None:
        dataset = sample_dataset("filters")
        self.assertTrue(len(dataset.filter(tag="math")) >= 1)
        first, second = dataset.split(0.5, seed=1)
        self.assertEqual(len(first) + len(second), len(dataset))

    def test_scoring_detects_a_good_answer(self) -> None:
        dataset = sample_dataset()
        case = next(item for item in dataset if item.id == "arith")
        result = self.agent().run("What is 6*7?")
        score = score_case(case, result)
        self.assertTrue(score.outcomes)
        self.assertIn(score.passed, (True, False))

    def test_metrics_registry_is_documented(self) -> None:
        self.assertIn("contains", metric_names())
        self.assertIn("tools", METRICS)

    def test_metric_failure_explains_itself(self) -> None:
        case = EvalCase(prompt="x", contains=("expected-text",))
        result = self.agent().run("x")
        score = score_case(case, result)
        contains = next(item for item in score.outcomes if item.name == "contains")
        self.assertFalse(contains.passed)
        self.assertIn("missing", contains.reason)

    def test_runner_reports_a_pass_rate(self) -> None:
        dataset = build_dataset([{"id": "noop", "prompt": "say hi", "not_contains": ["error"]}])
        report = run_eval(dataset, lambda: self.agent(), concurrency=2)
        self.assertEqual(len(report.runs), 1)
        self.assertGreaterEqual(report.pass_rate, 0.0)
        self.assertIn("summary", report.to_dict())

    def test_runner_repeats_and_reports_flakiness(self) -> None:
        dataset = build_dataset([{"id": "noop", "prompt": "say hi"}])
        report = EvalRunner(lambda: self.agent(), repeats=2).run(dataset)
        self.assertEqual(len(report.runs), 2)

    def test_report_renders_every_format(self) -> None:
        dataset = build_dataset([{"id": "noop", "prompt": "say hi"}])
        report = run_eval(dataset, lambda: self.agent())
        self.assertIn("noop", report.render("text"))
        self.assertIn("# Evaluation", report.render("markdown"))
        self.assertIn("testsuite", report.render("junit"))
        self.assertIn("case", report.render("csv"))

    def test_threshold_failure_raises(self) -> None:
        from magent.errors import EvalThresholdError

        dataset = build_dataset([{"id": "strict", "prompt": "say hi",
                                  "contains": ["definitely-not-present"]}])
        report = EvalRunner(lambda: self.agent(), threshold=0.99).run(dataset)
        with self.assertRaises(EvalThresholdError):
            report.check(0.99)

    def test_aggregate_summarises_scores(self) -> None:
        dataset = build_dataset([{"id": "a", "prompt": "say hi"}])
        report = run_eval(dataset, lambda: self.agent())
        summary = aggregate(report.scores)
        self.assertEqual(summary["cases"], 1)


if __name__ == "__main__":
    unittest.main()
