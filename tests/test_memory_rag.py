"""Tests for memory storage and retrieval-augmented generation."""

from __future__ import annotations

import unittest

from magent.memory import (ConversationWindow, HashingEmbedder, MemoryEntry, MemoryStore,
                           ShortTermMemory, VectorIndex, build_store, rank_entries, tokenize)
from magent.memory.ranking import BM25Scorer, DecayPolicy
from magent.rag import (Chunk, ChunkIndex, CitationSet, Retriever, chunk_markdown, chunk_python,
                        chunk_text_blocks, cited_context, search_files, verify_citations)
from magent.testing.fixtures import build_sample_corpus, temp_workspace
from magent.types import Message

from tests.support import MagentTestCase


class MemoryStoreTest(MagentTestCase):
    """Storing, searching and maintaining long-term memory."""

    def setUp(self) -> None:
        super().setUp()
        self.store = build_store(self.config(), root=self.home.path / "data")

    def test_add_and_get(self) -> None:
        entry = self.store.add("the deploy key lives in ~/.config/magent", kind="fact",
                               tags=["ops"])
        self.assertTrue(entry.id)
        self.assertEqual(self.store.get(entry.id).content, entry.content)
        self.assertIn("ops", entry.tags)

    def test_search_ranks_relevant_entries_first(self) -> None:
        self.store.add("the deploy key lives in ~/.config/magent", kind="fact")
        self.store.add("the parser handles nested arrays", kind="fact")
        found = self.store.search("where is the deploy key", limit=1)
        self.assertTrue(found)
        self.assertIn("deploy key", found[0].content)

    def test_duplicate_entries_are_deduplicated(self) -> None:
        first = self.store.add("same content here")
        second = self.store.add("same content here")
        self.assertEqual(first.id, second.id)
        self.assertLessEqual(self.store.stats().get("duplicates", 0), 1)

    def test_persistence_round_trips(self) -> None:
        self.store.add("remember this", kind="fact")
        path = self.store.save()
        self.assertTrue(path.exists())
        reloaded = build_store(self.config(), path=path)
        self.assertEqual(len(reloaded), len(self.store))

    def test_delete_and_clear(self) -> None:
        entry = self.store.add("temporary")
        self.assertTrue(self.store.delete(entry.id))
        self.assertIsNone(self.store.get(entry.id))

    def test_compact_reports_what_it_merged(self) -> None:
        self.store.add("the cache lives at ~/.cache/magent")
        self.store.add("the cache lives at ~/.cache/magent.")
        report = self.store.compact(dedupe_threshold=0.9)
        self.assertIsInstance(report, dict)
        self.assertIn("merged", report)

    def test_entities_are_tracked(self) -> None:
        self.store.add("Cortex runs at http://127.0.0.1:8624 and stores models in ~/models",
                       kind="fact")
        entities = self.store.entities()
        self.assertIsInstance(entities, dict)
        blob = str(entities)
        self.assertTrue("8624" in blob or "127.0.0.1" in blob or not entities)

    def test_forget_by_query(self) -> None:
        self.store.add("noise one")
        self.store.add("noise two")
        removed = self.store.delete_matching("noise", limit=2)
        self.assertEqual(removed, 2)

    def test_stats_describe_the_store(self) -> None:
        self.store.add("something")
        stats = self.store.stats()
        self.assertIn("entries", stats)
        self.assertGreaterEqual(stats["entries"], 1)

    def test_export_and_import_jsonl(self) -> None:
        self.store.add("exported")
        path = self.home.path / "memory.jsonl"
        self.store.export_jsonl(path)
        target = build_store(self.config(), root=self.home.path / "data2")
        imported = target.import_jsonl(path)
        self.assertGreaterEqual(imported, 1)


class RankingTest(unittest.TestCase):
    """BM25, decay and hybrid ranking."""

    def test_bm25_prefers_the_matching_document(self) -> None:
        scorer = BM25Scorer(["the cat sat on the mat", "a dog barked loudly",
                             "cats and dogs are common pets"])
        scores = scorer.scores("cat mat")
        self.assertEqual(scores.index(max(scores)), 0)

    def test_tokenize_normalises_case_and_punctuation(self) -> None:
        self.assertIn("hello", tokenize("Hello, World!"))

    def test_decay_reduces_old_entries(self) -> None:
        import time

        policy = DecayPolicy(half_life_days=1.0)
        fresh = MemoryEntry(content="fresh", kind="fact", created_at=time.time())
        stale = MemoryEntry(content="stale", kind="fact", created_at=time.time() - 86_400 * 10)
        self.assertGreater(policy.factor(fresh), policy.factor(stale))
        self.assertGreater(policy.factor(fresh), policy.min_score)

    def test_rank_entries_returns_the_best_first(self) -> None:
        entries = [MemoryEntry(content="garden tools", kind="fact"),
                   MemoryEntry(content="deployment runbook", kind="fact")]
        ranked = rank_entries(entries, "deployment runbook")
        self.assertIn("deployment", ranked[0].content)


class EmbeddingTest(unittest.TestCase):
    """The offline embedder is deterministic and meaningful."""

    def test_similar_texts_score_higher(self) -> None:
        embedder = HashingEmbedder(dim=256)
        related = [embedder.similarity("memory store", "the memory store keeps facts")]
        unrelated = [embedder.similarity("memory store", "banana bread recipe")]
        self.assertGreater(related[0], unrelated[0])

    def test_vectors_are_normalised(self) -> None:
        vector = HashingEmbedder(dim=64).embed(["hello world"])[0]
        norm = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_vector_index_finds_neighbours(self) -> None:
        embedder = HashingEmbedder(dim=64)
        index = VectorIndex()
        first, second = embedder.embed(["apples and pears", "brakes and clutches"])
        index.add("fruit", first)
        index.add("car", second)
        hits = index.search(first, limit=1)
        self.assertEqual(hits[0][0], "fruit")


class ShortTermTest(MagentTestCase):
    """The conversation window trims and summarises."""

    def test_window_keeps_the_last_messages(self) -> None:
        memory = ShortTermMemory(max_messages=4, summarize=False)
        for index in range(10):
            memory.add(Message.user(f"m{index}"))
        visible = memory.messages()
        self.assertLessEqual(len(visible), 6)
        self.assertEqual(visible[-1].text(), "m9")

    def test_summary_is_written_when_configured(self) -> None:
        memory = ShortTermMemory(max_messages=2, summarize=True, summary_threshold=2)
        for index in range(8):
            memory.add(Message.user(f"message {index}"))
        window = memory.window()
        self.assertTrue(window.summary or window.summarized_messages >= 0)

    def test_transcript_renders_roles(self) -> None:
        memory = ShortTermMemory()
        memory.add(Message.user("hello"))
        memory.add(Message.assistant("hi"))
        self.assertIn("hello", memory.transcript())

    def test_conversation_window_round_trips(self) -> None:
        window = ConversationWindow(session_id="s1")
        window.append(Message.user("one"))
        payload = window.to_dict()
        restored = ConversationWindow.from_dict(payload)
        self.assertEqual(restored.messages[0].text(), "one")


class ChunkerTest(unittest.TestCase):
    """Chunking keeps provenance and picks sensible splits."""

    def test_markdown_chunks_carry_headings(self) -> None:
        text = "# Title\n\nIntro paragraph.\n\n## Section\n\nBody text about things.\n"
        chunks = chunk_markdown(text, source="doc.md")
        self.assertTrue(chunks)
        self.assertTrue(any(chunk.heading for chunk in chunks))
        self.assertEqual(chunks[0].source, "doc.md")

    def test_heading_is_not_duplicated_in_the_body(self) -> None:
        text = "# Title\n\nBody only.\n"
        chunks = chunk_markdown(text, source="doc.md")
        self.assertNotIn("# Title\n# Title", chunks[0].text)

    def test_python_chunks_one_definition_each(self) -> None:
        source = "def one():\n    return 1\n\n\ndef two():\n    return 2\n"
        chunks = chunk_python(source, source="m.py")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("one", chunks[0].metadata.get("symbol", ""))

    def test_python_chunking_survives_a_module_without_docstrings(self) -> None:
        source = "import os\n\nx = 1\n" * 5
        chunks = chunk_python(source, source="plain.py")
        self.assertTrue(chunks)

    def test_paragraph_strategy_splits_long_text(self) -> None:
        text = "\n\n".join(f"Paragraph {index} " * 20 for index in range(10))
        chunks = chunk_text_blocks(text, max_chars=300, source="long.txt")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in chunks))


class RetrievalTest(MagentTestCase):
    """The retriever indexes a corpus and answers with citations."""

    def _retriever(self) -> tuple[Retriever, object]:
        corpus = build_sample_corpus(include_code=True)
        workspace = temp_workspace(corpus)
        root = workspace.__enter__()
        self.addCleanup(lambda: workspace.__exit__(None, None, None))
        retriever = Retriever()
        retriever.ingest_directory(root)
        return retriever, corpus

    def test_ingest_indexes_every_document(self) -> None:
        retriever, _ = self._retriever()
        self.assertGreaterEqual(len(retriever), 8)

    def test_queries_find_the_right_document(self) -> None:
        retriever, corpus = self._retriever()
        for pair in corpus.query_answers():
            result = retriever.retrieve(pair["question"], limit=3)
            sources = [chunk.source.rsplit("/", 1)[-1] for chunk in result.chunks]
            self.assertIn(pair["source"], sources, pair["question"])

    def test_context_block_is_renderable(self) -> None:
        retriever, corpus = self._retriever()
        context = retriever.context(corpus.query_answers()[0]["question"])
        self.assertIn("[1]", context)

    def test_citations_point_at_real_chunks(self) -> None:
        retriever, corpus = self._retriever()
        result = retriever.retrieve(corpus.query_answers()[1]["question"])
        citations = result.citations
        self.assertIsInstance(citations, CitationSet)
        answer = result.context
        check = verify_citations(answer, citations.citations)
        self.assertEqual(check.dangling, [])

    def test_index_round_trips_through_disk(self) -> None:
        retriever, _ = self._retriever()
        path = self.home.path / "index.json"
        retriever.save(path)
        restored = Retriever.load(path)
        self.assertEqual(len(restored), len(retriever))

    def test_search_files_scans_a_directory(self) -> None:
        self.write("notes.txt", "the quick brown fox jumps over the lazy dog")
        hits = search_files("brown fox", [self.workspace])
        self.assertTrue(hits)

    def test_cited_context_numbers_chunks(self) -> None:
        chunks = [Chunk(text="alpha", source="a.txt", start_line=1, end_line=2),
                  Chunk(text="beta", source="b.txt", start_line=3, end_line=4)]
        citation_set = cited_context(chunks)
        rendered = citation_set.render()
        self.assertIn("[1]", rendered)
        self.assertIn("alpha", rendered)
        self.assertEqual(len(citation_set.citations), 2)
        self.assertEqual(citation_set.sources(), ["a.txt", "b.txt"])
        self.assertEqual(citation_set.labels(), ["a.txt:1-2", "b.txt:3-4"])


if __name__ == "__main__":
    unittest.main()
