# RAG and citations

MAgent's retrieval stack ingests local documents, ranks chunks, builds numbered context, and verifies citations. It works with no external vector database and can use either deterministic local embeddings or Cortex embeddings.

## Retriever

```python
from magent.rag import Retriever

retriever = Retriever(limit=5, mode="hybrid", use_mmr=True)
report = retriever.ingest(["README.md", "docs/"])
print(report.to_dict())

result = retriever.retrieve("How are plugin permissions enforced?", limit=5)
print(result.context)
print(result.sources)
```

`RetrievalResult` exposes methods for context, concatenated text, source list, serialization, and citation verification.

## File-aware chunking

Strategies are selected by suffix/content:

- Markdown preserves headings and paragraph boundaries;
- Python groups top-level definitions with line locations;
- plain text uses paragraph/sentence boundaries;
- structured/text-like files use bounded blocks.

Chunks carry source, index, byte/character offsets, start/end lines, heading, strategy, metadata, score, and stable id. Defaults cap chunks around 1,200 characters with 120 characters of retrieval overlap.

```python
from magent.rag import chunk_file

for chunk in chunk_file("docs/plugins.md"):
    print(chunk.source, chunk.start_line, chunk.end_line, chunk.heading)
```

Binary files, unsupported suffixes, oversized files, hidden paths, build/cache directories, and configured exclusions are skipped with reasons in `IngestReport`.

## Index

`ChunkIndex` stores chunks, BM25 statistics, vectors, and source metadata. It can save/load a versioned JSON index, remove/replace sources, rebuild embeddings, and report statistics.

```python
retriever.save(".magent/docs-index.json")
restored = Retriever.load(".magent/docs-index.json")
```

## Ranking

Hybrid score:

```text
lexical_weight * BM25
+ (1 - lexical_weight) * semantic_weight * max(0, cosine)
+ heading boost
```

The result is clamped to a stable range. Semantic hashing has a non-zero noise floor, so its own `semantic_weight` prevents unrelated cosine noise from overpowering exact lexical evidence.

Retrieval modes:

- `hybrid` — lexical + vector;
- `bm25` — lexical only;
- `vector` — semantic only;
- `none` — insertion/source order for diagnostics.

## Reranking and diversity

Optional stages:

1. near-duplicate removal;
2. relevance sort;
3. MMR selection (`lambda=0.7` by default);
4. recency boost;
5. backend cross-encoder reranking.

MMR uses the relevance score already computed by the index; it does not recompute relevance with a different formula.

## Citations

```python
from magent.rag import cited_context, verify_citations

citation_set = cited_context(result.chunks)
print(citation_set.context)
# ... ask Cortex to answer with [1], [2], ...
check = verify_citations("The manifest gates network access [1].", citation_set.citations)
print(check.coverage, check.dangling, check.grounded)
```

A citation records number, source, heading, line range, chunk id, preview, and score. Verification reports used, valid, dangling, unused, coverage, and grounded status.

## CLI

```bash
magent rag ingest README.md docs/ --save .magent/docs-index.json
magent rag search 'Cortex authentication' --index .magent/docs-index.json
magent rag sources --index .magent/docs-index.json
magent rag stats --index .magent/docs-index.json
```

Use `--pattern` repeatedly to constrain ingestion and `--mode` to change retrieval.

## HTTP

- `POST /api/rag/ingest` ingests configured paths;
- `POST /api/rag/search` returns chunks/context/citations;
- `GET /api/rag/stats` describes the current index.

A server with no index returns an explicit unavailable/not-found response; it never fabricates results.

## Evaluation

The repository's sample corpus contains question/source pairs. The retrieval test requires the expected source at top-1 for its core questions and verifies citation references separately.
