# Memory and RAG

Two different jobs: **memory** keeps what the agent learned across sessions;
**RAG** answers from documents you point it at.

## Short-term memory

A token-aware sliding window with rolling summaries keeps the conversation
inside the model's context.

```python
from magent.memory import ShortTermMemory
from magent.types import Message

memory = ShortTermMemory(max_messages=40, max_tokens=6000, summarize=True)
memory.add(Message.user("we deploy on Fridays at 15:00 UTC"))
memory.add(Message.assistant("noted"))
print(len(memory.messages()), memory.window().summary)
```

Trimming never splits a tool call from its result, and evicted messages are
summarised in batches - one summary call per eviction would be wasteful.

## Long-term memory

Hybrid retrieval over BM25 and vectors, stored in SQLite (JSONL export), with
entity extraction, decay and de-duplication.

```python
from magent.memory import build_store

store = build_store()                       # honours [memory] in the config
store.add("the deploy key lives in ~/.config/magent/secrets.toml", kind="fact", tags=["ops"])
store.add("we deploy on Fridays at 15:00 UTC", kind="fact")
store.save()

for entry in store.search("when do we deploy", limit=3):
    print(f"{entry.score:.2f} {entry.kind} {entry.content}")
```

* `search` blends BM25, vector similarity, recency decay and importance
* `compact` merges near-duplicates above a threshold
* `delete_matching` forgets by query, kind or score
* `export_jsonl` / `import_jsonl` move a store between machines
* `stats`, `entities`, `tags`, `recent`, `by_kind` for inspection

`HashingEmbedder` works with no network; when a Cortex embedding model is
reachable, `build_embedder` uses it instead and the same API applies.

The agent recalls relevant memories automatically when `memory.enabled` is true,
and `--recall` / `--remember` control it per run.

## RAG

```python
from magent.rag import Retriever

retriever = Retriever(strategy="", max_chars=1200, limit=5)   # strategy auto-detected
report = retriever.ingest_directory("docs")
print(report.files, report.chunks, report.seconds)

result = retriever.retrieve("how does the sandbox work", limit=4)
print(result.context)                       # numbered [1] path:lines - heading blocks
print(result.citations.sources())
```

Chunking strategies: `markdown` (headings carried into each chunk), `code` (one
chunk per top-level definition, with the symbol name), `paragraph`, `sentence`
and `fixed`. `chunk_file` picks one by extension.

Ranking is hybrid: BM25 over text **plus** the heading, file stem and symbol
name (so a query naming a file or function finds it), blended with vector
similarity, then MMR for diversity, optional cross-encoder re-ranking through
Cortex, and near-duplicate removal.

Citations are checkable:

```python
result = retriever.retrieve("what port does cortex use")
answer = "Cortex listens on 8624 by default [1]."
print(result.verify(answer))        # coverage, dangling citations, unknown labels
```

## Grounded answers

```python
result = retriever.grounded_answer("what port does cortex use", backend=backend)
print(result["answer"])
print(result["citation_check"]["coverage"])
```

The prompt tells the model to answer only from the retrieved context and to say
so when the corpus does not contain the answer.

## From the CLI

```bash
magent rag ingest docs/                 # index, then cache the index on disk
magent rag status                       # chunks, sources, timings
magent rag search --query "guardrails"  # ranked chunks with locations
magent rag context --query "guardrails" # the exact block the model would see
magent memory list | search | add | forget | compact | stats
```
