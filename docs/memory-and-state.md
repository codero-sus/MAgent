# Memory and state

MAgent separates four forms of state:

1. model-visible conversation messages;
2. a token-bounded short-term window;
3. persistent long-term memory records;
4. saved run sessions/transcripts.

This separation makes retention and deletion explicit.

## Conversation messages

`Message` supports `system`, `user`, `assistant`, `tool`, and `developer` roles, multipart content, tool calls, ids, timestamps, and metadata. It converts to and from the OpenAI wire shape used by Cortex.

```python
from magent import Message

messages = [
    Message.system("Answer from evidence."),
    Message.user("What did we deploy?"),
]
result = agent.run(messages=messages)
```

## Short-term window

`ConversationWindow` tracks messages with a token estimator and configured maximum. Trim strategies are `middle`, `oldest`, and `newest`. System messages and the newest useful turns are protected according to policy. Optional summarization collapses older turns before dropping them.

The context assembler budgets system text, instructions, tool schemas, retrieved context, memory, and history independently. Slack goes to history, but required system/history sections are never silently omitted.

## Persistent memory

`MemoryStore` is a stdlib SQLite store with a deterministic fallback embedder:

```python
from magent import MemoryStore

store = MemoryStore(path=".magent/memory.db")
entry = store.add(
    "Production runs in ap-south-1",
    kind="fact",
    tags=["deployment", "production"],
    importance=0.9,
    source="runbook",
)

matches = store.search("production region", limit=5, semantic=True)
for match in matches:
    print(match.score, match.content)
```

Memory records include id, content, kind, tags, importance, source, arbitrary metadata, create/update/access timestamps, access count, optional embedding, and transient score.

Supported operations include add/add-many, get/update/touch, recent/list/by-kind, search/similar search, delete/delete-matching/delete-old, tags/entities, compact, backup, JSONL import/export, embedding reindex, clear, and statistics.

## Ranking

Hybrid ranking combines:

- BM25 lexical relevance;
- non-negative cosine similarity from embeddings;
- importance;
- recency/access decay;
- tag/entity evidence.

The zero-dependency `HashingEmbedder` maps token n-grams into a fixed vector. It is deterministic and useful for local semantic ranking without downloading a model. `BackendEmbedder` uses Cortex's `/v1/embeddings`; `CachingEmbedder` avoids repeated calls.

Semantic noise is deliberately downweighted rather than treated as equal to lexical evidence. Scores are clamped and reported, not hidden.

## Deduplication

By default, adding the same fingerprint updates/returns the existing memory instead of growing duplicates. Set `dedupe=False` only when duplicate observations are meaningful.

## Entities

Entity extraction recognizes common structured values such as emails, URLs, IP addresses, dates, paths, and identifiers. Entity tracking can be disabled in configuration.

## CLI

```bash
magent memory add 'Production runs in ap-south-1' \
  --kind fact --tag deployment --tag production --importance 0.9
magent memory search 'production region'
magent memory list --limit 20
magent memory stats
magent memory export ./memory.jsonl
magent memory compact
```

Run `magent memory --help` for every action and retention filter.

## Agent integration

When recall is enabled, the agent searches memory before constructing the model prompt and emits `memory.read`. When outcome persistence is enabled, it writes a bounded summary after the run and emits `memory.write`. Guardrail redaction happens before persistence.

Per-run controls:

```python
result = agent.run(
    "Continue the deployment task",
    recall_memories=True,
    remember_outcome=True,
)
```

## Sessions

Sessions are append-friendly JSONL transcripts with metadata. `SessionStore` supports open, list/query, delete, pruning, export, and repair of a corrupt tail. Byte limits apply from the first write and retain the newest entry when shrinking.

```bash
magent sessions list
magent sessions show SESSION_ID
magent sessions resume SESSION_ID
magent sessions export SESSION_ID transcript.json
magent sessions delete SESSION_ID
```

Session secret redaction is enabled by default. Session storage is not a replacement for encrypting a disk that holds sensitive work.

## Deletion and backup

Long-term memory and sessions are local files. Back up or delete them with the store/CLI rather than assuming a model “forgot.” `clear()` is destructive and approval policy should protect it in interactive operation.
