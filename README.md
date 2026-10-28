# MAgent

**Mr. Agent, Powerful.** A local-first Python agent framework for
[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster): it plans,
calls tools, remembers, retrieves, guards itself, automates full desktop
workflows, works in teams, and explains what it did — with **zero mandatory
dependencies** beyond the Python standard library.

> The product is **MAgent**—never “Magent.” Lowercase `magent` is reserved for
> the Python import package, executable command, paths, and environment namespace.

```bash
git clone https://github.com/codero-sus/MAgent && cd MAgent
python -m magent doctor
python -m magent ask "what can you do?"
python -m magent serve          # REST API + web UI
```

---

## Why it exists

Running a capable agent locally should not require a cloud account, a dozen
libraries or a vector database. MAgent is a complete agent framework in one
installable package: 143 tools behind real policy and approval controls, hybrid
memory, retrieval with citations you can verify, full desktop automation,
guardrails that fire, multi-agent orchestration, sessions, evaluation, tracing,
plugins, a 27-command CLI and an HTTP API — all talking to the models Cortex
LLMHoster already serves on your machine.

| | |
| --- | --- |
| **No dependencies** | `dependencies = []` in `pyproject.toml`. Stdlib only, by design and by test. |
| **Local first** | Cortex LLMHoster over its documented `/v1` and `/admin/api` endpoints; an echo backend keeps everything working with no models running. |
| **Always available** | Draggable topmost launcher, native task/queue/chat panel, durable SQLite work, eight selectable logo pairs, and no cloud control plane. |
| **Honest failure** | Typed errors with codes and exit codes; headless and partially supported desktops report exact capabilities instead of crashing. |
| **Verifiable** | 514 tests, plus an auto-generated manifest of **3,539 improvements** where every entry is re-checked against the code by `make verify`. |

## Install

```bash
pip install -e .              # optional — `python -m magent` works from a checkout
pip install 'MAgent[desktop]'  # optional cross-platform desktop backend
magent --help
make test                     # 514 tests, offline, ~35 seconds
make verify                   # re-check the improvements manifest
```

Requires Python 3.11+.

## Quick start

```python
from magent import Agent, build_backend

agent = Agent(backend=build_backend())
result = agent.run("read README.md and tell me what this project is")
print(result.output)
print(f"{result.usage.total_tokens} tokens, {result.duration_ms:.0f} ms, "
      f"{result.tool_call_count} tool calls")
```

```bash
magent run "summarise docs/" --stream           # watch it work
magent chat                                     # interactive session
magent plan "add pagination to the sessions API"
magent rag ingest docs/ && magent rag search --query "how do guardrails work"
magent eval sample --output evals/smoke.json    # evaluate an agent
magent desktop                                  # floating logo + task/queue/chat panel
magent serve                                    # http://127.0.0.1:8765
```

## What is in the box

### Agent loop

Four loop modes — `react`, `plan`, `reflect`, `single` — with streaming,
structured output (`structured=<schema>`), cancellation, budgets, retries,
approval gates and a `RunResult` that records every step, tool call, token and
event. `result.events` includes the terminal event, so a replay sees a run that
finished.

### Tools

143 tools across 17 categories (`files`, `text`, `data`, `git`, `shell`,
`python`, `web`, `cortex`, `desktop`, `memory`, `media`, `math`, `datetime`,
`system`, `search`, `output`, `meta`). Schemas are inferred from type hints and
docstrings. Files and commands use the workspace sandbox; desktop-changing
actions use risk policy, explicit approval, capability gates, limits, fail-safe
checks, and the audit trail.

```python
from magent.tools import tool

@tool(name="count_lines", description="Count the lines in a file", category="files")
def count_lines(path: str, skip_blank: bool = False) -> dict:
    """Count lines, optionally ignoring blank ones.

    Args:
        path: file to read, inside the workspace
        skip_blank: ignore lines that are empty after stripping
    """
    lines = open(path, encoding="utf-8").read().splitlines()
    return {"lines": len([line for line in lines if line.strip()]) if skip_blank else len(lines)}
```

### Always-available desktop agent

```bash
magent desktop status                         # exact live capabilities
magent desktop                                # floating always-on-top MAgent logo
magent desktop queue add "Prepare my report" # persistent prioritized work
magent desktop logos                          # eight icon + lockup pairs
magent desktop autostart install              # optional start-at-login
```

Clicking the draggable logo opens a native task/list, queue, agent-chat, status,
and settings panel. The SQLite queue survives restarts and uses queued, paused,
running, completed, failed, and cancelled states. Mouse, keyboard, clipboard, screen,
window, application, wait, and composed JSON-workflow operations are available
through 30 tools; desktop-changing calls require human approval. A top-left
pointer fail-safe and panel emergency stop provide immediate interruption.

The core stays dependency-free. `MAgent[desktop]` adds PyAutoGUI/Pillow for the
most consistent backend; native Windows, macOS, X11, and Wayland helpers are
probed individually. Headless installations remain fully importable and can use
the deterministic recording driver and workflow dry runs. See
[Desktop automation](docs/desktop.md).

### Backends

Cortex LLMHoster via `MAGENT_CORTEX_URL` (default `http://127.0.0.1:8624`),
streaming, embeddings, reranking, capability checks, model start/stop/restart
through the admin API, a `ModelRouter` with six strategies and per-route circuit
breakers, disk caching, token accounting and price books.

### Memory and retrieval

Hybrid BM25 + vector long-term memory with entity extraction, decay,
de-duplication and JSONL export; a token-aware short-term window with batched
rolling summaries; five chunking strategies; hybrid retrieval with MMR,
cross-encoder re-ranking and citations that `verify_citations` can check.

### Guardrails

Nine injection families, twelve PII detectors, policy rules, budgets, rate
limits, approval gates and an append-only audit log. Tool results are scanned
separately, and credentials the scanner recognises are redacted even in `warn`
mode.

### Teams

Five orchestration modes — `sequential`, `round_robin`, `supervisor`, `debate`,
`vote` — with structured handoffs between members, per-member token accounting
and full transcripts.

### Interface

```bash
magent run | ask | chat | plan | reflect     # the agent
magent tools | presets | skills | plugins    # what it can do
magent memory | rag | sessions | trace       # what it knows and did
magent models | cortex | config | doctor     # the machine it runs on
magent desktop                               # floating desktop agent + queue
magent eval | bench | improvements           # how good it is
magent serve | init | commands | completion  # operational
```

Every command accepts `--format text|json|jsonl|yaml|csv|md`, `--quiet`,
`--verbose`, `--no-color` and `--output-file`, before or after the subcommand.

### HTTP API and web UI

`magent serve` exposes the same objects over REST, plus a single-page UI at `/`
that uses relative URLs and works behind a proxy. Endpoints cover health, info,
models, tools, chat (JSON and server-sent events), sessions, memory, metrics
(Prometheus), RAG, desktop capabilities, and durable desktop-task lifecycle.
Optional bearer-token auth; configurable CORS. Direct desktop-changing tool
calls remain approval-gated.

### Extensibility

Twelve presets, ten personas, 23 prompt templates, twelve built-in skills,
plugins with manifests, hooks and validation, and a context builder that
compresses everything into your token budget in priority order.

## Architecture

```
src/magent/
├── agent.py core loop + prompts/ + context/     the thinking
├── llm/ http/ secrets.py                        the wiring to models
├── tools/ (113 builtins) guardrails/            what it may do
├── memory/ rag/                                 what it knows
├── desktop/                                     local automation + floating UI
├── agents/ (presets, teams) skills/ plugins/    how it is extended
├── session/ eval/ observability/ concurrency/   how it is operated
├── output/ server/ cli/                         how you talk to it
└── utils/ (13 modules) errors.py types.py       the foundation
```

* [docs/index.md](docs/index.md) — overview and page index
* [docs/quickstart.md](docs/quickstart.md) — five minutes to a working agent
* [docs/desktop.md](docs/desktop.md) — launcher, queue, tools, workflows, API, and safety
* [docs/contributing.md](docs/contributing.md) — layout, conventions, adding tools

## Testing and evaluation

```bash
make test            # 514 tests: unit, integration, CLI, HTTP, doctests
make smoke           # one end-to-end offline run of the whole stack
make smoke-tools     # dispatch every registered tool
make eval            # run the sample dataset
make verify          # verify every improvement reference
make check           # lint + type + test + smoke + verify
```

Evaluation uses deterministic, model-free metrics. A case only passes when its
hard assertions hold — a missing required substring is not averaged away.

## The improvements manifest

`IMPROVEMENTS.md` and `improvements.json` are generated from the source tree:

```bash
make improvements    # regenerate (the manifest is generated, not written by hand)
make count           # print the total
make verify          # re-resolve every reference against the tree
```

Each entry names a file and a symbol: a documented function, class or method; a
registered tool; a CLI command; a skill, persona, template or preset; a test; a
documentation section; a Makefile target. The verifier fails the build if a
reference no longer resolves, so the number cannot drift away from the code.

## Cortex LLMHoster licence

This integration is intended for **personal, non-commercial use**. MAgent talks
to Cortex purely as an external **HTTP client**. Cortex is distributed under its
own separate personal, non-commercial, no-modification licence; those terms
permit running an unmodified official copy for personal, non-commercial use and
forbid modification and redistribution. No Cortex source is vendored, patched
or copied into this repository. Cortex's licence remains separate from MAgent's.

## Licence

MAgent is licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
It permits use, modification and distribution for permitted non-commercial
purposes, including qualifying personal uses, but does not grant commercial-use
rights. See [LICENSE](LICENSE) for the complete terms.
