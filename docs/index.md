# MAgent

**Mr. Agent, Powerful.** A local-first Python agent framework for
[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster): it plans,
calls tools, remembers, retrieves, guards itself, automates desktop workflows,
works in teams and explains what it did - with **zero mandatory dependencies**
beyond the standard library.

## What you get

| Area | What is in the box |
| --- | --- |
| Agent loop | react, plan, reflect and single-shot modes, streaming, structured output |
| Tools | 143 built-in tools in 17 categories behind sandbox, policy, and approval controls |
| Desktop | draggable topmost launcher, native task/queue/chat panel, full input/screen/window automation, eight logo pairs |
| Backends | Cortex LLMHoster over its OpenAI-compatible `/v1` API, plus echo and scripted backends for tests |
| Memory | hybrid BM25 + vector long-term store, token-aware short-term window, rolling summaries |
| Retrieval | five chunking strategies, hybrid search, MMR re-ranking, citations you can verify |
| Safety | injection detection, PII redaction, policy rules, budgets, rate limits, audit log, approvals |
| Teams | sequential, round-robin, supervisor, debate and vote orchestration with handoffs |
| Sessions | JSON records, transcripts, search, export, pruning, redaction |
| Extensibility | presets, personas, prompt templates, skills, plugins with hooks |
| Interface | a 27-command CLI, floating desktop control, interactive REPL, REST API and web UI |
| Quality | evaluation datasets with deterministic metrics, tracing, metrics, 514 tests, 3,539 verified improvements |

## Install

```bash
git clone https://github.com/codero-sus/MAgent
cd MAgent
pip install -e .          # optional: the package runs from a checkout too
pip install 'MAgent[desktop]' # optional cross-platform desktop backend
magent doctor             # checks backend, sandbox, memory, sessions, plugins
```

## First run

```bash
magent ask "what does magent.config do?"          # one question, one answer
magent run "summarise docs/" --stream             # watch it work
magent chat                                       # interactive session
magent desktop                                    # floating local desktop agent
magent serve                                      # REST API + web UI
```

## Pages

* [quickstart.md](quickstart.md) - five minutes from clone to a working agent
* [agents.md](agents.md) - the loop, options, plans, reflection, streaming
* [tools.md](tools.md) - writing tools, schemas, the sandbox, approvals
* [desktop.md](desktop.md) - launcher, queue, desktop tools, workflows, API, safety
* [providers.md](providers.md) - Cortex LLMHoster, routing, caching, token budgets
* [memory-and-rag.md](memory-and-rag.md) - remembering things and citing sources
* [guardrails.md](guardrails.md) - injection, PII, policy, budgets, audit
* [teams.md](teams.md) - multi-agent orchestration and handoffs
* [cli.md](cli.md) - every command with examples
* [api.md](api.md) - the HTTP API and the web UI
* [plugins-and-skills.md](plugins-and-skills.md) - extending MAgent
* [testing.md](testing.md) - the test suite, evaluation and the improvements manifest
* [contributing.md](contributing.md) - layout, conventions, how to add a tool

## Licence note

MAgent is available under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
for permitted non-commercial purposes, including qualifying personal uses.
Commercial-use rights are not granted.

MAgent's Cortex integration is intended for personal, non-commercial use and
communicates only through Cortex's documented HTTP API. Cortex has its own
separate personal, non-commercial, no-modification licence, so MAgent never
vendors, patches or copies Cortex source. Each project remains subject to its
own licence.
