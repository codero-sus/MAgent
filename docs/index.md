# MAgent

**Mr. Agent, Powerful.** A local-first Python agent framework for
[Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster): it plans,
calls tools, remembers, retrieves, guards itself, works in teams and explains
what it did - with **zero mandatory dependencies** beyond the standard library.

## What you get

| Area | What is in the box |
| --- | --- |
| Agent loop | react, plan, reflect and single-shot modes, streaming, structured output |
| Tools | 113 built-in tools in 16 categories behind a deny-by-default sandbox |
| Backends | Cortex LLMHoster over its OpenAI-compatible `/v1` API, plus echo and scripted backends for tests |
| Memory | hybrid BM25 + vector long-term store, token-aware short-term window, rolling summaries |
| Retrieval | five chunking strategies, hybrid search, MMR re-ranking, citations you can verify |
| Safety | injection detection, PII redaction, policy rules, budgets, rate limits, audit log, approvals |
| Teams | sequential, round-robin, supervisor, debate and vote orchestration with handoffs |
| Sessions | JSON records, transcripts, search, export, pruning, redaction |
| Extensibility | presets, personas, prompt templates, skills, plugins with hooks |
| Interface | a 26-command CLI, an interactive REPL, a REST API and a single-page web UI |
| Quality | evaluation datasets with deterministic metrics, tracing, metrics, 429 tests, 3109 verified improvements |

## Install

```bash
git clone https://github.com/codero-sus/MAgent
cd MAgent
pip install -e .          # optional: the package runs from a checkout too
magent doctor             # checks backend, sandbox, memory, sessions, plugins
```

## First run

```bash
magent ask "what does magent.config do?"          # one question, one answer
magent run "summarise docs/" --stream             # watch it work
magent chat                                       # interactive session
magent serve                                      # REST API + web UI
```

## Pages

* [quickstart.md](quickstart.md) - five minutes from clone to a working agent
* [agents.md](agents.md) - the loop, options, plans, reflection, streaming
* [tools.md](tools.md) - writing tools, schemas, the sandbox, approvals
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

MAgent talks to Cortex LLMHoster over HTTP. Cortex is licensed
`LicenseRef-Cortex-Personal-NonCommercial-NoModification-NoRedistribution`, so
MAgent never vendors, patches or copies its source - integration happens through
the documented API only.
