# Contributing

## Layout

```
src/magent/
├── agent.py            the loop, plans, reflection, streaming
├── config.py            layering, profiles, env, redaction
├── errors.py            63 typed errors with codes and exit codes
├── paths.py             XDG layout, MAGENT_HOME, repo discovery
├── tools/               registry, schema inference, sandbox, approval, 113 builtins
├── llm/                 backends, router, cache, retry, tokens, usage, Cortex admin
├── memory/              entries, embeddings, vector index, ranking, store, window
├── rag/                 chunkers, index, re-ranking, citations, retriever
├── guardrails/          injection, PII, policy, audit, manager
├── context/             blocks, compression, workspace scanning, builder
├── prompts/             templates, personas, system prompt builder
├── agents/              presets, teams, handoffs
├── skills/              skill parsing and the built-in library
├── plugins/             manifests, loader, hooks
├── session/             records, store, transcripts, manager
├── eval/                datasets, metrics, runner, reports
├── observability/       spans, traces, metrics, wiring
├── concurrency/         pools, rate limits, debouncing
├── output/              formats, markdown, rendering, sinks
├── server/              HTTP API, router, web UI
├── cli/                 parser, context, REPL, 26 commands
├── testing/             fake Cortex server, fixtures, scripted backends
└── utils/               13 dependency-free helper modules
```

## Conventions

* **stdlib only** in `src/magent`. No third-party imports, ever.
* **Sync first.** Threads, not asyncio, for parallelism.
* **Type hints everywhere**, `from __future__ import annotations`, and ruff at
  100 columns.
* **Every public symbol has a docstring.** The improvements manifest requires it,
  and `make verify` fails without one.
* **Report, do not raise**, at the edges: a missing session, an unreachable
  model, a plugin that will not import and a broken summariser are all data the
  caller can inspect. Raise only when the caller asked for a contract
  (`StructuredOutputError`) or cannot continue.
* **No dead code.** If a branch cannot be reached, delete it.

## Adding a tool

1. Put it in the module that matches its category under `src/magent/tools/builtin/`.
2. Decorate it with `@tool(...)`, annotate the arguments, document them in a
   Google-style docstring.
3. If it touches paths, commands or URLs, take `context: ToolContext` and let the
   sandbox check first.
4. Add it to the module's `TOOLS` tuple.
5. Add a test in `tests/test_tools.py` and run `make smoke-tools` - the harness
   fails if a registered tool is neither exercised nor explicitly skipped.
6. Regenerate the manifest: `make improvements`.

## Adding a CLI command

Add a `Command(...)` to the matching `COMMANDS` tuple in
`src/magent/cli/commands/`. The parser, `--help`, `magent commands` and shell
completions follow from it. Give it examples - `magent <cmd> --help` shows them.

## Checklist before a pull request

```bash
make fmt          # ruff format
make lint         # ruff check
make type         # mypy on src
make test         # 487 tests
make smoke        # end-to-end, offline
make smoke-tools  # every registered tool
make verify       # the improvements manifest
```

## Working with Cortex LLMHoster

MAgent talks to Cortex over HTTP only. Cortex's licence forbids modification and
redistribution, so never vendor, patch or copy its source into this repository -
and never make MAgent depend on an undocumented endpoint.
