# MAgent documentation

**MAgent — Mr. Agent, Powerful.** is a Python 3.11+ agent framework, operator CLI, and embedded HTTP service designed around [Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster).

MAgent is sync-first and has no mandatory third-party runtime dependencies. It includes its own stdlib HTTP transport and connection pool, SSE parser, tool engine, approval policy, SQLite memory, retrieval stack, event bus, metrics, plugin loader, evaluation runner, terminal UI, threaded HTTP server, and deterministic test doubles.

> The product name is **MAgent**. `magent` is the lowercase import package and command.

## Read this first

1. [Quickstart](quickstart.md) — install, connect Cortex, and run the first task.
2. [Cortex LLMHoster](cortex.md) — exact routes, authentication, streaming, and admin control.
3. [Agents](agents.md) — loop modes, `RunResult`, events, and structured output.
4. [Tools](tools.md) — schemas, the 113 built-ins, sandbox policy, and approvals.
5. [Configuration](configuration.md) — TOML, profiles, environment variables, and precedence.

## Build applications

- [Python API](api.md)
- [Memory and state](memory-and-state.md)
- [RAG and citations](rag.md)
- [Guardrails](guardrails.md)
- [Plugins and hooks](plugins.md)
- [HTTP server and web console](server.md)

## Operate and verify

- [CLI reference](cli.md)
- [Testing](testing.md)
- [Verified improvements](improvements.md)
- [Contributing](contributing.md)

## Architecture at a glance

```text
CLI / Python / HTTP / web console
               │
               ▼
       Agent + prompt builder
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  Tools     Memory/RAG  Guardrails
     └─────────┼─────────┘
               ▼
      LLMBackend / router
               │ HTTP/SSE
               ▼
       Cortex LLMHoster
```

Cross-cutting services are sessions, events, traces, metrics, usage/cost budgets, concurrency pools, evaluation, and plugins.

## What “built for Cortex” means

MAgent does not guess an OpenAI-like route and hope it works. The client implements Cortex's root liveness/readiness endpoints, its OpenAI-compatible inference routes under `/v1`, and its administrative control plane under `/admin/api`. The test suite exercises these over real loopback HTTP against a protocol-compatible fake hoster. Cortex remains an external process; its source is never vendored or modified.

## Guarantees

- A model-requested tool is schema-validated before execution.
- Every dispatch returns a `ToolResult`, including validation and handler failures.
- Agent loops have iteration, tool, token, time, and cost bounds.
- Public HTTP binds require an API key.
- Configuration and diagnostic output redact secrets by default.
- Plugin entrypoints cannot leave their directory, and sensitive permissions require explicit approval.
- Each row of `IMPROVEMENTS.md` resolves to a real path and symbol.

## Get help

```bash
magent --help
magent doctor
magent COMMAND --help
make help
```

For a completely offline check:

```bash
magent --offline run 'hello'
make check
```
