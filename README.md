# MAgent

**Mr. Agent, Powerful.** A batteries-included, zero-mandatory-dependency Python agent framework and operator CLI built for [Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster).

> The product is **MAgent**. The lowercase word `magent` is used only for the Python import package, the executable command, configuration paths, and environment-variable namespace.

MAgent runs local models through Cortex's real HTTP API, gives them 113 schema-declared tools, keeps short- and long-term memory, retrieves cited context, applies guardrails and approval policy, records events and metrics, loads permission-declared plugins, evaluates datasets, and exposes a 31-route HTTP API with an embedded web console.

It has **3,200+ machine-verified improvements**—well above the promised 777. Every numbered row in [`IMPROVEMENTS.md`](IMPROVEMENTS.md) names a real file and symbol, and the verifier fails if any reference disappears.

## Highlights

- **Real Cortex integration** — `/health`, `/ready`, OpenAI-compatible `/v1`, streamed SSE tool calls, embeddings, and `/admin/api` model control.
- **Zero mandatory runtime dependencies** — the HTTP pool, SSE parser, TOML configuration, SQLite memory, BM25/vector retrieval, terminal UI, server, and test doubles use the Python 3.11+ standard library.
- **113 actual tools in 16 categories** — files, shell, Python, web, search, text, data, Git, memory, media, system inspection, Cortex administration, and more.
- **Four loop modes** — `react`, `plan`, `reflect`, and `single`, with bounded iterations and tool calls.
- **Safety by default** — loopback server bind, API key required for public binds, path roots, command/domain policy, risk-ranked approvals, prompt-injection checks, PII redaction, budgets, and append-only audit records.
- **Hybrid memory and RAG** — SQLite persistence, lexical + hashing-vector ranking, MMR diversity, file-aware chunkers, numbered citations, and grounding verification.
- **Observable and testable** — 27 typed event kinds, traces, Prometheus metrics, deterministic backends, and a real in-process fake of Cortex's HTTP contract.
- **Permission-declared plugins** — manifest validation, checksum pinning, sensitive-permission approval, declared tool/hook registration, lifecycle hooks, and static inspection before import.
- **One operator surface** — 14 CLI command groups, 31 HTTP routes, and a self-contained web console.

## Requirements

- CPython **3.11 or newer**
- [Cortex LLMHoster](https://github.com/codero-sus/Cortex_LLMHoster) for real local-model inference
- No cloud API key and no mandatory third-party Python package

MAgent integrates with Cortex as an external HTTP client. Cortex is not vendored, modified, or redistributed.

## Install

From this repository:

```bash
git clone https://github.com/codero-sus/MAgent.git
cd MAgent
python3 -m pip install -e . --no-deps
magent version
```

Or run directly from a checkout without installing:

```bash
PYTHONPATH=src python3 -m magent --help
```

The distribution is named **MAgent**. Package indexes normalize project names, so `pip install magent` and `pip install MAgent` resolve identically. Python imports and shell commands stay lowercase by convention:

```python
from magent import Agent
```

## Start Cortex

Cortex listens on `http://127.0.0.1:8624` by default. Start it using the instructions in its repository, then verify the actual endpoints MAgent uses:

```bash
curl http://127.0.0.1:8624/health
curl http://127.0.0.1:8624/ready
curl http://127.0.0.1:8624/v1/models
```

If Cortex authentication is enabled, set the key in the environment instead of putting it on a command line:

```bash
export MAGENT_CORTEX_API_KEY='your-cortex-key'
# CORTEX_API_KEY is also accepted; the MAgent-prefixed name wins.
```

Optional connection variables:

```bash
export MAGENT_CORTEX_URL='http://127.0.0.1:8624'
export MAGENT_CORTEX_MODEL='your-model-id'
export MAGENT_CORTEX_TIMEOUT='120'
```

Check and operate the hoster through MAgent:

```bash
magent cortex status
magent cortex models
magent cortex hardware
magent cortex metrics
magent cortex start YOUR_MODEL_ID
magent cortex test YOUR_MODEL_ID --prompt 'Reply with pong'
```

## First run

```bash
magent run 'Explain this repository in five bullets'
magent chat
```

Use a specific model, persona, and workspace:

```bash
magent run \
  --model YOUR_MODEL_ID \
  --persona coder \
  --workspace . \
  'Find the failing test, fix it, and show the exact verification command'
```

Use the deterministic local echo backend when Cortex is not running:

```bash
magent --offline run 'hello'
# echo:hello
```

Global flags work both before and after subcommands/actions:

```bash
magent -f json tools categories
magent tools categories -f json
magent plugins --dir ./plugins list
magent plugins list --dir ./plugins
```

## Python API

### Text run

```python
from magent import Agent, CortexBackend, CortexSettings, load_config

config = load_config()
backend = CortexBackend(CortexSettings.from_env())
agent = Agent(
    backend=backend,
    config=config.agent,
    settings=config,
    workspace=".",
)

result = agent.run("Summarize README.md and cite the path you read")
print(result.output)
print(result.stop_reason.value)
print(result.usage.total_tokens)
```

`Agent.run()` returns a `RunResult`, never a bare string. It records steps, messages, tool calls/results, usage, cost, stop reason, timings, metadata, and typed events. `agent.ask(prompt)` is the convenience method when only text is needed.

### Custom tool

```python
from magent import Agent, CortexBackend, CortexSettings, default_registry, tool

@tool(category="weather", risk="read", description="Read a local weather observation.")
def local_weather(city: str, unit: str = "celsius") -> dict[str, object]:
    """Return a demonstration observation.

    Args:
        city: city to report
        unit: celsius or fahrenheit
    """
    return {"city": city, "temperature": 24, "unit": unit, "source": "demo sensor"}

registry = default_registry()
registry.register(local_weather)
agent = Agent(
    backend=CortexBackend(CortexSettings.from_env()),
    registry=registry,
    workspace=".",
)
print(agent.ask("Use local_weather for Gurugram and report the observation."))
```

The decorator derives a JSON Schema from type hints and Google-style argument documentation. Tool handlers receive a `ToolContext` when they declare a `context` or `ctx` parameter.

### Structured output

```python
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "risk"],
    "additionalProperties": False,
}

result = agent.run("Assess this change", structured=schema)
print(result.structured)
```

MAgent asks Cortex for JSON, validates the object, and performs one schema-informed repair turn if necessary.

## Tools and approvals

Inspect the real catalogue:

```bash
magent tools categories
magent tools list --category files
magent tools show read_file
magent tools schema calculate read_file
magent tools search 'regular expression'
```

Risk order:

```text
safe < read < write < execute < destructive < network
```

Approval modes are `never`, `ask`, `always`, and `dangerous`. The default is `dangerous`: read-only work proceeds, while risky or irreversible operations require a decision. `--yes` is explicit non-interactive approval; `--dry-run` validates calls without executing handlers.

The sandbox is policy enforcement for well-behaved tools, not an operating-system security boundary against hostile Python code. Run untrusted plugins or generated code inside an OS/container sandbox as well.

## Configuration

Configuration precedence, lowest to highest:

1. built-in defaults;
2. system `magent.toml`;
3. user `magent.toml`;
4. project `./magent.toml`;
5. `$MAGENT_CONFIG`;
6. `MAGENT_*` environment variables;
7. CLI `--set section.key=value` overrides.

Create and inspect a complete file:

```bash
magent config init --file ./magent.toml
magent config paths
magent config show
magent config get agent.max_iterations
magent config validate
```

Minimal example:

```toml
profile = "default"

[server]
base_url = "http://127.0.0.1:8624"
api_key = "env:MAGENT_CORTEX_API_KEY"

[model]
backend = "cortex"
model = ""
fallback_to_echo = true

[agent]
loop_mode = "react"
max_iterations = 12
persona = "assistant"

[tools]
workspace = "."
approval_mode = "dangerous"
allow_network = false

[http]
host = "127.0.0.1"
port = 8700
api_key = "env:MAGENT_SERVER_KEY"
```

Secrets support `env:NAME` and `file:/path/to/value` references. Normal configuration serialization is redacted. The HTTP API refuses `?reveal=true` by design.

Named profiles deep-merge over the base:

```toml
[profiles.fast.agent]
max_iterations = 4
loop_mode = "single"

[profiles.research.tools]
allow_network = true
```

```bash
magent --profile fast run 'Give a quick answer'
```

## Memory and RAG

Long-term memory uses a local SQLite database and hybrid lexical/vector ranking:

```bash
magent memory add 'The deployment region is ap-south-1' --kind fact --tag operations
magent memory search 'deployment region'
magent memory stats
```

Ingest and query files:

```bash
magent rag ingest docs/ README.md
magent rag search 'How does plugin permission approval work?'
magent rag stats
```

MAgent has Markdown-, Python-, and text-aware chunkers; BM25 + hashing-vector hybrid scoring; optional backend embeddings; MMR diversity; source filters; numbered citations; and citation verification. See [`docs/rag.md`](docs/rag.md).

## Plugins

Create, inspect, and load a real plugin:

```bash
magent plugins scaffold ./plugins --name weather
magent plugins inspect ./plugins/weather
magent plugins list --dir ./plugins
magent plugins load --dir ./plugins
```

A `plugin.json` declares its entrypoint, tools, hooks, and permissions. Sensitive permissions—`filesystem.write`, `network`, `shell`, and `secrets.read`—are refused unless the operator approves them:

```bash
magent plugins load --dir ./plugins --allow-sensitive network
```

Checksums are optional but enforced when declared. Entrypoints cannot escape their plugin directory. Static `inspect` does not import plugin code. See [`docs/plugins.md`](docs/plugins.md).

## HTTP server and web console

Start the embedded server:

```bash
magent serve
# http://127.0.0.1:8700/
```

Useful modes:

```bash
magent serve --routes
magent serve --routes -f json
magent serve --port 0 --offline
magent serve --generate-key
magent serve --host 0.0.0.0 --server-key "$MAGENT_SERVER_KEY"
```

MAgent refuses a non-loopback bind without an API key. Authenticate with either header:

```bash
curl -H "Authorization: Bearer $MAGENT_SERVER_KEY" http://127.0.0.1:8700/api/tools
curl -H "X-API-Key: $MAGENT_SERVER_KEY" http://127.0.0.1:8700/api/tools
```

The 31 routes include health/readiness, configuration, tools/schema/dispatch, sync and async runs, SSE events, sessions, memory, RAG, guardrails, metrics, Cortex status/models, improvement totals, and server statistics. The console has Run, Tools, Memory, RAG, Sessions, Events, Metrics, and Settings tabs and is one embedded HTML file—no Node build or CDN.

## Evaluation

```bash
magent eval list --dataset builtin
magent eval run --dataset builtin
magent eval run --dataset ./my-eval.json --fail-below 0.80
magent eval init ./my-eval.json
magent eval metrics
```

Datasets can assert text, JSON, tool calls, stop reasons, latency, usage, and custom metrics. Exit code `8` indicates an evaluation threshold failure.

## CLI map

| Command | Purpose |
|---|---|
| `magent run` | execute one task |
| `magent chat` | interactive session with slash commands |
| `magent tools` | inspect tools, schemas, categories, and risk |
| `magent config` | initialize, inspect, validate, and edit configuration |
| `magent doctor` | diagnose Python, Cortex, models, tools, storage, and policy |
| `magent cortex` | inspect and operate Cortex LLMHoster |
| `magent memory` | add, find, export, compact, and remove memories |
| `magent rag` | ingest, search, inspect, save, and load indexes |
| `magent sessions` | list, show, resume, export, and delete sessions |
| `magent eval` | run and inspect evaluation datasets |
| `magent serve` | run the embedded API and web console |
| `magent plugins` | inspect, scaffold, policy-check, and load plugins |
| `magent improvements` | list, regenerate, and verify the manifest |
| `magent version` | print release metadata |

Run `magent COMMAND --help` for the exact action and flag surface.

## Verification and tests

Everything important can be checked without Cortex or the internet:

```bash
make test          # stdlib unittest suite
make smoke         # offline agent + fake-Cortex end-to-end scenario
make smoke-tools   # execute 83 safe tool cases
make verify        # resolve every IMPROVEMENTS.md reference
make check         # all of the above
```

Direct equivalents:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
PYTHONPATH=src python3 scripts/smoke.py
PYTHONPATH=src python3 scripts/smoke_tools.py
PYTHONPATH=src python3 scripts/verify_improvements.py
```

The tests use a real loopback HTTP server for both Cortex and MAgent routes, but no external network, GPU, model weights, or third-party testing package.

## 777+ real improvements

This repository does not satisfy the number with filler prose. [`scripts/count_improvements.py`](scripts/count_improvements.py) discovers real modules, public classes/functions/methods/properties/constants, error types, event kinds, tool implementations, CLI commands and flags, packaging files, scripts, tests, examples, and documentation. It writes the numbered manifest. [`scripts/verify_improvements.py`](scripts/verify_improvements.py) then resolves every reference against the current checkout and enforces the floor.

```bash
magent improvements --verify
python3 scripts/verify_improvements.py --json
python3 scripts/count_improvements.py --by-category
```

If a referenced symbol or file is deleted, verification exits non-zero.

## Architecture

```text
User / Python API / CLI / Web console
                  │
                  ▼
         Agent loop + prompt builder
     ┌────────────┼──────────────┐
     ▼            ▼              ▼
 Tool registry  Memory/RAG   Guardrails/budgets
     │            │              │
     └────────────┼──────────────┘
                  ▼
      LLMBackend / ModelRouter
                  │ HTTP + SSE
                  ▼
          Cortex LLMHoster
      /health /ready /v1 /admin/api

Cross-cutting: EventBus · sessions · traces · metrics · plugins · approvals
```

Backend selection happens in one factory: explicit backend → `MAGENT_BACKEND` → probe Cortex `/health` → offline `EchoBackend` when fallback is enabled.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Agents and loop modes](docs/agents.md)
- [Cortex LLMHoster integration](docs/cortex.md)
- [Tools, sandbox, and approvals](docs/tools.md)
- [Configuration](docs/configuration.md)
- [Memory](docs/memory-and-state.md)
- [RAG and citations](docs/rag.md)
- [Guardrails and security](docs/guardrails.md)
- [Plugins and hooks](docs/plugins.md)
- [HTTP server and console](docs/server.md)
- [CLI reference](docs/cli.md)
- [Python API](docs/api.md)
- [Testing](docs/testing.md)
- [Contributing](docs/contributing.md)

## License

MAgent is released under the [MIT License](LICENSE).

Cortex LLMHoster is a separate project with its own license. MAgent communicates with it only through its public HTTP API; Cortex source is not copied into this repository.
