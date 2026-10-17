# Configuration

MAgent configuration is a typed set of dataclass sections loaded from TOML, profiles, environment variables, and explicit overrides. Invalid values fail during load with `ConfigValidationError`, before a model or tool request.

## Precedence

Lowest to highest:

1. built-in defaults;
2. system config;
3. user config;
4. project `./magent.toml`;
5. the file named by `MAGENT_CONFIG`;
6. `MAGENT_*` environment variables;
7. CLI/API overrides.

The contributing file paths are retained in `Config.sources` for diagnostics.

## CLI

```bash
magent config paths
magent config init --file ./magent.toml
magent config show
magent config get agent.max_iterations
magent config set agent.max_iterations 8
magent config dump
magent config env
magent config validate
```

An explicit file:

```bash
magent --config ./production.toml run 'health check'
```

One-time overrides:

```bash
magent --set agent.max_iterations=4 \
       --set tools.allow_network=false \
       run 'bounded task'
```

## Sections

| Section | Responsibility |
|---|---|
| `server` | Cortex root, API prefix, key, HTTP timeout/pool/retries |
| `model` | backend, model ids, router, context size, fallback |
| `generation` | temperature, top-p, token limits, stop, seed, streaming |
| `agent` | loop, iterations, planning/reflection, persona, tool bounds |
| `memory` | window, persistence, vector search, retrieval |
| `tools` | workspace, paths, commands, network, approval, plugins |
| `guardrails` | injection/PII, budgets, rate limit, audit |
| `cache` | response cache location, size, TTL |
| `output` | format, color, verbosity, width, streaming display |
| `session` | directory, autosave, retention, transcript limits |
| `observability` | logs, traces, metrics, redaction, rotation |
| `http` | embedded server bind, key, CORS, body/worker/time bounds |
| `eval` | evaluation dataset and threshold defaults |

Unknown top-level tables are preserved in `Config.extra`, allowing plugins to keep configuration in the same TOML file. Unknown keys inside known sections are preserved under `extra.unknown_keys` rather than applied to core policy.

## Example

```toml
profile = "default"

[server]
base_url = "http://127.0.0.1:8624"
api_key = "env:MAGENT_CORTEX_API_KEY"
timeout = 120.0
max_retries = 3

[model]
backend = "auto"
model = ""
embedding_model = ""
router_strategy = "priority"
fallback_to_echo = true
context_size = 8192
max_output_tokens = 2048

[generation]
temperature = 0.2
stream = true

[agent]
loop_mode = "react"
max_iterations = 12
max_tool_calls_per_step = 8
persona = "assistant"
reflection = false
planning = false

[tools]
workspace = "."
approval_mode = "dangerous"
allow_network = false
autoload_plugins = true

[memory]
enabled = true
persistence = true
retrieval_mode = "hybrid"
retrieval_top_k = 5

[guardrails]
enabled = true
injection_detection = true
injection_action = "warn"
pii_redaction = true
pii_action = "redact"

[http]
host = "127.0.0.1"
port = 8700
api_key = "env:MAGENT_SERVER_KEY"
cors_origins = ["*"]
serve_ui = true
```

## Profiles

Profiles deep-merge into the base:

```toml
[profiles.fast.agent]
loop_mode = "single"
max_iterations = 3

[profiles.research.agent]
loop_mode = "plan"
max_iterations = 20

[profiles.research.tools]
allow_network = true
allow_domains = ["docs.python.org"]
```

```bash
magent --profile research run 'Compare the documented APIs'
```

A requested profile that is not defined is an error (except the implicit `default`).

## Environment variables

Examples:

```bash
export MAGENT_BACKEND=cortex
export MAGENT_CORTEX_URL=http://127.0.0.1:8624
export MAGENT_CORTEX_API_KEY='...'
export MAGENT_MODEL='model-id'
export MAGENT_MAX_ITERATIONS=8
export MAGENT_LOOP_MODE=react
export MAGENT_WORKSPACE="$PWD"
export MAGENT_APPROVAL_MODE=dangerous
export MAGENT_ALLOW_NETWORK=false
export MAGENT_MEMORY_ENABLED=true
export MAGENT_HTTP_HOST=127.0.0.1
export MAGENT_HTTP_PORT=8700
```

Booleans accept common true/false spellings; integer, float, and list variables are coerced to the declared field type. Empty values are ignored.

## Secrets

String values can reference:

```toml
api_key = "env:MAGENT_CORTEX_API_KEY"
# or
api_key = "file:/run/secrets/cortex_key"
```

Resolution is read-only. MAgent never writes a resolved secret back to configuration. `Config.to_dict()` redacts keys unless `redact=False` is deliberately requested in Python. The HTTP configuration route never permits reveal mode.

## Derived paths

When left empty, storage paths are derived under XDG directories:

- sessions, memory database, logs/audit/traces, and plugins under `XDG_DATA_HOME`;
- configuration under `XDG_CONFIG_HOME`;
- response cache under `XDG_CACHE_HOME`.

Existing explicit paths are not overwritten. Tests redirect all XDG directories into temporary homes.

## Python

```python
from magent import load_config

config = load_config(
    path="magent.toml",
    profile="fast",
    overrides={"agent.max_iterations": 2},
)
print(config.agent.loop_mode)
print(config.get("tools.workspace"))

copy = config.clone()
copy.agent.max_iterations = 99
assert config.agent.max_iterations != copy.agent.max_iterations
```

`clone()` is a deep independent copy, including mutable section lists and maps.
