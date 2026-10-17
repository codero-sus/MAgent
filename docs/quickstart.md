# Quickstart

This guide goes from a checkout to a real MAgent run through Cortex LLMHoster. It also shows the offline path used by CI and development.

## 1. Check Python

MAgent supports CPython 3.11 and newer:

```bash
python3 --version
```

No mandatory package is installed at runtime. Packaging uses Hatchling, but running from a checkout needs only the standard library.

## 2. Install MAgent

```bash
git clone https://github.com/codero-sus/MAgent.git
cd MAgent
python3 -m pip install -e . --no-deps
magent version
```

Without installation:

```bash
PYTHONPATH=src python3 -m magent --help
```

The product is **MAgent**; `magent` is the executable and import package.

## 3. Start Cortex LLMHoster

Follow Cortex's own repository instructions. Its defaults, which MAgent uses, are:

- root URL: `http://127.0.0.1:8624`;
- liveness: `GET /health`;
- readiness: `GET /ready`;
- model list: `GET /v1/models`;
- inference: `POST /v1/chat/completions`;
- control plane: `/admin/api`.

Check it directly:

```bash
curl http://127.0.0.1:8624/health
curl http://127.0.0.1:8624/ready
curl http://127.0.0.1:8624/v1/models
```

If authentication is enabled:

```bash
export MAGENT_CORTEX_API_KEY='replace-me'
```

Set a non-default URL when needed:

```bash
export MAGENT_CORTEX_URL='http://127.0.0.1:8624'
```

`MAGENT_CORTEX_*` variables take precedence over Cortex's generic variables.

## 4. Diagnose the connection

```bash
magent doctor
magent cortex status
magent cortex models
```

`doctor` checks the interpreter, package, configuration, backend, Cortex reachability, model list, tool registry, sandbox, workspace, memory, sessions, plugins, and improvement manifest. A Cortex connection failure is expected when deliberately working offline.

## 5. Run a task

```bash
magent run 'Read README.md and summarize the project in five bullets'
```

Useful options:

```bash
magent run --model MODEL_ID --persona coder --workspace . 'Run the tests and explain failures'
magent run --no-tools 'Answer without tools'
magent run --dry-run 'Show which operations would be attempted'
magent run -v 'Explain the repository'
```

Interactive mode:

```bash
magent chat
```

Inside chat, `/help`, `/model`, `/persona`, `/tools`, `/stats`, `/clear`, `/verbose`, and `/quit` control the session.

## 6. Run offline

The echo backend opens no socket and returns deterministic text:

```bash
magent --offline run 'hello'
# echo:hello
```

Python:

```python
from magent import Agent, EchoBackend

agent = Agent(backend=EchoBackend(), workspace=".")
result = agent.run("hello")
assert result.output == "echo:hello"
```

The testing package also has a scripted backend that asks for a tool, consumes its result, and answers. That lets the complete agent loop run in CI without a model.

## 7. Inspect tools

```bash
magent tools categories
magent tools list --category files
magent tools show read_file
magent tools schema calculate
```

MAgent loads 113 built-in tools in 16 categories. A tool declares JSON Schema, risk, side effects, timeout, output limit, tags, and approval requirements.

## 8. Add configuration

```bash
magent config init --file ./magent.toml
magent config validate
magent config show
```

Then edit `magent.toml`:

```toml
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
```

Environment and CLI overrides remain higher precedence:

```bash
magent --set agent.max_iterations=4 run 'Do one bounded task'
```

## 9. Start the web console

```bash
magent serve
```

Open `http://127.0.0.1:8700/`. The server binds only to loopback by default. A public bind is refused unless an API key is supplied:

```bash
export MAGENT_SERVER_KEY="$(magent serve --generate-key -q)"
magent serve --host 0.0.0.0 --server-key "$MAGENT_SERVER_KEY"
```

## 10. Verify the checkout

```bash
make test
make smoke
make smoke-tools
make verify
# or all four:
make check
```

All checks run without external network access or a live Cortex process.
