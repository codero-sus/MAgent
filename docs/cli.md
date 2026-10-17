# CLI reference

The `magent` and `mr-agent` console scripts both call `magent.cli.main:main`. The product is **MAgent**; command names follow normal lowercase shell conventions.

## Global syntax

```text
magent [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS] [ACTION [ACTION OPTIONS]]
```

Global flags may appear before the command, after the command, or after a nested action:

```bash
magent -f json tools categories
magent tools -f json categories
magent tools categories -f json
```

Group flags also work around nested actions:

```bash
magent plugins --dir ./plugins list
magent plugins list --dir ./plugins
```

## Global options

| Flag | Meaning |
|---|---|
| `-c, --config PATH` | explicit TOML configuration |
| `-p, --profile NAME` | profile to deep-merge |
| `-s, --set KEY=VALUE` | dotted override, repeatable |
| `--backend NAME` | `auto`, `cortex`, `echo`, or `router` |
| `--model ID` | model id |
| `--base-url URL` | Cortex root URL |
| `--api-key KEY` | Cortex key; environment preferred |
| `-f, --format FMT` | text, JSON, JSONL, Markdown, etc. per command |
| `--color / --no-color` | force terminal color policy |
| `-q, --quiet` | answer/result only |
| `-v, --verbose` | steps, tools, timings |
| `--debug` | debug logs and events |
| `--width COLS` | terminal wrapping width |
| `-w, --workspace DIR` | allowed working directory |
| `--offline` | force local EchoBackend |
| `--no-tools` | disable the tool catalogue for the run |
| `--timeout SECONDS` | request timeout |
| `-y, --yes` | non-interactive approval |
| `--dry-run` | validate/report without tool side effects |

Prefer `MAGENT_CORTEX_API_KEY` over `--api-key`: command arguments may be visible in process lists and shell history.

## Commands

### `run`

Execute one task and print the answer.

```bash
magent run --persona coder --max-steps 8 'Fix the test'
```

Run `magent run --help` for context/memory/task blocks, file inclusion, structured schema, session, tool selection, loop mode, and streaming controls.

### `chat`

Interactive conversation. Slash commands include `/help`, `/clear`, `/model`, `/persona`, `/tools`, `/stats`, `/verbose`, and `/quit`.

### `tools`

Actions: `list`, `categories`, `show`, `search`, `schema`, `risks`.

```bash
magent tools list --category files --max-risk read
magent tools show write_file
```

### `config`

Actions: `show`, `get`, `set`, `paths`, `validate`, `init`, `dump`, `env`.

```bash
magent config init --file ./magent.toml
magent config set agent.max_iterations 8
```

### `doctor`

Checks runtime and local services. `--offline` selects EchoBackend but still reports Cortex reachability separately. A failed Cortex check is actionable, not hidden by fallback.

### `cortex`

Actions: `status`, `models`, `start`, `stop`, `restart`, `stop-all`, `test`, `metrics`, `state`, `runtime`, `hardware`, `files`, `upload`, `config`, `update`, `auth`, `cookbook`.

The command operates Cortex's actual `/admin/api` surface.

### `memory`

Add/list/search/show/update/delete memories; inspect tags/entities/stats; import/export/compact/reindex/clear according to `--help`.

### `rag`

Ingest/search/stats/sources/chunks/save/load/clear index data. `--index`, `--pattern`, and retrieval `--mode` can be supplied around nested actions.

### `sessions`

List/show/resume/export/delete/prune session transcripts.

### `eval`

Actions: `run`, `list`, `show`, `metrics`, `tags`, `init`, `convert`.

```bash
magent eval run --dataset builtin --fail-below 0.5
magent eval init ./evaluation.json
```

### `serve`

Start the API/console, print routes, generate a server key, or dry-run resolved settings.

```bash
magent serve --routes
magent serve --host 127.0.0.1 --port 8700
```

The server credential is `--server-key`; global `--api-key` is the Cortex credential.

### `plugins`

Actions: `list`, `show`, `inspect`, `scaffold`, `permissions`, `hooks`, `load`, `manifest`, `stats`.

Group policy flags include `--dir`, `--allow-sensitive`, `--deny` (plugin name), `--only`, `--no-checksum`, `--strict`, and `--loaded`.

### `improvements`

List or verify generated entries:

```bash
magent improvements --verify
magent improvements --category tool
magent improvements --regenerate
```

### `version`

Print semantic version, codename, and channel.

## Output

`Printer` centralizes terminal/JSON output. Machine formats go to stdout; diagnostics and errors use stderr where appropriate. Color is disabled when output is not a TTY unless forced. Quiet mode still prints the requested answer.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | success/completed |
| 1 | runtime error |
| 2 | usage/config parse error |
| 3 | max iterations |
| 4 | budget exceeded |
| 5 | guardrail stop |
| 6 | needs input |
| 7 | tool limit |
| 8 | evaluation threshold failure |
| 78 | configuration failure in applicable command paths |
| 130 | interrupt/cancel |

## Shell completion

The CLI uses only `argparse` and does not install shell-specific completion scripts. The complete live grammar is always available through nested `--help`.
