# Contributing

Contributions should preserve MAgent's zero-mandatory-dependency design, Cortex HTTP boundary, safety defaults, naming, tests, and verified-improvement contract.

## Naming

The product is **MAgent**. Use exactly that spelling in prose, terminal banners, server headers, docs, and user-facing errors.

Use lowercase only where required by technical convention:

- `import magent`;
- the `magent` command;
- `src/magent` and user data/config paths;
- `MAGENT_*` environment variables (uppercase namespace);
- distribution-name normalization by package indexes.

Never write “Magent.”

## Setup

```bash
git clone https://github.com/codero-sus/MAgent.git
cd MAgent
python3 -m pip install -e . --no-deps
make test
```

Optional developer tools:

```bash
python3 -m pip install ruff mypy build
```

They are not runtime dependencies.

## Design rules

1. **Cortex stays external.** Do not copy, patch, vendor, or redistribute Cortex source. Integrate only through its public HTTP API.
2. **Stdlib baseline.** New mandatory behavior must work on stock Python 3.11+. Optional accelerators need graceful fallbacks.
3. **Sync first.** Use bounded threads for concurrency; preserve deterministic result ordering.
4. **Typed values.** Cross-layer data belongs in dataclasses/enums with JSON round trips.
5. **Explicit failures.** Return typed `ToolResult` failures or raise a specific `MAgentError`; never silently shrink a batch.
6. **Safe defaults.** Loopback binds, redaction, bounded bodies/output, approvals, and path checks must not regress.
7. **No guessed APIs.** Read the implementation/contract before calling a method. Add an integration test for wire behavior.
8. **Public docstrings.** Every public symbol requires a meaningful docstring; the improvement collector/verifier relies on it.
9. **No filler improvements.** Add implementation/tests/docs, regenerate the manifest, and let the count follow.

## Repository layout

```text
src/magent/          package
  llm/               Cortex, base backends, routing, cache, tokens, usage
  tools/             schema/registry/sandbox/approval + built-ins
  memory/            short- and long-term memory
  rag/               chunking/index/retrieval/reranking/citations
  guardrails/        policy checks, budgets, audit
  plugins/           manifests, hooks, loader
  server/            embedded HTTP server/routes/console
  cli/               argparse and command handlers
  testing/           deterministic backends and fake Cortex
tests/               stdlib unittest suite
docs/                documentation included in wheel
examples/            runnable examples
scripts/             smoke and improvement verification
```

## Tests

```bash
make test
make unit
make smoke
make smoke-tools
make verify
make check
```

Tests must isolate XDG homes and workspaces. Use `magent.testing.FakeCortexServer` for HTTP contract tests and `make_backend` for tool-loop tests. Use real loopback sockets when testing framing, CORS, authentication, or SSE.

When fixing a defect:

1. reproduce it with the narrowest failing test;
2. verify whether the assumption or implementation is wrong by reading the real API;
3. fix the implementation;
4. retain a regression assertion;
5. run the relevant module and then `make check`.

## Tools

Every built-in tool module exports `TOOLS` and `load(registry)`. Tool handlers:

- declare complete type hints and Google-style argument docs;
- use `ToolContext` for workspace/policy/services;
- perform sandbox checks before side effects;
- declare risk and side-effect flags honestly;
- bound output and time;
- avoid global mutable state;
- remain testable offline.

## HTTP and streams

- Bind test servers to `127.0.0.1` and ephemeral ports.
- Correct HTTP/1.1 streams use chunked framing.
- Apply CORS and auth on error paths as well as success.
- Never send duplicate framing/response headers.
- Sanitize URLs before including them in errors.
- Close pools, streams, servers, and worker threads in `finally`.

## Configuration

Do not mutate shared section objects when deriving policy; `Config.clone()` exists for independent copies. Keep precedence and environment mappings documented. Unknown plugin tables belong in `Config.extra`.

## Generated manifest

Never hand-edit `IMPROVEMENTS.md`:

```bash
PYTHONPATH=src python3 scripts/count_improvements.py --write
PYTHONPATH=src python3 scripts/verify_improvements.py
```

Commit the regenerated file with the implementation.

## Documentation

Commands and signatures in docs should be executed or compared with `--help`/`inspect.signature`. Explain security limits directly. Include a runnable example for a major new surface.

## Pull request checklist

- [ ] Product spelling is MAgent in prose.
- [ ] No Cortex source is copied.
- [ ] No mandatory dependency was added without an architectural reason.
- [ ] Public symbols have docstrings/type hints.
- [ ] New behavior has tests; bug fixes have regressions.
- [ ] `make check` passes.
- [ ] `IMPROVEMENTS.md` is regenerated and verifies.
- [ ] README/docs/examples are updated where behavior is user-visible.
