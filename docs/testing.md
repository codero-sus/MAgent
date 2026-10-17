# Testing

MAgent's suite uses the Python standard-library `unittest` runner. It does not require pytest, internet access, Cortex, model weights, a GPU, or writes to the developer's real configuration/data directories.

## Run the suite

```bash
make test
```

Equivalent:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
```

Fast core subset:

```bash
make unit
```

This selects files named `test_core*.py`.

## Isolation

`tests.support.MAgentTestCase` redirects:

- `XDG_DATA_HOME`;
- `XDG_CONFIG_HOME`;
- `XDG_CACHE_HOME`;
- `MAGENT_CONFIG`.

Temporary workspaces are removed after each test. Server/Cortex tests bind `127.0.0.1` on an ephemeral port and close the socket in teardown.

A test must never rely on or modify `~/.local/share/magent`, `~/.config/magent`, the user's active Cortex instance, or external network state.

## Deterministic backends

### Echo

```python
from magent.testing import offline_backend

backend = offline_backend()
```

Returns `echo:<last user text>` and supports deterministic streaming.

### Scripted

```python
from magent.testing import ScriptedBackend

backend = ScriptedBackend(script=["first answer", "second answer"])
```

Script entries may be strings, response dictionaries with tool calls, callables receiving `CompletionRequest`, or matcher/reply pairs.

### Tool script

```python
from magent.testing import make_backend

backend = make_backend(
    tool_name="calculate",
    tool_arguments={"expression": "6*7"},
    final_text="The answer is 42.",
)
```

Turn one asks for the tool; turn two answers after the result.

## Fake Cortex

```python
from magent import CortexBackend, CortexSettings
from magent.testing import start_fake_cortex

server = start_fake_cortex()
try:
    backend = CortexBackend(CortexSettings(
        base_url=server.url,
        require_auth=False,
        max_retries=0,
    ))
    assert backend.health()["ok"]
finally:
    server.stop()
```

This is not a monkey-patched client. It is a real `ThreadingHTTPServer` implementing the Cortex contract used by MAgent:

- `/health`, `/ready`, `/v1/models`;
- chat/completions, legacy completions, embeddings, responses;
- streamed SSE text and fragmented tool calls;
- Bearer authentication and Cortex error envelopes;
- state, metrics, runtime/hardware, and model control/test admin routes.

The fake records request counts, model counts, transcripts, admin calls, status codes, and latency. `fail_next(model, times)` injects controlled 503 responses.

## Sample corpus

```python
from magent.testing import build_sample_corpus, temp_workspace

corpus = build_sample_corpus(include_code=True)
with temp_workspace(corpus) as root:
    # ingest/search root
    ...
```

The corpus has known question/source answers for deterministic retrieval evaluation.

## Socket tests

`tests/test_server.py` starts MAgent's real HTTP server and calls it through `urllib.request`. It covers route dispatch, JSON failures, auth, CORS, static traversal, async polling, and chunked SSE. Do not replace these with direct handler calls; that would miss HTTP framing and header defects.

`tests/test_cortex.py` does the same for outbound Cortex integration. It has caught defects that unit-level payload tests cannot, such as non-2xx response handling and streamed argument fragments.

## Tool smoke suite

```bash
make smoke-tools
```

`scripts/smoke_tools.py` executes 83 safe tool scenarios in an isolated workspace. It skips cases that need external network, a live Cortex, a repository-mutating Git operation, or a separately configured memory store; targeted tests cover those surfaces.

## Application smoke test

```bash
make smoke
```

`scripts/smoke.py` verifies an offline agent tool loop, fake-Cortex HTTP and streaming, retrieval, plugin loading, and improvement verification as one scenario.

## Improvement verifier

```bash
make verify
python3 scripts/verify_improvements.py --json
```

The verifier is not a snapshot count. It resolves every numbered file/symbol reference and fails below 777.

## Writing a test

- Use `unittest.TestCase` or `MAgentTestCase`.
- Give the module, class, and test a docstring that says what contract is protected.
- Test public behavior, then add a narrow regression assertion for the bug.
- Use temporary paths and fakes.
- Assert one result per tool call and preserve call ids.
- For concurrency, wait on an event/condition with a timeout—do not sleep indefinitely.
- For sockets, always stop in `finally`/teardown.
- Never hide a failure with a broad `except Exception: pass`.

## Full verification

```bash
make check
```

This runs tests, manifest verification, application smoke, and tool smoke.
