# Testing, evaluation and the improvements manifest

## The test suite

```bash
make test                     # unittest, offline, ~10 seconds
python -m unittest discover -s tests -t .
python -m unittest tests.test_tools -v
```

Everything runs against the echo backend and a temporary `MAGENT_HOME`, so the
suite never touches your real memory, sessions, cache or logs. Tests that need a
live Cortex host are skipped unless `MAGENT_TEST_CORTEX_URL` is set.

| Suite | Covers |
| --- | --- |
| `test_core.py` | configuration, secrets, logging, events, types |
| `test_llm.py` | tokens, usage, caching, retries, routing, the factory |
| `test_tools.py` | schema inference, registry, sandbox, approvals, builtin tools |
| `test_agent.py` | loop modes, tools, streaming, structured output, cancellation |
| `test_memory_rag.py` | memory store, ranking, embeddings, chunking, retrieval |
| `test_guardrails_context.py` | injection, PII, policy, context assembly |
| `test_agents_skills_plugins.py` | presets, teams, handoffs, skills, plugins |
| `test_session_output_eval.py` | sessions, transcripts, rendering, evaluation |
| `test_observability_concurrency.py` | tracing, metrics, pools, limiters |
| `test_server.py` | the HTTP API over a real socket |
| `test_cli.py` | every command through the real parser |
| `test_doctests.py` | the examples in the package docstrings |

## Evaluation

```bash
magent eval sample --output evals/smoke.json
magent eval --dataset evals/smoke.json --threshold 0.9 --repeats 3 --concurrency 4
magent eval --dataset evals/smoke.json --report-format junit --output report.xml
magent eval metrics
```

Datasets load from JSON, JSONL, Markdown tables or Python literals:

```python
{"id": "arith", "prompt": "What is 6*7?", "contains": ["42"], "tools": ["calculate"],
 "max_steps": 4, "tags": ["math"]}
```

Metrics are deterministic and model-free, and a case only passes when its
**hard** assertions hold - a missing required substring is not averaged away by
the other metrics:

| Metric | Checks |
| --- | --- |
| `contains` / `not_contains` | required and forbidden substrings |
| `regex` / `exact_match` / `similarity` | pattern, equality, textual closeness |
| `json_valid` | a parseable JSON object when one was requested |
| `tools` | required and forbidden tool calls |
| `max_steps` / `max_tokens` / `max_seconds` | budgets |
| `no_error` | the run completed cleanly |

Reports render as text, JSON, Markdown, CSV or JUnit XML, and `--threshold`
makes the process exit non-zero below the bar.

## Tracing and metrics

```bash
magent trace                    # flame graph of the last run
magent trace --path ~/.local/state/magent/traces.jsonl
curl -s localhost:8765/api/metrics?format=prometheus
```

```python
from magent.observability import Observability
from magent.agent import Agent
from magent.llm.factory import build_backend

observability = Observability.from_config()
agent = Agent(backend=build_backend())
observability.attach(agent)
agent.run("do something")
print(agent.events.stats())
print(observability.metrics.prometheus())
```

## The improvements manifest

The build ships a machine-checkable claim: **3,539 improvements**, each traced to
a file and symbol that exists in this repository.

```bash
make improvements      # regenerate improvements.json + IMPROVEMENTS.md
make verify            # re-check every reference against the tree
make count             # print the total
```

`scripts/build_improvements.py` walks the source with `ast` and emits one entry
per documented function, class and method; per registered tool; per CLI command;
per skill, persona, prompt template and preset; per test; per documentation
section, Makefile target and workflow file.

`scripts/verify_improvements.py` re-resolves every reference:
the file must exist, a Python symbol must be a *documented* definition (or a name
the module declares), a token reference must appear verbatim, and the total must
meet the promised minimum. `--strict` additionally requires the manifest to
match what the generator would produce today, which is what `make verify` runs.
Delete a function and the build fails - the number cannot drift away from the
code.
