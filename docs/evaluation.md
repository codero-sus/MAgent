# Evaluation

MAgent's evaluation package runs repeatable cases against any backend and computes text, structure, tool-use, usage, latency, and composite metrics. It is intended for regression gates, model comparisons, prompt changes, and tool-policy changes.

## CLI

```bash
magent eval list --dataset builtin
magent eval show CASE_ID --dataset builtin
magent eval run --dataset builtin
magent eval metrics
magent eval tags --dataset builtin
magent eval init ./evaluation.json
magent eval convert INPUT OUTPUT
```

Gate CI by pass rate:

```bash
magent eval run --dataset ./evaluation.json --fail-below 0.80
```

A threshold miss exits with code 8, distinct from a runtime failure.

## Dataset

A JSON dataset contains metadata and cases. A case can specify:

- id/name/tags;
- prompt or message history;
- expected text/contains/regex;
- expected JSON or JSON Schema;
- expected tool names/arguments/results;
- expected stop reason;
- max steps, tokens, latency, or cost;
- metric names and weights;
- per-case backend/model/options;
- setup/teardown metadata.

Use `magent eval init` to write a valid starter instead of guessing the schema.

## Metrics

The registry includes exact/contains/regex, normalized text similarity, token/word overlap, JSON equality/subset/schema, tool selection, stop reason, and efficiency-style measures. Metrics return typed scores with details rather than only booleans.

Composite scoring combines configured weights. A case can have a minimum score; a dataset can have a pass-rate threshold.

## Offline evaluation

```bash
magent --offline eval run --dataset builtin
```

Echo mode is useful for exercising the runner itself. For deterministic tool behavior in Python, use `magent.testing.make_backend`.

## Reports

The runner records attempts, timings, usage, model, output, errors, metric results, and case status. Text, JSON, and JSONL renderers support humans and CI. Aggregation reports count, passed/failed/error, pass rate, mean score, percentile latency, and token totals.

## Python

```python
from magent.eval import load_dataset, run_dataset

# Inspect exact signatures in magent.eval for custom runner/backend wiring.
dataset = load_dataset("evaluation.json")
```

The CLI is the recommended stable operator surface. The `magent.eval` package exposes dataset, case, metric registry, runner, result, report, and comparison primitives for applications.

## Reproducibility

- Pin the Cortex model/runtime configuration.
- Set generation seed when supported.
- Record MAgent/Cortex versions and dataset hash.
- Keep tool/network policy the same across candidates.
- Use warmup/repetition for latency comparisons.
- Treat a single local timing as indicative, not statistically conclusive.

Evaluation output is evidence about the cases you wrote; it is not a universal model-quality claim.
