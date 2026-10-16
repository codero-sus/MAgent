# Providers

MAgent is built for **Cortex LLMHoster** and also runs fully offline.

## Cortex LLMHoster

Cortex serves local models behind an OpenAI-compatible API under `/v1` plus an
admin control plane under `/admin/api`. MAgent speaks both.

```bash
export CORTEX_API_KEY=...            # optional; Cortex can run without auth
magent models                        # GET /v1/models
magent cortex state                  # runtime, hardware, metrics, model states
magent cortex start --model qwen2.5-coder-7b
magent cortex metrics
```

Connection settings come from the configuration file or the environment:

| Setting | Default | Environment |
| --- | --- | --- |
| base URL | `http://127.0.0.1:8624` | `MAGENT_CORTEX_URL` |
| API key | none | `MAGENT_CORTEX_API_KEY` |
| prefix | `/v1` | `MAGENT_API_PREFIX` |
| timeout | 120 s | `MAGENT_TIMEOUT` |
| backend | `auto` | `MAGENT_BACKEND` |
| model | server default | `MAGENT_MODEL` |

`MAGENT_*` variables win over `CORTEX_*` ones, so MAgent settings can differ
from a shared Cortex installation.

## Backend selection

`build_backend()` resolves in this order: an explicit kind, then
`MAGENT_BACKEND`, then a probe of `GET /health` and `GET /ready` at the
configured URL. If nothing answers and `fallback_to_echo` is on (the default),
you get the offline echo backend instead of an exception - which is why
`magent doctor` warns rather than fails on a machine with no models.

```python
from magent.llm.factory import build_backend

backend = build_backend()                                  # auto
backend = build_backend(config=config)                      # from the config file
backend = build_backend(config=config, offline=True)        # force echo
```

## Routing and failover

```python
from magent.llm.router import ModelRouter, Route
from magent.llm.cortex import CortexBackend, CortexSettings

router = ModelRouter([
    Route(backend=CortexBackend(CortexSettings(model="big")), label="big", priority=10),
    Route(backend=CortexBackend(CortexSettings(model="small")), label="small", priority=20),
], strategy="priority", max_failovers=2)
```

Strategies: `priority`, `round_robin`, `least_used`, `fastest`, `cheapest`,
`random`. Each route has a circuit breaker, so a failing model stops being
tried until it recovers.

## Caching, retries and budgets

* responses cache to disk under `[cache].directory`, keyed on the request, with
  a TTL; streaming responses are not cached unless you ask
* retries use exponential backoff with jitter and never retry a schema error
* token budgets live in `[guardrails]`; usage is tracked per run, per model and
  in total, and `PriceBook` turns tokens into money when you configure rates

```python
from magent.llm.usage import BudgetConfig, UsageTracker

tracker = UsageTracker(budget=BudgetConfig(max_tokens=20_000))
tracker.record_tokens(1_500)
print(tracker.remaining(), tracker.utilization())
```

## Advanced capabilities

Cortex advertises capabilities per model (vision, OCR, image, audio, video,
embeddings, reranking, moderation). The backend exposes them so an agent can
check before calling:

```python
info = backend.models()[0]
print(info.id, info.capabilities)
```

`CortexAdmin` covers the control plane: `state`, `metrics`, `hardware`,
`cookbook`, `start_model`, `stop_model`, `restart_model`, `test_model`,
`model_files`, `check_update`, `auth_options`.
