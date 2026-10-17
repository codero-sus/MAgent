# Agents and loop modes

`Agent` owns the bounded reasoning/tool loop. It is cheap to construct and reusable: per-run state is returned in `RunResult`, while cumulative counters and conversation history stay on the agent until `reset()`.

## Construct an agent

```python
from magent import Agent, CortexBackend, CortexSettings, load_config

config = load_config()
agent = Agent(
    backend=CortexBackend(CortexSettings.from_env()),
    config=config.agent,
    settings=config,
    workspace=".",
    name="MAgent",
)
```

Optional constructor services include a `ToolRegistry`, `Sandbox`, `ApprovalManager`, memory store, `EventBus`, guardrail pipeline, session store, usage tracker, token estimator, and per-agent event callback.

## Run

```python
result = agent.run(
    "Inspect this repository and report the three highest-risk issues",
    persona="reviewer",
    max_iterations=8,
    model="",
)
```

Per-run options are validated. Unknown names fail immediately instead of being ignored. Options include:

- `system`, `persona`, and `model`;
- generation temperature and max tokens;
- `max_iterations` and `loop_mode`;
- enabled/disabled tools and network policy;
- `dry_run`, `stream`, and structured schema;
- context, memory, examples, and task prompt blocks;
- session id and metadata;
- memory recall/outcome persistence controls.

## Loop modes

### `react`

The default. Cortex returns text and/or tool calls. MAgent validates and executes calls, appends `tool` messages, then asks Cortex again until it receives a final answer or reaches a bound.

### `plan`

MAgent asks for a typed plan, tracks plan-step state, can replan after failures, and executes toward the goal. Plans have bounded step counts and serializable status.

### `reflect`

At configured intervals, the agent evaluates progress and can continue, change approach, or stop. Reflection is structured rather than hidden chain-of-thought.

### `single`

Uses the simplest loop policy for short work while still completing a necessary tool/result round trip.

Only these four names are valid. `plan_execute` is not a mode.

## `RunResult`

A run returns:

- stable `run_id`;
- final `output`;
- typed `StopReason`;
- ordered `StepRecord` values;
- cumulative `Usage` and `CostEstimate`;
- model and timings;
- model-visible `Message` transcript;
- parsed structured output when requested;
- the run's non-terminal event trace;
- error and metadata snapshots.

Convenience properties report success, iteration count, tool-call/error counts, duration, tokens per second, and cache hit rate.

```python
print(result.summary_line())
print(result.to_json(indent=2))
```

The terminal `run.finished`/`run.failed` event carries the `RunResult` itself and therefore lives on the event bus/stream rather than recursively inside `result.events`.

## Stop reasons

| Reason | Meaning | CLI exit |
|---|---|---:|
| `completed` | final answer produced | 0 |
| `max_iterations` | iteration bound reached | 3 |
| `budget_exceeded` | token/time/cost budget | 4 |
| `guardrail` | blocked by policy | 5 |
| `needs_input` | caller input required | 6 |
| `tool_limit` | tool-call bound reached | 7 |
| `error` | execution error | 1 |
| `cancelled` | caller cancellation/interrupt | 130 |

Cancellation is cooperative: `agent.cancel(reason=...)` asks the current run to stop at a safe boundary. The cancellation flag resets after the cancelled run.

## Events

There are 27 typed event kinds, including:

```text
run.started      run.finished     run.failed
step.started     step.finished
llm.request      llm.response     llm.retry      llm.cache_hit
tool.call        tool.result      tool.error
guard.blocked    guard.approval_requested       guard.approval_decided
memory.read      memory.write
plan.created     plan.step        agent.reflection
agent.thinking   agent.text_delta agent.text_done
budget.warning   router.model_switched          log custom
```

Subscribe to the bus:

```python
subscription = agent.events.subscribe(
    lambda event: print(event.kind.value, event.data),
    kinds=["tool", "run.*"],
)
result = agent.run("Calculate 19 * 23")
agent.events.unsubscribe(subscription)
```

Or stream a run:

```python
for event in agent.stream("Summarize README.md"):
    print(event.kind.value)
```

Each event is yielded exactly once. The final event stores the full result under `event.data["result"]`.

## Structured output

```python
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}
result = agent.run("Return the assessment as JSON", structured=schema)
print(result.structured)
```

MAgent extracts plain or fenced JSON, validates it with the same schema engine used for tools, and makes one repair request with validation feedback if needed.

## Reuse and reset

```python
first = agent.run("Remember the current task")
second = agent.run("Continue")
print(agent.stats())
agent.reset()
```

`reset()` clears conversation/cancellation state; it does not delete persistent memory or sessions. `close()` closes the backend and owned resources.

## Prompt construction

The system prompt composes rather than concatenates arbitrary walls of text:

- product identity and selected persona;
- environment facts and workspace state;
- tool-use policy and compact catalogue;
- safety policy;
- retrieved memory/context;
- output/structured instructions.

A token-aware context assembler allocates shares to system, instructions, tools, retrieved context, memory, and history, then trims or summarizes while preserving required sections.
