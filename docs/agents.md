# Agents

An agent is a backend plus a tool registry, a sandbox, guardrails, memory, a
session and an event bus. `Agent.run()` drives a loop, records every step, and
returns a `RunResult` you can inspect, serialise and replay.

## Loop modes

| Mode | Behaviour | Use it for |
| --- | --- | --- |
| `react` | think, call tools, observe, repeat | general work |
| `plan` | write a plan first, then execute step by step | multi-step tasks |
| `reflect` | work, then critique itself and continue if needed | quality-sensitive answers |
| `single` | one completion, no tools | classification, extraction, drafting |

```bash
magent run "refactor parse_args into three functions" --loop-mode plan
magent run "is this regex safe?" --loop-mode reflect --stream
magent ask "summarise this changelog" --loop-mode single
```

```python
from magent.agent import Agent
from magent.llm.factory import build_backend

agent = Agent(backend=build_backend(), loop_mode="plan", max_iterations=10)
result = agent.run("add pagination to the sessions API")
print(result.stop_reason, len(result.steps))
```

## What a run records

`RunResult` carries the answer, the stop reason, every `StepRecord` (thought,
tool calls, tool results, usage, timings), token usage, cost, the messages and
the event list. `result.to_dict()` is JSON-safe - including when the terminal
event refers back to the run.

```python
result = agent.run("count the Python files")
print(result.succeeded, result.stop_reason)
print(result.tool_call_count, result.tool_error_count)
for step in result.steps:
    print(step.index, [call.name for call in step.tool_calls])
```

## Streaming

`agent.stream()` yields events as they happen and finishes with a
`run.finished` event carrying the whole result. Every event is delivered exactly
once.

```python
from magent.types import EventKind

for event in agent.stream("write a haiku about caching"):
    if event.kind == EventKind.TEXT_DELTA:
        print(event.data["text"], end="")
    elif event.kind == EventKind.TOOL_CALL:
        print("\n->", event.data["call"].name)
```

## Structured output

```python
schema = {"type": "object",
          "properties": {"name": {"type": "string"}, "stars": {"type": "integer"}},
          "required": ["name", "stars"]}
result = agent.run("describe this repository as JSON", structured=schema)
print(result.structured["name"])
```

`structured=True` asks for JSON without a schema. If the model cannot produce
valid JSON even after one repair attempt, `StructuredOutputError` is raised -
a schema you asked for is a contract, not a suggestion.

## Plans and reflection

```python
plan = agent.plan("migrate the CLI to argparse subcommands", max_steps=6)
for step in plan.steps:
    print(step.index + 1, step.title, step.tool)
```

```python
critique = agent.critique("explain the sandbox", "the sandbox blocks nothing")
print(critique["findings"] if "findings" in critique else critique)
```

## Presets

Twelve tuned configurations ship with the package: `general`, `coder`,
`reviewer`, `researcher`, `analyst`, `writer`, `devops`, `planner`, `tutor`,
`librarian`, `sandboxed`, `minimal`.

```bash
magent presets
magent run "review this diff" --preset reviewer
```

```python
from magent.agents import build_preset_agent

agent = build_preset_agent("coder", backend=build_backend())
print(agent.registry.count())      # the preset narrows the tool set
```

## Cancellation and budgets

```python
import threading
from magent.agent import Agent

agent = Agent(backend=build_backend())
threading.Timer(5.0, lambda: agent.cancel(reason="deadline")).start()
result = agent.run("keep working until you are stopped")
print(result.stop_reason)          # cancelled
```

Budgets live in `[guardrails]`: `budget_max_tokens`, `budget_max_cost`,
`budget_max_seconds`, `budget_max_steps`, `budget_max_tool_calls`. Exceeding one
stops the run with `StopReason.BUDGET` and a `budget.warning` event on the way.
