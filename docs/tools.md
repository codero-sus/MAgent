# Tools, sandbox, and approvals

MAgent ships 113 tools in 16 categories. Each tool is an ordinary Python callable plus a `ToolSpec`: model-facing description, JSON Schema, return type, category/tags, risk, side-effect flags, timeout, output limit, parallel-safety declaration, examples, aliases, dependency hints, and approval policy.

## Inspect the catalogue

```bash
magent tools categories
magent tools list
magent tools list --category files
magent tools search 'regular expression'
magent tools show read_file
magent tools schema calculate read_file
magent tools risks
```

Categories are `cortex`, `data`, `datetime`, `files`, `git`, `math`, `media`, `memory`, `meta`, `output`, `python`, `search`, `shell`, `system`, `text`, and `web`.

## Define a tool

```python
from magent import tool
from magent.tools import ToolContext

@tool(
    category="inventory",
    risk="read",
    filesystem=True,
    tags=["warehouse"],
    timeout=5.0,
    output_limit=8_000,
)
def inventory_count(sku: str, *, context: ToolContext) -> dict[str, object]:
    """Read an inventory count from the current workspace.

    Args:
        sku: product identifier
        context: constrained invocation context supplied by MAgent
    """
    path = context.resolve_path("inventory.json")
    # Read and parse path here.
    return {"sku": sku, "count": 0, "source": str(path)}
```

Register it:

```python
from magent import default_registry

registry = default_registry()
registry.register(inventory_count)
```

The decorator derives schemas from type hints, defaults, literals/enums, lists, dictionaries, optionals, dataclasses, and Google-style `Args:` documentation. The function remains callable as normal Python; its `.tool` attribute contains the built `Tool`.

## Dispatch contract

```python
from magent.types import ToolCall
from magent.tools import ToolContext

call = ToolCall(id="call_1", name="calculate", arguments={"expression": "6*7"})
result = registry.dispatch(call, ToolContext(workspace="."))
```

Dispatch always returns `ToolResult`:

- valid arguments are safely coerced;
- missing/wrong arguments return `tool_validation_error`;
- unknown names return `tool_not_found` and suggestions;
- handler exceptions return `tool_execution_error`;
- call id and name always survive;
- output is truncated to the declared limit;
- timings and category/risk metadata are recorded.

A failure is sent back to Cortex as a `tool` message so the model can change approach. It does not tear down the entire run unless configured to stop on errors.

## Parallel dispatch

`dispatch_many()` uses a bounded thread pool only when every selected tool declares `parallel_safe=True`. Results are returned in input order even when they complete out of order. An overall timeout still produces one failure result per input call, so Cortex is never left waiting for a missing tool-call id.

## Risk ladder

```text
safe < read < write < execute < destructive < network
```

Risk is independent of side-effect flags. A tool can also declare:

- `destructive`;
- `network`;
- `filesystem`;
- `executes_code`;
- `requires_approval`.

These fields feed filtering, prompts, approvals, and operator displays.

## Approval modes

| Mode | Behavior |
|---|---|
| `never` | deny approval-requiring actions |
| `ask` | ask for every action covered by policy |
| `always` | approve automatically |
| `dangerous` | ask only for risky/destructive actions (default) |

Per-tool decisions, allow/deny lists, and an audit history are supported. `magent run --yes` is an explicit non-interactive approval. `--dry-run` validates calls and returns the would-run tool/arguments without invoking the handler.

The server returns HTTP 428 `approval_required` for a protected direct tool call unless the request explicitly includes approval.

## Sandbox policy

`ToolContext.resolve_path()` resolves relative paths under the workspace and delegates to `Sandbox`. Policy controls:

- allowed roots and read/write outside-root behavior;
- denied/sensitive path patterns;
- command allow/deny lists and executable policy;
- network enablement and domain allow/deny lists;
- process, read, write, and output bounds;
- subprocess environment passthrough and umask.

Built-in file, shell, Python, web, Git, and media tools call these checks before side effects.

> The Python-level sandbox is defense in depth for framework tools. It is not an OS isolation boundary against deliberately hostile Python. Run untrusted code/plugins in a container, VM, or restricted operating-system account.

## Built-in tool design

Every module under `magent.tools.builtin` exports `TOOLS` and `load(registry)`. Loading failures are reported; they are not silently swallowed. Optional external executables are checked at call time and represented in tool requirements.

Safe tools are covered by `scripts/smoke_tools.py`, which executes 83 cases in an isolated workspace. Network, live Cortex, repository-mutating Git, and persistent-memory cases are deliberately skipped there and covered by targeted tests/fakes instead.

## Filtering for one run

```python
result = agent.run(
    "Analyze the CSV",
    tools=["csv_read", "csv_stats", "compute_statistics"],
    disabled_tools=["run_command"],
    allow_network=False,
)
```

Filtering creates a run-local view; it does not mutate the shared registry.
