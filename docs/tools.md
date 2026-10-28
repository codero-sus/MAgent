# Tools

MAgent ships 143 tools in 17 categories. They are ordinary Python functions with
type hints and docstrings; the JSON schema the model sees is inferred from both.

## Writing a tool

```python
from pathlib import Path
from magent.tools import tool

@tool(name="count_lines", description="Count the lines in a file", category="files")
def count_lines(path: Path, skip_blank: bool = False) -> dict:
    """Count lines, optionally ignoring blank ones.

    Args:
        path: file to read, inside the workspace
        skip_blank: ignore lines that are empty after stripping
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if skip_blank:
        lines = [line for line in lines if line.strip()]
    return {"lines": len(lines)}
```

The decorator keeps the function callable, and attaches the tool:

```python
count_lines("README.md")               # normal call
count_lines.tool.run({"path": "README.md", "skip_blank": True})
count_lines.tool.spec.parameters       # JSON Schema
```

Inference handles `Literal` (enum), `X | None` (nullable), `Path` (path format),
defaults (not required) and `**Argument docs in the docstring** (descriptions).

## Categories and risk

| Category | Count | Examples |
| --- | --- | --- |
| files | 17 | `read_file`, `write_file`, `edit_file`, `glob_files`, `delete_path` |
| cortex | 12 | `cortex_health`, `cortex_models`, `cortex_model_control` |
| desktop | 30 | `desktop_click`, `desktop_screenshot`, `desktop_run_workflow`, `desktop_queue_add` |
| text | 10 | `text_stats`, `regex_replace`, `extractive_summary` |
| git | 10 | `git_status`, `git_diff`, `git_commit`, `git_blame` |
| data | 8 | `csv_read`, `csv_stats`, `sqlite_query`, `group_aggregate` |
| shell / system | 12 | `run_command`, `run_script`, `env_get`, `list_processes` |
| python | 3 | `python_exec`, `inspect_python`, `audit_python` |
| web / search | 7 | `fetch_page`, `http_get`, `search_documents` |

Risk levels are ordered: `safe < read < write < execute < destructive < network`.
`--preset sandboxed` caps the whole run at `max_risk="read"`.

## The sandbox

Every filesystem path a file, shell, or code tool touches is resolved and checked
against the workspace jail:

```python
from magent.tools import build_sandbox
from magent.config import default_config

sandbox = build_sandbox(default_config(), workspace=".")
sandbox.check_read("docs/index.md").allowed        # True
sandbox.check_read("/etc/hostname")                 # raises PermissionDeniedError
sandbox.check_command("ls -la").allowed             # True
sandbox.check_command("rm -rf /")                   # raises
```

Deny rules win over allow rules, `ALWAYS_DENIED_COMMANDS` can never be allowed,
byte ceilings apply to reads and writes, and every decision is recorded for
`magent doctor` and the audit log. Set `[tools].deny_action = "warn"` to get a
decision object instead of an exception.

> The sandbox is a guardrail, not a security boundary. Run untrusted code in a
> container. Desktop automation is outside the filesystem jail and is instead
> controlled by tool policy, explicit approvals, capability gates, limits, and
> fail-safe stops; it is not an operating-system security boundary either.

## Approvals

```toml
[tools]
approval_mode = "dangerous"   # never | ask | always | dangerous
```

* `never` - no prompts (use for CI with a read-only tool set)
* `ask` - prompt for anything risky
* `dangerous` - prompt for `write`/`execute`/`destructive`/`network`
* `always` - prompt for every call

A denied request becomes a tool result with `error_code="approval_denied"`; the
run continues and the model is told the call was refused. Desktop-changing tools
also have `requires_approval=true` and enforce approval when dispatched directly,
so an HTTP tool call cannot silently bypass the agent loop.

## Calling tools directly

```bash
magent tools                                   # list everything
magent tools --category git --format json
magent tools inspect write_file                # the exact schema the model sees
magent tools call calculate '{"expression": "6*7"}'
```
