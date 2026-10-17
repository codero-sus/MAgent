# Plugins and skills

Two extension points with different shapes: **skills** are instructions you drop
into a folder, **plugins** are Python that adds tools, hooks, personas, templates
and skills.

## Skills

A skill is Markdown with optional front matter:

```markdown
---
name: deploy-runbook
description: How this team ships to production
tags: [ops, release]
triggers: [deploy, release, ship]
tools: [run_command, read_file]
priority: 70
---
1. Run the test suite and read the output before anything else.
2. Deploy to staging, wait for the health check, then promote.
3. Record what you did in the incident channel.
```

Drop it in `.magent/skills/`, `skills/`, `docs/skills/` or
`~/.local/share/magent/skills/` and it is picked up automatically when the task
text matches its triggers.

```bash
magent skills --task "deploy the release"
```

```python
from magent.skills import build_library

library = build_library(workspace=".")
print(library.for_task("deploy the release")[0].name)
print(library.prompt_block("deploy the release"))
```

Twelve skills ship with the package: `code-review`, `debug-loop`,
`commit-messages`, `refactor-plan`, `test-plan`, `research-brief`,
`csv-analysis`, `release-notes`, `incident-report`, `api-design`,
`performance-triage` and `doc-writer`. A project file with the same name
overrides the built-in one.

## Plugins

A plugin is a directory with a manifest and an entry module:

```
.magent/plugins/acme/
├── plugin.json
└── acme.py
```

```json
{
  "name": "acme",
  "version": "1.2.0",
  "entry": "acme.py",
  "description": "Acme internal helpers",
  "provides": ["tools", "hooks", "skills"],
  "hooks": ["after_run"]
}
```

```python
from magent.skills import Skill
from magent.tools import tool

@tool(name="acme_ping", description="Ping the Acme service", category="network")
def acme_ping(host: str = "acme.internal") -> str:
    """Return a greeting from the Acme service."""
    return f"pong from {host}"

def after_run(payload, **context):
    """Note that a run finished."""
    return None

HOOKS = {"after_run": after_run}
SKILLS = [Skill(name="acme-style", description="Acme house style", body="Be brief.")]
```

Single-file plugins work without boilerplate: `@tool` functions, `HOOKS`,
`SKILLS`, `PERSONAS` and `TEMPLATES` at module level are discovered. A
`register(registry, context)` function or a `PLUGIN = {...}` dict is also
supported when you want explicit control.

### Loading rules

* the entry file must live inside the plugin directory - a path that escapes it
  is refused before import
* the manifest must validate: unknown capabilities and hooks are reported, not
  ignored
* oversized entry files are refused; checksums can be verified with
  `check_checksums=True`
* a plugin that raises during import or registration is recorded as failed
  instead of taking down the process; `strict=True` turns that into an exception

```bash
magent plugins paths      # where MAgent looks
magent plugins check      # load everything and report
magent plugins inspect acme
```

```python
from magent.plugins import PluginLoader, load_plugins

loader = PluginLoader(registry=registry, hooks=hooks)
loader.load_all(".magent/plugins")
print(loader.stats())
```

### Hooks

`before_run`, `after_run`, `before_prompt`, `after_prompt`, `before_tool`,
`after_tool`, `on_error`, `on_event`, `startup`, `shutdown`.

Handlers run in priority order and are isolated: one that raises is logged and
skipped. Returning a value replaces the payload; returning `None` leaves it
alone.

```python
from magent.plugins import HookRegistry

hooks = HookRegistry()
hooks.register("before_prompt", lambda payload, **kw: payload + "\nBe terse.", priority=10)
```
