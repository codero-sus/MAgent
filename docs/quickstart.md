# Quickstart

Five minutes from a clone to an agent that uses tools, memory and citations.

## 1. Check the installation

```bash
python -m magent version        # or `magent version` after pip install -e .
python -m magent doctor         # every check should pass or warn
```

`doctor` probes the backend, the sandbox, guardrails, memory, sessions, plugins,
skills and disk. A `warn` on the backend means Cortex LLMHoster is not running
yet, so MAgent fell back to its offline echo backend.

## 2. Talk to your own models

Start Cortex LLMHoster, then:

```bash
magent models                    # list what Cortex serves
magent cortex state              # admin view: runtime, hardware, metrics
magent ask "explain what you can do"
```

Point MAgent at a different host with `MAGENT_CORTEX_URL`, choose a model with
`--model`, or force the offline backend with `--offline`.

## 3. Use it from Python

```python
from magent import Agent, build_backend

agent = Agent(backend=build_backend())
result = agent.run("what tools do you have?")
print(result.output)
print(f"{result.usage.total_tokens} tokens in {result.duration_ms:.0f} ms")
```

## 4. Give it tools and a workspace

```python
from magent.agent import Agent
from magent.llm.factory import build_backend
from magent.tools import default_registry

agent = Agent(backend=build_backend(), workspace=".", registry=default_registry())
result = agent.run("read README.md and list the three most important sections")
for step in result.steps:
    for call in step.tool_calls:
        print("called", call.name, call.arguments)
```

Filesystem and command tools run inside a sandbox: paths are resolved and jailed
to the workspace, denied patterns win over allowed ones, and dangerous commands
are refused outright. Desktop tools use separate capability, limit, fail-safe,
and human-approval controls. See [tools.md](tools.md).

## 5. Remember something

```python
from magent.memory import build_store

store = build_store()
store.add("the deploy key lives in ~/.config/magent/secrets.toml", kind="fact", tags=["ops"])
store.save()

for entry in store.search("where is the deploy key"):
    print(entry.score, entry.content)
```

## 6. Answer from your own documents

```bash
magent rag ingest docs/
magent rag search --query "how do guardrails work"
```

```python
from magent.rag import Retriever

retriever = Retriever()
retriever.ingest_directory("docs")
result = retriever.retrieve("how do guardrails work", limit=3)
print(result.context)          # numbered, cited context
print(result.verify("..."))    # are the citations real?
```

## 7. Keep it on your desktop

```bash
pip install 'MAgent[desktop]'    # optional but recommended backend
magent desktop status
magent desktop                   # click the floating logo for task/queue/chat
```

The queue is local SQLite, actions that change the desktop ask for approval, and
the red **Stop** control cancels current work. See [desktop.md](desktop.md) before
using automation on a real session.

## 8. Evaluate it

```bash
magent eval sample --output evals/smoke.json
magent eval --dataset evals/smoke.json --threshold 0.9
```

The exit code is non-zero below the threshold, so it drops straight into CI.

## Next

* [agents.md](agents.md) for loop modes, plans and streaming
* [guardrails.md](guardrails.md) before you let an agent touch a real system
* [desktop.md](desktop.md) for launcher, queue, workflows, permissions, and safety
* [cli.md](cli.md) for the full command list
