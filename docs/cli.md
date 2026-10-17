# CLI

26 commands, all of which accept `--format text|json|jsonl|yaml|csv|md`,
`--quiet`, `--verbose`, `--no-color`, `--output-file PATH` and the global
`--config`, `--profile`, `--workspace`, `--backend`, `--model`, `--offline`,
`--set KEY=VALUE` flags - before or after the subcommand.

## Running the agent

```bash
magent run "summarise docs/" --stream
magent run "review this change" --preset reviewer --file src/app.py
git diff | magent run "review this change" --stdin
magent ask "what is 2+2?"                       # just the answer
magent chat --session refactor-2026              # interactive
magent plan "add pagination to the sessions API"
magent reflect "explain the guardrail pipeline"
```

## Tools, presets, skills

```bash
magent tools                                     # 113 tools, 16 categories
magent tools --category git --format json
magent tools inspect write_file                  # the schema the model sees
magent tools call calculate '{"expression": "6*7"}'
magent presets coder
magent skills --task "review this pull request"
magent prompts personas coder
```

## Knowledge

```bash
magent memory add "the deploy key lives in ~/.config/magent"
magent memory search "deploy key"
magent memory forget --id mem_1234
magent memory compact
magent rag ingest docs/
magent rag search --query "how do guardrails work"
magent sessions list
magent sessions show ses_abc123 --transcript-format markdown
magent sessions search "flaky test"
magent sessions export ses_abc123 --output transcript.md
```

## System

```bash
magent doctor                                    # every check, with fixes
magent doctor --format json
magent config show --section tools
magent config set model.temperature 0.2
magent config paths
magent models
magent cortex state | metrics | hardware | cookbook
magent cortex start --model qwen2.5-coder-7b
magent plugins paths | check | inspect acme
magent init --skills --plugin
magent completion bash > /etc/bash_completion.d/magent
```

## Quality

```bash
magent eval sample --output evals/smoke.json
magent eval --dataset evals/smoke.json --threshold 0.9 --repeats 3 --concurrency 4
magent eval --report-format junit --output report.xml
magent bench --repeats 5
magent trace                                     # flame graph of the last run
magent improvements verify                       # re-check the manifest
magent improvements count
```

## Serving

```bash
magent serve
magent serve --host 0.0.0.0 --port 8080
magent serve --port 0 --background
```

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | the command ran and reported failure (a failing eval, doctor) |
| 2 | usage error |
| 3 | configuration error |
| 4 | backend or model unavailable |
| 5 | permission, sandbox or approval refusal |
| 6 | guardrail block or budget exhaustion |
| 130 | interrupted |
