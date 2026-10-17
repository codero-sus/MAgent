# MAgent skill prompts

These are executable task prompts, not hidden framework magic. Pass one to the real `run --file` surface:

```bash
magent --config agents/coder.toml run --file skills/repository-review.md
magent --config agents/devops.toml run --file skills/cortex-diagnosis.md
```

Append task-specific context through a copied file or use the skill as a template. Every skill states its evidence, safety, and done conditions so completion can be checked.
