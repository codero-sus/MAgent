# MAgent configuration presets

Each file is a real `magent.toml` fragment and can be selected with `--config`:

```bash
magent --config agents/coder.toml run 'Review this repository'
magent --config agents/researcher.toml run --rag docs/ 'Explain the architecture'
```

The files only change documented typed settings. CLI flags, environment variables, and `--set` still have higher precedence. Copy a preset into your own configuration and add Cortex model ids or policy specific to your environment.

These presets do **not** contain credentials. Use `MAGENT_CORTEX_API_KEY` and `MAGENT_SERVER_KEY` for secrets.
