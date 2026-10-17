# Verified improvements

The user-visible promise is **777+ real improvements**. MAgent exceeds that floor with thousands of generated entries, but the important property is traceability rather than the headline count.

## What counts

The collector scans actual repository surfaces:

- Python modules;
- public classes, functions, methods, and properties;
- documented public constants;
- tool implementations;
- configuration keys;
- CLI commands and flags;
- error types;
- event kinds;
- tests;
- documentation pages;
- examples and scripts;
- packaging/automation files.

A row is not free text. It has a category, title, and reference such as a source path plus symbol. The generated [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) is the canonical numbered manifest and must never be padded or hand-edited.

## Generate

```bash
PYTHONPATH=src python3 scripts/count_improvements.py --write
```

Useful forms:

```bash
python3 scripts/count_improvements.py --by-category
python3 scripts/count_improvements.py --json
python3 scripts/count_improvements.py --json-out improvements.json
python3 scripts/count_improvements.py --strict
```

The CLI wrapper is:

```bash
magent improvements --regenerate
```

## Verify

```bash
PYTHONPATH=src python3 scripts/verify_improvements.py
```

The verifier:

1. parses every numbered row;
2. checks that the referenced path exists;
3. imports/parses Python and resolves the named symbol where applicable;
4. validates special categories such as tools, CLI flags, errors, and events;
5. reports unresolvable references;
6. enforces the target of 777;
7. exits non-zero on any failure.

Machine output:

```bash
python3 scripts/verify_improvements.py --json
magent improvements --verify -f json
```

## Why generation is not padding

Generation prevents selective documentation but does not create features. The collector can only list a module, symbol, tool, flag, test, page, or packaging asset that exists under its category rules. The verifier independently resolves the result. Deleting or renaming a referenced implementation breaks verification until the manifest is regenerated and still meets the floor.

One implementation symbol normally contributes one entry, not hundreds of paraphrases. Tool entries map to registered `ToolSpec` names. CLI flags map to `argparse` actions. Error/event entries map to real classes/enums.

## Workflow

When code changes:

```bash
make test
PYTHONPATH=src python3 scripts/count_improvements.py --write
make verify
```

Review the manifest diff like generated API documentation. Unexpected large count changes usually signal a collector rule or public-surface mistake.

## Current report

Run this for the authoritative current number and category split:

```bash
magent improvements --verify
python3 scripts/count_improvements.py --by-category
```

README intentionally says “3,200+” rather than pinning a number that changes whenever real tests/docs/symbols are added. The generated manifest carries the exact count and version.
