"""The ``eval`` command: score agent behaviour against a dataset.

Unit tests prove components work; evaluation proves the agent does. The
difference shows up in exactly the places integration breaks - prompt assembly,
context trimming, tool availability, guardrail interaction - which no component
test covers on its own.

The command is built to be used in CI: ``--fail-below`` sets a required pass
rate and the exit code reflects it, so a regression fails the build instead of
hiding in a report nobody reads. ``--dataset builtin`` runs offline against the
echo backend, which means the harness itself is exercisable without a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...errors import EvalError, EvalThresholdError
from ...eval import (
    METRIC_NAMES,
    EvalCase,
    EvalDataset,
    EvalRunner,
    builtin_dataset,
    describe_metrics,
    load_dataset,
    resolve_dataset,
)
from ...utils.tables import render_key_value, render_markdown_table, render_table
from ..args import CliContext, EXIT_ERROR, EXIT_OK


def _shared_parent() -> argparse.ArgumentParser:
    """Return a help-less parser carrying the flags every eval action accepts.

    Declaring ``--dataset`` only on the ``eval`` parser would make
    ``magent eval run --dataset x`` a usage error, because argparse hands
    everything after the action name to the action's own parser. Sharing the
    flags as a parent makes both positions work.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dataset", default="", metavar="NAME|PATH",
                        help="dataset name or file (default: builtin)")
    parent.add_argument("--fail-below", type=float, default=-1.0, metavar="RATE",
                        help="required pass rate 0-1; exit non-zero below it")
    return parent


def register(subparsers: Any) -> None:
    """Add the ``eval`` subcommand and its actions."""
    shared = _shared_parent()
    parents = [shared]
    parser = subparsers.add_parser("eval", help="evaluate agent behaviour against a dataset",
                                   description="Run evaluation cases and report the scores.",
                                   parents=parents)
    action = parser.add_subparsers(dest="action", metavar="ACTION")

    ran = action.add_parser("run", help="run the dataset", parents=parents)
    ran.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    ran.add_argument("--case", action="append", default=[], help="only these case ids")
    ran.add_argument("--limit", type=int, default=0, help="run at most N cases")
    ran.add_argument("--concurrency", type=int, default=0, help="cases to run at once")
    ran.add_argument("--repeats", type=int, default=0, help="executions per case")
    ran.add_argument("--sequential", action="store_true", help="force one case at a time")
    ran.add_argument("--report", default="", metavar="PATH", help="write the report here")
    ran.add_argument("--failures-only", action="store_true", help="print only failed cases")
    ran.add_argument("--metrics", default="", metavar="NAME,...",
                     help="metrics to apply when a case names none")
    ran.add_argument("--strict", action="store_true",
                     help="same as --fail-below 1.0: every case must pass")

    listed = action.add_parser("list", help="list the cases in a dataset", parents=parents)
    listed.add_argument("--tag", action="append", default=[], help="only cases with this tag")
    listed.add_argument("--limit", type=int, default=0, help="maximum cases")

    shown = action.add_parser("show", help="print one case in full", parents=parents)
    shown.add_argument("id", help="case identifier")

    action.add_parser("metrics", help="list the available scoring metrics", parents=parents)
    action.add_parser("tags", help="list the tags in a dataset", parents=parents)

    init = action.add_parser("init", help="write a starter dataset file", parents=parents)
    init.add_argument("path", help="destination file (.json or .jsonl)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")

    converted = action.add_parser("convert", help="convert a dataset between formats",
                                  parents=parents)
    converted.add_argument("source", help="input dataset")
    converted.add_argument("destination", help="output file (.json or .jsonl)")

    parser.set_defaults(handler=cmd_eval, action_default="run")


def _dataset(args: argparse.Namespace, context: CliContext) -> EvalDataset:
    """Resolve the dataset named by flags or configuration."""
    spec = str(getattr(args, "dataset", "") or "")
    configured = str(context.config.eval.dataset or "builtin")
    configured_path = str(context.config.eval.dataset_path or "")
    if spec:
        candidate = Path(spec).expanduser()
        if candidate.exists():
            return load_dataset(candidate)
        return resolve_dataset(spec)
    if configured_path:
        return load_dataset(Path(configured_path).expanduser())
    if configured and configured != "builtin":
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return load_dataset(candidate)
    return builtin_dataset()


def _filtered(dataset: EvalDataset, args: argparse.Namespace) -> EvalDataset:
    """Apply the ``--tag``, ``--case`` and ``--limit`` filters."""
    return dataset.filter(tags=list(getattr(args, "tag", []) or []),
                          ids=list(getattr(args, "case", []) or []),
                          limit=int(getattr(args, "limit", 0) or 0))


def cmd_eval(args: argparse.Namespace, context: CliContext) -> int:
    """Dispatch one ``eval`` action."""
    action = getattr(args, "action", "") or getattr(args, "action_default", "run")
    handlers: dict[str, Any] = {
        "run": _run, "list": _list, "show": _show, "metrics": _metrics,
        "tags": _tags, "init": _init, "convert": _convert,
    }
    try:
        return handlers.get(action, _run)(args, context)
    except EvalThresholdError as exc:
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR
    except EvalError as exc:
        context.printer.error(f"error: {exc}")
        return EXIT_ERROR


def _fail_below(args: argparse.Namespace, context: CliContext) -> float:
    """Resolve the required pass rate from flags and configuration."""
    if getattr(args, "strict", False):
        return 1.0
    value = float(getattr(args, "fail_below", -1.0))
    if value >= 0:
        return max(0.0, min(1.0, value))
    configured = float(context.config.eval.fail_below or 0.0)
    return max(0.0, min(1.0, configured))


def _run(args: argparse.Namespace, context: CliContext) -> int:
    """Run the dataset and report."""
    dataset = _filtered(_dataset(args, context), args)
    if not len(dataset):
        context.printer.error("error: no cases matched the filters")
        return EXIT_ERROR

    metrics = [item.strip() for item in str(getattr(args, "metrics", "") or "").split(",")
               if item.strip()]
    unknown = [name for name in metrics if name not in METRIC_NAMES]
    if unknown:
        context.printer.error(f"error: unknown metric(s) {', '.join(unknown)}; "
                              f"available: {', '.join(METRIC_NAMES)}")
        return EXIT_ERROR

    concurrency = 1 if getattr(args, "sequential", False) else \
        int(getattr(args, "concurrency", 0) or context.config.eval.concurrency or 4)
    repeats = int(getattr(args, "repeats", 0) or context.config.eval.repeats or 1)
    fail_below = _fail_below(args, context)

    agent = context.make_agent()
    backend = context.backend

    def progress(result: Any, position: int, total: int) -> None:
        mark = "+" if result.passed else "x"
        theme = "success" if result.passed else "error"
        context.printer.write(context.printer.themed(
            f"  {mark} [{position}/{total}] {result.case_id} {result.aggregate:.2f}", theme),
            stream=context.printer.err, min_level="verbose")

    context.printer.write(context.printer.themed(
        f"evaluating {len(dataset)} case(s) from {dataset.name} "
        f"(concurrency {concurrency}, repeats {repeats})", "heading"), min_level="normal")

    runner = EvalRunner(agent=agent, concurrency=concurrency, repeats=repeats,
                        metrics=metrics, fail_below=fail_below, backend=backend,
                        model=str(context.config.model.model or ""),
                        on_progress=progress if context.options.at_least("verbose") else None)
    report = runner.run(dataset)

    report_path = str(getattr(args, "report", "") or context.config.eval.report_path or "")
    if report_path and not context.dry_run:
        fmt = str(context.config.eval.report_format or "markdown")
        written = report.save(report_path, fmt=fmt,
                              failures_only=bool(getattr(args, "failures_only", False)))
        context.printer.write(f"report written to {written}", min_level="normal")

    if context.options.fmt == "json":
        context.emit(report.to_dict())
    else:
        fmt = "markdown" if context.options.fmt == "markdown" else "text"
        context.printer.write(
            report.render(fmt, width=context.options.width,
                          failures_only=bool(getattr(args, "failures_only", False))),
            min_level="quiet")

    return EXIT_OK if report.ok else EXIT_ERROR


def _list(args: argparse.Namespace, context: CliContext) -> int:
    """List the cases in a dataset."""
    dataset = _filtered(_dataset(args, context), args)
    if context.options.fmt == "json":
        context.emit({"dataset": dataset.name, "count": len(dataset),
                      "cases": [case.to_dict() for case in dataset]})
        return EXIT_OK
    if not len(dataset):
        context.printer.write("no cases matched", min_level="quiet")
        return EXIT_OK
    rows = [[case.id, ",".join(case.tags) or "-", f"{case.threshold:.2f}",
             " ".join(case.prompt.split())[:56]] for case in dataset]
    table = render_markdown_table(rows, headers=["id", "tags", "min", "prompt"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["id", "tags", "min", "prompt"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    context.printer.write(context.printer.themed(
        f"{len(dataset)} case(s) in {dataset.name}", "muted"), min_level="normal")
    return EXIT_OK


def _show(args: argparse.Namespace, context: CliContext) -> int:
    """Print one case in full."""
    dataset = _dataset(args, context)
    wanted = str(getattr(args, "id", "") or "")
    case = dataset.by_id(wanted)
    if case is None:
        context.printer.error(f"error: no case {wanted!r} in {dataset.name}")
        close = [item.id for item in dataset if wanted.lower() in item.id.lower()][:6]
        if close:
            context.printer.write(context.printer.themed(
                f"did you mean: {', '.join(close)}", "muted"), min_level="quiet")
        return EXIT_ERROR
    if context.options.fmt == "json":
        context.emit(case.to_dict())
        return EXIT_OK
    printer = context.printer
    printer.write(printer.themed(case.id, "heading"), min_level="quiet")
    payload = case.to_dict()
    printer.write(render_key_value([(key, value) for key, value in payload.items()
                                    if key not in {"id", "prompt"}]), min_level="quiet")
    printer.blank(min_level="quiet")
    printer.write(printer.themed("prompt", "muted"), min_level="quiet")
    printer.write(case.prompt, min_level="quiet")
    if case.context:
        printer.blank(min_level="quiet")
        printer.write(printer.themed("context", "muted"), min_level="quiet")
        printer.write(case.context, min_level="quiet")
    return EXIT_OK


def _metrics(args: argparse.Namespace, context: CliContext) -> int:
    """List the available scoring metrics."""
    help_text = describe_metrics()
    if context.options.fmt == "json":
        context.emit({"metrics": help_text, "count": len(help_text)})
        return EXIT_OK
    rows = [[name, help_text[name]] for name in sorted(help_text)]
    table = render_markdown_table(rows, headers=["metric", "what it checks"]) \
        if context.options.fmt == "markdown" \
        else render_table(rows, headers=["metric", "what it checks"],
                          unicode_borders=context.options.unicode)
    context.printer.write(table, min_level="quiet")
    return EXIT_OK


def _tags(args: argparse.Namespace, context: CliContext) -> int:
    """List the tags in a dataset with case counts."""
    dataset = _dataset(args, context)
    counts: dict[str, int] = {}
    for case in dataset:
        for tag in case.tags or ["(untagged)"]:
            counts[tag] = counts.get(tag, 0) + 1
    if context.options.fmt == "json":
        context.emit({"dataset": dataset.name, "tags": counts})
        return EXIT_OK
    rows = [[tag, count] for tag, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    context.printer.write(render_table(rows, headers=["tag", "cases"],
                                       alignments=["left", "right"],
                                       unicode_borders=context.options.unicode),
                          min_level="quiet")
    return EXIT_OK


def _init(args: argparse.Namespace, context: CliContext) -> int:
    """Write a starter dataset file."""
    target = Path(str(getattr(args, "path", "") or "")).expanduser()
    if not str(target):
        context.printer.error("error: a destination path is required")
        return EXIT_ERROR
    if target.exists() and not getattr(args, "force", False):
        context.printer.error(f"error: {target} exists; pass --force to overwrite")
        return EXIT_ERROR
    dataset = EvalDataset(
        name=target.stem,
        description="Starter MAgent evaluation dataset",
        cases=[
            EvalCase(id="arithmetic", prompt="What is 6 * 7?", expected=["42"],
                     expected_tools=["calculate"], tags=["math"],
                     description="the agent must use the calculator, not guess"),
            EvalCase(id="greeting", prompt="Say hello in one short sentence.",
                     tags=["chat"], expect_no_tools=True, max_chars=200,
                     description="chitchat must not call tools"),
            EvalCase(id="shape", prompt="Answer with a version string like 1.2.3.",
                     tags=["format"], expected_regex=r"\d+\.\d+\.\d+",
                     description="output must match a declared shape"),
        ],
    )
    if context.dry_run:
        context.printer.write(f"would write {len(dataset)} cases to {target}", min_level="quiet")
        return EXIT_OK
    written = dataset.save(target)
    context.printer.write(f"wrote {len(dataset)} cases to {written} "
                          f"({written.stat().st_size} bytes)", min_level="quiet")
    context.printer.write(context.printer.themed(
        "run them with: magent eval run --dataset " + str(written), "muted"), min_level="normal")
    return EXIT_OK


def _convert(args: argparse.Namespace, context: CliContext) -> int:
    """Convert a dataset between JSON and JSONL."""
    source = Path(str(getattr(args, "source", "") or "")).expanduser()
    destination = Path(str(getattr(args, "destination", "") or "")).expanduser()
    if not source.exists():
        context.printer.error(f"error: no such dataset: {source}")
        return EXIT_ERROR
    if destination.exists() and not context.assume_yes:
        from ..args import confirm

        if not confirm(f"overwrite {destination}?", context=context):
            context.printer.warn("cancelled")
            return EXIT_OK
    dataset = load_dataset(source)
    if context.dry_run:
        context.printer.write(f"would convert {len(dataset)} cases from {source.name} "
                              f"to {destination.name}", min_level="quiet")
        return EXIT_OK
    written = dataset.save(destination)
    context.printer.write(f"converted {len(dataset)} cases to {written}", min_level="quiet")
    return EXIT_OK


__all__ = ["cmd_eval", "register"]
