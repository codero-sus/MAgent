"""Quality commands: ``eval``, ``bench``, ``improvements``.

``eval`` runs a dataset of cases against an agent and fails the process when the
pass rate drops below a threshold - the shape a CI job needs.  ``bench`` measures
the backend itself (latency, tokens/s, cost) with no tools and no guardrails in
the way.  ``improvements`` verifies the machine-checkable manifest of what this
build actually contains, so the claim can be re-checked by anyone at any time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ...errors import EvalThresholdError
from ...utils.text import ellipsize
from ..commands import Command


def eval_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent eval`` - run an evaluation dataset or list what is available."""
    from ...eval import EvalRunner, load_dataset, metric_names, sample_dataset
    from ...eval.runner import REPORT_FORMATS

    action = getattr(args, "action", "") or "run"
    if action == "metrics":
        rows = [(name, "") for name in metric_names()]
        if ctx.config.output.format != "text":
            ctx.emit(list(metric_names()), fmt=ctx.config.output.format)
            return 0
        ctx.table(rows, ["metric"], title=f"{len(rows)} metrics")
        return 0
    if action == "sample":
        dataset = sample_dataset("sample")
        if args.output:
            from ...eval import save_dataset

            ctx.notice(f"wrote {save_dataset(dataset, args.output)}", level="success")
            return 0
        if ctx.config.output.format != "text":
            ctx.emit(dataset.to_dict(), fmt=ctx.config.output.format)
            return 0
        ctx.table(
            [(case.id, case.prompt, ",".join(case.tags)) for case in dataset],
            ["id", "prompt", "tags"],
            title="sample dataset",
        )
        return 0
    dataset = load_dataset(args.dataset) if args.dataset else sample_dataset("sample")
    if getattr(args, "tag", ""):
        dataset = dataset.filter(tag=args.tag)
    if getattr(args, "limit", 0):
        dataset = dataset.filter(limit=args.limit)
    if not len(dataset):
        ctx.error(f"dataset {dataset.name!r} has no cases after filtering")
        return 2
    preset = getattr(args, "preset", "") or ""
    factory = (lambda: ctx.build_preset_agent(preset)) if preset else (lambda: ctx.build_agent())
    runner = EvalRunner(
        factory,
        metrics=tuple(args.metrics or ()),
        repeats=args.repeats,
        concurrency=args.concurrency,
        threshold=args.threshold or 0.0,
        run_options={"max_iterations": args.max_iterations}
        if getattr(args, "max_iterations", 0)
        else None,
    )
    machine_format = bool(args.report_format) or ctx.config.output.format != "text"
    show_progress = not ctx.quiet and not machine_format
    report = runner.run(
        dataset,
        progress=(
            lambda done, total: (
                ctx.write(f"\r\033[Kevaluating {done}/{total}") if show_progress else None
            )
        ),
    )
    if show_progress:
        ctx.write("\r\033[K")
    fmt = args.report_format or (
        ctx.config.output.format if ctx.config.output.format != "text" else "text"
    )
    if args.output:
        path = report.save(args.output, fmt=fmt)
        ctx.notice(f"wrote {path}", level="success")
    if fmt == "text":
        ctx.print(report.render("text"))
    else:
        ctx.write(report.render(fmt if fmt in REPORT_FORMATS else "json"))
    failed = report.failures()
    if failed and fmt == "text":
        ctx.print("")
        for run in failed[:10]:
            ctx.notice(f"{run.case.id}: {run.score.summary()}", level="error")
    if args.threshold:
        try:
            report.check(args.threshold)
        except EvalThresholdError as exc:
            ctx.error(str(exc))
            return 1
    return 0 if report.ok else 1


def _price_book(config: Any) -> Any:
    """Build a :class:`~magent.llm.usage.PriceBook` from configuration.

    Prices live in ``[pricing.prompt_per_million]`` / ``[pricing.completion_per_million]``
    when the user configured them; an unknown model simply costs nothing, which
    is the honest answer for a local model running on the user's own hardware.
    """
    from ...llm.usage import PriceBook

    pricing = getattr(config, "extra", {}) or {}
    section = pricing.get("pricing") if isinstance(pricing, dict) else {}
    if not isinstance(section, dict):
        return PriceBook()
    prompt_rates = (
        {str(key): float(value) for key, value in (section.get("prompt_per_million") or {}).items()}
        if isinstance(section.get("prompt_per_million"), dict)
        else {}
    )
    completion_rates = (
        {
            str(key): float(value)
            for key, value in (section.get("completion_per_million") or {}).items()
        }
        if isinstance(section.get("completion_per_million"), dict)
        else {}
    )
    return PriceBook(
        prompt_per_million=prompt_rates,
        completion_per_million=completion_rates,
        default_prompt=float(section.get("default_prompt", 0.0) or 0.0),
        default_completion=float(section.get("default_completion", 0.0) or 0.0),
        currency=str(section.get("currency", "USD") or "USD"),
    )


def bench_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent bench`` - measure backend latency, throughput and cost."""
    from ...llm.base import CompletionRequest, Message
    from ...types import Usage

    backend = ctx.backend
    prompt = args.prompt or "Reply with a single short sentence about the weather."
    repeats = max(1, int(args.repeats or 3))
    latencies: list[float] = []
    prompt_tokens = completion_tokens = 0
    failures: list[str] = []
    started = time.perf_counter()
    for index in range(repeats):
        try:
            request = CompletionRequest(
                messages=[Message.user(f"{prompt} ({index + 1})")],
                model=getattr(backend, "default_model", "") or "",
                temperature=0.0,
                max_tokens=int(args.max_tokens or 64),
                timeout=float(getattr(ctx.config.server, "timeout", 120.0) or 120.0),
            )
            t0 = time.perf_counter()
            response = backend.complete(request)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            usage = getattr(response, "usage", None)
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started
    model = getattr(backend, "default_model", "") or getattr(backend, "name", "")
    prices = _price_book(ctx.config)
    estimate = prices.cost(
        model, Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )
    payload = {
        "backend": type(backend).__name__,
        "model": model,
        "repeats": repeats,
        "ok": len(latencies),
        "failed": len(failures),
        "seconds": round(elapsed, 4),
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 2) if latencies else 0.0,
        },
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
        "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed else 0.0,
        "cost": estimate.to_dict(),
        "errors": failures[:5],
    }
    if ctx.config.output.format != "text":
        ctx.emit(payload, fmt=ctx.config.output.format)
        return 0 if not failures else 1
    ctx.key_values(
        [
            ("backend", payload["backend"]),
            ("model", model),
            ("calls", f"{payload['ok']}/{repeats}"),
            ("p50", f"{payload['latency_ms']['p50']} ms"),
            ("mean", f"{payload['latency_ms']['mean']} ms"),
            ("tokens/s", payload["tokens_per_second"]),
            ("tokens", f"{prompt_tokens} prompt + {completion_tokens} completion"),
            ("cost", f"{estimate.total:.6f} {estimate.currency}"),
            ("wall clock", f"{elapsed:.2f}s"),
        ],
        title="benchmark",
    )
    if failures:
        ctx.notice(f"{len(failures)} call(s) failed: {failures[0]}", level="error")
    return 0 if not failures else 1


def improvements_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent improvements`` - verify the shipped improvements manifest."""
    from ...paths import improvements_manifest

    action = getattr(args, "action", "") or "verify"
    if args.manifest:
        manifest = Path(args.manifest).expanduser()
    else:
        candidate = improvements_manifest(root=ctx.workspace)
        if not candidate.is_file():
            # A project without a manifest of its own still ships this one, which
            # describes the installation rather than the project.
            candidate = improvements_manifest()
        manifest = candidate
    if not manifest.is_file():
        ctx.error(f"manifest not found at {manifest}; run `make improvements` first")
        return 1
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as exc:
        ctx.error(f"manifest is not valid JSON: {exc}")
        return 1
    entries = payload.get("improvements", payload if isinstance(payload, list) else [])
    if action == "count":
        if ctx.config.output.format != "text":
            ctx.emit({"count": len(entries)}, fmt=ctx.config.output.format)
            return 0
        ctx.print(str(len(entries)))
        return 0
    if action == "list":
        limit = args.limit or 40
        category = getattr(args, "category", "") or ""
        chosen = [item for item in entries if not category or item.get("category") == category]
        if ctx.config.output.format != "text":
            ctx.emit(chosen[:limit], fmt=ctx.config.output.format)
            return 0
        rows = [
            (
                str(item.get("id", "")),
                str(item.get("category", "")),
                ellipsize(str(item.get("title", "")), 60),
                str(item.get("ref", "")),
            )
            for item in chosen[:limit]
        ]
        ctx.table(rows, ["id", "category", "title", "ref"], title=f"{len(chosen)} improvements")
        return 0
    verified = skipped = 0
    failures: list[str] = []
    for item in entries:
        ref = str(item.get("ref", ""))
        if not ref:
            skipped += 1
            continue
        path, _, symbol = ref.partition("#")
        target = ctx.workspace / path if not Path(path).is_absolute() else Path(path)
        if not target.is_file():
            failures.append(f"{item.get('id')}: missing file {path}")
            continue
        if symbol:
            text = target.read_text(encoding="utf-8", errors="replace")
            if symbol not in text:
                failures.append(f"{item.get('id')}: {symbol} not found in {path}")
                continue
        verified += 1
    summary = {
        "manifest": str(manifest),
        "total": len(entries),
        "verified": verified,
        "skipped": skipped,
        "failures": failures[:20],
        "ok": not failures,
    }
    if ctx.config.output.format != "text":
        ctx.emit(summary, fmt=ctx.config.output.format)
        return 0 if not failures else 1
    ctx.key_values(
        [
            ("manifest", manifest),
            ("total", len(entries)),
            ("verified", verified),
            ("skipped", skipped),
            ("failures", len(failures)),
        ],
        title="improvements",
    )
    for line in failures[:10]:
        ctx.notice(line, level="error")
    if not failures:
        ctx.notice(f"all {verified} improvements verified against the repository", level="success")
    return 0 if not failures else 1


def prompts_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent prompts`` - list prompt templates and personas."""
    from ...prompts import PERSONAS, TEMPLATES, get_template, list_templates, template_categories

    action = getattr(args, "action", "") or "templates"
    if action == "personas":
        name = getattr(args, "name", "") or ""
        if name:
            if name not in PERSONAS:
                ctx.error(f"unknown persona {name!r}")
                return 1
            from ...prompts import persona_prompt

            ctx.write(persona_prompt(name) + "\n")
            return 0
        rows = [
            (persona.name, ellipsize(persona.role, 40), ellipsize(persona.tone, 30))
            for persona in PERSONAS.values()
        ]
        ctx.table(rows, ["persona", "role", "tone"], title=f"{len(rows)} personas")
        return 0
    name = getattr(args, "name", "") or ""
    if name:
        try:
            template = get_template(name)
        except KeyError as exc:
            ctx.error(str(exc))
            return 1
        if ctx.config.output.format != "text":
            ctx.emit(
                {
                    "name": template.name,
                    "category": template.category,
                    "description": template.description,
                    "variables": list(template.variables),
                    "text": template.text,
                },
                fmt=ctx.config.output.format,
            )
            return 0
        ctx.key_values(
            [
                ("template", template.name),
                ("category", template.category),
                ("variables", ", ".join(template.variables) or "-"),
                ("chars", len(template.text)),
            ],
            title=f"template {template.name}",
        )
        ctx.print("")
        ctx.write(template.text.rstrip() + "\n")
        return 0
    rows = [
        (
            item["name"],
            item["category"],
            ", ".join(item["variables"]),
            ellipsize(item["description"], 48),
        )
        for item in list_templates()
    ]
    ctx.table(
        rows,
        ["template", "category", "variables", "description"],
        title=f"{len(TEMPLATES)} templates in {len(template_categories())} categories",
    )
    return 0


def _configure_eval(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent eval``."""
    parser.add_argument(
        "action", nargs="?", default="run", choices=["run", "metrics", "sample"], help="what to do"
    )
    parser.add_argument("--dataset", default="", help="dataset file (.json/.jsonl/.md/.py)")
    parser.add_argument("--metrics", action="append", default=None, help="metric (repeatable)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per case")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel cases")
    parser.add_argument("--threshold", type=float, default=0.0, help="minimum pass rate")
    parser.add_argument("--tag", default="", help="only cases with this tag")
    parser.add_argument("--limit", type=int, default=0, help="maximum cases")
    parser.add_argument("--preset", default="", help="agent preset to evaluate")
    parser.add_argument(
        "--max-iterations",
        dest="max_iterations",
        type=int,
        default=0,
        help="ceiling on steps per case",
    )
    parser.add_argument(
        "--report-format",
        dest="report_format",
        default="",
        help="report format: text|json|markdown|csv|junit",
    )
    parser.add_argument("--output", default="", help="write the report here")


def _configure_bench(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent bench``."""
    parser.add_argument("--prompt", default="", help="prompt to send")
    parser.add_argument("--repeats", type=int, default=3, help="how many calls")
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=64, help="tokens per call"
    )


def _configure_improvements(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent improvements``."""
    parser.add_argument(
        "action",
        nargs="?",
        default="verify",
        choices=["verify", "count", "list"],
        help="what to do",
    )
    parser.add_argument("--manifest", default="", help="path to the manifest JSON")
    parser.add_argument("--category", default="", help="filter by category")
    parser.add_argument("--limit", type=int, default=40, help="rows to show")


def _configure_prompts(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent prompts``."""
    parser.add_argument(
        "action",
        nargs="?",
        default="templates",
        choices=["templates", "personas"],
        help="what to list",
    )
    parser.add_argument("--name", default="", help="template or persona to render")


COMMANDS: tuple[Command, ...] = (
    Command(
        name="eval",
        help="Run an evaluation dataset",
        run=eval_command,
        group="quality",
        needs_backend=True,
        configure=_configure_eval,
        description=(
            "Score an agent against a dataset of cases with deterministic metrics "
            "(substrings, regex, JSON, tool usage, budgets) and exit non-zero below a "
            "threshold, so it can gate a build."
        ),
        examples=(
            "magent eval",
            "magent eval --dataset evals/smoke.jsonl --threshold 0.9",
            "magent eval --dataset evals/retrieval.md --repeats 3 --concurrency 4",
        ),
    ),
    Command(
        name="bench",
        help="Measure backend latency, throughput and cost",
        run=bench_command,
        group="quality",
        needs_backend=True,
        configure=_configure_bench,
        examples=("magent bench", "magent bench --repeats 10 --max-tokens 256"),
    ),
    Command(
        name="improvements",
        help="Verify the improvements manifest",
        run=improvements_command,
        group="quality",
        configure=_configure_improvements,
        description=(
            "Every claimed improvement carries a reference to a real file and symbol. "
            "`verify` re-checks each reference against the checkout and fails when one "
            "no longer exists."
        ),
        examples=(
            "magent improvements",
            "magent improvements count",
            "magent improvements list --category tools",
        ),
    ),
    Command(
        name="prompts",
        help="List prompt templates and personas",
        run=prompts_command,
        group="tools",
        configure=_configure_prompts,
        examples=(
            "magent prompts",
            "magent prompts personas coder",
            "magent prompts templates summarize",
        ),
    ),
)


__all__ = [
    "COMMANDS",
    "bench_command",
    "eval_command",
    "improvements_command",
    "prompts_command",
]
