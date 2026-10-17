"""Evaluation runs: execute a dataset and report what passed.

:class:`EvalRunner` takes an agent factory (a callable returning a fresh agent)
rather than an agent, because a shared agent carries conversation state into the
next case and quietly changes what is being measured.  Cases run in parallel
with :mod:`magent.concurrency`, repeats let a flaky agent show its variance, and
:class:`EvalReport` turns the scores into a table, JSON, Markdown or a JUnit-style
summary for CI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..concurrency.pool import CancellationToken, Task, WorkerPool
from ..errors import EvalThresholdError
from ..logging_ import get_logger
from ..utils.files import ensure_dir
from ..utils.text import ellipsize
from ..utils.timeutil import format_duration
from .dataset import Dataset, EvalCase
from .metrics import CaseScore, MetricOutcome, aggregate, score_case, weighted_pass_rate

LOGGER = get_logger("eval.runner")
"""Logger for evaluation runs."""

REPORT_FORMATS: tuple[str, ...] = ("text", "json", "markdown", "csv", "junit")
"""Report renderings :class:`EvalReport` can produce."""


@dataclass(slots=True)
class CaseRun:
    """One case, scored, with the numbers a report needs."""

    case: EvalCase
    score: CaseScore
    repeat: int = 0
    seconds: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    model: str = ""
    stop_reason: str = ""

    def to_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        """Serialise the case run."""
        payload = {
            "case": self.case.id,
            "repeat": self.repeat,
            "prompt": ellipsize(self.case.prompt, 200),
            "score": self.score.score,
            "passed": self.score.passed,
            "seconds": round(self.seconds, 4),
            "tokens": self.tokens,
            "cost": round(self.cost, 6),
            "model": self.model,
            "stop_reason": self.stop_reason,
            "failures": [item.to_dict() for item in self.score.failures],
        }
        if include_output:
            payload["output"] = self.score.output
        return payload


@dataclass(slots=True)
class EvalReport:
    """Aggregate outcome of an evaluation run."""

    dataset: str = ""
    runs: list[CaseRun] = field(default_factory=list)
    seconds: float = 0.0
    started_at: float = 0.0
    metrics: tuple[str, ...] = ()
    repeats: int = 1
    threshold: float = 0.0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> list[CaseScore]:
        """One score per case run."""
        return [run.score for run in self.runs]

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregated statistics."""
        stats = aggregate(self.scores)
        stats.update({
            "dataset": self.dataset,
            "seconds": round(self.seconds, 4),
            "repeats": self.repeats,
            "weighted_pass_rate": weighted_pass_rate(self.scores),
            "tokens": sum(run.tokens for run in self.runs),
            "cost": round(sum(run.cost for run in self.runs), 6),
            "threshold": self.threshold,
            "model": self.model,
        })
        return stats

    @property
    def pass_rate(self) -> float:
        """Fraction of case runs that passed."""
        return float(self.summary["pass_rate"])

    @property
    def ok(self) -> bool:
        """True when the pass rate meets the configured threshold."""
        return self.pass_rate >= self.threshold if self.threshold else self.pass_rate >= 0.999

    def failures(self) -> list[CaseRun]:
        """Case runs that did not pass."""
        return [run for run in self.runs if not run.score.passed]

    def by_case(self) -> dict[str, list[CaseRun]]:
        """Runs grouped by case id, for repeat analysis."""
        grouped: dict[str, list[CaseRun]] = {}
        for run in self.runs:
            grouped.setdefault(run.case.id, []).append(run)
        return grouped

    def flaky(self, *, min_runs: int = 2) -> list[str]:
        """Case ids that both passed and failed across repeats."""
        out: list[str] = []
        for case_id, runs in self.by_case().items():
            if len(runs) < min_runs:
                continue
            passed = {run.score.passed for run in runs}
            if len(passed) > 1:
                out.append(case_id)
        return sorted(out)

    def metric_table(self) -> list[tuple[str, str, str]]:
        """Rows of ``(metric, mean score, cases)``."""
        rows: list[tuple[str, str, str]] = []
        for name, value in self.summary["metrics"].items():
            rows.append((name, f"{value:.3f}", str(len(self.runs))))
        return rows

    def case_table(self, *, limit: int = 0) -> list[tuple[str, str, str, str]]:
        """Rows of ``(case, score, status, first failure)``."""
        rows: list[tuple[str, str, str, str]] = []
        for run in self.runs[: limit or len(self.runs)]:
            failures = run.score.failures
            reason = failures[0].reason if failures else "ok"
            rows.append((run.case.id, f"{run.score.score:.2f}",
                         "pass" if run.score.passed else "fail", ellipsize(reason, 70)))
        return rows

    def to_dict(self, *, include_outputs: bool = True) -> dict[str, Any]:
        """Serialise the report."""
        return {
            "dataset": self.dataset,
            "summary": self.summary,
            "case_runs": [run.to_dict(include_output=include_outputs) for run in self.runs],
            "failures": [run.case.id for run in self.failures()],
            "flaky": self.flaky(),
            "metrics": list(self.metrics),
            "started_at": self.started_at,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2, include_outputs: bool = False) -> str:
        """Serialise as JSON."""
        return json.dumps(self.to_dict(include_outputs=include_outputs), indent=indent,
                          default=str, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render a Markdown report with a summary and a per-case table."""
        info = self.summary
        lines = [
            f"# Evaluation: {self.dataset}",
            "",
            f"- cases: {info['cases']} (repeats: {self.repeats})",
            f"- passed: {info['passed']} / {info['cases']} ({info['pass_rate']:.1%})",
            f"- mean score: {info['score']:.3f}",
            f"- weighted pass rate: {info['weighted_pass_rate']:.3f}",
            f"- tokens: {info['tokens']}, cost: {info['cost']:.4f}",
            f"- wall clock: {format_duration(self.seconds)}",
            f"- model: {self.model or 'n/a'}",
            "",
            "| case | score | status | note |",
            "| --- | --- | --- | --- |",
        ]
        for case_id, score, status, reason in self.case_table():
            lines.append(f"| {case_id} | {score} | {status} | {reason} |")
        if info["metrics"]:
            lines.extend(["", "| metric | mean |", "| --- | --- |"])
            for name, value in info["metrics"].items():
                lines.append(f"| {name} | {value:.3f} |")
        flaky = self.flaky()
        if flaky:
            lines.extend(["", f"Flaky cases: {', '.join(flaky)}"])
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        """Render one row per case run."""
        from ..output.format import to_csv

        return to_csv([{key: value for key, value in run.to_dict(include_output=False).items()
                        if key != "failures"} for run in self.runs])

    def to_junit(self) -> str:
        """Render a JUnit XML report, for CI systems that consume it."""
        from xml.sax.saxutils import escape

        failures = sum(1 for run in self.runs if not run.score.passed)
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{escape(self.dataset)}" tests="{len(self.runs)}" '
            f'failures="{failures}" time="{self.seconds:.3f}">',
        ]
        for run in self.runs:
            name = escape(f"{run.case.id}" + (f"#{run.repeat}" if run.repeat else ""))
            lines.append(f'  <testcase classname="{escape(self.dataset)}" name="{name}" '
                         f'time="{run.seconds:.3f}">')
            if not run.score.passed:
                reason = run.score.failures[0].reason if run.score.failures else "failed"
                lines.append(f'    <failure message="{escape(reason)}">'
                             f'{escape(run.score.output[:2_000])}</failure>')
            lines.append("  </testcase>")
        lines.append("</testsuite>")
        return "\n".join(lines) + "\n"

    def render(self, fmt: str = "text") -> str:
        """Render the report in one of :data:`REPORT_FORMATS`."""
        active = (fmt or "text").strip().lower()
        if active == "json":
            return self.to_json()
        if active == "markdown":
            return self.to_markdown()
        if active == "csv":
            return self.to_csv()
        if active == "junit":
            return self.to_junit()
        info = self.summary
        lines = [
            f"dataset {self.dataset}: {info['passed']}/{info['cases']} passed "
            f"({info['pass_rate']:.1%}), mean score {info['score']:.3f}, "
            f"{format_duration(self.seconds)}",
        ]
        for case_id, score, status, reason in self.case_table():
            lines.append(f"  {status:4} {case_id:<24} {score:>5}  {reason}")
        if info["metrics"]:
            lines.append("  metrics: " + ", ".join(f"{name}={value:.3f}"
                                                   for name, value in info["metrics"].items()))
        flaky = self.flaky()
        if flaky:
            lines.append("  flaky: " + ", ".join(flaky))
        return "\n".join(lines)

    def save(self, path: str | Path, *, fmt: str = "") -> Path:
        """Write the report to a file, inferring the format from the extension."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        active = fmt or {".json": "json", ".md": "markdown", ".markdown": "markdown",
                         ".csv": "csv", ".xml": "junit"}.get(target.suffix.lower(), "text")
        target.write_text(self.render(active), encoding="utf-8")
        return target

    def check(self, threshold: float = 0.0) -> EvalReport:
        """Raise when the pass rate is below ``threshold``.

        Raises:
            EvalThresholdError: when the run scores below the threshold
        """
        limit = threshold or self.threshold
        if limit and self.pass_rate < limit:
            raise EvalThresholdError(
                f"pass rate {self.pass_rate:.1%} is below the {limit:.1%} threshold",
                details={"dataset": self.dataset, "failed": [run.case.id for run in self.failures()],
                         "pass_rate": self.pass_rate, "threshold": limit},
            )
        return self

    def __len__(self) -> int:
        """Number of case runs."""
        return len(self.runs)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (f"EvalReport(dataset={self.dataset!r}, runs={len(self.runs)}, "
                f"pass_rate={self.pass_rate:.3f})")


class EvalRunner:
    """Run a dataset against an agent factory."""

    def __init__(self, agent_factory: Callable[[], Any], *, metrics: Sequence[str] = (),
                 repeats: int = 1, concurrency: int = 1, threshold: float = 0.0,
                 token: CancellationToken | None = None, on_case: Callable[[CaseRun], None] | None = None,
                 run_options: Mapping[str, Any] | None = None,
                 weights: Mapping[str, float] | None = None) -> None:
        """Configure the runner.

        Args:
            agent_factory: callable returning a fresh agent for each case
            metrics: metric names to apply
            repeats: how many times to run each case
            concurrency: parallel cases; 1 runs sequentially
            threshold: pass-rate threshold checked by :meth:`EvalRunner.check`
            token: cancellation token shared with the pool
            on_case: callback invoked after each case run
            run_options: extra options passed to ``agent.run``
            weights: per-metric weights
        """
        self.agent_factory = agent_factory
        self.metrics = tuple(metrics)
        self.repeats = max(1, int(repeats))
        self.concurrency = max(1, int(concurrency))
        self.threshold = max(0.0, float(threshold))
        self.token = token
        self.on_case = on_case
        self.run_options = dict(run_options or {})
        self.weights = dict(weights or {})
        self.cases_run = 0
        self.errors = 0

    def run(self, dataset: Dataset, *, progress: Callable[[int, int], None] | None = None) -> EvalReport:
        """Run a dataset and return the report.

        Args:
            dataset: cases to run
            progress: optional ``(done, total)`` callback
        """
        started_at = time.time()
        started = time.perf_counter()
        jobs: list[tuple[EvalCase, int]] = [(case, repeat) for case in dataset.cases
                                            for repeat in range(self.repeats)]
        report = EvalReport(dataset=dataset.name, metrics=self.metrics, repeats=self.repeats,
                            threshold=self.threshold, started_at=started_at,
                            metadata=dict(dataset.defaults))
        if not jobs:
            report.seconds = time.perf_counter() - started
            return report
        if self.concurrency <= 1 or len(jobs) == 1:
            for index, (case, repeat) in enumerate(jobs, start=1):
                report.runs.append(self._run_case(case, repeat))
                if progress is not None:
                    progress(index, len(jobs))
        else:
            tasks = [Task(name=f"{case.id}#{repeat}", fn=self._run_case, args=(case, repeat),
                          timeout=case.max_seconds * 4 if case.max_seconds else 900.0)
                     for case, repeat in jobs]
            with WorkerPool(min(self.concurrency, len(tasks)), name="eval", token=self.token) as pool:
                outcomes = pool.run_many(tasks)
            for (case, repeat), outcome in zip(jobs, outcomes, strict=False):
                if outcome.ok and isinstance(outcome.value, CaseRun):
                    report.runs.append(outcome.value)
                else:
                    self.errors += 1
                    report.runs.append(CaseRun(
                        case=case, repeat=repeat,
                        score=CaseScore(case_id=case.id, error=outcome.error or "task failed",
                                        tags=case.tags, weight=case.weight),
                    ))
        report.seconds = time.perf_counter() - started
        report.model = next((run.model for run in report.runs if run.model), "")
        LOGGER.debug("eval %s: %s", dataset.name, report.summary)
        return report

    def _run_case(self, case: EvalCase, repeat: int) -> CaseRun:
        """Execute one case against a fresh agent and score it."""
        started = time.perf_counter()
        try:
            agent = self.agent_factory()
        except Exception as exc:  # noqa: BLE001 - a broken factory fails the case, not the run
            self.errors += 1
            return CaseRun(case=case, repeat=repeat, seconds=time.perf_counter() - started,
                           score=CaseScore(case_id=case.id, tags=case.tags, weight=case.weight,
                                           error=f"agent factory failed: {exc}"))
        try:
            result = agent.run(case.prompt, **self.run_options)
        except Exception as exc:  # noqa: BLE001 - the failure is the measurement
            self.errors += 1
            return CaseRun(case=case, repeat=repeat, seconds=time.perf_counter() - started,
                           score=CaseScore(case_id=case.id, tags=case.tags, weight=case.weight,
                                           error=f"{type(exc).__name__}: {exc}"))
        score = score_case(case, result, metrics=self.metrics, weights=self.weights)
        usage = getattr(result, "usage", None)
        run = CaseRun(
            case=case, repeat=repeat, score=score, seconds=time.perf_counter() - started,
            tokens=getattr(usage, "total_tokens", 0) or 0,
            cost=float(getattr(getattr(result, "cost", None), "total", 0.0) or 0.0),
            model=str(getattr(result, "model", "") or ""),
            stop_reason=str(getattr(getattr(result, "stop_reason", ""), "value",
                                    getattr(result, "stop_reason", "")) or ""),
        )
        self.cases_run += 1
        if self.on_case is not None:
            try:
                self.on_case(run)
            except Exception as exc:  # noqa: BLE001 - callbacks must not break the run
                LOGGER.debug("on_case callback failed: %s", exc)
        return run

    def check(self, dataset: Dataset, *, threshold: float = 0.0) -> EvalReport:
        """Run a dataset and raise when the pass rate is below the threshold."""
        report = self.run(dataset)
        report.check(threshold or self.threshold)
        return report

    def describe(self) -> dict[str, Any]:
        """Describe the runner configuration."""
        return {
            "metrics": list(self.metrics),
            "repeats": self.repeats,
            "concurrency": self.concurrency,
            "threshold": self.threshold,
            "cases_run": self.cases_run,
            "errors": self.errors,
            "weights": dict(self.weights),
        }

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (f"EvalRunner(repeats={self.repeats}, concurrency={self.concurrency}, "
                f"metrics={list(self.metrics)})")


def run_eval(dataset: Dataset, agent_factory: Callable[[], Any], *, metrics: Sequence[str] = (),
             repeats: int = 1, concurrency: int = 1, threshold: float = 0.0,
             **kwargs: Any) -> EvalReport:
    """Run a dataset and return the report in one call."""
    runner = EvalRunner(agent_factory, metrics=metrics, repeats=repeats,
                        concurrency=concurrency, threshold=threshold, **kwargs)
    return runner.run(dataset)


def metrics_for_names(names: Iterable[str]) -> list[Callable[[EvalCase, Any], MetricOutcome]]:
    """Resolve metric names to callables, raising on an unknown name."""
    from .metrics import get_metric

    return [get_metric(name) for name in names]


__all__ = [
    "REPORT_FORMATS",
    "CaseRun",
    "EvalReport",
    "EvalRunner",
    "metrics_for_names",
    "run_eval",
]
