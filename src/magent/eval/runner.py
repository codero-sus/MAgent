"""Running an evaluation dataset and reporting what happened.

The runner executes each case against a real agent - not a mock of one - so a
regression in tool dispatch, context handling, or guardrails shows up as a
failed case rather than as a passing unit test over a stub.

Three properties matter and are implemented here rather than left to callers:

*isolation*
    each case gets a fresh agent state, so a conversation in case 5 cannot
    change the answer to case 6;

*repeatability*
    ``repeats`` runs a case several times and scores the mean, because a
    single sample of a stochastic model proves nothing;

*honest reporting*
    a case that errors is a failure with the exception recorded, not a skipped
    row, and the report separates "scored low" from "never ran".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..concurrency import parallel_map
from ..errors import EvalError, EvalThresholdError
from ..logging_ import get_logger
from ..utils.files import atomic_write_text, ensure_dir
from ..utils.tables import render_key_value, render_markdown_table, render_table
from ..utils.timeutil import format_duration
from .cases import EvalCase, EvalDataset, resolve_dataset
from .metrics import METRICS, MetricScore, aggregate, score as score_case

LOGGER = get_logger("eval.runner")
"""Logger for evaluation runs."""

DEFAULT_CONCURRENCY = 4
"""Cases run at once when the caller does not say."""

DEFAULT_PASS_RATE = 0.0
"""Aggregate pass rate below which a run fails, when unset."""


@dataclass(slots=True)
class CaseOutcome:
    """What one execution of one case produced."""

    output: str = ""
    stop_reason: str = ""
    tools_called: list[str] = field(default_factory=list)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    duration_seconds: float = 0.0
    iterations: int = 0
    structured: Any = None
    citations: list[Any] = field(default_factory=list)
    error: str = ""
    error_code: str = ""
    events: int = 0
    run_id: str = ""
    model: str = ""
    repeat_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the run completed without an error."""
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "output": self.output[:2000],
            "stop_reason": self.stop_reason,
            "tools_called": self.tools_called,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 6),
            "duration_seconds": round(self.duration_seconds, 4),
            "iterations": self.iterations,
            "structured": self.structured,
            "citations": len(self.citations),
            "error": self.error,
            "error_code": self.error_code,
            "run_id": self.run_id,
            "model": self.model,
            "repeat": self.repeat_index,
        }

    @classmethod
    def from_run(cls, result: Any, *, duration_seconds: float = 0.0,
                 repeat_index: int = 0, events: int = 0) -> CaseOutcome:
        """Build an outcome from a :class:`~magent.types.RunResult`."""
        usage = getattr(result, "usage", None)
        cost = getattr(result, "cost", None)
        tools: list[str] = []
        for step in getattr(result, "steps", []) or []:
            for call in getattr(step, "tool_calls", []) or []:
                name = getattr(call, "name", "") or ""
                if name:
                    tools.append(str(name))
        citations = list((getattr(result, "metadata", None) or {}).get("citations") or [])
        return cls(
            output=str(getattr(result, "output", "") or ""),
            stop_reason=str(getattr(getattr(result, "stop_reason", ""), "value",
                                    getattr(result, "stop_reason", "")) or ""),
            tools_called=tools,
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost=float(getattr(cost, "total", 0.0) or 0.0),
            duration_seconds=duration_seconds or
            float(getattr(result, "duration_ms", 0.0) or 0.0) / 1000.0,
            iterations=int(getattr(result, "iterations", 0) or 0),
            structured=getattr(result, "structured", None),
            citations=citations,
            error=str(getattr(result, "error", "") or ""),
            run_id=str(getattr(result, "run_id", "") or ""),
            model=str(getattr(result, "model", "") or ""),
            repeat_index=repeat_index,
            events=events,
        )


@dataclass(slots=True)
class CaseResult:
    """One case's outcome, its metric scores, and the verdict."""

    case_id: str = ""
    prompt: str = ""
    tags: list[str] = field(default_factory=list)
    outcome: CaseOutcome = field(default_factory=CaseOutcome)
    scores: list[MetricScore] = field(default_factory=list)
    aggregate: float = 0.0
    threshold: float = 0.6
    passed: bool = False
    repeats: int = 1
    attempts: list[CaseOutcome] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str = ""
    weight: float = 1.0
    blocked_by: list[str] = field(default_factory=list)

    @property
    def failed_metrics(self) -> list[MetricScore]:
        """Metrics that did not pass."""
        return [item for item in self.scores if not item.passed]

    @property
    def required_failures(self) -> list[MetricScore]:
        """Required metrics that did not pass, which gate the verdict."""
        return [item for item in self.scores if item.required and not item.passed]

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "id": self.case_id,
            "prompt": self.prompt[:300],
            "tags": self.tags,
            "passed": self.passed,
            "aggregate": round(self.aggregate, 4),
            "threshold": self.threshold,
            "weight": self.weight,
            "repeats": self.repeats,
            "duration_seconds": round(self.duration_seconds, 4),
            "error": self.error,
            "blocked_by": self.blocked_by,
            "metrics": [item.to_dict() for item in self.scores],
            "outcome": self.outcome.to_dict(),
        }

    def describe(self, *, width: int = 100) -> str:
        """Return one or two lines describing this result."""
        mark = "PASS" if self.passed else "FAIL"
        head = (f"{mark} {self.case_id:<24} {self.aggregate:.2f} "
                f"(threshold {self.threshold:.2f}) {format_duration(self.duration_seconds)}")
        lines = [head[: max(40, width)]]
        failures = self.failed_metrics
        if self.blocked_by:
            lines.append(f"      required metric(s) failed: {', '.join(self.blocked_by)}"
                         [: max(40, width)])
        if failures:
            detail = "; ".join(str(item) for item in failures[:3])
            lines.append(f"      {detail}"[: max(40, width)])
        if self.error:
            lines.append(f"      error: {self.error}"[: max(40, width)])
        return "\n".join(lines)


@dataclass(slots=True)
class EvalReport:
    """The result of running a whole dataset."""

    dataset: str = ""
    results: list[CaseResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    model: str = ""
    backend: str = ""
    fail_below: float = DEFAULT_PASS_RATE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Number of cases run."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of cases that passed."""
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        """Number of cases that failed."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """Weighted mean aggregate score."""
        total_weight = sum(result.weight for result in self.results)
        if total_weight <= 0:
            return 0.0
        return sum(result.aggregate * result.weight for result in self.results) / total_weight

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the whole run."""
        if self.finished_at and self.started_at:
            return max(0.0, self.finished_at - self.started_at)
        return sum(result.duration_seconds for result in self.results)

    @property
    def ok(self) -> bool:
        """Whether the run met its required pass rate."""
        return self.total > 0 and self.pass_rate >= self.fail_below

    def errors(self) -> list[CaseResult]:
        """Results whose run raised or errored."""
        return [result for result in self.results if result.error]

    def by_tag(self) -> dict[str, dict[str, Any]]:
        """Group pass rates by tag."""
        grouped: dict[str, dict[str, Any]] = {}
        for result in self.results:
            for tag in result.tags or ["(untagged)"]:
                record = grouped.setdefault(tag, {"total": 0, "passed": 0, "score": 0.0})
                record["total"] += 1
                record["score"] += result.aggregate
                if result.passed:
                    record["passed"] += 1
        for record in grouped.values():
            record["pass_rate"] = round(record["passed"] / record["total"], 4) if record["total"] else 0.0
            record["mean_score"] = round(record["score"] / record["total"], 4) if record["total"] else 0.0
            record.pop("score", None)
        return dict(sorted(grouped.items(), key=lambda kv: (-kv[1]["total"], kv[0])))

    def by_metric(self) -> dict[str, dict[str, Any]]:
        """Group results by metric name, showing which metrics fail most."""
        grouped: dict[str, dict[str, Any]] = {}
        for result in self.results:
            for metric in result.scores:
                record = grouped.setdefault(metric.name, {"runs": 0, "passed": 0, "score": 0.0})
                record["runs"] += 1
                record["score"] += metric.score
                if metric.passed:
                    record["passed"] += 1
        for record in grouped.values():
            record["pass_rate"] = round(record["passed"] / record["runs"], 4) if record["runs"] else 0.0
            record["mean_score"] = round(record["score"] / record["runs"], 4) if record["runs"] else 0.0
            record.pop("score", None)
        return dict(sorted(grouped.items(), key=lambda kv: (kv[1]["pass_rate"], -kv[1]["runs"])))

    def totals(self) -> dict[str, Any]:
        """Return aggregate cost and token accounting."""
        return {
            "tokens": sum(result.outcome.total_tokens for result in self.results),
            "prompt_tokens": sum(result.outcome.prompt_tokens for result in self.results),
            "completion_tokens": sum(result.outcome.completion_tokens for result in self.results),
            "cost": round(sum(result.outcome.cost for result in self.results), 6),
            "tool_calls": sum(len(result.outcome.tools_called) for result in self.results),
            "iterations": sum(result.outcome.iterations for result in self.results),
        }

    def summary(self) -> dict[str, Any]:
        """Return the report's headline numbers."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "backend": self.backend,
            "cases": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "fail_below": self.fail_below,
            "ok": self.ok,
            "errors": len(self.errors()),
            "duration_seconds": round(self.duration_seconds, 3),
            **self.totals(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the fully serialized report."""
        return {
            **self.summary(),
            "by_tag": self.by_tag(),
            "by_metric": self.by_metric(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
            "results": [result.to_dict() for result in self.results],
        }

    def render(self, fmt: str = "markdown", *, width: int = 100,
               failures_only: bool = False) -> str:
        """Render the report as Markdown, text, or JSON."""
        if fmt == "json":
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        if fmt == "text":
            return self._render_text(width=width, failures_only=failures_only)
        return self._render_markdown(failures_only=failures_only)

    def _render_markdown(self, *, failures_only: bool = False) -> str:
        """Render a Markdown report suitable for a pull request comment."""
        shown = [result for result in self.results if not (failures_only and result.passed)]
        lines = [f"# Evaluation: {self.dataset}", ""]
        lines.append(render_key_value([
            ("cases", self.total),
            ("passed", self.passed),
            ("failed", self.failed),
            ("pass rate", f"{self.pass_rate:.1%}"),
            ("mean score", f"{self.mean_score:.3f}"),
            ("required", f"{self.fail_below:.1%}"),
            ("verdict", "**PASS**" if self.ok else "**FAIL**"),
            ("duration", format_duration(self.duration_seconds)),
            ("backend", self.backend or "-"),
            ("model", self.model or "-"),
        ]))
        lines.append("")
        totals = self.totals()
        lines.append(f"Tokens {totals['tokens']} "
                     f"({totals['prompt_tokens']}+{totals['completion_tokens']}) · "
                     f"tool calls {totals['tool_calls']} · cost ${totals['cost']:.4f}")
        lines.append("")
        lines.append("## Cases")
        lines.append("")
        rows = [[result.case_id, "pass" if result.passed else "**FAIL**",
                 f"{result.aggregate:.2f}", f"{result.threshold:.2f}",
                 format_duration(result.duration_seconds),
                 ", ".join(result.outcome.tools_called[:3]) or "-",
                 (result.error or "")[:40]] for result in shown]
        lines.append(render_markdown_table(
            rows, headers=["id", "verdict", "score", "min", "time", "tools", "error"]))
        lines.append("")
        by_metric = self.by_metric()
        if by_metric:
            lines.append("## Metrics")
            lines.append("")
            lines.append(render_markdown_table(
                [[name, record["runs"], f"{record['pass_rate']:.1%}",
                  f"{record['mean_score']:.3f}"] for name, record in by_metric.items()],
                headers=["metric", "runs", "pass rate", "mean"]))
            lines.append("")
        failures = [result for result in shown if not result.passed]
        if failures:
            lines.append("## Failures")
            lines.append("")
            for result in failures:
                lines.append(f"### {result.case_id}")
                lines.append("")
                lines.append(f"- prompt: {result.prompt[:200]}")
                lines.append(f"- score: {result.aggregate:.3f} (threshold {result.threshold:.2f})")
                if result.error:
                    lines.append(f"- error: {result.error}")
                for metric in result.failed_metrics:
                    lines.append(f"- {metric.name}: {metric.score:.2f} - {metric.detail}")
                answer = " ".join((result.outcome.output or "").split())
                if answer:
                    lines.append(f"- answer: {answer[:300]}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_text(self, *, width: int = 100, failures_only: bool = False) -> str:
        """Render a terminal report."""
        lines = [f"evaluation: {self.dataset}", "-" * min(width, 72)]
        for result in self.results:
            if failures_only and result.passed:
                continue
            lines.append(result.describe(width=width))
        lines.append("-" * min(width, 72))
        lines.append(f"{self.passed}/{self.total} passed ({self.pass_rate:.1%}), "
                     f"mean score {self.mean_score:.3f}, required {self.fail_below:.1%} -> "
                     f"{'PASS' if self.ok else 'FAIL'}")
        lines.append(f"duration {format_duration(self.duration_seconds)}")
        return "\n".join(lines)

    def save(self, path: str | Path, *, fmt: str = "markdown",
             failures_only: bool = False) -> Path:
        """Write the report to a file."""
        target = Path(path).expanduser()
        ensure_dir(target.parent)
        active = fmt if fmt in {"markdown", "text", "json"} else "markdown"
        if target.suffix == ".json":
            active = "json"
        atomic_write_text(target, self.render(active, failures_only=failures_only))
        return target


class EvalRunner:
    """Executes a dataset against an agent and scores every case."""

    def __init__(self, agent_factory: Callable[[], Any] | None = None, *, agent: Any = None,
                 concurrency: int = DEFAULT_CONCURRENCY, repeats: int = 1,
                 metrics: Sequence[str] = (), fail_below: float = DEFAULT_PASS_RATE,
                 on_progress: Callable[[CaseResult, int, int], None] | None = None,
                 backend: Any = None, model: str = "", reset_between_cases: bool = True,
                 timeout: float = 0.0) -> None:
        """Configure a runner.

        Args:
            agent_factory: builds a fresh agent per case, for true isolation
            agent: a single shared agent, reset between cases when
                ``reset_between_cases`` is set
            concurrency: cases to run at once; 1 runs sequentially
            repeats: executions per case, scored by mean
            metrics: metrics to apply when a case names none
            fail_below: pass rate required for the report to count as passing
            on_progress: called with each result and its position
            backend: recorded in the report for provenance
            model: recorded in the report for provenance
            reset_between_cases: clear shared agent state between cases
            timeout: overall seconds for the whole run, 0 for none
        """
        self.agent_factory = agent_factory
        self.agent = agent
        self.concurrency = max(1, int(concurrency))
        self.repeats = max(1, int(repeats))
        self.metrics = [str(name) for name in metrics]
        self.fail_below = max(0.0, min(1.0, float(fail_below)))
        self.on_progress = on_progress
        self.backend = backend
        self.model = model
        self.reset_between_cases = bool(reset_between_cases)
        self.timeout = float(timeout)
        if agent_factory is None and agent is None:
            raise EvalError("an agent or an agent_factory is required")

    # -- execution -----------------------------------------------------------

    def _agent_for(self, index: int) -> Any:
        """Return the agent to use for one case."""
        if self.agent_factory is not None:
            return self.agent_factory()
        agent = self.agent
        if agent is None:
            raise EvalError("no agent available")
        if self.reset_between_cases and hasattr(agent, "reset"):
            try:
                agent.reset()
            except Exception as exc:  # noqa: BLE001 - a failed reset must not stop the run
                LOGGER.warning("agent reset failed before case %d: %s", index, exc)
        return agent

    def run_case(self, case: EvalCase, *, index: int = 0) -> CaseResult:
        """Run one case, including its repeats, and score the result."""
        attempts: list[CaseOutcome] = []
        started = time.perf_counter()
        repeats = max(1, int(case.repeats) if case.repeats else int(self.repeats))
        error = ""
        for repeat in range(repeats):
            outcome = self._execute(case, repeat_index=repeat, index=index)
            attempts.append(outcome)
            if outcome.error and not error:
                error = outcome.error
        elapsed = time.perf_counter() - started

        representative = _merge_attempts(attempts)
        representative.duration_seconds = elapsed / max(1, len(attempts))
        if error:
            representative.error = error

        options = dict(case.options or {})
        if case.context:
            options.setdefault("context_block", case.context)
        scores = score_case(case, representative, metrics=self._metrics_for(case),
                            weight=case.weight)
        value = aggregate(scores)
        # A required metric is a gate, not a suggestion: clearing the aggregate
        # while missing a declared tool call is a failure, not a 0.75.
        blocked = [item.name for item in scores if item.required and not item.passed]
        passed = value >= case.threshold and not representative.error and not blocked

        result = CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            tags=list(case.tags),
            outcome=representative,
            scores=scores,
            aggregate=value,
            threshold=case.threshold,
            passed=passed,
            repeats=len(attempts),
            attempts=attempts if len(attempts) > 1 else [],
            duration_seconds=elapsed,
            error=representative.error,
            weight=case.weight,
            blocked_by=blocked,
        )
        return result

    def _metrics_for(self, case: EvalCase) -> list[str]:
        """Return the metrics to apply to one case."""
        return list(case.metrics or self.metrics)

    def _execute(self, case: EvalCase, *, repeat_index: int = 0, index: int = 0) -> CaseOutcome:
        """Run the agent once for one case and capture the outcome."""
        started = time.perf_counter()
        try:
            agent = self._agent_for(index)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return CaseOutcome(error=f"agent setup failed: {exc}",
                               error_code="agent_setup", repeat_index=repeat_index,
                               duration_seconds=time.perf_counter() - started)
        options: dict[str, Any] = dict(case.options or {})
        if case.context:
            options.setdefault("context_block", case.context)
        try:
            result = agent.run(case.prompt, **options)
        except Exception as exc:  # noqa: BLE001 - a raising case is a failed case
            return CaseOutcome(error=f"{type(exc).__name__}: {exc}",
                               error_code=type(exc).__name__, repeat_index=repeat_index,
                               duration_seconds=time.perf_counter() - started)
        return CaseOutcome.from_run(result, repeat_index=repeat_index,
                                    duration_seconds=time.perf_counter() - started)

    def run(self, dataset: EvalDataset | Sequence[EvalCase], *,
            limit: int = 0) -> EvalReport:
        """Run a whole dataset and return the report.

        Args:
            dataset: cases to run
            limit: run at most this many
        """
        cases = list(dataset)
        if limit:
            cases = cases[: max(0, int(limit))]
        if not cases:
            raise EvalError("the dataset contains no cases")

        report = EvalReport(
            dataset=getattr(dataset, "name", "dataset"),
            started_at=time.time(),
            model=self.model,
            backend=getattr(self.backend, "name", "") if self.backend is not None else "",
            fail_below=self.fail_below,
            metadata={"concurrency": self.concurrency, "repeats": self.repeats,
                      "metrics": self.metrics},
        )

        if self.concurrency <= 1:
            for index, case in enumerate(cases):
                result = self.run_case(case, index=index)
                report.results.append(result)
                self._notify(result, index + 1, len(cases))
        else:
            batch = parallel_map(
                lambda pair: self.run_case(pair[1], index=pair[0]),
                list(enumerate(cases)),
                max_workers=self.concurrency,
                timeout=self.timeout,
                ordered=True,
                chunk_name="eval",
            )
            for position, outcome in enumerate(batch.outcomes):
                if outcome.ok:
                    result = outcome.value
                else:
                    case = cases[outcome.index]
                    result = CaseResult(case_id=case.id, prompt=case.prompt,
                                        tags=list(case.tags), passed=False,
                                        aggregate=0.0, threshold=case.threshold,
                                        error=outcome.error or "evaluation failed",
                                        weight=case.weight)
                report.results.append(result)
                self._notify(result, position + 1, len(cases))

        report.finished_at = time.time()
        return report

    def _notify(self, result: CaseResult, position: int, total: int) -> None:
        """Report progress, never letting a bad callback stop the run."""
        if self.on_progress is None:
            return
        try:
            self.on_progress(result, position, total)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("progress callback raised: %s", exc)

    def require(self, report: EvalReport) -> EvalReport:
        """Raise when a report does not meet its required pass rate.

        Args:
            report: the report to judge

        Raises:
            EvalThresholdError: the pass rate is below ``fail_below``
        """
        if report.ok:
            return report
        failures = [result.case_id for result in report.results if not result.passed]
        raise EvalThresholdError(
            f"evaluation {report.dataset} scored {report.pass_rate:.1%} "
            f"({report.passed}/{report.total}), below the required {report.fail_below:.1%}",
            details={"pass_rate": round(report.pass_rate, 4), "passed": report.passed,
                     "total": report.total, "failures": failures[:20]},
        )


def _merge_attempts(attempts: Sequence[CaseOutcome]) -> CaseOutcome:
    """Combine repeated attempts into one representative outcome.

    The mean score of a stochastic model is the honest number, but a single
    erroring attempt must not be averaged away into a pass: if any attempt
    errored, the merged outcome records it.
    """
    if not attempts:
        return CaseOutcome()
    if len(attempts) == 1:
        return attempts[0]
    count = len(attempts)
    errored = [attempt for attempt in attempts if attempt.error]
    best = max(attempts, key=lambda attempt: len(attempt.output or ""))
    merged = CaseOutcome(
        output=best.output,
        stop_reason=best.stop_reason,
        tools_called=sorted({name for attempt in attempts for name in attempt.tools_called}),
        total_tokens=int(sum(attempt.total_tokens for attempt in attempts) / count),
        prompt_tokens=int(sum(attempt.prompt_tokens for attempt in attempts) / count),
        completion_tokens=int(sum(attempt.completion_tokens for attempt in attempts) / count),
        cost=sum(attempt.cost for attempt in attempts) / count,
        duration_seconds=sum(attempt.duration_seconds for attempt in attempts) / count,
        iterations=int(sum(attempt.iterations for attempt in attempts) / count),
        structured=best.structured,
        citations=list(best.citations),
        error=errored[0].error if errored else "",
        error_code=errored[0].error_code if errored else "",
        run_id=best.run_id,
        model=best.model,
        metadata={"attempts": count, "errored_attempts": len(errored)},
    )
    return merged


def run_evaluation(dataset: EvalDataset | str | Path | None = None, *, agent: Any = None,
                   agent_factory: Callable[[], Any] | None = None, config: Any = None,
                   concurrency: int = 0, repeats: int = 0, metrics: Sequence[str] = (),
                   fail_below: float | None = None, limit: int = 0,
                   report_path: str | Path = "", raise_on_failure: bool = False) -> EvalReport:
    """Run an evaluation in one call, using configuration defaults.

    Args:
        dataset: a dataset, a path, a name, or ``None`` for the built-in set
        agent: shared agent to evaluate
        agent_factory: builds a fresh agent per case
        config: resolved configuration supplying defaults
        concurrency: cases at once, defaulting to ``[eval] concurrency``
        repeats: executions per case, defaulting to ``[eval] repeats``
        metrics: metrics to apply when a case names none
        fail_below: required pass rate, defaulting to ``[eval] fail_below``
        limit: run at most this many cases
        report_path: write the report here when non-empty
        raise_on_failure: raise :class:`EvalThresholdError` when the run fails

    Returns:
        The completed :class:`EvalReport`.
    """
    section = getattr(config, "eval", None)
    active_dataset: EvalDataset
    if isinstance(dataset, EvalDataset):
        active_dataset = dataset
    elif isinstance(dataset, (str, Path)) and dataset:
        active_dataset = resolve_dataset(str(dataset),
                                         path=str(getattr(section, "dataset_path", "") or "")
                                         or None)
    else:
        active_dataset = resolve_dataset(str(getattr(section, "dataset", "") or "builtin"),
                                         path=str(getattr(section, "dataset_path", "") or "")
                                         or None)

    runner = EvalRunner(
        agent_factory=agent_factory,
        agent=agent,
        concurrency=int(concurrency or getattr(section, "concurrency", 0) or DEFAULT_CONCURRENCY),
        repeats=int(repeats or getattr(section, "repeats", 0) or 1),
        metrics=list(metrics or getattr(section, "metrics", None) or []),
        fail_below=(fail_below if fail_below is not None
                    else float(getattr(section, "fail_below", DEFAULT_PASS_RATE) or 0.0)),
        model=str(getattr(getattr(config, "model", None), "model", "") or ""),
    )
    report = runner.run(active_dataset, limit=limit)
    if report_path:
        target = Path(report_path).expanduser()
        fmt = str(getattr(section, "report_format", "") or "markdown")
        report.save(target, fmt=fmt)
    if raise_on_failure:
        runner.require(report)
    return report


def metric_names() -> list[str]:
    """Return every available metric name."""
    return sorted(METRICS)


def describe_metrics() -> dict[str, str]:
    """Return each metric's one-line description."""
    from .metrics import metric_help

    return metric_help()


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_PASS_RATE",
    "CaseOutcome",
    "CaseResult",
    "EvalReport",
    "EvalRunner",
    "describe_metrics",
    "metric_names",
    "run_evaluation",
]
