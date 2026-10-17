"""Evaluation: score agent behaviour against declared expectations.

Unit tests prove the parts work; evaluation proves the *agent* works. The
difference matters because an agent's behaviour depends on prompt construction,
context trimming, tool availability, and guardrails interacting - none of which
a component test covers.

A dataset is data, not code::

    from magent.eval import EvalCase, EvalDataset, EvalRunner

    dataset = EvalDataset(name="smoke")
    dataset.add(EvalCase(
        id="arithmetic",
        prompt="What is 6 * 7?",
        expected=["42"],
        expected_tools=["calculate"],
        max_seconds=30,
    ))

    runner = EvalRunner(agent=agent, concurrency=2)
    report = runner.run(dataset)
    print(report.render("markdown"))
    runner.require(report)          # raises EvalThresholdError on regression

Declaring a field activates its metric: ``expected_tools`` is checked for tool
use, ``check_pii`` scans the answer for personal data, ``max_seconds`` enforces
a latency budget. A built-in dataset ships with the package so
``magent eval run`` is useful before anyone writes a case.
"""

from __future__ import annotations

from .cases import (
    DATASET_SUFFIXES,
    DEFAULT_THRESHOLD,
    EvalCase,
    EvalDataset,
    builtin_dataset,
    dataset_from_records,
    load_dataset,
    resolve_dataset,
)
from .metrics import (
    DEFAULT_PASS_THRESHOLD,
    METRIC_NAMES,
    METRICS,
    REQUIRED_METRICS,
    MetricScore,
    aggregate,
    contains,
    exact,
    json_schema,
    length_bounds,
    max_duration,
    max_tokens,
    metric_help,
    no_pii,
    no_tool,
    regex,
    required_for,
    score,
    stop_reason,
    tool_used,
)
from .runner import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PASS_RATE,
    CaseOutcome,
    CaseResult,
    EvalReport,
    EvalRunner,
    describe_metrics,
    metric_names,
    run_evaluation,
)

__all__ = [
    "DATASET_SUFFIXES",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_PASS_RATE",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_THRESHOLD",
    "METRIC_NAMES",
    "METRICS",
    "REQUIRED_METRICS",
    "CaseOutcome",
    "CaseResult",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "EvalRunner",
    "MetricScore",
    "aggregate",
    "builtin_dataset",
    "contains",
    "dataset_from_records",
    "describe_metrics",
    "exact",
    "json_schema",
    "length_bounds",
    "load_dataset",
    "max_duration",
    "max_tokens",
    "metric_help",
    "metric_names",
    "no_pii",
    "no_tool",
    "regex",
    "required_for",
    "resolve_dataset",
    "run_evaluation",
    "score",
    "stop_reason",
    "tool_used",
]
