"""Evaluation: datasets, deterministic metrics and batch runners.

Public API:

:class:`EvalCase` / :class:`Dataset`
    cases with expectations (substrings, regex, JSON, tools, budgets), loaded
    from JSON, JSONL, Markdown tables or Python literals;
:func:`score_case` / :data:`METRICS` / :class:`CaseScore`
    deterministic, model-free scoring with a per-metric reason;
:class:`EvalRunner` / :func:`run_eval` / :class:`EvalReport`
    run a dataset against an agent, in parallel, with repeats and reports.

>>> from magent.eval import sample_dataset, metric_names
>>> len(sample_dataset("demo")) >= 4
True
>>> "contains" in metric_names()
True
"""

from __future__ import annotations

from .dataset import (CASE_FIELDS, DATASET_FORMATS, MAX_CASES, Dataset, EvalCase,
                      build_dataset, load_dataset, merge_datasets, parse_markdown_dataset,
                      sample_dataset, save_dataset)
from .metrics import (DEFAULT_METRIC_NAMES, METRICS, MIN_PASS_SCORE, CaseScore, MetricOutcome,
                      aggregate, contains_all, exact_match, forbids_text, get_metric, json_valid,
                      metric_names, no_error, regex_match, register_metric, score_case,
                      similarity_score, step_budget, time_budget, token_budget, tool_usage,
                      weighted_pass_rate)
from .runner import EvalReport, EvalRunner, run_eval

__all__ = [
    "CASE_FIELDS",
    "DATASET_FORMATS",
    "DEFAULT_METRIC_NAMES",
    "MAX_CASES",
    "METRICS",
    "MIN_PASS_SCORE",
    "CaseScore",
    "Dataset",
    "EvalCase",
    "EvalReport",
    "EvalRunner",
    "MetricOutcome",
    "aggregate",
    "build_dataset",
    "contains_all",
    "exact_match",
    "forbids_text",
    "get_metric",
    "json_valid",
    "load_dataset",
    "merge_datasets",
    "metric_names",
    "no_error",
    "parse_markdown_dataset",
    "regex_match",
    "register_metric",
    "run_eval",
    "sample_dataset",
    "save_dataset",
    "score_case",
    "similarity_score",
    "step_budget",
    "time_budget",
    "token_budget",
    "tool_usage",
    "weighted_pass_rate",
]
