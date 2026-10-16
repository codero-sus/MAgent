"""Evaluation metrics: judge one run against one case.

Metrics are plain functions over ``(case, result)`` returning a score in
``0..1`` plus an explanation, so a failing case says *why* it failed.  The
registry in :data:`METRICS` lets a dataset request a specific set by name
(``--metrics contains,tools``), and :func:`score_case` runs the default set plus
anything the case asks for.

Nothing here calls a model: a grader that needs an LLM would make the suite
non-deterministic and slow, which is exactly what a regression suite must not be.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import MetricError
from ..utils.json_utils import extract_json
from ..utils.text import levenshtein, normalize_newlines, similarity
from .dataset import EvalCase

MIN_PASS_SCORE = 0.6
"""A case passes when its weighted score reaches this."""


@dataclass(slots=True)
class MetricOutcome:
    """Result of one metric on one case."""

    name: str
    score: float
    passed: bool = True
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the outcome."""
        return {"name": self.name, "score": round(self.score, 4), "passed": self.passed,
                "reason": self.reason, "detail": dict(self.detail)}


@dataclass(slots=True)
class CaseScore:
    """Every metric outcome for one case."""

    case_id: str
    outcomes: list[MetricOutcome] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    output: str = ""
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    @property
    def score(self) -> float:
        """Mean score across metrics; 1.0 when there are none."""
        if not self.outcomes:
            return 0.0 if self.error else 1.0
        return round(sum(item.score for item in self.outcomes) / len(self.outcomes), 4)

    @property
    def passed(self) -> bool:
        """True when the run produced no error and scored at or above the bar."""
        if self.error:
            return False
        return self.score >= MIN_PASS_SCORE

    @property
    def failures(self) -> list[MetricOutcome]:
        """Metrics that did not reach a passing score."""
        return [item for item in self.outcomes if not item.passed]

    def to_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        """Serialise the case score."""
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "score": self.score,
            "passed": self.passed,
            "seconds": round(self.seconds, 4),
            "error": self.error,
            "tags": list(self.tags),
            "weight": self.weight,
            "metrics": [item.to_dict() for item in self.outcomes],
            "failures": [item.name for item in self.failures],
        }
        if include_output:
            payload["output"] = self.output
        return payload

    def summary(self) -> str:
        """One-line result for the CLI."""
        state = "PASS" if self.passed else "FAIL"
        reasons = "; ".join(item.reason for item in self.failures[:2])
        return f"[{state}] {self.case_id} {self.score:.2f}{' - ' + reasons if reasons else ''}"


def exact_match(case: EvalCase, result: Any) -> MetricOutcome:
    """Score 1 when the answer equals the expectation after normalisation."""
    output = _text(result)
    expected = case.expected
    if not expected:
        return MetricOutcome("exact_match", 1.0, True, "no expectation")
    left = _normalise(output)
    right = _normalise(expected)
    score = 1.0 if left == right else 0.0
    reason = "identical" if score else "differs from the expected answer"
    return MetricOutcome("exact_match", score, score >= MIN_PASS_SCORE, reason,
                         {"expected": expected[:200], "got": output[:200]})


def contains_all(case: EvalCase, result: Any) -> MetricOutcome:
    """Score the fraction of required substrings present in the answer."""
    if not case.contains:
        return MetricOutcome("contains", 1.0, True, "no required text")
    haystack = _text(result).lower()
    missing = [needle for needle in case.contains if needle.lower() not in haystack]
    score = round(1.0 - len(missing) / len(case.contains), 4)
    return MetricOutcome("contains", score, not missing,
                         "all present" if not missing else f"missing: {', '.join(missing[:4])}",
                         {"missing": missing})


def forbids_text(case: EvalCase, result: Any) -> MetricOutcome:
    """Score 0 when any forbidden substring appears in the answer."""
    if not case.not_contains:
        return MetricOutcome("not_contains", 1.0, True, "nothing forbidden")
    haystack = _text(result).lower()
    hits = [needle for needle in case.not_contains if needle.lower() in haystack]
    return MetricOutcome("not_contains", 0.0 if hits else 1.0, not hits,
                         "clean" if not hits else f"found: {', '.join(hits[:4])}",
                         {"hits": hits})


def regex_match(case: EvalCase, result: Any) -> MetricOutcome:
    """Score 1 when the answer matches the case's regular expression."""
    if not case.regex:
        return MetricOutcome("regex", 1.0, True, "no pattern")
    match = re.search(case.regex, _text(result), re.IGNORECASE | re.MULTILINE)
    return MetricOutcome("regex", 1.0 if match else 0.0, bool(match),
                         "pattern matched" if match else f"no match for {case.regex!r}",
                         {"match": match.group(0)[:200] if match else ""})


def similarity_score(case: EvalCase, result: Any) -> MetricOutcome:
    """Score the textual similarity to the expected answer."""
    if not case.expected:
        return MetricOutcome("similarity", 1.0, True, "no expectation")
    score = round(similarity(_text(result), case.expected), 4)
    return MetricOutcome("similarity", score, score >= 0.6, f"similarity {score:.2f}",
                         {"distance": levenshtein(_text(result), case.expected)})


def json_valid(case: EvalCase, result: Any) -> MetricOutcome:
    """Score 1 when the answer contains a parseable JSON object."""
    if not case.expect_json:
        return MetricOutcome("json_valid", 1.0, True, "JSON not required")
    raw = _text(result)
    candidate = extract_json(raw)
    parsed: Any = None
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            parsed = None
    elif candidate is not None:
        parsed = candidate
    ok = parsed is not None
    return MetricOutcome("json_valid", 1.0 if ok else 0.0, ok,
                         "parsed JSON" if ok else "no JSON object in the answer",
                         {"keys": sorted(parsed)[:10] if isinstance(parsed, dict) else []})


def tool_usage(case: EvalCase, result: Any) -> MetricOutcome:
    """Score required and forbidden tool calls."""
    if not case.tools and not case.forbidden_tools:
        return MetricOutcome("tools", 1.0, True, "no tool requirements")
    used = _tools(result)
    missing = [name for name in case.tools if name not in used]
    banned = [name for name in case.forbidden_tools if name in used]
    total = len(case.tools) + len(case.forbidden_tools)
    bad = len(missing) + len(banned)
    score = round(1.0 - bad / total, 4) if total else 1.0
    if missing and banned:
        reason = f"missing {', '.join(missing[:3])}; called forbidden {', '.join(banned[:3])}"
    elif missing:
        reason = f"missing tool calls: {', '.join(missing[:3])}"
    elif banned:
        reason = f"called forbidden tools: {', '.join(banned[:3])}"
    else:
        reason = "tool usage correct"
    return MetricOutcome("tools", score, not bad, reason,
                         {"used": sorted(used)[:12], "missing": missing, "banned": banned})


def step_budget(case: EvalCase, result: Any) -> MetricOutcome:
    """Score whether the run stayed inside its step budget."""
    if not case.max_steps:
        return MetricOutcome("max_steps", 1.0, True, "no step budget")
    steps = len(getattr(result, "steps", []) or [])
    ok = steps <= case.max_steps
    return MetricOutcome("max_steps", 1.0 if ok else 0.0, ok,
                         f"{steps} steps (limit {case.max_steps})",
                         {"steps": steps, "limit": case.max_steps})


def token_budget(case: EvalCase, result: Any) -> MetricOutcome:
    """Score whether the run stayed inside its token budget."""
    if not case.max_tokens:
        return MetricOutcome("max_tokens", 1.0, True, "no token budget")
    usage = getattr(result, "usage", None)
    used = getattr(usage, "total_tokens", 0) or 0
    ok = used <= case.max_tokens
    return MetricOutcome("max_tokens", 1.0 if ok else 0.0, ok,
                         f"{used} tokens (limit {case.max_tokens})",
                         {"tokens": used, "limit": case.max_tokens})


def time_budget(case: EvalCase, result: Any) -> MetricOutcome:
    """Score whether the run finished inside its time budget."""
    if not case.max_seconds:
        return MetricOutcome("max_seconds", 1.0, True, "no time budget")
    seconds = getattr(result, "duration_ms", 0.0) / 1000.0
    ok = seconds <= case.max_seconds
    return MetricOutcome("max_seconds", 1.0 if ok else 0.0, ok,
                         f"{seconds:.2f}s (limit {case.max_seconds:.2f}s)",
                         {"seconds": seconds, "limit": case.max_seconds})


def no_error(case: EvalCase, result: Any) -> MetricOutcome:
    """Score 1 when the run completed without an error."""
    del case
    error = getattr(result, "error", "") or ""
    ok = not error
    return MetricOutcome("no_error", 1.0 if ok else 0.0, ok,
                         "clean run" if ok else error[:160], {"error": error[:400]})


DEFAULT_METRIC_NAMES: tuple[str, ...] = ("no_error", "contains", "not_contains", "regex", "tools",
                                         "json_valid", "max_steps")
"""Metrics applied to every case unless the caller narrows the list."""

METRICS: dict[str, Callable[[EvalCase, Any], MetricOutcome]] = {
    "no_error": no_error,
    "contains": contains_all,
    "not_contains": forbids_text,
    "regex": regex_match,
    "exact_match": exact_match,
    "similarity": similarity_score,
    "json_valid": json_valid,
    "tools": tool_usage,
    "max_steps": step_budget,
    "max_tokens": token_budget,
    "max_seconds": time_budget,
}
"""Callable metrics by name."""


def metric_names() -> list[str]:
    """Registered metric names."""
    return sorted(METRICS)


def get_metric(name: str) -> Callable[[EvalCase, Any], MetricOutcome]:
    """Look up a metric by name.

    Raises:
        MetricError: when the metric does not exist
    """
    key = str(name).strip().lower()
    if key not in METRICS:
        raise MetricError(f"unknown metric {name!r}",
                          details={"known": metric_names()})
    return METRICS[key]


def register_metric(name: str, fn: Callable[[EvalCase, Any], MetricOutcome],
                    *, replace: bool = False) -> Callable[[EvalCase, Any], MetricOutcome]:
    """Register a custom metric.

    Raises:
        MetricError: when the name is taken and replacing is not allowed
    """
    if name in METRICS and not replace:
        raise MetricError(f"metric {name!r} already exists")
    METRICS[name] = fn
    return fn


def score_case(case: EvalCase, result: Any, *, metrics: Sequence[str] = (),
               weights: Mapping[str, float] | None = None) -> CaseScore:
    """Score one run against one case.

    Args:
        case: the case
        result: a :class:`~magent.types.RunResult`
        metrics: metric names to apply; defaults to :data:`DEFAULT_METRIC_NAMES`
        weights: optional per-metric weights

    Returns:
        a :class:`CaseScore` with one outcome per metric
    """
    names = list(metrics or DEFAULT_METRIC_NAMES)
    if case.expected and "similarity" not in names and "exact_match" not in names:
        names.append("similarity")
    if case.expect_json and "json_valid" not in names:
        names.append("json_valid")
    if case.max_tokens and "max_tokens" not in names:
        names.append("max_tokens")
    if case.max_seconds and "max_seconds" not in names:
        names.append("max_seconds")
    started = time.perf_counter()
    score = CaseScore(case_id=case.id, output=_text(result)[:2_000], tags=case.tags,
                      weight=case.weight)
    for name in names:
        try:
            outcome = get_metric(name)(case, result)
        except Exception as exc:  # noqa: BLE001 - a broken metric is a failed metric
            outcome = MetricOutcome(name, 0.0, False, f"metric error: {type(exc).__name__}: {exc}")
        weight = float((weights or {}).get(name, 1.0))
        if weight != 1.0:
            outcome.detail["weight"] = weight
            outcome.score = round(min(1.0, outcome.score * weight), 4)
        score.outcomes.append(outcome)
    stopped = getattr(getattr(result, "stop_reason", ""), "value",
                      getattr(result, "stop_reason", ""))
    if stopped and stopped in {"error", "guardrail"}:
        score.error = f"stop reason: {stopped}"
    score.seconds = time.perf_counter() - started
    return score


def _text(result: Any) -> str:
    """Extract the answer text from a run result or a plain string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return normalize_newlines(result)
    return normalize_newlines(str(getattr(result, "output", "") or ""))


def _tools(result: Any) -> set[str]:
    """Every tool name a run called."""
    names: set[str] = set()
    for step in getattr(result, "steps", []) or []:
        for call in getattr(step, "tool_calls", []) or []:
            name = getattr(call, "name", "")
            if name:
                names.add(str(name))
    return names


def _normalise(text: str) -> str:
    """Normalise text for comparison."""
    return " ".join((text or "").lower().split()).strip(" .!?")


def aggregate(scores: Iterable[CaseScore]) -> dict[str, Any]:
    """Aggregate case scores into a report-friendly summary."""
    items = list(scores)
    if not items:
        return {"cases": 0, "passed": 0, "pass_rate": 0.0, "score": 0.0, "metrics": {}}
    passed = sum(1 for item in items if item.passed)
    per_metric: dict[str, list[float]] = {}
    for item in items:
        for outcome in item.outcomes:
            per_metric.setdefault(outcome.name, []).append(outcome.score)
    return {
        "cases": len(items),
        "passed": passed,
        "failed": len(items) - passed,
        "pass_rate": round(passed / len(items), 4),
        "score": round(sum(item.score for item in items) / len(items), 4),
        "seconds": round(sum(item.seconds for item in items), 4),
        "metrics": {name: round(sum(values) / len(values), 4)
                    for name, values in sorted(per_metric.items())},
    }


def weighted_pass_rate(scores: Sequence[CaseScore]) -> float:
    """Pass rate where each case counts for its weight."""
    total = sum(item.weight for item in scores)
    if not total:
        return 0.0
    earned = sum(item.weight for item in scores if item.passed)
    return round(earned / total, 4)


__all__ = [
    "DEFAULT_METRIC_NAMES",
    "METRICS",
    "MIN_PASS_SCORE",
    "CaseScore",
    "MetricOutcome",
    "aggregate",
    "contains_all",
    "exact_match",
    "forbids_text",
    "get_metric",
    "json_valid",
    "metric_names",
    "no_error",
    "regex_match",
    "register_metric",
    "score_case",
    "similarity_score",
    "step_budget",
    "time_budget",
    "token_budget",
    "tool_usage",
    "weighted_pass_rate",
]
