"""Scoring functions for agent evaluation.

A single "did it work?" number hides the thing that actually regresses, so
each metric scores one dimension and reports its own evidence. An agent that
starts answering correctly but twice as slowly, or that begins leaking an email
address, should fail on *that* metric rather than on a blended average.

Every metric is a pure function taking a :class:`~magent.eval.cases.EvalCase`
and a :class:`~magent.eval.runner.CaseOutcome` and returning a
:class:`MetricScore`. They are registered in :data:`METRICS`, so a dataset can
name the ones it wants and a new one can be added without touching the runner.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

DEFAULT_PASS_THRESHOLD = 0.5
"""Score at or above which a metric counts as passing."""

METRIC_NAMES: tuple[str, ...] = (
    "contains",
    "exact",
    "regex",
    "json_schema",
    "tool_used",
    "no_tool",
    "stop_reason",
    "max_tokens",
    "max_duration",
    "no_pii",
    "required_for",
    "citation_coverage",
    "length_bounds",
    "callable",
)
"""Every metric the runner can apply, by name."""


@dataclass(slots=True)
class MetricScore:
    """One metric's verdict, with the evidence behind it."""

    name: str
    score: float = 0.0
    passed: bool = False
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    required: bool = False

    def __post_init__(self) -> None:
        """Clamp the score into ``0..1``."""
        self.score = max(0.0, min(1.0, float(self.score)))
        self.weight = max(0.0, float(self.weight))

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "passed": self.passed,
            "weight": self.weight,
            "required": self.required,
            "detail": self.detail,
            "evidence": self.evidence,
        }

    def __str__(self) -> str:
        """Return a one-line description."""
        mark = "pass" if self.passed else "FAIL"
        return f"{self.name}={self.score:.2f} [{mark}]" + (f" {self.detail}" if self.detail else "")


def contains(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Score how many expected substrings appear in the answer.

    Case-insensitive, because capitalisation is almost never the thing under
    test, and partial credit is proportional: three of four expected facts is
    0.75, which distinguishes a near miss from a total failure.
    """
    expected = _as_list(getattr(case, "expected", None) or getattr(case, "expected_contains", None))
    text = str(getattr(outcome, "output", "") or "").lower()
    if not expected:
        return MetricScore("contains", 1.0, True, "no expectations declared", weight=weight)
    hits = [item for item in expected if str(item).lower() in text]
    misses = [item for item in expected if str(item).lower() not in text]
    score = len(hits) / len(expected)
    return MetricScore(
        "contains", score, score >= float(getattr(case, "threshold", DEFAULT_PASS_THRESHOLD)),
        f"{len(hits)}/{len(expected)} expected strings present",
        {"hits": hits, "misses": misses[:10]}, weight,
    )


def exact(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require the answer to equal an expected value exactly."""
    expected = getattr(case, "expected_exact", "") or ""
    text = str(getattr(outcome, "output", "") or "").strip()
    if not expected:
        return MetricScore("exact", 1.0, True, "no exact expectation declared", weight=weight)
    normalize = bool(getattr(case, "normalize", True))
    left = _normalize(text) if normalize else text
    right = _normalize(str(expected)) if normalize else str(expected)
    passed = left == right
    return MetricScore("exact", 1.0 if passed else 0.0, passed,
                       "matches" if passed else f"expected {str(expected)[:60]!r}",
                       {"expected": str(expected)[:200], "actual": text[:200]}, weight)


def regex(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require the answer to match a regular expression."""
    pattern = str(getattr(case, "expected_regex", "") or "")
    text = str(getattr(outcome, "output", "") or "")
    if not pattern:
        return MetricScore("regex", 1.0, True, "no pattern declared", weight=weight)
    flags = re.IGNORECASE | re.DOTALL
    try:
        match = re.search(pattern, text, flags)
    except re.error as exc:
        return MetricScore("regex", 0.0, False, f"invalid pattern: {exc}",
                           {"pattern": pattern}, weight)
    return MetricScore("regex", 1.0 if match else 0.0, bool(match),
                       "matched" if match else f"no match for {pattern[:60]!r}",
                       {"pattern": pattern, "match": match.group(0)[:120] if match else ""},
                       weight)


def json_schema(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require the answer to parse as JSON matching declared required keys."""
    required = _as_list(getattr(case, "expected_keys", None))
    text = str(getattr(outcome, "output", "") or "").strip()
    structured = getattr(outcome, "structured", None)
    payload = structured if isinstance(structured, dict) else _parse_json(text)
    if payload is None:
        return MetricScore("json_schema", 0.0, False, "answer is not JSON",
                           {"text": text[:200]}, weight)
    if not required:
        return MetricScore("json_schema", 1.0, True, f"valid JSON with {len(payload)} keys",
                           {"keys": sorted(payload)}, weight)
    missing = [str(key) for key in required if key not in payload]
    score = 1.0 - (len(missing) / len(required))
    return MetricScore("json_schema", score, not missing,
                       "all required keys present" if not missing else f"missing {missing}",
                       {"required": [str(key) for key in required], "present": sorted(payload)},
                       weight)


def tool_used(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require that specific tools were called."""
    expected = _as_list(getattr(case, "expected_tools", None))
    called = [str(name) for name in (getattr(outcome, "tools_called", None) or [])]
    if not expected:
        return MetricScore("tool_used", 1.0, True, f"{len(called)} tool call(s), none required",
                           {"called": called}, weight)
    hits = [name for name in expected if name in called]
    missing = [name for name in expected if name not in called]
    score = len(hits) / len(expected)
    return MetricScore("tool_used", score, not missing,
                       f"called {hits}" + (f", missing {missing}" if missing else ""),
                       {"expected": [str(name) for name in expected], "called": called,
                        "missing": missing}, weight)


def no_tool(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require that no tool was called at all."""
    if not getattr(case, "expect_no_tools", False):
        return MetricScore("no_tool", 1.0, True, "not required", weight=weight)
    called = [str(name) for name in (getattr(outcome, "tools_called", None) or [])]
    passed = not called
    return MetricScore("no_tool", 1.0 if passed else 0.0, passed,
                       "no tools called" if passed else f"unexpectedly called {called}",
                       {"called": called}, weight)


def stop_reason(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require a specific stop reason."""
    expected = str(getattr(case, "expected_stop", "") or "")
    actual = str(getattr(outcome, "stop_reason", "") or "")
    if not expected:
        return MetricScore("stop_reason", 1.0, True, f"stopped: {actual or 'unknown'}",
                           weight=weight)
    passed = actual == expected
    return MetricScore("stop_reason", 1.0 if passed else 0.0, passed,
                       f"expected {expected}, got {actual or 'unknown'}",
                       {"expected": expected, "actual": actual}, weight)


def max_tokens(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require token usage to stay under a ceiling."""
    limit = int(getattr(case, "max_tokens", 0) or 0)
    used = int(getattr(outcome, "total_tokens", 0) or 0)
    if not limit:
        return MetricScore("max_tokens", 1.0, True, f"{used} tokens, no ceiling",
                           {"used": used}, weight)
    passed = used <= limit
    score = min(1.0, limit / used) if used else 1.0
    return MetricScore("max_tokens", score if passed else score * 0.5, passed,
                       f"{used} of {limit} tokens", {"used": used, "limit": limit}, weight)


def max_duration(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require a run to finish within a wall-clock budget."""
    limit = float(getattr(case, "max_seconds", 0.0) or 0.0)
    elapsed = float(getattr(outcome, "duration_seconds", 0.0) or 0.0)
    if not limit:
        return MetricScore("max_duration", 1.0, True, f"{elapsed:.3f}s, no ceiling",
                           {"seconds": round(elapsed, 3)}, weight)
    passed = elapsed <= limit
    score = min(1.0, limit / elapsed) if elapsed else 1.0
    return MetricScore("max_duration", score, passed, f"{elapsed:.3f}s of {limit}s",
                       {"seconds": round(elapsed, 3), "limit": limit}, weight)


def no_pii(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require the answer to contain no detectable personal data."""
    if not getattr(case, "check_pii", False):
        return MetricScore("no_pii", 1.0, True, "not required", weight=weight)
    from ..guardrails import PIIScanner

    text = str(getattr(outcome, "output", "") or "")
    scanner = PIIScanner()
    findings = scanner.scan(text)
    kinds = sorted({str(getattr(item, "kind", "") or getattr(item, "label", ""))
                    for item in findings})
    passed = not findings
    return MetricScore("no_pii", 1.0 if passed else 0.0, passed,
                       "no personal data" if passed else f"leaked {kinds}",
                       {"findings": kinds, "count": len(findings)}, weight)


def citation_coverage(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require grounded answers to cite sources that actually appear."""
    if not getattr(case, "check_citations", False):
        return MetricScore("citation_coverage", 1.0, True, "not required", weight=weight)
    text = str(getattr(outcome, "output", "") or "")
    citations = list(getattr(outcome, "citations", None) or [])
    if not citations:
        numbers = re.findall(r"\[(\d+)\]", text)
        return MetricScore("citation_coverage", 0.0, False,
                           f"answer cites {len(numbers)} reference(s) but no sources were provided",
                           {"inline": numbers[:10]}, weight)
    from ..rag import verify_citations

    check = verify_citations(text, citations)
    coverage = float(getattr(check, "coverage", 0.0) or 0.0)
    dangling = list(getattr(check, "dangling", []) or [])
    passed = coverage >= float(getattr(case, "min_coverage", 0.6)) and not dangling
    return MetricScore("citation_coverage", coverage, passed,
                       f"coverage {coverage:.0%}" + (f", dangling {dangling}" if dangling else ""),
                       {"coverage": round(coverage, 4), "dangling": dangling,
                        "sources": len(citations)}, weight)


def length_bounds(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Require the answer's length to fall inside a range."""
    minimum = int(getattr(case, "min_chars", 0) or 0)
    maximum = int(getattr(case, "max_chars", 0) or 0)
    text = str(getattr(outcome, "output", "") or "")
    length = len(text.strip())
    if not minimum and not maximum:
        return MetricScore("length_bounds", 1.0, True, f"{length} chars, no bounds",
                           {"chars": length}, weight)
    if minimum and length < minimum:
        return MetricScore("length_bounds", length / minimum if minimum else 0.0, False,
                           f"{length} chars, below the {minimum} minimum",
                           {"chars": length, "min": minimum}, weight)
    if maximum and length > maximum:
        return MetricScore("length_bounds", maximum / length if length else 0.0, False,
                           f"{length} chars, above the {maximum} maximum",
                           {"chars": length, "max": maximum}, weight)
    return MetricScore("length_bounds", 1.0, True, f"{length} chars within bounds",
                       {"chars": length, "min": minimum, "max": maximum}, weight)


def callable_metric(case: Any, outcome: Any, *, weight: float = 1.0) -> MetricScore:
    """Apply a case-supplied checker, for assertions the built-ins cannot express."""
    checker = getattr(case, "checker", None)
    if not callable(checker):
        return MetricScore("callable", 1.0, True, "no custom checker", weight=weight)
    try:
        verdict = checker(case, outcome)
    except Exception as exc:  # noqa: BLE001 - a broken checker must not abort the run
        return MetricScore("callable", 0.0, False, f"checker raised {type(exc).__name__}: {exc}",
                           weight=weight)
    if isinstance(verdict, MetricScore):
        verdict.weight = weight
        return verdict
    if isinstance(verdict, tuple) and len(verdict) == 2:
        score, detail = verdict
        passed = float(score) >= DEFAULT_PASS_THRESHOLD
        return MetricScore("callable", float(score), passed, str(detail), weight=weight)
    if isinstance(verdict, bool):
        return MetricScore("callable", 1.0 if verdict else 0.0, verdict,
                           "checker returned " + str(verdict), weight=weight)
    score = float(verdict) if isinstance(verdict, (int, float)) else 0.0
    return MetricScore("callable", score, score >= DEFAULT_PASS_THRESHOLD,
                       f"checker returned {score}", weight=weight)


METRICS: dict[str, Callable[..., MetricScore]] = {
    "contains": contains,
    "exact": exact,
    "regex": regex,
    "json_schema": json_schema,
    "tool_used": tool_used,
    "no_tool": no_tool,
    "stop_reason": stop_reason,
    "max_tokens": max_tokens,
    "max_duration": max_duration,
    "no_pii": no_pii,
    "citation_coverage": citation_coverage,
    "length_bounds": length_bounds,
    "callable": callable_metric,
}
"""Metric name to scoring function."""


REQUIRED_METRICS: dict[str, str] = {
    "tool_used": "expected_tools",
    "no_tool": "expect_no_tools",
    "stop_reason": "expected_stop",
    "no_pii": "check_pii",
    "json_schema": "expected_keys",
    "exact": "expected_exact",
    "citation_coverage": "check_citations",
}
"""Metrics that gate a case, keyed by the field whose presence requires them.

Averaging alone lets a hard requirement be diluted: a case that must call
``read_file`` can still clear its threshold on ``stop_reason`` and
``max_duration`` while calling nothing at all. When the case *declares* one of
these expectations, the metric must pass on its own merits as well as
contributing to the aggregate.
"""


def required_for(case: Any) -> set[str]:
    """Return the metric names a case requires, based on what it declares."""
    required: set[str] = set()
    for name, field_name in REQUIRED_METRICS.items():
        value = getattr(case, field_name, None)
        if value:
            required.add(name)
    return required


def score(case: Any, outcome: Any, *, metrics: Sequence[str] = (),
          weight: float = 1.0) -> list[MetricScore]:
    """Apply every requested metric, plus any the case implies.

    A case that declares ``expected_tools`` gets the ``tool_used`` metric even
    if it was not named, because declaring an expectation and then not checking
    it is how a suite quietly stops testing anything. Metrics tied to an
    explicit declaration are also marked ``required``, so they gate the verdict.
    """
    wanted = list(metrics) if metrics else list(_default_metrics(case))
    for name in _implied_metrics(case):
        if name not in wanted:
            wanted.append(name)
    gating = required_for(case)
    scores: list[MetricScore] = []
    for name in wanted:
        metric = METRICS.get(name)
        if metric is None:
            scores.append(MetricScore(name, 0.0, False, f"unknown metric {name!r}",
                                      weight=weight, required=name in gating))
            continue
        result = metric(case, outcome, weight=weight)
        result.required = name in gating
        scores.append(result)
    return scores


def _default_metrics(case: Any) -> list[str]:
    """Return the metrics to apply when a case names none."""
    names = ["stop_reason"]
    if getattr(case, "expected", None) or getattr(case, "expected_contains", None):
        names.append("contains")
    if getattr(case, "expected_exact", ""):
        names.append("exact")
    if getattr(case, "expected_regex", ""):
        names.append("regex")
    if getattr(case, "expected_keys", None):
        names.append("json_schema")
    if not names[1:]:
        names.append("length_bounds")
    return names


def _implied_metrics(case: Any) -> list[str]:
    """Return metrics implied by fields the case declares."""
    implied: list[str] = []
    if getattr(case, "expected_tools", None):
        implied.append("tool_used")
    if getattr(case, "expect_no_tools", False):
        implied.append("no_tool")
    if getattr(case, "expected_stop", ""):
        implied.append("stop_reason")
    if getattr(case, "max_tokens", 0):
        implied.append("max_tokens")
    if getattr(case, "max_seconds", 0.0):
        implied.append("max_duration")
    if getattr(case, "check_pii", False):
        implied.append("no_pii")
    if getattr(case, "check_citations", False):
        implied.append("citation_coverage")
    if getattr(case, "min_chars", 0) or getattr(case, "max_chars", 0):
        implied.append("length_bounds")
    if callable(getattr(case, "checker", None)):
        implied.append("callable")
    return implied


def aggregate(scores: Sequence[MetricScore]) -> float:
    """Combine metric scores into one weighted value in ``0..1``."""
    total_weight = sum(item.weight for item in scores)
    if total_weight <= 0:
        return 0.0
    return sum(item.score * item.weight for item in scores) / total_weight


def _as_list(value: Any) -> list[Any]:
    """Coerce a scalar or sequence into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize(text: str) -> str:
    """Collapse whitespace and case, for forgiving comparisons."""
    return " ".join(str(text or "").split()).lower()


def _parse_json(text: str) -> Any | None:
    """Parse JSON from text that may carry prose around it."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    from ..utils.json_utils import extract_json

    try:
        return extract_json(text)
    except Exception:  # noqa: BLE001 - not JSON is a valid outcome
        return None


def metric_help() -> dict[str, str]:
    """Return each metric's first docstring line, for ``magent eval metrics``."""
    out: dict[str, str] = {}
    for name, metric in METRICS.items():
        doc = (metric.__doc__ or "").strip().splitlines()
        out[name] = doc[0].strip() if doc else name
    return out


def timed(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Call ``fn`` and return its result with the elapsed seconds."""
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, max(0.0, time.perf_counter() - started)


__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "METRIC_NAMES",
    "METRICS",
    "REQUIRED_METRICS",
    "MetricScore",
    "aggregate",
    "callable_metric",
    "citation_coverage",
    "contains",
    "exact",
    "json_schema",
    "length_bounds",
    "max_duration",
    "max_tokens",
    "metric_help",
    "no_pii",
    "no_tool",
    "regex",
    "score",
    "stop_reason",
    "timed",
    "tool_used",
]
