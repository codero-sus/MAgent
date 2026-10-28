"""Usage, cost, and budget accounting across a run or a whole session.

Cortex LLMHoster runs models locally, so there is no per-token cloud charge.
MAgent still tracks tokens rigorously because they are the real currency of a
local deployment: they bound context windows, determine wall-clock latency, and
drive the budget guardrail that stops a runaway agent loop.

Where a price *is* supplied (for example when MAgent is pointed at a hosted
OpenAI-compatible gateway instead of Cortex) the same accounting produces a
genuine cost figure.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..types import CostEstimate, Usage


@dataclass(slots=True)
class PriceBook:
    """Per-model pricing used to turn tokens into money."""

    prompt_per_million: dict[str, float] = field(default_factory=dict)
    completion_per_million: dict[str, float] = field(default_factory=dict)
    default_prompt: float = 0.0
    default_completion: float = 0.0
    currency: str = "USD"

    def price_for(self, model: str) -> tuple[float, float]:
        """Return ``(prompt, completion)`` price per million tokens."""
        return (
            self.prompt_per_million.get(model, self.default_prompt),
            self.completion_per_million.get(model, self.default_completion),
        )

    def cost(self, model: str, usage: Usage) -> CostEstimate:
        """Convert a usage record into a :class:`CostEstimate`."""
        prompt_price, completion_price = self.price_for(model)
        return CostEstimate(
            prompt_cost=round(usage.prompt_tokens / 1_000_000 * prompt_price, 8),
            completion_cost=round(usage.completion_tokens / 1_000_000 * completion_price, 8),
            currency=self.currency,
            source="price_book",
        )

    @classmethod
    def free(cls) -> PriceBook:
        """Return a price book where everything costs nothing (local hosting)."""
        return cls(currency="USD")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the price book."""
        return {
            "prompt_per_million": dict(self.prompt_per_million),
            "completion_per_million": dict(self.completion_per_million),
            "default_prompt": self.default_prompt,
            "default_completion": self.default_completion,
            "currency": self.currency,
        }


@dataclass(slots=True)
class BudgetConfig:
    """Hard ceilings for a run; ``0`` means unlimited for that dimension."""

    max_tokens: int = 0
    max_prompt_tokens: int = 0
    max_completion_tokens: int = 0
    max_cost: float = 0.0
    max_steps: int = 0
    max_tool_calls: int = 0
    max_seconds: float = 0.0
    max_requests: int = 0
    warn_at: float = 0.8

    def is_unlimited(self) -> bool:
        """Report whether no ceiling at all is configured."""
        return not any(
            [
                self.max_tokens,
                self.max_prompt_tokens,
                self.max_completion_tokens,
                self.max_cost,
                self.max_steps,
                self.max_tool_calls,
                self.max_seconds,
                self.max_requests,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""
        return {
            "max_tokens": self.max_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_cost": self.max_cost,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_seconds": self.max_seconds,
            "max_requests": self.max_requests,
            "warn_at": self.warn_at,
        }


@dataclass(slots=True)
class BudgetViolation:
    """A record that a ceiling was reached."""

    dimension: str
    limit: float
    observed: float
    ts: float = field(default_factory=time.time)

    @property
    def ratio(self) -> float:
        """Return observed/limit."""
        return round(self.observed / self.limit, 4) if self.limit else 0.0

    def message(self) -> str:
        """Return a human readable explanation."""
        return (
            f"Budget exceeded: {self.dimension} reached {self.observed:g} "
            f"of a {self.limit:g} limit ({self.ratio * 100:.0f}%)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the violation."""
        return {
            "dimension": self.dimension,
            "limit": self.limit,
            "observed": self.observed,
            "ratio": self.ratio,
            "ts": self.ts,
            "message": self.message(),
        }


class UsageTracker:
    """Thread-safe accumulator of token, cost, and latency statistics.

    One tracker is created per agent run and can be shared across a team of
    agents so the budget applies to the whole collaboration, not per member.
    """

    def __init__(
        self,
        *,
        budget: BudgetConfig | None = None,
        prices: PriceBook | None = None,
        on_warning: Any = None,
        on_violation: Any = None,
    ) -> None:
        self.budget = budget or BudgetConfig()
        self.prices = prices or PriceBook.free()
        self.on_warning = on_warning
        self.on_violation = on_violation
        self._lock = threading.RLock()
        self.usage = Usage()
        self.cost = CostEstimate(currency=self.prices.currency)
        self.requests = 0
        self.steps = 0
        self.tool_calls = 0
        self.errors = 0
        self.cache_hits = 0
        self.retries = 0
        self.latencies: list[float] = []
        self.by_model: dict[str, Usage] = defaultdict(Usage)
        self.cost_by_model: dict[str, float] = defaultdict(float)
        self.started_at = time.monotonic()
        self.violations: list[BudgetViolation] = []
        self.warnings_emitted: set[str] = set()
        self.tokens_per_model: dict[str, int] = defaultdict(int)

    # -- recording -----------------------------------------------------------

    def record_response(
        self, model: str, usage: Usage, *, latency_ms: float = 0.0, cache_hit: bool = False
    ) -> CostEstimate:
        """Record one completion, updating totals, per-model splits, and cost."""
        with self._lock:
            self.usage.add(usage)
            self.requests += 1
            if cache_hit:
                self.cache_hits += 1
            if latency_ms > 0:
                self.latencies.append(latency_ms)
                self.latencies = self.latencies[-1000:]
            model_key = model or "unknown"
            self.by_model[model_key].add(usage)
            self.tokens_per_model[model_key] += usage.total_tokens
            estimate = self.prices.cost(model_key, usage)
            self.cost.prompt_cost += estimate.prompt_cost
            self.cost.completion_cost += estimate.completion_cost
            self.cost_by_model[model_key] = round(self.cost_by_model[model_key] + estimate.total, 8)
            self._check_limits()
            return estimate

    def record_step(self, count: int = 1) -> None:
        """Record agent loop iterations."""
        with self._lock:
            self.steps += count
            self._check_limits()

    def record_tool_call(self, count: int = 1) -> None:
        """Record tool invocations."""
        with self._lock:
            self.tool_calls += count
            self._check_limits()

    def record_error(self) -> None:
        """Record a failed request."""
        with self._lock:
            self.errors += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        with self._lock:
            self.retries += 1

    def record_tokens(self, count: int, *, model: str = "") -> None:
        """Record token usage without a full response record."""
        with self._lock:
            self.usage.total_tokens += count
            if model:
                self.tokens_per_model[model] += count
            self._check_limits()

    # -- budget enforcement --------------------------------------------------

    def elapsed_seconds(self) -> float:
        """Return wall clock seconds since tracking began."""
        return time.monotonic() - self.started_at

    def _check_limits(self) -> None:
        """Test every configured ceiling and emit warnings or violations."""
        budget = self.budget
        checks: list[tuple[str, float, float]] = [
            ("tokens", self.usage.total_tokens, budget.max_tokens),
            ("prompt_tokens", self.usage.prompt_tokens, budget.max_prompt_tokens),
            ("completion_tokens", self.usage.completion_tokens, budget.max_completion_tokens),
            ("cost", self.cost.total, budget.max_cost),
            ("steps", self.steps, budget.max_steps),
            ("tool_calls", self.tool_calls, budget.max_tool_calls),
            ("seconds", self.elapsed_seconds(), budget.max_seconds),
            ("requests", self.requests, budget.max_requests),
        ]
        for dimension, observed, limit in checks:
            if limit <= 0:
                continue
            if observed >= limit:
                violation = BudgetViolation(
                    dimension=dimension, limit=float(limit), observed=float(observed)
                )
                if dimension not in {item.dimension for item in self.violations}:
                    self.violations.append(violation)
                    if self.on_violation is not None:
                        self.on_violation(violation)
            elif observed >= limit * budget.warn_at:
                key = f"warn:{dimension}"
                if key not in self.warnings_emitted:
                    self.warnings_emitted.add(key)
                    if self.on_warning is not None:
                        self.on_warning(
                            BudgetViolation(
                                dimension=dimension, limit=float(limit), observed=float(observed)
                            )
                        )

    def exceeded(self) -> list[BudgetViolation]:
        """Return every ceiling that has been breached."""
        with self._lock:
            self._check_limits()
            return list(self.violations)

    def is_exceeded(self) -> bool:
        """Report whether any ceiling has been breached."""
        return bool(self.exceeded())

    def remaining(self) -> dict[str, float]:
        """Return how much headroom is left on each configured ceiling."""
        budget = self.budget
        pairs = [
            ("tokens", self.usage.total_tokens, budget.max_tokens),
            ("prompt_tokens", self.usage.prompt_tokens, budget.max_prompt_tokens),
            ("completion_tokens", self.usage.completion_tokens, budget.max_completion_tokens),
            ("cost", self.cost.total, budget.max_cost),
            ("steps", self.steps, budget.max_steps),
            ("tool_calls", self.tool_calls, budget.max_tool_calls),
            ("seconds", self.elapsed_seconds(), budget.max_seconds),
            ("requests", self.requests, budget.max_requests),
        ]
        out: dict[str, float] = {}
        for dimension, observed, limit in pairs:
            if limit > 0:
                out[dimension] = round(max(0.0, limit - observed), 4)
        return out

    def utilization(self) -> dict[str, float]:
        """Return consumption as a 0..1 fraction of each configured ceiling."""
        budget = self.budget
        pairs = [
            ("tokens", self.usage.total_tokens, budget.max_tokens),
            ("cost", self.cost.total, budget.max_cost),
            ("steps", self.steps, budget.max_steps),
            ("tool_calls", self.tool_calls, budget.max_tool_calls),
            ("seconds", self.elapsed_seconds(), budget.max_seconds),
            ("requests", self.requests, budget.max_requests),
        ]
        return {
            dimension: round(min(1.0, observed / limit), 4)
            for dimension, observed, limit in pairs
            if limit > 0
        }

    # -- reporting -----------------------------------------------------------

    def tokens_per_second(self) -> float:
        """Return completion token throughput."""
        seconds = self.elapsed_seconds()
        if seconds <= 0 or self.usage.completion_tokens <= 0:
            return 0.0
        return round(self.usage.completion_tokens / seconds, 2)

    def average_latency_ms(self) -> float:
        """Return the mean request latency."""
        if not self.latencies:
            return 0.0
        return round(sum(self.latencies) / len(self.latencies), 2)

    def p95_latency_ms(self) -> float:
        """Return the 95th percentile request latency."""
        if not self.latencies:
            return 0.0
        from ..utils.numbers import percentile

        return percentile(self.latencies, 95)

    def cache_hit_rate(self) -> float:
        """Return the fraction of requests served from cache."""
        if not self.requests:
            return 0.0
        return round(self.cache_hits / self.requests, 4)

    def error_rate(self) -> float:
        """Return the fraction of requests that errored."""
        total = self.requests + self.errors
        return round(self.errors / total, 4) if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a full serializable telemetry snapshot."""
        with self._lock:
            return {
                "usage": self.usage.to_dict(),
                "cost": self.cost.to_dict(),
                "requests": self.requests,
                "steps": self.steps,
                "tool_calls": self.tool_calls,
                "errors": self.errors,
                "retries": self.retries,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hit_rate(),
                "error_rate": self.error_rate(),
                "elapsed_seconds": round(self.elapsed_seconds(), 3),
                "tokens_per_second": self.tokens_per_second(),
                "average_latency_ms": self.average_latency_ms(),
                "p95_latency_ms": self.p95_latency_ms(),
                "by_model": {model: usage.to_dict() for model, usage in self.by_model.items()},
                "cost_by_model": dict(self.cost_by_model),
                "budget": self.budget.to_dict(),
                "remaining": self.remaining(),
                "utilization": self.utilization(),
                "violations": [item.to_dict() for item in self.violations],
            }

    def summary_line(self) -> str:
        """Return a one line human summary of usage."""
        cost_text = f" cost={self.cost.total:.4f}{self.cost.currency}" if self.cost.total else ""
        return (
            f"tokens={self.usage.total_tokens} "
            f"(in={self.usage.prompt_tokens} out={self.usage.completion_tokens})"
            f" requests={self.requests} steps={self.steps} tools={self.tool_calls}"
            f" cache={self.cache_hit_rate() * 100:.0f}%"
            f" avg={self.average_latency_ms():.0f}ms{cost_text}"
        )

    def reset(self) -> None:
        """Clear every counter and restart the clock."""
        with self._lock:
            self.usage = Usage()
            self.cost = CostEstimate(currency=self.prices.currency)
            self.requests = self.steps = self.tool_calls = 0
            self.errors = self.cache_hits = self.retries = 0
            self.latencies.clear()
            self.by_model = defaultdict(Usage)
            self.cost_by_model = defaultdict(float)
            self.tokens_per_model = defaultdict(int)
            self.violations.clear()
            self.warnings_emitted.clear()
            self.started_at = time.monotonic()

    def merge(self, other: UsageTracker) -> None:
        """Fold another tracker's totals into this one."""
        with self._lock:
            self.usage.add(other.usage)
            self.cost.prompt_cost += other.cost.prompt_cost
            self.cost.completion_cost += other.cost.completion_cost
            self.requests += other.requests
            self.steps += other.steps
            self.tool_calls += other.tool_calls
            self.errors += other.errors
            self.cache_hits += other.cache_hits
            self.retries += other.retries
            self.latencies.extend(other.latencies)
            for model, usage in other.by_model.items():
                self.by_model[model].add(usage)
            for model, amount in other.cost_by_model.items():
                self.cost_by_model[model] = round(self.cost_by_model[model] + amount, 8)
            self.violations.extend(other.violations)


__all__ = [
    "BudgetConfig",
    "BudgetViolation",
    "PriceBook",
    "UsageTracker",
]
