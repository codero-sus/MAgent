"""Numeric helpers and small pure-Python statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    """Integer variant of :func:`clamp`."""
    return int(clamp(float(value), float(low), float(high)))


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between ``a`` and ``b`` at ``t`` in 0..1."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def round_to(value: float, digits: int = 2) -> float:
    """Round ``value`` to ``digits`` decimals, avoiding ``-0.0``."""
    result = round(float(value), digits)
    return 0.0 if result == 0 else result


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    items = [float(value) for value in values]
    return round(sum(items) / len(items), 6) if items else 0.0


def median(values: Iterable[float]) -> float:
    """Return the median, averaging the two central values when even."""
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2.0


def percentile(values: Iterable[float], pct: float) -> float:
    """Return the ``pct``-th percentile using linear interpolation."""
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    rank = clamp(pct, 0.0, 100.0) / 100.0 * (len(items) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return items[int(rank)]
    return lerp(items[low], items[high], rank - low)


def stddev(values: Iterable[float]) -> float:
    """Return the population standard deviation."""
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    average = sum(items) / len(items)
    variance = sum((item - average) ** 2 for item in items) / len(items)
    return round(math.sqrt(variance), 6)


def sum_floats(values: Iterable[float], *, digits: int = 6) -> float:
    """Sum floats with a final rounding pass to hide binary drift."""
    return round_to(math.fsum(float(value) for value in values), digits)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without raising on a zero denominator."""
    return numerator / denominator if denominator else default


def percentage(part: float, whole: float, *, digits: int = 2) -> float:
    """Return ``part`` as a percentage of ``whole``."""
    return round_to(safe_div(part, whole) * 100.0, digits)


def geometric_mean(values: Iterable[float]) -> float:
    """Return the geometric mean of positive values."""
    items = [float(value) for value in values if float(value) > 0]
    if not items:
        return 0.0
    return round(math.exp(math.fsum(math.log(item) for item in items) / len(items)), 6)


def harmonic_mean(values: Iterable[float]) -> float:
    """Return the harmonic mean, useful for averaging rates."""
    items = [float(value) for value in values if float(value) > 0]
    if not items:
        return 0.0
    return round(len(items) / math.fsum(1.0 / item for item in items), 6)


def moving_average(values: Sequence[float], window: int = 3) -> list[float]:
    """Return a simple moving average series of the same length."""
    if window <= 1 or not values:
        return [float(value) for value in values]
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out.append(mean(values[start : index + 1]))
    return out


def normalize(values: Iterable[float]) -> list[float]:
    """Min-max normalize a series into ``0..1``."""
    items = [float(value) for value in values]
    if not items:
        return []
    low, high = min(items), max(items)
    if high == low:
        return [0.0] * len(items)
    return [round((item - low) / (high - low), 6) for item in items]


def zscore(values: Iterable[float]) -> list[float]:
    """Return z-scores for a series, or zeros when variance is nil."""
    items = [float(value) for value in values]
    if not items:
        return []
    average = mean(items)
    deviation = stddev(items)
    if deviation == 0:
        return [0.0] * len(items)
    return [round((item - average) / deviation, 6) for item in items]


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exponent = math.exp(x)
    return exponent / (1.0 + exponent)


def softmax(values: Iterable[float]) -> list[float]:
    """Return a probability distribution over ``values``."""
    items = [float(value) for value in values]
    if not items:
        return []
    top = max(items)
    exponentials = [math.exp(item - top) for item in items]
    total = math.fsum(exponentials)
    return [round(item / total, 8) for item in exponentials]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(math.fsum(a * a for a in left))
    norm_right = math.sqrt(math.fsum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return round(dot / (norm_left * norm_right), 8)


def euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the L2 distance between two vectors."""
    if not left or len(left) != len(right):
        return float("inf")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=False)))


def manhattan_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the L1 distance between two vectors."""
    if not left or len(left) != len(right):
        return float("inf")
    return math.fsum(abs(a - b) for a, b in zip(left, right, strict=False))


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two vectors."""
    return math.fsum(a * b for a, b in zip(left, right, strict=False))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return a unit-length copy of ``vector``."""
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def int_from_bytes(data: bytes, *, signed: bool = False) -> int:
    """Convert bytes to an integer with explicit endianness."""
    return int.from_bytes(data, "big", signed=signed)


def bits_required(max_value: int) -> int:
    """Return how many bits are needed to represent ``max_value``."""
    return max(1, int(max_value).bit_length())


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two >= ``value``."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def buckets(count: int, size: int) -> int:
    """Return how many fixed-size buckets ``count`` items need."""
    if size <= 0:
        return 0
    return (count + size - 1) // size


def weighted_choice(weights: Sequence[float], random_value: float) -> int:
    """Pick an index from ``weights`` given a uniform ``random_value`` in 0..1."""
    total = math.fsum(max(0.0, float(weight)) for weight in weights)
    if total <= 0:
        return 0
    target = clamp(random_value, 0.0, 1.0) * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += max(0.0, float(weight))
        if running >= target:
            return index
    return len(weights) - 1


__all__ = [
    "bits_required",
    "buckets",
    "clamp",
    "clamp_int",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "geometric_mean",
    "harmonic_mean",
    "int_from_bytes",
    "l2_normalize",
    "lerp",
    "manhattan_distance",
    "mean",
    "median",
    "moving_average",
    "next_power_of_two",
    "normalize",
    "percentage",
    "percentile",
    "round_to",
    "safe_div",
    "sigmoid",
    "softmax",
    "stddev",
    "sum_floats",
    "weighted_choice",
    "zscore",
]
