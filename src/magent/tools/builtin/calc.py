"""Math, statistics, unit conversion, and date/time tools.

Models are unreliable at arithmetic. These tools give them an exact calculator,
a statistics package, a unit converter, and a timezone-aware clock, all
implemented with the standard library only.
"""

from __future__ import annotations

import ast
import calendar
import math
import random as random_module
import re
import statistics
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import tool
from ...utils.numbers import (
    cosine_similarity,
    geometric_mean,
    harmonic_mean,
    mean,
    median,
    percentile,
    stddev,
)
from ...utils.timeutil import format_duration, parse_duration

_ALLOWED_MATH: dict[str, Any] = {
    name: getattr(math, name)
    for name in (
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "comb",
        "copysign",
        "cos",
        "cosh",
        "degrees",
        "dist",
        "erf",
        "erfc",
        "exp",
        "expm1",
        "fabs",
        "factorial",
        "floor",
        "fmod",
        "frexp",
        "fsum",
        "gamma",
        "gcd",
        "hypot",
        "isclose",
        "isfinite",
        "isinf",
        "isnan",
        "ldexp",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "modf",
        "perm",
        "pow",
        "radians",
        "remainder",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
        "prod",
    )
}
"""Whitelisted math functions available to the calculator."""

_ALLOWED_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "len": len,
    "True": True,
    "False": False,
    "None": None,
    **_ALLOWED_MATH,
}
"""Names the calculator expression may reference."""

_EXPRESSION_BLOCKERS = re.compile(
    r"(__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bgetattr\b|\bsetattr\b|\bos\b|\bsys\b|"
    r"\bsubprocess\b|\blambda\b|\bglobals\b|\blocals\b|\bcompile\b|\binput\b|\bexit\b|\bquit\b)"
)
"""Patterns that make an expression unsafe to evaluate."""

UNIT_FACTORS: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "um": 1e-6,
        "nm": 1e-9,
        "in": 0.0254,
        "inch": 0.0254,
        "ft": 0.3048,
        "foot": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344,
        "mile": 1609.344,
        "nmi": 1852.0,
        "ly": 9.4607e15,
    },
    "mass": {
        "g": 1.0,
        "kg": 1000.0,
        "mg": 0.001,
        "ug": 1e-6,
        "t": 1e6,
        "tonne": 1e6,
        "lb": 453.59237,
        "oz": 28.349523125,
        "st": 6350.29318,
    },
    "time": {
        "s": 1.0,
        "ms": 0.001,
        "us": 1e-6,
        "ns": 1e-9,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
        "wk": 604800.0,
        "yr": 31557600.0,
    },
    "data": {
        "b": 1.0,
        "kb": 1000.0,
        "mb": 1000.0**2,
        "gb": 1000.0**3,
        "tb": 1000.0**4,
        "kib": 1024.0,
        "mib": 1024.0**2,
        "gib": 1024.0**3,
        "tib": 1024.0**4,
        "bit": 0.125,
        "kbit": 125.0,
        "mbit": 125000.0,
        "gbit": 125000000.0,
    },
    "temperature": {},
    "area": {
        "m2": 1.0,
        "km2": 1e6,
        "cm2": 1e-4,
        "ha": 10000.0,
        "acre": 4046.8564224,
        "ft2": 0.09290304,
        "in2": 0.00064516,
        "mi2": 2589988.110336,
    },
    "volume": {
        "l": 1.0,
        "ml": 0.001,
        "m3": 1000.0,
        "cm3": 0.001,
        "gal": 3.785411784,
        "qt": 0.946352946,
        "pt": 0.473176473,
        "cup": 0.2365882365,
        "floz": 0.0295735295625,
    },
    "speed": {
        "mps": 1.0,
        "kph": 1 / 3.6,
        "mph": 0.44704,
        "knot": 0.514444444,
        "fps": 0.3048,
    },
    "pressure": {
        "pa": 1.0,
        "kpa": 1000.0,
        "mpa": 1e6,
        "bar": 100000.0,
        "atm": 101325.0,
        "psi": 6894.757293168,
        "torr": 133.322368,
        "mmhg": 133.322387415,
    },
    "energy": {
        "j": 1.0,
        "kj": 1000.0,
        "cal": 4.184,
        "kcal": 4184.0,
        "wh": 3600.0,
        "kwh": 3.6e6,
        "ev": 1.602176634e-19,
        "btu": 1055.05585,
    },
    "power": {
        "w": 1.0,
        "kw": 1000.0,
        "mw": 1e6,
        "hp": 745.699872,
        "btuh": 0.29307107,
    },
    "frequency": {
        "hz": 1.0,
        "khz": 1000.0,
        "mhz": 1e6,
        "ghz": 1e9,
        "rpm": 1 / 60.0,
    },
    "angle": {
        "deg": 1.0,
        "rad": 180.0 / math.pi,
        "grad": 0.9,
        "turn": 360.0,
        "arcmin": 1 / 60.0,
        "arcsec": 1 / 3600.0,
    },
}
"""Conversion factors grouped by dimension, each normalized to a base unit."""

UNIT_ALIASES: dict[str, str] = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "kilometer": "km",
    "kilometers": "km",
    "centimeter": "cm",
    "millimeter": "mm",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
    "yard": "yd",
    "yards": "yd",
    "mile": "mi",
    "miles": "mi",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "pound": "lb",
    "pounds": "lb",
    "ounce": "oz",
    "ounces": "oz",
    "ton": "t",
    "tonne": "t",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "day": "d",
    "days": "d",
    "week": "wk",
    "weeks": "wk",
    "year": "yr",
    "byte": "b",
    "bytes": "b",
    "kilobyte": "kb",
    "megabyte": "mb",
    "gigabyte": "gb",
    "terabyte": "tb",
    "celsius": "c",
    "centigrade": "c",
    "fahrenheit": "f",
    "kelvin": "k",
    "liter": "l",
    "litre": "l",
    "liters": "l",
    "gallon": "gal",
    "gallons": "gal",
}
"""Long unit names mapped to their short symbols."""


@tool(category="math", risk="safe", deterministic=True, tags=["calculator", "core"])
def calculate(expression: str, precision: int = 10) -> dict[str, Any]:
    """Evaluate an arithmetic expression safely and exactly.

    Supports + - * / // % **, parentheses, and the math module's functions
    (sqrt, log, sin, cos, factorial, ...). No names, imports, or attributes are
    reachable, so the expression cannot execute arbitrary code.

    Args:
        expression: the arithmetic expression to evaluate
        precision: decimal places to round the result to
    """
    text = str(expression).strip()
    if not text:
        raise ToolExecutionError("Empty expression")
    if len(text) > 2000:
        raise ToolExecutionError("Expression too long (max 2000 characters)")
    if _EXPRESSION_BLOCKERS.search(text):
        raise ToolExecutionError(
            "Expression contains a disallowed token; only arithmetic and math functions are permitted"
        )
    normalized = text.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(f"Could not parse expression: {exc}") from exc
    audit_expression_ast(tree)
    try:
        code = compile(tree, "<expression>", "eval", dont_inherit=True)
        value = eval(code, {"__builtins__": {}}, dict(_ALLOWED_NAMES))
    except ZeroDivisionError as exc:
        raise ToolExecutionError("Division by zero") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Could not evaluate expression: {exc}") from exc
    if isinstance(value, complex):
        return {
            "expression": text,
            "result": {"real": value.real, "imag": value.imag},
            "type": "complex",
        }
    if isinstance(value, bool):
        return {"expression": text, "result": value, "type": "boolean"}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"expression": text, "result": str(value), "type": "float"}
        rounded = round(value, max(0, precision))
        return {
            "expression": text,
            "result": rounded,
            "exact": value,
            "type": "float",
            "integer_valued": float(rounded).is_integer(),
        }
    return {"expression": text, "result": value, "type": type(value).__name__}


ALLOWED_AST_NODES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.Num,
        ast.Str,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.IfExp,
        ast.Starred,
        ast.keyword,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.Invert,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    }
)
"""AST node types the calculator is willing to evaluate."""

ALLOWED_CALLEES: frozenset[str] = frozenset(_ALLOWED_NAMES)
"""Names the calculator may call or reference."""


def audit_expression_ast(tree: ast.AST) -> None:
    """Reject an expression whose AST contains anything outside the whitelist.

    This is the real safety layer: blocking names textually is easy to evade,
    but an attribute access, comprehension, or lambda cannot be expressed with
    only the node types above.
    """
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_NODES:
            raise ToolExecutionError(f"Expression uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and not _callee_allowed(node.func):
            raise ToolExecutionError("Only whitelisted math functions may be called")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_CALLEES:
            raise ToolExecutionError(f"Unknown name {node.id!r} in expression")


def _callee_allowed(node: ast.AST) -> bool:
    """Report whether a call target is a whitelisted plain name."""
    return isinstance(node, ast.Name) and node.id in ALLOWED_CALLEES


@tool(category="math", risk="safe", tags=["statistics"])
def compute_statistics(values: str, sort_result: bool = True) -> dict[str, Any]:
    """Compute descriptive statistics for a list of numbers.

    Args:
        values: comma, space, or newline separated numbers, or a JSON array
        sort_result: include the sorted series in the output
    """
    numbers = _parse_numbers(values)
    if not numbers:
        raise ToolExecutionError("No numbers found in the input")
    ordered = sorted(numbers)
    try:
        mode = statistics.multimode(numbers)
    except statistics.StatisticsError:
        mode = []
    quartiles = statistics.quantiles(numbers, n=4, method="inclusive") if len(numbers) >= 2 else []
    result: dict[str, Any] = {
        "count": len(numbers),
        "sum": round(math.fsum(numbers), 10),
        "mean": round(mean(numbers), 10),
        "median": round(median(numbers), 10),
        "mode": mode,
        "min": ordered[0],
        "max": ordered[-1],
        "range": round(ordered[-1] - ordered[0], 10),
        "stddev": round(stddev(numbers), 10),
        "variance": round(statistics.pvariance(numbers), 10) if len(numbers) > 1 else 0.0,
        "geometric_mean": round(geometric_mean(numbers), 10),
        "harmonic_mean": round(harmonic_mean([value for value in numbers if value > 0]), 10),
        "p25": round(percentile(numbers, 25), 10),
        "p75": round(percentile(numbers, 75), 10),
        "p90": round(percentile(numbers, 90), 10),
        "p95": round(percentile(numbers, 95), 10),
        "p99": round(percentile(numbers, 99), 10),
        "iqr": round(percentile(numbers, 75) - percentile(numbers, 25), 10),
        "zeros": sum(1 for value in numbers if value == 0),
        "negatives": sum(1 for value in numbers if value < 0),
    }
    if quartiles:
        result["quartiles"] = [round(value, 10) for value in quartiles]
    if sort_result:
        result["sorted"] = ordered[:500]
    return result


def _parse_numbers(values: Any) -> list[float]:
    """Parse numbers out of a string, list, or JSON payload."""
    if isinstance(values, (list, tuple)):
        source = list(values)
    else:
        text = str(values).strip()
        if text.startswith("["):
            import json

            try:
                source = json.loads(text)
            except json.JSONDecodeError:
                source = []
        else:
            source = re.split(r"[,\s;]+", text)
    out: list[float] = []
    for item in source:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            out.append(float(item))
            continue
        try:
            out.append(float(str(item).strip()))
        except ValueError:
            continue
    return out


@tool(category="math", risk="safe", tags=["convert", "units"])
def convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Convert a number between units of the same dimension.

    Handles length, mass, time, data, temperature, area, volume, speed,
    pressure, energy, power, frequency, and angle.

    Args:
        value: the numeric value to convert
        from_unit: source unit name or symbol
        to_unit: target unit name or symbol
    """
    source = UNIT_ALIASES.get(str(from_unit).lower(), str(from_unit).lower())
    target = UNIT_ALIASES.get(str(to_unit).lower(), str(to_unit).lower())
    if source in {"c", "f", "k"} and target in {"c", "f", "k"}:
        converted = _convert_temperature(float(value), source, target)
        return {
            "value": float(value),
            "from": source,
            "to": target,
            "result": round(converted, 10),
            "dimension": "temperature",
        }
    for dimension, table in UNIT_FACTORS.items():
        if source in table and target in table:
            base = float(value) * table[source]
            converted = base / table[target]
            return {
                "value": float(value),
                "from": source,
                "to": target,
                "result": round(converted, 10),
                "dimension": dimension,
                "base_unit": _base_unit(dimension),
            }
    if source in {"c", "f", "k"} or target in {"c", "f", "k"}:
        raise ToolExecutionError(
            f"Cannot convert {from_unit!r} to {to_unit!r}; temperature needs both ends specified"
        )
    known_source = [dimension for dimension, table in UNIT_FACTORS.items() if source in table]
    raise ToolExecutionError(
        f"Unknown or incompatible units {from_unit!r} -> {to_unit!r}."
        + (f" {from_unit!r} is a {known_source[0]} unit." if known_source else "")
    )


def _base_unit(dimension: str) -> str:
    """Return the unit whose factor is 1.0 for a dimension."""
    table = UNIT_FACTORS.get(dimension, {})
    for unit, factor in table.items():
        if factor == 1.0:
            return unit
    return ""


def _convert_temperature(value: float, source: str, target: str) -> float:
    """Convert between Celsius, Fahrenheit, and Kelvin."""
    if source == "c":
        celsius = value
    elif source == "f":
        celsius = (value - 32.0) * 5.0 / 9.0
    else:
        celsius = value - 273.15
    if target == "c":
        return celsius
    if target == "f":
        return celsius * 9.0 / 5.0 + 32.0
    return celsius + 273.15


@tool(category="math", risk="safe", tags=["base", "convert"])
def convert_base(number: str, from_base: int = 10, to_base: int = 16) -> dict[str, Any]:
    """Convert an integer between numeric bases from 2 to 36.

    Args:
        number: the digits to convert
        from_base: base of the input digits
        to_base: base for the output digits
    """
    if not 2 <= from_base <= 36 or not 2 <= to_base <= 36:
        raise ToolExecutionError("Bases must be between 2 and 36")
    try:
        decimal_value = int(str(number).strip().replace("_", ""), from_base)
    except ValueError as exc:
        raise ToolExecutionError(f"{number!r} is not valid base-{from_base}: {exc}") from exc
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if decimal_value == 0:
        rendered = "0"
    else:
        negative = decimal_value < 0
        remaining = abs(decimal_value)
        chars: list[str] = []
        while remaining:
            chars.append(digits[remaining % to_base])
            remaining //= to_base
        rendered = ("-" if negative else "") + "".join(reversed(chars))
    return {
        "input": number,
        "from_base": from_base,
        "to_base": to_base,
        "decimal": decimal_value,
        "binary": bin(decimal_value)[2:].lstrip("-"),
        "octal": oct(decimal_value)[2:].lstrip("-"),
        "hex": hex(decimal_value)[2:].lstrip("-").upper(),
        "result": rendered,
        "bits": abs(decimal_value).bit_length(),
    }


@tool(category="math", risk="safe", tags=["random"])
def random_number(
    minimum: float = 0.0,
    maximum: float = 1.0,
    count: int = 1,
    integer: bool = False,
    seed: int = 0,
    distribution: str = "uniform",
) -> dict[str, Any]:
    """Generate random numbers with an optional reproducible seed.

    Args:
        minimum: lower bound
        maximum: upper bound
        count: how many values to generate
        integer: generate integers instead of floats
        seed: seed for reproducibility, 0 for unseeded
        distribution: uniform, normal, or choice
    """
    generator = random_module.Random(seed) if seed else random_module
    low, high = (minimum, maximum) if minimum <= maximum else (maximum, minimum)
    count = max(1, min(int(count), 1000))
    values: list[float] = []
    for _ in range(count):
        if distribution == "normal":
            mid = (low + high) / 2.0
            spread = max(1e-9, (high - low) / 6.0)
            sample = generator.gauss(mid, spread)
            sample = max(low, min(high, sample))
        else:
            sample = generator.uniform(low, high)
        values.append(round(sample) if integer else round(sample, 10))
    return {
        "count": len(values),
        "values": values,
        "value": values[0],
        "min": low,
        "max": high,
        "integer": integer,
        "seed": seed,
        "distribution": distribution,
    }


@tool(category="math", risk="safe", tags=["vector", "similarity"])
def vector_math(left: str, right: str = "", operation: str = "norm") -> dict[str, Any]:
    """Run vector operations on one or two numeric vectors.

    Args:
        left: first vector as a JSON array or comma separated numbers
        right: second vector, required for dot, cosine, distance, and add
        operation: norm, normalize, dot, cosine, euclidean, manhattan, add, subtract, scale
    """
    first = _parse_numbers(left)
    second = _parse_numbers(right) if right else []
    if not first:
        raise ToolExecutionError("The left vector has no numbers")
    if operation in {"dot", "cosine", "euclidean", "manhattan", "add", "subtract"} and len(
        second
    ) != len(first):
        raise ToolExecutionError("Both vectors must have the same length for this operation")
    if operation == "norm":
        return {
            "operation": operation,
            "dim": len(first),
            "result": round(math.sqrt(math.fsum(v * v for v in first)), 10),
        }
    if operation == "normalize":
        norm = math.sqrt(math.fsum(v * v for v in first)) or 1.0
        return {
            "operation": operation,
            "dim": len(first),
            "result": [round(v / norm, 10) for v in first],
        }
    if operation == "dot":
        return {
            "operation": operation,
            "dim": len(first),
            "result": round(math.fsum(a * b for a, b in zip(first, second, strict=True)), 10),
        }
    if operation == "cosine":
        return {
            "operation": operation,
            "dim": len(first),
            "result": cosine_similarity(first, second),
        }
    if operation == "euclidean":
        return {
            "operation": operation,
            "dim": len(first),
            "result": round(
                math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(first, second, strict=True))), 10
            ),
        }
    if operation == "manhattan":
        return {
            "operation": operation,
            "dim": len(first),
            "result": round(math.fsum(abs(a - b) for a, b in zip(first, second, strict=True)), 10),
        }
    if operation == "add":
        return {
            "operation": operation,
            "dim": len(first),
            "result": [a + b for a, b in zip(first, second, strict=True)],
        }
    if operation == "subtract":
        return {
            "operation": operation,
            "dim": len(first),
            "result": [a - b for a, b in zip(first, second, strict=True)],
        }
    if operation == "scale":
        factor = second[0] if second else 1.0
        return {
            "operation": operation,
            "dim": len(first),
            "result": [round(v * factor, 10) for v in first],
        }
    raise ToolExecutionError(f"Unknown operation {operation!r}")


@tool(category="datetime", risk="safe", tags=["now", "core"])
def current_time(timezone_name: str = "UTC", format: str = "") -> dict[str, Any]:
    """Return the current date and time.

    Args:
        timezone_name: UTC, local, or a fixed offset like +05:30
        format: optional strftime format for an extra rendered field
    """
    tz = _resolve_timezone(timezone_name)
    moment = datetime.now(tz)
    payload: dict[str, Any] = {
        "iso": moment.isoformat(),
        "utc": moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unix": int(moment.timestamp()),
        "unix_ms": int(moment.timestamp() * 1000),
        "date": moment.strftime("%Y-%m-%d"),
        "time": moment.strftime("%H:%M:%S"),
        "weekday": moment.strftime("%A"),
        "weekday_index": moment.weekday(),
        "year": moment.year,
        "month": moment.month,
        "day": moment.day,
        "hour": moment.hour,
        "minute": moment.minute,
        "second": moment.second,
        "timezone": timezone_name,
        "utc_offset": moment.strftime("%z"),
        "iso_week": moment.isocalendar()[1],
        "day_of_year": moment.timetuple().tm_yday,
        "is_leap_year": calendar.isleap(moment.year),
        "quarter": (moment.month - 1) // 3 + 1,
    }
    if format:
        payload["formatted"] = moment.strftime(format)
    return payload


def _resolve_timezone(name: str) -> timezone:
    """Resolve a timezone name or offset string to a fixed offset."""
    text = str(name or "UTC").strip()
    lowered = text.lower()
    if lowered in {"utc", "gmt", "z", "zulu"}:
        return UTC
    if lowered in {"local", "system", ""}:
        return datetime.now().astimezone().tzinfo or UTC
    match = re.fullmatch(r"([+-])(\d{1,2})(?::?(\d{2}))?", text)
    if match:
        sign = -1 if match.group(1) == "-" else 1
        hours = int(match.group(2))
        minutes = int(match.group(3) or 0)
        offset = timedelta(hours=hours, minutes=minutes) * sign
        if abs(offset) > timedelta(hours=14):
            raise ToolExecutionError("UTC offset must be within +/-14 hours")
        return timezone(offset, text)
    raise ToolExecutionError(f"Unknown timezone {name!r}; use UTC, local, or an offset like +05:30")


@tool(category="datetime", risk="safe", tags=["parse", "format"])
def format_datetime(
    value: str, output_format: str = "%Y-%m-%d %H:%M:%S", timezone_name: str = "UTC"
) -> dict[str, Any]:
    """Parse a date/time string and re-render it in another format.

    Args:
        value: ISO-8601 timestamp, unix seconds, or a common date string
        output_format: strftime format for the result
        timezone_name: timezone to render the result in
    """
    parsed = _parse_datetime(value)
    tz = _resolve_timezone(timezone_name)
    localized = parsed.astimezone(tz)
    return {
        "input": value,
        "parsed_iso": parsed.isoformat(),
        "formatted": localized.strftime(output_format),
        "timezone": timezone_name,
        "unix": int(localized.timestamp()),
    }


def _parse_datetime(value: str) -> datetime:
    """Parse a wide range of date/time representations."""
    text = str(value).strip()
    if not text:
        raise ToolExecutionError("Empty date/time value")
    if re.fullmatch(r"-?\d{10}", text):
        return datetime.fromtimestamp(int(text), tz=UTC)
    if re.fullmatch(r"-?\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000.0, tz=UTC)
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d %b %Y",
            "%H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ToolExecutionError(f"Could not parse date/time value {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@tool(category="datetime", risk="safe", tags=["diff", "duration"])
def datetime_diff(start: str, end: str) -> dict[str, Any]:
    """Compute the interval between two date/time values.

    Args:
        start: earlier date/time
        end: later date/time
    """
    first = _parse_datetime(start)
    second = _parse_datetime(end)
    delta = second - first
    total_seconds = delta.total_seconds()
    days = abs(delta.days)
    return {
        "start": first.isoformat(),
        "end": second.isoformat(),
        "seconds": total_seconds,
        "minutes": round(total_seconds / 60.0, 4),
        "hours": round(total_seconds / 3600.0, 4),
        "days": round(total_seconds / 86400.0, 4),
        "weeks": round(total_seconds / 604800.0, 4),
        "whole_days": days,
        "human": format_duration(abs(total_seconds), precision=0),
        "negative": total_seconds < 0,
        "business_days": _business_days(first, second),
    }


def _business_days(start: datetime, end: datetime) -> int:
    """Count weekdays between two dates, excluding weekends."""
    first, second = (start, end) if start <= end else (end, start)
    count = 0
    cursor = first.date()
    while cursor < second.date():
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


@tool(category="datetime", risk="safe", tags=["add", "offset"])
def datetime_add(value: str, offset: str, timezone_name: str = "UTC") -> dict[str, Any]:
    """Add or subtract a duration from a date/time.

    Args:
        value: base date/time
        offset: duration such as "3d", "-2h", "90m", "1w 2d"
        timezone_name: timezone to render the result in
    """
    base = _parse_datetime(value)
    seconds = parse_duration(offset)
    if seconds == 0 and offset.strip() not in {"0", "0s", ""}:
        seconds = _parse_signed_duration(offset)
    result = base + timedelta(seconds=seconds)
    tz = _resolve_timezone(timezone_name)
    localized = result.astimezone(tz)
    return {
        "input": value,
        "offset": offset,
        "offset_seconds": seconds,
        "result_iso": localized.isoformat(),
        "result_utc": localized.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result_date": localized.strftime("%Y-%m-%d"),
        "weekday": localized.strftime("%A"),
        "unix": int(localized.timestamp()),
    }


def _parse_signed_duration(text: str) -> float:
    """Parse a duration that may start with a minus sign."""
    stripped = str(text).strip()
    negative = stripped.startswith("-")
    seconds = parse_duration(stripped.lstrip("+-"))
    return -seconds if negative else seconds


@tool(category="datetime", risk="safe", tags=["calendar"])
def calendar_month(year: int, month: int) -> dict[str, Any]:
    """Return a text calendar for one month plus its metadata.

    Args:
        year: full year, e.g. 2026
        month: month number from 1 to 12
    """
    if not 1 <= month <= 12:
        raise ToolExecutionError("month must be between 1 and 12")
    if not 1 <= year <= 9999:
        raise ToolExecutionError("year must be between 1 and 9999")
    text = calendar.month(year, month)
    days = calendar.monthrange(year, month)[1]
    today = datetime.now(UTC).date()
    return {
        "year": year,
        "month": month,
        "month_name": calendar.month_name[month],
        "days": days,
        "first_weekday": calendar.day_name[calendar.monthrange(year, month)[0]],
        "calendar": text,
        "is_current_month": today.year == year and today.month == month,
        "is_leap_year": calendar.isleap(year),
    }


@tool(category="datetime", risk="safe", tags=["duration", "parse"])
def parse_duration_tool(duration: str) -> dict[str, Any]:
    """Parse a human duration string into seconds and other units.

    Args:
        duration: value such as "1h30m", "90s", "2d", "500ms"
    """
    seconds = parse_duration(duration)
    if seconds == 0 and duration.strip() not in {"0", "0s", ""}:
        raise ToolExecutionError(f"Could not parse duration {duration!r}")
    return {
        "input": duration,
        "seconds": seconds,
        "milliseconds": round(seconds * 1000, 3),
        "minutes": round(seconds / 60.0, 6),
        "hours": round(seconds / 3600.0, 6),
        "days": round(seconds / 86400.0, 6),
        "human": format_duration(seconds),
    }


TOOLS: list[Any] = [
    calculate,
    compute_statistics,
    convert_units,
    convert_base,
    random_number,
    vector_math,
    current_time,
    format_datetime,
    datetime_diff,
    datetime_add,
    calendar_month,
    parse_duration_tool,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every math and datetime tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "ALLOWED_AST_NODES",
    "ALLOWED_CALLEES",
    "TOOLS",
    "UNIT_ALIASES",
    "UNIT_FACTORS",
    "audit_expression_ast",
    "calculate",
    "calendar_month",
    "compute_statistics",
    "convert_base",
    "convert_units",
    "current_time",
    "datetime_add",
    "datetime_diff",
    "format_datetime",
    "load",
    "parse_duration_tool",
    "random_number",
    "vector_math",
]
