"""A dependency-free JSON Schema subset validator.

Tool arguments come from a language model, which means they arrive malformed
often enough to matter. Rather than trusting them, MAgent validates every call
against the tool's declared schema and returns a precise, model-readable error
so the next iteration can self-correct.

Supported keywords
------------------
``type``, ``properties``, ``required``, ``additionalProperties``, ``items``,
``enum``, ``const``, ``default``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
``exclusiveMaximum``, ``multipleOf``, ``minLength``, ``maxLength``, ``pattern``,
``format``, ``minItems``, ``maxItems``, ``uniqueItems``, ``minProperties``,
``maxProperties``, ``anyOf``, ``oneOf``, ``allOf``, ``not``, ``nullable``.

Unsupported keywords are ignored rather than rejected, which matches how model
providers treat schemas they do not fully implement.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}
"""Mapping from JSON Schema type names to Python types."""

FORMATS: dict[str, re.Pattern[str] | None] = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "uri": None,
    "uuid": re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "date-time": re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
    "ipv4": None,
    "ipv6": None,
    "hostname": re.compile(r"^[A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?(\.[A-Za-z0-9\-]+)*$"),
    "path": None,
}
"""Recognized ``format`` values and their validation regexes."""


@dataclass(slots=True)
class ValidationError:
    """One schema violation with a JSON-pointer style path."""

    path: str
    message: str
    keyword: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for tool output payloads."""
        return {
            "path": self.path,
            "message": self.message,
            "keyword": self.keyword,
            "expected": self.expected,
            "actual": _safe(self.actual),
        }

    def __str__(self) -> str:
        location = self.path or "<root>"
        return f"{location}: {self.message}"


def _safe(value: Any) -> Any:
    """Truncate a value so error payloads stay small."""
    text = repr(value)
    return text[:120] + ("..." if len(text) > 120 else "")


@dataclass(slots=True)
class ValidationResult:
    """The outcome of validating one document against one schema."""

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    coerced: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "valid": self.valid,
            "errors": [error.to_dict() for error in self.errors],
        }

    def error_text(self, *, limit: int = 5) -> str:
        """Return a compact human/model readable error summary."""
        if self.valid:
            return ""
        lines = [str(error) for error in self.errors[:limit]]
        if len(self.errors) > limit:
            lines.append(f"...and {len(self.errors) - limit} more")
        return "; ".join(lines)


class SchemaValidator:
    """Validate and coerce values against a JSON Schema subset."""

    def __init__(self, *, coerce: bool = True, strict_types: bool = False) -> None:
        self.coerce = coerce
        self.strict_types = strict_types

    # -- public API ----------------------------------------------------------

    def validate(self, value: Any, schema: Mapping[str, Any] | None) -> ValidationResult:
        """Validate ``value`` and return every violation found."""
        errors: list[ValidationError] = []
        coerced = self._validate(value, schema or {}, "", errors)
        return ValidationResult(valid=not errors, errors=errors, coerced=coerced)

    def is_valid(self, value: Any, schema: Mapping[str, Any] | None) -> bool:
        """Report whether ``value`` satisfies ``schema``."""
        return self.validate(value, schema).valid

    def validate_or_raise(self, value: Any, schema: Mapping[str, Any] | None) -> Any:
        """Validate and return the coerced value, raising on failure."""
        from ..errors import ToolValidationError

        result = self.validate(value, schema)
        if not result.valid:
            raise ToolValidationError(result.error_text())
        return result.coerced

    def apply_defaults(self, value: Any, schema: Mapping[str, Any]) -> Any:
        """Fill in declared defaults for missing object properties."""
        if not isinstance(value, dict) or not isinstance(schema, Mapping):
            return value
        properties = schema.get("properties") or {}
        if not isinstance(properties, Mapping):
            return value
        out = dict(value)
        for key, sub_schema in properties.items():
            if key in out or not isinstance(sub_schema, Mapping):
                continue
            if "default" in sub_schema:
                out[key] = sub_schema["default"]
        return out

    # -- internals -----------------------------------------------------------

    def _validate(
        self, value: Any, schema: Mapping[str, Any], path: str, errors: list[ValidationError]
    ) -> Any:
        """Recursively validate, appending errors and returning a coerced value."""
        if not isinstance(schema, Mapping):
            return value
        current = value
        if schema.get("nullable") and current is None:
            return current
        declared = schema.get("type")
        if declared is not None:
            current = self._check_type(current, declared, path, errors, schema)
            if current is _INVALID:
                return value
        if "const" in schema and current != schema["const"]:
            errors.append(ValidationError(path, f"must equal {_safe(schema['const'])}", "const",
                                          schema["const"], current))
        if "enum" in schema and isinstance(schema["enum"], Sequence) and current not in schema["enum"]:
            errors.append(ValidationError(path, f"must be one of {list(schema['enum'])}", "enum",
                                          list(schema["enum"]), current))
        for combiner in ("allOf", "anyOf", "oneOf"):
            sub_schemas = schema.get(combiner)
            if isinstance(sub_schemas, Sequence) and sub_schemas:
                self._check_combiner(combiner, current, sub_schemas, path, errors)
        if "not" in schema and isinstance(schema["not"], Mapping):
            probe: list[ValidationError] = []
            self._validate(current, schema["not"], path, probe)
            if not probe:
                errors.append(ValidationError(path, "must NOT match the 'not' schema", "not"))
        if isinstance(current, str):
            self._check_string(current, schema, path, errors)
        elif isinstance(current, bool):
            pass
        elif isinstance(current, (int, float)):
            self._check_number(current, schema, path, errors)
        elif isinstance(current, list):
            current = self._check_array(current, schema, path, errors)
        elif isinstance(current, dict):
            current = self._check_object(current, schema, path, errors)
        return current

    def _check_type(self, value: Any, declared: Any, path: str, errors: list[ValidationError],
                    schema: Mapping[str, Any]) -> Any:
        """Verify (and optionally coerce) the declared type."""
        names = [declared] if isinstance(declared, str) else list(declared)
        for name in names:
            expected = TYPE_CHECKS.get(str(name))
            if expected is None:
                return value
            if isinstance(value, expected) and not (
                str(name) in {"integer", "number"} and isinstance(value, bool)
            ):
                return value
        if self.coerce and not self.strict_types and names:
            coerced = _coerce_to(value, str(names[0]))
            if coerced is not _INVALID:
                return coerced
        errors.append(
            ValidationError(path, f"expected type {'/'.join(map(str, names))}, got {type(value).__name__}",
                            "type", names, type(value).__name__)
        )
        return _INVALID

    def _check_combiner(self, combiner: str, value: Any, schemas: Sequence[Any], path: str,
                        errors: list[ValidationError]) -> None:
        """Validate ``allOf`` / ``anyOf`` / ``oneOf`` branches."""
        matches = 0
        for sub_schema in schemas:
            if not isinstance(sub_schema, Mapping):
                continue
            probe: list[ValidationError] = []
            self._validate(value, sub_schema, path, probe)
            if not probe:
                matches += 1
        if combiner == "allOf" and matches != len([s for s in schemas if isinstance(s, Mapping)]):
            errors.append(ValidationError(path, "must satisfy every 'allOf' branch", combiner))
        elif combiner == "anyOf" and matches == 0:
            errors.append(ValidationError(path, "must satisfy at least one 'anyOf' branch", combiner))
        elif combiner == "oneOf" and matches != 1:
            errors.append(ValidationError(path, f"must satisfy exactly one 'oneOf' branch (matched {matches})",
                                          combiner))

    def _check_string(self, value: str, schema: Mapping[str, Any], path: str,
                      errors: list[ValidationError]) -> None:
        """Validate string constraints."""
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(ValidationError(path, f"must be at least {schema['minLength']} characters",
                                          "minLength", schema["minLength"], len(value)))
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(ValidationError(path, f"must be at most {schema['maxLength']} characters",
                                          "maxLength", schema["maxLength"], len(value)))
        pattern = schema.get("pattern")
        if pattern:
            try:
                if not re.search(str(pattern), value):
                    errors.append(ValidationError(path, f"must match pattern {pattern!r}", "pattern",
                                                  pattern, value))
            except re.error as exc:
                errors.append(ValidationError(path, f"invalid pattern in schema: {exc}", "pattern"))
        fmt = schema.get("format")
        if fmt and isinstance(FORMATS.get(str(fmt), False), (re.Pattern, type(None))):
            if not _check_format(value, str(fmt)):
                errors.append(ValidationError(path, f"must be a valid {fmt}", "format", fmt, value))

    def _check_number(self, value: float, schema: Mapping[str, Any], path: str,
                      errors: list[ValidationError]) -> None:
        """Validate numeric constraints."""
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(ValidationError(path, f"must be >= {schema['minimum']}", "minimum",
                                          schema["minimum"], value))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(ValidationError(path, f"must be <= {schema['maximum']}", "maximum",
                                          schema["maximum"], value))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(ValidationError(path, f"must be > {schema['exclusiveMinimum']}",
                                          "exclusiveMinimum", schema["exclusiveMinimum"], value))
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(ValidationError(path, f"must be < {schema['exclusiveMaximum']}",
                                          "exclusiveMaximum", schema["exclusiveMaximum"], value))
        if "multipleOf" in schema and schema["multipleOf"]:
            divisor = float(schema["multipleOf"])
            if divisor and abs((value / divisor) - round(value / divisor)) > 1e-9:
                errors.append(ValidationError(path, f"must be a multiple of {divisor}", "multipleOf",
                                              divisor, value))

    def _check_array(self, value: list, schema: Mapping[str, Any], path: str,
                     errors: list[ValidationError]) -> list:
        """Validate array constraints and each item."""
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(ValidationError(path, f"must have at least {schema['minItems']} items",
                                          "minItems", schema["minItems"], len(value)))
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(ValidationError(path, f"must have at most {schema['maxItems']} items",
                                          "maxItems", schema["maxItems"], len(value)))
        if schema.get("uniqueItems"):
            seen: list[str] = []
            for item in value:
                key = repr(item)
                if key in seen:
                    errors.append(ValidationError(path, "items must be unique", "uniqueItems"))
                    break
                seen.append(key)
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [
                self._validate(item, item_schema, f"{path}[{index}]", errors)
                for index, item in enumerate(value)
            ]
        if isinstance(item_schema, Sequence) and not isinstance(item_schema, (str, bytes)):
            out = list(value)
            for index, sub_schema in enumerate(item_schema):
                if index < len(out) and isinstance(sub_schema, Mapping):
                    out[index] = self._validate(out[index], sub_schema, f"{path}[{index}]", errors)
            return out
        return value

    def _check_object(self, value: dict, schema: Mapping[str, Any], path: str,
                      errors: list[ValidationError]) -> dict:
        """Validate object constraints and each declared property."""
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        out = dict(value)
        if isinstance(properties, Mapping):
            for key, sub_schema in properties.items():
                if key in out and isinstance(sub_schema, Mapping):
                    out[key] = self._validate(out[key], sub_schema, f"{path}.{key}" if path else key, errors)
                elif key not in out and isinstance(sub_schema, Mapping) and "default" in sub_schema:
                    out[key] = sub_schema["default"]
        if isinstance(required, Sequence):
            for key in required:
                if key not in value or value[key] is None:
                    errors.append(ValidationError(
                        f"{path}.{key}" if path else str(key),
                        f"missing required property '{key}'",
                        "required",
                        list(required),
                        sorted(value),
                    ))
        additional = schema.get("additionalProperties")
        if additional is False and isinstance(properties, Mapping):
            extras = sorted(set(value) - set(properties))
            if extras:
                errors.append(ValidationError(path, f"unexpected properties: {extras}",
                                              "additionalProperties"))
        if "minProperties" in schema and len(value) < int(schema["minProperties"]):
            errors.append(ValidationError(path, f"must have at least {schema['minProperties']} properties",
                                          "minProperties"))
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            errors.append(ValidationError(path, f"must have at most {schema['maxProperties']} properties",
                                          "maxProperties"))
        pattern_properties = schema.get("patternProperties")
        if isinstance(pattern_properties, Mapping):
            for pattern, sub_schema in pattern_properties.items():
                if not isinstance(sub_schema, Mapping):
                    continue
                try:
                    matcher = re.compile(str(pattern))
                except re.error:
                    continue
                for key in list(out):
                    if matcher.search(key):
                        out[key] = self._validate(out[key], sub_schema,
                                                  f"{path}.{key}" if path else key, errors)
        return out


_INVALID = object()
"""Sentinel meaning 'type check failed, stop descending'."""


def _coerce_to(value: Any, target: str) -> Any:
    """Best-effort coercion of a model-supplied value to the declared type."""
    try:
        if target == "string":
            if isinstance(value, (dict, list)):
                import json

                return json.dumps(value, default=str, ensure_ascii=False)
            return str(value)
        if target == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                return int(value.strip(), 10)
            if isinstance(value, float):
                return int(value)
            return _INVALID
        if target == "number":
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            return float(value)
        if target == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1", "on", "y"}:
                    return True
                if lowered in {"false", "no", "0", "off", "n", ""}:
                    return False
                return _INVALID
            return bool(value)
        if target == "array":
            if isinstance(value, (list, tuple)):
                return list(value)
            if isinstance(value, str):
                import json

                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return [value]
                return parsed if isinstance(parsed, list) else [parsed]
            return [value]
        if target == "object":
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                import json

                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else _INVALID
            return _INVALID
        if target == "null":
            return None if value in {"", "null", "None"} else _INVALID
    except (ValueError, TypeError, json.JSONDecodeError):
        return _INVALID
    return _INVALID


def _check_format(value: str, fmt: str) -> bool:
    """Validate a string against a named ``format``."""
    pattern = FORMATS.get(fmt, False)
    if pattern is False:
        return True
    if fmt in {"ipv4", "ipv6"}:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return address.version == (4 if fmt == "ipv4" else 6)
    if fmt == "uri":
        parts = urlsplit(value)
        return bool(parts.scheme and parts.netloc)
    if fmt == "path":
        return bool(value) and "\x00" not in value
    if pattern is None:
        return True
    return bool(pattern.match(value))


def validate(value: Any, schema: Mapping[str, Any] | None) -> ValidationResult:
    """Module level shortcut for a one-off validation."""
    return SchemaValidator().validate(value, schema)


def is_valid(value: Any, schema: Mapping[str, Any] | None) -> bool:
    """Module level shortcut returning only validity."""
    return SchemaValidator().is_valid(value, schema)


def object_schema(properties: dict[str, Any], required: list[str] | None = None,
                  *, additional: bool = False) -> dict[str, Any]:
    """Build an object schema dictionary from a property map."""
    schema: dict[str, Any] = {"type": "object", "properties": properties,
                              "additionalProperties": additional}
    if required:
        schema["required"] = list(required)
    return schema


def string_schema(description: str = "", **kwargs: Any) -> dict[str, Any]:
    """Build a string property schema."""
    schema: dict[str, Any] = {"type": "string"}
    if description:
        schema["description"] = description
    schema.update(kwargs)
    return schema


def integer_schema(description: str = "", **kwargs: Any) -> dict[str, Any]:
    """Build an integer property schema."""
    schema: dict[str, Any] = {"type": "integer"}
    if description:
        schema["description"] = description
    schema.update(kwargs)
    return schema


def number_schema(description: str = "", **kwargs: Any) -> dict[str, Any]:
    """Build a number property schema."""
    schema: dict[str, Any] = {"type": "number"}
    if description:
        schema["description"] = description
    schema.update(kwargs)
    return schema


def boolean_schema(description: str = "", **kwargs: Any) -> dict[str, Any]:
    """Build a boolean property schema."""
    schema: dict[str, Any] = {"type": "boolean"}
    if description:
        schema["description"] = description
    schema.update(kwargs)
    return schema


def array_schema(items: dict[str, Any], description: str = "", **kwargs: Any) -> dict[str, Any]:
    """Build an array property schema."""
    schema: dict[str, Any] = {"type": "array", "items": items}
    if description:
        schema["description"] = description
    schema.update(kwargs)
    return schema


def enum_schema(values: Sequence[Any], description: str = "") -> dict[str, Any]:
    """Build an enum property schema."""
    schema: dict[str, Any] = {"enum": list(values)}
    if description:
        schema["description"] = description
    return schema


__all__ = [
    "FORMATS",
    "SchemaValidator",
    "TYPE_CHECKS",
    "ValidationError",
    "ValidationResult",
    "array_schema",
    "boolean_schema",
    "enum_schema",
    "integer_schema",
    "is_valid",
    "number_schema",
    "object_schema",
    "string_schema",
    "validate",
]
