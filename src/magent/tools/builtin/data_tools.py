"""Structured data tools: CSV, JSON, SQLite, and tabular analysis."""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from ...errors import ToolExecutionError
from ...tools.base import ToolContext, tool
from ...utils.json_utils import loads_lenient
from ...utils.tables import render_markdown_table

SAFE_SQL_PATTERN = re.compile(r"^\s*(select|with|pragma\s+table_info|explain)\b", re.IGNORECASE)
"""SQL statements the query tool is willing to run."""

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|vacuum|reindex|grant)\b",
    re.IGNORECASE,
)
"""SQL keywords that indicate a write, which the query tool refuses."""


@tool(category="data", risk="read", filesystem=True, tags=["csv", "core"])
def csv_read(
    path: str,
    delimiter: str = ",",
    has_header: bool = True,
    limit: int = 100,
    columns: str = "",
    where: str = "",
    sort_by: str = "",
    descending: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Read and filter a CSV or TSV file.

    Args:
        path: CSV file path inside the workspace
        delimiter: field separator, use "\\t" for TSV
        has_header: treat the first row as column names
        limit: maximum rows to return
        columns: comma separated column names to project
        where: simple filter as "column=value" or "column~substring"
        sort_by: column name to sort by
        descending: sort descending instead of ascending
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such CSV file: {target}")
    sep = "\t" if delimiter in {"\\t", "tab"} else (delimiter or ",")
    with target.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter=sep)
        rows = list(reader)
    if not rows:
        return {"path": str(target), "rows": 0, "columns": [], "records": []}
    if has_header:
        header = [name.strip() for name in rows[0]]
        body = rows[1:]
    else:
        width = max(len(row) for row in rows)
        header = [f"col_{index}" for index in range(width)]
        body = rows
    records = [
        dict(zip(header, row + [""] * (len(header) - len(row)), strict=False)) for row in body
    ]
    if where:
        records = _apply_where(records, where)
    if sort_by and records and sort_by in header:
        reverse = bool(descending)
        records.sort(key=lambda item: _sort_key(item.get(sort_by, "")), reverse=reverse)
    projection = [name.strip() for name in columns.split(",") if name.strip()] or header
    projected = [{key: record.get(key, "") for key in projection} for record in records[:limit]]
    return {
        "path": str(target),
        "rows": len(body),
        "returned": len(projected),
        "columns": header,
        "projected": projection,
        "truncated": len(records) > limit,
        "records": projected,
        "preview_markdown": render_markdown_table(
            [[record.get(key, "") for key in projection] for record in projected[:10]], projection
        ),
    }


def _apply_where(records: list[dict[str, Any]], clause: str) -> list[dict[str, Any]]:
    """Apply a ``column=value`` or ``column~substring`` filter."""
    if "~" in clause:
        key, _, needle = clause.partition("~")
        return [
            record
            for record in records
            if needle.lower() in str(record.get(key.strip(), "")).lower()
        ]
    if "=" in clause:
        key, _, value = clause.partition("=")
        return [record for record in records if str(record.get(key.strip(), "")) == value.strip()]
    if "!=" in clause:
        key, _, value = clause.partition("!=")
        return [record for record in records if str(record.get(key.strip(), "")) != value.strip()]
    raise ToolExecutionError("where must be 'column=value', 'column!=value', or 'column~substring'")


def _sort_key(value: Any) -> tuple[int, float, str]:
    """Return a sort key that orders numbers numerically and text lexically."""
    text = str(value)
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text.lower())


@tool(category="data", risk="read", tags=["csv", "stats"])
def csv_stats(
    path: str, delimiter: str = ",", columns: str = "", context: ToolContext | None = None
) -> dict[str, Any]:
    """Summarize a CSV file: row count, column types, and numeric ranges.

    Args:
        path: CSV file path
        delimiter: field separator
        columns: restrict the summary to these columns
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such CSV file: {target}")
    sep = "\t" if delimiter in {"\\t", "tab"} else (delimiter or ",")
    with target.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=sep)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    wanted = [name.strip() for name in columns.split(",") if name.strip()] or header
    summary: dict[str, Any] = {}
    for name in wanted:
        values = [str(row.get(name, "")) for row in rows]
        numeric: list[float] = []
        for value in values:
            try:
                numeric.append(float(value))
            except ValueError:
                continue
        entry: dict[str, Any] = {
            "type": "number"
            if numeric and len(numeric) == len([v for v in values if v])
            else "text",
            "non_empty": sum(1 for value in values if value.strip()),
            "empty": sum(1 for value in values if not value.strip()),
            "unique": len(set(values)),
        }
        if numeric:
            entry.update(
                {
                    "min": min(numeric),
                    "max": max(numeric),
                    "mean": round(sum(numeric) / len(numeric), 6),
                    "sum": round(sum(numeric), 6),
                }
            )
        else:
            top = Counter(value for value in values if value.strip()).most_common(3)
            entry["top_values"] = [{"value": value, "count": count} for value, count in top]
        summary[name] = entry
    return {"path": str(target), "rows": len(rows), "columns": header, "summary": summary}


@tool(category="data", risk="read", tags=["json", "query", "core"])
def json_query(data: str, path: str = "", default: Any = None) -> dict[str, Any]:
    """Extract a value from a JSON document using a dotted path.

    Args:
        data: the JSON document as text, or a JSON array/object
        path: dotted path such as "items[0].name" or "a.b.c", empty for the root
        default: value to return when the path is missing
    """
    parsed = loads_lenient(data)
    if parsed is None:
        raise ToolExecutionError("Input is not valid JSON")
    if not path:
        return {"path": "", "found": True, "value": parsed, "type": type(parsed).__name__}
    cursor: Any = parsed
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    for token in tokens:
        try:
            if token.startswith("[") and token.endswith("]"):
                cursor = cursor[int(token[1:-1])]
            elif isinstance(cursor, dict):
                cursor = cursor[token]
            elif isinstance(cursor, list) and token.isdigit():
                cursor = cursor[int(token)]
            else:
                cursor = None
                break
        except (IndexError, KeyError, TypeError, ValueError):
            cursor = None
            break
    found = cursor is not None
    return {
        "path": path,
        "found": found,
        "value": cursor if found else default,
        "type": type(cursor).__name__ if found else type(default).__name__,
        "items": len(cursor) if isinstance(cursor, (dict, list)) else None,
    }


@tool(category="data", risk="read", tags=["json", "flatten"])
def json_flatten(data: str, separator: str = ".", max_keys: int = 500) -> dict[str, Any]:
    """Flatten a nested JSON document into dotted-path keys.

    Args:
        data: the JSON document as text
        separator: path separator to use
        max_keys: maximum number of flattened keys to return
    """
    from ...utils.json_utils import flatten

    parsed = loads_lenient(data)
    if not isinstance(parsed, (dict, list)):
        raise ToolExecutionError("Input must be a JSON object or array")
    flat = flatten(parsed, separator=separator)
    items = list(flat.items())
    return {
        "keys": len(items),
        "returned": min(len(items), max_keys),
        "truncated": len(items) > max_keys,
        "flat": dict(items[:max_keys]),
    }


@tool(category="data", risk="read", filesystem=True, tags=["sqlite", "sql", "core"])
def sqlite_query(
    path: str,
    sql: str,
    params: str = "",
    limit: int = 200,
    allow_write: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Run a read-only SQL query against a SQLite database.

    Args:
        path: database file, or ":memory:" for an in-memory database
        sql: a SELECT or WITH statement
        params: JSON array of positional parameters
        limit: maximum rows to return
        allow_write: permit INSERT/UPDATE/DELETE (requires approval)
    """
    ctx = context or ToolContext()
    statement = str(sql).strip().rstrip(";")
    if not statement:
        raise ToolExecutionError("Empty SQL statement")
    if not allow_write and not SAFE_SQL_PATTERN.match(statement):
        raise ToolExecutionError(
            "Only SELECT/WITH statements are allowed unless allow_write is set"
        )
    if not allow_write and FORBIDDEN_SQL.search(statement):
        raise ToolExecutionError(
            "Statement contains a write keyword; pass allow_write=true to permit it"
        )
    if ";" in statement:
        raise ToolExecutionError("Multiple statements are not allowed")
    if path != ":memory:":
        target = ctx.resolve_path(path)
        if ctx.sandbox is not None:
            ctx.sandbox.check_read(target)
        if not target.is_file():
            raise ToolExecutionError(f"No such database: {target}")
        database = str(target)
    else:
        database = ":memory:"
    arguments: list[Any] = []
    if params:
        parsed = loads_lenient(params)
        if not isinstance(parsed, list):
            raise ToolExecutionError("params must be a JSON array")
        arguments = list(parsed)
    connection = sqlite3.connect(database, timeout=10.0)
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(statement, arguments)
        columns = (
            [description[0] for description in cursor.description] if cursor.description else []
        )
        rows = [dict(row) for row in cursor.fetchmany(max(1, limit))]
        total = cursor.rowcount
        if allow_write:
            connection.commit()
    except sqlite3.Error as exc:
        raise ToolExecutionError(f"SQL error: {exc}") from exc
    finally:
        connection.close()
    return {
        "database": database,
        "sql": statement,
        "columns": columns,
        "rows": len(rows),
        "rowcount": total,
        "truncated": len(rows) >= limit,
        "records": rows,
        "preview_markdown": render_markdown_table(
            [[_render_cell(record.get(column)) for column in columns] for record in rows[:10]],
            columns,
        )
        if columns
        else "",
    }


def _render_cell(value: Any) -> str:
    """Render one SQL cell for a Markdown preview."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False)[:80]
    return str(value)


@tool(category="data", risk="read", filesystem=True, tags=["sqlite", "schema"])
def sqlite_schema(path: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Describe every table, index, and view in a SQLite database.

    Args:
        path: database file inside the workspace
    """
    ctx = context or ToolContext()
    target = ctx.resolve_path(path)
    if ctx.sandbox is not None:
        ctx.sandbox.check_read(target)
    if not target.is_file():
        raise ToolExecutionError(f"No such database: {target}")
    connection = sqlite3.connect(str(target), timeout=10.0)
    try:
        cursor = connection.execute(
            "SELECT name, type, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
        tables: list[dict[str, Any]] = []
        for name, kind, sql in cursor.fetchall():
            entry: dict[str, Any] = {"name": name, "type": kind, "sql": sql or ""}
            if kind in {"table", "view"}:
                info = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
                entry["columns"] = [
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "pk": bool(row[5]),
                    }
                    for row in info
                ]
                count = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
                entry["row_count"] = int(count[0]) if count else 0
            tables.append(entry)
    except sqlite3.Error as exc:
        raise ToolExecutionError(f"Could not read schema: {exc}") from exc
    finally:
        connection.close()
    return {"path": str(target), "objects": len(tables), "schema": tables}


@tool(category="data", risk="safe", tags=["json", "csv", "convert"])
def json_to_csv(data: str, columns: str = "", flatten: bool = True) -> dict[str, Any]:
    """Convert a JSON array of objects into CSV text.

    Args:
        data: a JSON array of objects
        columns: explicit column order, empty to infer from the data
        flatten: flatten nested objects into dotted column names
    """
    from ...utils.json_utils import flatten as flatten_dict

    parsed = loads_lenient(data)
    if isinstance(parsed, dict):
        for key in ("data", "items", "results", "records", "rows"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list) or not parsed:
        raise ToolExecutionError("Input must be a non-empty JSON array")
    records: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            records.append({"value": item})
        elif flatten:
            records.append({key: value for key, value in flatten_dict(item).items()})
        else:
            records.append(item)
    if columns:
        header = [name.strip() for name in columns.split(",") if name.strip()]
    else:
        header = list(dict.fromkeys(key for record in records for key in record))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for record in records:
        writer.writerow([_render_cell(record.get(key, "")) for key in header])
    return {"rows": len(records), "columns": header, "csv": buffer.getvalue()}


@tool(category="data", risk="safe", tags=["group", "aggregate"])
def group_aggregate(
    records: str, group_by: str, aggregate: str = "count", value_field: str = "", top: int = 20
) -> dict[str, Any]:
    """Group JSON records by a field and aggregate them.

    Args:
        records: a JSON array of objects
        group_by: field name to group on
        aggregate: count, sum, avg, min, max, or unique
        value_field: field to aggregate, required for sum/avg/min/max
        top: how many groups to return, ordered by the aggregate
    """
    parsed = loads_lenient(records)
    if not isinstance(parsed, list):
        raise ToolExecutionError("records must be a JSON array")
    buckets: dict[str, list[Any]] = defaultdict(list)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = str(item.get(group_by, ""))
        buckets[key].append(item.get(value_field) if value_field else item)
    groups: list[dict[str, Any]] = []
    for key, values in buckets.items():
        entry: dict[str, Any] = {"group": key, "count": len(values)}
        if aggregate == "count":
            entry["value"] = len(values)
        elif aggregate in {"sum", "avg", "min", "max"}:
            numbers = [_to_number(value) for value in values if _to_number(value) is not None]
            if not numbers:
                entry["value"] = None
            elif aggregate == "sum":
                entry["value"] = round(sum(numbers), 6)
            elif aggregate == "avg":
                entry["value"] = round(sum(numbers) / len(numbers), 6)
            elif aggregate == "min":
                entry["value"] = min(numbers)
            else:
                entry["value"] = max(numbers)
        elif aggregate == "unique":
            entry["value"] = sorted({str(value) for value in values})[:20]
            entry["unique_count"] = len({str(value) for value in values})
        else:
            raise ToolExecutionError(f"Unknown aggregate {aggregate!r}")
        groups.append(entry)
    reverse = aggregate != "min"
    groups.sort(
        key=lambda item: (
            (
                item["value"] is None,
                -(item["value"] or 0) if isinstance(item.get("value"), (int, float)) else 0,
            )
            if reverse
            else (item["value"] is None, item.get("value") or 0)
        )
    )
    return {
        "group_by": group_by,
        "aggregate": aggregate,
        "groups": len(groups),
        "records": len(parsed),
        "returned": min(len(groups), top),
        "results": groups[:top],
    }


def _to_number(value: Any) -> float | None:
    """Coerce a value to float, or return ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


TOOLS: list[Any] = [
    csv_read,
    csv_stats,
    json_query,
    json_flatten,
    sqlite_query,
    sqlite_schema,
    json_to_csv,
    group_aggregate,
]
"""Every tool defined in this module."""


def load(registry: Any) -> int:
    """Register every data tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = [
    "FORBIDDEN_SQL",
    "SAFE_SQL_PATTERN",
    "TOOLS",
    "csv_read",
    "csv_stats",
    "group_aggregate",
    "json_flatten",
    "json_query",
    "json_to_csv",
    "load",
    "sqlite_query",
    "sqlite_schema",
]
