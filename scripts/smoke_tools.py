#!/usr/bin/env python3
"""Exercise every built-in tool against a throwaway workspace.

This is the tool layer's smoke test. It builds a temporary workspace, registers
the full built-in catalogue inside a sandbox that has no network access, then
dispatches a realistic call for every tool that can run offline. Tools that need
a live Cortex LLMHoster instance or an outbound network connection are listed in
:data:`SKIPPED` with the reason, and the script *fails* if a registered tool is
neither exercised nor explicitly skipped - so a new tool can never silently
escape coverage.

Run it with::

    make smoke-tools
    PYTHONPATH=src python3 scripts/smoke_tools.py --verbose
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root."""

sys.path.insert(0, str(REPO_ROOT / "src"))

from magent.config import load_config  # noqa: E402
from magent.tools.base import ToolContext, default_registry  # noqa: E402
from magent.tools.builtin import load_errors, tools_by_category  # noqa: E402
from magent.tools.sandbox import Sandbox, SandboxPolicy  # noqa: E402
from magent.types import ToolCall  # noqa: E402

SKIPPED: dict[str, str] = {
    "http_get": "requires outbound network access",
    "http_post": "requires outbound network access",
    "fetch_page": "requires outbound network access",
    "download_file": "requires outbound network access",
    "git_status": "requires a git repository",
    "git_log": "requires a git repository with history",
    "git_diff": "requires a git repository",
    "git_commit": "requires a git repository and user identity",
    "git_add": "requires a git repository",
    "git_branches": "requires a git repository",
    "git_blame": "requires a git repository with history",
    "git_remote_info": "requires a git repository",
    "git_push": "requires a git remote and credentials",
    "git_show": "requires a git repository with history",
    "run_script": "covered by run_command with an explicit interpreter",
    "remember": "requires an attached memory store",
    "recall": "requires an attached memory store",
    "list_memories": "requires an attached memory store",
    "forget": "requires an attached memory store",
    "export_memories": "requires an attached memory store",
    "compact_memories": "requires an attached memory store",
    "ask_llm": "requires a live LLM backend",
    "cortex_health": "requires a running Cortex LLMHoster instance",
    "cortex_models": "requires a running Cortex LLMHoster instance",
    "cortex_metrics": "requires a running Cortex LLMHoster instance",
    "cortex_state": "requires a running Cortex LLMHoster instance",
    "cortex_hardware": "requires a running Cortex LLMHoster instance",
    "cortex_model_control": "requires a running Cortex LLMHoster instance",
    "cortex_model_files": "requires a running Cortex LLMHoster instance",
    "cortex_update_check": "requires a running Cortex LLMHoster instance",
    "cortex_auth_options": "requires a running Cortex LLMHoster instance",
    "cortex_test_model": "requires a running Cortex LLMHoster instance",
    "cortex_cookbook": "requires a running Cortex LLMHoster instance",
}
"""Tools not exercised offline, mapped to the reason they are skipped."""

CSV_ROWS = "name,score,city\nalpha,10,pune\nbeta,22,pune\ngamma,7,delhi\n"
"""Sample CSV fixture content."""

NESTED_JSON = {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], "meta": {"version": 2}}
"""Sample nested JSON fixture."""

SCRIPT_SOURCE = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n\n\nTOTAL = 10\n'
"""Sample Python fixture source, used both on disk and by ``inspect_python``."""

GUIDE = (
    "# Guide\n\nMAgent is an agent framework for local models.\n\n"
    "## Tools\n\nTools are functions with a schema. The agent calls them and\n"
    "reads the structured result. Every call is sandboxed and audited.\n"
)
"""Sample Markdown fixture content."""


def build_workspace(root: Path) -> None:
    """Populate a throwaway workspace with representative fixture files."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "guide.md").write_text(GUIDE, encoding="utf-8")
    (root / "data.csv").write_text(CSV_ROWS, encoding="utf-8")
    (root / "notes.json").write_text(json.dumps(NESTED_JSON), encoding="utf-8")
    (root / "script.py").write_text(SCRIPT_SOURCE, encoding="utf-8")
    (root / "sample.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x20"
        b"\x08\x06\x00\x00\x00\x00\x00\x00\x00"
    )
    (root / "audio.wav").write_bytes(b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 24)
    build_sample_database(root / "smoke.db")


def build_sample_database(path: Path) -> None:
    """Create a tiny SQLite fixture so the SQL tools have something to read."""
    import sqlite3

    connection = sqlite3.connect(str(path))
    try:
        connection.execute("CREATE TABLE scores (name TEXT PRIMARY KEY, score INTEGER, city TEXT)")
        connection.executemany(
            "INSERT INTO scores (name, score, city) VALUES (?, ?, ?)",
            [("alpha", 10, "pune"), ("beta", 22, "pune"), ("gamma", 7, "delhi")],
        )
        connection.execute("CREATE INDEX scores_city ON scores (city)")
        connection.commit()
    finally:
        connection.close()


def smoke_calls(root: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return one realistic offline call per exercisable tool."""
    csv_text = CSV_ROWS
    json_text = json.dumps(NESTED_JSON)
    records = json.dumps([
        {"city": "pune", "score": 10}, {"city": "pune", "score": 22}, {"city": "delhi", "score": 7},
    ])
    return [
        # -- files -----------------------------------------------------------
        ("read_file", {"path": "docs/guide.md"}),
        ("read_file", {"path": "docs/guide.md", "start_line": 2, "end_line": 3}),
        ("write_file", {"path": "out/report.md", "content": "# Report\n\nhello\n"}),
        ("edit_file", {"path": "out/report.md", "old_text": "hello", "new_text": "world"}),
        ("list_dir", {"path": "."}),
        ("glob_files", {"pattern": "**/*.md"}),
        ("grep", {"pattern": "agent", "path": "."}),
        ("file_info", {"path": "data.csv"}),
        ("copy_path", {"source": "data.csv", "destination": "out/data-copy.csv"}),
        ("move_path", {"source": "out/data-copy.csv", "destination": "out/data-moved.csv"}),
        ("make_dir", {"path": "out/nested/deep"}),
        ("diff_files", {"left": "data.csv", "right": "out/data-moved.csv"}),
        ("directory_tree", {"path": "."}),
        ("hash_file", {"path": "data.csv"}),
        ("chunk_file", {"path": "docs/guide.md", "max_chars": 60}),
        ("find_in_files", {"query": "read_file", "paths": "."}),
        ("disk_usage", {"path": "."}),
        ("delete_path", {"path": "out/data-moved.csv"}),
        # -- shell -----------------------------------------------------------
        ("run_command", {"command": "echo magent-smoke"}),
        ("show_environment", {}),
        ("find_executable", {"name": "python3"}),
        ("list_processes", {}),
        # -- text ------------------------------------------------------------
        ("regex_find", {"text": "abc 123 def 456", "pattern": r"\d+"}),
        ("regex_replace", {"text": "a1 b2", "pattern": r"\d", "replacement": "9"}),
        ("text_stats", {"text": GUIDE}),
        ("transform_text", {"text": "Hello World", "operation": "upper"}),
        ("compare_text", {"left": "a b c", "right": "a x c"}),
        ("extract_lines", {"text": "one\ntwo\nthree", "grep": "tw"}),
        ("encode_text", {"text": "hello", "operation": "base64_encode"}),
        ("split_text", {"text": "a\n\nb\n\nc", "separator": "\n\n"}),
        ("extract_json_from_text", {"text": 'noise {"a": 1} noise'}),
        ("extractive_summary", {"text": GUIDE * 3, "sentences": 2}),
        # -- math and datetime ----------------------------------------------
        ("calculate", {"expression": "sqrt(144) + 2**10"}),
        ("compute_statistics", {"values": "1,2,3,4,100"}),
        ("convert_units", {"value": 3, "from_unit": "km", "to_unit": "miles"}),
        ("convert_base", {"number": "deadbeef", "from_base": 16, "to_base": 2}),
        ("random_number", {"minimum": 1, "maximum": 6}),
        ("vector_math", {"left": "1,2,3", "right": "4,5,6", "operation": "dot"}),
        ("current_time", {}),
        ("format_datetime", {"value": "2026-10-15T09:30:00", "output_format": "%Y/%m/%d"}),
        ("datetime_diff", {"start": "2026-01-01", "end": "2026-10-15"}),
        ("datetime_add", {"value": "2026-01-31T12:00:00", "offset": "1w 2d -3h"}),
        ("calendar_month", {"year": 2026, "month": 10}),
        ("parse_duration_tool", {"duration": "1h30m"}),
        # -- data ------------------------------------------------------------
        ("csv_read", {"path": "data.csv"}),
        ("csv_read", {"path": "data.csv", "where": "city=pune", "sort_by": "score", "descending": True}),
        ("csv_stats", {"path": "data.csv"}),
        ("json_query", {"data": json_text, "path": "items[1].name"}),
        ("json_flatten", {"data": json_text}),
        ("json_to_csv", {"data": json.dumps([{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}])}),
        ("group_aggregate", {"records": records, "group_by": "city", "aggregate": "sum",
                             "value_field": "score"}),
        ("sqlite_query", {"path": ":memory:", "sql": "SELECT 1 AS one, 'x' AS label"}),
        ("sqlite_query", {"path": "smoke.db", "sql": "SELECT name, score FROM scores ORDER BY score DESC"}),
        ("sqlite_schema", {"path": "smoke.db"}),
        # -- search ----------------------------------------------------------
        ("search_documents", {"query": "agent framework tools", "path": "."}),
        ("similar_texts", {"text": "read a file", "candidates": '["read_file","write_file","grep"]'}),
        # -- python ----------------------------------------------------------
        ("python_exec", {"code": "total = sum(i * i for i in range(10))\nprint(total)\nresult = total"}),
        ("audit_python", {"code": "import os\nos.system('id')"}),
        ("inspect_python", {"code": SCRIPT_SOURCE}),
        # -- system ----------------------------------------------------------
        ("system_info", {"include_env": True}),
        ("disk_space", {"path": "."}),
        ("env_get", {"names": "PATH,HOME,MAGENT_DEFINITELY_UNSET"}),
        ("env_list", {}),
        ("python_environment", {}),
        ("uptime_and_time", {}),
        ("wait", {"seconds": 0.01, "reason": "smoke test"}),
        # -- media -----------------------------------------------------------
        ("image_info", {"path": "sample.png"}),
        ("audio_info", {"path": "audio.wav"}),
        ("video_info", {"path": "sample.png"}),
        ("media_hash", {"path": "sample.png"}),
        # -- output ----------------------------------------------------------
        ("format_table", {"data": csv_text, "style": "box"}),
        ("render_chart", {"values": "3,7,2,9", "labels": "a,b,c,d"}),
        ("build_report", {"title": "Smoke", "sections": json.dumps(
            [{"heading": "Intro", "body": "text"}, {"heading": "More", "body": "body"}])}),
        ("write_artifact", {"path": "out/artifact.json", "content": '{"a": 1}', "mode": "json"}),
        ("render_box_tool", {"text": "hello box", "title": "Smoke"}),
        ("render_key_value_tool", {"pairs": '{"alpha": 1, "beta": 2}'}),
        # -- meta ------------------------------------------------------------
        ("list_tools", {"category": "files"}),
        ("describe_tool", {"name": "read_file"}),
        ("use_tool", {"name": "calculate", "arguments": '{"expression": "6*7"}'}),
        ("make_plan", {"goal": "ship the smoke test", "steps": '["write code","run it"]'}),
        ("summarize_for_context", {"text": "x" * 5000, "max_chars": 500}),
        ("parse_url_tool", {"url": "https://user:pass@example.com:8624/v1/models?x=1"}),
        # -- cortex ----------------------------------------------------------
        ("cortex_diagnose", {}),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the smoke suite and return a process exit code."""
    parser = argparse.ArgumentParser(description="Exercise every built-in tool offline.")
    parser.add_argument("-v", "--verbose", action="store_true", help="print result payloads")
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    parser.add_argument("--json", action="store_true", help="emit a machine readable report")
    args = parser.parse_args(argv)

    registry = default_registry()
    failures_on_load = load_errors()
    names = set(registry.names())
    covered = {name for name, _ in smoke_calls(Path("."))}
    unaccounted = sorted(names - covered - set(SKIPPED))
    unknown_skips = sorted(set(SKIPPED) - names)
    unknown_calls = sorted(covered - names)

    root = Path(tempfile.mkdtemp(prefix="magent-smoke-tools-"))
    passed = 0
    failed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        build_workspace(root)
        config = load_config()
        policy = SandboxPolicy(roots=[str(root)], command_policy="open", allow_network=False,
                               deny_action="raise")
        sandbox = Sandbox(policy, workspace=root)
        context = ToolContext(workspace=root, sandbox=sandbox, config=config,
                              registry=registry)
        for index, (name, arguments) in enumerate(smoke_calls(root), start=1):
            result = registry.dispatch(
                ToolCall(id=f"smoke_{index}", name=name, arguments=arguments), context
            )
            rows.append({
                "tool": name,
                "ok": result.ok,
                "duration_ms": round(result.duration_ms, 3),
                "error_code": result.error_code,
                "error": result.error,
            })
            if result.ok:
                passed += 1
                if args.verbose:
                    payload = json.dumps(result.content, default=str, ensure_ascii=False)
                    print(f"  ok   {name:24s} {result.duration_ms:7.1f}ms  {payload[:120]}")
            else:
                failed.append({"tool": name, "code": result.error_code, "error": result.error})
                print(f"  FAIL {name:24s} [{result.error_code}] {str(result.error)[:200]}")
        decisions = sandbox.stats()
    finally:
        if args.keep:
            print(f"workspace kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    elapsed = time.perf_counter() - started

    report = {
        "tools_registered": registry.count(),
        "calls": len(rows),
        "passed": passed,
        "failed": len(failed),
        "skipped": len(SKIPPED),
        "unaccounted_tools": unaccounted,
        "unknown_skips": unknown_skips,
        "unknown_calls": unknown_calls,
        "load_errors": failures_on_load,
        "categories": {key: len(value) for key, value in tools_by_category().items()},
        "sandbox": decisions,
        "seconds": round(elapsed, 2),
        "failures": failed,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print()
        print(f"registered tools : {report['tools_registered']} across {len(report['categories'])} categories")
        print(f"calls dispatched : {report['calls']}  passed {passed}  failed {len(failed)}")
        print(f"skipped          : {report['skipped']} (network, live Cortex, git repo, memory store)")
        print(f"sandbox denials  : {decisions.get('denials')}")
        print(f"elapsed          : {report['seconds']}s")
        for problem in ("unaccounted_tools", "unknown_skips", "unknown_calls", "load_errors"):
            if report[problem]:
                print(f"{problem}: {report[problem]}")

    ok = (
        not failed
        and not unaccounted
        and not unknown_skips
        and not unknown_calls
        and not failures_on_load
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
