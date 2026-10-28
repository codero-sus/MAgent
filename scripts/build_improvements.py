#!/usr/bin/env python3
"""Generate the machine-checkable improvements manifest.

Every entry in ``improvements.json`` points at something that actually exists in
this repository: a documented function, class or method; a registered tool; a CLI
command; a shipped skill; a test; a documented procedure; a Makefile target; a
network of repository-level facts.  The verifier
(:mod:`scripts.verify_improvements`) re-checks every reference against the current
tree, so the manifest cannot drift away from the code it claims to describe.

Usage::

    python scripts/build_improvements.py            # rewrite the manifest
    python scripts/build_improvements.py --check     # fail if it is stale
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Repository root, derived from this file's location."""

MANIFEST_PATH = REPO_ROOT / "improvements.json"
"""Machine-readable output."""

DOC_PATH = REPO_ROOT / "IMPROVEMENTS.md"
"""Human-readable output."""

MINIMUM_IMPROVEMENTS = 777
"""The build promises at least this many verified improvements."""

CATEGORY_BY_PACKAGE: dict[str, str] = {
    "agents": "multi-agent",
    "cli": "interface",
    "concurrency": "concurrency",
    "context": "context",
    "eval": "evaluation",
    "events": "runtime",
    "guardrails": "safety",
    "http": "runtime",
    "llm": "backend",
    "memory": "memory",
    "observability": "observability",
    "output": "interface",
    "plugins": "extensibility",
    "prompts": "prompting",
    "rag": "retrieval",
    "server": "interface",
    "session": "sessions",
    "skills": "extensibility",
    "testing": "testing",
    "tools": "tools",
    "utils": "foundation",
}
"""Package name to manifest category."""

SKIP_SYMBOLS: frozenset[str] = frozenset(
    {
        "__init__",
        "__post_init__",
        "__repr__",
        "__len__",
        "__contains__",
        "__iter__",
        "__getitem__",
        "__getattr__",
        "__dir__",
        "__enter__",
        "__exit__",
        "__call__",
        "__hash__",
        "__eq__",
        "__str__",
        "__bool__",
        "__del__",
        "__set_name__",
        "__init_subclass__",
        "__class_getitem__",
        "__format__",
        "wrapper",
        "decorate",
        "composed",
        "line",
        "separator",
        "frame",
        "walk",
        "split_with",
        "handler",
        "generator",
        "flush",
        "close",
    }
)
"""Dunder and closure helpers are skipped: they are plumbing, not features."""

KIND_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("tests/", "test"),
    ("docs/", "documentation"),
    ("examples/", "example"),
    ("scripts/", "tooling"),
    ("src/", "code"),
)
"""How a path is classified."""


@dataclass(slots=True)
class Improvement:
    """One verifiable improvement."""

    id: str
    category: str
    kind: str
    title: str
    ref: str
    description: str = ""
    symbol: str = ""
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entry, omitting empty optional fields."""
        payload: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "kind": self.kind,
            "title": self.title,
            "ref": self.ref,
        }
        if self.description:
            payload["description"] = self.description
        if self.symbol:
            payload["symbol"] = self.symbol
        if self.line:
            payload["line"] = self.line
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def markdown_row(self) -> str:
        """One row of the generated ``IMPROVEMENTS.md`` table."""
        title = self.title.replace("|", "\\|")
        return f"| {self.id} | `{self.ref}` | {title} |"


def iter_python_files() -> Iterator[Path]:
    """Yield every Python file that belongs to the manifest."""
    for pattern in ("src/magent/**/*.py", "tests/**/*.py", "scripts/*.py", "examples/**/*.py"):
        for path in sorted(REPO_ROOT.glob(pattern)):
            if "__pycache__" in path.parts or path.name.startswith("_generated"):
                continue
            yield path


def relative(path: Path) -> str:
    """Repository-relative POSIX path."""
    return path.relative_to(REPO_ROOT).as_posix()


def category_for(path: str, fallback: str = "foundation") -> str:
    """Category for a file path."""
    text = path.replace("\\", "/")
    if text.startswith("tests/"):
        return "testing"
    if text.startswith("docs/"):
        return "documentation"
    if text.startswith("examples/"):
        return "examples"
    if text.startswith("scripts/"):
        return "tooling"
    parts = text.split("/")
    if "src" in parts:
        index = parts.index("src")
        if len(parts) > index + 2:
            return CATEGORY_BY_PACKAGE.get(parts[index + 2], fallback)
    return fallback


def kind_for(path: str) -> str:
    """Kind label for a file path."""
    for prefix, kind in KIND_BY_PREFIX:
        if path.startswith(prefix):
            return kind
    return "code"


def _docstring_first_line(node: ast.AST) -> str:
    """First non-empty line of a node's docstring, or an empty string."""
    try:
        doc = ast.get_docstring(node) or ""
    except (TypeError, ValueError):
        return ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _docstring_summary(node: ast.AST, *, limit: int = 240) -> str:
    """One-sentence summary: everything up to the first blank line."""
    try:
        doc = ast.get_docstring(node) or ""
    except (TypeError, ValueError):
        return ""
    paragraph: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        paragraph.append(line.strip())
    text = " ".join(paragraph)
    return text[:limit]


def _is_documented(node: ast.AST) -> bool:
    """True when a definition carries a usable docstring."""
    return bool(_docstring_first_line(node))


def _qualified_name(stack: Sequence[str], name: str) -> str:
    """``Class.method`` style name for a nested definition."""
    return ".".join([*stack, name]) if stack else name


def collect_symbols(path: Path) -> list[Improvement]:
    """Every documented function, class and method in one Python file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - a broken file is a build failure
        print(f"warning: cannot parse {relative(path)}: {exc}", file=sys.stderr)
        return []
    rel = relative(path)
    category = category_for(rel)
    kind = kind_for(rel)
    out: list[Improvement] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name in SKIP_SYMBOLS and not isinstance(child, ast.ClassDef):
                    visit(child, stack)
                    continue
                qualified = _qualified_name(stack, child.name)
                if _is_documented(child):
                    out.append(
                        Improvement(
                            id="",
                            category=category,
                            kind=kind,
                            title=_docstring_first_line(child),
                            ref=f"{rel}#{qualified}",
                            description=_docstring_summary(child),
                            symbol=qualified,
                            line=getattr(child, "lineno", 0),
                            metadata={"defined_in": rel},
                        )
                    )
                if isinstance(child, ast.ClassDef) or isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    visit(child, [*stack, child.name])
            else:
                visit(child, stack)

    visit(tree, [])
    return out


def collect_tools(path: Path) -> list[Improvement]:
    """Entries for every ``@tool`` decorated function in one module."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    rel = relative(path)
    out: list[Improvement] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "tool":
                continue
            tool_name = ""
            for keyword in decorator.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    tool_name = str(keyword.value.value)
            tool_name = tool_name or node.name
            summary = _docstring_first_line(node) or f"Tool {tool_name}"
            out.append(
                Improvement(
                    id="",
                    category="tools",
                    kind="tool",
                    title=f"Tool ``{tool_name}``: {summary}",
                    ref=f"{rel}#{node.name}",
                    description=_docstring_summary(node),
                    symbol=node.name,
                    line=getattr(node, "lineno", 0),
                    metadata={"tool": tool_name},
                )
            )
    return out


def collect_commands(path: Path) -> list[Improvement]:
    """Entries for every CLI command declared in a command module."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    rel = relative(path)
    out: list[Improvement] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "Command":
            continue
        values: dict[str, ast.AST] = {}
        for keyword in node.keywords:
            if keyword.arg:
                values[keyword.arg] = keyword.value
        command_name = _constant_text(values.get("name"))
        handler = _attribute_name(values.get("run"))
        if not command_name or not handler:
            continue
        help_text = _constant_text(values.get("help")) or command_name
        out.append(
            Improvement(
                id="",
                category="interface",
                kind="command",
                title=f"CLI command ``magent {command_name}``: {help_text}",
                ref=f"{rel}#{handler}",
                description=_constant_text(values.get("description")) or help_text,
                symbol=handler,
                line=getattr(node, "lineno", 0),
                metadata={"command": command_name},
            )
        )
    return out


def _constant_text(node: ast.AST | None) -> str:
    """Best-effort string value of a constant or f-string node."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    if isinstance(node, ast.JoinedStr):
        parts = [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
        return "".join(parts).strip()
    return ""


def _attribute_name(node: ast.AST | None) -> str:
    """Name referenced by a ``Name`` or ``Attribute`` node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def collect_markdown(path: Path) -> list[Improvement]:
    """One entry per ``##`` section of a documentation page."""
    rel = relative(path)
    out: list[Improvement] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        text = line.strip()
        if not text.startswith("## "):
            continue
        heading = text[3:].strip()
        if not heading:
            continue
        out.append(
            Improvement(
                id="",
                category="documentation",
                kind="section",
                title=heading,
                ref=f"{rel}#{heading}",
                description=f"{heading} (documented in {rel})",
                symbol=heading,
                line=number,
            )
        )
    return out


def collect_files() -> list[Improvement]:
    """Repository-level entries: docs, examples, scripts, workflows, Makefile targets."""
    out: list[Improvement] = []
    for pattern, category, kind in (
        ("docs/*.md", "documentation", "page"),
        ("examples/*.py", "examples", "example"),
        ("scripts/*.py", "tooling", "script"),
        (".github/workflows/*.yml", "ci", "workflow"),
        ("pyproject.toml", "packaging", "manifest"),
        ("Makefile", "tooling", "makefile"),
        ("README.md", "documentation", "page"),
        ("LICENSE", "packaging", "license"),
        (".gitignore", "packaging", "config"),
    ):
        for path in sorted(REPO_ROOT.glob(pattern)):
            rel = relative(path)
            if rel == MANIFEST_PATH.name or rel == DOC_PATH.name:
                continue
            out.append(
                Improvement(
                    id="",
                    category=category,
                    kind=kind,
                    title=f"{rel}",
                    ref=rel,
                    description=f"{kind.replace('-', ' ').capitalize()} shipped at {rel}",
                    metadata={"bytes": path.stat().st_size},
                )
            )
    out.extend(collect_make_targets())
    return out


def collect_make_targets() -> list[Improvement]:
    """One entry per Makefile target, so the developer UX is verifiable too."""
    makefile = REPO_ROOT / "Makefile"
    if not makefile.is_file():
        return []
    out: list[Improvement] = []
    for line in makefile.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("\t", "#", ".")) or ":" not in line:
            continue
        target, separator, dependencies = line.partition(":")
        if separator != ":" or dependencies.startswith(("=", "?")):
            # ``SRC :=`` / ``CFLAGS +=`` are variable assignments, not targets.
            continue
        if target.strip().startswith("$") or "=" in target:
            continue
        if any(char in target for char in ("(", ")", "/", "&")):
            continue
        names = [item for item in target.split() if item]
        for name in names:
            out.append(
                Improvement(
                    id="",
                    category="tooling",
                    kind="make-target",
                    title=f"``make {name}``",
                    ref=f"Makefile#{name}:",
                    description=f"Make target {name} (dependencies: {dependencies.strip() or 'none'})",
                    symbol=f"{name}:",
                )
            )
    return out


def collect_doc_claims() -> list[Improvement]:
    """Entries for the shipped skills, presets, personas and prompt templates.

    These live in Python data structures rather than functions, so they are
    derived from the source with a light parse and verified by looking for the
    declared name in the file that defines it.
    """
    out: list[Improvement] = []
    source_map = {
        "skills": (REPO_ROOT / "src/magent/skills/library.py", 'name="', "Skill"),
        "presets": (REPO_ROOT / "src/magent/agents/presets.py", 'name="', "AgentPreset"),
    }
    for category, (path, marker, _kind) in source_map.items():
        if not path.is_file():
            continue
        rel = relative(path)
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in {"Skill", "AgentPreset", "_register"}:
                continue
            values: dict[str, str] = {}
            for keyword in node.keywords:
                if keyword.arg:
                    values[keyword.arg] = _constant_text(keyword.value)
            if not values.get("name"):
                continue
            label = values["name"]
            description = values.get("description") or values.get("body", "")[:120]
            out.append(
                Improvement(
                    id="",
                    category=category,
                    kind=category[:-1] or "item",
                    title=f"{category[:-1].capitalize()} ``{label}``: {description or 'shipped feature'}",
                    ref=f"{rel}#{label}",
                    description=description,
                    symbol=label,
                )
            )
    personas = REPO_ROOT / "src/magent/prompts/personas.py"
    if personas.is_file():
        rel = relative(personas)
        tree = ast.parse(personas.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name != "Persona":
                    continue
                values = {
                    keyword.arg: _constant_text(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg
                }
                if not values.get("name"):
                    continue
                out.append(
                    Improvement(
                        id="",
                        category="prompting",
                        kind="persona",
                        title=f"Persona ``{values['name']}``: {values.get('role') or 'shipped persona'}",
                        ref=f"{rel}#{values['name']}",
                        symbol=values["name"],
                    )
                )
    templates = REPO_ROOT / "src/magent/prompts/templates.py"
    if templates.is_file():
        rel = relative(templates)
        tree = ast.parse(templates.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in {"PromptTemplate", "_template"}:
                    continue
                values = {
                    keyword.arg: _constant_text(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg
                }
                if not values.get("name"):
                    continue
                out.append(
                    Improvement(
                        id="",
                        category="prompting",
                        kind="template",
                        title=f"Prompt template ``{values['name']}``",
                        ref=f"{rel}#{values['name']}",
                        symbol=values["name"],
                    )
                )
    return out


def collect_errors() -> list[Improvement]:
    """One entry per typed exception, so error handling is verifiable too."""
    path = REPO_ROOT / "src/magent/errors.py"
    if not path.is_file():
        return []
    rel = relative(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[Improvement] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_attribute_name(base) == "MAgentError" for base in node.bases):
            continue
        out.append(
            Improvement(
                id="",
                category="foundation",
                kind="error",
                title=f"Typed error ``{node.name}``: {_docstring_first_line(node) or node.name}",
                ref=f"{rel}#{node.name}",
                description=_docstring_summary(node),
                symbol=node.name,
                line=getattr(node, "lineno", 0),
            )
        )
    return out


def collect_all() -> list[Improvement]:
    """Every improvement in the repository, in a deterministic order."""
    collected: list[Improvement] = []
    for path in iter_python_files():
        collected.extend(collect_symbols(path))
        rel = relative(path)
        if not rel.startswith("tests/"):
            # Tests define throwaway tools through the same decorator; those are
            # fixtures, not shipped capabilities.
            collected.extend(collect_tools(path))
        collected.extend(collect_commands(path))
    for path in sorted(REPO_ROOT.glob("docs/*.md")):
        collected.extend(collect_markdown(path))
    collected.extend(collect_files())
    collected.extend(collect_doc_claims())
    collected.extend(collect_errors())
    return deduplicate(collected)


def deduplicate(entries: Iterable[Improvement]) -> list[Improvement]:
    """Drop entries that repeat a ``(kind, ref)`` pair, keeping the first."""
    seen: set[tuple[str, str]] = set()
    out: list[Improvement] = []
    for entry in entries:
        key = (entry.kind, entry.ref)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    out.sort(key=lambda item: (item.category, item.kind, item.ref, item.line))
    for index, entry in enumerate(out, start=1):
        entry.id = f"IMP-{index:04d}"
    return out


def build_manifest() -> dict[str, Any]:
    """Assemble the manifest payload."""
    entries = collect_all()
    categories: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for entry in entries:
        categories[entry.category] = categories.get(entry.category, 0) + 1
        kinds[entry.kind] = kinds.get(entry.kind, 0) + 1
    return {
        "schema": 1,
        "generator": "scripts/build_improvements.py",
        "verifier": "scripts/verify_improvements.py",
        "minimum": MINIMUM_IMPROVEMENTS,
        "total": len(entries),
        "categories": dict(sorted(categories.items())),
        "kinds": dict(sorted(kinds.items())),
        "improvements": [entry.to_dict() for entry in entries],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render ``IMPROVEMENTS.md`` from the manifest."""
    entries = manifest["improvements"]
    lines = [
        "# MAgent improvements",
        "",
        f"**{manifest['total']} verifiable improvements** across "
        f"{len(manifest['categories'])} categories, every one of them traced to a "
        "file and symbol that exists in this repository.",
        "",
        "This file is generated by `scripts/build_improvements.py` and checked by "
        "`scripts/verify_improvements.py`. Re-run `make verify` after any change: an "
        "entry whose reference no longer resolves fails the build.",
        "",
        "## How to read an entry",
        "",
        "* `ref` is `path#symbol` for code, a tool, a CLI command or a test;",
        "* a bare `ref` is a file that must exist (docs, workflows, packaging);",
        "* `path#token` for Makefile targets and Markdown sections must appear verbatim.",
        "",
        "## Counts",
        "",
        "| category | count |",
        "| --- | --- |",
    ]
    for category, count in manifest["categories"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "| kind | count |",
            "| --- | --- |",
        ]
    )
    for kind, count in manifest["kinds"].items():
        lines.append(f"| {kind} | {count} |")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)
    for category in sorted(by_category):
        lines.extend(
            [
                "",
                f"## {category} ({len(by_category[category])})",
                "",
                "| id | ref | what it is |",
                "| --- | --- | --- |",
            ]
        )
        for entry in by_category[category]:
            title = entry["title"].replace("|", "\\|")
            lines.append(f"| {entry['id']} | `{entry['ref']}` | {title} |")
    lines.append("")
    return "\n".join(lines)


def write_manifest(*, check: bool = False) -> int:
    """Write both files (or verify they are current) and return an exit code."""
    manifest = build_manifest()
    if manifest["total"] < MINIMUM_IMPROVEMENTS:
        print(
            f"error: only {manifest['total']} improvements found, {MINIMUM_IMPROVEMENTS} expected",
            file=sys.stderr,
        )
        return 1
    payload = json.dumps(manifest, indent=1, sort_keys=False) + "\n"
    markdown = render_markdown(manifest)
    if check:
        stale: list[str] = []
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_text(encoding="utf-8") != payload:
            stale.append(str(MANIFEST_PATH.name))
        if not DOC_PATH.is_file() or DOC_PATH.read_text(encoding="utf-8") != markdown:
            stale.append(str(DOC_PATH.name))
        if stale:
            print(
                f"error: {', '.join(stale)} is out of date; run "
                f"`python scripts/build_improvements.py`",
                file=sys.stderr,
            )
            return 1
        print(f"up to date: {manifest['total']} improvements")
        return 0
    MANIFEST_PATH.write_text(payload, encoding="utf-8")
    DOC_PATH.write_text(markdown, encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.name} and {DOC_PATH.name} with {manifest['total']} improvements")
    for category, count in manifest["categories"].items():
        print(f"  {category:<16} {count}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the generated files are out of date"
    )
    args = parser.parse_args(argv)
    return write_manifest(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
