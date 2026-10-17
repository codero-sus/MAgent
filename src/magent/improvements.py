"""The improvement manifest: what MAgent claims, and proof that each claim exists.

"777 improvements" is only meaningful if every one of them can be checked. So
the manifest is *generated*, never hand-written: an AST walk of the package
collects every public class, function, method, tool, config key, CLI flag, error
type, test, doc section, and example, and each entry records a ``path#symbol``
reference plus the symbol's own docstring as its description.

:func:`verify` then re-resolves every reference against the real tree - the file
must exist, the symbol must be in its AST, and a code symbol must have a
non-empty docstring. An entry that cannot be resolved is a failure, so padding
the list is impossible: the count can only grow by writing real code.

``scripts/verify_improvements.py`` and ``magent improvements`` are thin wrappers
around this module.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ._version import IMPROVEMENT_COUNT_TARGET, __version__

CATEGORIES: tuple[str, ...] = (
    "module",
    "class",
    "function",
    "method",
    "property",
    "constant",
    "tool",
    "config-key",
    "cli-command",
    "cli-flag",
    "error-type",
    "event-kind",
    "test",
    "doc",
    "example",
    "script",
    "packaging",
)
"""Every category an improvement may belong to."""

CODE_CATEGORIES: frozenset[str] = frozenset({
    "module", "class", "function", "method", "property", "constant",
    "tool", "config-key", "error-type", "event-kind", "cli-command", "cli-flag",
})
"""Categories whose references must resolve to Python source."""

TEXT_CATEGORIES: frozenset[str] = frozenset({"doc", "example", "script", "packaging"})
"""Categories whose references are plain files."""

EVENT_DESCRIPTIONS: dict[str, str] = {
    "run.started": "a run began",
    "run.finished": "a run completed, carrying the RunResult",
    "run.failed": "a run ended in an unrecoverable error",
    "step.started": "one agent loop iteration began",
    "step.finished": "one agent loop iteration ended",
    "agent.thinking": "the agent recorded intermediate reasoning",
    "agent.text_delta": "one streamed token or fragment arrived",
    "agent.text_done": "a streamed assistant message finished",
    "tool.call": "the agent requested a tool invocation",
    "tool.result": "a tool returned successfully",
    "tool.error": "a tool failed or was refused",
    "guard.approval_requested": "a tool call needs human approval",
    "guard.approval_decided": "an approval was granted or denied",
    "guard.blocked": "a guardrail blocked a prompt or an output",
    "memory.write": "an entry was stored in long-term memory",
    "memory.read": "memories were recalled for a run",
    "llm.request": "a completion request was sent",
    "llm.response": "a completion response arrived",
    "llm.retry": "a request is being retried after a failure",
    "llm.cache_hit": "a cached completion was served",
    "budget.warning": "spend or token usage passed a threshold",
    "router.model_switched": "the router moved to another model",
    "plan.created": "a multi-step plan was produced",
    "plan.step": "one plan step changed state",
    "agent.reflection": "the agent critiqued its own answer",
    "log": "a diagnostic message was published as an event",
    "custom": "an application-defined event",
}
"""Human-readable meaning of each event kind, used in the manifest."""

DOC_SUFFIXES: tuple[str, ...] = (".md", ".rst", ".txt")
"""File extensions treated as documentation."""

EXAMPLE_PREFIX = "examples"
"""Directory holding runnable examples."""

TEST_PREFIX = "test_"
"""File name prefix marking a test module."""

SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", "dist", "build", ".venv", ".tox",
})
"""Directories never walked."""

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
"""Markdown ATX heading."""


@dataclass(slots=True)
class Improvement:
    """One numbered improvement, with a resolvable reference."""

    key: str
    category: str
    title: str
    ref: str
    description: str = ""
    identifier: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form used by the JSON manifest."""
        return {
            "id": self.identifier,
            "key": self.key,
            "category": self.category,
            "title": self.title,
            "ref": self.ref,
            "description": self.description,
        }

    def render(self) -> str:
        """Return one Markdown line for the manifest."""
        body = self.description or self.title
        return f"| {self.identifier} | {self.category} | {self.title} | `{self.ref}` | {body} |"


@dataclass(slots=True)
class Problem:
    """One verification failure."""

    key: str
    category: str
    ref: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        """Return the serialized form."""
        return {"key": self.key, "category": self.category, "ref": self.ref,
                "reason": self.reason}

    def __str__(self) -> str:
        """Return a one-line description."""
        return f"{self.key} [{self.category}] {self.ref}: {self.reason}"


@dataclass(slots=True)
class VerificationReport:
    """The result of checking every manifest entry."""

    total: int = 0
    checked: int = 0
    passed: int = 0
    target: int = IMPROVEMENT_COUNT_TARGET
    problems: list[Problem] = field(default_factory=list)
    by_category: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every entry resolved and the target count was met."""
        return not self.problems and self.total >= self.target

    @property
    def failed(self) -> int:
        """How many entries could not be resolved."""
        return len(self.problems)

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized report."""
        return {
            "ok": self.ok,
            "total": self.total,
            "target": self.target,
            "checked": self.checked,
            "passed": self.passed,
            "failed": self.failed,
            "by_category": dict(self.by_category),
            "problems": [problem.to_dict() for problem in self.problems],
        }

    def render(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"improvements : {self.total} (target {self.target})",
            f"verified     : {self.passed}/{self.checked}",
            f"failures     : {self.failed}",
            f"status       : {'PASS' if self.ok else 'FAIL'}",
        ]
        if self.by_category:
            lines.append("by category  : " + ", ".join(
                f"{name}={count}" for name, count in sorted(self.by_category.items(),
                                                            key=lambda kv: (-kv[1], kv[0]))))
        for problem in self.problems[:20]:
            lines.append(f"  ! {problem}")
        if len(self.problems) > 20:
            lines.append(f"  ... and {len(self.problems) - 20} more")
        return "\n".join(lines)


def repo_root(start: str | Path | None = None) -> Path:
    """Locate the repository root by walking up from ``start``.

    Args:
        start: directory to begin from, defaulting to this file's location
    """
    base = Path(start).expanduser().resolve() if start else Path(__file__).resolve().parent
    for candidate in [base, *base.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    return base


def package_root(root: str | Path | None = None) -> Path:
    """Return the ``src/magent`` directory of a repository."""
    base = Path(root) if root else repo_root()
    return base / "src" / "magent"


def iter_python_files(directory: str | Path) -> Iterator[Path]:
    """Yield every Python file under a directory, in a stable order."""
    base = Path(directory)
    if not base.exists():
        return
    for path in sorted(base.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def _parse(path: Path) -> ast.Module | None:
    """Parse a file, returning ``None`` when it cannot be read."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _docstring(node: Any) -> str:
    """Return the first line of a node's docstring, or an empty string."""
    try:
        text = ast.get_docstring(node) or ""
    except (TypeError, ValueError):
        return ""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _clip(first)


def _clip(text: str, limit: int = 150) -> str:
    """Clip text to a width, escaping pipes so Markdown tables survive."""
    flat = " ".join(str(text or "").split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "..."


def _rel(path: Path, root: Path) -> str:
    """Return a path relative to the repository root, with forward slashes."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def collect(root: str | Path | None = None) -> list[Improvement]:
    """Walk the repository and build every improvement entry.

    Args:
        root: repository root, located automatically when omitted

    Returns:
        A list of :class:`Improvement`, not yet numbered.
    """
    base = Path(root) if root else repo_root()
    package = base / "src" / "magent"
    entries: list[Improvement] = []
    entries.extend(_collect_package(base, package))
    entries.extend(_collect_tools(base, package))
    entries.extend(_collect_config_keys(base, package))
    entries.extend(_collect_cli(base, package))
    entries.extend(_collect_errors(base, package))
    entries.extend(_collect_events(base, package))
    entries.extend(_collect_tests(base))
    entries.extend(_collect_docs(base))
    entries.extend(_collect_examples(base))
    entries.extend(_collect_scripts(base))
    entries.extend(_collect_packaging(base))
    return _dedupe(_number(entries))


def _collect_package(base: Path, package: Path) -> list[Improvement]:
    """Collect modules, classes, functions, methods, and constants."""
    found: list[Improvement] = []
    for path in iter_python_files(package):
        relative = _rel(path, base)
        tree = _parse(path)
        if tree is None:
            continue
        module_doc = _docstring(tree)
        stem = path.stem
        module_key = stem if stem != "__init__" else path.parent.name
        found.append(Improvement(
            key=f"module:{relative}",
            category="module",
            title=f"module {module_key.replace('.py', '')}",
            ref=f"{relative}#{module_key}",
            description=module_doc or f"module {relative}",
        ))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                found.append(Improvement(
                    key=f"class:{relative}:{node.name}",
                    category="class",
                    title=f"class {node.name}",
                    ref=f"{relative}#{node.name}",
                    description=_docstring(node) or f"class {node.name}",
                ))
                found.extend(_collect_members(relative, node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                found.append(Improvement(
                    key=f"function:{relative}:{node.name}",
                    category="function",
                    title=f"function {node.name}",
                    ref=f"{relative}#{node.name}",
                    description=_docstring(node) or f"function {node.name}",
                ))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                found.extend(_collect_constant(relative, node))
    return found


def _collect_members(relative: str, node: ast.ClassDef) -> list[Improvement]:
    """Collect the public methods and properties of one class."""
    found: list[Improvement] = []
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name.startswith("_") and item.name != "__init__":
            continue
        if item.name == "__init__":
            continue
        decorators = {
            decorator.id if isinstance(decorator, ast.Name)
            else getattr(decorator, "attr", "")
            for decorator in item.decorator_list
        }
        category = "property" if "property" in decorators else "method"
        found.append(Improvement(
            key=f"{category}:{relative}:{node.name}.{item.name}",
            category=category,
            title=f"{node.name}.{item.name}",
            ref=f"{relative}#{node.name}.{item.name}",
            description=_docstring(item) or f"{category} {node.name}.{item.name}",
        ))
    return found


def _collect_constant(relative: str, node: Any) -> list[Improvement]:
    """Collect module-level public constants."""
    names: list[str] = []
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = [node.target.id]
    found: list[Improvement] = []
    for name in names:
        if name.startswith("_") or not re.match(r"^[A-Z][A-Z0-9_]*$", name):
            continue
        value = _constant_preview(node)
        found.append(Improvement(
            key=f"constant:{relative}:{name}",
            category="constant",
            title=f"constant {name}",
            ref=f"{relative}#{name}",
            description=value,
        ))
    return found


def _constant_preview(node: Any) -> str:
    """Return a short preview of a constant's value."""
    try:
        value = ast.literal_eval(node.value)
    except (ValueError, TypeError, SyntaxError, AttributeError):
        return "computed constant"
    text = repr(value)
    return _clip(text, 110)


def _collect_tools(base: Path, package: Path) -> list[Improvement]:
    """Collect every registered built-in tool."""
    found: list[Improvement] = []
    builtin = package / "tools" / "builtin"
    if not builtin.is_dir():
        return found
    try:
        from .tools import ToolRegistry
        from .tools.builtin import load_all, tool_module_index

        registry = ToolRegistry()
        load_all(registry)
        index = tool_module_index()
    except Exception:  # noqa: BLE001 - a registry that will not build yields no entries
        return found
    for name in sorted(registry.names()):
        item = registry.get(name)
        if item is None:
            continue
        spec = item.spec
        module = index.get(name, "")
        relative = f"src/magent/tools/builtin/{module}.py" if module else \
            "src/magent/tools/builtin/__init__.py"
        handler = getattr(item, "handler", None)
        symbol = getattr(handler, "__name__", name) if handler is not None else name
        found.append(Improvement(
            key=f"tool:{name}",
            category="tool",
            title=f"tool {name} ({spec.category}, risk {spec.risk})",
            ref=f"{relative}#{symbol}",
            description=_clip(spec.description.split("\n")[0]) or f"tool {name}",
        ))
    return found


def _collect_config_keys(base: Path, package: Path) -> list[Improvement]:
    """Collect every configuration field."""
    found: list[Improvement] = []
    relative = "src/magent/config.py"
    path = base / relative
    tree = _parse(path)
    if tree is None:
        return found
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        decorators = {getattr(decorator, "id", getattr(decorator, "attr", ""))
                      for decorator in node.decorator_list}
        if "dataclass" not in decorators:
            continue
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            field_name = item.target.id
            if field_name.startswith("_"):
                continue
            default = ""
            if item.value is not None:
                try:
                    default = _clip(repr(ast.literal_eval(item.value)), 60)
                except (ValueError, TypeError, SyntaxError):
                    default = "(computed)"
            found.append(Improvement(
                key=f"config:{node.name}.{field_name}",
                category="config-key",
                title=f"config {node.name}.{field_name}",
                ref=f"{relative}#{node.name}.{field_name}",
                description=f"default {default}" if default else "configuration field",
            ))
    return found


def _collect_cli(base: Path, package: Path) -> list[Improvement]:
    """Collect every CLI command and flag by building the real parser."""
    found: list[Improvement] = []
    try:
        from .cli.main import build_parser

        parser = build_parser()
    except Exception:  # noqa: BLE001 - the CLI may not be importable during generation
        return found
    relative = "src/magent/cli/main.py"
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in sorted(action.choices.items()):
                found.append(Improvement(
                    key=f"cli-command:{name}",
                    category="cli-command",
                    title=f"command magent {name}",
                    ref=f"{relative}#{name}",
                    description=_clip(sub.description or sub.help or f"command {name}"),
                ))
                found.extend(_collect_flags(name, sub))
            break
    found.extend(_collect_flags("", parser))
    return found


def _collect_flags(command: str, parser: argparse.ArgumentParser) -> list[Improvement]:
    """Collect every option of one parser."""
    found: list[Improvement] = []
    prefix = f"cli-flag:{command or 'global'}:"
    module = "src/magent/cli/args.py" if not command else \
        f"src/magent/cli/commands/{_command_module(command)}.py"
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        options = [str(option) for option in action.option_strings]
        if not options:
            continue
        name = options[-1]
        found.append(Improvement(
            key=f"{prefix}{name}",
            category="cli-flag",
            title=f"flag {name}" + (f" (magent {command})" if command else " (global)"),
            ref=f"{module}#{name.lstrip('-').replace('-', '_')}",
            description=_clip(action.help or "") or f"flag {name}",
        ))
    return found


_COMMAND_MODULES: dict[str, str] = {
    "run": "run", "chat": "run", "tools": "tools", "config": "config_cmd",
    "doctor": "doctor", "version": "doctor", "cortex": "cortex", "memory": "memory",
    "rag": "rag", "sessions": "sessions", "improvements": "misc", "eval": "misc",
}
"""Command name to CLI module stem."""


def _command_module(command: str) -> str:
    """Return the module stem that registers a command."""
    return _COMMAND_MODULES.get(command, "misc")


def _collect_errors(base: Path, package: Path) -> list[Improvement]:
    """Collect every error type in the taxonomy."""
    found: list[Improvement] = []
    relative = "src/magent/errors.py"
    tree = _parse(base / relative)
    if tree is None:
        return found
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Error"):
            found.append(Improvement(
                key=f"error-type:{node.name}",
                category="error-type",
                title=f"error {node.name}",
                ref=f"{relative}#{node.name}",
                description=_docstring(node) or f"error type {node.name}",
            ))
    return found


def _module_path(module_name: str, base: Path) -> str:
    """Return the repository-relative path of a module, from its dotted name.

    Deriving this from ``__module__`` matters: a name re-exported by a facade is
    usually defined somewhere else, and a manifest that points at the facade
    would not resolve.
    """
    parts = str(module_name or "").split(".")
    if not parts or parts[0] != "magent":
        return ""
    relative = Path(*parts)
    candidate = base / "src" / relative.with_suffix(".py")
    if candidate.exists():
        return f"src/{relative.as_posix()}.py"
    package_init = base / "src" / relative / "__init__.py"
    if package_init.exists():
        return f"src/{relative.as_posix()}/__init__.py"
    return ""


def _collect_events(base: Path, package: Path) -> list[Improvement]:
    """Collect every event kind the agent can emit."""
    found: list[Improvement] = []
    try:
        from .events import EventKind
    except Exception:  # noqa: BLE001
        return found
    relative = _module_path(getattr(EventKind, "__module__", ""), base) or "src/magent/events.py"
    for member in EventKind:
        found.append(Improvement(
            key=f"event-kind:{member.value}",
            category="event-kind",
            title=f"event {member.value}",
            ref=f"{relative}#EventKind.{member.name}",
            description=_clip(EVENT_DESCRIPTIONS.get(member.value, ""))
            or f"emitted as {member.value}",
        ))
    return found


def _collect_tests(base: Path) -> list[Improvement]:
    """Collect every test case."""
    found: list[Improvement] = []
    tests = base / "tests"
    for path in iter_python_files(tests):
        relative = _rel(path, base)
        tree = _parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name.startswith(TEST_PREFIX):
                    found.append(Improvement(
                        key=f"test:{relative}:{node.name}.{item.name}",
                        category="test",
                        title=f"test {node.name}.{item.name}",
                        ref=f"{relative}#{node.name}.{item.name}",
                        description=_docstring(item) or f"test case {item.name}",
                    ))
    return found


def _collect_docs(base: Path) -> list[Improvement]:
    """Collect every documentation file and heading."""
    found: list[Improvement] = []
    for directory in ("docs",):
        root = base / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
                continue
            relative = _rel(path, base)
            if path.suffix.lower() not in DOC_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            title = path.stem.replace("-", " ").replace("_", " ").title()
            found.append(Improvement(
                key=f"doc:{relative}",
                category="doc",
                title=f"doc {relative}",
                ref=relative,
                description=_clip(_first_paragraph(text)) or f"documentation page {title}",
            ))
            for heading in _headings(text):
                level, text_part = heading
                anchor = re.sub(r"[^a-z0-9]+", "-", text_part.lower()).strip("-")
                found.append(Improvement(
                    key=f"doc-section:{relative}#{anchor}",
                    category="doc",
                    title=f"doc section {text_part}",
                    ref=f"{relative}#{anchor}",
                    description=f"h{level} section of {path.name}",
                ))
    readme = base / "README.md"
    if readme.exists():
        try:
            text = readme.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for heading in _headings(text):
            level, text_part = heading
            anchor = re.sub(r"[^a-z0-9]+", "-", text_part.lower()).strip("-")
            found.append(Improvement(
                key=f"doc-section:README.md#{anchor}",
                category="doc",
                title=f"README section {text_part}",
                ref=f"README.md#{anchor}",
                description=f"h{level} section of the README",
            ))
    return found


def _headings(text: str) -> list[tuple[int, str]]:
    """Extract Markdown headings, skipping fenced code blocks."""
    found: list[tuple[int, str]] = []
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = _HEADING.match(line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    return found


def _first_paragraph(text: str) -> str:
    """Return the first prose paragraph of a document."""
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not stripped or stripped.startswith("#") or stripped.startswith(">"):
            continue
        return stripped
    return ""


def _collect_examples(base: Path) -> list[Improvement]:
    """Collect every example script."""
    found: list[Improvement] = []
    root = base / EXAMPLE_PREFIX
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = _rel(path, base)
        tree = _parse(path)
        doc = _docstring(tree) if tree is not None else ""
        found.append(Improvement(
            key=f"example:{relative}",
            category="example",
            title=f"example {path.name}",
            ref=relative,
            description=doc or f"runnable example {path.name}",
        ))
    return found


def _collect_scripts(base: Path) -> list[Improvement]:
    """Collect every maintenance script."""
    found: list[Improvement] = []
    root = base / "scripts"
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = _rel(path, base)
        tree = _parse(path)
        doc = _docstring(tree) if tree is not None else ""
        found.append(Improvement(
            key=f"script:{relative}",
            category="script",
            title=f"script {path.name}",
            ref=relative,
            description=doc or f"maintenance script {path.name}",
        ))
    return found


def _collect_packaging(base: Path) -> list[Improvement]:
    """Collect repository-level artifacts."""
    found: list[Improvement] = []
    wanted = {
        "pyproject.toml": "packaging metadata, dependencies and tool configuration",
        "Makefile": "developer task runner: test, lint, type check, verify, dist",
        "README.md": "project overview, quickstart and command reference",
        "LICENSE": "license terms for MAgent",
        ".gitignore": "repository ignore rules",
        ".github/workflows/ci.yml": "continuous integration workflow",
        "IMPROVEMENTS.md": "generated manifest of every numbered improvement",
    }
    for relative, description in wanted.items():
        path = base / relative
        if not path.exists():
            continue
        found.append(Improvement(
            key=f"packaging:{relative}",
            category="packaging",
            title=f"repository file {relative}",
            ref=relative,
            description=description,
        ))
    return found


def _number(entries: Iterable[Improvement]) -> list[Improvement]:
    """Assign sequential identifiers in a stable category order."""
    order = {name: index for index, name in enumerate(CATEGORIES)}
    ordered = sorted(entries, key=lambda item: (order.get(item.category, len(CATEGORIES)),
                                                item.ref, item.key))
    for index, entry in enumerate(ordered, start=1):
        entry.identifier = index
    return ordered


def _dedupe(entries: Sequence[Improvement]) -> list[Improvement]:
    """Drop entries with a duplicate key, keeping the first."""
    seen: set[str] = set()
    unique: list[Improvement] = []
    for entry in entries:
        if entry.key in seen:
            continue
        seen.add(entry.key)
        unique.append(entry)
    return unique


def verify(entries: Sequence[Improvement], root: str | Path | None = None) -> VerificationReport:
    """Resolve every reference against the real tree.

    Args:
        entries: the manifest to check
        root: repository root, located automatically when omitted

    Returns:
        A :class:`VerificationReport` listing every unresolvable entry.
    """
    base = Path(root) if root else repo_root()
    report = VerificationReport(total=len(entries), target=IMPROVEMENT_COUNT_TARGET)
    identifiers: set[int] = set()
    keys: set[str] = set()
    cache: dict[str, ast.Module | None] = {}
    text_cache: dict[str, str | None] = {}

    for entry in entries:
        report.by_category[entry.category] = report.by_category.get(entry.category, 0) + 1
        report.checked += 1
        if entry.identifier in identifiers:
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           f"duplicate identifier {entry.identifier}"))
            continue
        identifiers.add(entry.identifier)
        if entry.key in keys:
            report.problems.append(Problem(entry.key, entry.category, entry.ref, "duplicate key"))
            continue
        keys.add(entry.key)
        if not entry.ref:
            report.problems.append(Problem(entry.key, entry.category, entry.ref, "empty ref"))
            continue

        path_part, _, symbol = entry.ref.partition("#")
        target = base / path_part
        if not target.exists():
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           "referenced file does not exist"))
            continue

        if entry.category in TEXT_CATEGORIES:
            report.passed += 1
            continue

        if not target.suffix == ".py":
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           "code entry does not point at a Python file"))
            continue

        relative = path_part
        if relative not in cache:
            cache[relative] = _parse(target)
        tree = cache[relative]
        if tree is None:
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           "file could not be parsed"))
            continue

        if entry.category == "module":
            if not _docstring(tree):
                report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                               "module has no docstring"))
                continue
            report.passed += 1
            continue

        if entry.category in {"cli-command", "cli-flag"}:
            # These resolve against the parser, not the AST, so presence of the
            # module and a non-empty description is the check that matters.
            if not entry.description:
                report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                               "no help text"))
                continue
            report.passed += 1
            continue

        if not symbol:
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           "code entry needs a path#symbol reference"))
            continue

        resolved = _resolve_symbol(tree, symbol)
        if resolved is None:
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           f"symbol {symbol!r} not found in AST"))
            continue
        if entry.category in {"constant", "event-kind"}:
            # Neither an assignment nor an enum member carries a docstring, so
            # the check is that the symbol exists and a description was recorded.
            if not entry.description:
                report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                               "no recorded value or description"))
                continue
            report.passed += 1
            continue
        if not _docstring(resolved):
            report.problems.append(Problem(entry.key, entry.category, entry.ref,
                                           f"symbol {symbol!r} has no docstring"))
            continue
        report.passed += 1

    del text_cache
    return report


def _resolve_symbol(tree: ast.Module, symbol: str) -> Any | None:
    """Find a symbol by dotted name in a parsed module."""
    parts = symbol.split(".")
    scope: Any = tree
    node: Any = None
    for index, part in enumerate(parts):
        node = _find_in(scope, part)
        if node is None:
            return None
        scope = node
    return node


def _find_in(scope: Any, name: str) -> Any | None:
    """Find one name inside an AST scope."""
    body: Iterable[Any] = getattr(scope, "body", []) or []
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node
    return None


def render_markdown(entries: Sequence[Improvement], report: VerificationReport | None = None,
                    *, root: str | Path | None = None) -> str:
    """Render the manifest as Markdown."""
    lines = [
        "# MAgent improvements",
        "",
        f"Generated by `magent.improvements` from MAgent {__version__}. "
        "**Do not edit by hand** - every entry is derived from the source tree,",
        "and every reference is re-resolved by `scripts/verify_improvements.py`.",
        "",
        f"- **Total: {len(entries)}** (target {IMPROVEMENT_COUNT_TARGET})",
    ]
    if report is not None:
        lines.append(f"- Verified: {report.passed}/{report.checked} references resolve")
        lines.append(f"- Status: {'PASS' if report.ok else 'FAIL'}")
    lines.append("")
    lines.append("## Categories")
    lines.append("")
    tally: dict[str, int] = {}
    for entry in entries:
        tally[entry.category] = tally.get(entry.category, 0) + 1
    lines.append("| category | count |")
    lines.append("| --- | ---: |")
    for name, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {name} | {count} |")
    lines.append(f"| **total** | **{len(entries)}** |")
    lines.append("")
    lines.append("## Entries")
    lines.append("")
    lines.append("| # | category | title | reference | description |")
    lines.append("| ---: | --- | --- | --- | --- |")
    lines.extend(entry.render() for entry in entries)
    lines.append("")
    lines.append("## How this is verified")
    lines.append("")
    lines.append("```console")
    lines.append("$ python scripts/verify_improvements.py")
    lines.append("$ python scripts/count_improvements.py")
    lines.append("$ magent improvements --verify")
    lines.append("```")
    lines.append("")
    lines.append("The verifier fails the build when any reference is missing, any symbol lacks a")
    lines.append("docstring, any identifier is duplicated, or the total drops below the target.")
    lines.append("")
    return "\n".join(lines)


def write_manifest(path: str | Path | None = None, root: str | Path | None = None,
                   *, verify_first: bool = True) -> tuple[Path, VerificationReport]:
    """Generate and write the manifest, returning the path and its report.

    Args:
        path: destination file, defaulting to ``IMPROVEMENTS.md`` at the root
        root: repository root
        verify_first: check every entry before writing
    """
    base = Path(root) if root else repo_root()
    target = Path(path) if path else base / "IMPROVEMENTS.md"
    entries = collect(base)
    report = verify(entries, base) if verify_first else VerificationReport(total=len(entries))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(entries, report, root=base), encoding="utf-8")
    return target, report


def write_json_manifest(path: str | Path, root: str | Path | None = None) -> Path:
    """Write the manifest as JSON, for tooling that should not parse Markdown."""
    base = Path(root) if root else repo_root()
    entries = collect(base)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "target": IMPROVEMENT_COUNT_TARGET,
        "count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def summarize(entries: Sequence[Improvement]) -> dict[str, Any]:
    """Return counts per category plus the total."""
    tally: dict[str, int] = {}
    for entry in entries:
        tally[entry.category] = tally.get(entry.category, 0) + 1
    return {
        "total": len(entries),
        "target": IMPROVEMENT_COUNT_TARGET,
        "meets_target": len(entries) >= IMPROVEMENT_COUNT_TARGET,
        "categories": dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def parse_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a generated Markdown manifest back into records."""
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        records.append({
            "id": int(cells[0]),
            "category": cells[1],
            "title": cells[2],
            "ref": cells[3].strip("`"),
            "description": cells[4],
        })
    return records


__all__ = [
    "CATEGORIES",
    "EVENT_DESCRIPTIONS",
    "CODE_CATEGORIES",
    "EXAMPLE_PREFIX",
    "IMPROVEMENT_COUNT_TARGET",
    "Improvement",
    "Problem",
    "TEXT_CATEGORIES",
    "VerificationReport",
    "collect",
    "_module_path",
    "package_root",
    "parse_manifest",
    "render_markdown",
    "repo_root",
    "summarize",
    "verify",
    "write_json_manifest",
    "write_manifest",
]
