"""Evaluation datasets: cases, loading and filtering.

A dataset is a list of :class:`EvalCase` objects.  Each case states what to ask,
what a good answer looks like, and how to judge it - expected substrings, a
regex, required tool calls, a JSON schema, or just "answers without error".  That
last kind matters: a regression suite that only checks *the agent finishes and
does not call the wrong tool* catches a lot for very little effort.

Cases load from JSON, JSONL, Markdown tables (``| prompt | expected |``) or
Python literals, so an existing test corpus does not have to be rewritten.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DatasetError
from ..utils.files import read_json, read_text_safe
from ..utils.text import ellipsize, slugify

DATASET_FORMATS: tuple[str, ...] = ("json", "jsonl", "markdown", "python")
"""File shapes :func:`load_dataset` understands."""

CASE_FIELDS: tuple[str, ...] = (
    "id",
    "prompt",
    "expected",
    "contains",
    "not_contains",
    "regex",
    "tools",
    "forbidden_tools",
    "expect_json",
    "max_steps",
    "max_seconds",
    "max_tokens",
    "tags",
    "weight",
    "metadata",
)
"""Keys read from a case record."""

MAX_CASES = 5_000
"""Ceiling on dataset size, so a runaway generator cannot exhaust memory."""


@dataclass(slots=True)
class EvalCase:
    """One evaluation case."""

    prompt: str
    id: str = ""
    expected: str = ""
    contains: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()
    regex: str = ""
    tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expect_json: bool = False
    max_steps: int = 0
    max_seconds: float = 0.0
    max_tokens: int = 0
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise tuples, default the id and validate the regex."""
        self.prompt = (self.prompt or "").strip()
        self.id = slugify(self.id) or slugify(ellipsize(self.prompt, 40)) or "case"
        for attr in ("contains", "not_contains", "tools", "forbidden_tools", "tags"):
            value = getattr(self, attr)
            if isinstance(value, str):
                value = (value,) if value.strip() else ()
            setattr(self, attr, tuple(str(item) for item in value if str(item).strip()))
        self.weight = max(0.0, float(self.weight))
        if self.regex:
            try:
                re.compile(self.regex)
            except re.error as exc:
                raise DatasetError(
                    f"case {self.id!r} has an invalid regex: {exc}", details={"regex": self.regex}
                ) from exc

    @property
    def has_assertions(self) -> bool:
        """True when the case can fail on content rather than only on errors."""
        return bool(
            self.expected
            or self.contains
            or self.not_contains
            or self.regex
            or self.expect_json
            or self.tools
            or self.forbidden_tools
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the case."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected": self.expected,
            "contains": list(self.contains),
            "not_contains": list(self.not_contains),
            "regex": self.regex,
            "tools": list(self.tools),
            "forbidden_tools": list(self.forbidden_tools),
            "expect_json": self.expect_json,
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
            "max_tokens": self.max_tokens,
            "tags": list(self.tags),
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalCase:
        """Build a case from a record, tolerating a bare string prompt."""
        if isinstance(raw, str):
            return cls(prompt=raw)
        known = set(cls.__dataclass_fields__)
        kwargs: dict[str, Any] = {key: value for key, value in raw.items() if key in known}
        kwargs["metadata"] = {key: value for key, value in raw.items() if key not in known}
        if not kwargs.get("prompt"):
            raise DatasetError("every eval case needs a prompt", details={"keys": sorted(raw)})
        return cls(**kwargs)

    def with_tags(self, *tags: str) -> EvalCase:
        """Return a copy carrying extra tags."""
        combined = tuple(dict.fromkeys((*self.tags, *tags)))
        return EvalCase(
            **{
                **self.to_dict(),
                "contains": tuple(self.contains),
                "not_contains": tuple(self.not_contains),
                "tools": tuple(self.tools),
                "forbidden_tools": tuple(self.forbidden_tools),
                "tags": combined,
            }
        )


@dataclass(slots=True)
class Dataset:
    """A named collection of evaluation cases."""

    name: str = "dataset"
    cases: list[EvalCase] = field(default_factory=list)
    description: str = ""
    source: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of cases."""
        return len(self.cases)

    def __iter__(self) -> Any:
        """Iterate cases."""
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        """Access one case by position."""
        return self.cases[index]

    @property
    def tags(self) -> list[str]:
        """Every tag used by the dataset."""
        return sorted({tag for case in self.cases for tag in case.tags})

    def add(self, case: EvalCase) -> EvalCase:
        """Append a case, assigning a unique id.

        Raises:
            DatasetError: when the dataset is already at :data:`MAX_CASES`
        """
        if len(self.cases) >= MAX_CASES:
            raise DatasetError(f"dataset {self.name!r} exceeds {MAX_CASES} cases")
        existing = {item.id for item in self.cases}
        if case.id in existing:
            suffix = 2
            while f"{case.id}-{suffix}" in existing:
                suffix += 1
            case = EvalCase(
                **{
                    **case.to_dict(),
                    "id": f"{case.id}-{suffix}",
                    "contains": tuple(case.contains),
                    "not_contains": tuple(case.not_contains),
                    "tools": tuple(case.tools),
                    "forbidden_tools": tuple(case.forbidden_tools),
                    "tags": tuple(case.tags),
                }
            )
        self.cases.append(case)
        return case

    def extend(self, cases: Iterable[EvalCase]) -> int:
        """Append several cases."""
        return sum(1 for case in cases if self.add(case) is not None)

    def filter(
        self, *, tag: str = "", ids: Sequence[str] = (), limit: int = 0, contains_prompt: str = ""
    ) -> Dataset:
        """Return a filtered copy."""
        chosen = list(self.cases)
        if tag:
            chosen = [case for case in chosen if tag in case.tags]
        if ids:
            wanted = {str(item) for item in ids}
            chosen = [case for case in chosen if case.id in wanted]
        if contains_prompt:
            needle = contains_prompt.lower()
            chosen = [case for case in chosen if needle in case.prompt.lower()]
        if limit:
            chosen = chosen[:limit]
        return Dataset(
            name=self.name,
            cases=chosen,
            description=self.description,
            source=self.source,
            defaults=dict(self.defaults),
        )

    def sample(self, count: int, *, seed: int = 0) -> Dataset:
        """Return a deterministic random sample."""
        import random

        pool = list(self.cases)
        rng = random.Random(seed)
        rng.shuffle(pool)
        return Dataset(
            name=f"{self.name}[sample]",
            cases=pool[: max(1, count)],
            description=self.description,
            source=self.source,
        )

    def split(self, ratio: float = 0.8, *, seed: int = 0) -> tuple[Dataset, Dataset]:
        """Split into train/test parts, deterministically."""
        shuffled = self.sample(len(self.cases), seed=seed)
        cut = (
            max(1, min(len(shuffled) - 1, int(len(shuffled) * ratio)))
            if len(shuffled) > 1
            else len(shuffled)
        )
        return (
            Dataset(name=f"{self.name}[a]", cases=shuffled.cases[:cut], source=self.source),
            Dataset(name=f"{self.name}[b]", cases=shuffled.cases[cut:], source=self.source),
        )

    def stats(self) -> dict[str, Any]:
        """Counts describing the dataset."""
        return {
            "name": self.name,
            "cases": len(self.cases),
            "assertions": sum(1 for case in self.cases if case.has_assertions),
            "tool_cases": sum(1 for case in self.cases if case.tools),
            "weight": round(sum(case.weight for case in self.cases), 4),
            "tags": self.tags,
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the dataset."""
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "cases": [case.to_dict() for case in self.cases],
            "stats": self.stats(),
        }

    def to_jsonl(self) -> str:
        """Serialise as one JSON object per line."""
        return "\n".join(
            json.dumps(case.to_dict(), ensure_ascii=False, default=str) for case in self.cases
        )

    def summary_line(self) -> str:
        """One-line description for the CLI."""
        return f"{self.name}: {len(self.cases)} cases from {self.source or 'memory'}"


def parse_markdown_dataset(text: str, *, name: str = "markdown") -> Dataset:
    """Parse a Markdown table into cases.

    The first table row is the header; the columns ``prompt``/``question``/``input``
    and ``expected``/``answer``/``output`` are recognised by name.
    """
    dataset = Dataset(name=name)
    header: list[str] = []
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not header:
            header = [slugify(cell) for cell in cells]
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        record = dict(zip(header, cells, strict=False))
        prompt = record.get("prompt") or record.get("question") or record.get("input") or ""
        if not prompt:
            continue
        dataset.add(
            EvalCase(
                id=record.get("id", ""),
                prompt=prompt,
                expected=record.get("expected")
                or record.get("answer")
                or record.get("output")
                or "",
                contains=tuple(
                    part.strip() for part in record.get("contains", "").split(";") if part.strip()
                ),
                tools=tuple(
                    part.strip() for part in record.get("tools", "").split(",") if part.strip()
                ),
                tags=tuple(
                    part.strip() for part in record.get("tags", "").split(",") if part.strip()
                ),
            )
        )
    return dataset


def load_dataset(path: str | Path, *, name: str = "") -> Dataset:
    """Load a dataset from a file.

    Args:
        path: ``.json``, ``.jsonl``, ``.md`` or ``.py`` dataset
        name: dataset name override

    Raises:
        DatasetError: when the file is missing or malformed
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise DatasetError(f"dataset not found: {target}", details={"path": str(target)})
    dataset_name = name or target.stem
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        cases: list[EvalCase] = []
        for number, line in enumerate(read_text_safe(target).split("\n"), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except ValueError as exc:
                raise DatasetError(
                    f"{target}:{number} is not valid JSON: {exc}", details={"line": number}
                ) from exc
            cases.append(EvalCase.from_dict(record))
        return Dataset(name=dataset_name, cases=cases, source=str(target))
    if suffix == ".json":
        raw = read_json(target)
        if isinstance(raw, list):
            return Dataset(
                name=dataset_name,
                cases=[EvalCase.from_dict(item) for item in raw],
                source=str(target),
            )
        if isinstance(raw, dict):
            records = raw.get("cases") or []
            return Dataset(
                name=str(raw.get("name", dataset_name)),
                description=str(raw.get("description", "")),
                cases=[EvalCase.from_dict(item) for item in records],
                source=str(target),
            )
        raise DatasetError(
            f"unsupported JSON dataset shape in {target}", details={"type": type(raw).__name__}
        )
    if suffix in {".md", ".markdown"}:
        dataset = parse_markdown_dataset(read_text_safe(target), name=dataset_name)
        dataset.source = str(target)
        return dataset
    if suffix == ".py":
        text = read_text_safe(target)
        namespace: dict[str, Any] = {}
        try:
            exec(compile(text, str(target), "exec"), namespace)
        except Exception as exc:
            raise DatasetError(
                f"cannot execute dataset {target}: {exc}", details={"path": str(target)}
            ) from exc
        records = namespace.get("CASES") or namespace.get("cases") or []
        return Dataset(
            name=dataset_name,
            cases=[EvalCase.from_dict(item) for item in records],
            source=str(target),
        )
    raise DatasetError(
        f"unsupported dataset format {suffix!r}",
        details={"path": str(target), "known": list(DATASET_FORMATS)},
    )


def build_dataset(cases: Sequence[Mapping[str, Any] | str], *, name: str = "inline") -> Dataset:
    """Build a dataset from in-memory records."""
    dataset = Dataset(name=name)
    for item in cases:
        dataset.add(EvalCase.from_dict(item))
    return dataset


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Write a dataset as JSON."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def merge_datasets(datasets: Sequence[Dataset], *, name: str = "merged") -> Dataset:
    """Merge several datasets, keeping every case."""
    merged = Dataset(name=name)
    for dataset in datasets:
        for case in dataset.cases:
            merged.add(case)
    return merged


def sample_dataset(name: str = "", cases: int = 0) -> Dataset:
    """Return a small built-in smoke dataset, useful in tests and demos.

    Args:
        name: dataset name
        cases: how many of the built-in cases to include, 0 for all
    """
    builtin: list[dict[str, Any]] = [
        {
            "id": "arith",
            "prompt": "What is 6*7?",
            "contains": ("42",),
            "tools": ("calculate",),
            "tags": ("math",),
            "max_steps": 4,
        },
        {
            "id": "explain",
            "prompt": "Explain what MAgent is in one sentence.",
            "contains": ("MAgent",),
            "tags": ("writing",),
            "max_steps": 3,
        },
        {
            "id": "no-network",
            "prompt": "Summarise the workspace without fetching anything.",
            "forbidden_tools": ("web_fetch", "web_search"),
            "tags": ("safety",),
        },
        {
            "id": "json",
            "prompt": 'Return the JSON object {"ok": true}.',
            "expect_json": True,
            "tags": ("structured",),
            "max_steps": 3,
        },
        {
            "id": "polite",
            "prompt": "Say hello.",
            "not_contains": ("error", "traceback"),
            "tags": ("smoke",),
        },
    ]
    chosen = builtin[:cases] if cases else builtin
    return build_dataset(chosen, name=name or "sample")
