"""Skills: reusable instructions a model can be handed for one kind of task.

A skill is prose plus metadata - the same shape Anthropic-style ``SKILL.md``
files use, minus any dependency.  Front matter carries the machine-readable
part (name, description, tags, triggers, preferred tools); the body is the
instruction text injected into the prompt when the skill applies.

The parser is deliberately forgiving: lists may be written ``[a, b]``,
comma-separated, or as ``- item`` lines, and a skill without front matter is
still a skill (its name comes from the file stem).  Anything malformed raises
:class:`~magent.errors.SkillFormatError` with the line number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import SkillFormatError
from ..utils.files import read_text_safe
from ..utils.text import ellipsize, slugify

FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n\s*---\s*\n?(.*)$", re.DOTALL)
"""Matcher for a leading ``---`` front-matter block."""

KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$")
"""Matcher for one front-matter key."""

LIST_FIELDS: tuple[str, ...] = ("tags", "triggers", "tools", "examples", "related")
"""Fields parsed as lists of strings."""

MAX_BODY_CHARS = 20_000
"""Ceiling on instruction text kept from one skill."""

SKILL_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".skill", ".txt")
"""File extensions recognised as skills."""


@dataclass(slots=True)
class Skill:
    """One reusable instruction set."""

    name: str
    description: str = ""
    body: str = ""
    tags: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    version: str = "1.0.0"
    author: str = ""
    source: str = ""
    priority: int = 50
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise the name, the tuples and the body length."""
        self.name = slugify(self.name or "skill")
        self.body = (self.body or "")[:MAX_BODY_CHARS].strip()
        for attr in ("tags", "triggers", "tools", "examples", "related"):
            value = getattr(self, attr)
            if isinstance(value, str):
                value = tuple(part.strip() for part in re.split(r"[,\n]", value) if part.strip())
            setattr(
                self,
                attr,
                tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip())),
            )
        self.priority = max(0, min(100, int(self.priority)))

    @property
    def words(self) -> int:
        """Word count of the instruction body."""
        return len(self.body.split())

    @property
    def is_empty(self) -> bool:
        """True when the skill carries no instructions."""
        return not self.body.strip() and not self.description.strip()

    def matches(self, text: str) -> float:
        """Score how well this skill applies to a task description, 0..1.

        Triggers are worth more than tags; a tag match alone is weak evidence.
        """
        if not text.strip():
            return 0.0
        haystack = f" {text.lower()} "
        score = 0.0
        for trigger in self.triggers:
            needle = trigger.lower().strip()
            if needle and (needle in haystack or f" {needle} " in haystack):
                score += 0.34
        for tag in self.tags:
            needle = tag.lower().strip()
            if needle and needle in haystack:
                score += 0.12
        name_words = [word for word in self.name.replace("-", " ").split() if len(word) > 3]
        for word in name_words:
            if word.lower() in haystack:
                score += 0.18
        return min(1.0, round(score, 4))

    def render(self, *, style: str = "heading", max_chars: int = 4_000) -> str:
        """Render the skill as prompt text.

        Args:
            style: ``heading``, ``plain`` or ``tagged``
            max_chars: ceiling on the rendered body
        """
        body = ellipsize(self.body.strip(), max_chars)
        if style == "plain":
            return body
        if style == "tagged":
            return f'<skill name="{self.name}">\n{body}\n</skill>'
        header = f"## Skill: {self.name}"
        if self.description:
            header += f" - {self.description}"
        return f"{header}\n\n{body}"

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        """Serialise the skill."""
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "triggers": list(self.triggers),
            "tools": list(self.tools),
            "related": list(self.related),
            "version": self.version,
            "author": self.author,
            "source": self.source,
            "priority": self.priority,
            "enabled": self.enabled,
            "words": self.words,
            "metadata": dict(self.metadata),
        }
        if include_body:
            payload["body"] = self.body
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Skill:
        """Build a skill from parsed data."""
        known = set(cls.__dataclass_fields__)
        kwargs: dict[str, Any] = {key: value for key, value in raw.items() if key in known}
        kwargs["metadata"] = {key: value for key, value in raw.items() if key not in known}
        kwargs.setdefault("name", str(raw.get("name", "") or "skill"))
        return cls(**kwargs)

    def summary_line(self) -> str:
        """One-line description for ``magent skills``."""
        tags = ",".join(self.tags[:3])
        return f"{self.name:<24} {ellipsize(self.description, 60)} {tags}"

    def __len__(self) -> int:
        """Number of words in the body."""
        return self.words


def parse_front_matter(text: str, *, source: str = "") -> tuple[dict[str, Any], str]:
    """Split a Markdown document into front matter and body.

    Args:
        text: the document
        source: path used in error messages

    Raises:
        SkillFormatError: when a front-matter line is neither a key nor a list item
    """
    match = FRONT_MATTER_RE.match(text or "")
    if not match:
        return {}, (text or "")
    raw_block, body = match.group(1), match.group(2)
    data: dict[str, Any] = {}
    current_list: str = ""
    for number, line in enumerate(raw_block.split("\n"), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list:
            data.setdefault(current_list, []).append(stripped[2:].strip())
            continue
        key_match = KEY_RE.match(stripped)
        if not key_match:
            raise SkillFormatError(
                f"bad front matter on line {number} of {source or 'skill'}: {line!r}",
                details={"line": number, "source": source, "text": line},
            )
        key = key_match.group(1).strip().lower().replace("-", "_")
        value = key_match.group(2).strip()
        current_list = ""
        if not value:
            data[key] = []
            current_list = key
            continue
        if value.startswith("[") and value.endswith("]"):
            items = [part.strip().strip("'\"") for part in value[1:-1].split(",")]
            data[key] = [item for item in items if item]
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        elif value.isdigit():
            data[key] = int(value)
        else:
            data[key] = value.strip("'\"")
    for key in LIST_FIELDS:
        if key in data and isinstance(data[key], str):
            data[key] = [part.strip() for part in data[key].split(",") if part.strip()]
    return data, body


def parse_skill(text: str, *, source: str = "", name: str = "") -> Skill:
    """Parse one skill document.

    Args:
        text: Markdown with optional front matter
        source: path the text came from
        name: fallback name when neither front matter nor the path provides one

    Raises:
        SkillFormatError: when the front matter is malformed
    """
    data, body = parse_front_matter(text, source=source)
    if not data and body.lstrip().startswith("# "):
        first, _, rest = body.partition("\n")
        data = {"name": first[2:].strip()}
        body = rest
    if not data.get("name"):
        stem = Path(source).stem if source else (name or "skill")
        data["name"] = name or stem
    if not data.get("description") and body.strip():
        for line in body.strip().split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                data["description"] = ellipsize(stripped, 160)
                break
    data["body"] = body
    data.setdefault("source", source)
    return Skill.from_dict(data)


def load_skill(path: str | Path) -> Skill:
    """Load one skill file.

    Raises:
        SkillFormatError: when the file is unreadable or malformed
    """
    target = Path(path).expanduser()
    try:
        text = read_text_safe(target, limit=MAX_BODY_CHARS * 2)
    except Exception as exc:
        raise SkillFormatError(
            f"cannot read skill {target}: {exc}", details={"path": str(target)}
        ) from exc
    return parse_skill(text, source=str(target))


def dump_skill(skill: Skill) -> str:
    """Render a skill back to Markdown with front matter."""
    lines = ["---", f"name: {skill.name}"]
    if skill.description:
        lines.append(f"description: {skill.description}")
    for key in ("tags", "triggers", "tools", "related"):
        value = getattr(skill, key)
        if value:
            lines.append(f"{key}: [{', '.join(value)}]")
    lines.append(f"version: {skill.version}")
    if skill.author:
        lines.append(f"author: {skill.author}")
    if skill.priority != 50:
        lines.append(f"priority: {skill.priority}")
    lines.append("---")
    lines.append("")
    lines.append(skill.body.strip())
    return "\n".join(lines) + "\n"


def skill_from_mapping(name: str, values: Mapping[str, Any]) -> Skill:
    """Build a skill from a config ``[skills.<name>]`` section."""
    data = dict(values)
    data.setdefault("name", name)
    if "instructions" in data and "body" not in data:
        data["body"] = data.pop("instructions")
    return Skill.from_dict(data)


def merge_skills(skills: Sequence[Skill], *, name: str = "combined") -> Skill:
    """Combine several skills into one, keeping the highest priority first."""
    ordered = sorted(skills, key=lambda item: -item.priority)
    body = "\n\n".join(item.render(style="heading") for item in ordered if not item.is_empty)
    tags = tuple(dict.fromkeys(tag for item in ordered for tag in item.tags))
    tools = tuple(dict.fromkeys(tool for item in ordered for tool in item.tools))
    return Skill(
        name=name,
        description=f"{len(ordered)} combined skills",
        body=body,
        tags=tags,
        tools=tools,
        metadata={"merged": [item.name for item in ordered]},
    )


def iter_skill_files(root: str | Path) -> Iterable[Path]:
    """Yield skill files under a directory."""
    base = Path(root).expanduser()
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in SKILL_EXTENSIONS:
            yield path


__all__ = [
    "FRONT_MATTER_RE",
    "KEY_RE",
    "LIST_FIELDS",
    "MAX_BODY_CHARS",
    "SKILL_EXTENSIONS",
    "Skill",
    "dump_skill",
    "iter_skill_files",
    "load_skill",
    "merge_skills",
    "parse_front_matter",
    "parse_skill",
    "skill_from_mapping",
]
