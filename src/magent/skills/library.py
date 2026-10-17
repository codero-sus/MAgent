"""The skill library: built-in skills, matching and prompt composition.

:data:`BUILTIN_SKILLS` ships twelve skills that cover the work MAgent is
actually used for - debugging, code review, writing commits, planning a
refactor, producing a research brief, analysing a CSV, writing release notes.
They are ordinary :class:`~magent.skills.skill.Skill` objects, so a project can
override one by dropping a file with the same name into ``.magent/skills/``.

Matching is keyword based and explainable: :meth:`SkillLibrary.for_task` returns
the highest-scoring skills together with the score, and
:meth:`SkillLibrary.prompt_block` renders them as a single context block that the
context builder can compress like any other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import SkillNotFoundError
from ..logging_ import get_logger
from ..utils.text import ellipsize
from .skill import Skill, iter_skill_files, load_skill, merge_skills, skill_from_mapping

LOGGER = get_logger("skills.library")
"""Logger for skill loading."""

DEFAULT_SKILL_DIRS: tuple[str, ...] = (".magent/skills", "skills", "docs/skills")
"""Directories searched for project skills."""

MAX_SKILLS_IN_PROMPT = 3
"""How many matched skills may be injected into one prompt."""

BUILTIN_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="code-review",
        description="Review a diff or file for defects and risky changes",
        tags=("code", "review", "quality"),
        triggers=("review", "code review", "pull request", "look at this code", "audit"),
        tools=("read_file", "grep", "git_diff", "git_log"),
        priority=70,
        body=(
            "Work through the change once, then report findings only.\n\n"
            "1. Read the whole changed file, not just the diff, to judge intent.\n"
            "2. For each finding, give: severity (critical/major/minor), file and line, "
            "what is wrong, and the smallest fix.\n"
            "3. Check the usual failure classes first: unhandled errors, boundary "
            "conditions, resource leaks, blocking calls, injection, secrets in logs.\n"
            "4. Say what you verified by reading versus what you suspect.\n"
            "5. Never rewrite the code; propose the change."
        ),
    ),
    Skill(
        name="debug-loop",
        description="Find a bug by reproducing it, then narrowing until the cause is proven",
        tags=("code", "debug", "testing"),
        triggers=("debug", "why does", "not working", "traceback", "fails", "crash", "error"),
        tools=("read_file", "grep", "run_command", "python_exec"),
        priority=75,
        body=(
            "Reproduce before theorising.\n\n"
            "1. Write the smallest command that shows the failure and run it.\n"
            "2. Read the full traceback; the frame you are about to change is rarely the frame "
            "that is wrong.\n"
            "3. Form one hypothesis at a time and test it with a log line or a print, then "
            "remove the probe.\n"
            "4. Fix the cause, not the symptom; if you must change behaviour, say what else "
            "the change affects.\n"
            "5. Re-run the reproduction and show the output."
        ),
    ),
    Skill(
        name="commit-messages",
        description="Write a commit message that explains why, not just what",
        tags=("git", "writing"),
        triggers=("commit", "commit message", "changelog entry"),
        tools=("git_status", "git_diff", "git_log"),
        priority=60,
        body=(
            "Format: a subject line of at most 72 characters in the imperative mood, a blank "
            "line, then a body explaining why the change was needed and what changed in "
            "behaviour. Reference issues by number when you know them. Mention anything the "
            "reviewer must check by hand. Never list the files you touched - the diff shows "
            "that."
        ),
    ),
    Skill(
        name="refactor-plan",
        description="Plan a refactor as small, individually verifiable steps",
        tags=("code", "planning"),
        triggers=("refactor", "clean up", "restructure", "extract", "rename across"),
        tools=("read_file", "grep", "directory_tree"),
        priority=65,
        body=(
            "A refactor is a sequence of behaviour-preserving steps, each of which can be "
            "tested on its own.\n\n"
            "1. State the smell precisely, with file and line.\n"
            "2. Name the target design in one sentence.\n"
            "3. Break the path into steps that each keep the tests green.\n"
            "4. Say which step is riskiest and how to verify it.\n"
            "5. Never combine a refactor with a behaviour change in one step."
        ),
    ),
    Skill(
        name="test-plan",
        description="Design tests that would catch the bugs this change could introduce",
        tags=("testing", "quality"),
        triggers=("test", "unit test", "coverage", "test plan"),
        tools=("read_file", "grep", "run_command"),
        priority=60,
        body=(
            "For each behaviour the change touches, list: the setup, the action, the expected "
            "result, and the bug it would catch. Include at least one failure path and one "
            "boundary value (empty, single item, maximum). Prefer fast unit tests; mark the "
            "slow integration ones explicitly. Do not write a test that asserts the "
            "implementation rather than the behaviour."
        ),
    ),
    Skill(
        name="research-brief",
        description="Answer a question from sources with citations and an explicit confidence",
        tags=("research", "writing"),
        triggers=("research", "find out", "investigate", "compare", "what is the state of"),
        tools=("web_search", "web_fetch", "read_file", "rag_search", "memory_search"),
        priority=65,
        body=(
            "1. Restate the question as the specific thing to establish.\n"
            "2. Gather at least two independent sources; prefer primary material over "
            "summaries.\n"
            "3. For every claim, cite the source. Quote numbers exactly, with units and dates.\n"
            "4. Where sources disagree, say so and explain which is better evidence and why.\n"
            "5. Finish with what you could not establish and how you would find it."
        ),
    ),
    Skill(
        name="csv-analysis",
        description="Analyse a tabular file and report findings with the numbers behind them",
        tags=("data", "analysis"),
        triggers=("csv", "dataset", "spreadsheet", "analyse this data", "summary statistics"),
        tools=("data_read", "data_query", "data_aggregate", "python_exec", "output_write"),
        priority=65,
        body=(
            "1. Show the schema: columns, types, row count.\n"
            "2. Report missing values and obvious outliers before any aggregate.\n"
            "3. Give each finding as a number with its denominator (e.g. 120 of 480 rows, "
            "25.0%).\n"
            "4. State the filtering you applied; a filtered number without its filter is "
            "misleading.\n"
            "5. Do not infer causation from a correlation you computed."
        ),
    ),
    Skill(
        name="release-notes",
        description="Turn a range of commits into release notes grouped by effect",
        tags=("writing", "git", "release"),
        triggers=("release notes", "what changed", "changelog", "between versions"),
        tools=("git_log", "git_diff", "git_show"),
        priority=55,
        body=(
            "Group changes under Added, Changed, Fixed, Removed. Write for the person "
            "upgrading: what breaks, what must they do. Include issue numbers. Leave out "
            "internal refactors with no user-visible effect unless they change performance."
        ),
    ),
    Skill(
        name="incident-report",
        description="Write an incident report with timeline, cause and follow-ups",
        tags=("operations", "writing"),
        triggers=("incident", "outage", "postmortem", "went down", "downtime"),
        tools=("read_file", "memory_search", "output_write"),
        priority=60,
        body=(
            "Sections: impact (who and how long), timeline (timestamps, UTC), detection, root "
            "cause, contributing factors, resolution, follow-up actions with owners. Be "
            "specific and blame-free: describe the conditions that made the failure possible "
            "rather than the person who triggered it."
        ),
    ),
    Skill(
        name="api-design",
        description="Design or critique an HTTP API for consistency and change-safety",
        tags=("design", "api"),
        triggers=("api design", "endpoint", "rest", "schema design", "http api"),
        tools=("read_file", "grep"),
        priority=55,
        body=(
            "Check: resource naming, idempotency of unsafe-looking methods, error shape, "
            "pagination, filtering, versioning, auth scope, and what a client must do when a "
            "field is added or removed. Give concrete request and response examples for the "
            "normal path and one failure path."
        ),
    ),
    Skill(
        name="performance-triage",
        description="Find where the time goes before changing anything",
        tags=("performance", "debug"),
        triggers=("slow", "performance", "latency", "takes too long", "optimise", "optimize"),
        tools=("read_file", "grep", "run_command", "benchmark"),
        priority=60,
        body=(
            "Measure first. 1. State the budget (e.g. p95 under 200 ms) and the current "
            "number. 2. Find the hot path with a timer, a profiler, or `time` - not by "
            "reading. 3. Name the complexity, the allocation, or the I/O that dominates. "
            "4. Change one thing, re-measure, and report both numbers. 5. Say what you "
            "deliberately did not optimise."
        ),
    ),
    Skill(
        name="doc-writer",
        description="Write documentation for the task at hand, not for the author",
        tags=("writing", "docs"),
        triggers=("document", "readme", "docstring", "write docs", "explain how to"),
        tools=("read_file", "grep", "output_write"),
        priority=50,
        body=(
            "Start with what the reader wants to accomplish. Show a working example before "
            "the explanation. Document the failure modes and the limits, not only the happy "
            "path. Every code sample must be runnable as written - no ellipses in commands."
        ),
    ),
)
"""Skills every MAgent install has."""


@dataclass(slots=True)
class SkillMatch:
    """One skill matched against a task, with its score and the reason."""

    skill: Skill
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the match."""
        return {
            "skill": self.skill.name,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


class SkillLibrary:
    """A collection of skills with lookup, matching and prompt rendering."""

    def __init__(
        self,
        skills: Iterable[Skill] = (),
        *,
        include_builtins: bool = True,
        directory: str | Path | None = None,
        use_triggers: bool = True,
    ) -> None:
        """Build a library.

        Args:
            skills: extra skills to register
            include_builtins: start from :data:`BUILTIN_SKILLS`
            directory: directory to load skills from
            use_triggers: allow trigger matching in :meth:`for_task`
        """
        self.skills: dict[str, Skill] = {}
        self.directory = Path(directory).expanduser() if directory else None
        self.use_triggers = bool(use_triggers)
        self.load_errors: list[str] = []
        if include_builtins:
            for skill in BUILTIN_SKILLS:
                self.add(skill)
        for skill in skills:
            self.add(skill)

    # -- collection ----------------------------------------------------------

    def add(self, skill: Skill, *, replace: bool = True) -> Skill:
        """Register a skill.

        Raises:
            SkillFormatError: when the object is not a :class:`Skill`
        """
        from ..errors import SkillFormatError

        if not isinstance(skill, Skill):
            raise SkillFormatError(f"expected a Skill, got {type(skill).__name__}")
        if skill.name in self.skills and not replace:
            return self.skills[skill.name]
        self.skills[skill.name] = skill
        return skill

    def add_many(self, skills: Iterable[Skill], *, replace: bool = True) -> int:
        """Register several skills."""
        return sum(1 for skill in skills if self.add(skill, replace=replace) is not None)

    def remove(self, name: str) -> bool:
        """Remove a skill by name."""
        return self.skills.pop(name, None) is not None

    def get(self, name: str) -> Skill:
        """Look up a skill.

        Raises:
            SkillNotFoundError: when the name is unknown
        """
        key = str(name).strip().lower().replace(" ", "-")
        if key in self.skills:
            return self.skills[key]
        raise SkillNotFoundError(f"unknown skill {name!r}", details={"known": self.names()})

    def has(self, name: str) -> bool:
        """True when a skill exists."""
        return str(name).strip().lower().replace(" ", "-") in self.skills

    def names(self) -> list[str]:
        """Skill names, sorted."""
        return sorted(self.skills)

    def tags(self) -> list[str]:
        """Every tag used by the library."""
        return sorted({tag for skill in self.skills.values() for tag in skill.tags})

    def by_tag(self, tag: str) -> list[Skill]:
        """Skills carrying a tag."""
        needle = tag.strip().lower()
        return [skill for skill in self.skills.values() if needle in skill.tags]

    def search(self, text: str, *, limit: int = 10) -> list[SkillMatch]:
        """Rank skills against free text, best first."""
        matches: list[SkillMatch] = []
        for skill in self.skills.values():
            if not skill.enabled:
                continue
            score = skill.matches(text)
            if score <= 0:
                continue
            reasons: list[str] = []
            lowered = text.lower()
            for trigger in skill.triggers:
                if trigger.lower() in lowered:
                    reasons.append(f"trigger:{trigger}")
            for tag in skill.tags:
                if tag.lower() in lowered:
                    reasons.append(f"tag:{tag}")
            matches.append(SkillMatch(skill=skill, score=score, reasons=reasons))
        matches.sort(key=lambda item: (-item.score, -item.skill.priority, item.skill.name))
        return matches[: max(1, limit)]

    def for_task(
        self, task: str, *, limit: int = MAX_SKILLS_IN_PROMPT, min_score: float = 0.24
    ) -> list[Skill]:
        """Return the skills that apply to a task, highest scoring first."""
        if not self.use_triggers or not task.strip():
            return []
        matches = [
            match for match in self.search(task, limit=limit * 3) if match.score >= min_score
        ]
        chosen: list[Skill] = []
        for match in matches:
            chosen.append(match.skill)
            if len(chosen) >= limit:
                break
        return chosen

    # -- loading -------------------------------------------------------------

    def load_directory(self, root: str | Path | None = None, *, replace: bool = True) -> int:
        """Load every skill file under a directory; returns how many loaded."""
        base = Path(root).expanduser() if root else self.directory
        if base is None or not base.exists():
            return 0
        loaded = 0
        for path in iter_skill_files(base):
            try:
                self.add(load_skill(path), replace=replace)
                loaded += 1
            except Exception as exc:
                self.load_errors.append(f"{path}: {exc}")
                LOGGER.warning("skill %s failed to load: %s", path, exc)
        return loaded

    def load_config(self, config: Any = None) -> int:
        """Load skills declared under ``[skills.<name>]`` in configuration."""
        raw = getattr(config, "extra", None) or {}
        section = raw.get("skills") if isinstance(raw, dict) else None
        if not isinstance(section, dict):
            return 0
        loaded = 0
        for name, values in section.items():
            if isinstance(values, dict):
                self.add(skill_from_mapping(str(name), values))
                loaded += 1
        return loaded

    # -- rendering -----------------------------------------------------------

    def render(
        self,
        skills: Sequence[Skill] | None = None,
        *,
        style: str = "heading",
        max_chars: int = 6_000,
    ) -> str:
        """Render skills as prompt text."""
        chosen = list(skills if skills is not None else self.skills.values())
        if not chosen:
            return ""
        body = "\n\n".join(skill.render(style=style) for skill in chosen if not skill.is_empty)
        return ellipsize(body, max_chars)

    def prompt_block(
        self,
        task: str,
        *,
        limit: int = MAX_SKILLS_IN_PROMPT,
        style: str = "heading",
        max_chars: int = 4_000,
    ) -> str:
        """Return instructions for whichever skills match a task (may be empty)."""
        chosen = self.for_task(task, limit=limit)
        if not chosen:
            return ""
        header = (
            "Procedures that apply to this task - follow them unless the user asks otherwise:\n"
        )
        return header + self.render(chosen, style=style, max_chars=max_chars)

    def combined(self, names: Sequence[str], *, name: str = "combined") -> Skill:
        """Merge several skills into one."""
        return merge_skills([self.get(item) for item in names], name=name)

    # -- views ---------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Library counters."""
        skills = list(self.skills.values())
        return {
            "skills": len(skills),
            "builtins": len(BUILTIN_SKILLS),
            "custom": sum(
                1 for skill in skills if skill.name not in {item.name for item in BUILTIN_SKILLS}
            ),
            "disabled": sum(1 for skill in skills if not skill.enabled),
            "tags": len(self.tags()),
            "words": sum(skill.words for skill in skills),
            "directory": str(self.directory) if self.directory else "",
            "load_errors": list(self.load_errors[-5:]),
        }

    def describe(self) -> str:
        """One-paragraph description of the library."""
        info = self.stats()
        return (
            f"{info['skills']} skills ({info['builtins']} built-in, {info['custom']} custom), "
            f"{info['words']} words of instructions"
        )

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        """Serialise the library."""
        return {
            "skills": {
                name: skill.to_dict(include_body=include_body)
                for name, skill in sorted(self.skills.items())
            },
            "stats": self.stats(),
        }

    def __len__(self) -> int:
        """Number of skills."""
        return len(self.skills)

    def __contains__(self, name: object) -> bool:
        """True when a skill exists."""
        return self.has(str(name))

    def __iter__(self) -> Any:
        """Iterate skills in name order."""
        return iter(self.skills[name] for name in self.names())

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"SkillLibrary(skills={len(self.skills)}, dir={self.directory})"


def register_skill(skill: Skill) -> Skill:
    """Register a skill in the process-wide default library.

    Used by plugins and ``[skills.*]`` config sections.
    """
    return get_library().add(skill)


_LIBRARY: SkillLibrary | None = None
"""Process-wide default library, built on first use."""


def get_library(*, reload: bool = False, directory: str | Path | None = None) -> SkillLibrary:
    """Return the process-wide default library.

    Args:
        reload: rebuild the library from scratch
        directory: directory to load project skills from
    """
    global _LIBRARY
    if _LIBRARY is None or reload:
        library = SkillLibrary(directory=directory)
        if directory is not None:
            library.load_directory()
        _LIBRARY = library
    return _LIBRARY


def build_library(
    config: Any = None, *, workspace: str | Path | None = None, include_builtins: bool = True
) -> SkillLibrary:
    """Build a library for a configuration and workspace.

    Skills are looked up in ``.magent/skills``, ``skills`` and ``docs/skills``
    under the workspace, then in the user data directory, and finally from the
    ``[skills.*]`` config section.  A project skill with the same name as a
    built-in wins, which is how a team overrides a shipped procedure.
    """
    root = Path(workspace or getattr(getattr(config, "tools", None), "workspace", "") or Path.cwd())
    library = SkillLibrary(include_builtins=include_builtins)
    for name in DEFAULT_SKILL_DIRS:
        library.load_directory(root / name)
    data_dir = getattr(config, "data_dir", "") or (Path.home() / ".local" / "share" / "magent")
    library.load_directory(Path(data_dir) / "skills")
    library.load_config(config)
    return library


def skill_table(library: SkillLibrary | None = None) -> list[tuple[str, str, str, str]]:
    """Rows of ``(name, priority, tags, description)`` for CLI tables."""
    lib = library or get_library()
    return [
        (skill.name, str(skill.priority), ",".join(skill.tags[:3]), skill.description)
        for skill in lib
    ]


def render_skill_prompt(skills: Sequence[Skill], *, style: str = "heading") -> str:
    """Render a list of skills as an instruction block."""
    library = SkillLibrary(include_builtins=False)
    return library.render(skills, style=style)


__all__ = [
    "BUILTIN_SKILLS",
    "DEFAULT_SKILL_DIRS",
    "MAX_SKILLS_IN_PROMPT",
    "SkillLibrary",
    "SkillMatch",
    "build_library",
    "get_library",
    "register_skill",
    "render_skill_prompt",
    "skill_table",
]
