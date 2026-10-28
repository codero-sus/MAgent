"""Reusable task procedures for the agent to follow.

Public API:

:class:`Skill`
    one instruction set plus metadata, parsed from Markdown front matter;
:class:`SkillLibrary` / :func:`build_library`
    built-in skills, project overrides, keyword matching and prompt rendering;
:data:`BUILTIN_SKILLS`
    twelve shipped procedures - code review, debugging, commits, refactors,
    tests, research briefs, CSV analysis, release notes, incidents, API design,
    performance triage and documentation.

>>> from magent.skills import get_library
>>> library = get_library()
>>> library.for_task("please review this pull request")[0].name
'code-review'
"""

from __future__ import annotations

from .library import (
    BUILTIN_SKILLS,
    DEFAULT_SKILL_DIRS,
    MAX_SKILLS_IN_PROMPT,
    SkillLibrary,
    SkillMatch,
    build_library,
    get_library,
    register_skill,
    render_skill_prompt,
    skill_table,
)
from .skill import (
    FRONT_MATTER_RE,
    KEY_RE,
    LIST_FIELDS,
    MAX_BODY_CHARS,
    SKILL_EXTENSIONS,
    Skill,
    dump_skill,
    iter_skill_files,
    load_skill,
    merge_skills,
    parse_front_matter,
    parse_skill,
    skill_from_mapping,
)

__all__ = [
    "BUILTIN_SKILLS",
    "DEFAULT_SKILL_DIRS",
    "FRONT_MATTER_RE",
    "KEY_RE",
    "LIST_FIELDS",
    "MAX_BODY_CHARS",
    "MAX_SKILLS_IN_PROMPT",
    "SKILL_EXTENSIONS",
    "Skill",
    "SkillLibrary",
    "SkillMatch",
    "build_library",
    "dump_skill",
    "get_library",
    "iter_skill_files",
    "load_skill",
    "merge_skills",
    "parse_front_matter",
    "parse_skill",
    "register_skill",
    "render_skill_prompt",
    "skill_from_mapping",
    "skill_table",
]
