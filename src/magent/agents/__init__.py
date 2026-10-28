"""Ready-made agents and multi-agent teams.

Public API:

:class:`AgentPreset` / :data:`PRESETS` / :func:`build_preset_agent`
    tuned starting points - coder, reviewer, researcher, analyst, writer,
    devops, planner, tutor, librarian, sandboxed, minimal, general;
:class:`Team` / :class:`TeamMember` / :func:`run_team`
    multi-agent orchestration in five modes: sequential pipeline, round robin,
    supervisor routing, debate and parallel vote;
:class:`Handoff` / :func:`build_handoff`
    the structured note one agent leaves for the next.

>>> from magent.agents import get_preset, preset_names
>>> get_preset("coder").persona
'coder'
>>> len(preset_names()) >= 12
True
"""

from __future__ import annotations

from .handoff import HANDOFF_PROMPT, Handoff, build_handoff, handoff_block, parse_handoff
from .presets import (
    PRESET_CATEGORIES,
    PRESETS,
    AgentPreset,
    build_preset_agent,
    describe_presets,
    get_preset,
    list_presets,
    preset_for_task,
    preset_names,
    preset_table,
    presets_from_config,
    register_preset,
)
from .team import (
    TEAM_MODES,
    DebateRound,
    Team,
    TeamMember,
    TeamResult,
    TeamTurn,
    run_team,
    team_from_presets,
)

__all__ = [
    "HANDOFF_PROMPT",
    "PRESETS",
    "PRESET_CATEGORIES",
    "TEAM_MODES",
    "AgentPreset",
    "DebateRound",
    "Handoff",
    "Team",
    "TeamMember",
    "TeamResult",
    "TeamTurn",
    "build_handoff",
    "build_preset_agent",
    "describe_presets",
    "get_preset",
    "handoff_block",
    "list_presets",
    "parse_handoff",
    "preset_for_task",
    "preset_names",
    "preset_table",
    "presets_from_config",
    "register_preset",
    "run_team",
    "team_from_presets",
]
