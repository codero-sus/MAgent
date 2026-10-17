#!/usr/bin/env python3
"""Example 06: two agents debating, with a judge and handoffs.

Run: ``python examples/06_team_debate.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import backend, heading, main_guard
from magent.agents import (
    Team,
    TeamMember,
    build_handoff,
    build_preset_agent,
    handoff_block,
    run_team,
)


def body() -> None:
    """Run a debate and a simple pipeline."""
    instance = backend()
    members = [
        build_preset_agent("analyst", backend=instance),
        build_preset_agent("researcher", backend=instance),
    ]
    judge = build_preset_agent("reviewer", backend=instance)

    heading("debate")
    roster = [
        TeamMember(name=f"member{index}", agent=member) for index, member in enumerate(members)
    ]
    roster.append(TeamMember(name="judge", agent=judge))
    team = Team(roster, mode="debate", judge="judge", max_rounds=1, max_turns=6)
    result = team.run("Is local model hosting better than a hosted API for a small team?")
    print(result.summary_line())
    print(result.output[:400])

    heading("turns")
    for turn in result.turns:
        print(f"  {turn.index + 1}. {turn.member:<10} {turn.tokens:>5} tokens  {turn.seconds:.2f}s")

    heading("handoff to the next agent")
    handoff = build_handoff(
        result, from_agent="judge", to_agent="writer", next_steps=["turn this into a short brief"]
    )
    print(handoff_block(handoff))

    heading("pipeline")
    pipeline = run_team(
        [(f"step{index}", member) for index, member in enumerate(members)],
        "List two risks of running models locally.",
        mode="sequential",
    )
    print(pipeline.summary_line())


if __name__ == "__main__":
    raise SystemExit(main_guard(body)())
