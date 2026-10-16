# Teams

A team is a roster of agents plus an orchestration mode.

```python
from magent.agents import Team, TeamMember, build_preset_agent
from magent.llm.factory import build_backend

backend = build_backend()
team = Team([
    TeamMember(name="researcher", agent=build_preset_agent("researcher", backend=backend)),
    TeamMember(name="writer", agent=build_preset_agent("writer", backend=backend)),
    TeamMember(name="reviewer", agent=build_preset_agent("reviewer", backend=backend)),
], mode="sequential")
result = team.run("write a short brief on local model hosting")
print(result.output)
print(result.summary_line())
print(result.transcript())
```

## Modes

| Mode | How it works |
| --- | --- |
| `sequential` | pipeline: each member refines the previous output |
| `round_robin` | repeated passes until a member says `DONE` or the turn budget ends |
| `supervisor` | one member routes work by naming the next speaker |
| `debate` | positions, critiques, convergence check, then a judge settles |
| `vote` | members answer in parallel, a judge synthesises |

```python
team = Team(members, mode="vote", judge="reviewer", max_workers=4)
team = Team(members, mode="debate", judge="writer", max_rounds=2)
```

Every mode records the same evidence: one `TeamTurn` per contribution with the
prompt, output, tokens, cost, duration, tools and stop reason, plus `DebateRound`
records for debates and per-member token totals.

## Handoffs

Consecutive members get a structured note rather than a wall of transcript:

```python
from magent.agents import build_handoff, handoff_block, parse_handoff

handoff = build_handoff(result, from_agent="researcher", to_agent="writer",
                        next_steps=["cite two sources"])
print(handoff_block(handoff))
```

A model can also produce one: `HANDOFF_PROMPT` asks for JSON, `parse_handoff`
tolerantly recovers it, and a prose reply still yields a usable handoff.

## From presets in one call

```python
from magent.agents import team_from_presets

team = team_from_presets(["researcher", "writer", "reviewer"], mode="sequential")
print(team.describe())
```

## When not to use a team

A team costs several times a single run. Use one when the work genuinely splits
(research then write then check), when you want an adversarial second opinion
(`debate`), or when sampling several answers and picking the best is cheaper
than making one model better (`vote`).
