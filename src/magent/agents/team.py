"""Multi-agent teams: five orchestration modes over ordinary agents.

A team is a list of :class:`TeamMember` objects plus a mode.  The modes cover
the patterns that are actually useful with a local model host:

``sequential``
    pipeline - each member refines the previous member's output;
``round_robin``
    repeated passes over the roster until the work is declared done;
``supervisor``
    one member routes the task, naming who speaks next;
``debate``
    members propose, critique each other, and a judge settles the argument;
``vote``
    members answer the same task in parallel and a judge synthesises.

Every mode records the same things - each turn's prompt, output, tokens, cost,
duration and stop reason - plus the handoffs between members, so a team run is
auditable in exactly the same way as a single-agent run.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..errors import AgentError
from ..logging_ import get_logger
from ..types import RunResult, Usage
from ..utils.text import ellipsize
from .handoff import Handoff, build_handoff, handoff_block

LOGGER = get_logger("agents.team")
"""Logger for team orchestration."""

TEAM_MODES: tuple[str, ...] = ("sequential", "round_robin", "supervisor", "debate", "vote")
"""Orchestration modes a :class:`Team` understands."""

STOP_PHRASES: tuple[str, ...] = ("DONE", "FINISHED", "COMPLETE", "NO_FURTHER_WORK")
"""Words a supervisor or member may use to end a loop."""

ROUTE_RE = re.compile(r"\b(?:next|route|assign|speaker)\s*[:=]?\s*([a-z0-9_\- ]+)", re.IGNORECASE)
"""Routing line emitted by a supervisor."""

MAX_TURNS = 12
"""Default ceiling on turns across a team run."""

JUDGE_PROMPT = """You are the judge of a team discussion.

Task: {task}

Candidate answers:
{answers}

Write the single best answer. Use the strongest points from the candidates,
drop anything unsupported, and say plainly when the candidates disagree on a
point that matters. Do not mention the candidates or the judging process.
"""
"""Prompt used to synthesise a final answer from several."""

CRITIQUE_PROMPT = """Task: {task}

Another team member proposed this answer:

{answer}

List the concrete weaknesses: missing evidence, wrong claims, unhandled edge
cases. Then give the corrected answer. Be specific and brief.
"""
"""Prompt used during a debate round."""

SUPERVISOR_PROMPT = """You are routing work in a team.

Task: {task}

Members:
{roster}

Progress so far:
{progress}

Reply with exactly one line: either `next: <member>` to assign the next speaker,
or `DONE` when the task is complete. Nothing else.
"""
"""Prompt used by the supervisor mode."""


@dataclass(slots=True)
class TeamMember:
    """One participant in a team."""

    name: str
    agent: Any
    role: str = ""
    preset: str = ""
    weight: float = 1.0
    max_turns: int = 3
    instructions: str = ""
    enabled: bool = True
    turns_used: int = 0

    def __post_init__(self) -> None:
        """Default the role to the preset name and clamp the weight."""
        if not self.role:
            self.role = self.preset or "member"
        self.weight = max(0.0, float(self.weight))
        self.max_turns = max(1, int(self.max_turns))

    def prompt_for(self, task: str, *, previous: str = "", extra: str = "") -> str:
        """Build the prompt this member receives."""
        parts = [task.strip()]
        if previous.strip():
            parts.append("Previous work:\n" + previous.strip())
        if self.instructions.strip():
            parts.append("Your role: " + self.instructions.strip())
        if extra.strip():
            parts.append(extra.strip())
        return "\n\n".join(part for part in parts if part)

    def ask(self, prompt: str, **options: Any) -> tuple[str, RunResult]:
        """Run the member's agent once; returns ``(text, result)``."""
        result = self.agent.run(prompt, **options)
        return result.output or "", result

    def reset(self) -> None:
        """Clear the per-run turn counter."""
        self.turns_used = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the member (without the agent object)."""
        return {
            "name": self.name,
            "role": self.role,
            "preset": self.preset,
            "weight": self.weight,
            "max_turns": self.max_turns,
            "instructions": self.instructions,
            "enabled": self.enabled,
            "turns_used": self.turns_used,
            "agent": type(self.agent).__name__,
        }


@dataclass(slots=True)
class TeamTurn:
    """One contribution by one member."""

    index: int
    member: str
    prompt: str
    output: str
    seconds: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    ok: bool = True
    model: str = ""
    stop_reason: str = ""
    error: str = ""
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, index: int, member: str, prompt: str, result: RunResult,
                    seconds: float) -> TeamTurn:
        """Build a turn from a finished run."""
        return cls(
            index=index, member=member, prompt=prompt, output=result.output or "",
            seconds=round(seconds, 4), tokens=result.usage.total_tokens,
            cost=round(getattr(result.cost, "total", 0.0) or 0.0, 6),
            ok=bool(result.succeeded), model=result.model,
            stop_reason=str(result.stop_reason), error=result.error or "",
            tools=sorted({call.name for step in result.steps for call in step.tool_calls}),
            metadata={"run_id": result.run_id, "steps": len(result.steps)},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the turn."""
        return {
            "index": self.index,
            "member": self.member,
            "prompt": ellipsize(self.prompt, 400),
            "output": self.output,
            "seconds": self.seconds,
            "tokens": self.tokens,
            "cost": self.cost,
            "ok": self.ok,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "tools": list(self.tools),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DebateRound:
    """One round of proposals and critiques."""

    index: int
    positions: dict[str, str] = field(default_factory=dict)
    critiques: dict[str, str] = field(default_factory=dict)
    converged: bool = False
    judgement: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the round."""
        return {
            "index": self.index,
            "positions": dict(self.positions),
            "critiques": dict(self.critiques),
            "converged": self.converged,
            "judgement": self.judgement,
            "seconds": round(self.seconds, 4),
        }


@dataclass(slots=True)
class TeamResult:
    """Outcome of a team run."""

    task: str = ""
    mode: str = "sequential"
    output: str = ""
    turns: list[TeamTurn] = field(default_factory=list)
    rounds: list[DebateRound] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    seconds: float = 0.0
    members: list[str] = field(default_factory=list)
    judge: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True when the team produced an answer and no turn failed hard."""
        return bool(self.output.strip()) and not self.error

    @property
    def tokens(self) -> int:
        """Total tokens spent by the team."""
        return self.usage.total_tokens

    @property
    def turn_count(self) -> int:
        """Number of member contributions."""
        return len(self.turns)

    def by_member(self) -> dict[str, list[TeamTurn]]:
        """Turns grouped by member name."""
        grouped: dict[str, list[TeamTurn]] = {}
        for turn in self.turns:
            grouped.setdefault(turn.member, []).append(turn)
        return grouped

    def member_tokens(self) -> dict[str, int]:
        """Tokens spent per member."""
        out: dict[str, int] = {}
        for turn in self.turns:
            out[turn.member] = out.get(turn.member, 0) + turn.tokens
        return out

    def transcript(self, *, max_chars: int = 1_200) -> str:
        """Human-readable transcript of the run."""
        lines = [f"# Team run ({self.mode}): {ellipsize(self.task, 120)}"]
        for turn in self.turns:
            lines.append(f"\n## {turn.index + 1}. {turn.member} "
                         f"({turn.seconds:.2f}s, {turn.tokens} tokens)")
            lines.append(ellipsize(turn.output.strip(), max_chars))
        if self.rounds:
            lines.append("\n## Debate")
            for round_ in self.rounds:
                lines.append(f"- round {round_.index + 1}: "
                             f"{len(round_.positions)} positions, "
                             f"{len(round_.critiques)} critiques, "
                             f"converged={round_.converged}")
        lines.append(f"\n## Final ({self.judge or 'team'})")
        lines.append(ellipsize(self.output.strip(), max_chars * 2))
        return "\n".join(lines)

    def summary_line(self) -> str:
        """One-line summary for the CLI."""
        return (f"team[{self.mode}] {len(self.turns)} turns, {self.tokens} tokens, "
                f"{self.cost:.4f} cost, {self.seconds:.2f}s -> "
                f"{'ok' if self.succeeded else self.error or 'no output'}")

    def to_dict(self, *, include_outputs: bool = True) -> dict[str, Any]:
        """Serialise the result."""
        turns = [turn.to_dict() for turn in self.turns]
        if not include_outputs:
            for turn in turns:
                turn["output"] = ellipsize(turn["output"], 120)
        return {
            "task": self.task,
            "mode": self.mode,
            "output": self.output,
            "succeeded": self.succeeded,
            "members": list(self.members),
            "judge": self.judge,
            "turns": turns,
            "rounds": [round_.to_dict() for round_ in self.rounds],
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "usage": self.usage.to_dict(),
            "cost": round(self.cost, 6),
            "seconds": round(self.seconds, 4),
            "tokens_per_member": self.member_tokens(),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class Team:
    """A roster of agents plus an orchestration mode."""

    def __init__(self, members: Sequence[TeamMember] = (), *, mode: str = "sequential",
                 judge: TeamMember | str | None = None, max_rounds: int = 2,
                 max_turns: int = MAX_TURNS, name: str = "team", parallel: bool = True,
                 max_workers: int = 4, stop_phrases: Sequence[str] = STOP_PHRASES,
                 record_handoffs: bool = True, on_turn: Callable[[TeamTurn], None] | None = None,
                 options: dict[str, Any] | None = None) -> None:
        """Build a team.

        Args:
            members: participants, in speaking order
            mode: one of :data:`TEAM_MODES`
            judge: member name or member used to settle debate/vote modes
            max_rounds: rounds for debate and round-robin modes
            max_turns: hard ceiling on total turns
            name: team name, used in logs and results
            parallel: allow parallel calls in vote mode
            max_workers: thread ceiling for parallel calls
            stop_phrases: words that end a loop early
            record_handoffs: build a handoff between consecutive members
            on_turn: callback invoked after every turn
            options: default run options passed to each member's agent

        Raises:
            AgentError: when the team has no enabled members
        """
        self.members = list(members)
        self.mode = mode if mode in TEAM_MODES else "sequential"
        self.judge_name = judge if isinstance(judge, str) else getattr(judge, "name", "")
        if isinstance(judge, TeamMember) and judge not in self.members:
            self.members.append(judge)
        self.max_rounds = max(1, int(max_rounds))
        self.max_turns = max(1, int(max_turns))
        self.name = name
        self.parallel = bool(parallel)
        self.max_workers = max(1, int(max_workers))
        self.stop_phrases = tuple(stop_phrases)
        self.record_handoffs = bool(record_handoffs)
        self.on_turn = on_turn
        self.options = dict(options or {})
        self.runs = 0
        if not self.active_members():
            raise AgentError("team has no enabled members", details={"team": name})

    # -- roster --------------------------------------------------------------

    @classmethod
    def from_presets(cls, names: Sequence[str], *, backend: Any = None, config: Any = None,
                     mode: str = "sequential", judge: str = "", **kwargs: Any) -> Team:
        """Build a team where each member is an agent created from a preset.

        Args:
            names: preset names, e.g. ``["researcher", "writer", "reviewer"]``
            backend: shared LLM backend for every member
            config: loaded configuration
            mode: orchestration mode
            judge: name of the member that settles debate/vote modes
            **kwargs: further :class:`Team` arguments
        """
        from .presets import build_preset_agent, get_preset

        members: list[TeamMember] = []
        for index, name in enumerate(names):
            preset = get_preset(name)
            agent = build_preset_agent(name, backend=backend, config=config)
            members.append(TeamMember(name=f"{preset.name}-{index + 1}", agent=agent,
                                      preset=preset.name, role=preset.description,
                                      instructions=preset.instructions))
        return cls(members, mode=mode, judge=judge, **kwargs)

    def add(self, member: TeamMember) -> TeamMember:
        """Append a member."""
        self.members.append(member)
        return member

    def add_agent(self, name: str, agent: Any, **kwargs: Any) -> TeamMember:
        """Append a member built from an existing agent."""
        return self.add(TeamMember(name=name, agent=agent, **kwargs))

    def active_members(self) -> list[TeamMember]:
        """Enabled members, in speaking order."""
        return [member for member in self.members if member.enabled]

    def member(self, name: str) -> TeamMember | None:
        """Find a member by exact name, then by prefix."""
        for member in self.members:
            if member.name == name:
                return member
        needle = (name or "").strip().lower()
        for member in self.members:
            if member.name.lower().startswith(needle) or member.preset == needle:
                return member
        return None

    def judge_member(self) -> TeamMember | None:
        """Resolve the judge: named member, else the heaviest, else the last."""
        if self.judge_name:
            found = self.member(self.judge_name)
            if found is not None:
                return found
        active = self.active_members()
        if not active:
            return None
        return max(active, key=lambda member: (member.weight, member.turns_used))

    def roster(self) -> list[dict[str, Any]]:
        """Serialise the roster."""
        return [member.to_dict() for member in self.members]

    def describe(self) -> str:
        """One-paragraph description of the team."""
        names = ", ".join(f"{member.name}({member.preset or member.role})"
                          for member in self.active_members())
        return f"team {self.name!r} mode={self.mode} judge={self.judge_name or '-'} members: {names}"

    # -- running -------------------------------------------------------------

    def run(self, task: str, *, context: str = "", **options: Any) -> TeamResult:
        """Run the team on a task.

        Args:
            task: what the team should achieve
            context: extra context prepended to the first member's prompt
            **options: run options forwarded to each member's agent

        Raises:
            AgentError: when the task is empty or no member can be reached
        """
        if not task.strip():
            raise AgentError("team task must not be empty")
        started = time.perf_counter()
        for member in self.members:
            member.reset()
        merged = {**self.options, **options}
        if context.strip():
            merged.setdefault("context_block", context.strip())
        runners: dict[str, Callable[[str, dict[str, Any]], TeamResult]] = {
            "sequential": self._run_sequential,
            "round_robin": self._run_round_robin,
            "supervisor": self._run_supervisor,
            "debate": self._run_debate,
            "vote": self._run_vote,
        }
        result = runners[self.mode](task, merged)
        result.seconds = time.perf_counter() - started
        result.members = [member.name for member in self.active_members()]
        result.judge = getattr(self.judge_member(), "name", "")
        result.metadata.setdefault("team", self.name)
        result.metadata.setdefault("roster", self.roster())
        self.runs += 1
        LOGGER.debug("%s", result.summary_line())
        return result

    def _new_result(self, task: str) -> TeamResult:
        """Create an empty result for the configured mode."""
        return TeamResult(task=task, mode=self.mode)

    def _speak(self, member: TeamMember, prompt: str, options: dict[str, Any],
               result: TeamResult) -> TeamTurn:
        """Run one member and record the turn."""
        started = time.perf_counter()
        try:
            text, run = member.ask(prompt, **options)
            turn = TeamTurn.from_result(len(result.turns), member.name, prompt, run,
                                        time.perf_counter() - started)
            result.usage = result.usage.merge(run.usage)
            result.cost = round(result.cost + turn.cost, 6)
        except Exception as exc:  # noqa: BLE001 - one failing member must not kill the team
            LOGGER.warning("member %s failed: %s", member.name, exc)
            turn = TeamTurn(index=len(result.turns), member=member.name, prompt=prompt,
                            output="", seconds=time.perf_counter() - started, ok=False,
                            error=f"{type(exc).__name__}: {exc}")
        member.turns_used += 1
        result.turns.append(turn)
        if self.on_turn is not None:
            try:
                self.on_turn(turn)
            except Exception as exc:  # noqa: BLE001 - callbacks must not break a run
                LOGGER.debug("on_turn callback failed: %s", exc)
        return turn

    def _handoff(self, result: TeamResult, previous: TeamTurn, current: TeamTurn,
                 task: str) -> Handoff:
        """Record the handoff between two consecutive turns."""
        handoff = build_handoff(
            task=task, from_agent=previous.member, to_agent=current.member,
            summary=previous.output,
            next_steps=[line for line in current.output.split("\n")[:3] if line.strip()],
            confidence=0.9 if previous.ok else 0.4,
            run_id=previous.metadata.get("run_id", ""),
        )
        result.handoffs.append(handoff)
        return handoff

    def _run_sequential(self, task: str, options: dict[str, Any]) -> TeamResult:
        """Pipeline mode: each member refines the previous output."""
        result = self._new_result(task)
        previous_text = ""
        previous_turn: TeamTurn | None = None
        for member in self.active_members():
            if len(result.turns) >= self.max_turns:
                result.metadata["stopped"] = "max_turns"
                break
            prompt = member.prompt_for(task, previous=previous_text)
            turn = self._speak(member, prompt, options, result)
            if previous_turn is not None and self.record_handoffs:
                self._handoff(result, previous_turn, turn, task)
            if turn.ok and turn.output.strip():
                previous_text = turn.output
                previous_turn = turn
        result.output = previous_text
        return result

    def _run_round_robin(self, task: str, options: dict[str, Any]) -> TeamResult:
        """Repeated passes over the roster until a member declares the work done."""
        result = self._new_result(task)
        previous_text = ""
        for round_index in range(self.max_rounds):
            for member in self.active_members():
                if len(result.turns) >= self.max_turns:
                    result.metadata["stopped"] = "max_turns"
                    break
                if member.turns_used >= member.max_turns:
                    continue
                prompt = member.prompt_for(task, previous=previous_text,
                                           extra=f"Pass {round_index + 1} of {self.max_rounds}: "
                                                 f"improve on the previous work, or reply "
                                                 f"{self.stop_phrases[0]} when it is complete.")
                turn = self._speak(member, prompt, options, result)
                if turn.ok and turn.output.strip():
                    previous_text = turn.output
                if self._declares_done(turn.output):
                    result.metadata["stopped"] = "member declared done"
                    result.output = previous_text
                    return result
        result.output = previous_text
        return result

    def _run_supervisor(self, task: str, options: dict[str, Any]) -> TeamResult:
        """Supervisor mode: one member routes the work turn by turn."""
        result = self._new_result(task)
        supervisor = self.judge_member()
        members = [member for member in self.active_members() if member is not supervisor]
        if supervisor is None or not members:
            LOGGER.debug("supervisor mode needs a judge and at least one other member")
            return self._run_sequential(task, options)
        roster = "\n".join(f"- {member.name}: {member.role}" for member in members)
        progress = "(nothing yet)"
        chosen: TeamMember | None = None
        while len(result.turns) < self.max_turns:
            prompt = SUPERVISOR_PROMPT.format(task=task, roster=roster, progress=progress)
            routing = self._speak(supervisor, prompt, options, result)
            if self._declares_done(routing.output):
                result.metadata["stopped"] = "supervisor declared done"
                break
            chosen = self._parse_route(routing.output, members)
            if chosen is None:
                result.metadata["stopped"] = "no route parsed"
                break
            if len(result.turns) + 1 >= self.max_turns:
                # Two turns per loop: routing plus the assigned work. Stop before
                # the pair would push the run past its ceiling.
                result.metadata["stopped"] = "max_turns"
                break
            work = chosen.prompt_for(task, previous=self._latest_output(result))
            turn = self._speak(chosen, work, options, result)
            progress = f"{chosen.name}: {ellipsize(turn.output, 600)}"
        else:
            result.metadata.setdefault("stopped", "max_turns")
        if chosen is not None:
            final_prompt = (f"Task: {task}\n\nWork produced by the team:\n"
                            f"{self._latest_output(result)}\n\n"
                            "Write the final answer. Use the team's work, correct anything "
                            "wrong, and do not mention the team.")
            final = self._speak(supervisor, final_prompt, options, result)
            result.output = final.output or self._latest_output(result)
        else:
            result.output = self._latest_output(result)
        return result

    def _run_debate(self, task: str, options: dict[str, Any]) -> TeamResult:
        """Debate mode: propose, critique, converge, judge."""
        result = self._new_result(task)
        members = self.active_members()
        judge = self.judge_member()
        debaters = [member for member in members if member is not judge] or members
        positions: dict[str, str] = {}
        for round_index in range(self.max_rounds):
            debate = DebateRound(index=round_index)
            started = time.perf_counter()
            for member in debaters:
                if len(result.turns) >= self.max_turns:
                    break
                previous = positions.get(member.name, "")
                prompt = member.prompt_for(
                    task, previous=previous,
                    extra=("Give your position. Be concrete and defend it."
                           if not previous else
                           "Revise your position in light of the critiques below.\n"
                           + "\n".join(f"{name}: {ellipsize(text, 300)}"
                                       for name, text in debate.critiques.items())))
                turn = self._speak(member, prompt, options, result)
                if turn.output.strip():
                    positions[member.name] = turn.output
                    debate.positions[member.name] = turn.output
            for member in debaters:
                target_name, target_text = self._critique_target(member, positions)
                if not target_text:
                    continue
                prompt = CRITIQUE_PROMPT.format(task=task, answer=ellipsize(target_text, 2_000))
                turn = self._speak(member, prompt, options, result)
                if turn.output.strip():
                    debate.critiques[f"{member.name}->{target_name}"] = turn.output
            debate.seconds = time.perf_counter() - started
            debate.converged = round_index > 0 and self._converged(positions)
            result.rounds.append(debate)
            if debate.converged:
                result.metadata["stopped"] = "positions converged"
                break
        answers = "\n\n".join(f"### {name}\n{text}" for name, text in positions.items())
        if judge is not None and answers:
            prompt = JUDGE_PROMPT.format(task=task, answers=ellipsize(answers, 6_000))
            final = self._speak(judge, prompt, options, result)
            result.output = final.output or answers
        else:
            result.output = answers
        return result

    def _run_vote(self, task: str, options: dict[str, Any]) -> TeamResult:
        """Vote mode: parallel answers, then a judge synthesises."""
        result = self._new_result(task)
        members = self.active_members()
        judge = self.judge_member()
        voters = [member for member in members if member is not judge] or members
        answers: dict[str, str] = {}
        if self.parallel and len(voters) > 1:
            turns: list[TeamTurn] = []
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(voters)),
                                    thread_name_prefix="team-vote") as pool:
                futures = {}
                for member in voters:
                    prompt = member.prompt_for(task)
                    started = time.perf_counter()
                    futures[pool.submit(self._safe_ask, member, prompt, options)] = (member,
                                                                                    prompt,
                                                                                    started)
                for future in as_completed(futures):
                    member, prompt, started = futures[future]
                    text, run, error = future.result()
                    turn = (TeamTurn.from_result(len(result.turns) + len(turns), member.name,
                                                 prompt, run, time.perf_counter() - started)
                            if run is not None else
                            TeamTurn(index=len(result.turns) + len(turns), member=member.name,
                                     prompt=prompt, output=text,
                                     seconds=time.perf_counter() - started, ok=False,
                                     error=error))
                    turns.append(turn)
                    if run is not None:
                        result.usage = result.usage.merge(run.usage)
                    if text.strip():
                        answers[member.name] = text
            for turn in turns:
                member = self.member(turn.member)
                if member is not None:
                    member.turns_used += 1
                result.turns.append(turn)
                result.cost = round(result.cost + turn.cost, 6)
                if self.on_turn is not None:
                    self.on_turn(turn)
        else:
            for member in voters:
                prompt = member.prompt_for(task)
                turn = self._speak(member, prompt, options, result)
                if turn.output.strip():
                    answers[member.name] = turn.output
        if judge is not None and answers:
            body = "\n\n".join(f"### {name}\n{text}" for name, text in answers.items())
            prompt = JUDGE_PROMPT.format(task=task, answers=ellipsize(body, 6_000))
            final = self._speak(judge, prompt, options, result)
            result.output = final.output or body
        else:
            result.output = self._best_answer(answers)
        result.metadata["votes"] = {name: len(text) for name, text in answers.items()}
        return result

    def _safe_ask(self, member: TeamMember, prompt: str,
                  options: dict[str, Any]) -> tuple[str, RunResult | None, str]:
        """Ask one member, converting failures into a triple instead of raising."""
        try:
            text, run = member.ask(prompt, **options)
            run.metadata.setdefault("team_member", member.name)
            return text, run, ""
        except Exception as exc:  # noqa: BLE001 - parallel votes must not raise into the pool
            LOGGER.warning("vote from %s failed: %s", member.name, exc)
            return "", None, f"{type(exc).__name__}: {exc}"

    # -- helpers -------------------------------------------------------------

    def _latest_output(self, result: TeamResult) -> str:
        """Most recent successful output."""
        for turn in reversed(result.turns):
            if turn.ok and turn.output.strip():
                return turn.output
        return ""

    def _declares_done(self, text: str) -> bool:
        """True when a member's reply is one of the stop phrases."""
        cleaned = (text or "").strip().upper().replace(" ", "_")
        if not cleaned:
            return False
        first_line = cleaned.split("\n", 1)[0].strip(" .!:")
        return any(first_line.startswith(phrase) for phrase in self.stop_phrases)

    def _parse_route(self, text: str, members: Sequence[TeamMember]) -> TeamMember | None:
        """Extract the next speaker from a supervisor reply."""
        match = ROUTE_RE.search(text or "")
        if match:
            found = self.member(match.group(1).strip())
            if found in members:
                return found
        for member in members:
            if member.name.lower() in (text or "").lower():
                return member
        return None

    def _critique_target(self, member: TeamMember,
                         positions: dict[str, str]) -> tuple[str, str]:
        """Pick the position this member should critique (not its own)."""
        others = [(name, text) for name, text in positions.items() if name != member.name]
        if not others:
            return "", ""
        others.sort(key=lambda item: -len(item[1]))
        return others[0]

    def _converged(self, positions: dict[str, str]) -> bool:
        """True when every pair of positions is textually similar."""
        from ..utils.text import similarity

        texts = list(positions.values())
        if len(texts) < 2:
            return True
        for index, first in enumerate(texts):
            for second in texts[index + 1:]:
                if similarity(first, second) < 0.75:
                    return False
        return True

    def _best_answer(self, answers: dict[str, str]) -> str:
        """Pick the longest answer when there is no judge."""
        if not answers:
            return ""
        return max(answers.items(), key=lambda item: len(item[1]))[1]

    def __len__(self) -> int:
        """Number of members."""
        return len(self.members)

    def __repr__(self) -> str:
        """Compact developer representation."""
        return (f"Team(name={self.name!r}, mode={self.mode!r}, "
                f"members={[m.name for m in self.members]})")


def run_team(members: Iterable[TeamMember] | Iterable[Any], task: str, *, mode: str = "sequential",
             judge: str = "", max_rounds: int = 2, backend: Any = None, config: Any = None,
             **kwargs: Any) -> TeamResult:
    """Build a team and run one task in a single call.

    Args:
        members: :class:`TeamMember` objects, or ``(name, agent)`` pairs
        task: what to achieve
        mode: orchestration mode
        judge: name of the member that settles debate/vote
        max_rounds: rounds for debate and round-robin
        backend: unused here, accepted for symmetry with :meth:`Team.from_presets`
        config: unused here, accepted for symmetry
        **kwargs: further :class:`Team` arguments
    """
    del backend, config
    roster: list[TeamMember] = []
    for item in members:
        if isinstance(item, TeamMember):
            roster.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            roster.append(TeamMember(name=str(item[0]), agent=item[1]))
        else:
            roster.append(TeamMember(name=getattr(item, "name", f"member-{len(roster) + 1}"),
                                     agent=item))
    team = Team(roster, mode=mode, judge=judge, max_rounds=max_rounds, **kwargs)
    return team.run(task)


def team_from_presets(names: Sequence[str], *, mode: str = "sequential", backend: Any = None,
                      config: Any = None, judge: str = "", **kwargs: Any) -> Team:
    """Convenience wrapper around :meth:`Team.from_presets`."""
    return Team.from_presets(names, backend=backend, config=config, mode=mode, judge=judge,
                             **kwargs)


def handoff_chain(result: TeamResult) -> str:
    """Render the chain of handoffs produced by a team run."""
    if not result.handoffs:
        return ""
    return "\n\n".join(handoff_block(handoff) for handoff in result.handoffs)


__all__ = [
    "CRITIQUE_PROMPT",
    "JUDGE_PROMPT",
    "MAX_TURNS",
    "ROUTE_RE",
    "STOP_PHRASES",
    "SUPERVISOR_PROMPT",
    "TEAM_MODES",
    "DebateRound",
    "Team",
    "TeamMember",
    "TeamResult",
    "TeamTurn",
    "handoff_chain",
    "run_team",
    "team_from_presets",
]
