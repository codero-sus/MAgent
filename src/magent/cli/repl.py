"""The interactive chat loop behind ``magent chat``.

A REPL is not just ``input()``: it keeps one agent so the conversation has
continuity, streams tokens as they arrive, handles Ctrl-C without losing the
session, and offers slash commands for the things that are awkward to type in
prose - switching persona, listing tools, searching memory, saving a transcript.
Multi-line input is supported by ending a line with a backslash or leaving a
code fence open.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..errors import MAgentError
from ..output.render import RenderOptions, ResultRenderer
from ..types import EventKind, RunResult
from ..utils.files import write_json
from ..utils.text import ellipsize

PROMPT = "› "
"""Input prompt."""

CONTINUATION = "… "
"""Continuation prompt for multi-line input."""

MAX_INPUT_LINES = 60
"""Safety ceiling on a single pasted message."""


@dataclass(slots=True)
class SlashCommand:
    """One ``/command`` available inside the REPL."""

    name: str
    help: str
    handler: Callable[["Repl", str], bool]
    aliases: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"SlashCommand(name={self.name!r})"


class Repl:
    """Interactive session over one agent."""

    def __init__(self, ctx: Any, *, session_id: str = "", preset: str = "",
                 stream: bool = True, agent: Any = None) -> None:
        """Create the REPL.

        Args:
            ctx: the CLI context
            session_id: session to resume or create
            preset: agent preset to use for the whole session
            stream: stream tokens as they arrive
            agent: explicit agent, overriding the preset/default
        """
        self.ctx = ctx
        self.session_id = session_id or ""
        self.preset = preset
        self.stream = stream
        self.renderer = ResultRenderer(RenderOptions.from_config(ctx.config), sink=ctx.sink)
        self.agent = agent if agent is not None else self._build_agent()
        self.turns = 0
        self.started_at = time.time()
        self.history: list[tuple[str, str]] = []
        self.commands = {item.name: item for item in self.default_commands()}
        for item in self.default_commands():
            for alias in item.aliases:
                self.commands[alias] = item
        self._running = True

    def _build_agent(self) -> Any:
        """Create the agent this session will use."""
        if self.preset:
            return self.ctx.build_preset_agent(self.preset,
                                               name=f"MAgent[{self.preset}]")
        return self.ctx.build_agent()

    # -- slash commands ------------------------------------------------------

    @staticmethod
    def default_commands() -> list[SlashCommand]:
        """The built-in slash commands."""
        return [
            SlashCommand("help", "list commands", lambda repl, arg: repl.cmd_help(arg), ("?", "h")),
            SlashCommand("tools", "list or search tools", lambda repl, arg: repl.cmd_tools(arg)),
            SlashCommand("memory", "search long-term memory", lambda repl, arg: repl.cmd_memory(arg)),
            SlashCommand("recall", "inject memories into the next prompt",
                         lambda repl, arg: repl.cmd_recall(arg)),
            SlashCommand("preset", "switch agent preset", lambda repl, arg: repl.cmd_preset(arg)),
            SlashCommand("persona", "switch persona", lambda repl, arg: repl.cmd_persona(arg)),
            SlashCommand("files", "attach files to the next message",
                         lambda repl, arg: repl.cmd_files(arg)),
            SlashCommand("save", "save the session transcript", lambda repl, arg: repl.cmd_save(arg)),
            SlashCommand("sessions", "list stored sessions",
                         lambda repl, arg: repl.cmd_sessions(arg)),
            SlashCommand("stats", "show session statistics", lambda repl, arg: repl.cmd_stats(arg)),
            SlashCommand("skills", "show which skills apply to text",
                         lambda repl, arg: repl.cmd_skills(arg)),
            SlashCommand("clear", "forget the conversation so far",
                         lambda repl, arg: repl.cmd_clear(arg)),
            SlashCommand("quit", "leave the session", lambda repl, arg: repl.cmd_quit(arg),
                         ("exit", "q")),
        ]

    def register(self, command: SlashCommand) -> SlashCommand:
        """Add a slash command (used by plugins and tests)."""
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command
        return command

    def dispatch(self, line: str) -> bool:
        """Handle one line, returning False when the session should end."""
        text = line.strip()
        if not text.startswith("/"):
            return self.ask(text) is not None
        name, _, argument = text[1:].partition(" ")
        command = self.commands.get(name.strip().lower())
        if command is None:
            self.ctx.notice(f"unknown command /{name} - try /help", level="warning")
            return True
        return command.handler(self, argument.strip())

    # -- slash implementations ----------------------------------------------

    def cmd_help(self, argument: str = "") -> bool:
        """Print the command list."""
        del argument
        rows = [(f"/{item.name}", item.help) for item in self.commands.values()]
        seen: set[str] = set()
        unique = [row for row in rows if not (row[0] in seen or seen.add(row[0]))]
        self.ctx.table(unique, ["command", "what it does"], title="slash commands")
        self.ctx.print("")
        self.ctx.print("Anything else you type is sent to the agent. Ctrl-D or /quit leaves.")
        return True

    def cmd_tools(self, argument: str = "") -> bool:
        """List tools, optionally filtered by a search term.

        ``registry.search`` returns specifications (the discovery path used by the
        ``list_tools`` meta tool), so the rows are read from those rather than
        from ``Tool`` objects.
        """
        registry = self.ctx.registry
        specs = registry.search(text=argument) if argument else [tool.spec for tool in registry.tools()]
        rows = [(spec.name, spec.category, spec.risk,
                 ellipsize(spec.description.split("\n")[0], 60)) for spec in specs[:40]]
        self.ctx.table(rows, ["name", "category", "risk", "description"],
                       title=f"{len(specs)} tools" + (f" matching {argument!r}" if argument else ""))
        return True

    def cmd_memory(self, argument: str = "") -> bool:
        """Search memory."""
        store = self.ctx.memory
        if store is None:
            self.ctx.notice("memory is disabled", level="warning")
            return True
        entries = store.search(argument, limit=10) if argument else store.recent(10)
        if not entries:
            self.ctx.notice("nothing found")
            return True
        rows = [(entry.id[:10], entry.kind, f"{entry.score:.2f}", ellipsize(entry.content, 60))
                for entry in entries]
        self.ctx.table(rows, ["id", "kind", "score", "content"])
        return True

    def cmd_recall(self, argument: str = "") -> bool:
        """Attach remembered context to the next message."""
        store = self.ctx.memory
        if store is None:
            self.ctx.notice("memory is disabled", level="warning")
            return True
        entries = store.search(argument, limit=5) if argument else []
        if not entries:
            self.ctx.notice("no matching memories to attach")
            return True
        block = "\n".join(f"- {entry.content}" for entry in entries)
        self.agent.prompt_defaults["memory_block"] = block
        self.ctx.notice(f"attached {len(entries)} memories to the next prompt")
        return True

    def cmd_preset(self, argument: str = "") -> bool:
        """Switch to another preset, keeping the session."""
        from ..agents.presets import preset_names

        if not argument:
            self.ctx.print("presets: " + ", ".join(preset_names()))
            return True
        try:
            self.agent = self.ctx.build_preset_agent(argument, name=f"MAgent[{argument}]")
        except MAgentError as exc:
            self.ctx.notice(str(exc), level="error")
            return True
        self.preset = argument
        self.ctx.notice(f"switched to preset {argument}")
        return True

    def cmd_persona(self, argument: str = "") -> bool:
        """Switch persona on the running agent."""
        from ..prompts.personas import PERSONAS

        if not argument:
            self.ctx.print("personas: " + ", ".join(sorted(PERSONAS)))
            return True
        if argument not in PERSONAS:
            self.ctx.notice(f"unknown persona {argument!r}", level="error")
            return True
        self.agent.config.persona = argument
        self.ctx.notice(f"persona set to {argument}")
        return True

    def cmd_files(self, argument: str = "") -> bool:
        """Attach files to the next prompt."""
        paths = [item for item in argument.split() if item]
        if not paths:
            self.ctx.print("usage: /files path [path...]")
            return True
        from ..utils.files import read_text_safe

        blocks = []
        for path in paths:
            try:
                text = read_text_safe(Path(path).expanduser(), limit=100_000)
            except Exception as exc:  # noqa: BLE001 - reported to the user
                self.ctx.notice(f"{path}: {exc}", level="warning")
                continue
            blocks.append(f"## File: {path}\n\n{text.strip()}")
        if blocks:
            self.agent.prompt_defaults["context_block"] = "\n\n".join(blocks)
            self.ctx.notice(f"attached {len(blocks)} files to the next prompt")
        return True

    def cmd_save(self, argument: str = "") -> bool:
        """Write the session transcript to disk."""
        target = Path(argument).expanduser() if argument else Path("magent-session.json")
        sessions = self.ctx.sessions
        record = getattr(sessions, "current", None)
        if record is None:
            write_json(target, {"turns": self.history, "preset": self.preset})
        else:
            sessions.save()
            write_json(target, record.to_dict())
        self.ctx.notice(f"saved transcript to {target}")
        return True

    def cmd_sessions(self, argument: str = "") -> bool:
        """List stored sessions."""
        del argument
        infos = self.ctx.sessions.list(limit=15)
        if not infos:
            self.ctx.notice("no stored sessions")
            return True
        rows = [(info.age, info.id[:16], str(info.messages),
                 ellipsize(info.title, 44)) for info in infos]
        self.ctx.table(rows, ["age", "id", "msgs", "title"])
        return True

    def cmd_stats(self, argument: str = "") -> bool:
        """Show session counters."""
        del argument
        self.ctx.key_values([
            ("turns", self.turns),
            ("minutes", round((time.time() - self.started_at) / 60, 2)),
            ("preset", self.preset or "(default)"),
            ("session", self.session_id or "(none)"),
            ("model", getattr(self.agent, "model", "") or "unknown"),
            ("tools", self.ctx.registry.count()),
        ], title="session")
        return True

    def cmd_skills(self, argument: str = "") -> bool:
        """Show which skills match some text."""
        library = self.ctx.skills
        matches = library.search(argument) if argument else []
        if not argument:
            self.ctx.print("skills: " + ", ".join(library.names()))
            return True
        if not matches:
            self.ctx.notice("no skill matched")
            return True
        rows = [(match.skill.name, f"{match.score:.2f}", ", ".join(match.reasons[:3]))
                for match in matches[:8]]
        self.ctx.table(rows, ["skill", "score", "why"])
        return True

    def cmd_clear(self, argument: str = "") -> bool:
        """Forget the conversation so far."""
        del argument
        self.agent.reset()
        self.history.clear()
        self.ctx.notice("conversation cleared")
        return True

    def cmd_quit(self, argument: str = "") -> bool:
        """Leave the session."""
        del argument
        self._running = False
        return False

    # -- running -------------------------------------------------------------

    def ask(self, text: str) -> RunResult | None:
        """Send one message and render the answer."""
        if not text.strip():
            return None
        try:
            if self.stream and not self.ctx.quiet and self.ctx.config.output.format == "text":
                result = self._stream(text)
            else:
                result = self.agent.run(text, session_id=self.session_id)
                self.renderer.print_result(result)
        except KeyboardInterrupt:
            self.ctx.notice("interrupted - the session continues", level="warning")
            return None
        except MAgentError as exc:
            self.ctx.error(str(exc))
            return None
        self.turns += 1
        output = result.output if result is not None else ""
        self.history.append((text, output))
        return result

    def _stream(self, text: str) -> RunResult | None:
        """Run with token streaming and return the final result."""
        result: RunResult | None = None
        in_text = False
        for event in self.agent.stream(text, session_id=self.session_id):
            if event.kind == EventKind.TEXT_DELTA:
                self.ctx.write(str(event.data.get("text", "")))
                in_text = True
                continue
            line = self.renderer.render_event(event)
            if line:
                if in_text:
                    self.ctx.write("\n")
                    in_text = False
                self.ctx.print(line)
            if event.kind == EventKind.RUN_FINISHED:
                result = event.data.get("result")
        if in_text:
            self.ctx.write("\n")
        if result is not None:
            self.ctx.print("")
            self.renderer.print(self.renderer.render_footer(result))
        return result

    def read_message(self) -> str:
        """Read one message, honouring continuation lines and fenced blocks."""
        lines: list[str] = []
        pending = False
        fences = 0
        while True:
            try:
                line = input(CONTINUATION if pending else PROMPT)
            except EOFError:
                return "" if not lines else "\n".join(lines)
            except KeyboardInterrupt:
                print()
                if lines:
                    lines.clear()
                    continue
                return ""
            stripped = line.strip()
            if stripped.count("```") % 2:
                fences += 1 if fences % 2 == 0 else -1
            lines.append(line)
            pending = bool(line.endswith("\\")) or fences % 2 == 1
            if not pending or len(lines) >= MAX_INPUT_LINES:
                break
        return "\n".join(lines).replace("\\\n", "\n").strip()

    def banner(self) -> None:
        """Print the opening banner."""
        if self.ctx.quiet or self.ctx.config.output.format != "text":
            return
        preset = self.preset or "default"
        model = getattr(self.agent, "model", "") or "auto"
        self.ctx.print(self.renderer.options.style(f"MAgent chat · preset {preset} · model {model}",
                                                  "heading"))
        self.ctx.print(self.renderer.options.style(
            "type /help for commands, Ctrl-D to leave", "muted"))

    def run(self) -> int:
        """Run the loop until it ends."""
        self.banner()
        while self._running:
            try:
                message = self.read_message()
            except KeyboardInterrupt:
                print()
                continue
            if not message:
                break
            self.dispatch(message)
        self.close()
        return 0

    def close(self) -> None:
        """Save the session and say goodbye."""
        if self.ctx.sessions is not None:
            path = self.ctx.sessions.save()
            if path and self.turns:
                self.ctx.notice(f"session saved to {path}", level="muted")
        if self.turns and not self.ctx.quiet:
            self.ctx.print(self.renderer.options.style(
                f"{self.turns} turns in {time.time() - self.started_at:.0f}s", "muted"))

    def __repr__(self) -> str:
        """Compact developer representation."""
        return f"Repl(session={self.session_id!r}, preset={self.preset!r}, turns={self.turns})"


def run_repl(ctx: Any, *, prompt: str = "", **kwargs: Any) -> int:
    """Create a REPL, optionally ask one question first, then run it."""
    repl = Repl(ctx, **kwargs)
    if prompt:
        repl.ask(prompt)
    return repl.run()


def slash_command_names(repl: Repl) -> Sequence[str]:
    """Names of the slash commands, sorted."""
    return sorted({item.name for item in repl.commands.values()})


def feed(repl: Repl, lines: Iterable[str]) -> list[Any]:
    """Feed lines into a REPL programmatically; used by the tests."""
    results = []
    for line in lines:
        results.append(repl.dispatch(line))
    return results


__all__ = [
    "CONTINUATION",
    "MAX_INPUT_LINES",
    "PROMPT",
    "Repl",
    "SlashCommand",
    "feed",
    "run_repl",
    "slash_command_names",
]
