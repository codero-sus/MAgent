"""Commands that run the agent: ``run``, ``ask``, ``chat``, ``plan``, ``reflect``.

These are the commands people actually type, so they share one implementation
of "load the prompt, choose an agent, stream or not, render the result".  The
differences are only in the default options and the output shape: ``ask`` prints
just the answer (so it can be piped), ``run`` prints the whole result with usage,
``plan`` stops after planning, and ``reflect`` adds a self-critique.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..commands import Command
from ...errors import MAgentError
from ...output.render import RenderOptions, ResultRenderer
from ...types import EventKind
from ...utils.files import read_text_safe, write_json
from ...utils.text import ellipsize

PROMPT_FILE_LIMIT = 200_000
"""Characters read from ``--prompt-file`` or ``--file``."""


def _read_prompt(args: argparse.Namespace, extra: str = "") -> str:
    """Assemble the prompt from the argument, a file, and piped stdin."""
    parts: list[str] = []
    prompt = (getattr(args, "prompt", "") or "").strip()
    if prompt:
        parts.append(prompt)
    path = getattr(args, "prompt_file", "") or ""
    if path:
        parts.append(read_text_safe(Path(path).expanduser(), limit=PROMPT_FILE_LIMIT).strip())
    if getattr(args, "stdin", False) and not sys.stdin.isatty():
        piped = sys.stdin.read(PROMPT_FILE_LIMIT).strip()
        if piped:
            parts.append(piped)
    if extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(part for part in parts if part)


def _run_options(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into ``Agent.run`` options."""
    options: dict[str, Any] = {}
    for flag, key in (("loop_mode", "loop_mode"), ("persona", "persona"),
                      ("system", "system"), ("task_block", "task_block"),
                      ("context_block", "context_block")):
        value = getattr(args, flag, "") or ""
        if value:
            options[key] = value
    if getattr(args, "max_iterations", 0):
        options["max_iterations"] = int(args.max_iterations)
    if getattr(args, "temperature", None) is not None:
        options["temperature"] = float(args.temperature)
    if getattr(args, "max_tokens", 0):
        options["max_tokens"] = int(args.max_tokens)
    if getattr(args, "model", ""):
        options["model"] = args.model
    if getattr(args, "dry_run", False):
        options["dry_run"] = True
    if getattr(args, "structured", False):
        options["structured"] = True
    if getattr(args, "include_files", False):
        options["include_files"] = True
    if getattr(args, "memory_recall", False):
        options["recall_memories"] = True
    if getattr(args, "remember", False):
        options["remember_outcome"] = True
    if getattr(args, "session", ""):
        options["session_id"] = args.session
    if getattr(args, "tools", None):
        options["tools"] = list(args.tools)
        options["disabled_tools"] = list(args.tools)
    return options


def _context_block(args: argparse.Namespace) -> str:
    """Build the context block from ``--file`` arguments and skill matches."""
    blocks: list[str] = []
    for path in getattr(args, "file", None) or []:
        target = Path(path).expanduser()
        try:
            text = read_text_safe(target, limit=PROMPT_FILE_LIMIT)
        except Exception as exc:  # noqa: BLE001 - a missing file is reported, not fatal
            blocks.append(f"File {target}: unreadable ({exc})")
            continue
        blocks.append(f"## File: {target}\n\n{text.strip()}")
    return "\n\n".join(blocks)


def _agent_command(ctx: Any, args: argparse.Namespace, *, extra_prompt: str = "") -> int:
    """Shared implementation for the agent-facing commands."""
    prompt = _read_prompt(args, extra_prompt)
    if not prompt:
        ctx.error("no prompt given (pass text, --prompt-file, or pipe stdin)")
        return 2
    preset = getattr(args, "preset", "") or ""
    options = _run_options(args)
    context_block = _context_block(args)
    if context_block:
        options["context_block"] = "\n\n".join(
            part for part in (options.get("context_block", ""), context_block) if part)
    skills = ctx.skills
    if not getattr(args, "no_skills", False) and skills is not None:
        block = skills.prompt_block(prompt)
        if block:
            options["context_block"] = "\n\n".join(
                part for part in (options.get("context_block", ""), block) if part)
    agent = ctx.build_preset_agent(preset, name=f"MAgent[{preset}]" if preset else "MAgent") \
        if preset else ctx.build_agent()
    want_stream = bool(getattr(args, "stream", False)) and not ctx.quiet \
        and ctx.config.output.format == "text"
    renderer = ResultRenderer(RenderOptions.from_config(ctx.config), sink=ctx.sink,
                              width=ctx.config.output.width or 0)
    if want_stream:
        events = list(renderer.stream(agent.stream(prompt, **options)))
        result = None
        for event in reversed(events):
            if event.kind == EventKind.RUN_FINISHED:
                result = event.data.get("result")
                break
        if result is None:
            ctx.error("the run produced no result")
            return 1
    else:
        spinner = renderer.spinner("thinking") if _wants_spinner(ctx, args) else None
        try:
            if spinner is not None:
                spinner.start()
            result = agent.run(prompt, **options)
        except KeyboardInterrupt:
            ctx.error("interrupted")
            return 130
        finally:
            if spinner is not None:
                spinner.stop()
    if getattr(args, "save", ""):
        target = Path(args.save).expanduser()
        write_json(target, result.to_dict())
        ctx.notice(f"wrote {target}", level="muted")
    if ctx.config.output.format != "text":
        ctx.emit(result, fmt=ctx.config.output.format)
    elif getattr(args, "quiet_output", False):
        ctx.write(result.output + ("\n" if result.output else ""))
    else:
        renderer.print_result(result)
    if getattr(args, "json_output", False):
        ctx.emit(result, fmt="json")
    return 0 if result.succeeded else 1


def _wants_spinner(ctx: Any, args: argparse.Namespace) -> bool:
    """Show a spinner only for interactive, non-streaming, non-json runs."""
    if getattr(args, "no_spinner", False) or ctx.quiet:
        return False
    if ctx.config.output.format != "text":
        return False
    if getattr(args, "stream", False):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001 - assume non-interactive
        return False


def configure_run(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the agent-facing commands."""
    parser.add_argument("prompt", nargs="?", default="", help="what to ask the agent to do")
    parser.add_argument("--prompt-file", default="", help="read the prompt from a file")
    parser.add_argument("--file", action="append", default=[], help="attach a file as context")
    parser.add_argument("--stdin", action="store_true", help="append piped stdin to the prompt")
    parser.add_argument("--preset", default="", help="start from an agent preset")
    parser.add_argument("--persona", default="", help="override the persona")
    parser.add_argument("--loop-mode", dest="loop_mode", default="",
                        choices=["", "react", "plan", "reflect", "single"],
                        help="reasoning loop to use")
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=0,
                        help="ceiling on reasoning steps")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=0,
                        help="ceiling on generated tokens")
    parser.add_argument("--tool", dest="tools", action="append", default=None,
                        help="restrict the run to these tools (repeatable)")
    parser.add_argument("--session", default="", help="session id to record into")
    parser.add_argument("--system", default="", help="replace the system prompt")
    parser.add_argument("--context", dest="context_block", default="", help="extra context block")
    parser.add_argument("--task", dest="task_block", default="", help="explicit task block")
    parser.add_argument("--stream", action="store_true", help="stream tokens as they arrive")
    parser.add_argument("--no-spinner", dest="no_spinner", action="store_true",
                        help="do not animate a spinner")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="plan tool calls without executing them")
    parser.add_argument("--structured", action="store_true", help="ask for JSON and parse it")
    parser.add_argument("--include-files", dest="include_files", action="store_true",
                        help="attach workspace files that match the prompt")
    parser.add_argument("--recall", dest="memory_recall", action="store_true",
                        help="recall memories into the prompt")
    parser.add_argument("--remember", action="store_true", help="store the outcome as a memory")
    parser.add_argument("--no-skills", dest="no_skills", action="store_true",
                        help="do not inject matching skills")
    parser.add_argument("--save", default="", help="write the run result as JSON to this path")


def configure_chat(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent chat``."""
    parser.add_argument("prompt", nargs="?", default="", help="first message (optional)")
    parser.add_argument("--preset", default="", help="start from an agent preset")
    parser.add_argument("--session", default="", help="resume or create this session id")
    parser.add_argument("--no-stream", dest="no_stream", action="store_true",
                        help="disable token streaming")
    parser.add_argument("--interactive", action="store_true",
                        help="keep the session open after the first message")


def configure_plan(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent plan``."""
    parser.add_argument("prompt", nargs="?", default="", help="the goal to plan")
    parser.add_argument("--prompt-file", default="", help="read the goal from a file")
    parser.add_argument("--stdin", action="store_true", help="append piped stdin")
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=6,
                        help="ceiling on plan steps")


def run_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent run`` - execute one task and render the full result."""
    return _agent_command(ctx, args)


def ask_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent ask`` - print only the answer, for pipes and scripts."""
    args.quiet_output = True
    args.stream = getattr(args, "stream", False)
    code = _agent_command(ctx, args)
    return code


def chat_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent chat`` - an interactive session."""
    from ..repl import Repl

    repl = Repl(ctx, session_id=getattr(args, "session", "") or "",
                preset=getattr(args, "preset", "") or "",
                stream=not getattr(args, "no_stream", False))
    if args.prompt:
        repl.ask(args.prompt)
        if not getattr(args, "interactive", False):
            return 0
    return repl.run()


def plan_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent plan`` - produce a plan without executing it."""
    prompt = _read_prompt(args)
    if not prompt:
        ctx.error("no goal given")
        return 2
    agent = ctx.build_agent()
    plan = agent.plan(prompt, max_steps=int(getattr(args, "max_steps", 6) or 6))
    if ctx.config.output.format != "text":
        ctx.emit(plan, fmt=ctx.config.output.format)
        return 0
    ctx.print(f"plan for: {ellipsize(prompt, 100)}")
    ctx.print("")
    rows = []
    for index, step in enumerate(plan.steps, start=1):
        rows.append((str(index), step.description or step.goal, step.tool or "-",
                     step.status or "pending"))
    ctx.table(rows, ["#", "step", "tool", "status"])
    if getattr(plan, "notes", ""):
        ctx.print("")
        ctx.print(str(plan.notes))
    return 0


def reflect_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent reflect`` - run a task, then critique the answer."""
    prompt = _read_prompt(args)
    if not prompt:
        ctx.error("no prompt given")
        return 2
    agent = ctx.build_agent()
    result = agent.run(prompt, **{**_run_options(args), "loop_mode": "reflect"})
    renderer = ResultRenderer(RenderOptions.from_config(ctx.config), sink=ctx.sink)
    renderer.print_result(result)
    critique = agent.critique(prompt, result.output)
    if ctx.config.output.format != "text":
        ctx.emit({"result": result.to_dict(), "critique": critique},
                 fmt=ctx.config.output.format)
        return 0
    ctx.print("")
    ctx.notice("self-critique")
    ctx.print(json.dumps(critique, indent=2, default=str) if isinstance(critique, (dict, list))
              else str(critique))
    return 0


COMMANDS: tuple[Command, ...] = (
    Command(
        name="run",
        help="Run the agent on a task",
        run=run_command,
        group="agent",
        needs_backend=True,
        configure=configure_run,
        description=(
            "Execute one task with the full agent loop: tools, guardrails, memory, "
            "skills and session recording. Use --stream to watch tokens arrive and "
            "--format json to hand the result to another program."
        ),
        examples=('magent run "summarise the failures in ci.log"',
                  'magent run "refactor parse_args" --preset coder --stream',
                  'git diff | magent run "review this change" --stdin'),
    ),
    Command(
        name="ask",
        help="Ask one question and print only the answer",
        run=ask_command,
        group="agent",
        needs_backend=True,
        aliases=("a",),
        configure=configure_run,
        description="Like ``run`` but the answer is printed alone, so it composes in a pipe.",
        examples=('magent ask "what does magent.config do?"',
                  'magent ask "list the tools" --preset planner'),
    ),
    Command(
        name="chat",
        help="Start an interactive session",
        run=chat_command,
        group="agent",
        needs_backend=True,
        configure=configure_chat,
        description=(
            "An interactive prompt with slash commands (/help, /tools, /memory, /save, "
            "/quit). History is written to a session, so /save and --session let you "
            "come back to it."
        ),
        examples=("magent chat", "magent chat --session refactor-2026"),
    ),
    Command(
        name="plan",
        help="Produce a plan without executing it",
        run=plan_command,
        group="agent",
        needs_backend=True,
        configure=configure_plan,
        examples=('magent plan "add pagination to the sessions API"',),
    ),
    Command(
        name="reflect",
        help="Run a task, then critique the answer",
        run=reflect_command,
        group="agent",
        needs_backend=True,
        configure=configure_run,
        examples=('magent reflect "explain the guardrail pipeline"',),
    ),
)


__all__ = [
    "COMMANDS",
    "configure_chat",
    "configure_plan",
    "configure_run",
    "ask_command",
    "chat_command",
    "plan_command",
    "reflect_command",
    "run_command",
]
