"""The ``run`` and ``chat`` commands: how a task is actually executed.

``run`` is one shot - a task in, an answer out, an exit code that a script can
branch on. ``chat`` is a REPL over a single persistent agent, so history,
memory, and the session transcript carry across turns.

Both stream events rather than waiting for the end, because a run that calls
tools can take minutes and silence reads as a hang. The event kinds decide what
is shown: text deltas go straight to the terminal, tool calls and results get
one line each, and timings appear only at ``--verbose``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...errors import MAgentError
from ...output import Printer, render_result
from ...types import StopReason
from ..args import CliContext, EXIT_ERROR, EXIT_OK, read_prompt_text

CHAT_COMMANDS: dict[str, str] = {
    "/help": "show this help",
    "/tools": "list the tools available to the agent",
    "/clear": "forget the conversation history",
    "/history": "show the conversation so far",
    "/save": "save the session transcript to disk",
    "/session": "show the current session id and statistics",
    "/model": "show or set the model (usage: /model [id])",
    "/mode": "show or set the loop mode (react, plan, reflect, single)",
    "/persona": "show or set the persona (usage: /persona [name])",
    "/usage": "show token and cost totals for this chat",
    "/verbose": "toggle verbose step output",
    "/quit": "exit the chat",
}
"""Slash commands understood inside ``magent chat``."""

STOP_EXIT_CODES: dict[str, int] = {
    StopReason.COMPLETED.value: EXIT_OK,
    StopReason.MAX_ITERATIONS.value: 3,
    StopReason.BUDGET.value: 4,
    StopReason.CANCELLED.value: 130,
    StopReason.ERROR.value: EXIT_ERROR,
    StopReason.GUARDRAIL.value: 5,
    StopReason.NEEDS_INPUT.value: 6,
    StopReason.TOOL_LIMIT.value: 7,
}
"""Exit code per stop reason, so a script can tell *why* a run stopped."""


def register(subparsers: Any) -> None:
    """Add the ``run`` and ``chat`` subcommands."""
    run = subparsers.add_parser("run", help="run one task and print the answer",
                                description="Execute a single task with the agent.")
    run.add_argument("prompt", nargs="*", help="the task to perform")
    run.add_argument("--file", metavar="PATH", default="",
                     help="read the task from a file (-f is --format)")
    run.add_argument("--stdin", action="store_true", help="read the task from stdin")
    run.add_argument("--tools", metavar="NAME", action="append", default=[],
                     help="restrict to these tools (repeatable)")
    run.add_argument("--disable-tools", metavar="NAME", action="append", default=[],
                     help="exclude these tools (repeatable)")
    run.add_argument("--max-steps", type=int, default=0, metavar="N",
                     help="maximum agent loop iterations")
    run.add_argument("--mode", default="", choices=["", "react", "plan", "reflect", "single"],
                     help="agent loop mode")
    run.add_argument("--persona", default="", metavar="NAME", help="persona to adopt")
    run.add_argument("--system", default="", metavar="TEXT", help="extra system instructions")
    run.add_argument("--structured", default="", metavar="JSON",
                     help="JSON schema the answer must satisfy")
    run.add_argument("--rag", metavar="PATH", action="append", default=[],
                     help="ingest these paths and cite them in the answer")
    run.add_argument("--remember", action="store_true", help="store the outcome in memory")
    run.add_argument("--no-memory", action="store_true", help="do not read or write memory")
    run.add_argument("--session", default="", metavar="ID", help="resume this session")
    run.add_argument("--new-session", action="store_true", help="start and record a new session")
    run.add_argument("--no-stream", action="store_true", help="print only the final answer")
    run.set_defaults(handler=cmd_run)

    chat = subparsers.add_parser("chat", help="start an interactive conversation",
                                 description="Interactive REPL over one persistent agent.")
    chat.add_argument("prompt", nargs="*", help="optional first message")
    chat.add_argument("--tools", metavar="NAME", action="append", default=[],
                      help="restrict to these tools (repeatable)")
    chat.add_argument("--max-steps", type=int, default=0, metavar="N",
                      help="maximum iterations per turn")
    chat.add_argument("--mode", default="", choices=["", "react", "plan", "reflect", "single"],
                      help="agent loop mode")
    chat.add_argument("--persona", default="", metavar="NAME", help="persona to adopt")
    chat.add_argument("--resume", action="store_true", help="resume the most recent session")
    chat.add_argument("--no-memory", action="store_true", help="do not read or write memory")
    chat.set_defaults(handler=cmd_chat)


def _run_options(args: argparse.Namespace, context: CliContext) -> dict[str, Any]:
    """Translate command line flags into :class:`RunOptions` keywords."""
    options: dict[str, Any] = {}
    if getattr(args, "tools", None):
        options["tools"] = list(args.tools)
    if getattr(args, "disable_tools", None):
        options["disabled_tools"] = list(args.disable_tools)
    if getattr(args, "max_steps", 0):
        options["max_iterations"] = int(args.max_steps)
    if getattr(args, "mode", ""):
        options["loop_mode"] = args.mode
    if getattr(args, "persona", ""):
        options["persona"] = args.persona
    if getattr(args, "system", ""):
        options["system"] = args.system
    if getattr(args, "structured", ""):
        try:
            options["structured"] = json.loads(args.structured)
        except json.JSONDecodeError as exc:
            raise MAgentError(f"--structured is not valid JSON: {exc}") from exc
    if getattr(args, "remember", False):
        options["remember_outcome"] = True
    if getattr(args, "no_memory", False):
        options["recall_memories"] = False
        options["remember_outcome"] = False
    return options


def _rag_block(context: CliContext, paths: list[str], query: str) -> str:
    """Ingest the requested paths and return cited context for the query."""
    if not paths:
        return ""
    from ...rag import Retriever

    retriever = Retriever(name="cli")
    for path in paths:
        from pathlib import Path

        target = Path(path).expanduser()
        if target.is_dir():
            retriever.ingest_directory(target)
        elif target.exists():
            retriever.ingest([target])
        else:
            retriever.ingest_text(path, source="inline")
    if not len(retriever):
        return ""
    context.printer.debug(f"rag: indexed {len(retriever)} chunks from {len(paths)} paths")
    return retriever.context(query, limit=6)


def cmd_run(args: argparse.Namespace, context: CliContext) -> int:
    """Execute one task and print the result."""
    task = read_prompt_text(args)
    if not task:
        context.printer.error("error: no task given; pass text, --file, or --stdin")
        return EXIT_ERROR

    options = _run_options(args, context)
    retrieved = _rag_block(context, list(getattr(args, "rag", []) or []), task)
    if retrieved:
        options["context_block"] = retrieved

    agent = context.make_agent()
    if getattr(args, "session", ""):
        options["session_id"] = args.session
    recorder = None
    if getattr(args, "new_session", False) or getattr(args, "session", ""):
        recorder = _session_recorder(context, args)

    stream = not getattr(args, "no_stream", False) and context.options.fmt == "text" \
        and not context.options.quiet
    if stream:
        result = _run_streaming(agent, task, context, recorder=recorder, **options)
    else:
        if recorder is not None:
            recorder.record_prompt(task)
        result = agent.run(task, **options)
        if recorder is not None:
            recorder.record_result(result)

    if context.options.fmt == "json":
        context.printer.write(render_result(result, options=context.options), min_level="quiet")
    elif context.options.fmt == "markdown":
        context.printer.write(render_result(result, options=context.options), min_level="quiet")
    else:
        if not stream:
            context.printer.write(result.output.rstrip(), min_level="quiet")
        if context.options.at_least("normal") and context.options.show_timing:
            context.printer.write(_stats_line(result, context), min_level="normal")
    if recorder is not None:
        recorder.finish()
    reason = str(getattr(result.stop_reason, "value", result.stop_reason))
    return STOP_EXIT_CODES.get(reason, EXIT_OK if result.output else EXIT_ERROR)


class _SessionRecorder:
    """Records a run into a session transcript, if sessions are enabled."""

    def __init__(self, context: CliContext, session: Any) -> None:
        """Capture the context and the open session."""
        self.context = context
        self.session = session

    def record_prompt(self, task: str) -> None:
        """Record the user's task."""
        from ...types import Message

        self.session.record(Message.user(task), run_id="")

    def record_result(self, result: Any) -> None:
        """Record every message of a finished run."""
        for message in result.messages:
            if message.role == "system":
                continue
            self.session.record(message, model=result.model, run_id=result.run_id)
        for step in result.steps:
            for tool_result in step.tool_results:
                self.session.record_tool(tool_result, step=step.index, run_id=result.run_id)

    def finish(self) -> None:
        """Save the transcript and report where it went."""
        self.session.transcript.close()
        path = self.session.save()
        self.context.printer.debug(f"session saved: {path}")


def _session_recorder(context: CliContext, args: argparse.Namespace) -> _SessionRecorder | None:
    """Open (or resume) a session for recording, or return ``None``."""
    if not context.config.session.enabled:
        return None
    from ...session import SessionStore

    store = SessionStore.from_config(context.config)
    wanted = getattr(args, "session", "")
    session = store.open(wanted) if wanted else store.create(model=context.config.model.model)
    return _SessionRecorder(context, session)


def _run_streaming(agent: Any, task: str, context: CliContext, *,
                   recorder: _SessionRecorder | None = None, **options: Any) -> Any:
    """Run the agent while printing events as they arrive."""
    printer = context.printer
    if recorder is not None:
        recorder.record_prompt(task)
    result: Any = None
    text_started = False
    streamed_text = False
    for event in agent.stream(task, **options):
        kind = str(getattr(event.kind, "value", event.kind))
        data = dict(event.data or {})
        if kind == "agent.text_delta":
            chunk = str(data.get("text") or data.get("delta") or "")
            if chunk:
                streamed_text = True
                if not text_started:
                    text_started = True
                sys.stdout.write(chunk)
                sys.stdout.flush()
        elif kind == "tool.call":
            if text_started:
                sys.stdout.write("\n")
                text_started = False
            printer.write(printer.themed(
                f"  -> {data.get('tool') or data.get('name')} "
                f"{_compact(data.get('arguments'))}", "tool"), min_level="normal")
        elif kind == "tool.result":
            ok = data.get("ok", True)
            name = data.get("tool") or data.get("name") or ""
            summary = _compact(data.get("content"), 140)
            theme = "success" if ok else "error"
            printer.write(printer.themed(f"  {'+' if ok else 'x'} {name} {summary}", theme),
                          min_level="normal")
        elif kind == "guard.blocked":
            printer.warn(f"blocked: {data.get('reason', '')}")
        elif kind == "llm.retry":
            printer.debug(f"retrying after {data.get('error', '')}")
        elif kind == "step.finished" and printer.at_least("verbose"):
            printer.debug(f"step {data.get('index', '?')} finished in "
                          f"{data.get('duration_ms', 0):.0f}ms")
        elif kind == "run.finished":
            result = data.get("result")
        elif kind == "run.failed":
            printer.error(f"run failed: {data.get('error', '')}")
    if text_started:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if result is None:
        result = agent.last_result if hasattr(agent, "last_result") else None
    if result is None:
        raise MAgentError("the agent produced no result")
    if not text_started and not streamed_text and getattr(result, "output", ""):
        # A backend that does not emit deltas (the offline echo backend, or any
        # non-streaming model) would otherwise leave the answer unprinted.
        printer.write(str(result.output).rstrip(), min_level="quiet")
    elif text_started:
        pass
    if recorder is not None:
        recorder.record_result(result)
    return result


def _stats_line(result: Any, context: CliContext) -> str:
    """Build the trailing statistics line for a run."""
    from ...utils.tables import pluralize
    from ...utils.timeutil import format_duration

    parts = [format_duration(result.duration_ms / 1000.0),
             pluralize(result.iterations, "step")]
    if result.tool_call_count:
        parts.append(pluralize(result.tool_call_count, "tool call"))
    if context.options.show_usage:
        usage = getattr(result, "usage", None)
        if usage is not None and getattr(usage, "total_tokens", 0):
            parts.append(f"{usage.total_tokens} tokens")
        cost = getattr(result, "cost", None)
        if cost is not None and getattr(cost, "total", 0.0):
            parts.append(f"${cost.total:.4f}")
    reason = str(getattr(result.stop_reason, "value", result.stop_reason))
    if reason != StopReason.COMPLETED.value:
        parts.append(f"stop: {reason}")
    return context.printer.themed(" · ".join(parts), "muted")


def _compact(value: Any, limit: int = 100) -> str:
    """Flatten a value onto one line no longer than ``limit``."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    flat = " ".join(value.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def cmd_chat(args: argparse.Namespace, context: CliContext) -> int:
    """Run an interactive conversation over one persistent agent."""
    printer = context.printer
    if context.options.fmt == "json":
        printer.error("error: chat needs a terminal; use 'magent run --format json' instead")
        return EXIT_ERROR

    agent = context.make_agent()
    session = None
    if context.config.session.enabled and (getattr(args, "resume", False) or True):
        from ...session import SessionStore

        store = SessionStore.from_config(context.config)
        session = store.latest() if getattr(args, "resume", False) else store.create(
            title="chat", model=context.config.model.model)
    options = _run_options(args, context)

    _print_banner(context, agent, session)
    if session is not None and getattr(args, "resume", False) and session.entries:
        replayed = session.last_messages(context.config.session.keep_last)
        if replayed:
            agent.history.extend(replayed)
            printer.write(printer.themed(
                f"resumed {session.id} with {len(replayed)} earlier messages", "muted"),
                min_level="normal")

    first = " ".join(getattr(args, "prompt", []) or []).strip()
    turns = 0
    while True:
        try:
            if first:
                user_input, first = first, ""
            else:
                user_input = _read_line(printer)
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        if user_input is None:
            break
        text = user_input.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _handle_chat_command(text, context, agent, session, options):
                break
            continue
        turns += 1
        try:
            result = agent.run(text, **options)
        except MAgentError as exc:
            printer.error(f"error: {exc}")
            continue
        except KeyboardInterrupt:
            printer.warn("interrupted; the agent will stop after this step")
            agent.cancel(reason="interrupted in chat")
            continue
        printer.write(result.output.rstrip(), min_level="quiet")
        if context.options.at_least("verbose"):
            printer.write(_stats_line(result, context), min_level="verbose")
        if session is not None:
            from ...types import Message

            session.record(Message.user(text))
            for message in result.messages:
                if message.role != "system":
                    session.record(message, model=result.model, run_id=result.run_id)
            session.flush()
    if session is not None:
        session.transcript.close()
        path = session.save()
        printer.write(printer.themed(f"saved {len(session)} turns to {path}", "muted"),
                      min_level="normal")
    printer.write(printer.themed(f"{turns} turns", "muted"), min_level="normal")
    return EXIT_OK


def _read_line(printer: Printer) -> str | None:
    """Read one line of input, returning ``None`` at end of input."""
    try:
        return input(printer.themed("you> ", "user"))
    except EOFError:
        return None


def _print_banner(context: CliContext, agent: Any, session: Any) -> None:
    """Print the chat greeting."""
    from ..._version import version_string

    printer = context.printer
    printer.write(printer.themed(version_string(), "brand"), min_level="normal")
    backend = getattr(agent.backend, "name", type(agent.backend).__name__)
    model = getattr(agent, "model", "") or context.config.model.model or backend
    printer.write(printer.themed(f"backend {backend} · model {model} · "
                                 f"{context.registry.count()} tools", "muted"), min_level="normal")
    if session is not None:
        printer.write(printer.themed(f"session {session.id}", "muted"), min_level="minimal")
    printer.write(printer.themed("type /help for commands, Ctrl-D to exit", "muted"),
                  min_level="normal")
    printer.blank()


def _handle_chat_command(text: str, context: CliContext, agent: Any, session: Any,
                         options: dict[str, Any]) -> bool:
    """Execute one slash command; return ``True`` when the chat should exit."""
    printer = context.printer
    parts = text.split(maxsplit=1)
    name = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if name in {"/quit", "/exit", "/q"}:
        return True
    if name == "/help":
        for command, description in CHAT_COMMANDS.items():
            printer.write(f"  {command:<12} {description}", min_level="quiet")
        return False
    if name == "/tools":
        names = sorted(context.registry.names())
        printer.write(printer.themed(f"{len(names)} tools:", "heading"), min_level="quiet")
        for index in range(0, len(names), 4):
            printer.write("  " + "  ".join(item.ljust(20) for item in names[index:index + 4]),
                          min_level="quiet")
        return False
    if name == "/clear":
        agent.reset()
        printer.write("history cleared", min_level="quiet")
        return False
    if name == "/history":
        for message in agent.history:
            role = str(getattr(message.role, "value", message.role))
            body = " ".join((message.content or "").split())[:120]
            printer.write(f"  {role}: {body}", min_level="quiet")
        if not agent.history:
            printer.write("  (empty)", min_level="quiet")
        return False
    if name == "/usage":
        tracker = getattr(agent, "tracker", None)
        payload = tracker.to_dict() if hasattr(tracker, "to_dict") else {}
        printer.write(json.dumps(payload, indent=2, default=str), min_level="quiet")
        return False
    if name == "/model":
        if argument:
            agent.model = argument
            printer.write(f"model set to {argument}", min_level="quiet")
        else:
            printer.write(f"model: {getattr(agent, 'model', '') or '(backend default)'}",
                          min_level="quiet")
        return False
    if name == "/mode":
        if argument:
            agent.config.loop_mode = argument
        printer.write(f"loop mode: {agent.config.loop_mode}", min_level="quiet")
        return False
    if name == "/persona":
        if argument:
            agent.config.persona = argument
        printer.write(f"persona: {agent.config.persona or '(default)'}", min_level="quiet")
        return False
    if name == "/verbose":
        context.options.verbosity = "normal" if context.options.verbosity == "verbose" else "verbose"
        printer.write(f"verbosity: {context.options.verbosity}", min_level="quiet")
        return False
    if name == "/save":
        if session is None:
            printer.warn("sessions are disabled")
            return False
        path = session.save()
        printer.write(f"saved {len(session)} turns to {path}", min_level="quiet")
        return False
    if name == "/session":
        if session is None:
            printer.warn("sessions are disabled")
            return False
        printer.write(json.dumps(session.stats(), indent=2, default=str), min_level="quiet")
        return False
    printer.warn(f"unknown command {name}; try /help")
    return False


__all__ = [
    "CHAT_COMMANDS",
    "STOP_EXIT_CODES",
    "cmd_chat",
    "cmd_run",
    "register",
]
