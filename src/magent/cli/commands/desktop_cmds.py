"""The ``magent desktop`` command: panel, queue, logos, and startup control."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...desktop import DesktopAppearanceStore, DesktopTaskQueue
from ...desktop.autostart import autostart_status, install_autostart, remove_autostart
from ...desktop.branding import LOGO_THEMES, logo_ids, logo_theme
from ...desktop.driver import build_desktop_controller
from ..commands import Command


def _configure_desktop(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by desktop panel and queue actions."""
    parser.add_argument(
        "action",
        nargs="?",
        default="open",
        choices=["open", "status", "queue", "logos", "autostart"],
        help="open the panel or manage one desktop subsystem",
    )
    parser.add_argument(
        "operation",
        nargs="?",
        default="",
        help="queue: add/list/show/cancel/retry/delete/clear/pause/resume; logos: theme id",
    )
    parser.add_argument("text", nargs="*", help="task prompt or task id")
    parser.add_argument("--prompt", default="", help="task prompt (alternative to positional text)")
    parser.add_argument("--title", default="", help="queue task title")
    parser.add_argument("--priority", type=int, default=50, help="queue priority, 0..100")
    parser.add_argument("--max-attempts", type=int, default=1, help="automatic attempts, 1..20")
    parser.add_argument("--status", default="", help="filter queue list by status")
    parser.add_argument("--limit", type=int, default=50, help="maximum tasks to list")
    parser.add_argument("--background", action="store_true", help="detach the floating panel")
    parser.add_argument("--logo", choices=logo_ids(), default="", help="initial logo theme")
    parser.add_argument(
        "--no-topmost", action="store_true", help="do not force the panel above windows"
    )


def desktop_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent desktop`` - operate the full desktop automation surface."""
    action = getattr(args, "action", "open") or "open"
    if action == "status":
        return _desktop_status(ctx)
    if action == "queue":
        return _desktop_queue(ctx, args)
    if action == "logos":
        return _desktop_logos(ctx, args)
    if action == "autostart":
        return _desktop_autostart(ctx, args)
    return _desktop_open(ctx, args)


def _desktop_status(ctx: Any) -> int:
    from ...desktop.app import tkinter_status

    controller = build_desktop_controller(ctx.config)
    selected_logo = _selected_logo(ctx.config)
    with DesktopTaskQueue(ctx.config.desktop.queue_path, recover=False) as queue:
        payload: dict[str, Any] = {
            "desktop": controller.describe(),
            "ui": tkinter_status(),
            "queue": queue.stats(),
            "autostart": autostart_status(),
            "logo": logo_theme(selected_logo).to_dict(),
            "logo_options": [theme.to_dict() for theme in LOGO_THEMES],
        }
    if ctx.config.output.format != "text":
        ctx.emit(payload, fmt=ctx.config.output.format)
        return 0
    capabilities: dict[str, Any] = payload["desktop"]["capabilities"]
    counts: dict[str, int] = payload["queue"]["counts"]
    ctx.key_values(
        [
            ("driver", capabilities["driver"]),
            ("session", capabilities["session"]),
            ("display", "ready" if capabilities["display_available"] else "unavailable"),
            ("mouse / keyboard", f"{capabilities['mouse']} / {capabilities['keyboard']}"),
            ("screenshots", capabilities["screenshots"]),
            ("windows", capabilities["window_control"]),
            (
                "clipboard",
                f"read={capabilities['clipboard_read']} write={capabilities['clipboard_write']}",
            ),
            ("floating UI", "ready" if payload["ui"]["available"] else payload["ui"]["error"]),
            (
                "queue",
                f"{counts['queued']} queued, {counts['paused']} paused, "
                f"{counts['running']} running, {counts['failed']} failed",
            ),
            ("logo", payload["logo"]["name"]),
            ("starts at login", payload["autostart"]["installed"]),
        ],
        title="MAgent desktop",
    )
    for note in capabilities.get("notes", []):
        ctx.notice(note, level="warning")
    return 0


def _desktop_queue(ctx: Any, args: argparse.Namespace) -> int:
    operation = (args.operation or "list").lower()
    with DesktopTaskQueue(ctx.config.desktop.queue_path, recover=False) as queue:
        if operation == "add":
            prompt = (args.prompt or " ".join(args.text)).strip()
            if not prompt:
                ctx.error("queue add needs a prompt")
                return 2
            created = queue.enqueue(
                prompt,
                title=args.title,
                priority=args.priority,
                max_attempts=args.max_attempts,
                session_id="desktop-cli",
                metadata={"source": "cli"},
            )
            return _emit_task(ctx, created, notice="queued")
        if operation == "show":
            shown = queue.get(_subject(args))
            if shown is None:
                ctx.error("unknown desktop task")
                return 1
            ctx.emit(
                shown.to_dict(),
                fmt=ctx.config.output.format if ctx.config.output.format != "text" else "yaml",
            )
            return 0
        if operation == "cancel":
            cancelled = queue.cancel(_subject(args))
            return _emit_optional_task(ctx, cancelled, "cancelled")
        if operation == "retry":
            retried = queue.retry(_subject(args), reset_attempts=True)
            return _emit_optional_task(ctx, retried, "queued for retry")
        if operation == "delete":
            removed = queue.delete(_subject(args))
            if not removed:
                ctx.error("task not found or currently running")
                return 1
            ctx.notice("desktop task deleted", level="success")
            return 0
        if operation == "clear":
            statuses = [args.status] if args.status else None
            count = queue.clear(statuses=statuses)
            ctx.notice(f"removed {count} desktop task(s)", level="success")
            return 0
        if operation in {"pause", "resume"}:
            task_id = _subject(args)
            if task_id:
                transitioned = (
                    queue.pause_task(task_id)
                    if operation == "pause"
                    else queue.resume_task(task_id)
                )
                return _emit_optional_task(ctx, transitioned, f"{operation}d")
            queue.set_paused(operation == "pause")
            ctx.notice(f"desktop queue {operation}d", level="success")
            return 0
        if operation not in {"list", "ls"}:
            ctx.error(f"unknown queue operation {operation!r}")
            return 2
        tasks = queue.list(status=args.status, limit=args.limit)
        if ctx.config.output.format != "text":
            ctx.emit(
                {"tasks": [task.to_dict() for task in tasks], "stats": queue.stats()},
                fmt=ctx.config.output.format,
            )
            return 0
        if not tasks:
            ctx.notice("desktop queue is empty", level="muted")
            return 0
        ctx.table(
            [
                (
                    task.id,
                    task.status.value,
                    task.priority,
                    f"{task.attempts}/{task.max_attempts}",
                    task.title,
                )
                for task in tasks
            ],
            ["id", "status", "priority", "tries", "task"],
            title=f"desktop queue · {len(tasks)} task(s)",
        )
        return 0


def _desktop_logos(ctx: Any, args: argparse.Namespace) -> int:
    selected = (args.logo or args.operation or "").lower()
    if selected:
        if selected not in logo_ids():
            ctx.error(f"unknown logo {selected!r}; choose {', '.join(logo_ids())}")
            return 2
        store = DesktopAppearanceStore(ctx.config.desktop.state_path)
        state = store.load()
        state.logo_theme = selected
        store.save(state)
        ctx.config.desktop.logo_theme = selected
        ctx.notice(f"selected {logo_theme(selected).name}", level="success")
        return 0
    current = _selected_logo(ctx.config)
    rows = [
        (
            index,
            theme.id,
            theme.name,
            theme.tagline,
            "selected" if theme.id == current else "",
        )
        for index, theme in enumerate(LOGO_THEMES, start=1)
    ]
    if ctx.config.output.format != "text":
        ctx.emit([theme.to_dict() for theme in LOGO_THEMES], fmt=ctx.config.output.format)
    else:
        ctx.table(rows, ["#", "id", "name", "tagline", "state"], title="8 MAgent logo pairs")
        ctx.notice("select with `magent desktop logos <id>`", level="muted")
    return 0


def _desktop_autostart(ctx: Any, args: argparse.Namespace) -> int:
    operation = (args.operation or "status").lower()
    if operation == "install":
        path = install_autostart()
        ctx.notice(f"installed startup entry at {path}", level="success")
        return 0
    if operation == "remove":
        removed = remove_autostart()
        ctx.notice("removed startup entry" if removed else "startup entry was not installed")
        return 0
    if operation != "status":
        ctx.error("autostart operation must be status, install, or remove")
        return 2
    payload = autostart_status()
    if ctx.config.output.format != "text":
        ctx.emit(payload, fmt=ctx.config.output.format)
    else:
        ctx.key_values(list(payload.items()), title="desktop autostart")
    return 0


def _desktop_open(ctx: Any, args: argparse.Namespace) -> int:
    if args.logo:
        ctx.config.desktop.logo_theme = args.logo
        store = DesktopAppearanceStore(ctx.config.desktop.state_path)
        state = store.load()
        state.logo_theme = args.logo
        store.save(state)
    if args.no_topmost:
        ctx.config.desktop.always_on_top = False
    if args.background and not os.environ.get("MAGENT_DESKTOP_CHILD"):
        command = [sys.executable, "-m", "magent", "desktop", "open"]
        for name, flag in (
            ("config", "--config"),
            ("profile", "--profile"),
            ("workspace", "--workspace"),
        ):
            value = getattr(args, name, "") or ""
            if value:
                command.extend([flag, str(value)])
        if args.logo:
            command.extend(["--logo", args.logo])
        if args.no_topmost:
            command.append("--no-topmost")
        env = dict(os.environ)
        env["MAGENT_DESKTOP_CHILD"] = "1"
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x00000008 | 0x00000200
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        ctx.notice(f"MAgent desktop started in background (pid {process.pid})", level="success")
        return 0
    from ...desktop.app import launch_desktop_app

    def build_agent(approval: Any) -> Any:
        return ctx.build_agent(approval=approval, name="MAgent[desktop]")

    try:
        return launch_desktop_app(ctx.config, agent_builder=build_agent)
    except RuntimeError as exc:
        ctx.error(str(exc))
        ctx.notice("run `magent desktop status` for desktop capability details", level="muted")
        return 1


def _selected_logo(config: Any) -> str:
    path = Path(config.desktop.state_path).expanduser()
    if path.is_file():
        return DesktopAppearanceStore(path).load().logo_theme
    return str(config.desktop.logo_theme)


def _subject(args: argparse.Namespace) -> str:
    return str((args.text[0] if args.text else args.prompt) or "").strip()


def _emit_task(ctx: Any, task: Any, *, notice: str) -> int:
    if ctx.config.output.format != "text":
        ctx.emit(task.to_dict(), fmt=ctx.config.output.format)
    else:
        ctx.notice(f"{notice}: {task.id} · {task.title}", level="success")
    return 0


def _emit_optional_task(ctx: Any, task: Any, notice: str) -> int:
    if task is None:
        ctx.error("unknown desktop task")
        return 1
    return _emit_task(ctx, task, notice=notice)


COMMANDS: tuple[Command, ...] = (
    Command(
        name="desktop",
        help="Automate the desktop and open the floating control",
        run=desktop_command,
        group="desktop",
        configure=_configure_desktop,
        description=(
            "Open MAgent's always-on-top logo and task/queue/chat panel, inspect desktop "
            "capabilities, manage durable work, choose one of eight logo pairs, or install "
            "the per-user startup entry."
        ),
        examples=(
            "magent desktop",
            "magent desktop open --background",
            "magent desktop status --format json",
            'magent desktop queue add "Prepare the weekly report" --priority 80',
            "magent desktop queue list",
            "magent desktop logos orbit",
            "magent desktop autostart install",
        ),
    ),
)


__all__ = ["COMMANDS", "desktop_command"]
