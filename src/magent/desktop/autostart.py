"""Install and remove per-user startup entries for the floating MAgent control."""

from __future__ import annotations

import html
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

from ..utils.files import atomic_write_text

APP_ID = "ai.arena.magent.desktop"
"""Stable id used by launchd and desktop environments."""


def autostart_path(*, system: str = "", home: str | Path | None = None) -> Path:
    """Return the platform's per-user autostart file path."""
    kind = (system or platform.system()).lower()
    root = Path(home).expanduser() if home is not None else Path.home()
    if kind == "windows":
        appdata = Path(os.environ.get("APPDATA", root / "AppData" / "Roaming"))
        return (
            appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "MAgent.cmd"
        )
    if kind == "darwin":
        return root / "Library" / "LaunchAgents" / f"{APP_ID}.plist"
    return root / ".config" / "autostart" / "magent.desktop"


def autostart_status(*, system: str = "", home: str | Path | None = None) -> dict[str, Any]:
    """Describe the current user's startup entry."""
    path = autostart_path(system=system, home=home)
    return {
        "platform": system or platform.system(),
        "path": str(path),
        "installed": path.is_file(),
    }


def install_autostart(
    *,
    executable: str | Path | None = None,
    system: str = "",
    home: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Install an idempotent per-user startup entry."""
    kind = (system or platform.system()).lower()
    python = str(Path(executable or sys.executable).expanduser())
    args = [python, "-m", "magent", "desktop", "open", *(extra_args or [])]
    target = autostart_path(system=kind, home=home)
    if kind == "windows":
        command = " ".join(_cmd_quote(item) for item in args)
        content = '@echo off\r\nstart "MAgent" /b ' + command + "\r\n"
    elif kind == "darwin":
        array = "".join(f"<string>{html.escape(item)}</string>" for item in args)
        log_dir = (
            Path(home).expanduser() / "Library" / "Logs"
            if home is not None
            else Path.home() / "Library" / "Logs"
        )
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            f"<key>Label</key><string>{APP_ID}</string>\n"
            f"<key>ProgramArguments</key><array>{array}</array>\n"
            "<key>RunAtLoad</key><true/>\n"
            "<key>KeepAlive</key><false/>\n"
            f"<key>StandardOutPath</key><string>{html.escape(str(log_dir / 'magent-desktop.log'))}</string>\n"
            f"<key>StandardErrorPath</key><string>{html.escape(str(log_dir / 'magent-desktop-error.log'))}</string>\n"
            "</dict></plist>\n"
        )
    else:
        command = shlex.join(args)
        content = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Version=1.0\n"
            "Name=MAgent\n"
            "Comment=Always-on-top local desktop agent\n"
            f"Exec={command}\n"
            "Terminal=false\n"
            "StartupNotify=false\n"
            "X-GNOME-Autostart-enabled=true\n"
        )
    atomic_write_text(target, content)
    try:
        target.chmod(0o755 if kind in {"windows", "linux"} else 0o644)
    except OSError:
        pass
    return target


def remove_autostart(*, system: str = "", home: str | Path | None = None) -> bool:
    """Remove this user's startup entry."""
    target = autostart_path(system=system, home=home)
    try:
        target.unlink()
    except FileNotFoundError:
        return False
    return True


def _cmd_quote(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


__all__ = [
    "APP_ID",
    "autostart_path",
    "autostart_status",
    "install_autostart",
    "remove_autostart",
]
