"""Cross-platform drivers and the guarded desktop controller.

MAgent prefers the optional ``pyautogui`` backend for consistent pointer,
keyboard, screenshot, and image-location behaviour.  With no Python extras it
falls back to native operating-system facilities: Win32 APIs on Windows,
AppleScript and command-line helpers on macOS, and xdotool/wmctrl/Wayland tools
on Linux.  Imports and capability probes are lazy, so headless servers can still
load the complete tool catalogue and report exactly what is unavailable.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import webbrowser
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..errors import ToolExecutionError
from .models import DesktopCapabilities, MonitorInfo, Point, Rect, WindowInfo


class DesktopUnavailableError(ToolExecutionError):
    """Raised when the current desktop session cannot perform an action."""

    code = "desktop_unavailable"


class DesktopDriver(ABC):
    """Operating-system-neutral interface used by tools and workflows."""

    name = "desktop"

    @abstractmethod
    def capabilities(self) -> DesktopCapabilities:
        """Probe supported desktop operations without changing the desktop."""

    def monitors(self) -> list[MonitorInfo]:
        width, height = self.screen_size()
        return [MonitorInfo(id="primary", bounds=Rect(0, 0, width, height), primary=True)]

    def screen_size(self) -> tuple[int, int]:
        raise DesktopUnavailableError(f"{self.name} cannot read screen size")

    def pointer_position(self) -> Point:
        raise DesktopUnavailableError(f"{self.name} cannot read pointer position")

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot move the pointer")

    def click(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot click")

    def drag(
        self,
        x: int,
        y: int,
        *,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot drag")

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot scroll")

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot type")

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot press keys")

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot send hotkeys")

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        raise DesktopUnavailableError(f"{self.name} cannot capture screenshots")

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        raise DesktopUnavailableError(f"{self.name} cannot inspect pixels")

    def locate(
        self,
        image: str | Path,
        *,
        confidence: float = 0.9,
        region: Rect | None = None,
    ) -> Rect | None:
        raise DesktopUnavailableError(f"{self.name} cannot locate images")

    def clipboard_read(self) -> str:
        raise DesktopUnavailableError(f"{self.name} cannot read the clipboard")

    def clipboard_write(self, text: str) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot write the clipboard")

    def windows(self) -> list[WindowInfo]:
        raise DesktopUnavailableError(f"{self.name} cannot list windows")

    def active_window(self) -> WindowInfo | None:
        windows = self.windows()
        return next((window for window in windows if window.active), None)

    def focus_window(self, window_id: str) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot focus windows")

    def window_action(self, window_id: str, action: str) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot manage windows")

    def move_resize_window(self, window_id: str, bounds: Rect) -> None:
        raise DesktopUnavailableError(f"{self.name} cannot move windows")

    def launch(self, app: str, args: list[str] | None = None) -> int:
        command = [app, *(args or [])]
        try:
            process = subprocess.Popen(command, start_new_session=True)
        except OSError as exc:
            raise DesktopUnavailableError(f"could not launch {app!r}: {exc}") from exc
        return int(process.pid)

    def open_target(self, target: str) -> bool:
        return bool(webbrowser.open(target))


class PyAutoGUIDriver(DesktopDriver):
    """Adapter around the optional PyAutoGUI/Pillow desktop stack."""

    name = "pyautogui"

    def __init__(self, module: Any = None, *, fail_safe: bool = True, pause: float = 0.05) -> None:
        if module is None:
            try:
                import pyautogui as module  # type: ignore[no-redef]
            except Exception as exc:
                raise DesktopUnavailableError(f"PyAutoGUI is unavailable: {exc}") from exc
        self.gui = module
        self.gui.FAILSAFE = bool(fail_safe)
        self.gui.PAUSE = max(0.0, float(pause))

    def capabilities(self) -> DesktopCapabilities:
        return DesktopCapabilities(
            driver=self.name,
            platform=platform.system(),
            session=_session_name(),
            display_available=True,
            mouse=True,
            keyboard=True,
            screenshots=True,
            image_location=True,
            monitors=True,
            dependencies={"pyautogui": True, "pillow": _module_available("PIL")},
        )

    def screen_size(self) -> tuple[int, int]:
        size = self.gui.size()
        return int(size.width), int(size.height)

    def pointer_position(self) -> Point:
        point = self.gui.position()
        return Point(int(point.x), int(point.y))

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> None:
        self.gui.moveTo(int(x), int(y), duration=max(0.0, float(duration)))

    def click(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        self.gui.click(x=x, y=y, button=button, clicks=max(1, clicks), interval=max(0.0, interval))

    def drag(self, x: int, y: int, *, duration: float = 0.5, button: str = "left") -> None:
        self.gui.dragTo(int(x), int(y), duration=max(0.0, duration), button=button)

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        if horizontal and hasattr(self.gui, "hscroll"):
            self.gui.hscroll(int(amount))
        else:
            self.gui.scroll(int(amount))

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        self.gui.write(str(text), interval=max(0.0, interval))

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self.gui.press(key, presses=max(1, int(presses)), interval=max(0.0, interval))

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        if not keys:
            raise ValueError("hotkey requires at least one key")
        self.gui.hotkey(*keys, interval=max(0.0, interval))

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        area = (region.x, region.y, region.width, region.height) if region else None
        image = self.gui.screenshot(region=area)
        image.save(target)
        return target

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        color = self.gui.pixel(int(x), int(y))
        return int(color[0]), int(color[1]), int(color[2])

    def locate(
        self,
        image: str | Path,
        *,
        confidence: float = 0.9,
        region: Rect | None = None,
    ) -> Rect | None:
        kwargs: dict[str, Any] = {"confidence": max(0.0, min(1.0, confidence))}
        if region:
            kwargs["region"] = (region.x, region.y, region.width, region.height)
        try:
            box = self.gui.locateOnScreen(str(image), **kwargs)
        except TypeError:
            kwargs.pop("confidence", None)
            box = self.gui.locateOnScreen(str(image), **kwargs)
        if box is None:
            return None
        return Rect(int(box.left), int(box.top), int(box.width), int(box.height))


class NativeDesktopDriver(DesktopDriver):
    """Best-effort native driver with no Python package dependencies."""

    name = "native"

    def __init__(self, *, system: str = "", environ: dict[str, str] | None = None) -> None:
        self.system = (system or platform.system()).lower()
        self.environ = dict(os.environ if environ is None else environ)
        self._which = {name: shutil.which(name) for name in self._dependency_names()}

    def _dependency_names(self) -> tuple[str, ...]:
        return (
            "xdotool",
            "wmctrl",
            "grim",
            "scrot",
            "gnome-screenshot",
            "import",
            "wl-copy",
            "wl-paste",
            "xclip",
            "xsel",
            "cliclick",
            "osascript",
            "screencapture",
            "powershell",
        )

    def capabilities(self) -> DesktopCapabilities:
        windows = self.system == "windows"
        mac = self.system == "darwin"
        linux = self.system == "linux"
        display = (
            windows
            or mac
            or bool(self.environ.get("DISPLAY") or self.environ.get("WAYLAND_DISPLAY"))
        )
        xdotool = bool(self._which.get("xdotool"))
        wmctrl = bool(self._which.get("wmctrl"))
        cliclick = bool(self._which.get("cliclick"))
        screenshot = (
            windows
            or (mac and bool(self._which.get("screencapture")))
            or bool(self._screenshot_command())
        )
        clip_read, clip_write = self._clipboard_capabilities()
        notes: list[str] = []
        if not display:
            notes.append("no graphical display session detected")
        if linux and self.environ.get("WAYLAND_DISPLAY") and not xdotool:
            notes.append("Wayland input automation needs PyAutoGUI or a compositor-specific helper")
        if mac:
            notes.append("macOS may request Accessibility and Screen Recording permissions")
        platform_name = {"windows": "Windows", "darwin": "Darwin", "linux": "Linux"}.get(
            self.system, self.system or platform.system()
        )
        session = self.system if windows or mac else _session_name(self.environ)
        return DesktopCapabilities(
            driver=self.name,
            platform=platform_name,
            session=session,
            display_available=display,
            mouse=display and (windows or xdotool or cliclick),
            keyboard=display and (windows or mac or xdotool),
            screenshots=display and screenshot,
            image_location=False,
            clipboard_read=clip_read,
            clipboard_write=clip_write,
            windows=display and (windows or mac or wmctrl or xdotool),
            window_control=display and (windows or mac or wmctrl or xdotool),
            launch=True,
            monitors=display and (windows or mac or xdotool),
            dependencies={name: bool(path) for name, path in self._which.items()},
            notes=notes,
        )

    # -- command helpers ---------------------------------------------------

    def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: float = 15.0,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=check,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            label = Path(args[0]).name if args else "desktop helper"
            raise DesktopUnavailableError(f"{label} failed: {exc}") from exc

    def _require(self, name: str) -> str:
        path = self._which.get(name) or shutil.which(name)
        if not path:
            raise DesktopUnavailableError(f"desktop action needs {name!r} on PATH")
        return path

    # -- display and pointer ----------------------------------------------

    def screen_size(self) -> tuple[int, int]:
        if self.system == "windows":
            import ctypes

            return int(ctypes.windll.user32.GetSystemMetrics(0)), int(  # type: ignore[attr-defined]
                ctypes.windll.user32.GetSystemMetrics(1)  # type: ignore[attr-defined]
            )
        if self.system == "linux":
            result = self._run([self._require("xdotool"), "getdisplaygeometry"])
            width, height = result.stdout.split()[:2]
            return int(width), int(height)
        if self.system == "darwin":
            script = 'tell application "Finder" to get bounds of window of desktop'
            result = self._run([self._require("osascript"), "-e", script])
            values = [int(item.strip()) for item in result.stdout.split(",")]
            return values[2] - values[0], values[3] - values[1]
        return super().screen_size()

    def pointer_position(self) -> Point:
        if self.system == "windows":
            import ctypes
            from ctypes import wintypes

            point = wintypes.POINT()
            if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):  # type: ignore[attr-defined]
                raise DesktopUnavailableError("GetCursorPos failed")
            return Point(int(point.x), int(point.y))
        if self.system == "linux":
            result = self._run([self._require("xdotool"), "getmouselocation", "--shell"])
            values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            return Point(int(values["X"]), int(values["Y"]))
        if self.system == "darwin" and self._which.get("cliclick"):
            result = self._run([self._require("cliclick"), "p"])
            match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", result.stdout)
            if match:
                return Point(int(match.group(1)), int(match.group(2)))
        return super().pointer_position()

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> None:
        if self.system == "windows":
            import ctypes

            if not ctypes.windll.user32.SetCursorPos(int(x), int(y)):  # type: ignore[attr-defined]
                raise DesktopUnavailableError("SetCursorPos failed")
            return
        if self.system == "linux":
            args = [self._require("xdotool"), "mousemove", "--sync", str(int(x)), str(int(y))]
            self._run(args)
            return
        if self.system == "darwin" and self._which.get("cliclick"):
            self._run([self._require("cliclick"), f"m:{int(x)},{int(y)}"])
            return
        super().move_pointer(x, y, duration=duration)

    def click(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        if x is not None and y is not None:
            self.move_pointer(x, y)
        normalized = _button(button)
        if self.system == "windows":
            import ctypes

            flags = {
                "left": (0x0002, 0x0004),
                "right": (0x0008, 0x0010),
                "middle": (0x0020, 0x0040),
            }[normalized]
            for index in range(max(1, int(clicks))):
                ctypes.windll.user32.mouse_event(flags[0], 0, 0, 0, 0)  # type: ignore[attr-defined]
                ctypes.windll.user32.mouse_event(flags[1], 0, 0, 0, 0)  # type: ignore[attr-defined]
                if index + 1 < clicks:
                    time.sleep(max(0.0, interval))
            return
        if self.system == "linux":
            number = {"left": "1", "middle": "2", "right": "3"}[normalized]
            self._run(
                [
                    self._require("xdotool"),
                    "click",
                    "--repeat",
                    str(max(1, int(clicks))),
                    "--delay",
                    str(max(0, int(interval * 1000))),
                    number,
                ]
            )
            return
        if self.system == "darwin":
            if self._which.get("cliclick"):
                point = self.pointer_position()
                command = "rc" if normalized == "right" else "c"
                sequence = [f"{command}:{point.x},{point.y}"] * max(1, int(clicks))
                self._run([self._require("cliclick"), *sequence])
                return
            point = self.pointer_position()
            script = f'tell application "System Events" to click at {{{point.x}, {point.y}}}'
            self._run([self._require("osascript"), "-e", script])
            return
        super().click(x=x, y=y, button=button, clicks=clicks, interval=interval)

    def drag(self, x: int, y: int, *, duration: float = 0.5, button: str = "left") -> None:
        normalized = _button(button)
        if self.system == "linux":
            number = {"left": "1", "middle": "2", "right": "3"}[normalized]
            self._run(
                [
                    self._require("xdotool"),
                    "mousedown",
                    number,
                    "mousemove",
                    "--sync",
                    str(int(x)),
                    str(int(y)),
                    "mouseup",
                    number,
                ]
            )
            return
        if self.system == "darwin" and self._which.get("cliclick"):
            start = self.pointer_position()
            self._run(
                [
                    self._require("cliclick"),
                    f"dd:{start.x},{start.y}",
                    f"dm:{int(x)},{int(y)}",
                    f"du:{int(x)},{int(y)}",
                ]
            )
            return
        if self.system == "windows":
            start = self.pointer_position()
            import ctypes

            flags = {
                "left": (0x0002, 0x0004),
                "right": (0x0008, 0x0010),
                "middle": (0x0020, 0x0040),
            }[normalized]
            ctypes.windll.user32.mouse_event(flags[0], 0, 0, 0, 0)  # type: ignore[attr-defined]
            steps = max(1, min(120, int(max(0.0, duration) * 60)))
            for index in range(1, steps + 1):
                ratio = index / steps
                self.move_pointer(
                    round(start.x + (int(x) - start.x) * ratio),
                    round(start.y + (int(y) - start.y) * ratio),
                )
                if duration:
                    time.sleep(duration / steps)
            ctypes.windll.user32.mouse_event(flags[1], 0, 0, 0, 0)  # type: ignore[attr-defined]
            return
        super().drag(x, y, duration=duration, button=button)

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        if self.system == "windows":
            import ctypes

            flag = 0x01000 if horizontal else 0x0800
            ctypes.windll.user32.mouse_event(flag, 0, 0, int(amount) * 120, 0)  # type: ignore[attr-defined]
            return
        if self.system == "linux":
            if horizontal:
                button = "6" if amount > 0 else "7"
            else:
                button = "4" if amount > 0 else "5"
            self._run(
                [self._require("xdotool"), "click", "--repeat", str(abs(int(amount))), button]
            )
            return
        if self.system == "darwin" and self._which.get("cliclick"):
            self._run([self._require("cliclick"), f"w:{int(amount)}"])
            return
        super().scroll(amount, horizontal=horizontal)

    # -- keyboard ----------------------------------------------------------

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        if self.system == "linux":
            self._run(
                [
                    self._require("xdotool"),
                    "type",
                    "--clearmodifiers",
                    "--delay",
                    str(max(0, int(interval * 1000))),
                    "--",
                    str(text),
                ]
            )
            return
        if self.system == "darwin":
            script = f'tell application "System Events" to keystroke {_apple_string(text)}'
            self._run([self._require("osascript"), "-e", script])
            return
        if self.system == "windows":
            _windows_type_unicode(str(text), interval=max(0.0, interval))
            return
        super().type_text(text, interval=interval)

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        normalized = _key(key)
        if self.system == "linux":
            self._run(
                [
                    self._require("xdotool"),
                    "key",
                    "--clearmodifiers",
                    "--repeat",
                    str(max(1, int(presses))),
                    "--delay",
                    str(max(0, int(interval * 1000))),
                    normalized,
                ]
            )
            return
        if self.system == "darwin":
            code = _MAC_KEY_CODES.get(normalized)
            if code is None and len(normalized) == 1:
                script = (
                    f'tell application "System Events" to keystroke {_apple_string(normalized)}'
                )
            elif code is not None:
                script = f'tell application "System Events" to key code {code}'
            else:
                raise ValueError(f"unsupported macOS key {key!r}")
            for index in range(max(1, int(presses))):
                self._run([self._require("osascript"), "-e", script])
                if index + 1 < presses:
                    time.sleep(max(0.0, interval))
            return
        if self.system == "windows":
            _windows_key(normalized, presses=presses, interval=interval)
            return
        super().press(key, presses=presses, interval=interval)

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        normalized = [_key(key) for key in keys]
        if not normalized:
            raise ValueError("hotkey requires at least one key")
        if self.system == "linux":
            self._run([self._require("xdotool"), "key", "+".join(normalized)])
            return
        if self.system == "darwin":
            modifiers = {
                "ctrl": "control down",
                "control": "control down",
                "alt": "option down",
                "option": "option down",
                "shift": "shift down",
                "command": "command down",
                "cmd": "command down",
                "win": "command down",
                "meta": "command down",
            }
            main = normalized[-1]
            used = [modifiers[key] for key in normalized[:-1] if key in modifiers]
            code = _MAC_KEY_CODES.get(main)
            action = f"key code {code}" if code is not None else f"keystroke {_apple_string(main)}"
            using = f" using {{{', '.join(used)}}}" if used else ""
            self._run(
                [
                    self._require("osascript"),
                    "-e",
                    f'tell application "System Events" to {action}{using}',
                ]
            )
            return
        if self.system == "windows":
            _windows_hotkey(normalized, interval=interval)
            return
        super().hotkey(keys, interval=interval)

    # -- screenshots -------------------------------------------------------

    def _screenshot_command(self) -> str:
        for name in ("grim", "gnome-screenshot", "scrot", "import"):
            if self._which.get(name):
                return name
        return ""

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.system == "darwin":
            args = [self._require("screencapture"), "-x"]
            if region:
                args.extend(["-R", f"{region.x},{region.y},{region.width},{region.height}"])
            args.append(str(target))
            self._run(args, timeout=30.0)
        elif self.system == "linux":
            helper = self._screenshot_command()
            if helper == "grim":
                args = [self._require("grim")]
                if region:
                    args.extend(["-g", f"{region.x},{region.y} {region.width}x{region.height}"])
                args.append(str(target))
            elif helper == "gnome-screenshot":
                if region:
                    raise DesktopUnavailableError("gnome-screenshot region capture is interactive")
                args = [self._require("gnome-screenshot"), "-f", str(target)]
            elif helper == "scrot":
                if region:
                    raise DesktopUnavailableError("scrot region capture is interactive")
                args = [self._require("scrot"), str(target)]
            elif helper == "import":
                root = (
                    f"root[geometry={region.width}x{region.height}+{region.x}+{region.y}]"
                    if region
                    else "root"
                )
                args = [self._require("import"), "-window", root, str(target)]
            else:
                raise DesktopUnavailableError(
                    "install grim, gnome-screenshot, scrot, or ImageMagick"
                )
            self._run(args, timeout=30.0)
        elif self.system == "windows":
            powershell = self._require("powershell")
            if region:
                x, y, width, height = region.x, region.y, region.width, region.height
            else:
                width, height = self.screen_size()
                x = y = 0
            escaped = str(target).replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Drawing;"
                f"$b=New-Object Drawing.Bitmap({width},{height});"
                "$g=[Drawing.Graphics]::FromImage($b);"
                f"$g.CopyFromScreen({x},{y},0,0,$b.Size);"
                f"$b.Save('{escaped}',[Drawing.Imaging.ImageFormat]::Png);"
                "$g.Dispose();$b.Dispose()"
            )
            self._run([powershell, "-NoProfile", "-Command", script], timeout=30.0)
        else:
            return super().screenshot(path, region=region)
        if not target.is_file():
            raise DesktopUnavailableError(f"screenshot helper did not create {target}")
        return target

    # -- clipboard ---------------------------------------------------------

    def _clipboard_capabilities(self) -> tuple[bool, bool]:
        if self.system == "windows":
            return bool(self._which.get("powershell")), bool(self._which.get("powershell"))
        if self.system == "darwin":
            return bool(shutil.which("pbpaste")), bool(shutil.which("pbcopy"))
        if self._which.get("wl-paste") and self._which.get("wl-copy"):
            return True, True
        if self._which.get("xclip") or self._which.get("xsel"):
            return True, True
        return False, False

    def clipboard_read(self) -> str:
        if self.system == "windows":
            result = self._run(
                [self._require("powershell"), "-NoProfile", "-Command", "Get-Clipboard -Raw"]
            )
            return result.stdout
        if self.system == "darwin":
            return self._run([self._require_external("pbpaste")]).stdout
        if self._which.get("wl-paste"):
            return self._run([self._require("wl-paste"), "--no-newline"]).stdout
        if self._which.get("xclip"):
            return self._run([self._require("xclip"), "-selection", "clipboard", "-o"]).stdout
        if self._which.get("xsel"):
            return self._run([self._require("xsel"), "--clipboard", "--output"]).stdout
        return super().clipboard_read()

    def clipboard_write(self, text: str) -> None:
        if self.system == "windows":
            self._run(
                [
                    self._require("powershell"),
                    "-NoProfile",
                    "-Command",
                    "$input | Set-Clipboard",
                ],
                input_text=text,
            )
            return
        if self.system == "darwin":
            self._run([self._require_external("pbcopy")], input_text=text)
            return
        if self._which.get("wl-copy"):
            self._run([self._require("wl-copy")], input_text=text)
            return
        if self._which.get("xclip"):
            self._run([self._require("xclip"), "-selection", "clipboard"], input_text=text)
            return
        if self._which.get("xsel"):
            self._run([self._require("xsel"), "--clipboard", "--input"], input_text=text)
            return
        super().clipboard_write(text)

    def _require_external(self, name: str) -> str:
        path = shutil.which(name)
        if not path:
            raise DesktopUnavailableError(f"desktop action needs {name!r} on PATH")
        return path

    # -- windows -----------------------------------------------------------

    def windows(self) -> list[WindowInfo]:
        if self.system == "linux":
            if self._which.get("wmctrl"):
                result = self._run([self._require("wmctrl"), "-lGpx"])
                active = self._linux_active_id()
                output: list[WindowInfo] = []
                for line in result.stdout.splitlines():
                    parts = line.split(None, 8)
                    if len(parts) < 9:
                        continue
                    ident, _desktop, x, y, width, height, pid, app, title = parts
                    output.append(
                        WindowInfo(
                            id=ident,
                            title=title,
                            app=app,
                            pid=int(pid) if pid.isdigit() else 0,
                            bounds=Rect(int(x), int(y), int(width), int(height)),
                            active=ident.lower() == active.lower(),
                        )
                    )
                return output
            ids = self._run(
                [self._require("xdotool"), "search", "--onlyvisible", "--name", "."],
                check=False,
            ).stdout.split()
            return [
                WindowInfo(
                    id=ident,
                    title=self._run(
                        [self._require("xdotool"), "getwindowname", ident], check=False
                    ).stdout.strip(),
                    active=ident == self._linux_active_id(),
                )
                for ident in ids
            ]
        if self.system == "darwin":
            script = (
                'tell application "System Events" to set rows to {}\n'
                'tell application "System Events"\n'
                "repeat with p in (application processes whose background only is false)\n"
                "set appName to name of p\n"
                "repeat with w in windows of p\n"
                'set end of rows to appName & "\\t" & name of w\n'
                "end repeat\nend repeat\nend tell\nreturn rows"
            )
            result = self._run([self._require("osascript"), "-e", script], check=False)
            output = []
            for index, value in enumerate(result.stdout.strip().split(", ")):
                app, _, title = value.partition("\t")
                if app or title:
                    output.append(WindowInfo(id=f"{app}:{index}", title=title, app=app))
            return output
        if self.system == "windows":
            return _windows_list_windows()
        return super().windows()

    def _linux_active_id(self) -> str:
        if not self._which.get("xdotool"):
            return ""
        return self._run([self._require("xdotool"), "getactivewindow"], check=False).stdout.strip()

    def active_window(self) -> WindowInfo | None:
        if self.system == "windows":
            return _windows_active_window()
        return super().active_window()

    def focus_window(self, window_id: str) -> None:
        if self.system == "linux":
            if self._which.get("wmctrl"):
                self._run([self._require("wmctrl"), "-ia", window_id])
            else:
                self._run([self._require("xdotool"), "windowactivate", "--sync", window_id])
            return
        if self.system == "darwin":
            app = window_id.split(":", 1)[0]
            script = f"tell application {_apple_string(app)} to activate"
            self._run([self._require("osascript"), "-e", script])
            return
        if self.system == "windows":
            _windows_focus(window_id)
            return
        super().focus_window(window_id)

    def window_action(self, window_id: str, action: str) -> None:
        action = action.lower().strip()
        if action not in {"minimize", "maximize", "restore", "close"}:
            raise ValueError("window action must be minimize, maximize, restore, or close")
        if self.system == "linux":
            if action == "close":
                command = (
                    [self._require("wmctrl"), "-ic", window_id]
                    if self._which.get("wmctrl")
                    else [self._require("xdotool"), "windowclose", window_id]
                )
            elif self._which.get("wmctrl"):
                mode = "remove" if action == "restore" else "add"
                prop = (
                    "maximized_vert,maximized_horz"
                    if action in {"maximize", "restore"}
                    else "hidden"
                )
                command = [self._require("wmctrl"), "-ir", window_id, "-b", f"{mode},{prop}"]
            else:
                command = [self._require("xdotool"), "windowminimize", window_id]
            self._run(command)
            return
        if self.system == "darwin":
            app, _, index = window_id.rpartition(":")
            number = int(index or 0) + 1
            if action == "close":
                statement = f"click button 1 of window {number}"
            elif action == "minimize":
                statement = f'set value of attribute "AXMinimized" of window {number} to true'
            elif action == "restore":
                statement = f'set value of attribute "AXMinimized" of window {number} to false'
            else:
                statement = f'set value of attribute "AXFullScreen" of window {number} to true'
            script = f'tell application "System Events" to tell process {_apple_string(app)} to {statement}'
            self._run([self._require("osascript"), "-e", script])
            return
        if self.system == "windows":
            _windows_window_action(window_id, action)
            return
        super().window_action(window_id, action)

    def move_resize_window(self, window_id: str, bounds: Rect) -> None:
        if self.system == "linux":
            if self._which.get("wmctrl"):
                self._run(
                    [
                        self._require("wmctrl"),
                        "-ir",
                        window_id,
                        "-e",
                        f"0,{bounds.x},{bounds.y},{bounds.width},{bounds.height}",
                    ]
                )
            else:
                self._run(
                    [
                        self._require("xdotool"),
                        "windowmove",
                        window_id,
                        str(bounds.x),
                        str(bounds.y),
                        "windowsize",
                        window_id,
                        str(bounds.width),
                        str(bounds.height),
                    ]
                )
            return
        if self.system == "darwin":
            app, _, index = window_id.rpartition(":")
            number = int(index or 0) + 1
            script = (
                f'tell application "System Events" to tell process {_apple_string(app)} '
                f"to tell window {number} to set {{position, size}} to "
                f"{{{{{bounds.x}, {bounds.y}}}, {{{bounds.width}, {bounds.height}}}}}"
            )
            self._run([self._require("osascript"), "-e", script])
            return
        if self.system == "windows":
            _windows_move_resize(window_id, bounds)
            return
        super().move_resize_window(window_id, bounds)

    # -- applications ------------------------------------------------------

    def launch(self, app: str, args: list[str] | None = None) -> int:
        arguments = list(args or [])
        if self.system == "darwin" and not Path(app).exists():
            process = subprocess.Popen(
                ["open", "-a", app, *(["--args", *arguments] if arguments else [])],
                start_new_session=True,
            )
            return int(process.pid)
        return super().launch(app, arguments)

    def open_target(self, target: str) -> bool:
        try:
            if self.system == "windows":
                os.startfile(target)  # type: ignore[attr-defined]
                return True
            command = ["open", target] if self.system == "darwin" else ["xdg-open", target]
            subprocess.Popen(command, start_new_session=True)
            return True
        except OSError:
            return super().open_target(target)


class AutoDesktopDriver(DesktopDriver):
    """Composite driver preferring PyAutoGUI and filling gaps natively."""

    name = "auto"

    def __init__(self, *, fail_safe: bool = True, pause: float = 0.05) -> None:
        self.native = NativeDesktopDriver()
        self.gui: PyAutoGUIDriver | None = None
        self.gui_error = ""
        try:
            self.gui = PyAutoGUIDriver(fail_safe=fail_safe, pause=pause)
        except Exception as exc:
            self.gui_error = str(exc)

    def capabilities(self) -> DesktopCapabilities:
        native = self.native.capabilities()
        if self.gui is None:
            if self.gui_error:
                native.notes.append(self.gui_error)
            native.driver = "native"
            return native
        gui = self.gui.capabilities()
        gui.driver = "pyautogui+native"
        gui.clipboard_read = native.clipboard_read
        gui.clipboard_write = native.clipboard_write
        gui.windows = native.windows
        gui.window_control = native.window_control
        gui.dependencies.update(native.dependencies)
        gui.notes.extend(native.notes)
        return gui

    def _visual(self) -> DesktopDriver:
        return self.gui or self.native

    def monitors(self) -> list[MonitorInfo]:
        return self._visual().monitors()

    def screen_size(self) -> tuple[int, int]:
        return self._visual().screen_size()

    def pointer_position(self) -> Point:
        return self._visual().pointer_position()

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> None:
        self._visual().move_pointer(x, y, duration=duration)

    def click(self, **kwargs: Any) -> None:
        self._visual().click(**kwargs)

    def drag(self, x: int, y: int, **kwargs: Any) -> None:
        self._visual().drag(x, y, **kwargs)

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        self._visual().scroll(amount, horizontal=horizontal)

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        self._visual().type_text(text, interval=interval)

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self._visual().press(key, presses=presses, interval=interval)

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        self._visual().hotkey(keys, interval=interval)

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        return self._visual().screenshot(path, region=region)

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        return self._visual().pixel(x, y)

    def locate(
        self, image: str | Path, *, confidence: float = 0.9, region: Rect | None = None
    ) -> Rect | None:
        return self._visual().locate(image, confidence=confidence, region=region)

    def clipboard_read(self) -> str:
        return self.native.clipboard_read()

    def clipboard_write(self, text: str) -> None:
        self.native.clipboard_write(text)

    def windows(self) -> list[WindowInfo]:
        return self.native.windows()

    def active_window(self) -> WindowInfo | None:
        return self.native.active_window()

    def focus_window(self, window_id: str) -> None:
        self.native.focus_window(window_id)

    def window_action(self, window_id: str, action: str) -> None:
        self.native.window_action(window_id, action)

    def move_resize_window(self, window_id: str, bounds: Rect) -> None:
        self.native.move_resize_window(window_id, bounds)

    def launch(self, app: str, args: list[str] | None = None) -> int:
        return self.native.launch(app, args)

    def open_target(self, target: str) -> bool:
        return self.native.open_target(target)


class RecordingDesktopDriver(DesktopDriver):
    """Deterministic in-memory driver for tests, dry runs, and demos."""

    name = "recording"

    def __init__(self, *, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height
        self.pointer = Point(0, 0)
        self.clipboard = ""
        self.actions: list[dict[str, Any]] = []
        self._windows = [
            WindowInfo(
                id="1",
                title="MAgent Test Window",
                app="magent",
                active=True,
                bounds=Rect(0, 0, 800, 600),
            )
        ]

    def _record(self, action: str, **data: Any) -> None:
        self.actions.append({"action": action, **data})

    def capabilities(self) -> DesktopCapabilities:
        return DesktopCapabilities(
            driver=self.name,
            platform="test",
            session="recording",
            display_available=True,
            mouse=True,
            keyboard=True,
            screenshots=True,
            image_location=True,
            clipboard_read=True,
            clipboard_write=True,
            windows=True,
            window_control=True,
            monitors=True,
        )

    def screen_size(self) -> tuple[int, int]:
        return self.width, self.height

    def pointer_position(self) -> Point:
        return self.pointer

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> None:
        self.pointer = Point(int(x), int(y))
        self._record("move", x=x, y=y, duration=duration)

    def click(self, **kwargs: Any) -> None:
        if kwargs.get("x") is not None and kwargs.get("y") is not None:
            self.pointer = Point(int(kwargs["x"]), int(kwargs["y"]))
        self._record("click", **kwargs)

    def drag(self, x: int, y: int, **kwargs: Any) -> None:
        self.pointer = Point(int(x), int(y))
        self._record("drag", x=x, y=y, **kwargs)

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        self._record("scroll", amount=amount, horizontal=horizontal)

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        self._record("type", text=text, interval=interval)

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self._record("press", key=key, presses=presses, interval=interval)

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        self._record("hotkey", keys=list(keys), interval=interval)

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_ONE_PIXEL_PNG)
        self._record("screenshot", path=str(target), region=region.to_dict() if region else None)
        return target

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        self._record("pixel", x=x, y=y)
        return (82, 220, 255)

    def locate(
        self, image: str | Path, *, confidence: float = 0.9, region: Rect | None = None
    ) -> Rect | None:
        self._record("locate", image=str(image), confidence=confidence)
        return Rect(10, 20, 30, 40)

    def clipboard_read(self) -> str:
        self._record("clipboard_read")
        return self.clipboard

    def clipboard_write(self, text: str) -> None:
        self.clipboard = text
        self._record("clipboard_write", text=text)

    def windows(self) -> list[WindowInfo]:
        return list(self._windows)

    def focus_window(self, window_id: str) -> None:
        self._record("focus_window", window_id=window_id)

    def window_action(self, window_id: str, action: str) -> None:
        self._record("window_action", window_id=window_id, operation=action)

    def move_resize_window(self, window_id: str, bounds: Rect) -> None:
        self._record("move_window", window_id=window_id, bounds=bounds.to_dict())

    def launch(self, app: str, args: list[str] | None = None) -> int:
        self._record("launch", app=app, args=list(args or []))
        return 4242

    def open_target(self, target: str) -> bool:
        self._record("open", target=target)
        return True


class DesktopController:
    """Safety-aware façade used by tools, workflows, and the panel."""

    def __init__(self, config: Any = None, *, driver: DesktopDriver | None = None) -> None:
        self.config = getattr(config, "desktop", config)
        backend = str(getattr(self.config, "backend", "auto") or "auto").lower()
        if driver is not None:
            self.driver = driver
        elif backend == "native":
            self.driver = NativeDesktopDriver()
        elif backend == "recording":
            self.driver = RecordingDesktopDriver()
        elif backend == "pyautogui":
            self.driver = PyAutoGUIDriver(
                fail_safe=bool(getattr(self.config, "fail_safe", True)),
                pause=float(getattr(self.config, "action_pause", 0.05)),
            )
        elif backend == "auto":
            self.driver = AutoDesktopDriver(
                fail_safe=bool(getattr(self.config, "fail_safe", True)),
                pause=float(getattr(self.config, "action_pause", 0.05)),
            )
        else:
            raise ValueError("desktop backend must be auto, pyautogui, native, or recording")

    @property
    def capabilities(self) -> DesktopCapabilities:
        """Current driver capabilities."""
        return self.driver.capabilities()

    def describe(self) -> dict[str, Any]:
        """Return configuration and live capability details."""
        return {
            "enabled": bool(getattr(self.config, "enabled", True)),
            "driver": self.driver.name,
            "capabilities": self.capabilities.to_dict(),
            "fail_safe": bool(getattr(self.config, "fail_safe", True)),
            "action_pause": float(getattr(self.config, "action_pause", 0.05)),
        }

    def _allow(self, capability: str, *, changing: bool = False) -> None:
        if not bool(getattr(self.config, "enabled", True)):
            raise DesktopUnavailableError("desktop automation is disabled in [desktop]")
        flag = {
            "mouse": "allow_mouse",
            "keyboard": "allow_keyboard",
            "screenshots": "allow_screenshots",
            "clipboard_read": "allow_clipboard",
            "clipboard_write": "allow_clipboard",
            "windows": "allow_window_control",
            "window_control": "allow_window_control",
            "launch": "allow_launch",
        }.get(capability)
        if flag and not bool(getattr(self.config, flag, True)):
            raise DesktopUnavailableError(
                f"desktop capability {capability!r} is disabled by {flag}"
            )
        if not bool(getattr(self.capabilities, capability, False)):
            raise DesktopUnavailableError(
                f"desktop driver {self.capabilities.driver!r} does not support {capability}"
            )
        if changing:
            self._panic_check()

    def _panic_check(self) -> None:
        if not bool(getattr(self.config, "fail_safe", True)):
            return
        margin = max(0, int(getattr(self.config, "panic_corner_size", 4)))
        try:
            point = self.driver.pointer_position()
        except Exception:
            return
        if point.x <= margin and point.y <= margin:
            raise DesktopUnavailableError(
                "desktop fail-safe engaged: move the pointer away from the top-left corner"
            )

    def _pause(self) -> None:
        delay = max(0.0, float(getattr(self.config, "action_pause", 0.05)))
        if delay:
            time.sleep(delay)

    def _validate_point(self, x: int, y: int) -> None:
        """Reject coordinates outside every reported monitor."""
        try:
            monitors = self.driver.monitors()
        except Exception:
            width, height = self.driver.screen_size()
            monitors = [MonitorInfo(id="primary", bounds=Rect(0, 0, width, height), primary=True)]
        for monitor in monitors:
            bounds = monitor.bounds
            if (
                bounds.x <= int(x) < bounds.x + bounds.width
                and bounds.y <= int(y) < bounds.y + bounds.height
            ):
                return
        raise ValueError(f"desktop coordinate ({x}, {y}) is outside every monitor")

    def _check_app_allowlist(self, app: str) -> None:
        allowed = [
            str(item).strip().casefold() for item in getattr(self.config, "allowed_apps", [])
        ]
        if not allowed:
            return
        raw = str(app).strip().casefold()
        name = Path(raw).name
        stem = Path(name).stem
        if not any(candidate in allowed for candidate in (raw, name, stem)):
            raise PermissionError(f"application {app!r} is not in desktop.allowed_apps")

    def _check_target_scheme(self, target: str) -> None:
        raw = str(target).strip()
        scheme = urlsplit(raw).scheme.casefold()
        if not scheme or (len(scheme) == 1 and len(raw) > 1 and raw[1:2] == ":"):
            scheme = "file"
        allowed = {
            str(item).strip().casefold()
            for item in getattr(
                self.config,
                "allowed_open_schemes",
                ["file", "http", "https", "mailto"],
            )
        }
        if scheme not in allowed:
            raise PermissionError(
                f"target scheme {scheme!r} is not in desktop.allowed_open_schemes"
            )

    def monitors(self) -> list[MonitorInfo]:
        self._allow("monitors")
        return self.driver.monitors()

    def screen_size(self) -> tuple[int, int]:
        if not bool(getattr(self.config, "enabled", True)):
            raise DesktopUnavailableError("desktop automation is disabled")
        return self.driver.screen_size()

    def pointer_position(self) -> Point:
        self._allow("mouse")
        return self.driver.pointer_position()

    def move_pointer(self, x: int, y: int, *, duration: float = 0.0) -> Point:
        self._allow("mouse", changing=True)
        self._validate_point(x, y)
        duration = max(0.0, min(float(duration), 30.0))
        self.driver.move_pointer(x, y, duration=duration)
        self._pause()
        return self.driver.pointer_position()

    def click(self, **kwargs: Any) -> Point:
        self._allow("mouse", changing=True)
        x, y = kwargs.get("x"), kwargs.get("y")
        if (x is None) != (y is None):
            raise ValueError("click x and y must be provided together")
        if x is not None and y is not None:
            self._validate_point(int(x), int(y))
            kwargs["x"], kwargs["y"] = int(x), int(y)
        button = _button(str(kwargs.get("button", "left")))
        kwargs["button"] = button
        kwargs["clicks"] = max(1, min(int(kwargs.get("clicks", 1)), 20))
        kwargs["interval"] = max(0.0, min(float(kwargs.get("interval", 0.1)), 10.0))
        self.driver.click(**kwargs)
        self._pause()
        return self.driver.pointer_position()

    def drag(self, x: int, y: int, *, duration: float = 0.5, button: str = "left") -> Point:
        self._allow("mouse", changing=True)
        self._validate_point(x, y)
        duration = max(0.0, min(float(duration), 30.0))
        self.driver.drag(x, y, duration=duration, button=_button(button))
        self._pause()
        return self.driver.pointer_position()

    def scroll(self, amount: int, *, horizontal: bool = False) -> None:
        self._allow("mouse", changing=True)
        self.driver.scroll(amount, horizontal=horizontal)
        self._pause()

    def type_text(self, text: str, *, interval: float = 0.01) -> None:
        self._allow("keyboard", changing=True)
        if len(text) > int(getattr(self.config, "max_type_chars", 10000)):
            raise ValueError("text exceeds desktop.max_type_chars")
        self.driver.type_text(text, interval=interval)
        self._pause()

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self._allow("keyboard", changing=True)
        self.driver.press(key, presses=max(1, min(int(presses), 100)), interval=interval)
        self._pause()

    def hotkey(self, keys: list[str], *, interval: float = 0.05) -> None:
        self._allow("keyboard", changing=True)
        if not 1 <= len(keys) <= 8:
            raise ValueError("hotkey requires 1..8 keys")
        self.driver.hotkey(keys, interval=interval)
        self._pause()

    def screenshot(self, path: str | Path, *, region: Rect | None = None) -> Path:
        self._allow("screenshots")
        return self.driver.screenshot(path, region=region)

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        self._allow("screenshots")
        self._validate_point(x, y)
        return self.driver.pixel(x, y)

    def locate(
        self, image: str | Path, *, confidence: float = 0.9, region: Rect | None = None
    ) -> Rect | None:
        self._allow("image_location")
        return self.driver.locate(image, confidence=confidence, region=region)

    def clipboard_read(self) -> str:
        self._allow("clipboard_read")
        text = self.driver.clipboard_read()
        limit = max(1, int(getattr(self.config, "max_clipboard_chars", 100000)))
        return text[:limit]

    def clipboard_write(self, text: str) -> None:
        self._allow("clipboard_write", changing=True)
        if len(text) > int(getattr(self.config, "max_clipboard_chars", 100000)):
            raise ValueError("text exceeds desktop.max_clipboard_chars")
        self.driver.clipboard_write(text)
        self._pause()

    def windows(self, *, query: str = "") -> list[WindowInfo]:
        self._allow("windows")
        items = self.driver.windows()
        needle = query.casefold().strip()
        if needle:
            items = [
                item
                for item in items
                if needle in item.title.casefold() or needle in item.app.casefold()
            ]
        return items

    def active_window(self) -> WindowInfo | None:
        self._allow("windows")
        return self.driver.active_window()

    def focus_window(self, window_id: str) -> None:
        self._allow("window_control", changing=True)
        self.driver.focus_window(window_id)
        self._pause()

    def window_action(self, window_id: str, action: str) -> None:
        self._allow("window_control", changing=True)
        self.driver.window_action(window_id, action)
        self._pause()

    def move_resize_window(self, window_id: str, bounds: Rect) -> None:
        self._allow("window_control", changing=True)
        if bounds.width < 50 or bounds.height < 50:
            raise ValueError("window width and height must both be at least 50")
        self.driver.move_resize_window(window_id, bounds)
        self._pause()

    def launch(self, app: str, args: list[str] | None = None) -> int:
        self._allow("launch", changing=True)
        if not str(app).strip():
            raise ValueError("application name must not be empty")
        self._check_app_allowlist(app)
        bounded_args = [str(item) for item in (args or [])]
        if len(bounded_args) > 100 or sum(len(item) for item in bounded_args) > 32_000:
            raise ValueError("application arguments exceed desktop safety limits")
        pid = self.driver.launch(app, bounded_args)
        self._pause()
        return pid

    def open_target(self, target: str) -> bool:
        self._allow("launch", changing=True)
        if not str(target).strip():
            raise ValueError("open target must not be empty")
        self._check_target_scheme(target)
        result = self.driver.open_target(target)
        self._pause()
        return result

    def wait_for_window(
        self,
        query: str,
        *,
        timeout: float = 10.0,
        interval: float = 0.25,
        active: bool = False,
    ) -> WindowInfo | None:
        """Wait for a matching window without sleeping longer than configured."""
        maximum = min(max(0.0, timeout), float(getattr(self.config, "max_wait_seconds", 120.0)))
        deadline = time.monotonic() + maximum
        while True:
            matches = self.windows(query=query)
            match = next((item for item in matches if not active or item.active), None)
            if match is not None:
                return match
            if time.monotonic() >= deadline:
                return None
            time.sleep(max(0.05, min(interval, 2.0)))


# -- platform helpers -------------------------------------------------------


def build_desktop_controller(
    config: Any = None, *, driver: DesktopDriver | None = None
) -> DesktopController:
    """Build the configured desktop controller."""
    return DesktopController(config, driver=driver)


def _session_name(environ: dict[str, str] | None = None) -> str:
    env = environ or os.environ
    return str(
        env.get("XDG_SESSION_TYPE")
        or ("wayland" if env.get("WAYLAND_DISPLAY") else "")
        or ("x11" if env.get("DISPLAY") else "")
        or platform.system().lower()
    )


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _button(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in {"left", "right", "middle"}:
        raise ValueError("mouse button must be left, right, or middle")
    return normalized


def _key(value: str) -> str:
    normalized = value.lower().strip().replace(" ", "_")
    aliases = {
        "return": "enter",
        "escape": "esc",
        "control": "ctrl",
        "windows": "win",
        "super": "win",
        "page_up": "pageup",
        "page_down": "pagedown",
    }
    return aliases.get(normalized, normalized)


def _apple_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


_MAC_KEY_CODES = {
    "enter": 36,
    "tab": 48,
    "space": 49,
    "backspace": 51,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "delete": 117,
}

_WINDOWS_VK = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "esc": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "delete": 0x2E,
    "win": 0x5B,
}
_WINDOWS_VK.update({f"f{index}": 0x6F + index for index in range(1, 13)})


def _windows_vk(key: str) -> int:
    import ctypes

    if key in _WINDOWS_VK:
        return _WINDOWS_VK[key]
    if len(key) == 1:
        return int(ctypes.windll.user32.VkKeyScanW(ord(key)) & 0xFF)  # type: ignore[attr-defined]
    raise ValueError(f"unsupported Windows key {key!r}")


def _windows_key(key: str, *, presses: int = 1, interval: float = 0.05) -> None:
    import ctypes

    vk = _windows_vk(key)
    for index in range(max(1, int(presses))):
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)  # type: ignore[attr-defined]
        ctypes.windll.user32.keybd_event(vk, 0, 0x0002, 0)  # type: ignore[attr-defined]
        if index + 1 < presses:
            time.sleep(max(0.0, interval))


def _windows_hotkey(keys: list[str], *, interval: float = 0.05) -> None:
    import ctypes

    values = [_windows_vk(key) for key in keys]
    for value in values:
        ctypes.windll.user32.keybd_event(value, 0, 0, 0)  # type: ignore[attr-defined]
        if interval:
            time.sleep(max(0.0, interval))
    for value in reversed(values):
        ctypes.windll.user32.keybd_event(value, 0, 0x0002, 0)  # type: ignore[attr-defined]


def _windows_type_unicode(text: str, *, interval: float = 0.01) -> None:
    """Type Unicode through Win32 ``SendInput`` without touching the clipboard."""
    import ctypes
    from ctypes import wintypes

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]

    encoded = text.encode("utf-16-le")
    units = [
        int.from_bytes(encoded[index : index + 2], "little") for index in range(0, len(encoded), 2)
    ]
    for unit in units:
        down = INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(0, unit, 0x0004, 0, None)))
        up = INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(0, unit, 0x0004 | 0x0002, 0, None)))
        payload = (INPUT * 2)(down, up)
        sent = ctypes.windll.user32.SendInput(2, payload, ctypes.sizeof(INPUT))  # type: ignore[attr-defined]
        if sent != 2:
            raise DesktopUnavailableError("SendInput failed while typing")
        if interval:
            time.sleep(interval)


def _windows_handle(window_id: str) -> int:
    try:
        return int(window_id, 0)
    except ValueError as exc:
        raise ValueError(f"invalid Windows window id {window_id!r}") from exc


def _windows_list_windows() -> list[WindowInfo]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    active = int(user32.GetForegroundWindow())
    output: list[WindowInfo] = []
    callback_type: Any = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

    @callback_type(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        output.append(
            WindowInfo(
                id=hex(int(hwnd)),
                title=buffer.value,
                pid=int(pid.value),
                bounds=Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top),
                active=int(hwnd) == active,
                minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    user32.EnumWindows(callback, 0)
    return output


def _windows_active_window() -> WindowInfo | None:
    import ctypes

    handle = int(ctypes.windll.user32.GetForegroundWindow())  # type: ignore[attr-defined]
    return next(
        (item for item in _windows_list_windows() if _windows_handle(item.id) == handle), None
    )


def _windows_focus(window_id: str) -> None:
    import ctypes

    hwnd = _windows_handle(window_id)
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.ShowWindow(hwnd, 9)
    if not user32.SetForegroundWindow(hwnd):
        raise DesktopUnavailableError(f"could not focus window {window_id}")


def _windows_window_action(window_id: str, action: str) -> None:
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = _windows_handle(window_id)
    if action == "close":
        user32.PostMessageW(hwnd, 0x0010, 0, 0)
        return
    command = {"minimize": 6, "maximize": 3, "restore": 9}[action]
    user32.ShowWindow(hwnd, command)


def _windows_move_resize(window_id: str, bounds: Rect) -> None:
    import ctypes

    ok = ctypes.windll.user32.MoveWindow(  # type: ignore[attr-defined]
        _windows_handle(window_id),
        bounds.x,
        bounds.y,
        bounds.width,
        bounds.height,
        True,
    )
    if not ok:
        raise DesktopUnavailableError(f"could not move window {window_id}")


_ONE_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


__all__ = [
    "AutoDesktopDriver",
    "DesktopController",
    "DesktopDriver",
    "DesktopUnavailableError",
    "NativeDesktopDriver",
    "PyAutoGUIDriver",
    "RecordingDesktopDriver",
    "build_desktop_controller",
]
