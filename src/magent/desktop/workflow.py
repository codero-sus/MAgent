"""Declarative, stoppable desktop workflows.

A workflow is a JSON list of explicit actions rather than arbitrary Python.  It
can be previewed with ``dry_run``, bounded by a maximum step count, paused or
stopped from the floating panel, and inspected step by step afterwards.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .driver import DesktopController
from .models import Rect, WorkflowResult, WorkflowStepResult

WorkflowListener = Callable[[str, dict[str, Any]], Any]


@dataclass(slots=True)
class DesktopStep:
    """One allowlisted action in a desktop workflow."""

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    delay_after: float = 0.0
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        self.action = str(self.action).strip().lower().replace("-", "_")
        if not self.action:
            raise ValueError("desktop workflow action must not be empty")
        self.params = dict(self.params or {})
        self.delay_after = max(0.0, min(float(self.delay_after), 60.0))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "action": self.action,
            "params": dict(self.params),
            "name": self.name,
            "delay_after": self.delay_after,
            "continue_on_error": self.continue_on_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DesktopStep:
        """Parse both nested ``params`` and compact action dictionaries."""
        known = {"action", "name", "delay_after", "continue_on_error", "params"}
        params = dict(value.get("params") or {})
        params.update({key: item for key, item in value.items() if key not in known})
        return cls(
            action=str(value.get("action", "")),
            params=params,
            name=str(value.get("name", "")),
            delay_after=float(value.get("delay_after", 0.0) or 0.0),
            continue_on_error=bool(value.get("continue_on_error", False)),
        )


@dataclass(slots=True)
class DesktopWorkflow:
    """Named sequence of desktop actions."""

    steps: list[DesktopStep]
    name: str = "desktop-workflow"
    description: str = ""
    variables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = [
            step if isinstance(step, DesktopStep) else DesktopStep.from_dict(step)
            for step in self.steps
        ]
        if not self.steps:
            raise ValueError("desktop workflow must contain at least one step")
        if len(self.steps) > 500:
            raise ValueError("desktop workflow cannot exceed 500 steps")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "name": self.name,
            "description": self.description,
            "variables": dict(self.variables),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def parse(cls, value: str | list[Any] | dict[str, Any]) -> DesktopWorkflow:
        """Parse a JSON string, compact list, or full workflow object."""
        payload: Any = value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"workflow is not valid JSON: {exc}") from exc
        if isinstance(payload, list):
            return cls(steps=[DesktopStep.from_dict(item) for item in payload])
        if not isinstance(payload, dict):
            raise ValueError("workflow must be a JSON object or list")
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise ValueError("workflow object needs a steps list")
        return cls(
            name=str(payload.get("name", "desktop-workflow")),
            description=str(payload.get("description", "")),
            variables=dict(payload.get("variables") or {}),
            steps=[DesktopStep.from_dict(item) for item in steps],
        )


class DesktopWorkflowRunner:
    """Execute allowlisted steps through a :class:`DesktopController`."""

    def __init__(
        self,
        controller: DesktopController,
        *,
        listener: WorkflowListener | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.controller = controller
        self.listener = listener
        self.cancelled = cancelled
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.context: dict[str, Any] = {}

    @property
    def stopped(self) -> bool:
        """Whether an emergency stop or parent-agent cancellation was requested."""
        if self._stop.is_set():
            return True
        if self.cancelled is not None:
            try:
                return bool(self.cancelled())
            except Exception:
                return False
        return False

    @property
    def paused(self) -> bool:
        """Whether execution is waiting for resume."""
        return self._pause.is_set()

    def stop(self) -> None:
        """Request a stop before the next step."""
        self._stop.set()

    def pause(self) -> None:
        """Pause before the next step."""
        self._pause.set()

    def resume(self) -> None:
        """Resume a paused workflow."""
        self._pause.clear()

    def reset(self) -> None:
        """Clear stop/pause flags and captured outputs."""
        self._stop.clear()
        self._pause.clear()
        self.context.clear()

    def run(
        self,
        workflow: DesktopWorkflow | str | list[Any] | dict[str, Any],
        *,
        dry_run: bool = False,
        stop_on_error: bool = True,
    ) -> WorkflowResult:
        """Execute or preview a workflow."""
        parsed = (
            workflow if isinstance(workflow, DesktopWorkflow) else DesktopWorkflow.parse(workflow)
        )
        limit = max(1, min(int(getattr(self.controller.config, "max_workflow_steps", 100)), 500))
        if len(parsed.steps) > limit:
            raise ValueError(
                f"desktop workflow has {len(parsed.steps)} steps; configured limit is {limit}"
            )
        self.reset()
        self.context.update(parsed.variables)
        result = WorkflowResult(name=parsed.name, ok=True)
        self._emit("workflow.started", workflow=parsed.to_dict(), dry_run=dry_run)
        for index, step in enumerate(parsed.steps):
            self._wait_if_paused()
            if self.stopped:
                result.ok = False
                result.stopped = True
                break
            started = time.perf_counter()
            self._emit("step.started", index=index, step=step.to_dict())
            if dry_run:
                outcome = WorkflowStepResult(
                    index=index,
                    action=step.action,
                    ok=True,
                    output={"dry_run": True, "params": self._resolve(step.params)},
                    skipped=True,
                )
            else:
                try:
                    output = self._execute(step)
                except Exception as exc:
                    outcome = WorkflowStepResult(
                        index=index,
                        action=step.action,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                else:
                    outcome = WorkflowStepResult(
                        index=index,
                        action=step.action,
                        ok=True,
                        output=output,
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
            result.steps.append(outcome)
            self.context[f"step_{index}"] = outcome.output
            self.context[step.name or f"step_{index}"] = outcome.output
            self._emit("step.finished", result=outcome.to_dict())
            if not outcome.ok:
                result.ok = False
                if stop_on_error and not step.continue_on_error:
                    break
            if step.delay_after and not self._interruptible_sleep(step.delay_after):
                result.ok = False
                result.stopped = True
                break
        result.completed_at = time.time()
        result.stopped = result.stopped or self.stopped
        self._emit("workflow.finished", result=result.to_dict())
        return result

    def _execute(self, step: DesktopStep) -> Any:
        params = self._resolve(step.params)
        action = _ACTION_ALIASES.get(step.action, step.action)
        controller = self.controller
        if action == "move":
            return controller.move_pointer(
                int(params["x"]), int(params["y"]), duration=float(params.get("duration", 0.0))
            ).to_dict()
        if action == "click":
            return controller.click(
                x=_optional_int(params.get("x")),
                y=_optional_int(params.get("y")),
                button=str(params.get("button", "left")),
                clicks=int(params.get("clicks", 1)),
                interval=float(params.get("interval", 0.1)),
            ).to_dict()
        if action == "drag":
            return controller.drag(
                int(params["x"]),
                int(params["y"]),
                duration=float(params.get("duration", 0.5)),
                button=str(params.get("button", "left")),
            ).to_dict()
        if action == "scroll":
            controller.scroll(
                int(params.get("amount", 0)), horizontal=bool(params.get("horizontal", False))
            )
            return {"scrolled": int(params.get("amount", 0))}
        if action == "type":
            text = str(params.get("text", ""))
            controller.type_text(text, interval=float(params.get("interval", 0.01)))
            return {"typed_chars": len(text)}
        if action == "press":
            controller.press(
                str(params["key"]),
                presses=int(params.get("presses", 1)),
                interval=float(params.get("interval", 0.05)),
            )
            return {"key": str(params["key"]), "presses": int(params.get("presses", 1))}
        if action == "hotkey":
            keys = _keys(params.get("keys", []))
            controller.hotkey(keys, interval=float(params.get("interval", 0.05)))
            return {"keys": keys}
        if action == "wait":
            seconds = max(0.0, min(float(params.get("seconds", 0.0)), 120.0))
            completed = self._interruptible_sleep(seconds)
            return {"waited": seconds, "completed": completed}
        if action == "screenshot":
            region = _region(params)
            path = controller.screenshot(str(params["path"]), region=region)
            return {"path": str(path), "region": region.to_dict() if region else None}
        if action == "locate":
            region = _region(params)
            located = controller.locate(
                str(params["image"]),
                confidence=float(params.get("confidence", 0.9)),
                region=region,
            )
            if located is None:
                return {"found": False}
            if bool(params.get("click", False)):
                controller.click(
                    x=located.x + located.width // 2,
                    y=located.y + located.height // 2,
                )
            return {"found": True, "bounds": located.to_dict()}
        if action == "clipboard_read":
            return {"text": controller.clipboard_read()}
        if action == "clipboard_write":
            text = str(params.get("text", ""))
            controller.clipboard_write(text)
            return {"written_chars": len(text)}
        if action == "focus_window":
            controller.focus_window(str(params["window_id"]))
            return {"window_id": str(params["window_id"])}
        if action == "window_action":
            controller.window_action(str(params["window_id"]), str(params["operation"]))
            return {"window_id": str(params["window_id"]), "operation": params["operation"]}
        if action == "move_window":
            bounds = _required_region(params)
            controller.move_resize_window(str(params["window_id"]), bounds)
            return {"window_id": str(params["window_id"]), "bounds": bounds.to_dict()}
        if action == "wait_window":
            window = controller.wait_for_window(
                str(params["query"]),
                timeout=float(params.get("timeout", 10.0)),
                interval=float(params.get("interval", 0.25)),
                active=bool(params.get("active", False)),
            )
            return {"found": bool(window), "window": window.to_dict() if window else None}
        if action == "launch":
            args = params.get("args", [])
            if isinstance(args, str):
                args = [args]
            return {"pid": controller.launch(str(params["app"]), [str(item) for item in args])}
        if action == "open":
            return {"opened": controller.open_target(str(params["target"]))}
        raise ValueError(f"unsupported desktop workflow action {step.action!r}")

    def _resolve(self, value: Any) -> Any:
        """Resolve ``${name}`` placeholders from prior step outputs."""
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if not isinstance(value, str):
            return value
        if value.startswith("${") and value.endswith("}"):
            return _dig(self.context, value[2:-1])
        rendered = value
        for key, item in self.context.items():
            marker = "${" + key + "}"
            if marker in rendered and isinstance(item, (str, int, float, bool)):
                rendered = rendered.replace(marker, str(item))
        return rendered

    def _wait_if_paused(self) -> None:
        while self.paused and not self.stopped:
            time.sleep(0.05)

    def _interruptible_sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while not self.stopped:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._stop.wait(min(remaining, 0.05))
        return False

    def _emit(self, event: str, **payload: Any) -> None:
        if self.listener is not None:
            try:
                self.listener(event, payload)
            except Exception:
                pass


_ACTION_ALIASES = {
    "move_mouse": "move",
    "mouse_move": "move",
    "mouse_click": "click",
    "drag_mouse": "drag",
    "type_text": "type",
    "press_key": "press",
    "sleep": "wait",
    "take_screenshot": "screenshot",
    "find_image": "locate",
    "set_clipboard": "clipboard_write",
    "get_clipboard": "clipboard_read",
    "activate_window": "focus_window",
    "launch_app": "launch",
    "open_target": "open",
}


def _optional_int(value: Any) -> int | None:
    return None if value is None or value == "" else int(value)


def _keys(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace("+", ",").split(",") if item.strip()]
    return [str(item) for item in (value or [])]


def _region(params: Mapping[str, Any]) -> Rect | None:
    value = params.get("region")
    if isinstance(value, Mapping):
        return _required_region(value)
    names = ("x", "y", "width", "height")
    if all(name in params and params[name] is not None for name in names):
        return Rect(*(int(params[name]) for name in names))
    return None


def _required_region(params: Mapping[str, Any]) -> Rect:
    try:
        rect = Rect(
            x=int(params["x"]),
            y=int(params["y"]),
            width=int(params["width"]),
            height=int(params["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("region needs integer x, y, width, and height") from exc
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("region width and height must be positive")
    return rect


def _dig(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError(f"unknown workflow placeholder {path!r}")
    return current


def load_workflow(path: str | Path) -> DesktopWorkflow:
    """Load a workflow JSON document from disk."""
    return DesktopWorkflow.parse(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "DesktopStep",
    "DesktopWorkflow",
    "DesktopWorkflowRunner",
    "WorkflowListener",
    "load_workflow",
]
