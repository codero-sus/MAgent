"""Value objects shared by MAgent's desktop automation subsystem.

The desktop layer deliberately keeps operating-system handles out of its public
API.  Windows, monitors, queued jobs, and workflow outcomes are represented by
small JSON-safe dataclasses, which makes the native UI, CLI, HTTP API, tools,
and tests speak the same language.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..types import new_id


class TaskStatus(StrEnum):
    """Lifecycle states for a persistent desktop task."""

    QUEUED = "queued"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})
"""States from which a task only moves after an explicit retry."""


@dataclass(slots=True, frozen=True)
class Point:
    """A physical desktop coordinate."""

    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe mapping."""
        return {"x": self.x, "y": self.y}


@dataclass(slots=True, frozen=True)
class Rect:
    """A rectangular screen or window region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Right-most coordinate, exclusive."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Bottom-most coordinate, exclusive."""
        return self.y + self.height

    def contains(self, point: Point) -> bool:
        """Return whether ``point`` lies inside this rectangle."""
        return self.x <= point.x < self.right and self.y <= point.y < self.bottom

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe mapping."""
        return asdict(self)


@dataclass(slots=True, frozen=True)
class MonitorInfo:
    """One monitor in the virtual desktop."""

    id: str
    bounds: Rect
    primary: bool = False
    scale: float = 1.0
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "id": self.id,
            "name": self.name or self.id,
            "bounds": self.bounds.to_dict(),
            "primary": self.primary,
            "scale": self.scale,
        }


@dataclass(slots=True, frozen=True)
class WindowInfo:
    """A top-level desktop window."""

    id: str
    title: str
    app: str = ""
    pid: int = 0
    bounds: Rect | None = None
    active: bool = False
    minimized: bool = False
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "id": self.id,
            "title": self.title,
            "app": self.app,
            "pid": self.pid,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "active": self.active,
            "minimized": self.minimized,
            "visible": self.visible,
        }


@dataclass(slots=True)
class DesktopCapabilities:
    """Feature probes for the selected desktop driver."""

    driver: str
    platform: str
    session: str = ""
    display_available: bool = False
    mouse: bool = False
    keyboard: bool = False
    screenshots: bool = False
    image_location: bool = False
    clipboard_read: bool = False
    clipboard_write: bool = False
    windows: bool = False
    window_control: bool = False
    launch: bool = True
    monitors: bool = False
    dependencies: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Whether the driver can perform at least input or window automation."""
        return self.display_available and (self.mouse or self.keyboard or self.window_control)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


@dataclass(slots=True)
class DesktopTask:
    """One durable prompt waiting for the desktop worker."""

    prompt: str
    title: str = ""
    id: str = field(default_factory=lambda: new_id("task"))
    status: TaskStatus = TaskStatus.QUEUED
    priority: int = 50
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    attempts: int = 0
    max_attempts: int = 1
    worker_id: str = ""
    session_id: str = "desktop"
    result: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize status and bounded queue values."""
        if not isinstance(self.status, TaskStatus):
            self.status = TaskStatus(str(self.status))
        self.prompt = str(self.prompt).strip()
        self.title = str(self.title).strip() or self._default_title()
        self.priority = max(0, min(100, int(self.priority)))
        self.max_attempts = max(1, min(20, int(self.max_attempts)))
        self.attempts = max(0, int(self.attempts))

    def _default_title(self) -> str:
        """Build a compact title from the first non-empty prompt line."""
        first = next((line.strip() for line in self.prompt.splitlines() if line.strip()), "Task")
        return first if len(first) <= 72 else first[:69].rstrip() + "..."

    @property
    def terminal(self) -> bool:
        """Whether the task has reached a terminal state."""
        return self.status in TERMINAL_TASK_STATUSES

    @property
    def age_seconds(self) -> float:
        """Seconds since the task was created."""
        return max(0.0, time.time() - self.created_at)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "result": self.result,
            "error": self.error,
            "metadata": dict(self.metadata),
            "terminal": self.terminal,
            "age_seconds": round(self.age_seconds, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesktopTask:
        """Rebuild a task from storage or API data."""
        known = {
            "id",
            "title",
            "prompt",
            "status",
            "priority",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
            "attempts",
            "max_attempts",
            "worker_id",
            "session_id",
            "result",
            "error",
            "metadata",
        }
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class WorkflowStepResult:
    """Outcome of one action in a desktop workflow."""

    index: int
    action: str
    ok: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)


@dataclass(slots=True)
class WorkflowResult:
    """Outcome of a complete desktop workflow."""

    name: str
    ok: bool
    steps: list[WorkflowStepResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    stopped: bool = False

    @property
    def duration_ms(self) -> float:
        """Elapsed workflow time in milliseconds."""
        end = self.completed_at or time.time()
        return max(0.0, (end - self.started_at) * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "name": self.name,
            "ok": self.ok,
            "steps": [step.to_dict() for step in self.steps],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 3),
            "stopped": self.stopped,
        }


__all__ = [
    "TERMINAL_TASK_STATUSES",
    "DesktopCapabilities",
    "DesktopTask",
    "MonitorInfo",
    "Point",
    "Rect",
    "TaskStatus",
    "WindowInfo",
    "WorkflowResult",
    "WorkflowStepResult",
]
