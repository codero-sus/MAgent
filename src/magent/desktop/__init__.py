"""Desktop automation, durable tasks, workflows, branding, and floating UI.

The package is safe to import on a headless machine.  Tkinter, PyAutoGUI, and
operating-system helpers are only touched when the corresponding feature runs.
"""

from .branding import (
    DEFAULT_LOGO_THEME,
    LOGO_THEMES,
    DesktopAppearance,
    DesktopAppearanceStore,
    LogoTheme,
    icon_svg,
    lockup_svg,
    logo_ids,
    logo_theme,
    write_logo_assets,
)
from .driver import (
    AutoDesktopDriver,
    DesktopController,
    DesktopDriver,
    DesktopUnavailableError,
    NativeDesktopDriver,
    PyAutoGUIDriver,
    RecordingDesktopDriver,
    build_desktop_controller,
)
from .models import (
    DesktopCapabilities,
    DesktopTask,
    MonitorInfo,
    Point,
    Rect,
    TaskStatus,
    WindowInfo,
    WorkflowResult,
    WorkflowStepResult,
)
from .queue import DesktopQueueWorker, DesktopTaskQueue
from .workflow import DesktopStep, DesktopWorkflow, DesktopWorkflowRunner, load_workflow

__all__ = [
    "DEFAULT_LOGO_THEME",
    "LOGO_THEMES",
    "AutoDesktopDriver",
    "DesktopAppearance",
    "DesktopAppearanceStore",
    "DesktopCapabilities",
    "DesktopController",
    "DesktopDriver",
    "DesktopQueueWorker",
    "DesktopStep",
    "DesktopTask",
    "DesktopTaskQueue",
    "DesktopUnavailableError",
    "DesktopWorkflow",
    "DesktopWorkflowRunner",
    "LogoTheme",
    "MonitorInfo",
    "NativeDesktopDriver",
    "Point",
    "PyAutoGUIDriver",
    "RecordingDesktopDriver",
    "Rect",
    "TaskStatus",
    "WindowInfo",
    "WorkflowResult",
    "WorkflowStepResult",
    "build_desktop_controller",
    "icon_svg",
    "load_workflow",
    "lockup_svg",
    "logo_ids",
    "logo_theme",
    "write_logo_assets",
]
