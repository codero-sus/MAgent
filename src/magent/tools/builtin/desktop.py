"""Desktop automation and durable work-queue tools.

Changing the user's desktop is never implicit: input, window, launch, clipboard
write, and workflow tools are marked for human approval.  Capability probes and
window listings remain available in headless environments so the model can
reason about what is possible instead of repeatedly attempting unavailable
operations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...desktop import DesktopTaskQueue, DesktopWorkflow, DesktopWorkflowRunner, Rect
from ...desktop.driver import DesktopController, build_desktop_controller
from ...errors import ApprovalRequiredError
from ...tools.approval import build_request
from ...tools.base import ToolContext, tool
from ...types import ApprovalDecision


def _config(context: ToolContext | None) -> Any:
    if context is not None and context.config is not None:
        return context.config
    from ...config import default_config

    return default_config()


def _controller(context: ToolContext | None) -> DesktopController:
    if context is not None:
        _approval_gate(context)
        cached = context.extras.get("__desktop_controller__")
        if isinstance(cached, DesktopController):
            return cached
        instance = build_desktop_controller(_config(context))
        context.extras["__desktop_controller__"] = instance
        return instance
    return build_desktop_controller(_config(context))


def _approval_gate(context: ToolContext) -> None:
    """Protect direct desktop-tool dispatches that bypass the agent loop.

    Agent-dispatched calls carry a per-call approval marker. CLI/API calls do
    not, so they are evaluated here through the same approval policy before a
    desktop controller is returned.
    """
    registry = context.registry
    instance = registry.get(context.tool_name) if registry is not None else None
    spec = getattr(instance, "spec", None)
    if spec is None or spec.category != "desktop" or not spec.requires_approval:
        return
    approved = context.extras.get("__approved_call_ids__", set())
    if isinstance(approved, set) and context.call_id in approved:
        return
    manager = context.approval
    if manager is None:
        raise ApprovalRequiredError(
            f"Approval required for {context.tool_name}; no approver is configured"
        )
    arguments_by_call = context.extras.get("__tool_arguments__", {})
    arguments = (
        dict(arguments_by_call.get(context.call_id, {}))
        if isinstance(arguments_by_call, dict)
        else {}
    )
    request = build_request(
        context.tool_name,
        arguments,
        risk=spec.risk,
        category=spec.category,
        destructive=spec.destructive,
        network=spec.network,
        required=spec.requires_approval,
        reason="desktop tool requires explicit approval",
        session_id=context.session_id,
        run_id=context.run_id,
    )
    record = manager.evaluate(request)
    if record.decision not in {ApprovalDecision.APPROVE, ApprovalDecision.APPROVE_SESSION}:
        raise ApprovalRequiredError(f"Approval denied for {context.tool_name}")
    marker = context.extras.setdefault("__approved_call_ids__", set())
    if isinstance(marker, set):
        marker.add(context.call_id)


def _queue_path(context: ToolContext | None) -> Path:
    config = _config(context)
    return Path(config.desktop.queue_path).expanduser()


def _rect(x: int | None, y: int | None, width: int | None, height: int | None) -> Rect | None:
    values = (x, y, width, height)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("region requires x, y, width, and height together")
    assert x is not None and y is not None and width is not None and height is not None
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return Rect(x, y, width, height)


@tool(category="desktop", risk="safe", tags=["desktop", "capabilities", "core"])
def desktop_capabilities(context: ToolContext | None = None) -> dict[str, Any]:
    """Report desktop driver support, safety settings, and missing helpers."""
    return _controller(context).describe()


@tool(category="desktop", risk="read", tags=["screen", "monitor"])
def desktop_screen(context: ToolContext | None = None) -> dict[str, Any]:
    """Report virtual screen size and monitor geometry."""
    controller = _controller(context)
    width, height = controller.screen_size()
    monitors: list[dict[str, Any]] = []
    try:
        monitors = [monitor.to_dict() for monitor in controller.monitors()]
    except Exception:
        monitors = [
            {"id": "primary", "bounds": Rect(0, 0, width, height).to_dict(), "primary": True}
        ]
    return {"width": width, "height": height, "monitors": monitors}


@tool(category="desktop", risk="read", tags=["mouse", "position"])
def desktop_pointer(context: ToolContext | None = None) -> dict[str, int]:
    """Return the current pointer coordinates."""
    return _controller(context).pointer_position().to_dict()


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["mouse", "move"],
)
def desktop_move(
    x: int,
    y: int,
    duration: float = 0.0,
    context: ToolContext | None = None,
) -> dict[str, int]:
    """Move the pointer to an absolute screen coordinate.

    Args:
        x: target horizontal coordinate
        y: target vertical coordinate
        duration: animated movement time in seconds
    """
    return _controller(context).move_pointer(x, y, duration=duration).to_dict()


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["mouse", "click"],
)
def desktop_click(
    x: int | None = None,
    y: int | None = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Click at a coordinate, or at the current pointer location.

    Args:
        x: optional horizontal coordinate; provide together with y
        y: optional vertical coordinate; provide together with x
        button: left, middle, or right
        clicks: number of clicks, capped at 10
        interval: delay between repeated clicks
    """
    if (x is None) != (y is None):
        raise ValueError("x and y must be provided together")
    point = _controller(context).click(
        x=x,
        y=y,
        button=button,
        clicks=max(1, min(clicks, 10)),
        interval=max(0.0, min(interval, 2.0)),
    )
    return {"clicked": True, "button": button, "clicks": clicks, "pointer": point.to_dict()}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["mouse", "drag"],
)
def desktop_drag(
    x: int,
    y: int,
    duration: float = 0.5,
    button: str = "left",
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Drag from the current pointer position to a coordinate."""
    point = _controller(context).drag(x, y, duration=duration, button=button)
    return {"dragged": True, "pointer": point.to_dict(), "button": button}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["mouse", "scroll"],
)
def desktop_scroll(
    amount: int,
    horizontal: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Scroll vertically or horizontally; positive values move up/right."""
    bounded = max(-100, min(int(amount), 100))
    _controller(context).scroll(bounded, horizontal=horizontal)
    return {"scrolled": bounded, "horizontal": horizontal}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["keyboard", "type"],
)
def desktop_type(
    text: str,
    interval: float = 0.01,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Type text into the currently focused control."""
    _controller(context).type_text(text, interval=max(0.0, min(interval, 2.0)))
    return {"typed_chars": len(text)}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["keyboard", "key"],
)
def desktop_key(
    key: str,
    presses: int = 1,
    interval: float = 0.05,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Press one named keyboard key one or more times."""
    bounded = max(1, min(int(presses), 100))
    _controller(context).press(key, presses=bounded, interval=interval)
    return {"key": key, "presses": bounded}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["keyboard", "hotkey"],
)
def desktop_hotkey(
    keys: str,
    interval: float = 0.05,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Press a key chord such as ``ctrl,shift,s``.

    Args:
        keys: comma- or plus-separated key names
        interval: delay between key-down events
    """
    parsed = [item.strip() for item in keys.replace("+", ",").split(",") if item.strip()]
    _controller(context).hotkey(parsed, interval=interval)
    return {"keys": parsed}


@tool(
    category="desktop",
    risk="read",
    filesystem=True,
    requires_approval=True,
    parallel_safe=False,
    tags=["screen", "screenshot"],
)
def desktop_screenshot(
    path: str = "",
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Capture the desktop or a rectangular region to a PNG file.

    Args:
        path: output path; defaults to the configured screenshot directory
        x: optional region x coordinate
        y: optional region y coordinate
        width: optional region width
        height: optional region height
    """
    config = _config(context)
    if path:
        target = context.resolve_path(path) if context is not None else Path(path).expanduser()
        if context is not None and context.sandbox is not None:
            context.sandbox.check_write(target)
    else:
        directory = Path(config.desktop.screenshot_directory).expanduser()
        target = directory / time.strftime("magent-%Y%m%d-%H%M%S.png")
    region = _rect(x, y, width, height)
    saved = _controller(context).screenshot(target, region=region)
    return {
        "path": str(saved),
        "bytes": saved.stat().st_size if saved.is_file() else 0,
        "region": region.to_dict() if region else None,
    }


@tool(
    category="desktop",
    risk="read",
    requires_approval=True,
    tags=["screen", "pixel"],
)
def desktop_pixel(x: int, y: int, context: ToolContext | None = None) -> dict[str, Any]:
    """Read the RGB colour at one screen coordinate."""
    red, green, blue = _controller(context).pixel(x, y)
    return {"x": x, "y": y, "rgb": [red, green, blue], "hex": f"#{red:02x}{green:02x}{blue:02x}"}


@tool(
    category="desktop",
    risk="read",
    filesystem=True,
    requires_approval=True,
    tags=["screen", "vision", "locate"],
)
def desktop_locate(
    image: str,
    confidence: float = 0.9,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Locate a reference image on screen using the optional desktop extras."""
    target = context.resolve_path(image) if context is not None else Path(image).expanduser()
    if context is not None and context.sandbox is not None:
        context.sandbox.check_read(target)
    region = _rect(x, y, width, height)
    match = _controller(context).locate(target, confidence=confidence, region=region)
    return {"found": bool(match), "bounds": match.to_dict() if match else None}


@tool(category="desktop", risk="read", tags=["window", "list"])
def desktop_windows(query: str = "", context: ToolContext | None = None) -> dict[str, Any]:
    """List visible top-level windows, optionally filtering by title or app."""
    windows = _controller(context).windows(query=query)
    return {
        "windows": [window.to_dict() for window in windows],
        "count": len(windows),
        "query": query,
    }


@tool(category="desktop", risk="read", tags=["window", "active"])
def desktop_active_window(context: ToolContext | None = None) -> dict[str, Any]:
    """Describe the currently focused top-level window."""
    window = _controller(context).active_window()
    return {"window": window.to_dict() if window else None}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["window", "focus"],
)
def desktop_focus_window(window_id: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Bring a top-level window to the foreground by its reported id."""
    _controller(context).focus_window(window_id)
    return {"focused": window_id}


@tool(
    category="desktop",
    risk="destructive",
    destructive=True,
    requires_approval=True,
    parallel_safe=False,
    tags=["window", "manage"],
)
def desktop_window_action(
    window_id: str,
    action: str,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Minimize, maximize, restore, or close a window."""
    _controller(context).window_action(window_id, action)
    return {"window_id": window_id, "action": action}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["window", "move", "resize"],
)
def desktop_move_window(
    window_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Move and resize one top-level window."""
    bounds = Rect(x, y, width, height)
    _controller(context).move_resize_window(window_id, bounds)
    return {"window_id": window_id, "bounds": bounds.to_dict()}


@tool(
    category="desktop",
    risk="execute",
    executes_code=True,
    requires_approval=True,
    parallel_safe=False,
    tags=["application", "launch"],
)
def desktop_launch(
    app: str,
    args: list[str] | None = None,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Launch an application with an argument list, without a shell."""
    pid = _controller(context).launch(app, list(args or []))
    return {"app": app, "args": list(args or []), "pid": pid}


@tool(
    category="desktop",
    risk="network",
    network=True,
    requires_approval=True,
    parallel_safe=False,
    tags=["application", "open", "url"],
)
def desktop_open(target: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Open a URL, document, or folder with the system default application."""
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"} and context is not None and context.sandbox is not None:
        context.sandbox.check_url(target)
    opened = _controller(context).open_target(target)
    return {"target": target, "opened": opened}


@tool(
    category="desktop",
    risk="read",
    requires_approval=True,
    tags=["clipboard", "read", "sensitive"],
)
def desktop_clipboard_read(context: ToolContext | None = None) -> dict[str, Any]:
    """Read plain text from the system clipboard; may expose sensitive data."""
    text = _controller(context).clipboard_read()
    return {"text": text, "chars": len(text)}


@tool(
    category="desktop",
    risk="write",
    requires_approval=True,
    parallel_safe=False,
    tags=["clipboard", "write"],
)
def desktop_clipboard_write(text: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Replace the system clipboard with plain text."""
    _controller(context).clipboard_write(text)
    return {"written": True, "chars": len(text)}


@tool(category="desktop", risk="read", timeout=130.0, tags=["window", "wait"])
def desktop_wait_window(
    query: str,
    timeout: float = 10.0,
    active: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Wait until a window title or app contains the query."""
    match = _controller(context).wait_for_window(query, timeout=timeout, active=active)
    return {"found": bool(match), "window": match.to_dict() if match else None}


@tool(category="desktop", risk="read", tags=["workflow", "preview", "dry-run"])
def desktop_preview_workflow(
    workflow: str,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Validate and preview a JSON desktop workflow without side effects."""
    parsed = DesktopWorkflow.parse(workflow)
    runner = DesktopWorkflowRunner(
        _controller(context),
        cancelled=(lambda: bool(context and context.agent and context.agent.cancelled)),
    )
    return runner.run(parsed, dry_run=True).to_dict()


@tool(
    category="desktop",
    risk="execute",
    executes_code=True,
    requires_approval=True,
    parallel_safe=False,
    timeout=600.0,
    tags=["workflow", "automation"],
)
def desktop_run_workflow(
    workflow: str,
    dry_run: bool = False,
    stop_on_error: bool = True,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Run a bounded JSON desktop workflow or preview it without side effects.

    Args:
        workflow: JSON object with ``steps`` or a JSON list of action objects
        dry_run: validate and list actions without operating the desktop
        stop_on_error: stop at the first failed step
    """
    parsed = DesktopWorkflow.parse(workflow)
    runner = DesktopWorkflowRunner(
        _controller(context),
        cancelled=(lambda: bool(context and context.agent and context.agent.cancelled)),
    )
    return runner.run(parsed, dry_run=dry_run, stop_on_error=stop_on_error).to_dict()


@tool(category="desktop", risk="write", tags=["queue", "task"])
def desktop_queue_add(
    prompt: str,
    title: str = "",
    priority: int = 50,
    max_attempts: int = 1,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Add an agent prompt to the persistent desktop queue."""
    with DesktopTaskQueue(_queue_path(context), recover=False) as queue:
        task = queue.enqueue(
            prompt,
            title=title,
            priority=priority,
            max_attempts=max_attempts,
            session_id=(context.session_id if context else "") or "desktop",
            metadata={"source": "agent-tool", "run_id": context.run_id if context else ""},
        )
    return task.to_dict()


@tool(category="desktop", risk="read", tags=["queue", "list"])
def desktop_queue_list(
    status: str = "",
    limit: int = 50,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """List queued, running, and completed desktop tasks."""
    with DesktopTaskQueue(_queue_path(context), recover=False) as queue:
        tasks = queue.list(status=status, limit=limit)
        stats = queue.stats()
    return {"tasks": [task.to_dict() for task in tasks], "stats": stats}


@tool(category="desktop", risk="write", requires_approval=True, tags=["queue", "cancel"])
def desktop_queue_cancel(task_id: str, context: ToolContext | None = None) -> dict[str, Any]:
    """Cancel a queued or currently running desktop task."""
    with DesktopTaskQueue(_queue_path(context), recover=False) as queue:
        task = queue.cancel(task_id)
    return {"found": bool(task), "task": task.to_dict() if task else None}


@tool(category="desktop", risk="write", tags=["queue", "retry"])
def desktop_queue_retry(
    task_id: str,
    reset_attempts: bool = False,
    context: ToolContext | None = None,
) -> dict[str, Any]:
    """Return a terminal desktop task to the queue."""
    with DesktopTaskQueue(_queue_path(context), recover=False) as queue:
        task = queue.retry(task_id, reset_attempts=reset_attempts)
    return {"found": bool(task), "task": task.to_dict() if task else None}


@tool(category="desktop", risk="safe", tags=["queue", "stats"])
def desktop_queue_stats(context: ToolContext | None = None) -> dict[str, Any]:
    """Report persistent desktop queue counts and pause state."""
    with DesktopTaskQueue(_queue_path(context), recover=False) as queue:
        return queue.stats()


TOOLS: list[Any] = [
    desktop_capabilities,
    desktop_screen,
    desktop_pointer,
    desktop_move,
    desktop_click,
    desktop_drag,
    desktop_scroll,
    desktop_type,
    desktop_key,
    desktop_hotkey,
    desktop_screenshot,
    desktop_pixel,
    desktop_locate,
    desktop_windows,
    desktop_active_window,
    desktop_focus_window,
    desktop_window_action,
    desktop_move_window,
    desktop_launch,
    desktop_open,
    desktop_clipboard_read,
    desktop_clipboard_write,
    desktop_wait_window,
    desktop_preview_workflow,
    desktop_run_workflow,
    desktop_queue_add,
    desktop_queue_list,
    desktop_queue_cancel,
    desktop_queue_retry,
    desktop_queue_stats,
]
"""Every desktop tool in registration order."""


def load(registry: Any) -> int:
    """Register every desktop tool; return how many were added."""
    registry.register_all([item.tool for item in TOOLS])
    return len(TOOLS)


__all__ = ["TOOLS", "load", *(item.__name__ for item in TOOLS)]
