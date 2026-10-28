# Desktop automation

MAgent can stay available as a small, draggable, always-on-top logo. Clicking it
opens a native local panel for tasks, the durable automation queue, agent chat,
status, approvals, and emergency controls. No browser or cloud service is
required, and model requests still go to Cortex LLMHoster over its external HTTP
API.

## Install and launch

The core package remains dependency-free. Native operating-system helpers are
used when available; the optional extra adds the most consistent cross-platform
mouse, keyboard, screenshot, and image-matching backend:

```bash
pip install 'MAgent[desktop]'
magent desktop status
magent desktop
```

Use `magent desktop open --background` to detach it. Install a per-user startup
entry with:

```bash
magent desktop autostart install
magent desktop autostart status
magent desktop autostart remove
```

The startup integration writes only to the current user's standard location:
XDG autostart on Linux, LaunchAgents on macOS, or the Startup folder on Windows.
It never requests administrator privileges.

### Platform permissions

- **macOS:** allow the Python/terminal application under Accessibility and Screen
  Recording when prompted.
- **Windows:** normal user sessions work; a non-elevated process cannot control
  an elevated window.
- **Linux/X11:** PyAutoGUI is recommended. Native fallbacks detect `xdotool`,
  `wmctrl`, `scrot`/`gnome-screenshot`, and `xclip`/`xsel` individually.
- **Linux/Wayland:** compositors intentionally restrict synthetic input. Screen
  and clipboard fallbacks detect `grim` and `wl-copy`/`wl-paste`; input support
  depends on the compositor or an XWayland session.
- **Headless sessions:** the modules, CLI, API, queue, workflow preview, and
  recording driver still work. `magent desktop status` explains which live
  capabilities are unavailable instead of crashing.

Tk is part of many Python distributions but is packaged separately on some
Linux systems (often as `python3-tk`). The status command reports when it is
missing.

## Floating control

The 64-pixel launcher can be dragged to any screen edge and remains above normal
windows by default. Click it to open or hide the panel. The panel provides:

- **Tasks:** persisted history and task details.
- **Queue:** add, pause, resume, retry, cancel, clear, and inspect work.
- **Agent:** chat with the same MAgent/Cortex agent used by the CLI.
- **Status:** live driver capability checks, active work, and queue counts.
- **Settings:** choose a logo, toggle always-on-top, and manage start-at-login.
- **Stop:** an emergency control that cancels the active agent run and stops the
  current worker before another task is claimed.

Panel and launcher placement, topmost preference, and logo choice are saved in
`desktop.state_path`. The work list is stored separately in a SQLite database at
`desktop.queue_path`, so tasks survive closing the panel or restarting MAgent.

## Eight bundled logo pairs

Each concept includes a square launcher icon and a horizontal product lockup:

1. `orbit` — Orbit M (default)
2. `core` — Agent Core
3. `command` — Command Wing
4. `neural` — Neural M
5. `flow` — Infinite Flow
6. `window` — Window Pilot
7. `pulse` — Pulse M
8. `beacon` — Agent Beacon

List and select them without editing configuration:

```bash
magent desktop logos
magent desktop logos beacon
```

The deterministic SVG files and a visual gallery are bundled under
`src/magent/desktop/assets/logos/`. Run
`python scripts/build_desktop_logos.py --check` to verify that generated assets
match their source definitions.

## Queue and task lifecycle

A task moves through explicit states:

```text
queued <-> paused
queued -> running -> completed
                  -> failed -> queued (retry)
queued/paused/running -> cancelled -> queued (manual retry)
```

Only one task may be `running` in the shared SQLite queue, even if a panel and an
API process open it at the same time. Priority is 0–100 (higher first), with FIFO
ordering for equal priorities. Interrupted `running` tasks are recovered to
`queued` when the desktop panel next starts.

```bash
magent desktop queue add "Open the latest report and prepare a concise summary" \
  --priority 80 --max-attempts 2
magent desktop queue list
magent desktop queue show TASK_ID
magent desktop queue pause                  # pause all claiming
magent desktop queue resume
magent desktop queue pause TASK_ID          # pause one queued task
magent desktop queue resume TASK_ID
magent desktop queue cancel TASK_ID
magent desktop queue retry TASK_ID
magent desktop queue delete TASK_ID
magent desktop queue clear                 # terminal tasks only
```

Queue prompts run through a normal MAgent agent. Any desktop-changing tool call
still reaches the panel's approval dialog; queuing a prompt is not blanket
permission to control the machine.

## Desktop tools

The default registry exposes 30 `desktop_*` tools in the `desktop` category:

- capability, monitor, pointer, pixel, screenshot, and image inspection;
- move, click, drag, vertical/horizontal scroll;
- type, key press, and multi-key hotkeys;
- clipboard read/write;
- window list, active/focus, minimize/maximize/restore/close, and move/resize;
- application launch, URL/file open, and wait-for-window;
- bounded JSON workflow execution with a dry-run preview mode;
- durable queue add/list/cancel/retry/status operations.

Discover the exact installed schemas with:

```bash
magent tools --category desktop
magent tools inspect desktop_run_workflow
```

## Declarative workflows

Workflows are JSON, not arbitrary Python. Every step must be one of the explicit
actions understood by `DesktopWorkflowRunner`, and a configurable step ceiling
is enforced before execution. This makes plans inspectable and dry-runnable.

```json
{
  "name": "save-note",
  "variables": {"message": "Prepared locally by MAgent"},
  "steps": [
    {"action": "move", "x": 420, "y": 260, "duration": 0.2},
    {"action": "click"},
    {"action": "type", "text": "${message}", "interval": 0.02},
    {"action": "hotkey", "keys": ["ctrl", "s"]}
  ]
}
```

Supported actions are `move`, `click`, `drag`, `scroll`, `type`, `press`,
`hotkey`, `wait`, `screenshot`, `locate`, `clipboard_read`, `clipboard_write`,
`focus_window`, `window_action`, `move_window`, `wait_window`, `launch`, and
`open`. Friendly aliases such as `move_mouse`, `type_text`, and `launch_app` are
also accepted. `${name}` placeholders resolve from workflow variables or prior
named step outputs.

Preview before execution:

```bash
magent tools call desktop_preview_workflow \
  '{"workflow":"{\"steps\":[{\"action\":\"click\",\"x\":100,\"y\":100}]}"}'
```

`examples/09_desktop_automation.py` runs the workflow in
`examples/desktop_workflow.json` against `RecordingDesktopDriver`, allowing the
full path to be tested without touching the real desktop.

## REST API

`magent serve` exposes the same persistent queue:

```text
GET    /api/desktop
GET    /api/desktop/tasks?status=queued&limit=100
POST   /api/desktop/tasks
GET    /api/desktop/tasks/{id}
PATCH  /api/desktop/tasks/{id}     {"action":"cancel"|"retry"|"pause"|"resume"}
DELETE /api/desktop/tasks/{id}
POST   /api/desktop/queue          {"action":"pause"|"resume"|"clear"}
```

Example:

```bash
curl -X POST http://127.0.0.1:8765/api/desktop/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Prepare the weekly local report","priority":70}'
```

Set `MAGENT_HTTP_API_KEY` before exposing the server beyond loopback. Direct REST
calls to desktop-changing tools are denied by default when no human approver is
available; queue tasks receive interactive approval in the desktop panel.

## Safety model

Desktop automation has several independent controls:

1. tool risk metadata and MAgent's existing `ApprovalManager` policy;
2. explicit `requires_approval` metadata on every desktop-changing tool;
3. per-call approval in the agent loop and a second direct-dispatch guard;
4. capability switches (`allow_mouse`, `allow_keyboard`, `allow_clipboard`,
   `allow_window_control`, `allow_launch`, and `allow_screenshots`);
5. executable and open-target scheme allowlists;
6. coordinate, text, clipboard, wait, hotkey, argument, and workflow limits;
7. PyAutoGUI and controller top-left fail-safe checks;
8. queue pause/cancel controls and the floating emergency stop.

A desktop automation library is **not an operating-system security boundary**.
Run MAgent as a normal user, leave approval mode at `dangerous` or stricter, use
an HTTP API key when appropriate, and do not grant the process permissions it
does not need.

## Configuration

```toml
[desktop]
enabled = true
backend = "auto"                 # auto | pyautogui | native | recording
allow_mouse = true
allow_keyboard = true
allow_screenshots = true
allow_clipboard = true
allow_window_control = true
allow_launch = true
fail_safe = true
panic_corner_size = 4
action_pause = 0.05
max_type_chars = 10000
max_clipboard_chars = 100000
max_wait_seconds = 120.0
max_workflow_steps = 100
allowed_apps = []                # empty means any executable
allowed_open_schemes = ["file", "http", "https", "mailto"]
logo_theme = "orbit"
always_on_top = true
panel_width = 500
panel_height = 720
queue_poll_interval = 0.25
recover_tasks = true
autostart = false
```

Paths default under MAgent's local data directory and can be overridden with
`queue_path`, `state_path`, and `screenshot_directory`. Common environment
overrides include `MAGENT_DESKTOP_ENABLED`, `MAGENT_DESKTOP_BACKEND`,
`MAGENT_DESKTOP_FAIL_SAFE`, `MAGENT_DESKTOP_LOGO`,
`MAGENT_DESKTOP_ALLOWED_APPS`, `MAGENT_DESKTOP_OPEN_SCHEMES`, and
`MAGENT_DESKTOP_MAX_WORKFLOW_STEPS`.
