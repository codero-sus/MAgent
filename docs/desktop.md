# Desktop automation

MAgent can stay available as a small, draggable, always-on-top logo. Clicking it
opens a native, local-first agent workbench: projects and sessions on the left,
the active conversation in the center, and workspace files, background activity,
and reusable skills on the right. No browser or cloud control plane is required;
model requests still go to Cortex LLMHoster over its external HTTP API.

The workbench takes broad interaction inspiration from project/session agent
interfaces such as LM Studio Bionic while retaining an original MAgent visual
system, implementation, safety model, and data format. No LM Studio source,
assets, branding, or proprietary implementation is copied.

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

## Launcher and workbench

The 76-pixel launcher can be dragged to any screen edge and remains above normal
windows by default. Click it to open or hide the workbench. Its three-column
layout keeps context visible instead of hiding each subsystem behind unrelated
tabs:

- **Projects and sessions:** pinned/recent navigation, rename and archive menus,
  new sessions, and nested sub-sessions.
- **Session:** a durable conversation reconstructed from queue history, an
  explicit skill picker, file attachment, priority, and background send.
- **Project files:** bounded, traversal-safe workspace indexing and text preview.
  Attachments are copied into the project's `attachments/` directory.
- **Activity:** every queued, paused, or running task remains visible even after
  switching to another project or session.
- **Skills:** the twelve bundled skills plus project skills from
  `.magent/skills/`, `skills/`, and `docs/skills/`; type `@skill-name` or use the
  picker to apply one explicitly. Otherwise MAgent matches skills to the task.
- **Tasks:** full project history with retry, pause/resume, cancel, delete, result,
  error, and applied-skill details.
- **Project settings:** per-project Cortex model ID and instructions, logo,
  topmost preference, and start-at-login controls.
- **Stop:** an emergency control that cancels the active agent run before another
  task is claimed.

Project sessions continue in the background when another session is selected.
Desktop automation remains single-consumer by design: two independent tasks must
not race to control one mouse and keyboard.

Panel/launcher placement, active context, topmost preference, and logo choice are
saved in `desktop.state_path`. Durable tasks use `desktop.queue_path`; project and
session metadata use `desktop.projects_path`; ordinary project folders live under
`desktop.workspace_directory` unless a code project points at an existing local
folder. Deleting project metadata never deletes workspace files.

## Projects, sessions, and files

Projects supply a workspace, type, optional Cortex model, and reusable
instructions. Three presets are available:

- `general` for quick mixed work;
- `code` for a repository or source folder (workspace files are included for
  orientation and file/shell tools are sandboxed to that folder);
- `documents` for research, writing, spreadsheets, presentations, and artifacts.

A first-run **Inbox** project is created locally. Every queued workbench task is
stamped with its project and session IDs, workspace, model choice, and explicit
skills. Existing pre-workbench queue databases migrate in place by adding a
`project_id` column; legacy tasks remain readable.

```bash
magent desktop projects create --title "Website refresh" --kind code --workspace .
magent desktop projects list
magent desktop projects show PROJECT_ID
magent desktop projects update PROJECT_ID --instructions "Run focused checks"
magent desktop projects pin PROJECT_ID
magent desktop projects archive PROJECT_ID

magent desktop sessions create --project PROJECT_ID --title "Research"
magent desktop sessions create --project PROJECT_ID --title "Fact check" --parent SESSION_ID
magent desktop sessions list --project PROJECT_ID
magent desktop sessions pin SESSION_ID

magent desktop files attach ./brief.pdf --project PROJECT_ID
magent desktop files list --project PROJECT_ID
magent desktop files preview attachments/brief.txt --project PROJECT_ID
```

Project/session deletion removes only metadata. Queue history and workspace files
are deliberately preserved so a mistaken cleanup cannot destroy user work.

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
GET/POST    /api/desktop/projects
GET/PATCH   /api/desktop/projects/{id}
DELETE      /api/desktop/projects/{id}          # workspace is preserved
GET/POST    /api/desktop/projects/{id}/sessions
GET         /api/desktop/projects/{id}/files    # ?path= for bounded preview
GET/PATCH   /api/desktop/sessions/{id}
DELETE      /api/desktop/sessions/{id}          # task history is preserved
GET         /api/desktop/tasks?project_id=...&session_id=...&status=queued
POST        /api/desktop/tasks
GET         /api/desktop/tasks/{id}
PATCH       /api/desktop/tasks/{id}  {"action":"cancel"|"retry"|"pause"|"resume"}
DELETE      /api/desktop/tasks/{id}
POST        /api/desktop/queue       {"action":"pause"|"resume"|"clear"}
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
panel_width = 1180
panel_height = 760
max_workspace_files = 1000
max_file_preview_bytes = 131072
queue_poll_interval = 0.25
recover_tasks = true
autostart = false
```

Paths default under MAgent's local data directory and can be overridden with
`queue_path`, `projects_path`, `workspace_directory`, `state_path`, and
`screenshot_directory`. Common environment overrides include
`MAGENT_DESKTOP_ENABLED`, `MAGENT_DESKTOP_BACKEND`,
`MAGENT_DESKTOP_FAIL_SAFE`, `MAGENT_DESKTOP_LOGO`, `MAGENT_DESKTOP_PROJECTS`,
`MAGENT_DESKTOP_WORKSPACES`, `MAGENT_DESKTOP_ALLOWED_APPS`,
`MAGENT_DESKTOP_OPEN_SCHEMES`, and `MAGENT_DESKTOP_MAX_WORKFLOW_STEPS`.
