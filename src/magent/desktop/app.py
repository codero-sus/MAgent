"""Always-on-top MAgent logo and native task/queue/chat control panel.

The UI uses Tk from the Python standard library.  It starts as a draggable
76-pixel MAgent mark that stays above other windows.  A click opens the panel;
tasks, queued work, chat messages, approvals, logo selection, and emergency
controls all live there.  Agent work runs on a background worker so Tk's event
loop never blocks.
"""

from __future__ import annotations

import queue as thread_queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from ..tools.approval import ApprovalDecision, ApprovalManager, ApprovalPolicy
from .autostart import autostart_status, install_autostart, remove_autostart
from .branding import (
    LOGO_THEMES,
    DesktopAppearanceStore,
    draw_logo,
    logo_theme,
)
from .models import DesktopTask, TaskStatus
from .queue import DesktopQueueWorker, DesktopTaskQueue

LOGGER = get_logger("desktop.app")
AgentBuilder = Callable[[ApprovalManager], Any]

BUBBLE_SIZE = 76
"""Pixel size of the always-on-top control."""

POLL_MS = 200
"""UI event and queue refresh cadence."""


def tkinter_status() -> dict[str, Any]:
    """Probe Tk availability without opening a visible window."""
    try:
        import tkinter as tk

        version = str(tk.TkVersion)
    except Exception as exc:
        return {"available": False, "version": "", "error": str(exc)}
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except Exception as exc:
        return {"available": False, "version": version, "error": str(exc)}
    return {"available": True, "version": version, "error": ""}


class DesktopApp:
    """Floating logo, persistent queue worker, and expandable desktop panel."""

    def __init__(
        self,
        config: Any,
        *,
        queue: DesktopTaskQueue | None = None,
        agent_builder: AgentBuilder | None = None,
    ) -> None:
        self.config = config
        self.desktop = config.desktop
        self.agent_builder = agent_builder
        self.task_queue = queue or DesktopTaskQueue(
            self.desktop.queue_path,
            recover=bool(self.desktop.recover_tasks),
        )
        self.appearance_store = DesktopAppearanceStore(self.desktop.state_path)
        state_exists = Path(self.desktop.state_path).expanduser().is_file()
        self.appearance = self.appearance_store.load()
        if not state_exists or self.appearance.logo_theme not in {
            theme.id for theme in LOGO_THEMES
        }:
            self.appearance.logo_theme = self.desktop.logo_theme
        self.appearance.always_on_top = bool(
            self.appearance.always_on_top and self.desktop.always_on_top
        )
        self.events: thread_queue.Queue[tuple[str, DesktopTask | None]] = thread_queue.Queue()
        self.worker = DesktopQueueWorker(
            self.task_queue,
            self._execute_task,
            poll_interval=float(self.desktop.queue_poll_interval),
            on_event=self._worker_event,
            cancel_current=self._cancel_agent,
        )
        self.root: Any = None
        self.panel: Any = None
        self.bubble_canvas: Any = None
        self.panel_logo: Any = None
        self.task_tree: Any = None
        self.queue_tree: Any = None
        self.task_detail: Any = None
        self.chat_transcript: Any = None
        self.chat_prompt: Any = None
        self.queue_prompt: Any = None
        self.queue_title: Any = None
        self.priority_var: Any = None
        self.attempts_var: Any = None
        self.status_var: Any = None
        self.pause_var: Any = None
        self.topmost_var: Any = None
        self.logo_var: Any = None
        self.logo_preview: Any = None
        self._tk: Any = None
        self._ttk: Any = None
        self._agent: Any = None
        self._closing = threading.Event()
        self._dirty = True
        self._selected_task_id = ""
        self._chat_tasks: set[str] = set()
        self._rendered_chat_tasks: set[str] = set()
        self._drag_origin: tuple[int, int, int, int] | None = None
        self._dragged = False

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        """Create Tk windows and block until the operator exits."""
        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
        except Exception as exc:
            raise RuntimeError(
                "Tkinter is required for the desktop panel; install your Python Tk package"
            ) from exc
        self._tk = tk
        self._ttk = ttk
        self._messagebox = messagebox
        try:
            self.root = tk.Tk()
        except Exception as exc:
            raise RuntimeError(f"could not open the desktop display: {exc}") from exc
        self._build_style()
        self._build_bubble()
        self._build_panel()
        self.worker.start()
        self.root.after(POLL_MS, self._poll)
        self.root.mainloop()
        return 0

    def close(self) -> None:
        """Stop active work, save placement, and destroy every window."""
        if self._closing.is_set():
            return
        self._closing.set()
        self.worker.stop(wait=False, cancel=True)
        self._capture_geometry()
        try:
            self.appearance_store.save(self.appearance)
        except OSError as exc:
            LOGGER.debug("could not persist desktop appearance: %s", exc)
        try:
            self.task_queue.close()
        except Exception:
            pass
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass

    # -- bubble ------------------------------------------------------------

    def _build_bubble(self) -> None:
        tk = self._tk
        root = self.root
        root.title("MAgent")
        root.overrideredirect(True)
        root.attributes("-topmost", bool(self.appearance.always_on_top))
        try:
            root.attributes("-alpha", 0.98)
        except Exception:
            pass
        root.configure(bg="#070b14")
        x = self.appearance.bubble_x
        if x < 0:
            x = max(0, root.winfo_screenwidth() - BUBBLE_SIZE - 24)
        y = max(0, self.appearance.bubble_y)
        root.geometry(f"{BUBBLE_SIZE}x{BUBBLE_SIZE}+{x}+{y}")
        canvas = tk.Canvas(
            root,
            width=BUBBLE_SIZE,
            height=BUBBLE_SIZE,
            bg="#070b14",
            highlightthickness=1,
            highlightbackground="#29415f",
            cursor="hand2",
        )
        canvas.pack(fill="both", expand=True)
        self.bubble_canvas = canvas
        self._draw_bubble()
        canvas.bind("<ButtonPress-1>", self._bubble_press)
        canvas.bind("<B1-Motion>", self._bubble_drag)
        canvas.bind("<ButtonRelease-1>", self._bubble_release)
        canvas.bind("<Button-3>", self._bubble_menu)
        root.bind("<Escape>", lambda _event: self._hide_panel())

    def _draw_bubble(self) -> None:
        if self.bubble_canvas is None:
            return
        draw_logo(
            self.bubble_canvas,
            self.appearance.logo_theme,
            BUBBLE_SIZE,
            show_background=True,
        )
        stats = self.task_queue.stats()
        running = stats["counts"][TaskStatus.RUNNING.value]
        queued = stats["counts"][TaskStatus.QUEUED.value]
        color = "#ffbf69" if self.task_queue.paused else "#61e6a7"
        if running:
            color = "#52dcff"
        elif queued:
            color = "#9c7cff"
        self.bubble_canvas.create_oval(56, 56, 70, 70, fill=color, outline="#07101d", width=2)
        if queued:
            self.bubble_canvas.create_text(
                63,
                63,
                text=str(min(queued, 9)) if queued < 10 else "+",
                fill="#07101d",
                font=("TkDefaultFont", 7, "bold"),
            )

    def _bubble_press(self, event: Any) -> None:
        self._drag_origin = (
            int(event.x_root),
            int(event.y_root),
            int(self.root.winfo_x()),
            int(self.root.winfo_y()),
        )
        self._dragged = False

    def _bubble_drag(self, event: Any) -> None:
        if self._drag_origin is None:
            return
        start_x, start_y, window_x, window_y = self._drag_origin
        dx, dy = int(event.x_root) - start_x, int(event.y_root) - start_y
        if abs(dx) + abs(dy) > 4:
            self._dragged = True
        x = max(0, min(self.root.winfo_screenwidth() - BUBBLE_SIZE, window_x + dx))
        y = max(0, min(self.root.winfo_screenheight() - BUBBLE_SIZE, window_y + dy))
        self.root.geometry(f"+{x}+{y}")

    def _bubble_release(self, _event: Any) -> None:
        if self._dragged:
            self._capture_geometry()
            self.appearance_store.save(self.appearance)
        else:
            self.toggle_panel()
        self._drag_origin = None

    def _bubble_menu(self, event: Any) -> None:
        menu = self._tk.Menu(self.root, tearoff=False, bg="#121d30", fg="#f4f8ff")
        menu.add_command(label="Open MAgent", command=self._show_panel)
        menu.add_command(
            label="Pause queue" if not self.task_queue.paused else "Resume queue",
            command=self._toggle_pause,
        )
        menu.add_command(label="Emergency stop", command=self._stop_current)
        menu.add_separator()
        menu.add_command(label="Exit", command=self.close)
        menu.tk_popup(event.x_root, event.y_root)

    # -- panel -------------------------------------------------------------

    def _build_style(self) -> None:
        ttk = self._ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background="#101827", foreground="#edf5ff", fieldbackground="#0c1422")
        style.configure("TFrame", background="#101827")
        style.configure("TLabel", background="#101827", foreground="#edf5ff")
        style.configure("Muted.TLabel", foreground="#91a0b8")
        style.configure(
            "Accent.TButton", background="#52dcff", foreground="#07101d", padding=(11, 7)
        )
        style.map("Accent.TButton", background=[("active", "#82e8ff")])
        style.configure("TButton", background="#1c2a40", foreground="#edf5ff", padding=(9, 6))
        style.configure("TNotebook", background="#0b111d", borderwidth=0)
        style.configure(
            "TNotebook.Tab", background="#121d30", foreground="#a8b5c8", padding=(12, 7)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#20314b")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Treeview",
            background="#0c1422",
            fieldbackground="#0c1422",
            foreground="#dfeaff",
            rowheight=27,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading", background="#19263b", foreground="#91a0b8", relief="flat"
        )
        style.map("Treeview", background=[("selected", "#27466a")])

    def _build_panel(self) -> None:
        tk, ttk = self._tk, self._ttk
        panel = tk.Toplevel(self.root)
        self.panel = panel
        panel.title("MAgent Desktop")
        panel.configure(bg="#0b111d")
        panel.geometry(self.appearance.panel_geometry)
        panel.minsize(430, 560)
        panel.attributes("-topmost", bool(self.appearance.always_on_top))
        panel.protocol("WM_DELETE_WINDOW", self._hide_panel)
        panel.withdraw()

        header = ttk.Frame(panel, padding=(15, 12))
        header.pack(fill="x")
        self.panel_logo = tk.Canvas(header, width=50, height=50, bg="#101827", highlightthickness=0)
        self.panel_logo.pack(side="left", padx=(0, 10))
        draw_logo(self.panel_logo, self.appearance.logo_theme, 50)
        title = ttk.Frame(header)
        title.pack(side="left", fill="x", expand=True)
        ttk.Label(title, text="MAgent", font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
        self.status_var = tk.StringVar(value="Desktop worker starting…")
        ttk.Label(title, textvariable=self.status_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Button(header, text="Stop", command=self._stop_current).pack(side="right", padx=(6, 0))
        self.pause_var = tk.StringVar(value="Pause")
        ttk.Button(header, textvariable=self.pause_var, command=self._toggle_pause).pack(
            side="right"
        )

        notebook = ttk.Notebook(panel)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        tasks = ttk.Frame(notebook, padding=10)
        queue_tab = ttk.Frame(notebook, padding=10)
        chat = ttk.Frame(notebook, padding=10)
        settings = ttk.Frame(notebook, padding=10)
        notebook.add(tasks, text="Tasks")
        notebook.add(queue_tab, text="Queue")
        notebook.add(chat, text="Chat")
        notebook.add(settings, text="Logo & settings")
        self._build_tasks_tab(tasks)
        self._build_queue_tab(queue_tab)
        self._build_chat_tab(chat)
        self._build_settings_tab(settings)

    def _build_tasks_tab(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="All work", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(toolbar, text="Refresh", command=self._refresh_tasks).pack(side="right")
        columns = ("status", "title", "attempts", "priority")
        tree = ttk.Treeview(
            parent, columns=columns, show="headings", selectmode="browse", height=12
        )
        for column, label, width in (
            ("status", "Status", 82),
            ("title", "Task", 240),
            ("attempts", "Try", 45),
            ("priority", "Priority", 55),
        ):
            tree.heading(column, text=label)
            tree.column(column, width=width, minwidth=40, stretch=column == "title")
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", self._select_task)
        self.task_tree = tree
        detail = tk.Text(
            parent,
            height=8,
            wrap="word",
            bg="#0c1422",
            fg="#dfeaff",
            insertbackground="#52dcff",
            relief="flat",
            padx=10,
            pady=8,
        )
        detail.pack(fill="x", pady=(8, 7))
        detail.configure(state="disabled")
        self.task_detail = detail
        buttons = ttk.Frame(parent)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self._cancel_selected).pack(side="left")
        ttk.Button(buttons, text="Pause / resume", command=self._pause_selected).pack(
            side="left", padx=5
        )
        ttk.Button(buttons, text="Retry", command=self._retry_selected).pack(side="left")
        ttk.Button(buttons, text="Delete", command=self._delete_selected).pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear finished", command=self._clear_finished).pack(side="right")

    def _build_queue_tab(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        ttk.Label(parent, text="Add work to the agent", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w"
        )
        self.queue_title = tk.StringVar()
        ttk.Entry(parent, textvariable=self.queue_title).pack(fill="x", pady=(7, 6))
        self.queue_title.set("")
        prompt = tk.Text(
            parent,
            height=7,
            wrap="word",
            bg="#0c1422",
            fg="#edf5ff",
            insertbackground="#52dcff",
            relief="flat",
            padx=10,
            pady=8,
        )
        prompt.pack(fill="x")
        prompt.insert("1.0", "")
        self.queue_prompt = prompt
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=8)
        ttk.Label(controls, text="Priority").pack(side="left")
        self.priority_var = tk.IntVar(value=50)
        ttk.Spinbox(controls, from_=0, to=100, width=5, textvariable=self.priority_var).pack(
            side="left", padx=(5, 12)
        )
        ttk.Label(controls, text="Attempts").pack(side="left")
        self.attempts_var = tk.IntVar(value=1)
        ttk.Spinbox(controls, from_=1, to=10, width=4, textvariable=self.attempts_var).pack(
            side="left", padx=5
        )
        ttk.Button(
            controls, text="Add to queue", style="Accent.TButton", command=self._add_queue_task
        ).pack(side="right")
        ttk.Label(parent, text="Waiting and active", style="Muted.TLabel").pack(
            anchor="w", pady=(7, 5)
        )
        tree = ttk.Treeview(
            parent, columns=("status", "title", "priority"), show="headings", height=10
        )
        for column, label, width in (
            ("status", "Status", 85),
            ("title", "Task", 260),
            ("priority", "Priority", 60),
        ):
            tree.heading(column, text=label)
            tree.column(column, width=width, stretch=column == "title")
        tree.pack(fill="both", expand=True)
        self.queue_tree = tree

    def _build_chat_tab(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        transcript = tk.Text(
            parent,
            wrap="word",
            state="disabled",
            bg="#0c1422",
            fg="#edf5ff",
            insertbackground="#52dcff",
            relief="flat",
            padx=11,
            pady=10,
        )
        transcript.pack(fill="both", expand=True)
        transcript.tag_configure("user", foreground="#74e6ff", font=("TkDefaultFont", 10, "bold"))
        transcript.tag_configure("agent", foreground="#75e7ad", font=("TkDefaultFont", 10, "bold"))
        transcript.tag_configure("error", foreground="#ff7d8b")
        self.chat_transcript = transcript
        prompt = tk.Text(
            parent,
            height=5,
            wrap="word",
            bg="#101b2c",
            fg="#edf5ff",
            insertbackground="#52dcff",
            relief="flat",
            padx=10,
            pady=8,
        )
        prompt.pack(fill="x", pady=(8, 6))
        prompt.bind("<Control-Return>", lambda _event: self._send_chat())
        prompt.bind("<Command-Return>", lambda _event: self._send_chat())
        self.chat_prompt = prompt
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(
            row, text="Chat is queued at priority 100 · Ctrl/Cmd+Enter", style="Muted.TLabel"
        ).pack(side="left")
        ttk.Button(row, text="Send", style="Accent.TButton", command=self._send_chat).pack(
            side="right"
        )

    def _build_settings_tab(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        ttk.Label(
            parent, text="Choose the floating MAgent logo", font=("TkDefaultFont", 11, "bold")
        ).pack(anchor="w")
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, pady=(9, 10))
        choices = ttk.Frame(body)
        choices.pack(side="left", fill="y")
        self.logo_var = tk.StringVar(value=self.appearance.logo_theme)
        for index, theme in enumerate(LOGO_THEMES, start=1):
            ttk.Radiobutton(
                choices,
                text=f"{index}. {theme.name}",
                value=theme.id,
                variable=self.logo_var,
                command=self._select_logo,
            ).pack(anchor="w", pady=3)
        preview_box = ttk.Frame(body)
        preview_box.pack(side="right", fill="both", expand=True, padx=(14, 0))
        self.logo_preview = tk.Canvas(
            preview_box,
            width=180,
            height=180,
            bg="#0b1422",
            highlightthickness=1,
            highlightbackground="#2b4263",
        )
        self.logo_preview.pack(anchor="center", pady=8)
        draw_logo(self.logo_preview, self.appearance.logo_theme, 180)
        self.logo_description = tk.StringVar(
            value=logo_theme(self.appearance.logo_theme).description
        )
        ttk.Label(
            preview_box, textvariable=self.logo_description, style="Muted.TLabel", wraplength=180
        ).pack()

        separator = ttk.Separator(parent)
        separator.pack(fill="x", pady=9)
        self.topmost_var = tk.BooleanVar(value=self.appearance.always_on_top)
        ttk.Checkbutton(
            parent,
            text="Keep logo and panel always on top",
            variable=self.topmost_var,
            command=self._change_topmost,
        ).pack(anchor="w")
        ttk.Label(
            parent,
            text="Move the pointer to the top-left corner for the automation fail-safe.",
            style="Muted.TLabel",
            wraplength=430,
        ).pack(anchor="w", pady=(6, 10))
        startup = ttk.Frame(parent)
        startup.pack(fill="x")
        self.autostart_var = tk.StringVar()
        ttk.Label(startup, textvariable=self.autostart_var, style="Muted.TLabel").pack(side="left")
        ttk.Button(startup, text="Enable at login", command=self._enable_autostart).pack(
            side="right"
        )
        ttk.Button(startup, text="Disable", command=self._disable_autostart).pack(
            side="right", padx=5
        )
        self._refresh_autostart()
        ttk.Button(parent, text="Exit MAgent desktop", command=self.close).pack(
            anchor="e", pady=(18, 0)
        )

    # -- panel actions -----------------------------------------------------

    def toggle_panel(self) -> None:
        """Show or hide the expanded panel."""
        if self.panel.state() == "withdrawn":
            self._show_panel()
        else:
            self._hide_panel()

    def _show_panel(self) -> None:
        self._place_panel()
        self.panel.deiconify()
        self.panel.lift()
        self._refresh_tasks()
        self.panel.focus_force()

    def _hide_panel(self) -> None:
        if self.panel is not None:
            self._capture_geometry()
            self.panel.withdraw()
            self.appearance_store.save(self.appearance)

    def _place_panel(self) -> None:
        self.panel.update_idletasks()
        width = max(430, int(self.desktop.panel_width))
        height = max(560, int(self.desktop.panel_height))
        bubble_x, bubble_y = self.root.winfo_x(), self.root.winfo_y()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = bubble_x - width - 10 if bubble_x > width + 20 else bubble_x + BUBBLE_SIZE + 10
        y = min(max(10, bubble_y), max(10, screen_h - height - 40))
        x = min(max(10, x), max(10, screen_w - width - 10))
        self.panel.geometry(f"{width}x{height}+{x}+{y}")

    def _capture_geometry(self) -> None:
        if self.root is not None:
            self.appearance.bubble_x = int(self.root.winfo_x())
            self.appearance.bubble_y = int(self.root.winfo_y())
        if self.panel is not None and self.panel.state() != "withdrawn":
            geometry = self.panel.geometry().split("+", 1)[0]
            if "x" in geometry:
                self.appearance.panel_geometry = geometry

    def _add_queue_task(self) -> None:
        prompt = self.queue_prompt.get("1.0", "end").strip()
        if not prompt:
            self._messagebox.showinfo(
                "MAgent", "Describe the work to add first.", parent=self.panel
            )
            return
        task = self.task_queue.enqueue(
            prompt,
            title=self.queue_title.get().strip(),
            priority=int(self.priority_var.get()),
            max_attempts=int(self.attempts_var.get()),
            session_id="desktop-queue",
            metadata={"source": "panel"},
        )
        self.queue_prompt.delete("1.0", "end")
        self.queue_title.set("")
        self._selected_task_id = task.id
        self._dirty = True
        self.worker.wake()

    def _send_chat(self) -> str:
        prompt = self.chat_prompt.get("1.0", "end").strip()
        if not prompt:
            return "break"
        task = self.task_queue.enqueue(
            prompt,
            title=f"Chat: {prompt[:60]}",
            priority=100,
            max_attempts=1,
            session_id="desktop-chat",
            metadata={"source": "chat"},
        )
        self._chat_tasks.add(task.id)
        self._append_chat("You", prompt, "user")
        self.chat_prompt.delete("1.0", "end")
        self._append_chat("MAgent", "Queued…", "agent")
        self.worker.wake()
        return "break"

    def _append_chat(self, role: str, text: str, tag: str) -> None:
        widget = self.chat_transcript
        widget.configure(state="normal")
        widget.insert("end", f"{role}\n", tag)
        widget.insert("end", text.strip() + "\n\n", tag if tag == "error" else "")
        widget.configure(state="disabled")
        widget.see("end")

    def _select_task(self, _event: Any = None) -> None:
        selected = self.task_tree.selection()
        if not selected:
            return
        task_id = str(selected[0])
        self._selected_task_id = task_id
        task = self.task_queue.get(task_id)
        if task is None:
            return
        text = (
            f"{task.prompt}\n\n"
            f"Status: {task.status.value} · attempt {task.attempts}/{task.max_attempts}\n"
            f"Result: {task.result or '(none)'}\n"
            f"Error: {task.error or '(none)'}"
        )
        self.task_detail.configure(state="normal")
        self.task_detail.delete("1.0", "end")
        self.task_detail.insert("1.0", text)
        self.task_detail.configure(state="disabled")

    def _cancel_selected(self) -> None:
        task = self.task_queue.get(self._selected_task_id)
        if task is None:
            return
        if self.worker.current is not None and self.worker.current.id == task.id:
            self.worker.cancel_active()
        else:
            self.task_queue.cancel(task.id)
        self._dirty = True

    def _pause_selected(self) -> None:
        task = self.task_queue.get(self._selected_task_id)
        if task is None:
            return
        if task.status is TaskStatus.QUEUED:
            self.task_queue.pause_task(task.id)
        elif task.status is TaskStatus.PAUSED:
            self.task_queue.resume_task(task.id)
            self.worker.wake()
        self._dirty = True

    def _retry_selected(self) -> None:
        if self._selected_task_id:
            self.task_queue.retry(self._selected_task_id, reset_attempts=True)
            self.worker.wake()
            self._dirty = True

    def _delete_selected(self) -> None:
        if self._selected_task_id and self.task_queue.delete(self._selected_task_id):
            self._selected_task_id = ""
            self._dirty = True

    def _clear_finished(self) -> None:
        self.task_queue.clear()
        self._dirty = True

    def _toggle_pause(self) -> None:
        self.task_queue.set_paused(not self.task_queue.paused)
        self.worker.wake()
        self._update_status()

    def _stop_current(self) -> None:
        stopped = self.worker.cancel_active()
        if not stopped:
            self.task_queue.set_paused(True)
        self._update_status()

    # -- worker and approvals ---------------------------------------------

    def _build_agent(self) -> Any:
        policy = ApprovalPolicy(mode=self.config.tools.approval_mode)
        approval = ApprovalManager(policy, prompt=self._request_approval)
        if self.agent_builder is not None:
            return self.agent_builder(approval)
        from ..agent import Agent
        from ..llm.factory import build_backend

        return Agent(
            backend=build_backend(config=self.config),
            settings=self.config,
            approval=approval,
            name="MAgent[desktop]",
        )

    def _execute_task(self, task: DesktopTask) -> str:
        if self._agent is None:
            self._agent = self._build_agent()
        self._agent.reset_cancel()
        result = self._agent.run(task.prompt, session_id=task.session_id)
        if getattr(result, "error", "") and not getattr(result, "output", ""):
            raise RuntimeError(result.error)
        return str(getattr(result, "output", result))

    def _cancel_agent(self) -> None:
        if self._agent is not None:
            self._agent.cancel(reason="stopped from MAgent desktop panel")

    def _request_approval(self, request: Any) -> ApprovalDecision:
        completed = threading.Event()
        answer: dict[str, ApprovalDecision] = {"value": ApprovalDecision.DENY}

        def ask() -> None:
            if self._closing.is_set():
                completed.set()
                return
            text = (
                f"Tool: {request.tool}\nRisk: {request.risk}\n\n"
                f"{request.summary(width=500)}\n\nAllow this desktop action?"
            )
            allowed = self._messagebox.askyesno("MAgent approval required", text, parent=self.panel)
            answer["value"] = ApprovalDecision.APPROVE if allowed else ApprovalDecision.DENY
            completed.set()

        self.root.after(0, ask)
        while not completed.wait(0.1):
            if self._closing.is_set():
                return ApprovalDecision.DENY
        return answer["value"]

    def _worker_event(self, event: str, task: DesktopTask | None) -> None:
        self.events.put((event, task))

    # -- refresh -----------------------------------------------------------

    def _poll(self) -> None:
        if self._closing.is_set():
            return
        while True:
            try:
                event, task = self.events.get_nowait()
            except thread_queue.Empty:
                break
            self._dirty = True
            if (
                task is not None
                and task.id in self._chat_tasks
                and task.id not in self._rendered_chat_tasks
            ):
                if event == "completed":
                    self._append_chat("MAgent", task.result or "Done.", "agent")
                    self._rendered_chat_tasks.add(task.id)
                elif event in {"failed", "cancelled"} and task.status in {
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    self._append_chat("MAgent", task.error or task.status.value, "error")
                    self._rendered_chat_tasks.add(task.id)
        if self._dirty:
            self._refresh_tasks()
            self._dirty = False
        self._update_status()
        self._draw_bubble()
        self.root.after(POLL_MS, self._poll)

    def _refresh_tasks(self) -> None:
        tasks = self.task_queue.list(limit=250)
        if self.task_tree is not None:
            selected = set(self.task_tree.selection())
            self.task_tree.delete(*self.task_tree.get_children())
            for task in tasks:
                self.task_tree.insert(
                    "",
                    "end",
                    iid=task.id,
                    values=(
                        task.status.value,
                        task.title,
                        f"{task.attempts}/{task.max_attempts}",
                        task.priority,
                    ),
                )
            for task_id in selected:
                if self.task_tree.exists(task_id):
                    self.task_tree.selection_add(task_id)
        if self.queue_tree is not None:
            self.queue_tree.delete(*self.queue_tree.get_children())
            for task in reversed(tasks):
                if task.status not in {
                    TaskStatus.QUEUED,
                    TaskStatus.PAUSED,
                    TaskStatus.RUNNING,
                }:
                    continue
                self.queue_tree.insert(
                    "",
                    "end",
                    iid="queue-" + task.id,
                    values=(task.status.value, task.title, task.priority),
                )

    def _update_status(self) -> None:
        stats = self.task_queue.stats()
        counts = stats["counts"]
        if stats["paused"]:
            text = "Queue paused · click Resume to continue"
        elif self.worker.current is not None:
            text = f"Working · {self.worker.current.title}"
        elif counts[TaskStatus.QUEUED.value]:
            text = f"Ready · {counts[TaskStatus.QUEUED.value]} queued"
        else:
            text = "Ready · click the logo anytime"
        if self.status_var is not None:
            self.status_var.set(text)
        if self.pause_var is not None:
            self.pause_var.set("Resume" if stats["paused"] else "Pause")

    # -- appearance and startup -------------------------------------------

    def _select_logo(self) -> None:
        selected = self.logo_var.get()
        theme = logo_theme(selected)
        self.appearance.logo_theme = theme.id
        self.desktop.logo_theme = theme.id
        self.logo_description.set(theme.description)
        draw_logo(self.logo_preview, theme.id, 180)
        draw_logo(self.panel_logo, theme.id, 50)
        self._draw_bubble()
        self.appearance_store.save(self.appearance)

    def _change_topmost(self) -> None:
        enabled = bool(self.topmost_var.get())
        self.appearance.always_on_top = enabled
        self.root.attributes("-topmost", enabled)
        self.panel.attributes("-topmost", enabled)
        self.appearance_store.save(self.appearance)

    def _refresh_autostart(self) -> None:
        status = autostart_status()
        self.autostart_var.set("Starts at login" if status["installed"] else "Not started at login")

    def _enable_autostart(self) -> None:
        try:
            path = install_autostart()
        except OSError as exc:
            self._messagebox.showerror(
                "MAgent", f"Could not enable startup:\n{exc}", parent=self.panel
            )
        else:
            self._messagebox.showinfo(
                "MAgent", f"Startup entry installed:\n{path}", parent=self.panel
            )
        self._refresh_autostart()

    def _disable_autostart(self) -> None:
        remove_autostart()
        self._refresh_autostart()


def launch_desktop_app(config: Any, *, agent_builder: AgentBuilder | None = None) -> int:
    """Construct and run the native desktop application."""
    return DesktopApp(config, agent_builder=agent_builder).run()


__all__ = [
    "BUBBLE_SIZE",
    "POLL_MS",
    "AgentBuilder",
    "DesktopApp",
    "launch_desktop_app",
    "tkinter_status",
]
