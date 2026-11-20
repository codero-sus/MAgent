"""Always-available MAgent launcher and project-oriented desktop workbench.

The interface is an original MAgent design informed by modern agent workspaces:
projects and sessions stay visible on the left, the active conversation occupies
the center, and files/background activity/skills remain visible on the right.
Changing sessions never stops queued work.  All durable state is local SQLite
or ordinary workspace files; Cortex remains an external HTTP model host.

Tk is imported only by :meth:`DesktopApp.run`, so project, queue, and workbench
logic remains usable and testable on headless machines.
"""

from __future__ import annotations

import inspect
import queue as thread_queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from ..tools.approval import ApprovalDecision, ApprovalManager, ApprovalPolicy
from .autostart import autostart_status, install_autostart, remove_autostart
from .branding import LOGO_THEMES, DesktopAppearanceStore, draw_logo, logo_theme
from .models import DesktopTask, TaskStatus
from .projects import DesktopProjectStore, ProjectKind
from .queue import DesktopQueueWorker, DesktopTaskQueue
from .workbench import DesktopWorkbench

LOGGER = get_logger("desktop.app")
AgentBuilder = Callable[..., Any]

BUBBLE_SIZE = 76
"""Pixel size of the draggable always-on-top launcher."""

POLL_MS = 250
"""UI event and queue refresh cadence."""

COLORS = {
    "window": "#080b11",
    "panel": "#0d1118",
    "surface": "#121821",
    "surface2": "#171f2b",
    "surface3": "#1d2735",
    "border": "#273345",
    "text": "#f1f5fa",
    "muted": "#8d9aae",
    "accent": "#55d6be",
    "accent2": "#73a7ff",
    "purple": "#a991ff",
    "danger": "#ff7185",
    "warning": "#f2be67",
}


def tkinter_status() -> dict[str, Any]:
    """Probe Tk availability without leaving a visible window."""
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
    """Floating launcher, local workbench, and persistent background worker."""

    def __init__(
        self,
        config: Any,
        *,
        queue: DesktopTaskQueue | None = None,
        projects: DesktopProjectStore | None = None,
        agent_builder: AgentBuilder | None = None,
    ) -> None:
        self.config = config
        self.desktop = config.desktop
        self.agent_builder = agent_builder
        self._owns_queue = queue is None
        self._owns_projects = projects is None
        self.task_queue = queue or DesktopTaskQueue(
            self.desktop.queue_path,
            recover=bool(self.desktop.recover_tasks),
        )
        self.project_store = projects or DesktopProjectStore(
            self.desktop.projects_path,
            workspace_root=self.desktop.workspace_directory,
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
        self.workbench = DesktopWorkbench(
            self.project_store,
            self.task_queue,
            project_id=self.appearance.project_id,
            session_id=self.appearance.session_id,
        )
        self._persist_selection()

        self.events: thread_queue.Queue[tuple[str, DesktopTask | None]] = thread_queue.Queue()
        self.worker = DesktopQueueWorker(
            self.task_queue,
            self._execute_task,
            poll_interval=float(self.desktop.queue_poll_interval),
            on_event=self._worker_event,
            cancel_current=self._cancel_agent,
        )

        # Tk objects are populated only by run().
        self.root: Any = None
        self.panel: Any = None
        self.bubble_canvas: Any = None
        self.panel_logo: Any = None
        self.project_tree: Any = None
        self.session_tree: Any = None
        self.transcript: Any = None
        self.chat_prompt: Any = None
        self.skill_var: Any = None
        self.priority_var: Any = None
        self.file_tree: Any = None
        self.file_preview: Any = None
        self.activity_tree: Any = None
        self.skills_list: Any = None
        self.task_tree: Any = None
        self.task_detail: Any = None
        self.project_title_var: Any = None
        self.session_title_var: Any = None
        self.workspace_var: Any = None
        self.status_var: Any = None
        self.model_status_var: Any = None
        self.pause_var: Any = None
        self.topmost_var: Any = None
        self.logo_var: Any = None
        self.logo_preview: Any = None
        self.logo_description: Any = None
        self.autostart_var: Any = None
        self.project_model_var: Any = None
        self.project_model_entry: Any = None
        self.project_instructions: Any = None
        self.center_notebook: Any = None
        self._tk: Any = None
        self._ttk: Any = None
        self._messagebox: Any = None
        self._filedialog: Any = None
        self._simpledialog: Any = None

        self._agents: dict[str, Any] = {}
        self._active_agent: Any = None
        self._skills: Any = None
        self._closing = threading.Event()
        self._dirty = True
        self._selected_task_id = ""
        self._selected_activity_id = ""
        self._transcript_signature: tuple[Any, ...] = ()
        self._drag_origin: tuple[int, int, int, int] | None = None
        self._dragged = False

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        """Create both Tk windows and block until the operator exits."""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, simpledialog, ttk
        except Exception as exc:
            raise RuntimeError(
                "Tkinter is required for the desktop workbench; install your Python Tk package"
            ) from exc
        self._tk = tk
        self._ttk = ttk
        self._messagebox = messagebox
        self._filedialog = filedialog
        self._simpledialog = simpledialog
        try:
            self.root = tk.Tk()
        except Exception as exc:
            raise RuntimeError(f"could not open the desktop display: {exc}") from exc
        self._build_style()
        self._build_bubble()
        self._build_panel()
        self._reload_skills()
        self._refresh_all()
        self.worker.start()
        self.root.after(POLL_MS, self._poll)
        self.root.mainloop()
        return 0

    def close(self) -> None:
        """Stop active work, save state, and release local stores."""
        if self._closing.is_set():
            return
        self._closing.set()
        self.worker.stop(wait=False, cancel=True)
        self._capture_geometry()
        self._persist_selection()
        try:
            self.appearance_store.save(self.appearance)
        except OSError as exc:
            LOGGER.debug("could not persist desktop appearance: %s", exc)
        if self._owns_queue:
            try:
                self.task_queue.close()
            except Exception:
                pass
        if self._owns_projects:
            try:
                self.project_store.close()
            except Exception:
                pass
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass

    # -- floating launcher -------------------------------------------------

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
        root.configure(bg=COLORS["window"])
        x = self.appearance.bubble_x
        if x < 0:
            x = max(0, root.winfo_screenwidth() - BUBBLE_SIZE - 24)
        y = max(0, self.appearance.bubble_y)
        root.geometry(f"{BUBBLE_SIZE}x{BUBBLE_SIZE}+{x}+{y}")
        canvas = tk.Canvas(
            root,
            width=BUBBLE_SIZE,
            height=BUBBLE_SIZE,
            bg=COLORS["window"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
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
        color = COLORS["warning"] if self.task_queue.paused else COLORS["accent"]
        if running:
            color = COLORS["accent2"]
        elif queued:
            color = COLORS["purple"]
        self.bubble_canvas.create_oval(55, 55, 70, 70, fill=color, outline="#07101d", width=2)
        if queued:
            self.bubble_canvas.create_text(
                62.5,
                62.5,
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
        menu = self._tk.Menu(
            self.root,
            tearoff=False,
            bg=COLORS["surface2"],
            fg=COLORS["text"],
            activebackground=COLORS["surface3"],
            activeforeground=COLORS["text"],
        )
        menu.add_command(label="Open workbench", command=self._show_panel)
        menu.add_command(label="New session", command=self._new_session)
        menu.add_command(
            label="Pause background work" if not self.task_queue.paused else "Resume work",
            command=self._toggle_pause,
        )
        menu.add_command(label="Emergency stop", command=self._stop_current)
        menu.add_separator()
        menu.add_command(label="Exit MAgent", command=self.close)
        menu.tk_popup(event.x_root, event.y_root)

    # -- visual system and shell ------------------------------------------

    def _build_style(self) -> None:
        ttk = self._ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            ".",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["surface"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
        )
        style.configure("TFrame", background=COLORS["panel"])
        style.configure("Surface.TFrame", background=COLORS["surface"])
        style.configure("Raised.TFrame", background=COLORS["surface2"])
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Surface.TLabel", background=COLORS["surface"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"])
        style.configure(
            "Eyebrow.TLabel",
            foreground=COLORS["muted"],
            font=("TkDefaultFont", 8, "bold"),
        )
        style.configure(
            "Pill.TLabel",
            background=COLORS["surface3"],
            foreground=COLORS["accent"],
            padding=(8, 4),
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#071410",
            padding=(13, 8),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#7be7d1")])
        style.configure(
            "TButton",
            background=COLORS["surface3"],
            foreground=COLORS["text"],
            padding=(9, 6),
            borderwidth=0,
        )
        style.map("TButton", background=[("active", "#293648")])
        style.configure(
            "Danger.TButton",
            background="#40202a",
            foreground="#ffbdc7",
            padding=(9, 6),
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#592733")])
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0, tabmargins=0)
        style.configure(
            "TNotebook.Tab",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            padding=(13, 8),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["surface2"])],
            foreground=[("selected", COLORS["text"])],
        )
        for name in ("Treeview", "Nav.Treeview"):
            style.configure(
                name,
                background=COLORS["surface"],
                fieldbackground=COLORS["surface"],
                foreground=COLORS["text"],
                rowheight=28,
                borderwidth=0,
            )
            style.map(
                name,
                background=[("selected", "#25364a")],
                foreground=[("selected", "#ffffff")],
            )
        style.configure(
            "Treeview.Heading",
            background=COLORS["surface2"],
            foreground=COLORS["muted"],
            relief="flat",
            padding=(7, 6),
        )
        style.configure(
            "TEntry",
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            insertcolor=COLORS["accent"],
            padding=7,
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["surface"],
            background=COLORS["surface2"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["muted"],
            padding=5,
        )
        style.configure("TPanedwindow", background=COLORS["border"], sashwidth=1)
        style.configure("TSeparator", background=COLORS["border"])

    def _build_panel(self) -> None:
        tk, ttk = self._tk, self._ttk
        panel = tk.Toplevel(self.root)
        self.panel = panel
        panel.title("MAgent Workbench")
        panel.configure(bg=COLORS["window"])
        panel.geometry(self.appearance.panel_geometry)
        panel.minsize(900, 620)
        panel.attributes("-topmost", bool(self.appearance.always_on_top))
        panel.protocol("WM_DELETE_WINDOW", self._hide_panel)
        panel.withdraw()

        self._build_header(panel)
        body = ttk.Panedwindow(panel, orient="horizontal")
        body.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        left = ttk.Frame(body, style="Surface.TFrame", width=230, padding=(12, 12))
        center = ttk.Frame(body, padding=(12, 10))
        right = ttk.Frame(body, style="Surface.TFrame", width=290, padding=(10, 10))
        body.add(left, weight=0)
        body.add(center, weight=1)
        body.add(right, weight=0)
        self._build_navigation(left)
        self._build_center(center)
        self._build_inspector(right)

    def _build_header(self, parent: Any) -> None:
        tk, ttk = self._tk, self._ttk
        header = ttk.Frame(parent, style="Raised.TFrame", padding=(14, 10))
        header.pack(fill="x", padx=1, pady=1)
        self.panel_logo = tk.Canvas(
            header,
            width=42,
            height=42,
            bg=COLORS["surface2"],
            highlightthickness=0,
        )
        self.panel_logo.pack(side="left", padx=(0, 10))
        draw_logo(self.panel_logo, self.appearance.logo_theme, 42)

        brand = ttk.Frame(header, style="Raised.TFrame")
        brand.pack(side="left")
        ttk.Label(
            brand,
            text="MAgent",
            style="Surface.TLabel",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            brand,
            text="LOCAL AGENT WORKBENCH",
            style="Surface.TLabel",
            foreground=COLORS["muted"],
            font=("TkDefaultFont", 7, "bold"),
        ).pack(anchor="w")

        ttk.Separator(header, orient="vertical").pack(side="left", fill="y", padx=14)
        context = ttk.Frame(header, style="Raised.TFrame")
        context.pack(side="left", fill="x", expand=True)
        self.project_title_var = tk.StringVar()
        self.session_title_var = tk.StringVar()
        ttk.Label(
            context,
            textvariable=self.project_title_var,
            style="Surface.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            context,
            textvariable=self.session_title_var,
            style="Surface.TLabel",
            foreground=COLORS["muted"],
        ).pack(anchor="w")

        status = ttk.Frame(header, style="Raised.TFrame")
        status.pack(side="right")
        self.model_status_var = tk.StringVar()
        ttk.Label(status, textvariable=self.model_status_var, style="Pill.TLabel").pack(
            side="left", padx=(0, 8)
        )
        self.status_var = tk.StringVar(value="Starting…")
        ttk.Label(
            status,
            textvariable=self.status_var,
            style="Surface.TLabel",
            foreground=COLORS["muted"],
        ).pack(side="left", padx=(0, 9))
        self.pause_var = tk.StringVar(value="Pause")
        ttk.Button(status, textvariable=self.pause_var, command=self._toggle_pause).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(status, text="Stop", style="Danger.TButton", command=self._stop_current).pack(
            side="left"
        )

    # -- left navigation ---------------------------------------------------

    def _build_navigation(self, parent: Any) -> None:
        ttk = self._ttk
        projects_header = ttk.Frame(parent, style="Surface.TFrame")
        projects_header.pack(fill="x")
        ttk.Label(projects_header, text="PROJECTS", style="Eyebrow.TLabel").pack(side="left")
        ttk.Button(projects_header, text="+", width=3, command=self._new_project).pack(side="right")
        project_tree = ttk.Treeview(
            parent,
            show="tree",
            selectmode="browse",
            height=8,
            style="Nav.Treeview",
        )
        project_tree.column("#0", width=190, minwidth=120, stretch=True)
        project_tree.pack(fill="x", pady=(7, 13))
        project_tree.bind("<<TreeviewSelect>>", self._select_project)
        project_tree.bind("<Button-3>", self._project_menu)
        self.project_tree = project_tree

        session_header = ttk.Frame(parent, style="Surface.TFrame")
        session_header.pack(fill="x")
        ttk.Label(session_header, text="SESSIONS", style="Eyebrow.TLabel").pack(side="left")
        ttk.Button(session_header, text="+", width=3, command=self._new_session).pack(side="right")
        session_tree = ttk.Treeview(
            parent,
            show="tree",
            selectmode="browse",
            style="Nav.Treeview",
        )
        session_tree.column("#0", width=190, minwidth=120, stretch=True)
        session_tree.pack(fill="both", expand=True, pady=(7, 10))
        session_tree.bind("<<TreeviewSelect>>", self._select_session)
        session_tree.bind("<Button-3>", self._session_menu)
        self.session_tree = session_tree

        ttk.Separator(parent).pack(fill="x", pady=(3, 9))
        local = ttk.Frame(parent, style="Surface.TFrame")
        local.pack(fill="x")
        ttk.Label(local, text="●", style="Surface.TLabel", foreground=COLORS["accent"]).pack(
            side="left"
        )
        ttk.Label(
            local,
            text="  Local project state",
            style="Surface.TLabel",
            foreground=COLORS["muted"],
        ).pack(side="left")

    # -- center workspace --------------------------------------------------

    def _build_center(self, parent: Any) -> None:
        ttk = self._ttk
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        chat = ttk.Frame(notebook, padding=(1, 8))
        tasks = ttk.Frame(notebook, padding=(1, 8))
        settings = ttk.Frame(notebook, padding=(1, 8))
        notebook.add(chat, text="Session")
        notebook.add(tasks, text="Tasks")
        notebook.add(settings, text="Project settings")
        self.center_notebook = notebook
        self._build_chat(chat)
        self._build_tasks(tasks)
        self._build_project_settings(settings)

    def _build_chat(self, parent: Any) -> None:
        tk, ttk = self._tk, self._ttk
        context = ttk.Frame(parent)
        context.pack(fill="x", pady=(0, 8))
        self.workspace_var = tk.StringVar()
        ttk.Label(context, textvariable=self.workspace_var, style="Muted.TLabel").pack(side="left")
        ttk.Button(context, text="New session", command=self._new_session).pack(side="right")

        transcript = tk.Text(
            parent,
            wrap="word",
            state="disabled",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            selectbackground="#29405a",
            relief="flat",
            borderwidth=0,
            padx=17,
            pady=14,
            spacing1=2,
            spacing3=6,
        )
        transcript.pack(fill="both", expand=True)
        transcript.tag_configure(
            "role_user",
            foreground=COLORS["accent2"],
            font=("TkDefaultFont", 8, "bold"),
            spacing1=12,
        )
        transcript.tag_configure(
            "role_agent",
            foreground=COLORS["accent"],
            font=("TkDefaultFont", 8, "bold"),
            spacing1=12,
        )
        transcript.tag_configure("body", foreground=COLORS["text"], lmargin2=8)
        transcript.tag_configure("meta", foreground=COLORS["muted"], font=("TkDefaultFont", 8))
        transcript.tag_configure("error", foreground=COLORS["danger"])
        transcript.tag_configure(
            "welcome_title",
            foreground=COLORS["text"],
            font=("TkDefaultFont", 16, "bold"),
            justify="center",
            spacing1=80,
        )
        transcript.tag_configure(
            "welcome_body", foreground=COLORS["muted"], justify="center", spacing1=8
        )
        self.transcript = transcript

        composer = ttk.Frame(parent, style="Raised.TFrame", padding=(10, 9))
        composer.pack(fill="x", pady=(8, 0))
        prompt = tk.Text(
            composer,
            height=4,
            wrap="word",
            bg=COLORS["surface2"],
            fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            selectbackground="#29405a",
            relief="flat",
            borderwidth=0,
            padx=5,
            pady=4,
        )
        prompt.pack(fill="x")
        prompt.bind("<Control-Return>", lambda _event: self._send_chat())
        prompt.bind("<Command-Return>", lambda _event: self._send_chat())
        self.chat_prompt = prompt
        row = ttk.Frame(composer, style="Raised.TFrame")
        row.pack(fill="x", pady=(7, 0))
        self.skill_var = tk.StringVar(value="Auto skills")
        self.skill_picker = ttk.Combobox(
            row,
            textvariable=self.skill_var,
            state="readonly",
            width=23,
            values=("Auto skills",),
        )
        self.skill_picker.pack(side="left")
        ttk.Button(row, text="Attach", command=self._attach_file).pack(side="left", padx=5)
        self.priority_var = tk.StringVar(value="Normal · 70")
        ttk.Combobox(
            row,
            textvariable=self.priority_var,
            state="readonly",
            width=14,
            values=("Now · 100", "High · 85", "Normal · 70", "Later · 40"),
        ).pack(side="left")
        ttk.Label(
            row,
            text="Runs in background · Ctrl/Cmd+Enter",
            style="Surface.TLabel",
            foreground=COLORS["muted"],
        ).pack(side="left", padx=9)
        ttk.Button(row, text="Send", style="Accent.TButton", command=self._send_chat).pack(
            side="right"
        )

    def _build_tasks(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(
            toolbar,
            text="Project task history",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side="left")
        ttk.Button(toolbar, text="Clear finished", command=self._clear_finished).pack(side="right")
        columns = ("status", "session", "title", "attempts")
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        for column, label, width in (
            ("status", "Status", 88),
            ("session", "Session", 130),
            ("title", "Task", 330),
            ("attempts", "Try", 48),
        ):
            tree.heading(column, text=label)
            tree.column(column, width=width, minwidth=45, stretch=column == "title")
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", self._select_task)
        self.task_tree = tree
        detail = tk.Text(
            parent,
            height=8,
            wrap="word",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["accent"],
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

    def _build_project_settings(self, parent: Any) -> None:
        ttk, tk = self._ttk, self._tk
        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True, padx=8, pady=5)
        ttk.Label(
            content,
            text="Project behavior",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            content,
            text="Instructions and model choice apply only to this project.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 12))
        ttk.Label(content, text="Cortex model ID", style="Eyebrow.TLabel").pack(anchor="w")
        self.project_model_var = tk.StringVar()
        self.project_model_entry = ttk.Entry(content, textvariable=self.project_model_var)
        self.project_model_entry.pack(fill="x", pady=(5, 12))
        ttk.Label(content, text="PROJECT INSTRUCTIONS", style="Eyebrow.TLabel").pack(anchor="w")
        instructions = tk.Text(
            content,
            height=9,
            wrap="word",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            relief="flat",
            padx=10,
            pady=8,
        )
        instructions.pack(fill="x", pady=(5, 8))
        self.project_instructions = instructions
        ttk.Button(
            content,
            text="Save project settings",
            style="Accent.TButton",
            command=self._save_project_settings,
        ).pack(anchor="e")

        ttk.Separator(content).pack(fill="x", pady=18)
        ttk.Label(
            content,
            text="Workbench appearance",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        logo_row = ttk.Frame(content)
        logo_row.pack(fill="x", pady=(9, 8))
        self.logo_var = tk.StringVar(value=self.appearance.logo_theme)
        ttk.Combobox(
            logo_row,
            textvariable=self.logo_var,
            values=tuple(theme.id for theme in LOGO_THEMES),
            state="readonly",
            width=20,
        ).pack(side="left")
        ttk.Button(logo_row, text="Apply logo", command=self._select_logo).pack(side="left", padx=6)
        self.logo_preview = tk.Canvas(
            logo_row,
            width=70,
            height=70,
            bg=COLORS["panel"],
            highlightthickness=0,
        )
        self.logo_preview.pack(side="right")
        draw_logo(self.logo_preview, self.appearance.logo_theme, 70)
        self.logo_description = tk.StringVar(
            value=logo_theme(self.appearance.logo_theme).description
        )
        ttk.Label(content, textvariable=self.logo_description, style="Muted.TLabel").pack(
            anchor="w"
        )
        self.topmost_var = tk.BooleanVar(value=self.appearance.always_on_top)
        ttk.Checkbutton(
            content,
            text="Keep launcher and workbench always on top",
            variable=self.topmost_var,
            command=self._change_topmost,
        ).pack(anchor="w", pady=(12, 5))
        startup = ttk.Frame(content)
        startup.pack(fill="x", pady=(7, 0))
        self.autostart_var = tk.StringVar()
        ttk.Label(startup, textvariable=self.autostart_var, style="Muted.TLabel").pack(side="left")
        ttk.Button(startup, text="Enable at login", command=self._enable_autostart).pack(
            side="right"
        )
        ttk.Button(startup, text="Disable", command=self._disable_autostart).pack(
            side="right", padx=5
        )
        self._refresh_autostart()

    # -- right inspector ---------------------------------------------------

    def _build_inspector(self, parent: Any) -> None:
        ttk = self._ttk
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        files = ttk.Frame(notebook, style="Surface.TFrame", padding=(3, 8))
        activity = ttk.Frame(notebook, style="Surface.TFrame", padding=(3, 8))
        skills = ttk.Frame(notebook, style="Surface.TFrame", padding=(3, 8))
        notebook.add(files, text="Files")
        notebook.add(activity, text="Activity")
        notebook.add(skills, text="Skills")
        self._build_files(files)
        self._build_activity(activity)
        self._build_skills(skills)

    def _build_files(self, parent: Any) -> None:
        tk, ttk = self._tk, self._ttk
        header = ttk.Frame(parent, style="Surface.TFrame")
        header.pack(fill="x", pady=(0, 7))
        ttk.Label(header, text="PROJECT FILES", style="Eyebrow.TLabel").pack(side="left")
        ttk.Button(header, text="+", width=3, command=self._attach_file).pack(side="right")
        ttk.Button(header, text="↻", width=3, command=self._refresh_files).pack(
            side="right", padx=4
        )
        tree = ttk.Treeview(
            parent,
            columns=("size",),
            show="tree headings",
            selectmode="browse",
            height=12,
            style="Nav.Treeview",
        )
        tree.heading("#0", text="Path")
        tree.heading("size", text="Size")
        tree.column("#0", width=180, stretch=True)
        tree.column("size", width=55, anchor="e", stretch=False)
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", self._preview_file)
        self.file_tree = tree
        preview = tk.Text(
            parent,
            height=11,
            wrap="none",
            state="disabled",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            relief="flat",
            padx=8,
            pady=7,
            font=("TkFixedFont", 9),
        )
        preview.pack(fill="x", pady=(8, 0))
        self.file_preview = preview

    def _build_activity(self, parent: Any) -> None:
        ttk = self._ttk
        ttk.Label(parent, text="BACKGROUND WORK", style="Eyebrow.TLabel").pack(
            anchor="w", pady=(0, 7)
        )
        tree = ttk.Treeview(
            parent,
            columns=("status",),
            show="tree headings",
            selectmode="browse",
            style="Nav.Treeview",
        )
        tree.heading("#0", text="Task")
        tree.heading("status", text="State")
        tree.column("#0", width=175, stretch=True)
        tree.column("status", width=72, stretch=False)
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", self._select_activity)
        self.activity_tree = tree
        buttons = ttk.Frame(parent, style="Surface.TFrame")
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Pause / resume", command=self._pause_activity).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self._cancel_activity).pack(side="right")

    def _build_skills(self, parent: Any) -> None:
        tk, ttk = self._tk, self._ttk
        ttk.Label(parent, text="REUSABLE SKILLS", style="Eyebrow.TLabel").pack(
            anchor="w", pady=(0, 3)
        )
        ttk.Label(
            parent,
            text="Double-click a skill to apply it to the composer.",
            style="Surface.TLabel",
            foreground=COLORS["muted"],
            wraplength=240,
        ).pack(anchor="w", pady=(0, 8))
        skills = tk.Listbox(
            parent,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            selectbackground="#25364a",
            selectforeground="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        skills.pack(fill="both", expand=True)
        skills.bind("<Double-Button-1>", self._choose_skill_from_list)
        self.skills_list = skills

    # -- window placement --------------------------------------------------

    def toggle_panel(self) -> None:
        """Show or hide the expanded workbench."""
        if self.panel.state() == "withdrawn":
            self._show_panel()
        else:
            self._hide_panel()

    def _show_panel(self) -> None:
        self._place_panel()
        self.panel.deiconify()
        self.panel.lift()
        self._refresh_all()
        self.panel.focus_force()

    def _hide_panel(self) -> None:
        if self.panel is not None:
            self._capture_geometry()
            self._persist_selection()
            self.panel.withdraw()
            self.appearance_store.save(self.appearance)

    def _place_panel(self) -> None:
        self.panel.update_idletasks()
        width = max(900, int(self.desktop.panel_width))
        height = max(620, int(self.desktop.panel_height))
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = min(width, max(900, screen_w - 24))
        height = min(height, max(620, screen_h - 48))
        bubble_x, bubble_y = self.root.winfo_x(), self.root.winfo_y()
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

    def _persist_selection(self) -> None:
        self.appearance.project_id = self.workbench.project_id
        self.appearance.session_id = self.workbench.session_id

    # -- project and session actions --------------------------------------

    def _new_project(self) -> None:
        name = self._simpledialog.askstring(
            "New MAgent project",
            "Project name",
            parent=self.panel,
        )
        if not name:
            return
        kind = self._simpledialog.askstring(
            "Project type",
            "Type: general, code, or documents",
            initialvalue="general",
            parent=self.panel,
        )
        if kind is None:
            return
        kind = kind.strip().lower()
        if kind not in {item.value for item in ProjectKind}:
            self._messagebox.showerror(
                "MAgent", "Project type must be general, code, or documents.", parent=self.panel
            )
            return
        workspace = ""
        if kind == ProjectKind.CODE.value:
            workspace = self._filedialog.askdirectory(
                title="Choose a code workspace (Cancel for a managed folder)",
                parent=self.panel,
            )
        try:
            self.workbench.new_project(name, kind=kind, workspace=workspace)
        except (OSError, ValueError) as exc:
            self._messagebox.showerror("MAgent", str(exc), parent=self.panel)
            return
        self._context_changed()

    def _new_session(self) -> None:
        if self._simpledialog is None:
            return
        title = self._simpledialog.askstring(
            "New MAgent session",
            "Session title",
            initialvalue="New session",
            parent=self.panel,
        )
        if title is None:
            return
        self.workbench.new_session(title.strip() or "New session")
        self._context_changed()

    def _select_project(self, _event: Any = None) -> None:
        selected = self.project_tree.selection()
        if not selected:
            return
        project_id = str(selected[0])
        if project_id == self.workbench.project_id:
            return
        try:
            self.workbench.select_project(project_id)
        except (KeyError, ValueError):
            return
        self._context_changed()

    def _select_session(self, _event: Any = None) -> None:
        selected = self.session_tree.selection()
        if not selected:
            return
        session_id = str(selected[0])
        if session_id == self.workbench.session_id:
            return
        try:
            self.workbench.select_session(session_id)
        except (KeyError, ValueError):
            return
        self._context_changed(refresh_projects=False)

    def _context_changed(self, *, refresh_projects: bool = True) -> None:
        self._persist_selection()
        self.appearance_store.save(self.appearance)
        self._transcript_signature = ()
        self._reload_skills()
        if refresh_projects:
            self._refresh_projects()
        self._refresh_sessions()
        self._refresh_context()
        self._refresh_files()
        self._refresh_tasks()
        self._refresh_activity()
        self._render_transcript()

    def _project_menu(self, event: Any) -> None:
        row = self.project_tree.identify_row(event.y)
        if not row:
            return
        self.project_tree.selection_set(row)
        project = self.project_store.get_project(row)
        if project is None:
            return
        menu = self._tk.Menu(
            self.panel,
            tearoff=False,
            bg=COLORS["surface2"],
            fg=COLORS["text"],
            activebackground=COLORS["surface3"],
            activeforeground=COLORS["text"],
        )
        menu.add_command(
            label="Unpin" if project.pinned else "Pin",
            command=lambda: self._set_project_state(row, pinned=not project.pinned),
        )
        menu.add_command(label="Rename", command=lambda: self._rename_project(row))
        menu.add_separator()
        menu.add_command(label="Archive", command=lambda: self._archive_project(row))
        menu.tk_popup(event.x_root, event.y_root)

    def _session_menu(self, event: Any) -> None:
        row = self.session_tree.identify_row(event.y)
        if not row:
            return
        self.session_tree.selection_set(row)
        session = self.project_store.get_session(row)
        if session is None:
            return
        menu = self._tk.Menu(
            self.panel,
            tearoff=False,
            bg=COLORS["surface2"],
            fg=COLORS["text"],
            activebackground=COLORS["surface3"],
            activeforeground=COLORS["text"],
        )
        menu.add_command(
            label="Unpin" if session.pinned else "Pin",
            command=lambda: self._set_session_state(row, pinned=not session.pinned),
        )
        menu.add_command(label="Rename", command=lambda: self._rename_session(row))
        menu.add_command(
            label="Create sub-session",
            command=lambda: self._new_subsession(row),
        )
        menu.add_separator()
        menu.add_command(label="Archive", command=lambda: self._archive_session(row))
        menu.tk_popup(event.x_root, event.y_root)

    def _set_project_state(self, project_id: str, **changes: Any) -> None:
        self.project_store.update_project(project_id, **changes)
        self._refresh_projects()

    def _set_session_state(self, session_id: str, **changes: Any) -> None:
        self.project_store.update_session(session_id, **changes)
        self._refresh_sessions()

    def _rename_project(self, project_id: str) -> None:
        project = self.project_store.get_project(project_id)
        if project is None:
            return
        value = self._simpledialog.askstring(
            "Rename project", "Project name", initialvalue=project.name, parent=self.panel
        )
        if value:
            self.project_store.update_project(project_id, name=value)
            self._refresh_projects()
            self._refresh_context()

    def _rename_session(self, session_id: str) -> None:
        session = self.project_store.get_session(session_id)
        if session is None:
            return
        value = self._simpledialog.askstring(
            "Rename session", "Session title", initialvalue=session.title, parent=self.panel
        )
        if value:
            self.project_store.update_session(session_id, title=value)
            self._refresh_sessions()
            self._refresh_context()

    def _new_subsession(self, parent_id: str) -> None:
        title = self._simpledialog.askstring(
            "New sub-session", "Sub-session title", parent=self.panel
        )
        if title:
            self.workbench.new_session(title, parent_id=parent_id)
            self._context_changed(refresh_projects=False)

    def _archive_project(self, project_id: str) -> None:
        projects = self.project_store.list_projects()
        if len(projects) <= 1:
            self._messagebox.showinfo(
                "MAgent", "Create another project before archiving this one.", parent=self.panel
            )
            return
        self.project_store.update_project(project_id, archived=True)
        if project_id == self.workbench.project_id:
            replacement = next(item for item in projects if item.id != project_id)
            self.workbench.select_project(replacement.id)
        self._context_changed()

    def _archive_session(self, session_id: str) -> None:
        sessions = self.project_store.list_sessions(self.workbench.project_id)
        if len(sessions) <= 1:
            self._messagebox.showinfo(
                "MAgent", "Create another session before archiving this one.", parent=self.panel
            )
            return
        self.project_store.update_session(session_id, archived=True)
        if session_id == self.workbench.session_id:
            replacement = next(item for item in sessions if item.id != session_id)
            self.workbench.select_session(replacement.id)
        self._context_changed(refresh_projects=False)

    # -- composer and task actions ----------------------------------------

    def _send_chat(self) -> str:
        prompt = self.chat_prompt.get("1.0", "end").strip()
        if not prompt:
            return "break"
        selected = self.skill_var.get().strip()
        skills = [] if selected == "Auto skills" else [selected]
        priority = {
            "Now · 100": 100,
            "High · 85": 85,
            "Normal · 70": 70,
            "Later · 40": 40,
        }.get(self.priority_var.get(), 70)
        try:
            task = self.workbench.enqueue(prompt, priority=priority, skills=skills)
        except ValueError as exc:
            self._messagebox.showerror("MAgent", str(exc), parent=self.panel)
            return "break"
        self.chat_prompt.delete("1.0", "end")
        self._selected_task_id = task.id
        self._dirty = True
        self.worker.wake()
        self._render_transcript()
        self._refresh_sessions()
        return "break"

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
            f"Project: {task.project_id or '(legacy)'}\nSession: {task.session_id}\n"
            f"Skills: {', '.join(task.metadata.get('skills', [])) or 'automatic'}\n\n"
            f"Result:\n{task.result or '(none)'}\n\n"
            f"Error:\n{task.error or '(none)'}"
        )
        self._replace_text(self.task_detail, text)

    def _cancel_selected(self) -> None:
        self._cancel_task(self._selected_task_id)

    def _pause_selected(self) -> None:
        self._pause_task(self._selected_task_id)

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

    def _select_activity(self, _event: Any = None) -> None:
        selected = self.activity_tree.selection()
        self._selected_activity_id = str(selected[0]) if selected else ""

    def _pause_activity(self) -> None:
        self._pause_task(self._selected_activity_id)

    def _cancel_activity(self) -> None:
        self._cancel_task(self._selected_activity_id)

    def _pause_task(self, task_id: str) -> None:
        task = self.task_queue.get(task_id)
        if task is None:
            return
        if task.status is TaskStatus.QUEUED:
            self.task_queue.pause_task(task.id)
        elif task.status is TaskStatus.PAUSED:
            self.task_queue.resume_task(task.id)
            self.worker.wake()
        self._dirty = True

    def _cancel_task(self, task_id: str) -> None:
        task = self.task_queue.get(task_id)
        if task is None:
            return
        if self.worker.current is not None and self.worker.current.id == task.id:
            self.worker.cancel_active()
        else:
            self.task_queue.cancel(task.id)
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

    # -- files and skills --------------------------------------------------

    def _attach_file(self) -> None:
        path = self._filedialog.askopenfilename(title="Attach a project file", parent=self.panel)
        if not path:
            return
        try:
            entry = self.project_store.attach_file(self.workbench.project_id, path)
        except (OSError, ValueError) as exc:
            self._messagebox.showerror("MAgent", str(exc), parent=self.panel)
            return
        self._refresh_files()
        if self.file_tree.exists(entry.path):
            self.file_tree.selection_set(entry.path)
            self.file_tree.see(entry.path)
            self._preview_file()

    def _preview_file(self, _event: Any = None) -> None:
        selected = self.file_tree.selection()
        if not selected:
            return
        try:
            preview = self.project_store.preview_file(
                self.workbench.project_id,
                str(selected[0]),
                max_bytes=int(self.desktop.max_file_preview_bytes),
            )
        except (OSError, ValueError) as exc:
            self._replace_text(self.file_preview, f"Preview unavailable: {exc}")
            return
        if preview["binary"]:
            text = f"Binary file · {preview['size']} bytes\n{preview['path']}"
        else:
            text = preview["content"]
            if preview["truncated"]:
                text += "\n\n… preview truncated …"
        self._replace_text(self.file_preview, text)

    def _reload_skills(self) -> None:
        try:
            from ..skills import build_library

            self._skills = build_library(self.config, workspace=self.workbench.project.workspace)
        except Exception as exc:
            LOGGER.debug("could not load workbench skills: %s", exc)
            try:
                from ..skills import SkillLibrary

                self._skills = SkillLibrary()
            except Exception:
                self._skills = None
        self._refresh_skills()

    def _refresh_skills(self) -> None:
        names = self._skills.names() if self._skills is not None else []
        if getattr(self, "skill_picker", None) is not None:
            self.skill_picker.configure(values=("Auto skills", *names))
            if self.skill_var.get() not in {"Auto skills", *names}:
                self.skill_var.set("Auto skills")
        if self.skills_list is not None:
            self.skills_list.delete(0, "end")
            for name in names:
                try:
                    skill = self._skills.get(name)
                    label = f"{name}\n  {skill.description}"
                except Exception:
                    label = name
                self.skills_list.insert("end", label)

    def _choose_skill_from_list(self, _event: Any = None) -> None:
        selected = self.skills_list.curselection()
        if not selected or self._skills is None:
            return
        names = self._skills.names()
        index = int(selected[0])
        if index < len(names):
            self.skill_var.set(names[index])
            self.chat_prompt.focus_set()

    # -- project settings and appearance ----------------------------------

    def _save_project_settings(self) -> None:
        project = self.workbench.project
        instructions = self.project_instructions.get("1.0", "end").strip()
        self.project_store.update_project(
            project.id,
            model=self.project_model_var.get().strip(),
            instructions=instructions,
        )
        self._refresh_context()
        self._messagebox.showinfo("MAgent", "Project settings saved.", parent=self.panel)

    def _select_logo(self) -> None:
        selected = self.logo_var.get()
        theme = logo_theme(selected)
        self.appearance.logo_theme = theme.id
        self.desktop.logo_theme = theme.id
        self.logo_description.set(theme.description)
        draw_logo(self.logo_preview, theme.id, 70)
        draw_logo(self.panel_logo, theme.id, 42)
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
        if self.autostart_var is not None:
            self.autostart_var.set(
                "Starts at login" if status["installed"] else "Not started at login"
            )

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

    # -- worker and agent context -----------------------------------------

    def _agent_for_workspace(self, workspace: str) -> Any:
        key = str(Path(workspace).expanduser().resolve())
        agent = self._agents.get(key)
        if agent is not None:
            self._active_agent = agent
            return agent
        policy = ApprovalPolicy(mode=self.config.tools.approval_mode)
        approval = ApprovalManager(policy, prompt=self._request_approval)
        if self.agent_builder is not None:
            try:
                signature = inspect.signature(self.agent_builder)
                accepts_workspace = len(signature.parameters) >= 2 or any(
                    parameter.kind is inspect.Parameter.VAR_POSITIONAL
                    for parameter in signature.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_workspace = False
            agent = (
                self.agent_builder(approval, key)
                if accepts_workspace
                else self.agent_builder(approval)
            )
        else:
            from ..agent import Agent
            from ..llm.factory import build_backend

            agent = Agent(
                backend=build_backend(config=self.config),
                settings=self.config,
                approval=approval,
                workspace=key,
                name="MAgent[desktop]",
            )
        self._agents[key] = agent
        self._active_agent = agent
        return agent

    def _execute_task(self, task: DesktopTask) -> str:
        project = self.project_store.get_project(
            task.project_id or str(task.metadata.get("project_id", ""))
        )
        workspace = (
            project.workspace
            if project is not None
            else str(task.metadata.get("workspace") or self.config.tools.workspace)
        )
        agent = self._agent_for_workspace(workspace)
        agent.reset_cancel()
        if project is not None:
            try:
                from ..skills import build_library

                skills = build_library(self.config, workspace=project.workspace)
            except Exception:
                skills = self._skills
        else:
            skills = self._skills
        context_block = self.workbench.context_block(task, skills)
        model = str(task.metadata.get("model", ""))
        result = agent.run(
            task.prompt,
            session_id=task.session_id,
            context_block=context_block,
            model=model,
            include_files=bool(project is not None and project.kind is ProjectKind.CODE),
            metadata={
                "desktop_project_id": task.project_id,
                "desktop_task_id": task.id,
                "skills": list(task.metadata.get("skills", [])),
            },
        )
        self.project_store.touch_session(task.session_id, task_id=task.id)
        if getattr(result, "error", "") and not getattr(result, "output", ""):
            raise RuntimeError(result.error)
        return str(getattr(result, "output", result))

    def _cancel_agent(self) -> None:
        if self._active_agent is not None:
            self._active_agent.cancel(reason="stopped from MAgent desktop workbench")

    def _request_approval(self, request: Any) -> ApprovalDecision:
        completed = threading.Event()
        answer: dict[str, ApprovalDecision] = {"value": ApprovalDecision.DENY}

        def ask() -> None:
            if self._closing.is_set():
                completed.set()
                return
            if self.panel.state() == "withdrawn":
                self._show_panel()
            self.panel.lift()
            text = (
                f"Tool: {request.tool}\nRisk: {request.risk}\n\n"
                f"{request.summary(width=500)}\n\nAllow this action in the active project?"
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
                self.events.get_nowait()
            except thread_queue.Empty:
                break
            self._dirty = True
        if self._dirty:
            self._refresh_all()
            self._dirty = False
        self._update_status()
        self._draw_bubble()
        self.root.after(POLL_MS, self._poll)

    def _refresh_all(self) -> None:
        self._refresh_projects()
        self._refresh_sessions()
        self._refresh_context()
        self._refresh_tasks()
        self._refresh_activity()
        self._refresh_files()
        self._render_transcript()
        self._update_status()

    def _refresh_projects(self) -> None:
        if self.project_tree is None:
            return
        self.project_tree.delete(*self.project_tree.get_children())
        for project in self.project_store.list_projects(limit=250):
            prefix = "★ " if project.pinned else ""
            kind = {"code": "⌘", "documents": "▤", "general": "◆"}.get(project.kind.value, "◆")
            self.project_tree.insert(
                "", "end", iid=project.id, text=f"{prefix}{kind}  {project.name}"
            )
        if self.project_tree.exists(self.workbench.project_id):
            self.project_tree.selection_set(self.workbench.project_id)
            self.project_tree.see(self.workbench.project_id)

    def _refresh_sessions(self) -> None:
        if self.session_tree is None:
            return
        self.session_tree.delete(*self.session_tree.get_children())
        for session in self.project_store.list_sessions(self.workbench.project_id, limit=300):
            prefix = "★ " if session.pinned else ""
            branch = "↳ " if session.parent_id else ""
            active = self.task_queue.stats(session_id=session.id)["counts"][
                TaskStatus.RUNNING.value
            ]
            suffix = "  ●" if active else ""
            self.session_tree.insert(
                "", "end", iid=session.id, text=f"{prefix}{branch}{session.title}{suffix}"
            )
        if self.session_tree.exists(self.workbench.session_id):
            self.session_tree.selection_set(self.workbench.session_id)
            self.session_tree.see(self.workbench.session_id)

    def _refresh_context(self) -> None:
        project = self.workbench.project
        session = self.workbench.session
        if self.project_title_var is not None:
            self.project_title_var.set(project.name)
        if self.session_title_var is not None:
            self.session_title_var.set(f"{session.title}  ·  {project.kind.value.title()} project")
        if self.workspace_var is not None:
            self.workspace_var.set(f"Workspace  ·  {project.workspace}")
        model = session.model or project.model or self.config.model.model or "Cortex auto"
        if self.model_status_var is not None:
            self.model_status_var.set(f"Cortex  ·  {model}")
        focused = self.panel.focus_get() if self.panel is not None else None
        if self.project_model_var is not None and focused is not self.project_model_entry:
            self.project_model_var.set(project.model)
        if self.project_instructions is not None and focused is not self.project_instructions:
            self.project_instructions.delete("1.0", "end")
            self.project_instructions.insert("1.0", project.instructions)

    def _render_transcript(self) -> None:
        if self.transcript is None:
            return
        entries = self.workbench.transcript(limit=300)
        signature = tuple(
            (item.task_id, item.role, item.status, len(item.text), hash(item.text))
            for item in entries
        )
        if signature == self._transcript_signature:
            return
        self._transcript_signature = signature
        widget = self.transcript
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        if not entries:
            project = self.workbench.project
            widget.insert("end", f"Start work in {project.name}\n", "welcome_title")
            widget.insert(
                "end",
                "Ask MAgent to research, write, code, organize files, or automate the desktop.\n"
                "Work keeps running when you switch projects or sessions.",
                "welcome_body",
            )
        else:
            for item in entries:
                role = "YOU" if item.role == "user" else "MAGENT"
                role_tag = "role_user" if item.role == "user" else "role_agent"
                widget.insert("end", role + "\n", role_tag)
                widget.insert("end", item.text.strip() + "\n", "error" if item.error else "body")
                widget.insert("end", item.status.upper() + "\n", "meta")
        widget.configure(state="disabled")
        widget.see("end")

    def _refresh_tasks(self) -> None:
        if self.task_tree is None:
            return
        tasks = self.task_queue.list(project_id=self.workbench.project_id, limit=500)
        sessions = {
            session.id: session.title
            for session in self.project_store.list_sessions(
                self.workbench.project_id, include_archived=True, limit=1000
            )
        }
        selected = set(self.task_tree.selection())
        self.task_tree.delete(*self.task_tree.get_children())
        for task in tasks:
            self.task_tree.insert(
                "",
                "end",
                iid=task.id,
                values=(
                    task.status.value,
                    sessions.get(task.session_id, task.session_id[:12]),
                    task.title,
                    f"{task.attempts}/{task.max_attempts}",
                ),
            )
        for task_id in selected:
            if self.task_tree.exists(task_id):
                self.task_tree.selection_add(task_id)

    def _refresh_activity(self) -> None:
        if self.activity_tree is None:
            return
        self.activity_tree.delete(*self.activity_tree.get_children())
        for task in self.workbench.activity(limit=300):
            self.activity_tree.insert(
                "", "end", iid=task.id, text=task.title, values=(task.status.value,)
            )
        if self._selected_activity_id and self.activity_tree.exists(self._selected_activity_id):
            self.activity_tree.selection_set(self._selected_activity_id)

    def _refresh_files(self) -> None:
        if self.file_tree is None:
            return
        selected = set(self.file_tree.selection())
        self.file_tree.delete(*self.file_tree.get_children())
        try:
            entries = self.project_store.list_files(
                self.workbench.project_id,
                limit=int(self.desktop.max_workspace_files),
            )
        except (OSError, KeyError):
            entries = []
        for entry in entries:
            self.file_tree.insert(
                "", "end", iid=entry.path, text=entry.path, values=(_human_bytes(entry.size),)
            )
        for path in selected:
            if self.file_tree.exists(path):
                self.file_tree.selection_add(path)

    def _update_status(self) -> None:
        stats = self.task_queue.stats()
        counts = stats["counts"]
        if stats["paused"]:
            text = "Paused"
        elif self.worker.current is not None:
            text = "Working"
        elif counts[TaskStatus.QUEUED.value]:
            text = f"{counts[TaskStatus.QUEUED.value]} queued"
        else:
            text = "Ready"
        if self.status_var is not None:
            self.status_var.set(text)
        if self.pause_var is not None:
            self.pause_var.set("Resume" if stats["paused"] else "Pause")

    @staticmethod
    def _replace_text(widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def _human_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def launch_desktop_app(config: Any, *, agent_builder: AgentBuilder | None = None) -> int:
    """Construct and run the native desktop workbench."""
    return DesktopApp(config, agent_builder=agent_builder).run()


__all__ = [
    "BUBBLE_SIZE",
    "POLL_MS",
    "AgentBuilder",
    "DesktopApp",
    "launch_desktop_app",
    "tkinter_status",
]
