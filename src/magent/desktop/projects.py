"""Local-first project, session, and workspace persistence for the desktop app.

The workbench groups durable queue items into projects and sessions.  Project
metadata lives in SQLite while the actual workspace remains an ordinary folder
on disk, so users can inspect, back up, and edit their files without MAgent.
No model provider owns this state: Cortex is still only the external HTTP model
host used when a queued prompt runs.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..types import new_id
from ..utils.text import slugify


class ProjectKind(StrEnum):
    """Workspace presets exposed by the desktop workbench."""

    GENERAL = "general"
    CODE = "code"
    DOCUMENTS = "documents"


@dataclass(slots=True)
class DesktopProject:
    """A durable unit of work with its own folder and instructions."""

    name: str
    id: str = field(default_factory=lambda: new_id("project"))
    kind: ProjectKind = ProjectKind.GENERAL
    workspace: str = ""
    instructions: str = ""
    model: str = ""
    pinned: bool = False
    archived: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()[:160]
        if not self.name:
            raise ValueError("project name must not be empty")
        if not isinstance(self.kind, ProjectKind):
            self.kind = ProjectKind(str(self.kind))
        self.workspace = str(self.workspace).strip()
        self.instructions = str(self.instructions).strip()[:50_000]
        self.model = str(self.model).strip()[:300]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["managed_workspace"] = bool(self.metadata.get("managed_workspace", False))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DesktopProject:
        """Build a project while ignoring derived or future fields."""
        known = {
            "id",
            "name",
            "kind",
            "workspace",
            "instructions",
            "model",
            "pinned",
            "archived",
            "created_at",
            "updated_at",
            "metadata",
        }
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass(slots=True)
class DesktopSession:
    """One conversation/background-work stream inside a project."""

    project_id: str
    title: str = "New session"
    id: str = field(default_factory=lambda: new_id("session"))
    model: str = ""
    pinned: bool = False
    archived: bool = False
    parent_id: str = ""
    last_task_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.project_id = str(self.project_id).strip()
        if not self.project_id:
            raise ValueError("session project_id must not be empty")
        self.title = str(self.title).strip()[:160] or "New session"
        self.model = str(self.model).strip()[:300]
        self.parent_id = str(self.parent_id).strip()
        self.last_task_id = str(self.last_task_id).strip()
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DesktopSession:
        """Build a session while ignoring future fields."""
        known = {
            "id",
            "project_id",
            "title",
            "model",
            "pinned",
            "archived",
            "parent_id",
            "last_task_id",
            "created_at",
            "updated_at",
            "metadata",
        }
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass(slots=True, frozen=True)
class WorkspaceEntry:
    """One bounded file-browser result inside a project workspace."""

    path: str
    name: str
    kind: str
    size: int = 0
    modified_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return asdict(self)


_PROJECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS desktop_projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    workspace TEXT NOT NULL,
    instructions TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_desktop_projects_list
    ON desktop_projects(archived, pinned DESC, updated_at DESC);
CREATE TABLE IF NOT EXISTS desktop_project_sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES desktop_projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT NOT NULL DEFAULT '',
    last_task_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_desktop_sessions_project
    ON desktop_project_sessions(project_id, archived, pinned DESC, updated_at DESC);
"""


class DesktopProjectStore:
    """Thread-safe SQLite store for desktop projects and their sessions."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if str(path) != ":memory:" else Path(":memory:")
        if workspace_root is None:
            base = self.path.parent if str(self.path) != ":memory:" else Path.cwd()
            workspace_root = base / "projects"
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.executescript(_PROJECT_SCHEMA)
            if str(self.path) != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.commit()

    def close(self) -> None:
        """Close the SQLite connection once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    def __enter__(self) -> DesktopProjectStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- projects ---------------------------------------------------------

    def create_project(
        self,
        name: str,
        *,
        kind: str | ProjectKind = ProjectKind.GENERAL,
        workspace: str | Path = "",
        instructions: str = "",
        model: str = "",
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DesktopProject:
        """Create a project and its folder, returning the persisted record."""
        normalized_kind = ProjectKind(str(getattr(kind, "value", kind)))
        project_id = new_id("project")
        details = dict(metadata or {})
        if str(workspace).strip():
            root = Path(workspace).expanduser().resolve()
            if not root.exists() or not root.is_dir():
                raise ValueError(f"project workspace is not a directory: {root}")
            details.setdefault("managed_workspace", False)
        else:
            label = slugify(str(name))[:48] or "project"
            root = (self.workspace_root / f"{label}-{project_id[-6:]}").resolve()
            root.mkdir(parents=True, exist_ok=True)
            details["managed_workspace"] = True
        project = DesktopProject(
            id=project_id,
            name=name,
            kind=normalized_kind,
            workspace=str(root),
            instructions=instructions,
            model=model,
            pinned=pinned,
            metadata=details,
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO desktop_projects (
                    id,name,kind,workspace,instructions,model,pinned,archived,
                    created_at,updated_at,metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                self._project_values(project),
            )
            self._connection.commit()
        return project

    def ensure_default(self) -> DesktopProject:
        """Return the newest active project or create a local Inbox project."""
        projects = self.list_projects(limit=1)
        if projects:
            return projects[0]
        project = self.create_project(
            "Inbox",
            kind=ProjectKind.GENERAL,
            instructions=(
                "Use this project for quick tasks. Keep generated files in the project workspace."
            ),
            pinned=True,
            metadata={"default": True},
        )
        self.create_session(project.id, title="Welcome")
        return project

    def get_project(self, project_id: str) -> DesktopProject | None:
        """Return one project by id."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM desktop_projects WHERE id=?", (str(project_id),)
            ).fetchone()
        return self._project_from_row(row) if row is not None else None

    def require_project(self, project_id: str) -> DesktopProject:
        """Return a project or raise ``KeyError``."""
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(f"unknown desktop project {project_id!r}")
        return project

    def list_projects(
        self,
        *,
        include_archived: bool = False,
        query: str = "",
        limit: int = 100,
    ) -> list[DesktopProject]:
        """List projects with pinned projects first."""
        clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            clauses.append("archived=0")
        if query.strip():
            clauses.append("LOWER(name) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        sql = "SELECT * FROM desktop_projects"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY pinned DESC, updated_at DESC, name ASC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._project_from_row(row) for row in rows]

    def update_project(self, project_id: str, **changes: Any) -> DesktopProject | None:
        """Edit mutable project fields using a validated read/replace write."""
        project = self.get_project(project_id)
        if project is None:
            return None
        for name in ("name", "instructions", "model", "pinned", "archived"):
            if name in changes:
                setattr(project, name, changes[name])
        if "kind" in changes:
            project.kind = ProjectKind(str(getattr(changes["kind"], "value", changes["kind"])))
        if "workspace" in changes:
            root = Path(str(changes["workspace"])).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"project workspace is not a directory: {root}")
            project.workspace = str(root)
            project.metadata["managed_workspace"] = False
        project.__post_init__()
        project.updated_at = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE desktop_projects
                   SET name=?,kind=?,workspace=?,instructions=?,model=?,pinned=?,archived=?,
                       updated_at=?,metadata=?
                 WHERE id=?
                """,
                (
                    project.name,
                    project.kind.value,
                    project.workspace,
                    project.instructions,
                    project.model,
                    int(project.pinned),
                    int(project.archived),
                    project.updated_at,
                    self._json(project.metadata),
                    project.id,
                ),
            )
            self._connection.commit()
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete metadata and sessions, but never delete workspace files."""
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM desktop_projects WHERE id=?", (str(project_id),)
            )
            self._connection.commit()
        return bool(cursor.rowcount)

    # -- sessions ---------------------------------------------------------

    def create_session(
        self,
        project_id: str,
        *,
        title: str = "New session",
        model: str = "",
        pinned: bool = False,
        parent_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DesktopSession:
        """Create a background-capable session inside a project."""
        self.require_project(project_id)
        if parent_id:
            parent = self.get_session(parent_id)
            if parent is None or parent.project_id != project_id:
                raise ValueError("parent session must belong to the same project")
        session = DesktopSession(
            project_id=project_id,
            title=title,
            model=model,
            pinned=pinned,
            parent_id=parent_id,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO desktop_project_sessions (
                    id,project_id,title,model,pinned,archived,parent_id,last_task_id,
                    created_at,updated_at,metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                self._session_values(session),
            )
            self._connection.execute(
                "UPDATE desktop_projects SET updated_at=? WHERE id=?",
                (session.updated_at, project_id),
            )
            self._connection.commit()
        return session

    def ensure_session(self, project_id: str) -> DesktopSession:
        """Return the newest active session, creating one if needed."""
        sessions = self.list_sessions(project_id, limit=1)
        return sessions[0] if sessions else self.create_session(project_id)

    def get_session(self, session_id: str) -> DesktopSession | None:
        """Return one project session by id."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM desktop_project_sessions WHERE id=?", (str(session_id),)
            ).fetchone()
        return self._session_from_row(row) if row is not None else None

    def list_sessions(
        self,
        project_id: str,
        *,
        include_archived: bool = False,
        limit: int = 200,
    ) -> list[DesktopSession]:
        """List sessions for a project with pinned and recent work first."""
        sql = "SELECT * FROM desktop_project_sessions WHERE project_id=?"
        params: list[Any] = [str(project_id)]
        if not include_archived:
            sql += " AND archived=0"
        sql += " ORDER BY pinned DESC, updated_at DESC, created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._session_from_row(row) for row in rows]

    def update_session(self, session_id: str, **changes: Any) -> DesktopSession | None:
        """Edit session title/model/state metadata."""
        session = self.get_session(session_id)
        if session is None:
            return None
        for name in ("title", "model", "pinned", "archived", "last_task_id"):
            if name in changes:
                setattr(session, name, changes[name])
        session.__post_init__()
        session.updated_at = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE desktop_project_sessions
                   SET title=?,model=?,pinned=?,archived=?,last_task_id=?,updated_at=?,metadata=?
                 WHERE id=?
                """,
                (
                    session.title,
                    session.model,
                    int(session.pinned),
                    int(session.archived),
                    session.last_task_id,
                    session.updated_at,
                    self._json(session.metadata),
                    session.id,
                ),
            )
            self._connection.execute(
                "UPDATE desktop_projects SET updated_at=? WHERE id=?",
                (session.updated_at, session.project_id),
            )
            self._connection.commit()
        return session

    def touch_session(self, session_id: str, *, task_id: str = "") -> DesktopSession | None:
        """Record activity without changing user-facing session fields."""
        return self.update_session(session_id, **({"last_task_id": task_id} if task_id else {}))

    def delete_session(self, session_id: str) -> bool:
        """Delete one session's desktop metadata; queue history remains."""
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM desktop_project_sessions WHERE id=?", (str(session_id),)
            )
            self._connection.commit()
        return bool(cursor.rowcount)

    # -- workspace files --------------------------------------------------

    def list_files(
        self,
        project_id: str,
        *,
        query: str = "",
        limit: int = 500,
        max_depth: int = 6,
        include_hidden: bool = False,
    ) -> list[WorkspaceEntry]:
        """Return a bounded, symlink-safe recursive workspace listing."""
        project = self.require_project(project_id)
        root = Path(project.workspace).expanduser().resolve()
        if not root.is_dir():
            return []
        needle = query.strip().lower()
        bounded = max(1, min(int(limit), 5000))
        depth_limit = max(0, min(int(max_depth), 20))
        entries: list[WorkspaceEntry] = []
        for current, directories, files in os.walk(root, followlinks=False):
            base = Path(current)
            try:
                depth = len(base.relative_to(root).parts)
            except ValueError:
                continue
            directories[:] = sorted(
                item
                for item in directories
                if (include_hidden or not item.startswith("."))
                and item not in {"__pycache__", "node_modules", ".git"}
                and not (base / item).is_symlink()
            )
            if depth >= depth_limit:
                directories[:] = []
            for name in sorted(files):
                if not include_hidden and name.startswith("."):
                    continue
                path = base / name
                if path.is_symlink():
                    continue
                try:
                    relative = path.relative_to(root).as_posix()
                    stat = path.stat()
                except (OSError, ValueError):
                    continue
                if needle and needle not in relative.lower():
                    continue
                entries.append(
                    WorkspaceEntry(
                        path=relative,
                        name=name,
                        kind="file",
                        size=int(stat.st_size),
                        modified_at=float(stat.st_mtime),
                    )
                )
                if len(entries) >= bounded:
                    return entries
        return entries

    def preview_file(
        self,
        project_id: str,
        relative_path: str,
        *,
        max_bytes: int = 128 * 1024,
    ) -> dict[str, Any]:
        """Read a bounded text preview without allowing path traversal."""
        target, relative = self._resolve_project_file(project_id, relative_path)
        if not target.is_file():
            raise ValueError("workspace entry is not a file")
        limit = max(256, min(int(max_bytes), 2 * 1024 * 1024))
        with target.open("rb") as handle:
            payload = handle.read(limit + 1)
        truncated = len(payload) > limit
        payload = payload[:limit]
        binary = b"\x00" in payload
        text = "" if binary else payload.decode("utf-8", errors="replace")
        return {
            "path": relative,
            "size": target.stat().st_size,
            "modified_at": target.stat().st_mtime,
            "binary": binary,
            "truncated": truncated,
            "content": text,
        }

    def attach_file(
        self,
        project_id: str,
        source: str | Path,
        *,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> WorkspaceEntry:
        """Copy a user-selected file into the project's ``attachments`` folder."""
        project = self.require_project(project_id)
        origin = Path(source).expanduser().resolve()
        if not origin.is_file():
            raise ValueError(f"attachment is not a file: {origin}")
        size = origin.stat().st_size
        if size > max(1, int(max_bytes)):
            raise ValueError(f"attachment exceeds the {max_bytes}-byte limit")
        root = Path(project.workspace).expanduser().resolve()
        attachment_dir = root / "attachments"
        attachment_dir.mkdir(parents=True, exist_ok=True)
        destination = attachment_dir / origin.name
        stem, suffix = destination.stem, destination.suffix
        counter = 2
        while destination.exists():
            destination = attachment_dir / f"{stem}-{counter}{suffix}"
            counter += 1
        shutil.copy2(origin, destination)
        stat = destination.stat()
        return WorkspaceEntry(
            path=destination.relative_to(root).as_posix(),
            name=destination.name,
            kind="file",
            size=stat.st_size,
            modified_at=stat.st_mtime,
        )

    # -- telemetry --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return project/session totals and the local storage paths."""
        with self._lock:
            project_rows = self._connection.execute(
                "SELECT archived, COUNT(*) AS count FROM desktop_projects GROUP BY archived"
            ).fetchall()
            session_rows = self._connection.execute(
                "SELECT archived, COUNT(*) AS count FROM desktop_project_sessions GROUP BY archived"
            ).fetchall()
        projects = {bool(row["archived"]): int(row["count"]) for row in project_rows}
        sessions = {bool(row["archived"]): int(row["count"]) for row in session_rows}
        return {
            "path": str(self.path),
            "workspace_root": str(self.workspace_root),
            "projects": projects.get(False, 0),
            "archived_projects": projects.get(True, 0),
            "sessions": sessions.get(False, 0),
            "archived_sessions": sessions.get(True, 0),
        }

    def __len__(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM desktop_projects"
            ).fetchone()
        return int(row["count"] if row else 0)

    # -- internals --------------------------------------------------------

    def _resolve_project_file(self, project_id: str, relative_path: str) -> tuple[Path, str]:
        project = self.require_project(project_id)
        root = Path(project.workspace).expanduser().resolve()
        relative = str(relative_path).replace("\\", "/").lstrip("/")
        target = (root / relative).resolve()
        try:
            normalized = target.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("workspace path escapes the project root") from exc
        return target, normalized

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, default=str)

    @classmethod
    def _project_values(cls, project: DesktopProject) -> tuple[Any, ...]:
        return (
            project.id,
            project.name,
            project.kind.value,
            project.workspace,
            project.instructions,
            project.model,
            int(project.pinned),
            int(project.archived),
            project.created_at,
            project.updated_at,
            cls._json(project.metadata),
        )

    @classmethod
    def _session_values(cls, session: DesktopSession) -> tuple[Any, ...]:
        return (
            session.id,
            session.project_id,
            session.title,
            session.model,
            int(session.pinned),
            int(session.archived),
            session.parent_id,
            session.last_task_id,
            session.created_at,
            session.updated_at,
            cls._json(session.metadata),
        )

    @classmethod
    def _project_from_row(cls, row: sqlite3.Row) -> DesktopProject:
        return DesktopProject(
            id=str(row["id"]),
            name=str(row["name"]),
            kind=ProjectKind(str(row["kind"])),
            workspace=str(row["workspace"]),
            instructions=str(row["instructions"]),
            model=str(row["model"]),
            pinned=bool(row["pinned"]),
            archived=bool(row["archived"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            metadata=cls._decode_json(row["metadata"]),
        )

    @classmethod
    def _session_from_row(cls, row: sqlite3.Row) -> DesktopSession:
        return DesktopSession(
            id=str(row["id"]),
            project_id=str(row["project_id"]),
            title=str(row["title"]),
            model=str(row["model"]),
            pinned=bool(row["pinned"]),
            archived=bool(row["archived"]),
            parent_id=str(row["parent_id"]),
            last_task_id=str(row["last_task_id"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            metadata=cls._decode_json(row["metadata"]),
        )

    @staticmethod
    def _decode_json(raw: Any) -> dict[str, Any]:
        try:
            payload = json.loads(str(raw or "{}"))
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}


__all__ = [
    "DesktopProject",
    "DesktopProjectStore",
    "DesktopSession",
    "ProjectKind",
    "WorkspaceEntry",
]
