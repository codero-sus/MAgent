"""Headless workbench state shared by the Tk panel and tests.

The workbench intentionally contains no Tk code.  It owns the selected project
and session, turns prompts into durable background tasks, reconstructs a chat
transcript from the queue, and builds the project/skill context used by an
agent run.  The native interface is therefore a view over testable local state
rather than the only place the workflow exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..utils.text import ellipsize
from .models import DesktopTask, TaskStatus
from .projects import DesktopProject, DesktopProjectStore, DesktopSession, ProjectKind
from .queue import DesktopTaskQueue

_SKILL_REFERENCE = re.compile(r"(?<!\w)@([a-zA-Z0-9][a-zA-Z0-9_-]{0,79})")


@dataclass(slots=True, frozen=True)
class TranscriptItem:
    """One user or agent entry reconstructed from a durable task."""

    role: str
    text: str
    task_id: str
    status: str
    created_at: float
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping."""
        return {
            "role": self.role,
            "text": self.text,
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "error": self.error,
        }


class DesktopWorkbench:
    """Project/session coordinator for background desktop agent work."""

    def __init__(
        self,
        projects: DesktopProjectStore,
        queue: DesktopTaskQueue,
        *,
        project_id: str = "",
        session_id: str = "",
    ) -> None:
        self.projects = projects
        self.queue = queue
        self.project_id = ""
        self.session_id = ""
        project = projects.get_project(project_id) if project_id else None
        self.select_project((project or projects.ensure_default()).id, preferred_session=session_id)

    @property
    def project(self) -> DesktopProject:
        """Return the active project."""
        return self.projects.require_project(self.project_id)

    @property
    def session(self) -> DesktopSession:
        """Return the active session."""
        session = self.projects.get_session(self.session_id)
        if session is None:
            session = self.projects.ensure_session(self.project_id)
            self.session_id = session.id
        return session

    def select_project(self, project_id: str, *, preferred_session: str = "") -> DesktopProject:
        """Activate a project and one of its non-archived sessions."""
        project = self.projects.require_project(project_id)
        if project.archived:
            raise ValueError("cannot select an archived project")
        selected = self.projects.get_session(preferred_session) if preferred_session else None
        if selected is None or selected.project_id != project.id or selected.archived:
            selected = self.projects.ensure_session(project.id)
        self.project_id = project.id
        self.session_id = selected.id
        return project

    def select_session(self, session_id: str) -> DesktopSession:
        """Activate a session and its owning project."""
        session = self.projects.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown desktop session {session_id!r}")
        if session.archived:
            raise ValueError("cannot select an archived session")
        self.project_id = session.project_id
        self.session_id = session.id
        return session

    def new_project(
        self,
        name: str,
        *,
        kind: str | ProjectKind = ProjectKind.GENERAL,
        workspace: str = "",
        instructions: str = "",
        model: str = "",
    ) -> DesktopProject:
        """Create and select a project with its first session."""
        project = self.projects.create_project(
            name,
            kind=kind,
            workspace=workspace,
            instructions=instructions,
            model=model,
        )
        session = self.projects.create_session(project.id, title="New session")
        self.project_id = project.id
        self.session_id = session.id
        return project

    def new_session(
        self,
        title: str = "New session",
        *,
        parent_id: str = "",
        model: str = "",
    ) -> DesktopSession:
        """Create and select a session in the active project."""
        session = self.projects.create_session(
            self.project_id,
            title=title,
            parent_id=parent_id,
            model=model,
        )
        self.session_id = session.id
        return session

    def enqueue(
        self,
        prompt: str,
        *,
        title: str = "",
        priority: int = 100,
        max_attempts: int = 1,
        skills: list[str] | tuple[str, ...] = (),
        source: str = "workbench",
        metadata: dict[str, Any] | None = None,
    ) -> DesktopTask:
        """Queue work in the active project/session and record explicit skills."""
        clean_prompt = str(prompt).strip()
        if not clean_prompt:
            raise ValueError("workbench prompt must not be empty")
        selected_skills = _normalize_skills(skills)
        referenced = _skill_references(clean_prompt)
        for name in referenced:
            if name not in selected_skills:
                selected_skills.append(name)
        details = dict(metadata or {})
        details.update(
            {
                "source": source,
                "project_id": self.project_id,
                "session_id": self.session_id,
                "skills": selected_skills,
            }
        )
        project = self.project
        session = self.session
        details.setdefault("workspace", project.workspace)
        details.setdefault("project_kind", project.kind.value)
        details.setdefault("model", session.model or project.model)
        task = self.queue.enqueue(
            clean_prompt,
            title=title,
            priority=priority,
            max_attempts=max_attempts,
            project_id=project.id,
            session_id=session.id,
            metadata=details,
        )
        if session.title == "New session":
            self.projects.update_session(session.id, title=_prompt_title(clean_prompt))
        self.projects.touch_session(session.id, task_id=task.id)
        return task

    def tasks(self, *, limit: int = 500) -> list[DesktopTask]:
        """Return newest tasks in the active session."""
        return self.queue.list(session_id=self.session_id, limit=limit)

    def activity(self, *, project_only: bool = False, limit: int = 200) -> list[DesktopTask]:
        """Return active queue entries, optionally constrained to this project."""
        tasks = self.queue.list(
            project_id=self.project_id if project_only else "",
            limit=limit,
        )
        return [
            task
            for task in tasks
            if task.status in {TaskStatus.QUEUED, TaskStatus.PAUSED, TaskStatus.RUNNING}
        ]

    def transcript(self, *, limit: int = 200) -> list[TranscriptItem]:
        """Reconstruct user/agent messages in chronological order."""
        tasks = self.queue.list(session_id=self.session_id, limit=limit)
        tasks.sort(key=lambda task: (task.created_at, task.id))
        entries: list[TranscriptItem] = []
        for task in tasks:
            entries.append(
                TranscriptItem(
                    role="user",
                    text=task.prompt,
                    task_id=task.id,
                    status=task.status.value,
                    created_at=task.created_at,
                )
            )
            answer = task.result
            error = False
            if task.status is TaskStatus.FAILED:
                answer = task.error or "Task failed."
                error = True
            elif task.status is TaskStatus.CANCELLED:
                answer = task.error or "Task cancelled."
                error = True
            elif task.status is TaskStatus.PAUSED:
                answer = "Paused"
            elif task.status is TaskStatus.QUEUED:
                answer = "Queued in the background…"
            elif task.status is TaskStatus.RUNNING:
                answer = "Working in the background…"
            if answer:
                entries.append(
                    TranscriptItem(
                        role="agent",
                        text=answer,
                        task_id=task.id,
                        status=task.status.value,
                        created_at=task.completed_at or task.updated_at,
                        error=error,
                    )
                )
        return entries

    def context_block(self, task: DesktopTask, skill_library: Any = None) -> str:
        """Build project and skill instructions for one queued task."""
        project = self.projects.get_project(
            task.project_id or str(task.metadata.get("project_id", ""))
        )
        session = self.projects.get_session(task.session_id)
        sections: list[str] = []
        if project is not None:
            sections.append(
                "## Active desktop project\n"
                f"Name: {project.name}\n"
                f"Type: {project.kind.value}\n"
                f"Workspace: {project.workspace}\n"
                "Keep generated or modified artifacts inside this workspace unless the user "
                "explicitly approves another location."
            )
            if project.instructions:
                sections.append("## Project instructions\n" + project.instructions)
        if session is not None:
            sections.append(f"## Session\n{session.title} ({session.id})")
            history = self._recent_session_context(
                session.id,
                before=task.created_at,
                current_task_id=task.id,
            )
            if history:
                sections.append("## Recent session context\n" + history)
        skills = _normalize_skills(task.metadata.get("skills", []))
        if skill_library is not None:
            selected = []
            for name in skills:
                try:
                    selected.append(skill_library.get(name))
                except Exception:
                    continue
            if not selected:
                try:
                    selected = list(skill_library.for_task(task.prompt))
                except Exception:
                    selected = []
            if selected:
                rendered = skill_library.render(selected, style="heading", max_chars=6_000)
                if rendered:
                    sections.append(
                        "## Reusable skills\nFollow these procedures unless the user says otherwise.\n"
                        + rendered
                    )
        return "\n\n".join(sections)

    def _recent_session_context(
        self,
        session_id: str,
        *,
        before: float,
        current_task_id: str,
    ) -> str:
        """Render bounded prior turns so switching sessions preserves continuity."""
        tasks = self.queue.list(session_id=session_id, limit=1000)
        previous = [
            item
            for item in tasks
            if item.id != current_task_id
            and item.created_at <= before
            and item.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        ]
        previous.sort(key=lambda item: (item.created_at, item.id))
        lines: list[str] = []
        for item in previous[-4:]:
            lines.append("User: " + ellipsize(item.prompt.replace("\n", " "), 800))
            response = item.result or item.error
            if response:
                lines.append("MAgent: " + ellipsize(response.replace("\n", " "), 1_500))
        return ellipsize("\n".join(lines), 7_000)

    def snapshot(self) -> dict[str, Any]:
        """Return the active context and local telemetry for APIs/status views."""
        return {
            "project": self.project.to_dict(),
            "session": self.session.to_dict(),
            "projects": self.projects.stats(),
            "queue": self.queue.stats(project_id=self.project_id),
        }


def _skill_references(prompt: str) -> list[str]:
    return _normalize_skills(match.group(1) for match in _SKILL_REFERENCE.finditer(prompt))


def _normalize_skills(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    result: list[str] = []
    for value in values or []:
        name = str(value).strip().lower().replace(" ", "-")
        if name and name not in result:
            result.append(name)
    return result


def _prompt_title(prompt: str) -> str:
    first = next((line.strip() for line in prompt.splitlines() if line.strip()), "New session")
    first = _SKILL_REFERENCE.sub("", first).strip() or "New session"
    return first if len(first) <= 72 else first[:69].rstrip() + "..."


__all__ = ["DesktopWorkbench", "TranscriptItem"]
