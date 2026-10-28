"""Persistent, process-safe task queue for always-on desktop automation.

The queue uses SQLite from the Python standard library.  A desktop panel can be
closed and reopened without losing work, the HTTP API and CLI can add tasks to
the same file, and claiming uses ``BEGIN IMMEDIATE`` so two workers cannot run
the same task.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..logging_ import get_logger
from .models import DesktopTask, TaskStatus

LOGGER = get_logger("desktop.queue")
TaskListener = Callable[[str, DesktopTask | None], Any]
TaskExecutor = Callable[[DesktopTask], Any]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS desktop_tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL NOT NULL DEFAULT 0,
    completed_at REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 1,
    worker_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT 'desktop',
    result TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_desktop_tasks_next
    ON desktop_tasks(status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_desktop_tasks_updated
    ON desktop_tasks(updated_at DESC);
CREATE TABLE IF NOT EXISTS desktop_queue_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class DesktopTaskQueue:
    """A durable priority queue shared by the desktop UI, CLI, and HTTP API."""

    def __init__(self, path: str | Path = ":memory:", *, recover: bool = True) -> None:
        """Open a queue and create its schema when necessary.

        Args:
            path: SQLite path or ``:memory:``
            recover: put tasks left ``running`` by a crashed worker back in the queue
        """
        self.path = Path(path).expanduser() if str(path) != ":memory:" else Path(":memory:")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._listeners: list[TaskListener] = []
        self._closed = False
        self._connection = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            if str(self.path) != ":memory:":
                self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.commit()
        if recover:
            self.recover_running()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    def __enter__(self) -> DesktopTaskQueue:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- events ------------------------------------------------------------

    def subscribe(self, listener: TaskListener) -> Callable[[], None]:
        """Register a task-change callback and return an unsubscribe function."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: str, task: DesktopTask | None = None) -> None:
        """Notify listeners without allowing one callback to break the queue."""
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event, task)
            except Exception as exc:
                LOGGER.debug("desktop queue listener failed: %s", exc)

    # -- CRUD --------------------------------------------------------------

    def enqueue(
        self,
        prompt: str,
        *,
        title: str = "",
        priority: int = 50,
        max_attempts: int = 1,
        session_id: str = "desktop",
        metadata: dict[str, Any] | None = None,
    ) -> DesktopTask:
        """Append one prompt and return the persisted task."""
        if not str(prompt).strip():
            raise ValueError("desktop task prompt must not be empty")
        task = DesktopTask(
            prompt=prompt,
            title=title,
            priority=priority,
            max_attempts=max_attempts,
            session_id=session_id or "desktop",
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO desktop_tasks (
                    id,title,prompt,status,priority,created_at,updated_at,started_at,
                    completed_at,attempts,max_attempts,worker_id,session_id,result,error,metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                self._values(task),
            )
            self._connection.commit()
        self._emit("enqueued", task)
        return task

    def get(self, task_id: str) -> DesktopTask | None:
        """Return one task by id."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM desktop_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(
        self,
        *,
        status: str | TaskStatus = "",
        limit: int = 100,
        newest_first: bool = True,
    ) -> list[DesktopTask]:
        """List tasks, optionally filtered by status."""
        bounded = max(1, min(int(limit), 1000))
        order = "updated_at DESC" if newest_first else "priority DESC, created_at ASC"
        params: list[Any] = []
        sql = "SELECT * FROM desktop_tasks"
        if status:
            normalized = TaskStatus(str(getattr(status, "value", status))).value
            sql += " WHERE status = ?"
            params.append(normalized)
        sql += f" ORDER BY {order} LIMIT ?"
        params.append(bounded)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def cancel(self, task_id: str) -> DesktopTask | None:
        """Cancel a queued or running task."""
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks
                   SET status=?, updated_at=?, completed_at=?, worker_id=''
                 WHERE id=? AND status IN (?,?,?)
                """,
                (
                    TaskStatus.CANCELLED.value,
                    now,
                    now,
                    task_id,
                    TaskStatus.QUEUED.value,
                    TaskStatus.PAUSED.value,
                    TaskStatus.RUNNING.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("cancelled", task)
        return task

    def pause_task(self, task_id: str) -> DesktopTask | None:
        """Pause queued work without interrupting a task already running."""
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks SET status=?, updated_at=?
                 WHERE id=? AND status=?
                """,
                (
                    TaskStatus.PAUSED.value,
                    now,
                    task_id,
                    TaskStatus.QUEUED.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("task_paused", task)
        return task

    def resume_task(self, task_id: str) -> DesktopTask | None:
        """Return an individually paused task to the runnable queue."""
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks SET status=?, updated_at=?
                 WHERE id=? AND status=?
                """,
                (
                    TaskStatus.QUEUED.value,
                    now,
                    task_id,
                    TaskStatus.PAUSED.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("task_resumed", task)
        return task

    def retry(self, task_id: str, *, reset_attempts: bool = False) -> DesktopTask | None:
        """Return a terminal task to the queue."""
        now = time.time()
        attempts_sql = "attempts=0," if reset_attempts else ""
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE desktop_tasks
                   SET status=?, {attempts_sql} updated_at=?, started_at=0, completed_at=0,
                       worker_id='', result='', error=''
                 WHERE id=? AND status IN (?,?,?)
                """,
                (
                    TaskStatus.QUEUED.value,
                    now,
                    task_id,
                    TaskStatus.FAILED.value,
                    TaskStatus.CANCELLED.value,
                    TaskStatus.COMPLETED.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("retried", task)
        return task

    def delete(self, task_id: str, *, include_running: bool = False) -> bool:
        """Delete a task, refusing active work unless explicitly requested."""
        with self._lock:
            if include_running:
                cursor = self._connection.execute(
                    "DELETE FROM desktop_tasks WHERE id=?", (task_id,)
                )
            else:
                cursor = self._connection.execute(
                    "DELETE FROM desktop_tasks WHERE id=? AND status != ?",
                    (task_id, TaskStatus.RUNNING.value),
                )
            self._connection.commit()
        removed = bool(cursor.rowcount)
        if removed:
            self._emit("deleted", None)
        return removed

    def clear(self, *, statuses: Iterable[str | TaskStatus] | None = None) -> int:
        """Delete terminal tasks, or the explicitly supplied states."""
        values = [
            TaskStatus(str(getattr(item, "value", item))).value
            for item in (
                statuses
                or tuple(
                    TaskStatus.__members__[name] for name in ("COMPLETED", "FAILED", "CANCELLED")
                )
            )
        ]
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self._lock:
            cursor = self._connection.execute(
                f"DELETE FROM desktop_tasks WHERE status IN ({placeholders})", values
            )
            self._connection.commit()
        count = int(cursor.rowcount)
        if count:
            self._emit("cleared", None)
        return count

    # -- worker transitions ------------------------------------------------

    def claim_next(self, worker_id: str) -> DesktopTask | None:
        """Atomically claim the highest-priority queued task unless paused."""
        if self.paused:
            return None
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                active = self._connection.execute(
                    "SELECT id FROM desktop_tasks WHERE status=? LIMIT 1",
                    (TaskStatus.RUNNING.value,),
                ).fetchone()
                if active is not None:
                    self._connection.commit()
                    return None
                row = self._connection.execute(
                    """
                    SELECT id FROM desktop_tasks
                     WHERE status=?
                     ORDER BY priority DESC, created_at ASC
                     LIMIT 1
                    """,
                    (TaskStatus.QUEUED.value,),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                task_id = str(row["id"])
                cursor = self._connection.execute(
                    """
                    UPDATE desktop_tasks
                       SET status=?, worker_id=?, started_at=?, updated_at=?, attempts=attempts+1,
                           completed_at=0, error=''
                     WHERE id=? AND status=?
                    """,
                    (
                        TaskStatus.RUNNING.value,
                        worker_id,
                        now,
                        now,
                        task_id,
                        TaskStatus.QUEUED.value,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("started", task)
        return task

    def complete(self, task_id: str, result: Any = "") -> DesktopTask | None:
        """Mark a running task successful."""
        now = time.time()
        rendered = self._render_result(result)
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks
                   SET status=?, result=?, error='', updated_at=?, completed_at=?, worker_id=''
                 WHERE id=? AND status=?
                """,
                (
                    TaskStatus.COMPLETED.value,
                    rendered,
                    now,
                    now,
                    task_id,
                    TaskStatus.RUNNING.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("completed", task)
        return task

    def fail(
        self, task_id: str, error: BaseException | str, *, retry: bool = True
    ) -> DesktopTask | None:
        """Record a failure and optionally requeue while attempts remain."""
        existing = self.get(task_id)
        if existing is None:
            return None
        will_retry = retry and existing.attempts < existing.max_attempts
        status = TaskStatus.QUEUED if will_retry else TaskStatus.FAILED
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks
                   SET status=?, error=?, updated_at=?, completed_at=?, worker_id=''
                 WHERE id=? AND status=?
                """,
                (
                    status.value,
                    str(error)[:8000],
                    now,
                    0.0 if will_retry else now,
                    task_id,
                    TaskStatus.RUNNING.value,
                ),
            )
            self._connection.commit()
        if not cursor.rowcount:
            return None
        task = self.get(task_id)
        self._emit("requeued" if will_retry else "failed", task)
        return task

    def recover_running(self) -> int:
        """Requeue tasks orphaned when a previous worker stopped."""
        now = time.time()
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE desktop_tasks
                   SET status=?, worker_id='', started_at=0, updated_at=?,
                       error=CASE WHEN error='' THEN 'recovered after worker exit' ELSE error END
                 WHERE status=?
                """,
                (TaskStatus.QUEUED.value, now, TaskStatus.RUNNING.value),
            )
            self._connection.commit()
        count = int(cursor.rowcount)
        if count:
            self._emit("recovered", None)
        return count

    # -- queue state -------------------------------------------------------

    @property
    def paused(self) -> bool:
        """Whether workers should refrain from claiming new tasks."""
        return self._state("paused", "0") == "1"

    def set_paused(self, paused: bool) -> None:
        """Persist the queue pause state."""
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO desktop_queue_state(key,value) VALUES('paused',?)",
                ("1" if paused else "0",),
            )
            self._connection.commit()
        self._emit("paused" if paused else "resumed", None)

    def stats(self) -> dict[str, Any]:
        """Return counts and basic queue telemetry."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM desktop_tasks GROUP BY status"
            ).fetchall()
            oldest = self._connection.execute(
                "SELECT MIN(created_at) AS value FROM desktop_tasks WHERE status=?",
                (TaskStatus.QUEUED.value,),
            ).fetchone()
        counts = {status.value: 0 for status in TaskStatus}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        queued_since = float(oldest["value"] or 0.0) if oldest else 0.0
        return {
            "path": str(self.path),
            "paused": self.paused,
            "counts": counts,
            "total": sum(counts.values()),
            "queued_age_seconds": round(max(0.0, time.time() - queued_since), 3)
            if queued_since
            else 0.0,
        }

    def __len__(self) -> int:
        """Number of tasks in all states."""
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM desktop_tasks").fetchone()
        return int(row["count"] if row else 0)

    # -- internals ---------------------------------------------------------

    def _state(self, key: str, default: str) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM desktop_queue_state WHERE key=?", (key,)
            ).fetchone()
        return str(row["value"]) if row else default

    @staticmethod
    def _values(task: DesktopTask) -> tuple[Any, ...]:
        return (
            task.id,
            task.title,
            task.prompt,
            task.status.value,
            task.priority,
            task.created_at,
            task.updated_at,
            task.started_at,
            task.completed_at,
            task.attempts,
            task.max_attempts,
            task.worker_id,
            task.session_id,
            task.result,
            task.error,
            json.dumps(task.metadata, ensure_ascii=False, default=str),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> DesktopTask:
        try:
            metadata = json.loads(str(row["metadata"] or "{}"))
        except (TypeError, ValueError):
            metadata = {}
        return DesktopTask(
            id=str(row["id"]),
            title=str(row["title"]),
            prompt=str(row["prompt"]),
            status=TaskStatus(str(row["status"])),
            priority=int(row["priority"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            started_at=float(row["started_at"]),
            completed_at=float(row["completed_at"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            worker_id=str(row["worker_id"]),
            session_id=str(row["session_id"]),
            result=str(row["result"]),
            error=str(row["error"]),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    @staticmethod
    def _render_result(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        output = getattr(result, "output", None)
        if isinstance(output, str):
            return output
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)


class DesktopQueueWorker:
    """Single-consumer background worker for :class:`DesktopTaskQueue`."""

    def __init__(
        self,
        queue: DesktopTaskQueue,
        executor: TaskExecutor,
        *,
        worker_id: str = "desktop-ui",
        poll_interval: float = 0.25,
        on_event: TaskListener | None = None,
        cancel_current: Callable[[], Any] | None = None,
    ) -> None:
        self.queue = queue
        self.executor = executor
        self.worker_id = worker_id
        self.poll_interval = max(0.05, float(poll_interval))
        self.on_event = on_event
        self.cancel_callback = cancel_current
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: DesktopTask | None = None
        self._unsubscribe = self.queue.subscribe(self._queue_event)

    @property
    def running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def current(self) -> DesktopTask | None:
        """The task currently handed to the executor."""
        return self._current

    def start(self) -> DesktopQueueWorker:
        """Start the worker once and return ``self``."""
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"magent-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, wait: bool = True, cancel: bool = False) -> None:
        """Stop claiming work and optionally cancel the active agent run."""
        self._stop.set()
        self._wake.set()
        if cancel:
            self.cancel_active()
        thread = self._thread
        if (
            wait
            and thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)
        self._thread = None

    def cancel_active(self) -> bool:
        """Cancel the active task and notify the executor when supported."""
        task = self._current
        if task is None:
            return False
        self.queue.cancel(task.id)
        if self.cancel_callback is not None:
            try:
                self.cancel_callback()
            except Exception as exc:
                LOGGER.debug("desktop executor cancellation failed: %s", exc)
        return True

    def wake(self) -> None:
        """Wake a sleeping worker."""
        self._wake.set()

    def _queue_event(self, event: str, task: DesktopTask | None) -> None:
        if event in {"enqueued", "retried", "resumed", "task_resumed", "recovered"}:
            self._wake.set()

    def _publish(self, event: str, task: DesktopTask | None) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, task)
            except Exception as exc:
                LOGGER.debug("desktop worker callback failed: %s", exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.queue.paused:
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            task = self.queue.claim_next(self.worker_id)
            if task is None:
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            self._current = task
            self._publish("started", task)
            watch_stop = threading.Event()
            watch_thread: threading.Thread | None = None
            if self.cancel_callback is not None:
                watch_thread = threading.Thread(
                    target=self._watch_cancellation,
                    args=(task.id, watch_stop),
                    name=f"magent-{self.worker_id}-cancel",
                    daemon=True,
                )
                watch_thread.start()
            try:
                result = self.executor(task)
            except Exception as exc:
                updated = self.queue.fail(task.id, exc, retry=True)
                self._publish("failed", updated)
            else:
                latest = self.queue.get(task.id)
                if latest is not None and latest.status is TaskStatus.CANCELLED:
                    self._publish("cancelled", latest)
                else:
                    updated = self.queue.complete(task.id, result)
                    self._publish("completed", updated)
            finally:
                watch_stop.set()
                if watch_thread is not None:
                    watch_thread.join(timeout=max(0.1, self.poll_interval * 2))
                self._current = None
        self._unsubscribe()

    def _watch_cancellation(self, task_id: str, stop: threading.Event) -> None:
        """Relay cancellation written by another queue connection to the executor."""
        while not stop.wait(min(self.poll_interval, 0.25)):
            latest = self.queue.get(task_id)
            if latest is None or latest.status is not TaskStatus.CANCELLED:
                continue
            callback = self.cancel_callback
            if callback is not None:
                try:
                    callback()
                except Exception as exc:
                    LOGGER.debug("desktop executor cancellation watcher failed: %s", exc)
            return

    def __enter__(self) -> DesktopQueueWorker:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


__all__ = ["DesktopQueueWorker", "DesktopTaskQueue", "TaskExecutor", "TaskListener"]
