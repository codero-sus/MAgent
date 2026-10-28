"""Deterministic tests for MAgent desktop automation and its durable queue."""

from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from unittest import mock

from magent.config import DesktopConfig, config_from_dict
from magent.desktop.autostart import autostart_status, install_autostart, remove_autostart
from magent.desktop.branding import (
    LOGO_THEMES,
    DesktopAppearance,
    DesktopAppearanceStore,
    logo_ids,
    logo_theme,
    write_logo_assets,
)
from magent.desktop.driver import (
    DesktopController,
    DesktopUnavailableError,
    NativeDesktopDriver,
    RecordingDesktopDriver,
)
from magent.desktop.models import Point, Rect, TaskStatus
from magent.desktop.queue import DesktopQueueWorker, DesktopTaskQueue
from magent.desktop.workflow import DesktopWorkflow, DesktopWorkflowRunner
from tests.support import MAgentTestCase


class DesktopControllerTest(MAgentTestCase):
    """Every desktop primitive is testable without controlling the real machine."""

    def controller(self) -> tuple[DesktopController, RecordingDesktopDriver]:
        config = DesktopConfig(
            backend="recording",
            fail_safe=False,
            action_pause=0,
            allowed_apps=["editor"],
            allowed_open_schemes=["https", "file"],
        )
        driver = RecordingDesktopDriver(width=1280, height=720)
        return DesktopController(config, driver=driver), driver

    def test_recording_driver_covers_input_capture_clipboard_windows_and_launch(self) -> None:
        controller, driver = self.controller()
        self.assertEqual(controller.screen_size(), (1280, 720))
        self.assertEqual(controller.move_pointer(10, 20), Point(10, 20))
        controller.click(x=12, y=24, clicks=2)
        controller.drag(50, 60, duration=0)
        controller.scroll(-3)
        controller.type_text("hello", interval=0)
        controller.press("enter", presses=2, interval=0)
        controller.hotkey(["ctrl", "s"], interval=0)
        controller.clipboard_write("local text")
        self.assertEqual(controller.clipboard_read(), "local text")
        capture = controller.screenshot(self.workspace / "capture.png")
        self.assertTrue(capture.read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(controller.pixel(1, 2), (82, 220, 255))
        self.assertEqual(controller.locate(capture), Rect(10, 20, 30, 40))
        self.assertEqual(controller.windows(query="test")[0].id, "1")
        controller.focus_window("1")
        controller.window_action("1", "minimize")
        controller.move_resize_window("1", Rect(1, 2, 300, 200))
        self.assertEqual(controller.launch("editor", ["notes.txt"]), 4242)
        self.assertTrue(controller.open_target("https://example.com"))
        actions = {entry["action"] for entry in driver.actions}
        self.assertTrue(
            {
                "move",
                "click",
                "drag",
                "scroll",
                "type",
                "press",
                "hotkey",
                "clipboard_write",
                "clipboard_read",
                "screenshot",
                "pixel",
                "locate",
                "focus_window",
                "window_action",
                "move_window",
                "launch",
                "open",
            }.issubset(actions)
        )

    def test_controller_enforces_bounds_limits_allowlists_and_fail_safe(self) -> None:
        controller, _ = self.controller()
        with self.assertRaises(ValueError):
            controller.move_pointer(1280, 10)
        with self.assertRaises(ValueError):
            controller.type_text("x" * (controller.config.max_type_chars + 1))
        with self.assertRaises(PermissionError):
            controller.launch("terminal")
        with self.assertRaises(PermissionError):
            controller.open_target("javascript:alert(1)")

        fail_safe = DesktopConfig(backend="recording", fail_safe=True, action_pause=0)
        guarded = DesktopController(fail_safe, driver=RecordingDesktopDriver())
        with self.assertRaises(DesktopUnavailableError):
            guarded.click(x=10, y=10)

    def test_describe_reports_recording_capabilities(self) -> None:
        controller, _ = self.controller()
        payload = controller.describe()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["capabilities"]["driver"], "recording")
        self.assertTrue(payload["capabilities"]["ready"])

    def test_native_capability_probes_are_platform_and_headless_safe(self) -> None:
        linux = NativeDesktopDriver(system="Linux", environ={})
        linux._which = dict.fromkeys(linux._which, None)
        headless = linux.capabilities()
        self.assertEqual(headless.platform, "Linux")
        self.assertFalse(headless.display_available)
        self.assertIn("no graphical display", " ".join(headless.notes))

        windows = NativeDesktopDriver(system="Windows", environ={})
        windows._which = dict.fromkeys(windows._which, None)
        supported = windows.capabilities()
        self.assertEqual(supported.platform, "Windows")
        self.assertEqual(supported.session, "windows")
        self.assertTrue(supported.mouse)
        self.assertTrue(supported.keyboard)


class DesktopWorkflowTest(MAgentTestCase):
    """JSON workflows stay bounded, previewable, and fully observable."""

    def setUp(self) -> None:
        super().setUp()
        self.driver = RecordingDesktopDriver(width=800, height=600)
        self.controller = DesktopController(
            DesktopConfig(backend="recording", fail_safe=False, action_pause=0),
            driver=self.driver,
        )

    def test_parse_run_aliases_and_variables(self) -> None:
        workflow = DesktopWorkflow.parse(
            {
                "name": "compose",
                "variables": {"message": "Hello MAgent"},
                "steps": [
                    {"action": "move_mouse", "x": 100, "y": 150},
                    {"action": "mouse_click"},
                    {"action": "type_text", "text": "${message}", "interval": 0},
                    {"action": "set_clipboard", "text": "${message}"},
                    {"action": "launch_app", "app": "editor"},
                    {"action": "sleep", "seconds": 0},
                ],
            }
        )
        events: list[str] = []
        result = DesktopWorkflowRunner(
            self.controller, listener=lambda event, _payload: events.append(event)
        ).run(workflow)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.steps), 6)
        self.assertEqual(self.driver.clipboard, "Hello MAgent")
        typed = next(item for item in self.driver.actions if item["action"] == "type")
        self.assertEqual(typed["text"], "Hello MAgent")
        self.assertEqual(events[0], "workflow.started")
        self.assertEqual(events[-1], "workflow.finished")

    def test_dry_run_executes_nothing_and_resolves_placeholders(self) -> None:
        runner = DesktopWorkflowRunner(self.controller)
        result = runner.run(
            {
                "variables": {"x": 44},
                "steps": [{"action": "move", "x": "${x}", "y": 9}],
            },
            dry_run=True,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.steps[0].skipped)
        self.assertEqual(result.steps[0].output["params"]["x"], 44)
        self.assertEqual(self.driver.actions, [])

    def test_parent_cancellation_interrupts_wait_and_remaining_steps(self) -> None:
        cancelled = threading.Event()
        timer = threading.Timer(0.05, cancelled.set)
        timer.start()
        try:
            result = DesktopWorkflowRunner(self.controller, cancelled=cancelled.is_set).run(
                [
                    {"action": "wait", "seconds": 2},
                    {"action": "type", "text": "must not execute"},
                ]
            )
        finally:
            timer.cancel()
        self.assertFalse(result.ok)
        self.assertTrue(result.stopped)
        self.assertEqual(len(result.steps), 1)
        self.assertFalse(any(item["action"] == "type" for item in self.driver.actions))

    def test_invalid_and_failed_workflows_stop_safely(self) -> None:
        with self.assertRaises(ValueError):
            DesktopWorkflow.parse("not json")
        with self.assertRaises(ValueError):
            DesktopWorkflow.parse([])
        result = DesktopWorkflowRunner(self.controller).run(
            [
                {"action": "unknown"},
                {"action": "type", "text": "must not execute"},
            ]
        )
        self.assertFalse(result.ok)
        self.assertEqual(len(result.steps), 1)
        self.assertIn("unsupported", result.steps[0].error)
        self.assertEqual(self.driver.actions, [])


class DesktopQueueTest(MAgentTestCase):
    """The task list is persistent and has explicit lifecycle transitions."""

    def queue_path(self) -> Path:
        return self.home.path / "desktop" / "queue.db"

    def test_priority_persistence_and_lifecycle(self) -> None:
        events: list[str] = []
        with DesktopTaskQueue(self.queue_path()) as queue:
            queue.subscribe(lambda event, _task: events.append(event))
            low = queue.enqueue("low priority", priority=5)
            high = queue.enqueue("high priority", title="Important", priority=90)
            claimed = queue.claim_next("test-worker")
            self.assertIsNotNone(claimed)
            assert claimed is not None
            self.assertEqual(claimed.id, high.id)
            self.assertEqual(claimed.status, TaskStatus.RUNNING)
            completed = queue.complete(claimed.id, {"answer": 42})
            self.assertEqual(completed.status, TaskStatus.COMPLETED)  # type: ignore[union-attr]
            cancelled = queue.cancel(low.id)
            self.assertEqual(cancelled.status, TaskStatus.CANCELLED)  # type: ignore[union-attr]
            retried = queue.retry(low.id, reset_attempts=True)
            self.assertEqual(retried.status, TaskStatus.QUEUED)  # type: ignore[union-attr]
            self.assertIn("enqueued", events)
            self.assertIn("started", events)
            self.assertIn("completed", events)
        with DesktopTaskQueue(self.queue_path(), recover=False) as reopened:
            self.assertEqual(len(reopened), 2)
            self.assertEqual(reopened.get(high.id).result, '{"answer": 42}')  # type: ignore[union-attr]

    def test_fail_retry_pause_cancel_delete_and_clear(self) -> None:
        with DesktopTaskQueue(":memory:") as queue:
            task = queue.enqueue("retry me", max_attempts=2)
            claimed = queue.claim_next("worker")
            self.assertEqual(claimed.id, task.id)  # type: ignore[union-attr]
            queued = queue.fail(task.id, "temporary", retry=True)
            self.assertEqual(queued.status, TaskStatus.QUEUED)  # type: ignore[union-attr]
            queue.claim_next("worker")
            failed = queue.fail(task.id, ValueError("permanent"), retry=True)
            self.assertEqual(failed.status, TaskStatus.FAILED)  # type: ignore[union-attr]
            self.assertIsNone(queue.cancel(task.id))
            self.assertIsNotNone(queue.retry(task.id, reset_attempts=True))
            paused = queue.pause_task(task.id)
            self.assertEqual(paused.status, TaskStatus.PAUSED)  # type: ignore[union-attr]
            self.assertIsNone(queue.claim_next("worker"))
            resumed = queue.resume_task(task.id)
            self.assertEqual(resumed.status, TaskStatus.QUEUED)  # type: ignore[union-attr]
            queue.set_paused(True)
            self.assertTrue(queue.paused)
            self.assertIsNone(queue.claim_next("worker"))
            queue.set_paused(False)
            self.assertIsNotNone(queue.claim_next("worker"))
            self.assertFalse(queue.delete(task.id))
            queue.cancel(task.id)
            self.assertTrue(queue.delete(task.id))
            queue.enqueue("a")
            queue.enqueue("b")
            self.assertEqual(queue.clear(statuses=[TaskStatus.QUEUED]), 2)

    def test_worker_completes_tasks_sequentially(self) -> None:
        done = threading.Event()
        order: list[str] = []
        with DesktopTaskQueue(":memory:") as queue:
            first = queue.enqueue("first", priority=10)
            second = queue.enqueue("second", priority=5)

            def execute(task: object) -> str:
                prompt = str(task.prompt)
                order.append(prompt)
                if len(order) == 2:
                    done.set()
                return prompt.upper()

            worker = DesktopQueueWorker(queue, execute, poll_interval=0.01).start()
            self.assertTrue(done.wait(2))
            worker.stop(wait=True)
            self.assertEqual(order, ["first", "second"])
            self.assertEqual(queue.get(first.id).status, TaskStatus.COMPLETED)  # type: ignore[union-attr]
            self.assertEqual(queue.get(second.id).result, "SECOND")  # type: ignore[union-attr]

    def test_external_cancellation_reaches_the_active_executor(self) -> None:
        path = self.queue_path()
        started = threading.Event()
        cancelled = threading.Event()
        release = threading.Event()
        with (
            DesktopTaskQueue(path, recover=False) as first,
            DesktopTaskQueue(path, recover=False) as second,
        ):
            task = first.enqueue("long task")

            def execute(_task: object) -> str:
                started.set()
                release.wait(2)
                return "stopped"

            def cancel_executor() -> None:
                cancelled.set()
                release.set()

            worker = DesktopQueueWorker(
                first,
                execute,
                poll_interval=0.05,
                cancel_current=cancel_executor,
            ).start()
            self.assertTrue(started.wait(1))
            self.assertIsNotNone(second.cancel(task.id))
            self.assertTrue(cancelled.wait(1))
            worker.stop(wait=True)
            self.assertEqual(first.get(task.id).status, TaskStatus.CANCELLED)  # type: ignore[union-attr]

    def test_two_process_connections_cannot_run_tasks_concurrently(self) -> None:
        path = self.queue_path()
        with (
            DesktopTaskQueue(path, recover=False) as first,
            DesktopTaskQueue(path, recover=False) as second,
        ):
            first.enqueue("one", priority=10)
            first.enqueue("two", priority=5)
            active = first.claim_next("worker-one")
            self.assertIsNotNone(active)
            self.assertIsNone(second.claim_next("worker-two"))
            first.complete(active.id, "done")  # type: ignore[union-attr]
            following = second.claim_next("worker-two")
            self.assertIsNotNone(following)
            self.assertEqual(following.prompt, "two")  # type: ignore[union-attr]

    def test_recover_running_after_unclean_shutdown(self) -> None:
        path = self.queue_path()
        queue = DesktopTaskQueue(path)
        task = queue.enqueue("recover")
        queue.claim_next("dead-worker")
        queue.close()
        with DesktopTaskQueue(path, recover=True) as recovered:
            current = recovered.get(task.id)
            self.assertEqual(current.status, TaskStatus.QUEUED)  # type: ignore[union-attr]
            self.assertIn("recovered", current.error)  # type: ignore[union-attr]


class DesktopBrandingAndConfigTest(MAgentTestCase):
    """Branding, persisted appearance, config, and startup files stay deterministic."""

    def test_eight_logo_pairs_generate_and_match_bundled_assets(self) -> None:
        self.assertEqual(len(LOGO_THEMES), 8)
        self.assertEqual(len(set(logo_ids())), 8)
        output = self.workspace / "logos"
        paths = write_logo_assets(output)
        self.assertEqual(len(paths), 17)
        for index, theme in enumerate(LOGO_THEMES, start=1):
            self.assertTrue((output / f"{index:02d}-{theme.id}-icon.svg").is_file())
            self.assertTrue((output / f"{index:02d}-{theme.id}-lockup.svg").is_file())
        self.assertIn("Orbit M", (output / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(logo_theme("does-not-exist").id, "orbit")

    def test_appearance_store_round_trips_and_recovers_bad_json(self) -> None:
        path = self.home.path / "appearance.json"
        store = DesktopAppearanceStore(path)
        state = DesktopAppearance(logo_theme="pulse", bubble_x=123, bubble_y=456)
        store.save(state)
        loaded = store.load()
        self.assertEqual(loaded.logo_theme, "pulse")
        self.assertEqual((loaded.bubble_x, loaded.bubble_y), (123, 456))
        path.write_text("not json", encoding="utf-8")
        self.assertEqual(store.load().logo_theme, "orbit")

    def test_desktop_config_loads_env_style_values_and_validates(self) -> None:
        config = config_from_dict(
            {
                "desktop": {
                    "backend": "recording",
                    "allowed_apps": ["Editor", "Browser"],
                    "max_workflow_steps": 22,
                }
            }
        )
        config.validate()
        self.assertEqual(config.desktop.backend, "recording")
        self.assertEqual(config.desktop.allowed_apps, ["Editor", "Browser"])
        invalid = DesktopConfig(logo_theme="missing")
        with self.assertRaises(Exception):
            invalid.validate()

    def test_macos_and_windows_autostart_formats(self) -> None:
        mac = install_autostart(
            executable="/Applications/Python 3/python",
            system="Darwin",
            home=self.home.path,
        )
        mac_text = mac.read_text(encoding="utf-8")
        self.assertIn("ai.arena.magent.desktop", mac_text)
        self.assertIn("<key>RunAtLoad</key><true/>", mac_text)
        self.assertTrue(remove_autostart(system="Darwin", home=self.home.path))

        with mock.patch.dict(os.environ):
            os.environ.pop("APPDATA", None)
            windows = install_autostart(
                executable="C:/Program Files/Python/python.exe",
                system="Windows",
                home=self.home.path,
            )
            windows_text = windows.read_text(encoding="utf-8")
            self.assertIn("@echo off", windows_text)
            self.assertIn('"C:/Program Files/Python/python.exe"', windows_text)
            self.assertTrue(remove_autostart(system="Windows", home=self.home.path))

    def test_linux_autostart_install_status_and_remove(self) -> None:
        target = self.home.path / ".config" / "autostart" / "magent.desktop"
        installed = install_autostart(
            executable="/opt/magent/bin/python", system="Linux", home=self.home.path
        )
        self.assertEqual(installed, target)
        text = target.read_text(encoding="utf-8")
        self.assertIn("Name=MAgent", text)
        self.assertIn("-m magent desktop open", text)
        self.assertTrue(autostart_status(system="Linux", home=self.home.path)["installed"])
        self.assertTrue(remove_autostart(system="Linux", home=self.home.path))
        self.assertFalse(remove_autostart(system="Linux", home=self.home.path))


class DesktopAppHeadlessTest(MAgentTestCase):
    """The native UI stays importable and configurable without a display."""

    def test_headless_probe_and_pre_window_initialization(self) -> None:
        from magent.desktop.app import DesktopApp, tkinter_status

        status = tkinter_status()
        self.assertEqual(set(status), {"available", "version", "error"})
        config = self.config()
        config.desktop.backend = "recording"
        config.desktop.logo_theme = "neural"
        app = DesktopApp(config)
        try:
            self.assertEqual(app.appearance.logo_theme, "neural")
            self.assertEqual(app.task_queue.stats()["total"], 0)
            self.assertIsNone(app.root)
        finally:
            app.close()


class DesktopCliAndToolsTest(MAgentTestCase):
    """The desktop feature is discoverable from CLI and the default registry."""

    def test_cli_status_queue_and_logo_selection(self) -> None:
        code, output, _ = self.run_cli("desktop", "status", "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertIn("desktop", payload)
        self.assertEqual(len(payload["logo_options"]), 8)

        code, output, _ = self.run_cli(
            "desktop", "queue", "add", "Write", "release", "notes", "--priority", "70"
        )
        self.assertEqual(code, 0)
        self.assertIn("queued", output)
        code, output, _ = self.run_cli("desktop", "queue", "list")
        self.assertEqual(code, 0)
        self.assertIn("Write release notes", output)
        code, output, _ = self.run_cli("desktop", "logos", "beacon")
        self.assertEqual(code, 0)
        self.assertIn("Agent Beacon", output)
        code, output, _ = self.run_cli("desktop", "status", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["logo"]["id"], "beacon")

    def test_registry_exposes_approval_aware_desktop_tools(self) -> None:
        registry = self.registry()
        desktop = [tool for tool in registry.tools() if tool.spec.category == "desktop"]
        self.assertGreaterEqual(len(desktop), 29)
        changing = registry.require("desktop_click")
        self.assertTrue(changing.spec.requires_approval)
        self.assertIn(changing.spec.risk, {"write", "execute", "destructive"})
        capabilities = registry.require("desktop_capabilities").run(
            {}, context=self.agent().tool_context()
        )
        self.assertTrue(capabilities.ok)
        self.assertIn("capabilities", capabilities.content)

    def test_direct_desktop_tool_dispatch_uses_the_approval_manager(self) -> None:
        from magent.tools import ApprovalManager, ApprovalPolicy, ToolContext
        from magent.tools.approval import auto_approve

        config = self.config()
        config.desktop.backend = "recording"
        config.desktop.fail_safe = False
        config.desktop.action_pause = 0
        registry = self.registry()
        denied_context = ToolContext(config=config, registry=registry)
        denied = registry.require("desktop_click").run({"x": 10, "y": 20}, context=denied_context)
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error_code, "approval_required")

        manager = ApprovalManager(ApprovalPolicy(mode="dangerous"), prompt=auto_approve)
        approved_context = ToolContext(config=config, registry=registry, approval=manager)
        approved = registry.require("desktop_click").run(
            {"x": 10, "y": 20}, context=approved_context
        )
        self.assertTrue(approved.ok)
        self.assertEqual(manager.asked, 1)
        controller = approved_context.extras["__desktop_controller__"]
        self.assertEqual(controller.driver.actions[-1]["action"], "click")

    def test_agent_approval_marker_prevents_a_second_desktop_prompt(self) -> None:
        from magent.testing import ToolScriptBackend
        from magent.tools import ApprovalManager, ApprovalPolicy
        from magent.tools.approval import auto_approve

        config = self.config()
        config.desktop.backend = "recording"
        config.desktop.fail_safe = False
        config.desktop.action_pause = 0
        manager = ApprovalManager(ApprovalPolicy(mode="dangerous"), prompt=auto_approve)
        backend = ToolScriptBackend(
            script=[
                {"tool_calls": [{"name": "desktop_click", "arguments": {"x": 10, "y": 20}}]},
                {"text": "Clicked after approval."},
            ]
        )
        result = self.agent(config=config, backend=backend, approval=manager).run(
            "Click the test desktop"
        )
        self.assertTrue(result.succeeded)
        tool_results = [item for step in result.steps for item in step.tool_results]
        self.assertTrue(tool_results[0].ok)
        self.assertEqual(manager.asked, 1)


if __name__ == "__main__":
    unittest.main()
