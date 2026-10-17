"""The ``doctor`` command: diagnose an installation in one pass.

When MAgent misbehaves, the cause is usually environmental - no Cortex
running, a model that is configured but not loaded, a workspace that is not
writable, a config file that was never read. Each of those produces a confusing
symptom somewhere else.

``doctor`` runs every check independently and reports each with a status, so
one broken check cannot hide the others. It never modifies anything: the
command is safe to run against a production host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from ..._version import (
    MIN_CORTEX_VERSION,
    MIN_PYTHON,
    USER_AGENT,
    __version__,
    is_supported_python,
    version_string,
)
from ..args import CliContext, EXIT_ERROR, EXIT_OK

STATUS_OK = "ok"
"""The check passed."""

STATUS_WARN = "warn"
"""The check passed with a caveat worth reading."""

STATUS_FAIL = "fail"
"""The check failed and something will not work."""

STATUS_SKIP = "skip"
"""The check did not apply."""

STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_WARN, STATUS_FAIL, STATUS_SKIP)
"""Every status a check may report."""

STATUS_EXIT: dict[str, int] = {STATUS_OK: EXIT_OK, STATUS_WARN: EXIT_OK,
                               STATUS_FAIL: EXIT_ERROR, STATUS_SKIP: EXIT_OK}
"""Exit code implied by the worst status seen."""


@dataclass(slots=True)
class CheckResult:
    """The outcome of one diagnostic check."""

    name: str
    status: str = STATUS_OK
    detail: str = ""
    hint: str = ""
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize an unknown status to a warning."""
        if self.status not in STATUSES:
            self.status = STATUS_WARN

    def to_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.data:
            payload["data"] = self.data
        return payload

    def render(self, *, color: bool = True, width: int = 100) -> str:
        """Return one or two lines describing this check."""
        from ...utils.colors import themed

        icon = {STATUS_OK: "+", STATUS_WARN: "!", STATUS_FAIL: "x", STATUS_SKIP: "-"}[self.status]
        name = themed(icon, {"ok": "success", "warn": "warning", "fail": "error",
                             "skip": "muted"}[self.status], enabled=color)
        head = f"{name} {self.name.ljust(20)} {self.detail}"
        lines = [_clip(head, width)]
        if self.hint and self.status in {STATUS_WARN, STATUS_FAIL}:
            lines.append(_clip(f"    hint: {self.hint}", width))
        return "\n".join(lines)


def _clip(text: str, width: int) -> str:
    """Clip one line to a width."""
    limit = max(30, int(width))
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class Doctor:
    """Runs MAgent's diagnostic checks against one context."""

    def __init__(self, context: CliContext, *, quick: bool = False) -> None:
        """Create a doctor.

        Args:
            context: the CLI context carrying config and options
            quick: skip checks that touch the network or the disk heavily
        """
        self.context = context
        self.config = context.config
        self.quick = bool(quick)
        self.results: list[CheckResult] = []

    # -- running -------------------------------------------------------------

    def run(self, *, only: list[str] | None = None) -> list[CheckResult]:
        """Run every check, or only the named ones."""
        checks: list[tuple[str, Callable[[], CheckResult]]] = [
            ("python", self.check_python),
            ("package", self.check_package),
            ("config", self.check_config),
            ("environment", self.check_environment),
            ("backend", self.check_backend),
            ("cortex", self.check_cortex),
            ("models", self.check_models),
            ("tools", self.check_tools),
            ("sandbox", self.check_sandbox),
            ("workspace", self.check_workspace),
            ("memory", self.check_memory),
            ("sessions", self.check_sessions),
            ("guardrails", self.check_guardrails),
            ("disk", self.check_disk),
            ("observability", self.check_observability),
        ]
        wanted = {name.strip().lower() for name in (only or []) if name.strip()}
        self.results = []
        for name, check in checks:
            if wanted and name not in wanted:
                continue
            started = time.perf_counter()
            try:
                result = check()
            except Exception as exc:  # noqa: BLE001 - a broken check must not stop the run
                result = CheckResult(name=name, status=STATUS_FAIL,
                                     detail=f"check raised {type(exc).__name__}: {exc}",
                                     hint="report this as a bug in magent doctor")
            result.duration_ms = (time.perf_counter() - started) * 1000.0
            self.results.append(result)
        return self.results

    @property
    def worst(self) -> str:
        """The most severe status observed."""
        order = [STATUS_FAIL, STATUS_WARN, STATUS_OK, STATUS_SKIP]
        present = {result.status for result in self.results}
        for status in order:
            if status in present:
                return status
        return STATUS_OK

    @property
    def exit_code(self) -> int:
        """Process exit code implied by :attr:`worst`."""
        return STATUS_EXIT.get(self.worst, EXIT_ERROR)

    def counts(self) -> dict[str, int]:
        """Count results per status."""
        tally = {status: 0 for status in STATUSES}
        for result in self.results:
            tally[result.status] += 1
        return tally

    def summary(self) -> dict[str, Any]:
        """Return an aggregate report."""
        return {
            "version": __version__,
            "python": platform.python_version(),
            "checks": len(self.results),
            "counts": self.counts(),
            "worst": self.worst,
            "exit_code": self.exit_code,
            "duration_ms": round(sum(result.duration_ms for result in self.results), 2),
            "results": [result.to_dict() for result in self.results],
        }

    def render(self) -> str:
        """Render the report as text."""
        lines = [self.context.printer.themed(version_string(), "brand")]
        color = self.context.options.color
        width = self.context.options.width
        for result in self.results:
            lines.append(result.render(color=color, width=width))
        tally = self.counts()
        lines.append("")
        lines.append(self.context.printer.themed(
            f"{tally[STATUS_OK]} ok · {tally[STATUS_WARN]} warn · {tally[STATUS_FAIL]} fail · "
            f"{tally[STATUS_SKIP]} skipped",
            "success" if self.worst == STATUS_OK else
            ("warning" if self.worst == STATUS_WARN else "error"),
        ))
        return "\n".join(lines)

    # -- individual checks ---------------------------------------------------

    def check_python(self) -> CheckResult:
        """Verify the interpreter is a supported release."""
        info = sys.version_info
        supported = is_supported_python(info.major, info.minor)
        required = ".".join(str(part) for part in MIN_PYTHON)
        return CheckResult(
            name="python",
            status=STATUS_OK if supported else STATUS_FAIL,
            detail=f"{platform.python_version()} on {platform.system()} "
                   f"{platform.machine()} (needs {required}+)",
            hint="" if supported else f"upgrade to Python {required} or newer",
            data={"version": platform.python_version(), "supported": supported,
                  "implementation": platform.python_implementation()},
        )

    def check_package(self) -> CheckResult:
        """Verify the package imports and report where it came from."""
        import magent

        location = str(Path(getattr(magent, "__file__", "")).parent)
        installed = "site-packages" in location or "dist-packages" in location
        return CheckResult(
            name="package",
            status=STATUS_OK,
            detail=f"magent {magent.__version__} from "
                   f"{'installed package' if installed else 'source checkout'}",
            data={"version": magent.__version__, "path": location, "installed": installed,
                  "user_agent": USER_AGENT},
        )

    def check_config(self) -> CheckResult:
        """Verify configuration resolved and report which files contributed."""
        sources = list(getattr(self.config, "sources", []) or [])
        path = str(getattr(self.config, "config_path", "") or "")
        problems: list[str] = []
        for name in ("server", "model", "agent", "tools", "memory", "guardrails"):
            try:
                self.config.section(name).validate()
            except Exception as exc:  # noqa: BLE001 - collect every problem
                problems.append(f"[{name}] {exc}")
        if problems:
            return CheckResult(name="config", status=STATUS_FAIL,
                               detail=f"{len(problems)} invalid section(s): {problems[0]}",
                               hint="run 'magent config validate' for the full list",
                               data={"problems": problems})
        return CheckResult(
            name="config",
            status=STATUS_OK if sources else STATUS_WARN,
            detail=(f"{len(sources)} file(s) contributed" if sources
                    else "no config file found; using defaults"),
            hint="" if sources else "run 'magent config init' to write a starter file",
            data={"sources": sources, "config_path": path,
                  "profile": str(getattr(self.config, "profile", "") or "")},
        )

    def check_environment(self) -> CheckResult:
        """Report the MAGENT_* and CORTEX_* variables that shape behaviour."""
        from ...config import ENV_MAP

        active = {name: value for name, value in os.environ.items()
                  if name.startswith(("MAGENT_", "CORTEX_"))}
        unknown = [name for name in active if name not in ENV_MAP
                   and not name.startswith("CORTEX_")]
        status = STATUS_OK
        if unknown:
            status = STATUS_WARN
        return CheckResult(
            name="environment",
            status=status,
            detail=(f"{len(active)} variable(s) set" if active else "no MAgent variables set"),
            hint=(f"unrecognized: {', '.join(sorted(unknown))}" if unknown
                  else "" if active else "defaults will be used"),
            data={"set": sorted(active), "unrecognized": sorted(unknown)},
        )

    def check_backend(self) -> CheckResult:
        """Resolve the backend and report what the factory decided."""
        try:
            backend = self.context.backend
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            return CheckResult(name="backend", status=STATUS_FAIL,
                               detail=f"could not build a backend: {exc}",
                               hint="check [server] base_url and [model] backend")
        name = getattr(backend, "name", type(backend).__name__)
        decision = self.context.decision
        reason = getattr(decision, "reason", "") if decision is not None else ""
        offline = name.startswith("echo")
        return CheckResult(
            name="backend",
            status=STATUS_WARN if offline and not getattr(self.context.args, "offline", False)
            else STATUS_OK,
            detail=f"{name} ({reason or 'selected'})",
            hint="" if not offline else
            "using the offline echo backend; answers will not come from a model",
            data={"backend": name, "reason": reason, "offline": offline,
                  "model": str(getattr(decision, "model", "") or "")},
        )

    def check_cortex(self) -> CheckResult:
        """Probe the Cortex LLMHoster HTTP endpoints."""
        if self.quick:
            return CheckResult(name="cortex", status=STATUS_SKIP, detail="skipped (--quick)")
        base_url = str(self.config.server.base_url or "").rstrip("/")
        if not base_url:
            return CheckResult(name="cortex", status=STATUS_WARN,
                               detail="no base_url configured",
                               hint="set MAGENT_CORTEX_URL or [server] base_url")
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        probe_timeout = float(self.config.server.probe_timeout or 3.0)
        reachable = _tcp_reachable(host, port, timeout=min(2.0, probe_timeout))
        if not reachable:
            return CheckResult(
                name="cortex", status=STATUS_FAIL,
                detail=f"{base_url} is not accepting connections",
                hint=f"start Cortex LLMHoster, or check that {host}:{port} is correct",
                data={"base_url": base_url, "host": host, "port": port, "reachable": False},
            )
        health = _http_json(f"{base_url}/health", timeout=probe_timeout,
                            api_key=self._api_key())
        ready = _http_json(f"{base_url}/ready", timeout=probe_timeout,
                           api_key=self._api_key())
        if health is None:
            return CheckResult(name="cortex", status=STATUS_WARN,
                               detail=f"{base_url} accepts connections but /health did not answer",
                               hint="is something else listening on that port?",
                               data={"base_url": base_url, "reachable": True})
        status = STATUS_OK
        detail = f"{base_url} healthy"
        hint = ""
        if isinstance(ready, dict) and ready.get("status") != "ready":
            status = STATUS_WARN
            reason = ready.get("reason", "unknown")
            detail = f"{base_url} up but not ready: {reason}"
            hint = "load or start a model: 'magent cortex models' then 'magent cortex start <id>'"
        return CheckResult(
            name="cortex", status=status, detail=detail, hint=hint,
            data={"base_url": base_url, "health": health, "ready": ready, "reachable": True},
        )

    def check_models(self) -> CheckResult:
        """List the models Cortex exposes and whether the configured one exists."""
        if self.quick:
            return CheckResult(name="models", status=STATUS_SKIP, detail="skipped (--quick)")
        admin = self._admin()
        if admin is None:
            return CheckResult(name="models", status=STATUS_SKIP,
                               detail="no backend admin interface available")
        try:
            models = admin.models(refresh=True)
        except Exception as exc:  # noqa: BLE001 - unreachable server is a finding, not a crash
            return CheckResult(name="models", status=STATUS_WARN,
                               detail=f"could not list models: {exc}",
                               hint="check /v1/models on the Cortex server")
        ids = [str(getattr(item, "id", item) if not isinstance(item, dict) else item.get("id", ""))
               for item in models]
        wanted = str(self.config.model.model or "")
        if not ids:
            return CheckResult(name="models", status=STATUS_WARN,
                               detail="Cortex reports no models",
                               hint="configure a model in Cortex, then re-run doctor")
        if wanted and wanted not in ids:
            return CheckResult(
                name="models", status=STATUS_WARN,
                detail=f"configured model {wanted!r} is not among {len(ids)} available",
                hint=f"choose one of: {', '.join(ids[:6])}",
                data={"available": ids, "configured": wanted},
            )
        return CheckResult(
            name="models", status=STATUS_OK,
            detail=f"{len(ids)} model(s)" + (f", using {wanted}" if wanted else ", using default"),
            data={"available": ids, "configured": wanted},
        )

    def check_tools(self) -> CheckResult:
        """Verify the tool catalogue loads without errors."""
        from ...tools.builtin import load_errors

        try:
            registry = self.context.registry
            count = registry.count()
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name="tools", status=STATUS_FAIL,
                               detail=f"registry could not be built: {exc}")
        errors = load_errors()
        categories = len(registry.categories())
        if errors:
            return CheckResult(
                name="tools", status=STATUS_WARN,
                detail=f"{count} tools in {categories} categories, "
                       f"{len(errors)} module(s) failed to load",
                hint=f"{', '.join(sorted(errors))}",
                data={"count": count, "categories": categories,
                      "errors": {key: value.splitlines()[0] for key, value in errors.items()}},
            )
        if count == 0:
            return CheckResult(name="tools", status=STATUS_WARN,
                               detail="no tools are registered",
                               hint="check [tools] enabled/disabled, or drop --no-tools")
        return CheckResult(name="tools", status=STATUS_OK,
                           detail=f"{count} tools in {categories} categories",
                           data={"count": count, "categories": categories})

    def check_sandbox(self) -> CheckResult:
        """Verify the sandbox can be built and reports a sane policy."""
        from ...tools.sandbox import build_sandbox

        try:
            sandbox = build_sandbox(self.config, workspace=self.context.workspace)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name="sandbox", status=STATUS_FAIL,
                               detail=f"could not build the sandbox: {exc}")
        roots_attr = getattr(sandbox, "roots", None)
        roots_value = roots_attr() if callable(roots_attr) else (roots_attr or [])
        roots = [str(path) for path in roots_value]
        policy = str(getattr(getattr(sandbox, "policy", None), "command_policy", "") or "")
        decision = sandbox.check_path(str(self.context.workspace))
        allowed = bool(getattr(decision, "allowed", False))
        return CheckResult(
            name="sandbox",
            status=STATUS_OK if allowed else STATUS_WARN,
            detail=f"policy {policy or 'default'}, {len(roots)} root(s), workspace "
                   f"{'allowed' if allowed else 'DENIED'}",
            hint="" if allowed else f"{self.context.workspace} is outside the sandbox roots",
            data={"roots": roots, "command_policy": policy, "workspace_allowed": allowed},
        )

    def check_workspace(self) -> CheckResult:
        """Verify the workspace exists and is readable and writable."""
        target = self.context.workspace
        if not target.exists():
            return CheckResult(name="workspace", status=STATUS_FAIL,
                               detail=f"{target} does not exist",
                               hint="create it, or pass --workspace to choose another")
        if not target.is_dir():
            return CheckResult(name="workspace", status=STATUS_FAIL,
                               detail=f"{target} is not a directory")
        readable = os.access(target, os.R_OK)
        writable = os.access(target, os.W_OK)
        status = STATUS_OK if readable and writable else STATUS_WARN
        return CheckResult(
            name="workspace", status=status,
            detail=f"{target} read={'yes' if readable else 'no'} write={'yes' if writable else 'no'}",
            hint="" if writable else "the agent will not be able to create files here",
            data={"path": str(target), "readable": readable, "writable": writable},
        )

    def check_memory(self) -> CheckResult:
        """Verify the memory store can be opened and report its size."""
        if not self.config.memory.enabled:
            return CheckResult(name="memory", status=STATUS_SKIP,
                               detail="disabled by [memory] enabled=false")
        from ...memory import MemoryStore

        try:
            store = MemoryStore.from_config(self.config)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name="memory", status=STATUS_FAIL,
                               detail=f"could not open the memory store: {exc}",
                               hint="check [memory] path and that its directory is writable")
        path = str(getattr(store, "path", "") or "")
        try:
            entries = len(store)
        except Exception:  # noqa: BLE001 - an unreadable store is still a finding
            entries = -1
        writable = _ancestor_writable(Path(path).parent) if path else False
        return CheckResult(
            name="memory",
            status=STATUS_OK if writable else STATUS_WARN,
            detail=(f"{entries} entries at {path}" if entries >= 0
                    else f"store at {path} could not be read"),
            hint="" if writable else "its directory is not writable",
            data={"path": path, "entries": entries, "writable": writable,
                  "retrieval_mode": str(self.config.memory.retrieval_mode)},
        )

    def check_sessions(self) -> CheckResult:
        """Verify the session directory is usable and report stored sessions."""
        if not self.config.session.enabled:
            return CheckResult(name="sessions", status=STATUS_SKIP,
                               detail="disabled by [session] enabled=false")
        from ...session import SessionStore

        store = SessionStore.from_config(self.config)
        directory = Path(store.directory)
        try:
            infos = store.list(limit=1000)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name="sessions", status=STATUS_FAIL,
                               detail=f"could not list sessions: {exc}")
        exists = directory.exists()
        writable = _ancestor_writable(directory)
        status = STATUS_OK if writable else STATUS_WARN
        return CheckResult(
            name="sessions", status=status,
            detail=f"{len(infos)} session(s) in {directory}"
                   + ("" if exists else " (not created yet)"),
            hint="" if writable else "the directory is not writable; transcripts will be lost",
            data={"directory": str(directory), "sessions": len(infos), "writable": writable,
                  "keep_last": store.keep_last},
        )

    def check_guardrails(self) -> CheckResult:
        """Verify guardrails construct and report their configured actions."""
        if not self.config.guardrails.enabled:
            return CheckResult(name="guardrails", status=STATUS_WARN,
                               detail="disabled by [guardrails] enabled=false",
                               hint="prompt injection and PII will not be checked")
        from ...guardrails import build_guardrails

        try:
            guardrails = build_guardrails(self.config)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(name="guardrails", status=STATUS_FAIL,
                               detail=f"could not build guardrails: {exc}")
        verdict = guardrails.inspect_prompt("hello")
        return CheckResult(
            name="guardrails",
            status=STATUS_OK if not verdict.blocked else STATUS_WARN,
            detail=(f"injection={self.config.guardrails.injection_action}, "
                    f"pii={self.config.guardrails.pii_action}, "
                    f"budget={self.config.guardrails.budget_max_tokens or 'off'} tokens"),
            hint="" if not verdict.blocked else "a benign prompt was blocked; check the policy",
            data={"injection_action": self.config.guardrails.injection_action,
                  "pii_action": self.config.guardrails.pii_action,
                  "self_test_blocked": verdict.blocked},
        )

    def check_disk(self) -> CheckResult:
        """Report free space where MAgent writes."""
        from ...utils.files import dir_size, free_space, human_size

        target = Path(getattr(self.config, "data_dir", "") or self.context.workspace)
        try:
            available = free_space(target)
        except OSError as exc:
            return CheckResult(name="disk", status=STATUS_WARN,
                               detail=f"could not stat {target}: {exc}")
        low = available < 100 * 1024 * 1024
        used = 0
        try:
            used = dir_size(target) if target.exists() else 0
        except OSError:
            used = -1
        return CheckResult(
            name="disk",
            status=STATUS_WARN if low else STATUS_OK,
            detail=f"{human_size(available)} free at {target}"
                   + (f", {human_size(used)} used" if used >= 0 else ""),
            hint="less than 100 MB free; sessions and caches may fail to write" if low else "",
            data={"path": str(target), "free_bytes": available, "used_bytes": used},
        )

    def check_observability(self) -> CheckResult:
        """Report where logs, metrics and traces are written."""
        section = self.config.observability
        log_file = str(getattr(section, "log_file", "") or "")
        trace_file = str(getattr(section, "trace_file", "") or "")
        problems: list[str] = []
        for label, path in (("log", log_file), ("trace", trace_file)):
            if path and not _ancestor_writable(Path(path).parent):
                problems.append(f"{label} directory is not writable: {path}")
        detail = (f"level {section.log_level}, format {section.log_format}, "
                  f"metrics={'on' if section.metrics_enabled else 'off'}, "
                  f"traces={'on' if section.trace_enabled else 'off'}")
        return CheckResult(
            name="observability",
            status=STATUS_WARN if problems else STATUS_OK,
            detail=detail,
            hint="; ".join(problems),
            data={"log_level": section.log_level, "log_file": log_file,
                  "trace_file": trace_file, "metrics_enabled": section.metrics_enabled,
                  "trace_enabled": section.trace_enabled, "problems": problems},
        )

    # -- helpers -------------------------------------------------------------

    def _api_key(self) -> str:
        """Return the configured API key, resolving a secret reference."""
        from ...secrets import is_secret_reference, resolve_secret

        raw = str(self.config.server.api_key or "")
        return resolve_secret(raw) if is_secret_reference(raw) else raw

    def _admin(self) -> Any:
        """Return the backend's Cortex admin interface, or ``None``.

        The offline backends also expose ``models``, but with a different
        signature and no control plane, so they are deliberately not treated as
        admins: probing them would report a false failure.
        """
        from ...llm.cortex_admin import CortexAdmin

        backend = self.context.backend
        if isinstance(backend, CortexAdmin):
            return backend
        for attribute in ("admin", "cortex_admin"):
            candidate = getattr(backend, attribute, None)
            if isinstance(candidate, CortexAdmin):
                return candidate
        return None


def _ancestor_writable(path: Path) -> bool:
    """Return whether a directory is writable, creating it in spirit if absent.

    MAgent makes its data directories on first write, so a path that does not
    exist yet is not a problem - only an existing ancestor that refuses writes
    is. Checking the path itself would report every fresh install as broken.
    """
    candidate = Path(path) if str(path) else Path(".")
    for parent in [candidate, *candidate.parents]:
        if parent.exists():
            return os.access(parent, os.W_OK)
    return False


def _tcp_reachable(host: str, port: int, *, timeout: float = 2.0) -> bool:
    """Return whether a TCP connection to ``host:port`` succeeds."""
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=max(0.1, timeout)):
            return True
    except OSError:
        return False


def _http_json(url: str, *, timeout: float = 3.0, api_key: str = "") -> Any:
    """Fetch a URL and parse JSON, returning ``None`` on any failure."""
    from ...http import HTTPClient, HTTPRequest

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with HTTPClient(timeout=timeout) as client:
            response = client.send(HTTPRequest(url=url, method="GET", headers=headers))
        if not response.ok:
            return {"status": "error", "code": response.status}
        return response.json()
    except Exception:  # noqa: BLE001 - a probe failure is data, not an error
        return None


def register(subparsers: Any) -> None:
    """Add the ``doctor`` subcommand."""
    parser = subparsers.add_parser("doctor", help="diagnose the installation",
                                   description="Run every health check and report the results.")
    parser.add_argument("--quick", action="store_true",
                        help="skip network probes")
    parser.add_argument("--only", metavar="NAME", action="append", default=[],
                        help="run only these checks (repeatable)")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on warnings as well as failures")
    parser.set_defaults(handler=cmd_doctor)


def cmd_doctor(args: argparse.Namespace, context: CliContext) -> int:
    """Run the diagnostics and print the report."""
    doctor = Doctor(context, quick=bool(getattr(args, "quick", False)))
    doctor.run(only=list(getattr(args, "only", []) or []))
    if context.options.fmt == "json":
        context.emit(doctor.summary())
    else:
        context.printer.write(doctor.render(), min_level="quiet")
    code = doctor.exit_code
    if getattr(args, "strict", False) and doctor.worst == STATUS_WARN:
        code = EXIT_ERROR
    return code


def cmd_version(args: argparse.Namespace, context: CliContext) -> int:
    """Print version information."""
    import magent

    payload = {
        "version": magent.__version__,
        "codename": magent.CODENAME,
        "channel": magent.BUILD_CHANNEL,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "min_python": ".".join(str(part) for part in MIN_PYTHON),
        "min_cortex": MIN_CORTEX_VERSION,
        "cortex_repo": magent.CORTEX_REPO,
        "user_agent": USER_AGENT,
    }
    if getattr(args, "long", False) or context.options.at_least("verbose"):
        payload["tools"] = context.registry.count()
        payload["path"] = str(Path(magent.__file__).parent)
    if context.options.fmt == "json":
        context.emit(payload)
    else:
        context.printer.write(version_string(), min_level="quiet")
        if getattr(args, "long", False) or context.options.at_least("verbose"):
            from ...utils.tables import render_key_value

            context.printer.write(render_key_value(
                [(key, value) for key, value in payload.items() if key != "version"]),
                min_level="quiet")
    return EXIT_OK


def register_version(subparsers: Any) -> None:
    """Add the ``version`` subcommand."""
    parser = subparsers.add_parser("version", help="print version information",
                                   description="Show the MAgent, Python and Cortex versions.")
    parser.add_argument("-l", "--long", action="store_true", help="include paths and tool count")
    parser.set_defaults(handler=cmd_version)


__all__ = [
    "STATUSES",
    "STATUS_FAIL",
    "STATUS_OK",
    "STATUS_SKIP",
    "STATUS_WARN",
    "CheckResult",
    "Doctor",
    "cmd_doctor",
    "cmd_version",
    "register",
    "register_version",
]
