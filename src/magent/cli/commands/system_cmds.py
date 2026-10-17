"""System commands: ``doctor``, ``config``, ``models``, ``cortex``, ``init``, ``version``.

``magent doctor`` is the one to run when something is wrong: it probes the
backend, checks the sandbox, guardrails, memory, sessions, plugin and skill
directories, prints a row per check with a fix hint, and exits non-zero when a
hard requirement fails.  ``magent config show`` prints the effective
configuration with secrets redacted, so a support conversation can start from the
actual values instead of a guess.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from ..._version import __version__
from ...errors import MAgentError
from ...utils.files import ensure_dir, human_size
from ...utils.text import ellipsize
from ..commands import Command


def version_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent version`` - print version and build details."""
    payload = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "package": str(Path(__file__).resolve().parents[3]),
    }
    if ctx.config.output.format != "text":
        ctx.emit(payload, fmt=ctx.config.output.format)
        return 0
    ctx.print(f"MAgent {__version__}")
    ctx.print(f"python  {payload['python']} ({payload['platform']})")
    ctx.print(f"package {payload['package']}")
    if getattr(args, "verbose", False):
        ctx.print("")
        ctx.key_values(
            [
                ("profile", ctx.config.profile),
                ("config", ctx.config.config_path or "(none)"),
                ("data", ctx.config.data_dir),
                ("cache", ctx.config.cache_dir),
            ]
        )
    return 0


def doctor_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent doctor`` - check the installation and print fixes."""
    checks: list[tuple[str, str, str, str]] = []
    problems = 0

    # configuration
    try:
        ctx.config.validate()
        checks.append(
            ("config", "ok", f"profile {ctx.config.profile}", "edit with `magent config show`")
        )
    except MAgentError as exc:
        problems += 1
        checks.append(("config", "fail", str(exc), "run `magent config init --force`"))

    # backend
    try:
        backend = ctx.backend
        name = getattr(backend, "name", type(backend).__name__)
        kind = type(backend).__name__
        if kind == "EchoBackend":
            checks.append(
                (
                    "backend",
                    "warn",
                    f"{name} (offline)",
                    "start Cortex LLMHoster, then re-run doctor",
                )
            )
        else:
            checks.append(("backend", "ok", name, ""))
        if hasattr(backend, "health"):
            try:
                health = backend.health()
                status = "ok" if health.get("ok", True) else "warn"
                if status == "warn":
                    problems += 0
                checks.append(
                    (
                        "backend health",
                        status,
                        ellipsize(json.dumps(health, default=str), 60),
                        "check the Cortex URL and API key",
                    )
                )
            except Exception as exc:
                checks.append(
                    ("backend health", "warn", str(exc)[:60], "the model host may not be running")
                )
    except MAgentError as exc:
        problems += 1
        checks.append(("backend", "fail", str(exc), "check [model] and [server] in the config"))

    # tools
    try:
        registry = ctx.registry
        categories = len(registry.categories())
        checks.append(("tools", "ok", f"{registry.count()} tools in {categories} categories", ""))
    except Exception as exc:
        problems += 1
        checks.append(("tools", "fail", str(exc), "reinstall or check plugin errors"))

    # sandbox
    try:
        from ...tools import build_sandbox

        sandbox = build_sandbox(ctx.config, workspace=ctx.workspace)
        roots = ", ".join(str(root) for root in sandbox.roots()[:3])
        checks.append(("sandbox", "ok", f"roots: {roots or '(none)'}", ""))
    except Exception as exc:
        problems += 1
        checks.append(("sandbox", "fail", str(exc), "check [tools].workspace/allow_paths"))

    # guardrails
    try:
        guardrails = ctx.guardrails
        if guardrails is None:
            checks.append(("guardrails", "warn", "disabled", "set [guardrails].enabled = true"))
        else:
            checks.append(("guardrails", "ok", guardrails.describe()[:60], ""))
    except Exception as exc:
        problems += 1
        checks.append(("guardrails", "fail", str(exc), "check [guardrails]"))

    # memory
    try:
        store = ctx.memory
        if store is None:
            checks.append(("memory", "warn", "disabled", "set [memory].enabled = true"))
        else:
            stats = store.stats()
            checks.append(
                ("memory", "ok", f"{stats.get('entries', 0)} entries at {store.path}", "")
            )
    except Exception as exc:
        problems += 1
        checks.append(("memory", "fail", str(exc), "check the data directory is writable"))

    # sessions
    try:
        sessions = ctx.sessions
        directory = sessions.store.directory
        ensure_dir(directory)
        checks.append(("sessions", "ok", f"{len(sessions.store)} stored in {directory}", ""))
    except Exception as exc:
        problems += 1
        checks.append(("sessions", "fail", str(exc), "check [session].directory"))

    # plugins
    try:
        loader = ctx.plugin_loader
        if loader is None:
            checks.append(("plugins", "warn", "not loaded", "run `magent plugins check`"))
        else:
            stats = loader.stats()
            level = "ok" if not stats["failed"] else "warn"
            checks.append(
                (
                    "plugins",
                    level,
                    f"{stats['loaded']} loaded, {stats['failed']} failed",
                    "run `magent plugins check`",
                )
            )
    except Exception as exc:
        problems += 1
        checks.append(("plugins", "fail", str(exc), "check the plugin directory"))

    # skills
    try:
        library = ctx.skills
        checks.append(("skills", "ok", library.describe(), ""))
    except Exception as exc:
        problems += 1
        checks.append(("skills", "fail", str(exc), "check .magent/skills"))

    # workspace
    checks.append(
        (
            "workspace",
            "ok" if ctx.workspace.exists() else "warn",
            str(ctx.workspace),
            "pass --workspace",
        )
    )

    # disk
    try:
        usage = shutil.disk_usage(ctx.workspace if ctx.workspace.exists() else Path.cwd())
        level = "ok" if usage.free > 100 * 1024 * 1024 else "warn"
        checks.append(
            ("disk", level, f"{human_size(usage.free)} free", "free space before large runs")
        )
    except OSError as exc:
        checks.append(("disk", "warn", str(exc)[:50], ""))

    if ctx.config.output.format != "text":
        ctx.emit(
            {
                "checks": [
                    {"check": name, "status": status, "detail": detail, "fix": fix}
                    for name, status, detail, fix in checks
                ],
                "problems": problems,
                "version": __version__,
            },
            fmt=ctx.config.output.format,
        )
        return 0 if problems == 0 else 1
    rows = [
        (name, status, ellipsize(detail, 56), ellipsize(fix, 40))
        for name, status, detail, fix in checks
    ]
    ctx.table(rows, ["check", "status", "detail", "fix"], title=f"MAgent {__version__} doctor")
    failed = [name for name, status, _, _ in checks if status == "fail"]
    if failed:
        ctx.notice(f"{len(failed)} check(s) failed: {', '.join(failed)}", level="error")
    else:
        ctx.notice("all required checks passed", level="success")
    return 0 if problems == 0 else 1


def config_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent config`` - show, inspect and write configuration."""
    action = getattr(args, "action", "") or "show"
    config = ctx.config
    if action == "show":
        payload = config.to_dict(redact=not getattr(args, "show_secrets", False))
        if ctx.config.output.format != "text":
            ctx.emit(payload, fmt=ctx.config.output.format)
            return 0
        if getattr(args, "section", ""):
            section = payload.get(args.section)
            if section is None:
                ctx.error(f"unknown section {args.section!r}")
                return 1
            ctx.emit(section, fmt="yaml")
            return 0
        ctx.emit(payload, fmt="yaml")
        return 0
    if action == "paths":
        from ...config import default_config_paths, system_config_dir, user_config_dir

        rows = [(str(path), "yes" if path.exists() else "no") for path in default_config_paths()]
        rows.append((f"user: {user_config_dir()}", "dir"))
        rows.append((f"system: {system_config_dir()}", "dir"))
        rows.append((f"data: {config.data_dir}", "dir"))
        rows.append((f"cache: {config.cache_dir}", "dir"))
        ctx.table(rows, ["path", "state"], title="configuration paths")
        return 0
    if action == "get":
        if not args.key:
            ctx.error("get needs a dotted key, e.g. model.model")
            return 2
        value = _dig(config, args.key)
        if value is None:
            ctx.error(f"no such key: {args.key}")
            return 1
        ctx.emit(
            value, fmt=ctx.config.output.format if ctx.config.output.format != "text" else "text"
        )
        return 0
    if action == "set":
        if not args.key or args.value is None:
            ctx.error("set needs a dotted key and a value")
            return 2
        return _set_config_key(ctx, args.key, args.value)
    if action == "init":
        path = (
            Path(args.output).expanduser()
            if args.output
            else (Path(ctx.config.data_dir) / "magent.toml")
        )
        if path.exists() and not args.force:
            ctx.error(f"{path} exists; pass --force to overwrite")
            return 1
        from ...config import write_example_config

        written = write_example_config(path, force=True)
        ctx.notice(f"wrote {written}", level="success")
        return 0
    if action == "validate":
        try:
            config.validate()
        except MAgentError as exc:
            ctx.error(str(exc))
            return 1
        ctx.notice("configuration is valid", level="success")
        return 0
    ctx.error(f"unknown action {action!r}")
    return 2


def _dig(config: Any, key: str) -> Any:
    """Read a dotted configuration key."""
    current: Any = config
    for part in key.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _set_config_key(ctx: Any, key: str, value: str) -> int:
    """Write one dotted key into the user's configuration file."""
    target = Path(
        ctx.config.config_path or (Path(ctx.config.data_dir) / "magent.toml")
    ).expanduser()
    data: dict[str, Any] = {}
    if target.is_file():
        from ...config import load_toml

        data = load_toml(target) or {}
    section, _, leaf = key.rpartition(".")
    if not section:
        ctx.error("key must be dotted, e.g. `model.model`")
        return 2
    node = data
    for part in section.split("."):
        node = node.setdefault(part, {})
    parsed = _parse_value(value)
    node[leaf] = parsed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_dump_toml(data), encoding="utf-8")
    ctx.notice(f"{key} = {parsed!r} written to {target}", level="success")
    return 0


def _parse_value(value: str) -> Any:
    """Parse a CLI string into a bool, number, list or string."""
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if text.startswith("[") and text.endswith("]"):
        return [item.strip().strip("'\"") for item in text[1:-1].split(",") if item.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _dump_toml(data: dict[str, Any]) -> str:
    """Serialise a nested dict as TOML for the keys MAgent writes."""
    lines: list[str] = []

    def emit(node: dict[str, Any], prefix: str = "") -> None:
        scalars = {key: value for key, value in node.items() if not isinstance(value, dict)}
        tables = {key: value for key, value in node.items() if isinstance(value, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_literal(value)}")
        if scalars or prefix:
            lines.append("")
        for key, value in tables.items():
            emit(value, f"{prefix}.{key}" if prefix else key)

    emit(data)
    return "\n".join(lines).rstrip() + "\n"


def _toml_literal(value: Any) -> str:
    """Render one TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    return json.dumps(str(value))


def models_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent models`` - list the models the backend reports."""
    backend = ctx.backend
    models: list[Any] = []
    error = ""
    try:
        models = list(backend.models()) if hasattr(backend, "models") else []
    except MAgentError as exc:
        error = str(exc)
    if ctx.config.output.format != "text":
        ctx.emit(
            {
                "backend": type(backend).__name__,
                "models": [getattr(m, "id", str(m)) for m in models],
                "error": error,
            },
            fmt=ctx.config.output.format,
        )
        return 0 if not error else 1
    if error:
        ctx.notice(f"could not list models: {error}", level="warning")
    if not models:
        ctx.notice("no models reported; is Cortex LLMHoster running?", level="warning")
        ctx.print(f"backend: {type(backend).__name__} ({getattr(backend, 'name', '')})")
        return 0 if not error else 1
    rows = []
    for model in models:
        rows.append(
            (
                getattr(model, "id", str(model)),
                getattr(model, "runtime", ""),
                ", ".join(getattr(model, "capabilities", []) or [])[:40],
                "default" if getattr(model, "default", False) else "",
            )
        )
    ctx.table(
        rows,
        ["id", "runtime", "capabilities", "note"],
        title=f"{len(rows)} models on {type(backend).__name__}",
    )
    return 0


def cortex_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent cortex`` - talk to Cortex LLMHoster's admin API."""
    from ...llm.cortex_admin import CortexAdmin

    action = getattr(args, "action", "") or "state"
    try:
        admin = CortexAdmin(backend=ctx.backend if hasattr(ctx.backend, "settings") else None)
    except Exception as exc:
        ctx.error(f"cannot reach the Cortex admin API: {exc}")
        return 1
    try:
        if action == "state":
            payload = admin.state()
        elif action == "metrics":
            payload = admin.metrics()
        elif action == "runtime":
            payload = admin.runtime()
        elif action == "hardware":
            payload = admin.hardware()
        elif action == "models":
            payload = {
                "running": [
                    dict(item) if isinstance(item, dict) else {"model": item}
                    for item in admin.iter_running_models()
                ]
            }
        elif action == "cookbook":
            payload = admin.cookbook()
        elif action == "start":
            if not ctx.config.model.model:
                ctx.error("start needs a model id (pass --model)")
                return 2
            payload = admin.start_model(ctx.config.model.model)
        elif action == "stop":
            payload = (
                admin.stop_model(ctx.config.model.model)
                if ctx.config.model.model
                else admin.stop_all_models()
            )
        elif action == "restart":
            if not ctx.config.model.model:
                ctx.error("restart needs a model id (pass --model)")
                return 2
            payload = admin.restart_model(ctx.config.model.model)
        elif action == "update":
            payload = admin.check_update()
        elif action == "auth":
            payload = admin.auth_options()
        elif action == "files":
            payload = admin.model_files()
        else:
            ctx.error(f"unknown cortex action {action!r}")
            return 2
    except MAgentError as exc:
        ctx.error(str(exc))
        return 1
    ctx.emit(
        payload, fmt=ctx.config.output.format if ctx.config.output.format != "text" else "yaml"
    )
    return 0


def init_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent init`` - write a starter configuration, skills directory and plugin."""
    root = Path(args.directory).expanduser() if args.directory else ctx.workspace
    ensure_dir(root)
    written: list[Path] = []
    config_path = root / "magent.toml"
    if config_path.exists() and not args.force:
        ctx.notice(f"{config_path} already exists (use --force to overwrite)", level="warning")
    else:
        from ...config import write_example_config

        written.append(write_example_config(config_path, force=True))
    if getattr(args, "skills", False):
        skills_dir = root / ".magent" / "skills"
        ensure_dir(skills_dir)
        sample = skills_dir / "house-style.md"
        if not sample.exists() or args.force:
            sample.write_text(
                "---\n"
                "name: house-style\n"
                "description: How this project writes code and commits\n"
                "tags: [style]\n"
                "triggers: [house style, our conventions]\n"
                "priority: 70\n"
                "---\n\n"
                "Match the existing code style exactly. Prefer the smallest change that\n"
                "works. Commit messages: imperative subject, then why in the body.\n",
                encoding="utf-8",
            )
            written.append(sample)
    if getattr(args, "plugin", False):
        plugin_dir = root / ".magent" / "plugins" / "example"
        ensure_dir(plugin_dir)
        manifest = plugin_dir / "plugin.json"
        if not manifest.exists() or args.force:
            manifest.write_text(
                json.dumps(
                    {
                        "name": "example",
                        "version": "0.1.0",
                        "entry": "example.py",
                        "description": "An example MAgent plugin",
                        "provides": ["tools", "hooks"],
                        "hooks": ["after_run"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (plugin_dir / "example.py").write_text(
                '"""Example MAgent plugin: one tool and one hook."""\n\n'
                "from magent.tools import tool\n\n\n"
                '@tool(name="example_echo", description="Echo text back", category="text")\n'
                'def example_echo(text: str = "") -> str:\n'
                '    """Return the text unchanged."""\n'
                "    return text\n\n\n"
                "def after_run(payload, **context):\n"
                '    """Note that a run finished."""\n'
                "    return None\n\n\n"
                'HOOKS = {"after_run": after_run}\n',
                encoding="utf-8",
            )
            written.append(manifest)
    if ctx.config.output.format != "text":
        ctx.emit([str(path) for path in written], fmt=ctx.config.output.format)
        return 0
    for path in written:
        ctx.notice(f"wrote {path}", level="success")
    if not written:
        ctx.notice("nothing to do")
    return 0


def completion_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent completion`` - print a shell completion script."""
    from ..parser import completions

    if args.shell not in {"bash", "zsh"}:
        ctx.error(f"unsupported shell {args.shell!r}; try bash or zsh")
        return 2
    ctx.write(completions(args.shell))
    return 0


def _configure_completion(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent completion``."""
    parser.add_argument(
        "shell",
        nargs="?",
        default="bash",
        choices=["bash", "zsh"],
        help="shell to generate completions for",
    )


def commands_command(ctx: Any, args: argparse.Namespace) -> int:
    """``magent commands`` - list every subcommand."""
    from ..commands import describe_commands

    commands = _all_commands()
    if ctx.config.output.format != "text":
        ctx.emit([command.to_dict() for command in commands], fmt=ctx.config.output.format)
        return 0
    ctx.write(describe_commands(commands) + "\n")
    return 0


def _all_commands() -> list[Any]:
    """Every registered command, from the command registry."""
    from ..commands.registry import all_commands

    return all_commands()


def _configure_config(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent config``."""
    parser.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "paths", "get", "set", "init", "validate"],
        help="what to do",
    )
    parser.add_argument("key", nargs="?", default="", help="dotted key for get/set")
    parser.add_argument("value", nargs="?", default=None, help="value for set")
    parser.add_argument("--section", default="", help="show only this section")
    parser.add_argument(
        "--show-secrets",
        dest="show_secrets",
        action="store_true",
        help="do not redact secret values",
    )
    parser.add_argument("--output", default="", help="path for `config init`")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")


def _configure_cortex(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent cortex``."""
    parser.add_argument(
        "action",
        nargs="?",
        default="state",
        choices=[
            "state",
            "metrics",
            "runtime",
            "hardware",
            "models",
            "cookbook",
            "start",
            "stop",
            "restart",
            "update",
            "auth",
            "files",
        ],
        help="admin endpoint to query",
    )
    parser.add_argument(
        "--target",
        default="",
        choices=["", "auto"],
        help="deprecated alias kept for scripts; use --model instead",
    )


def _configure_init(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``magent init``."""
    parser.add_argument("directory", nargs="?", default="", help="where to write (default: .)")
    parser.add_argument("--skills", action="store_true", help="also create .magent/skills")
    parser.add_argument("--plugin", action="store_true", help="also create an example plugin")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")


COMMANDS: tuple[Command, ...] = (
    Command(
        name="version",
        help="Print version information",
        run=version_command,
        group="system",
        aliases=("v",),
        examples=("magent version", "magent version --verbose"),
    ),
    Command(
        name="doctor",
        help="Check the installation and print fixes",
        run=doctor_command,
        group="system",
        needs_backend=True,
        description=(
            "Probes the backend, sandbox, guardrails, memory, sessions, plugins, skills "
            "and disk, prints one row per check with the fix for anything wrong, and "
            "exits non-zero when a required check fails."
        ),
        examples=("magent doctor", "magent doctor --format json"),
    ),
    Command(
        name="config",
        help="Show, inspect and write configuration",
        run=config_command,
        group="system",
        aliases=("cfg",),
        configure=_configure_config,
        examples=(
            "magent config show",
            "magent config show --section tools",
            "magent config set model.temperature 0.2",
            "magent config paths",
        ),
    ),
    Command(
        name="models",
        help="List the models the backend reports",
        run=models_command,
        group="models",
        needs_backend=True,
        examples=("magent models", "magent models --format json"),
    ),
    Command(
        name="cortex",
        help="Query and control Cortex LLMHoster",
        run=cortex_command,
        group="models",
        needs_backend=True,
        configure=_configure_cortex,
        description=(
            "The control plane behind the models: state, metrics, hardware, running "
            "models, the cookbook of known model recipes, model start/stop/restart, "
            "update checks and the auth options Cortex exposes."
        ),
        examples=(
            "magent cortex state",
            "magent cortex metrics",
            "magent cortex start --model qwen2.5-coder-7b",
            "magent cortex stop --model qwen2.5-coder-7b",
        ),
    ),
    Command(
        name="init",
        help="Write a starter config, skills and plugin",
        run=init_command,
        group="system",
        configure=_configure_init,
        examples=("magent init", "magent init --skills --plugin --force"),
    ),
    Command(
        name="commands",
        help="List every subcommand",
        run=commands_command,
        group="system",
        examples=("magent commands", "magent commands --format json"),
    ),
    Command(
        name="completion",
        help="Print a shell completion script",
        run=completion_command,
        group="system",
        configure=_configure_completion,
        examples=('eval "$(magent completion bash)"', "magent completion zsh > _magent"),
    ),
)


__all__ = [
    "COMMANDS",
    "commands_command",
    "completion_command",
    "config_command",
    "cortex_command",
    "doctor_command",
    "init_command",
    "models_command",
    "version_command",
]
