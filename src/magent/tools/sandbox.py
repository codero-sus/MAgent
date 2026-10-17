"""The tool sandbox: path jail, command policy, and resource limits.

An agent that can run tools can damage a machine. The sandbox is the single
chokepoint every filesystem and shell tool passes through, and it is
deny-by-default:

* paths are resolved (symlinks included) and must land inside an allowed root,
* an explicit deny list wins over the allow list,
* shell commands must match an allowlist pattern unless the policy is open,
* a hard list of command names is always refused,
* reads and writes are byte-capped,
* every decision is recorded for the audit log.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..errors import PermissionDeniedError, SandboxViolationError
from ..logging_ import get_logger

LOGGER = get_logger("sandbox")

ALWAYS_DENIED_COMMANDS: frozenset[str] = frozenset(
    {
        "mkfs",
        "mkfs.ext4",
        "dd",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "rm",
        "rmdir",
        "del",
        "format",
        "fdisk",
        "parted",
        "wipefs",
        "chmod",
        "chown",
        "chgrp",
        "chroot",
        "sudo",
        "su",
        "doas",
        "pkexec",
        "passwd",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "mount",
        "umount",
        "insmod",
        "modprobe",
        "rmmod",
        "iptables",
        "nft",
        "ufw",
        "firewall-cmd",
        "systemctl",
        "service",
        "launchctl",
        "kill",
        "killall",
        "pkill",
        "taskkill",
        "curl",
        "wget",
    }
)
"""Commands never permitted through the sandbox, whatever the policy says."""

SAFE_DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "dir",
    "pwd",
    "echo",
    "cat",
    "type",
    "head",
    "tail",
    "wc",
    "sort",
    "uniq",
    "grep",
    "rg",
    "find",
    "tree",
    "stat",
    "file",
    "du",
    "df",
    "date",
    "whoami",
    "python",
    "python3",
    "node",
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git remote",
    "pytest",
    "make help",
    "cargo check",
    "go vet",
)
"""Commands allowed when the policy is ``allowlist`` and no list is configured."""

DANGEROUS_PATH_PATTERNS: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
    "/root/.ssh",
    "/proc",
    "/sys",
    "/dev",
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gcloud",
    "id_rsa",
    "id_ed25519",
    ".env",
    "credentials",
    "secrets",
)
"""Substrings that make a path sensitive even inside the workspace."""

DenyAction = Literal["raise", "warn"]
"""What the sandbox does when a rule matches."""


@dataclass(slots=True)
class SandboxDecision:
    """One allow/deny decision, recorded for auditing."""

    allowed: bool
    action: str
    target: str
    reason: str
    rule: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision."""
        return {
            "allowed": self.allowed,
            "action": self.action,
            "target": self.target,
            "reason": self.reason,
            "rule": self.rule,
            "ts": self.ts,
        }


@dataclass(slots=True)
class SandboxPolicy:
    """Declarative sandbox configuration."""

    roots: list[str] = field(default_factory=list)
    allow_paths: list[str] = field(default_factory=list)
    deny_paths: list[str] = field(default_factory=list)
    allow_write: bool = True
    allow_read_outside: bool = False
    allow_exec: bool = True
    command_policy: str = "allowlist"
    allow_commands: list[str] = field(default_factory=list)
    deny_commands: list[str] = field(default_factory=list)
    always_deny: list[str] = field(default_factory=lambda: sorted(ALWAYS_DENIED_COMMANDS))
    max_read_bytes: int = 2 * 1024 * 1024
    max_write_bytes: int = 8 * 1024 * 1024
    max_output_chars: int = 32000
    block_sensitive: bool = True
    sensitive_patterns: list[str] = field(default_factory=lambda: list(DANGEROUS_PATH_PATTERNS))
    deny_action: DenyAction = "raise"
    allow_network: bool = False
    allow_domains: list[str] = field(default_factory=list)
    deny_domains: list[str] = field(default_factory=list)
    max_processes: int = 4
    umask: int | None = None

    def validate(self) -> None:
        """Reject policies that cannot be enforced."""
        if self.command_policy not in {"allowlist", "denylist", "open", "disabled"}:
            raise ValueError("command_policy must be allowlist, denylist, open, or disabled")
        if self.max_read_bytes < 0 or self.max_write_bytes < 0:
            raise ValueError("byte limits cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy."""
        import dataclasses

        return dataclasses.asdict(self)


class Sandbox:
    """Enforces a :class:`SandboxPolicy` on paths, commands, and domains."""

    def __init__(
        self, policy: SandboxPolicy | None = None, *, workspace: str | Path | None = None
    ) -> None:
        self.policy = policy or SandboxPolicy()
        self.policy.validate()
        self.workspace = Path(
            workspace or (self.policy.roots[0] if self.policy.roots else Path.cwd())
        ).expanduser()
        if not self.policy.roots:
            self.policy.roots = [str(self.workspace)]
        self._lock = threading.RLock()
        self.decisions: list[SandboxDecision] = []
        self.max_decisions = 500
        self.denials = Counter()
        self.allowances = Counter()

    # -- paths ---------------------------------------------------------------

    def roots(self) -> list[Path]:
        """Return every resolved allowed root directory."""
        out: list[Path] = []
        for entry in list(self.policy.roots) + list(self.policy.allow_paths):
            path = Path(entry).expanduser()
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved not in out:
                out.append(resolved)
        if not out:
            out.append(self.workspace.resolve())
        return out

    def resolve(self, candidate: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve ``candidate`` inside the jail, raising on escape attempts."""
        raw = Path(str(candidate)).expanduser()
        if not raw.is_absolute():
            raw = self.workspace / raw
        try:
            resolved = raw.resolve()
        except OSError as exc:
            raise SandboxViolationError(f"Cannot resolve path {candidate!r}: {exc}") from exc
        self.check_path(resolved, action="resolve")
        if must_exist and not resolved.exists():
            raise SandboxViolationError(f"Path does not exist: {resolved}")
        return resolved

    def check_path(self, path: str | Path, *, action: str = "read") -> SandboxDecision:
        """Verify a path is permitted for ``action``; raise or warn per policy."""
        target = Path(str(path)).expanduser()
        try:
            resolved = target.resolve()
        except OSError:
            resolved = target
        text = str(resolved)
        lowered = text.lower()

        for pattern in self.policy.deny_paths:
            if self._path_matches(resolved, pattern):
                return self._deny(
                    action, text, f"matches deny_paths rule {pattern!r}", "deny_paths"
                )

        if self.policy.block_sensitive:
            for pattern in self.policy.sensitive_patterns:
                needle = str(Path(pattern).expanduser()).lower()
                if needle in lowered or pattern.lower() in lowered:
                    return self._deny(
                        action, text, f"path looks sensitive ({pattern})", "sensitive"
                    )

        allowed_roots = self.roots()
        inside = any(resolved == root or root in resolved.parents for root in allowed_roots)
        if not inside:
            if action == "read" and self.policy.allow_read_outside:
                return self._allow(
                    action, text, "read outside the jail is permitted", "allow_read_outside"
                )
            return self._deny(
                action,
                text,
                f"path is outside the workspace jail ({', '.join(str(r) for r in allowed_roots)})",
                "root",
            )
        if action in {"write", "append", "delete", "chmod"} and not self.policy.allow_write:
            return self._deny(action, text, "writes are disabled by policy", "allow_write")
        return self._allow(action, text, "inside an allowed root", "root")

    def _path_matches(self, resolved: Path, pattern: str) -> bool:
        """Test a resolved path against a glob, prefix, or regex deny rule."""
        text = str(resolved)
        expanded = str(Path(pattern).expanduser())
        if text == expanded or text.startswith(expanded.rstrip("/") + "/"):
            return True
        if any(char in pattern for char in "*?["):
            if fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(resolved.name, pattern):
                return True
        if pattern.startswith("/") or pattern.startswith("~"):
            return False
        try:
            return bool(re.search(pattern, text))
        except re.error:
            return pattern.lower() in text.lower()

    def check_read(self, path: str | Path) -> SandboxDecision:
        """Authorize a read and enforce the byte ceiling."""
        decision = self.check_path(path, action="read")
        target = Path(str(path)).expanduser()
        try:
            size = target.stat().st_size if target.exists() else 0
        except OSError:
            size = 0
        if size > self.policy.max_read_bytes:
            return self._deny(
                "read",
                str(target),
                f"file is {size} bytes, above the {self.policy.max_read_bytes} byte read limit",
                "max_read_bytes",
            )
        return decision

    def check_write(self, path: str | Path, size: int = 0) -> SandboxDecision:
        """Authorize a write and enforce the byte ceiling."""
        decision = self.check_path(path, action="write")
        if size > self.policy.max_write_bytes:
            return self._deny(
                "write",
                str(path),
                f"payload is {size} bytes, above the {self.policy.max_write_bytes} byte write limit",
                "max_write_bytes",
            )
        return decision

    def check_delete(self, path: str | Path) -> SandboxDecision:
        """Authorize a deletion."""
        return self.check_path(path, action="delete")

    # -- commands ------------------------------------------------------------

    def check_command(self, command: str) -> SandboxDecision:
        """Authorize a shell command against the configured policy."""
        if not self.policy.allow_exec:
            return self._deny(
                "exec", command, "command execution is disabled by policy", "allow_exec"
            )
        if self.policy.command_policy == "disabled":
            return self._deny("exec", command, "command_policy is 'disabled'", "policy")
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            return self._deny("exec", command, f"could not parse command: {exc}", "parse")
        if not tokens:
            return self._deny("exec", command, "empty command", "empty")
        program = Path(tokens[0]).name
        always_denied = set(self.policy.always_deny) | ALWAYS_DENIED_COMMANDS
        if program in always_denied:
            return self._deny("exec", command, f"{program!r} is always denied", "always_deny")
        for denied in self.policy.deny_commands:
            if self._command_matches(tokens, denied):
                return self._deny(
                    "exec", command, f"matches deny_commands rule {denied!r}", "deny_commands"
                )
        if self.policy.command_policy == "open":
            return self._allow("exec", command, "command_policy is open", "open")
        allowlist = list(self.policy.allow_commands) or list(SAFE_DEFAULT_ALLOWLIST)
        for allowed in allowlist:
            if self._command_matches(tokens, allowed):
                return self._allow(
                    "exec", command, f"matches allow_commands rule {allowed!r}", "allow_commands"
                )
        if self.policy.command_policy == "denylist":
            return self._allow("exec", command, "denylist policy and no rule matched", "denylist")
        return self._deny(
            "exec",
            command,
            f"{program!r} is not on the command allowlist; add it to tools.allow_commands",
            "allowlist",
        )

    def _command_matches(self, tokens: list[str], rule: str) -> bool:
        """Match a tokenized command against a rule string or glob."""
        joined = " ".join(tokens)
        rule_tokens = rule.split()
        if not rule_tokens:
            return False
        if joined == rule or joined.startswith(rule + " "):
            return True
        if Path(tokens[0]).name == Path(rule_tokens[0]).name and len(rule_tokens) == 1:
            return True
        if len(rule_tokens) > 1 and tokens[: len(rule_tokens)] == rule_tokens:
            return True
        if any(char in rule for char in "*?["):
            return fnmatch.fnmatch(joined, rule) or fnmatch.fnmatch(tokens[0], rule)
        return False

    # -- network -------------------------------------------------------------

    def check_url(self, url: str) -> SandboxDecision:
        """Authorize an outbound URL against the network policy."""
        from ..utils.urls import is_local_url, url_origin

        if not self.policy.allow_network:
            return self._deny(
                "network", url, "network access is disabled by policy", "allow_network"
            )
        origin = url_origin(url)
        for denied in self.policy.deny_domains:
            if self._domain_matches(origin, denied):
                return self._deny("network", url, f"origin {origin} is denied", "deny_domains")
        if self.policy.allow_domains:
            for allowed in self.policy.allow_domains:
                if self._domain_matches(origin, allowed):
                    return self._allow(
                        "network", url, f"origin {origin} is allowlisted", "allow_domains"
                    )
            return self._deny(
                "network", url, f"origin {origin} is not allowlisted", "allow_domains"
            )
        if is_local_url(url):
            return self._allow("network", url, "loopback/private address", "local")
        return self._allow("network", url, "network access permitted", "open")

    def _domain_matches(self, origin: str, rule: str) -> bool:
        """Test an origin against a domain rule, honoring leading wildcards."""
        from urllib.parse import urlsplit

        host = (urlsplit(origin).hostname or "").lower()
        needle = rule.strip().lower().replace("*.", "")
        if not needle:
            return False
        if host == needle:
            return True
        return host.endswith("." + needle) or (rule.startswith("*.") and host.endswith(needle))

    # -- decisions -----------------------------------------------------------

    def _allow(self, action: str, target: str, reason: str, rule: str) -> SandboxDecision:
        """Record and return an allow decision."""
        decision = SandboxDecision(
            allowed=True, action=action, target=target, reason=reason, rule=rule
        )
        self._record(decision)
        return decision

    def _deny(self, action: str, target: str, reason: str, rule: str) -> SandboxDecision:
        """Record a denial and raise unless the policy only warns."""
        decision = SandboxDecision(
            allowed=False, action=action, target=target, reason=reason, rule=rule
        )
        self._record(decision)
        self.denials[rule] += 1
        if self.policy.deny_action == "warn":
            LOGGER.warning("sandbox warning: %s", reason)
            return decision
        raise PermissionDeniedError(
            f"Sandbox blocked {action} on {target}: {reason}",
            details={"action": action, "target": target, "rule": rule},
        )

    def _record(self, decision: SandboxDecision) -> None:
        """Append a decision to the bounded audit trail."""
        with self._lock:
            self.decisions.append(decision)
            if len(self.decisions) > self.max_decisions:
                del self.decisions[: len(self.decisions) - self.max_decisions]
            if decision.allowed:
                self.allowances[decision.rule] += 1

    def recent_decisions(self, limit: int = 50) -> list[SandboxDecision]:
        """Return the most recent decisions."""
        with self._lock:
            return list(self.decisions[-max(1, limit) :])

    def stats(self) -> dict[str, Any]:
        """Return sandbox telemetry."""
        with self._lock:
            return {
                "roots": [str(root) for root in self.roots()],
                "workspace": str(self.workspace),
                "decisions": len(self.decisions),
                "allowances": dict(self.allowances),
                "denials": dict(self.denials),
                "policy": self.policy.to_dict(),
            }

    # -- helpers -------------------------------------------------------------

    def relative(self, path: str | Path) -> str:
        """Render a path relative to the workspace for display."""
        target = Path(str(path)).expanduser()
        try:
            return str(target.resolve().relative_to(self.workspace.resolve()))
        except (ValueError, OSError):
            return str(target)

    def env_for_subprocess(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build a scrubbed environment for a subprocess."""
        base = dict(os.environ)
        for key in list(base):
            upper = key.upper()
            if any(
                token in upper for token in ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "CREDENTIAL")
            ):
                if not upper.startswith("MAGENT_KEEP_"):
                    base.pop(key, None)
        if extra:
            base.update({str(k): str(v) for k, v in extra.items()})
        return base


def build_sandbox(config: Any, *, workspace: str | Path | None = None) -> Sandbox:
    """Build a :class:`Sandbox` from a loaded MAgent configuration."""
    tools = getattr(config, "tools", None)
    guardrails = getattr(config, "guardrails", None)
    root = workspace or (getattr(tools, "workspace", "") if tools else "") or str(Path.cwd())
    policy = SandboxPolicy(
        roots=[str(Path(root).expanduser())],
        allow_paths=[
            str(Path(item).expanduser()) for item in (getattr(tools, "allow_paths", []) or [])
        ],
        deny_paths=list(getattr(tools, "deny_paths", []) or []),
        allow_commands=list(getattr(tools, "allow_commands", []) or []),
        deny_commands=list(getattr(tools, "deny_commands", []) or []),
        max_read_bytes=int(getattr(tools, "max_file_read_bytes", 2 * 1024 * 1024)),
        max_write_bytes=int(getattr(tools, "max_file_write_bytes", 8 * 1024 * 1024)),
        max_output_chars=int(getattr(tools, "max_output_chars", 32000)),
        allow_network=bool(getattr(tools, "allow_network", False)),
        allow_domains=list(getattr(tools, "allow_domains", []) or []),
        deny_domains=list(getattr(tools, "deny_domains", []) or []),
        command_policy="allowlist" if getattr(tools, "allow_commands", []) else "allowlist",
    )
    if guardrails is not None and getattr(guardrails, "url_allowlist_only", False):
        policy.allow_domains = policy.allow_domains or ["localhost", "127.0.0.1"]
    return Sandbox(policy, workspace=root)


def permissive_sandbox(workspace: str | Path = ".") -> Sandbox:
    """Return a sandbox that allows most operations, for trusted local use."""
    policy = SandboxPolicy(
        roots=[str(Path(workspace).expanduser())],
        command_policy="open",
        allow_network=True,
        allow_read_outside=True,
        deny_action="warn",
        block_sensitive=False,
    )
    return Sandbox(policy, workspace=workspace)


def strict_sandbox(workspace: str | Path = ".") -> Sandbox:
    """Return a sandbox that refuses anything not explicitly allowed."""
    policy = SandboxPolicy(
        roots=[str(Path(workspace).expanduser())],
        command_policy="allowlist",
        allow_commands=list(SAFE_DEFAULT_ALLOWLIST),
        allow_network=False,
        allow_read_outside=False,
        block_sensitive=True,
        deny_action="raise",
    )
    return Sandbox(policy, workspace=workspace)


__all__ = [
    "ALWAYS_DENIED_COMMANDS",
    "DANGEROUS_PATH_PATTERNS",
    "SAFE_DEFAULT_ALLOWLIST",
    "Sandbox",
    "SandboxDecision",
    "SandboxPolicy",
    "build_sandbox",
    "permissive_sandbox",
    "strict_sandbox",
]
