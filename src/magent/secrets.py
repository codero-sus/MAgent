"""Secret resolution and redaction.

Configuration files may reference a secret instead of containing one:

* ``env:CORTEX_API_KEY``        read from the environment
* ``file:~/.secrets/cortex``    read the first line of a file
* ``exec:gpg -dq key.gpg``      run a command and use its stdout
* ``plain:some-value``          literal (discouraged)

Resolution happens once, at load time. Nothing in MAgent ever writes a resolved
secret back to disk, into a log line, into an event payload, or into a session
transcript: :func:`redact_text` is applied at every one of those boundaries.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SecretResolutionError

SCHEMES = ("env", "file", "exec", "plain")
"""Supported secret reference schemes."""

_SECRET_ASSIGN = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|authorization|bearer|cookie|credential)"
    r"\b(\s*[:=]\s*)(['\"]?)([^\s'\",;}]+)\3"
)
"""Regex matching ``key=value`` and ``key: value`` secret assignments."""

_BEARER = re.compile(r"(?i)\b(bearer|basic|token)\s+[A-Za-z0-9._~+/\-=]{8,}")
"""Regex matching an HTTP authorization header value."""

_KNOWN_PREFIXES = re.compile(
    r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|ghr|github_pat|xox[baprs]|AIza|ya29)[-_][A-Za-z0-9_\-]{10,}\b"
)
"""Regex matching well known provider key formats."""

_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{32,}\b")
"""Regex matching long hex strings that are likely credentials."""

REDACTED = "[REDACTED]"
"""Replacement text used by every redaction routine."""


def is_secret_reference(value: Any) -> bool:
    """Report whether a config value is a ``scheme:target`` secret reference."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return any(text.startswith(f"{scheme}:") for scheme in SCHEMES)


def resolve_secret(
    value: str, *, default: str = "", environ: dict[str, str] | None = None, allow_exec: bool = True
) -> str:
    """Resolve a secret reference into its plaintext value.

    Non-reference values are returned unchanged, so callers can pass any config
    string through without checking first.
    """
    if not isinstance(value, str):
        return default
    text = value.strip()
    if not text:
        return default
    env = dict(os.environ if environ is None else environ)
    for scheme in SCHEMES:
        prefix = f"{scheme}:"
        if not text.startswith(prefix):
            continue
        target = text[len(prefix) :]
        if scheme == "env":
            resolved = env.get(target, "")
            if not resolved:
                fallback = env.get(target.upper(), "")
                if not fallback and default:
                    return default
                return fallback or ""
            return resolved.strip()
        if scheme == "file":
            path = Path(target).expanduser()
            try:
                return path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            except OSError as exc:
                raise SecretResolutionError(f"Could not read secret file {path}: {exc}") from exc
            except IndexError as exc:
                raise SecretResolutionError(f"Secret file {path} is empty") from exc
        if scheme == "exec":
            if not allow_exec:
                raise SecretResolutionError("exec: secret references are disabled by policy")
            try:
                completed = subprocess.run(
                    shlex.split(target),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise SecretResolutionError(f"Secret command failed: {exc}") from exc
            if completed.returncode != 0:
                raise SecretResolutionError(
                    f"Secret command exited {completed.returncode}: {completed.stderr.strip()[:200]}"
                )
            return completed.stdout.strip()
        if scheme == "plain":
            return target
    return text


@dataclass(slots=True)
class SecretStore:
    """A tiny encrypted-at-rest-free secret file with strict permissions.

    Values live in ``~/.config/magent/secrets.toml`` (or a path you choose) with
    mode ``0600``. This is deliberately simple: it exists so a user can keep a
    Cortex key out of their shell history, not to be a vault.
    """

    path: Path
    _values: dict[str, str]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._values = {}
        self.load()

    def load(self) -> SecretStore:
        """Read the store from disk, creating it with mode 0600 if missing."""
        if not self.path.is_file():
            self._values = {}
            return self
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SecretResolutionError(f"Could not read secret store {self.path}: {exc}") from exc
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, _, value = stripped.partition("=")
            self._values[key.strip()] = value.strip().strip('"').strip("'")
        return self

    def save(self) -> Path:
        """Write the store atomically and restrict its permissions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# MAgent secret store - mode 0600, never commit this file"]
        for key in sorted(self._values):
            lines.append(f'{key} = "{self._values[key]}"')
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self.path

    def get(self, key: str, default: str = "") -> str:
        """Return one secret value."""
        return self._values.get(key, default)

    def set(self, key: str, value: str, *, save: bool = True) -> None:
        """Store one secret value."""
        self._values[key] = value
        if save:
            self.save()

    def delete(self, key: str, *, save: bool = True) -> bool:
        """Remove one secret value; report whether it existed."""
        existed = self._values.pop(key, None) is not None
        if existed and save:
            self.save()
        return existed

    def keys(self) -> list[str]:
        """Return the stored key names (never the values)."""
        return sorted(self._values)

    def as_references(self) -> dict[str, str]:
        """Return ``key -> file:...`` references safe to put in config."""
        return dict.fromkeys(self._values, f"file:{self.path}")


def redact_text(text: str, *, extra_secrets: list[str] | None = None) -> str:
    """Replace anything that looks like a credential with ``[REDACTED]``."""
    if not text:
        return text
    out = _SECRET_ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    out = _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
    out = _KNOWN_PREFIXES.sub(REDACTED, out)
    for secret in extra_secrets or []:
        if secret and len(secret) >= 6 and secret in out:
            out = out.replace(secret, REDACTED)
    return out


def redact_mapping(data: Any, *, extra_secrets: list[str] | None = None) -> Any:
    """Recursively redact secrets inside mappings, lists, and strings."""
    if isinstance(data, str):
        return redact_text(data, extra_secrets=extra_secrets)
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            name = str(key).lower()
            if any(
                bad in name
                for bad in (
                    "api_key",
                    "apikey",
                    "secret",
                    "token",
                    "password",
                    "authorization",
                    "cookie",
                    "credential",
                )
            ):
                out[key] = REDACTED if value else value
            else:
                out[key] = redact_mapping(value, extra_secrets=extra_secrets)
        return out
    if isinstance(data, (list, tuple)):
        kind = type(data)
        return kind(redact_mapping(item, extra_secrets=extra_secrets) for item in data)
    return data


def mask_value(value: str, *, keep: int = 4) -> str:
    """Mask a secret keeping only its last ``keep`` characters."""
    from .utils.text import mask_secret

    return mask_secret(value, keep=keep)


def contains_secret(text: str, secrets: list[str]) -> list[str]:
    """Return which of ``secrets`` appear verbatim in ``text``."""
    return [secret for secret in secrets if secret and secret in text]


def generate_api_key(prefix: str = "mag") -> str:
    """Generate a fresh random API key for the MAgent HTTP server."""
    from .utils.hashing import random_token

    return f"{prefix}_{random_token(24)}"


__all__ = [
    "REDACTED",
    "SCHEMES",
    "SecretStore",
    "contains_secret",
    "generate_api_key",
    "is_secret_reference",
    "mask_value",
    "redact_mapping",
    "redact_text",
    "resolve_secret",
]
