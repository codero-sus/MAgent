"""MAgent version metadata.

The version string is the single source of truth for the package, the CLI
``--version`` flag, the HTTP ``/api/info`` route, and the ``User-Agent`` header
MAgent sends to Cortex LLMHoster.
"""

from __future__ import annotations

__version__ = "1.0.0"
"""Semantic version of MAgent."""

__version_info__ = (1, 0, 0)
"""Version split into integers for comparison."""

RELEASE_NAME = "Mr. Agent"
"""Human friendly release name."""

CODENAME = "Genesis"
"""Codename for the 1.0 line."""

BUILD_CHANNEL = "stable"
"""Distribution channel: ``stable``, ``beta`` or ``dev``."""

MIN_PYTHON = (3, 11)
"""Oldest supported CPython release."""

MIN_CORTEX_VERSION = "0.2.0"
"""Oldest Cortex LLMHoster version whose API MAgent relies on."""

CORTEX_REPO = "https://github.com/codero-sus/Cortex_LLMHoster"
"""Upstream Cortex LLMHoster repository."""

HOMEPAGE = "https://the-codero.web.app/"
"""Author project homepage."""

USER_AGENT = f"MAgent/{__version__} (+{HOMEPAGE})"
"""Default HTTP ``User-Agent`` for outbound requests."""

IMPROVEMENT_COUNT_TARGET = 777
"""Minimum number of registered, verified improvements MAgent promises."""


def version_string() -> str:
    """Return the full human readable version banner."""
    return f"MAgent {__version__} ({CODENAME}) channel={BUILD_CHANNEL}"


def is_supported_python(major: int, minor: int) -> bool:
    """Report whether a given interpreter release is supported."""
    return (major, minor) >= MIN_PYTHON


__all__ = [
    "BUILD_CHANNEL",
    "CODENAME",
    "CORTEX_REPO",
    "HOMEPAGE",
    "IMPROVEMENT_COUNT_TARGET",
    "MIN_CORTEX_VERSION",
    "MIN_PYTHON",
    "RELEASE_NAME",
    "USER_AGENT",
    "__version__",
    "__version_info__",
    "is_supported_python",
    "version_string",
]
