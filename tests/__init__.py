"""MAgent's test suite.

Everything here runs on a stock CPython 3.11+ interpreter with ``unittest``:
no pytest, no network, no running Cortex LLMHoster, and no writes outside a
temporary directory. Run it with::

    make test          # everything
    make unit          # the fast core subset (test_core_*.py)

Tests that need configuration isolate themselves through
:func:`tests.support.isolated_config`, which points ``XDG_DATA_HOME`` and
``XDG_CONFIG_HOME`` at a temporary directory so a test run cannot touch - or be
touched by - the developer's real ``~/.local/share/magent``.

Anything that would need a live model uses the doubles in
:mod:`magent.testing` instead: :class:`~magent.llm.echo.EchoBackend` for text,
:func:`~magent.testing.fixtures.make_backend` for scripted tool calls, and
:class:`~magent.testing.fake_cortex.FakeCortex` for the hoster's HTTP surface.
"""

from __future__ import annotations
