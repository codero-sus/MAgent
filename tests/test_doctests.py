"""Run the doctests that are embedded in MAgent's modules.

The package documents its public API with short ``>>>`` examples; running them
here keeps those examples honest instead of letting them rot.
"""

from __future__ import annotations

import doctest
import importlib
import unittest

MODULES: tuple[str, ...] = (
    "magent",
    "magent.agents",
    "magent.cli",
    "magent.concurrency",
    "magent.context",
    "magent.eval",
    "magent.guardrails",
    "magent.memory",
    "magent.observability",
    "magent.output",
    "magent.plugins",
    "magent.prompts",
    "magent.rag",
    "magent.server",
    "magent.session",
    "magent.skills",
    "magent.tools",
    "magent.utils",
)
"""Modules whose docstrings are executed as tests."""

OPTIONFLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
"""Matching options: long output is shortened with ``...`` in the examples."""


def _suite_for(module_name: str) -> unittest.TestSuite:
    """Build a doctest suite for one module.

    Args:
        module_name: dotted module path
    """
    module = importlib.import_module(module_name)
    return doctest.DocTestSuite(module, optionflags=OPTIONFLAGS)


class DoctestTest(unittest.TestCase):
    """Every documented example in the package runs."""

    def test_package_examples(self) -> None:
        failures: list[str] = []
        for module_name in MODULES:
            suite = _suite_for(module_name)
            for case in _iter_cases(suite):
                result = case.run()
                if result.failures or result.errors:
                    detail = (result.failures or result.errors)[0][1].strip().splitlines()[-1]
                    failures.append(f"{module_name}: {detail[:160]}")
        self.assertEqual(failures, [], "\n".join(failures))


def _iter_cases(suite: unittest.TestSuite) -> list[doctest.DocTestCase]:
    """Flatten a suite into its doctest cases."""
    cases: list[doctest.DocTestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            cases.extend(_iter_cases(item))
        elif isinstance(item, doctest.DocTestCase):
            cases.append(item)
    return cases


if __name__ == "__main__":
    unittest.main()
