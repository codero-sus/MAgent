"""Repository-level tests for naming, docs, examples, presets, and automation.

These contracts are easy to regress while the Python suite remains green: the
product spelling, a usable README, packaged documentation, runnable examples,
valid preset TOML, build assets, and CI commands are all part of a real release.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from magent._version import IMPROVEMENT_COUNT_TARGET, version_string
from magent.config import load_config
from magent.improvements import collect, verify

ROOT = Path(__file__).resolve().parents[1]
"""Repository root used by repository-shape tests."""


class NamingTests(unittest.TestCase):
    """The user correction is permanent: product spelling is MAgent."""

    def test_readme_starts_with_product_name(self) -> None:
        """The main landing page is titled exactly MAgent."""
        first = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first, "# MAgent")

    def test_distribution_name_is_magent_with_exact_display_case(self) -> None:
        """Project metadata preserves display casing while indexes normalize it."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "MAgent"', pyproject)

    def test_source_has_no_wrong_magent_spelling(self) -> None:
        """User-facing source must never spell the product 'Magent'."""
        offenders: list[str] = []
        for path in (ROOT / "src").rglob("*"):
            if path.suffix not in {".py", ".html"} or "__pycache__" in path.parts:
                continue
            if "Magent" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_version_banner_starts_with_magent(self) -> None:
        """Release metadata uses the correct product spelling."""
        self.assertRegex(version_string(), r"^MAgent \d+\.\d+\.\d+")

    def test_lowercase_name_is_reserved_for_technical_identifiers(self) -> None:
        """The package and console script remain conventional lowercase ids."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('magent = "magent.cli.main:main"', pyproject)
        self.assertTrue((ROOT / "src" / "magent" / "__init__.py").is_file())


class DocumentationTests(unittest.TestCase):
    """Documentation is broad, substantive, linked, and packaged."""

    def test_readme_is_substantive(self) -> None:
        """The old two-line stub must never return."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertGreater(len(text), 10_000)
        for heading in ("## Install", "## Python API", "## Plugins",
                        "## HTTP server and web console", "## Verification and tests"):
            self.assertIn(heading, text)

    def test_documentation_has_at_least_fifteen_pages(self) -> None:
        """MAgent mirrors the breadth expected from a framework."""
        pages = sorted((ROOT / "docs").glob("*.md"))
        self.assertGreaterEqual(len(pages), 15)
        for page in pages:
            text = page.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# "), page.name)
            self.assertGreater(len(text), 500, page.name)

    def test_readme_local_doc_links_resolve(self) -> None:
        """Every docs/*.md link on the landing page names a real file."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        links = re.findall(r"\((docs/[^)#]+\.md)\)", text)
        self.assertGreaterEqual(len(links), 10)
        missing = [link for link in links if not (ROOT / link).is_file()]
        self.assertEqual(missing, [])

    def test_mkdocs_navigation_names_real_pages(self) -> None:
        """The documentation site cannot reference a missing page."""
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        names = re.findall(r": ([a-z0-9-]+\.md)$", config, flags=re.MULTILINE)
        self.assertGreaterEqual(len(names), 15)
        self.assertEqual([name for name in names if not (ROOT / "docs" / name).is_file()], [])

    def test_wheel_force_includes_docs_and_console(self) -> None:
        """Installed users receive docs and the embedded UI."""
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"docs" = "magent/_docs"', pyproject)
        self.assertIn('"src/magent/server/static/index.html"', pyproject)


class PresetTests(unittest.TestCase):
    """Agent presets parse through the actual layered configuration loader."""

    def test_every_agent_preset_validates(self) -> None:
        """No preset contains an unknown or invalid typed value."""
        presets = sorted((ROOT / "agents").glob("*.toml"))
        self.assertGreaterEqual(len(presets), 6)
        for path in presets:
            with self.subTest(path=path.name):
                config = load_config(path=path, use_env=False, required=True)
                self.assertEqual(config.agent.persona, path.stem)
                self.assertIn(config.agent.loop_mode, {"react", "plan", "reflect", "single"})

    def test_skills_are_executable_prompt_files(self) -> None:
        """Each skill has a title, steps, and a done/safety condition."""
        skills = [path for path in sorted((ROOT / "skills").glob("*.md"))
                  if path.name != "README.md"]
        self.assertGreaterEqual(len(skills), 6)
        for path in skills:
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Skill:"), path.name)
            self.assertGreaterEqual(len(re.findall(r"^\d+\. ", text, flags=re.MULTILINE)), 5,
                                    path.name)
            self.assertGreater(len(text), 500, path.name)

    def test_skill_readme_uses_real_run_file_flag(self) -> None:
        """Skill instructions point at the implemented CLI surface."""
        text = (ROOT / "skills" / "README.md").read_text(encoding="utf-8")
        self.assertIn("magent", text)
        self.assertIn("run --file", text)


class ExampleTests(unittest.TestCase):
    """Examples compile and execute without external services."""

    def test_eight_numbered_examples_exist(self) -> None:
        """The examples cover the same breadth documented in the README."""
        examples = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
        self.assertEqual(len(examples), 8)

    def test_every_example_executes(self) -> None:
        """Examples are runnable programs, not illustrative pseudocode."""
        examples = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
        with tempfile.TemporaryDirectory(prefix="magent-example-tests-") as temporary:
            env = dict(os.environ)
            env.update({
                "PYTHONPATH": str(ROOT / "src"),
                "XDG_DATA_HOME": str(Path(temporary) / "data"),
                "XDG_CONFIG_HOME": str(Path(temporary) / "config"),
                "XDG_CACHE_HOME": str(Path(temporary) / "cache"),
            })
            env.pop("MAGENT_CONFIG", None)
            for path in examples:
                with self.subTest(path=path.name):
                    completed = subprocess.run(
                        [sys.executable, str(path)], cwd=ROOT, env=env,
                        text=True, capture_output=True, timeout=30, check=False,
                    )
                    self.assertEqual(completed.returncode, 0,
                                     f"{completed.stdout}\n{completed.stderr}")
                    self.assertTrue(completed.stdout.strip(), path.name)


class AutomationTests(unittest.TestCase):
    """Build, CI, and verification entry points exist and call real checks."""

    def test_ci_matrix_and_build_job_exist(self) -> None:
        """CI covers supported Pythons and inspects wheel contents."""
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        for version in ("3.11", "3.12", "3.13"):
            self.assertIn(version, workflow)
        self.assertIn("unittest discover", workflow)
        self.assertIn("verify_improvements.py", workflow)
        self.assertIn("python -m build", workflow)
        self.assertIn("magent/_docs/index.md", workflow)

    def test_makefile_check_runs_all_verifiers(self) -> None:
        """One local target covers tests, manifest, and both smoke suites."""
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("check: test verify smoke smoke-tools", makefile)
        self.assertIn("scripts/smoke.py", makefile)
        self.assertIn("scripts/smoke_tools.py", makefile)

    def test_improvement_inventory_resolves(self) -> None:
        """The live collector exceeds 777 and every reference resolves."""
        entries = collect(ROOT)
        report = verify(entries, ROOT)
        self.assertTrue(report.ok, report.render())
        self.assertEqual(report.failed, 0)
        self.assertGreaterEqual(report.total, IMPROVEMENT_COUNT_TARGET)


if __name__ == "__main__":
    unittest.main()
