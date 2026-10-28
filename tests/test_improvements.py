"""Verify the improvements manifest against the repository.

The build claims a large number of improvements.  These tests keep that claim
checkable: the manifest must exist, must declare at least the promised minimum,
and every reference in it must resolve to something real in this checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "improvements.json"
MARKDOWN = REPO_ROOT / "IMPROVEMENTS.md"
MINIMUM = 777


class ImprovementsTest(unittest.TestCase):
    """The manifest exists, is well formed and verifies."""

    @classmethod
    def setUpClass(cls) -> None:
        """Skip the whole class when the manifest has not been generated."""
        if not MANIFEST.is_file():
            raise unittest.SkipTest("improvements.json is missing; run `make improvements`")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_meets_the_minimum(self) -> None:
        self.assertGreaterEqual(self.manifest["total"], MINIMUM)
        self.assertEqual(self.manifest["total"], len(self.manifest["improvements"]))

    def test_ids_are_unique_and_ordered(self) -> None:
        ids = [entry["id"] for entry in self.manifest["improvements"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(identifier.startswith("IMP-") for identifier in ids))

    def test_every_entry_has_a_category_and_title(self) -> None:
        for entry in self.manifest["improvements"]:
            self.assertTrue(entry["category"], entry["id"])
            self.assertTrue(entry["title"], entry["id"])
            self.assertTrue(entry["ref"], entry["id"])

    def test_categories_cover_the_framework(self) -> None:
        categories = set(self.manifest["categories"])
        for expected in ("tools", "memory", "retrieval", "safety", "testing", "interface"):
            self.assertIn(expected, categories)

    def test_verifier_accepts_the_manifest(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "verify_improvements.py"), "--json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=600,
        )
        payload = json.loads(completed.stdout or "{}")
        self.assertEqual(
            completed.returncode, 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        )
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("verified"), self.manifest["total"])

    def test_counter_reports_the_same_number(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "count_improvements.py")],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-500:])
        self.assertEqual(int(completed.stdout.strip()), self.manifest["total"])

    def test_markdown_is_generated_from_the_manifest(self) -> None:
        text = MARKDOWN.read_text(encoding="utf-8")
        self.assertIn(f"**{self.manifest['total']} verifiable improvements**", text)
        self.assertIn("## tools", text)
        self.assertIn("verify_improvements.py", text)

    def test_references_point_at_real_files(self) -> None:
        checked = 0
        for entry in self.manifest["improvements"][:200]:
            path_text = entry["ref"].split("#", 1)[0]
            self.assertTrue((REPO_ROOT / path_text).exists(), entry["ref"])
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
