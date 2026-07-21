from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


class PaperResultsPackageTest(unittest.TestCase):
    def test_results_manifest_verifies_every_packaged_file(self) -> None:
        manifest = json.loads((PAPER / "RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_main_commit"], "0b2405d9cca43fd04e35f792fdc4664405154fc6")
        self.assertFalse(manifest["hypothesis_pass"])
        self.assertGreaterEqual(len(manifest["files"]), 30)
        for record in manifest["files"]:
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(path.stat().st_size, record["bytes"], record["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
                record["path"],
            )

    def test_completion_and_hypothesis_are_not_conflated(self) -> None:
        raw = PAPER / "data" / "raw"
        success = json.loads((raw / "_SUCCESS.json").read_text(encoding="utf-8"))
        hypothesis = json.loads(
            (raw / "HYPOTHESIS_RESULT.json").read_text(encoding="utf-8")
        )
        self.assertEqual(success["status"], "PASS")
        self.assertTrue(success["details"]["experiment_complete"])
        self.assertTrue(success["details"]["fresh_restore_verified"])
        self.assertFalse(hypothesis["hypothesis_pass"])
        self.assertFalse(hypothesis["energy_reduction_pass"])
        self.assertTrue(hypothesis["quality_noninferiority_pass"])
        self.assertTrue(hypothesis["operating_constraints_pass"])

    def test_trace_audit_has_exact_frozen_coverage(self) -> None:
        path = PAPER / "data" / "derived" / "trace_audit.csv"
        with path.open(encoding="utf-8") as handle:
            rows = {row["trace_set"]: row for row in csv.DictReader(handle)}
        self.assertEqual(int(rows["clean"]["rows"]), 2040)
        self.assertEqual(int(rows["clean"]["routes"]), 17)
        self.assertEqual(int(rows["clean"]["duplicate_unit_ids"]), 0)
        self.assertEqual(int(rows["clean"]["execution_failures"]), 0)
        self.assertEqual(float(rows["clean"]["energy_coverage"]), 1.0)
        self.assertEqual(int(rows["robustness"]["rows"]), 1440)
        self.assertEqual(int(rows["robustness"]["duplicate_unit_ids"]), 0)
        self.assertEqual(int(rows["robustness"]["execution_failures"]), 0)

    def test_primary_claims_are_honest(self) -> None:
        report = (PAPER / "RESULTS_AUDIT.md").read_text(encoding="utf-8")
        self.assertIn("confirmatory hypothesis **did not pass**", report)
        self.assertIn("floor effect", report)
        self.assertIn("not an execution failure", report)
        self.assertIn("action-pool routability", report)
        self.assertIn("0.0 pp", report)
        self.assertNotIn("hf_", report)


if __name__ == "__main__":
    unittest.main()
