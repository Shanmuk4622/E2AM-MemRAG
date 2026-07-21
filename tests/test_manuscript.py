from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_manuscript.py"
SPEC = importlib.util.spec_from_file_location("validate_manuscript", SCRIPT)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ManuscriptTests(unittest.TestCase):
    def test_complete_manuscript_validates(self) -> None:
        report = VALIDATOR.validate()
        self.assertEqual(report["status"], "PASS")
        self.assertGreaterEqual(report["bibliography_entries"], 25)
        self.assertEqual(report["bibliography_entries"], report["citations_used"])

    def test_stage06_policy_is_pinned_and_fail_closed(self) -> None:
        policy_path = ROOT / "paper" / "data" / "raw" / "stage06_frozen_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(VALIDATOR.sha256(policy_path), "fda2bda0a715227162ceead0ff032f2228999f6dc9f4bb22890ec23a4acdab14")
        self.assertFalse(policy["validation_selection"]["feasible"])
        self.assertEqual(policy["tau"], 1.0)
        self.assertFalse(policy["test_accessed"])
        self.assertFalse(policy["validation_selection"]["protocol_amendment"]["hypothesis_claim_allowed_when_infeasible"])

    def test_measurement_boundary_is_explicit(self) -> None:
        text = VALIDATOR.all_tex().lower()
        self.assertIn("generation-window selected-gpu board energy", text)
        self.assertIn("whole-system", text)
        self.assertIn("carbon", text)
        self.assertNotIn("end-to-end selected-gpu board-energy accounting", text)

    def test_routability_diagnostics_match_frozen_matrix(self) -> None:
        pools = {
            row["pool_id"]: row
            for row in VALIDATOR.read_csv("routability_pools.csv")
        }
        self.assertEqual(int(pools["deployable"]["actions"]), 11)
        self.assertAlmostEqual(float(pools["deployable"]["best_fixed_success"]), 15 / 120)
        self.assertAlmostEqual(float(pools["deployable"]["oracle_success"]), 15 / 120)
        self.assertAlmostEqual(float(pools["deployable"]["routing_headroom"]), 0.0)
        self.assertAlmostEqual(float(pools["offline_reference"]["oracle_success"]), 87 / 120)
        self.assertAlmostEqual(float(pools["all_retained"]["oracle_success"]), 88 / 120)

    def test_reviewer_sensitive_language_is_absent(self) -> None:
        text = VALIDATOR.all_tex().lower()
        for phrase in ("safe route", "safe fallback", "complete telemetry", "prevent parametric memorization"):
            self.assertNotIn(phrase, text)

    def test_success_preserving_cost_bound_is_not_overstated(self) -> None:
        row = VALIDATOR.read_csv("success_preserving_cost_oracle.csv")[0]
        self.assertEqual(int(row["baseline_successes"]), 15)
        self.assertEqual(int(row["oracle_successes"]), 15)
        self.assertEqual(int(row["selections_changed"]), 1)
        self.assertAlmostEqual(float(row["mean_saving_gpu_joules"]), 0.04890513398299845)
        text = VALIDATOR.all_tex().lower()
        self.assertIn("zero success headroom", text)
        self.assertNotIn("has zero cost headroom", text)

    def test_no_reference_or_bibliography_drift(self) -> None:
        text = VALIDATOR.all_tex()
        bib = VALIDATOR.bibliography_keys()
        citations = VALIDATOR.citation_keys(text)
        self.assertEqual(len(bib), len(set(bib)))
        self.assertEqual(citations, set(bib))
        label_report = VALIDATOR.label_audit(text)
        self.assertEqual(label_report["duplicates"], [])
        self.assertEqual(label_report["unresolved"], [])


if __name__ == "__main__":
    unittest.main()
