from __future__ import annotations

import unittest

from e2am_memrag.gates import evaluate_bootstrap_gate


class BootstrapGateTests(unittest.TestCase):
    def _evaluate(self, phase: str, **changes):
        values = {
            "phase": phase,
            "preflight": {
                "disk_ok": True,
                "visible_gpu_count": 1,
                "gpu_name": "Tesla T4",
            },
            "completed_units": 32,
            "expected_units": 32,
            "shard_validation": {"rows": 32, "shards": 4},
            "energy": {
                "available": True,
                "energy_joules": 12.5,
                "duration_seconds": 2.2,
                "samples": 12,
                "gpu_uuid": "GPU-test",
            },
            "checkpoint_valid": True,
            "source_verified": True,
            "environment_verified": True,
            "sync_results": [{"status": "CLEAN", "complete": True, "verified": True}],
            "restore_result": {"status": "RESTORED"},
            "fresh_restore_root": True,
        }
        values.update(changes)
        return evaluate_bootstrap_gate(**values)

    def test_upload_alone_can_never_be_a_hard_pass(self) -> None:
        report = self._evaluate("REMOTE_UPLOAD")
        self.assertFalse(report["hard_pass"])
        self.assertTrue(report["upload_verified"])
        self.assertEqual(report["status"], "UPLOAD_VERIFIED_RESTART_REQUIRED")

    def test_only_fresh_verified_restore_is_a_hard_pass(self) -> None:
        report = self._evaluate("REMOTE_RESTORE")
        self.assertTrue(report["hard_pass"])
        self.assertEqual(report["status"], "PASS")

    def test_nonfresh_restore_is_rejected(self) -> None:
        report = self._evaluate("REMOTE_RESTORE", fresh_restore_root=False)
        self.assertFalse(report["hard_pass"])
        self.assertEqual(report["status"], "RESTORE_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
