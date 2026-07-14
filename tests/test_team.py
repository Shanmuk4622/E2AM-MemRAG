from __future__ import annotations

import unittest

from e2am_memrag.team import (
    build_team_plan,
    effective_periodic_interval_seconds,
    verify_team_plan,
)


class TeamPlanTests(unittest.TestCase):
    def test_four_worker_plan_is_stable_complete_and_staggered(self) -> None:
        left = build_team_plan(
            experiment_id="pilot-v1",
            worker_ids=["friend-02", "owner-00", "friend-01", "friend-03"],
        )
        right = build_team_plan(
            experiment_id="pilot-v1",
            worker_ids=["owner-00", "friend-01", "friend-02", "friend-03"],
        )
        self.assertEqual(left, right)
        self.assertEqual(left["source"], {})
        self.assertEqual(left["worker_count"], 4)
        self.assertEqual(left["sync"]["effective_interval_seconds"], 1200)
        self.assertTrue(left["sync"]["ten_minute_target_supported"])
        partitions = [
            partition
            for worker in left["workers"]
            for partition in worker["logical_partitions"]
        ]
        self.assertEqual(sorted(partitions), list(range(256)))
        self.assertEqual(len(partitions), len(set(partitions)))
        self.assertEqual(
            [worker["sync_stagger_seconds"] for worker in left["workers"]],
            [0, 300, 600, 900],
        )
        self.assertEqual(verify_team_plan(left), left)

    def test_six_workers_slow_automatically_under_shared_budget(self) -> None:
        self.assertEqual(effective_periodic_interval_seconds(5), 1200)
        self.assertEqual(effective_periodic_interval_seconds(6), 1200)

    def test_tampered_plan_is_rejected(self) -> None:
        plan = build_team_plan(
            experiment_id="pilot-v1",
            worker_ids=["owner-00", "friend-01"],
            partition_count=8,
        )
        plan["workers"][0]["logical_partitions"].append(7)
        with self.assertRaises(ValueError):
            verify_team_plan(plan)


if __name__ == "__main__":
    unittest.main()
