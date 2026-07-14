from __future__ import annotations

import unittest

from e2am_memrag.identity import make_unit_id
from e2am_memrag.provenance import ExperimentSpec
from e2am_memrag.work_plan import WorkerAssignment, validate_work_plan


class ProvenanceAndWorkPlanTests(unittest.TestCase):
    def test_spec_is_stable_and_rejects_secret_keys(self) -> None:
        left = ExperimentSpec.create(config={"a": 1, "b": 2})
        right = ExperimentSpec.create(config={"b": 2, "a": 1})
        self.assertEqual(left.spec_hash, right.spec_hash)
        with self.assertRaises(ValueError):
            ExperimentSpec.create(config={"hf_token": "do-not-store"})
        benign = ExperimentSpec.create(
            model={"tokenizer_revision": "abc"}, decoding={"max_new_tokens": 128}
        )
        self.assertTrue(benign.spec_hash)
        with self.assertRaises(ValueError):
            ExperimentSpec.create(metrics={"loss": float("nan")})

    def test_frozen_partitions_can_be_reassigned_without_changing_unit(self) -> None:
        unit_id = make_unit_id({"query": "q1", "route": "r1"})
        workers = [
            WorkerAssignment("worker-a", tuple(range(0, 128)), 256),
            WorkerAssignment("worker-b", tuple(range(128, 256)), 256),
        ]
        validate_work_plan(workers, require_full=True)
        owners = [worker for worker in workers if worker.owns(unit_id)]
        self.assertEqual(len(owners), 1)
        partition = next(value for value in range(256) if WorkerAssignment("x", (value,), 256).owns(unit_id))
        replacement = WorkerAssignment("replacement", (partition,), 256)
        self.assertTrue(replacement.owns(unit_id))

    def test_duplicate_partition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_work_plan(
                [WorkerAssignment("a", (0, 1), 4), WorkerAssignment("b", (1, 2), 4)]
            )


if __name__ == "__main__":
    unittest.main()
