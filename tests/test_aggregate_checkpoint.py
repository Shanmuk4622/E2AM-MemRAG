from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from e2am_memrag.aggregate import aggregate_rows
from e2am_memrag.checkpoints import CheckpointStore
from e2am_memrag.utils import atomic_write_json, read_json


class AggregateCheckpointTests(unittest.TestCase):
    def test_identical_duplicate_is_counted_and_divergence_blocks(self) -> None:
        row = {
            "unit_id": "u1",
            "spec_hash": "spec-a",
            "status": "SUCCESS",
            "input": {"q": 1},
            "output": {"a": 2},
        }
        duplicate = {**row, "duration_seconds": 99, "attempt": 2}
        result = aggregate_rows([[row], [duplicate]])
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.identical_duplicates, 1)
        divergent = {**row, "output": {"a": 3}}
        with self.assertRaises(RuntimeError):
            aggregate_rows([[row], [divergent]])

    def test_aggregate_rejects_missing_or_mixed_spec_hashes(self) -> None:
        base = {
            "unit_id": "u1",
            "spec_hash": "spec-a",
            "status": "SUCCESS",
            "input": {},
            "output": {},
        }
        with self.assertRaises(ValueError):
            aggregate_rows([[{key: value for key, value in base.items() if key != "spec_hash"}]])
        with self.assertRaises(RuntimeError):
            aggregate_rows([[base, {**base, "unit_id": "u2", "spec_hash": "spec-b"}]])

    def test_only_complete_verified_checkpoint_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary))
            metadata = {
                "model": "router-v0",
                "optimizer": "adamw",
                "scheduler": "linear",
                "rng": {"python": 1, "numpy": 2, "torch": 3},
                "data_cursor": {"epoch": 2, "batch": 7},
            }
            first = store.seal(10, {"state.bin": b"checkpoint-ten"}, metadata)
            second = store.seal(20, {"state.bin": b"checkpoint-twenty"}, metadata)
            self.assertEqual(first.step, 10)
            self.assertEqual(store.latest_valid().path, second.path)
            (Path(temporary) / ".step-000000030-crashed").mkdir()
            self.assertEqual(store.latest_valid().step, 20)

    def test_same_checkpoint_step_with_changed_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary))
            base = {
                "model": "router-v0",
                "optimizer": "adamw",
                "scheduler": "linear",
                "rng": {"torch": 1},
                "data_cursor": {"epoch": 1, "batch": 2},
            }
            store.seal(5, {"state.bin": b"same"}, base)
            with self.assertRaises(RuntimeError):
                store.seal(5, {"state.bin": b"same"}, {**base, "optimizer": "sgd"})

    def test_checkpoint_validation_recomputes_seal_and_complete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary))
            metadata = {
                "model": "router-v0",
                "optimizer": "adamw",
                "scheduler": "linear",
                "rng": {"torch": 1},
                "data_cursor": {"epoch": 1, "batch": 2},
            }
            checkpoint = store.seal(7, {"state.bin": b"state"}, metadata)
            (checkpoint.path / "unsealed.bin").write_bytes(b"not-in-seal")
            with self.assertRaises(RuntimeError):
                store.latest_valid()

        with tempfile.TemporaryDirectory() as temporary:
            store = CheckpointStore(Path(temporary))
            checkpoint = store.seal(8, {"state.bin": b"state"}, metadata)
            seal_path = checkpoint.path / "_COMPLETE.json"
            seal = read_json(seal_path)
            seal["metadata"]["optimizer"] = "tampered"
            atomic_write_json(seal_path, seal)
            with self.assertRaises(RuntimeError):
                store.latest_valid()


if __name__ == "__main__":
    unittest.main()
