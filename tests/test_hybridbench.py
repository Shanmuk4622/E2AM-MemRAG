from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from e2am_memrag.hybridbench import (
    BM25Index,
    TASK_TYPES,
    freeze_dataset,
    generate_hybridbench,
    leakage_audit,
    read_jsonl,
)


class HybridBenchTests(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_leak_free(self) -> None:
        left = generate_hybridbench(80, seed=4622)
        right = generate_hybridbench(80, seed=4622)
        self.assertEqual(left, right)
        self.assertEqual(len(left["queries"]), 80)
        self.assertEqual({row["task_type"] for row in left["queries"]}, set(TASK_TYPES))
        self.assertTrue(leakage_audit(left["queries"])["hard_pass"])
        label_ids = {row["query_id"] for row in left["labels"]}
        self.assertEqual(label_ids, {row["query_id"] for row in left["queries"]})
        self.assertFalse(any("answer" in row for row in left["queries"]))

    def test_tombstone_targets_an_existing_memory_event(self) -> None:
        data = generate_hybridbench(40)
        ids = {row["event_id"] for row in data["memory_events"]}
        tombstones = [row for row in data["memory_events"] if row["event_type"] == "tombstone"]
        self.assertTrue(tombstones)
        self.assertTrue(all(row["tombstone_target"] in ids for row in tombstones))

    def test_retrieval_targets_are_not_constant_by_task(self) -> None:
        data = generate_hybridbench(160)
        labels = {row["query_id"]: row for row in data["labels"]}
        for task in (
            "knowledge_only",
            "memory_only",
            "temporal_update",
            "authority_conflict",
            "multi_hop",
        ):
            answers = {
                labels[row["query_id"]]["answer"]
                for row in data["queries"]
                if row["task_type"] == task
            }
            self.assertGreater(len(answers), 1, task)

    def test_bm25_round_trip_retrieves_opaque_project_record(self) -> None:
        records = [
            {"doc_id": "a", "text": "project PRJ-AAAA uses AdamW"},
            {"doc_id": "b", "text": "project PRJ-BBBB uses Lion"},
        ]
        index = BM25Index.build(records, id_field="doc_id")
        restored = BM25Index.from_dict(index.as_dict())
        self.assertEqual(restored.search("PRJ-BBBB optimizer", top_k=1)[0][0], "b")

    def test_freeze_writes_separate_labels_and_verified_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = generate_hybridbench(40)
            freeze = freeze_dataset(temporary, data)
            self.assertEqual(freeze["benchmark"], "E2AM-HybridBench")
            self.assertEqual(len(read_jsonl(Path(temporary) / "labels.jsonl")), 40)
            self.assertTrue((Path(temporary) / "BENCHMARK_FREEZE.json").is_file())


if __name__ == "__main__":
    unittest.main()
