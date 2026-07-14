from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from e2am_memrag.events import EventLog
from e2am_memrag.identity import RunIdentity, make_unit_id
from e2am_memrag.manifest import ManifestStore
from e2am_memrag.paths import RunPaths
from e2am_memrag.runner import ResumableRunner
from e2am_memrag.shards import ShardStore
from e2am_memrag.signals import StopController


class ResumeTests(unittest.TestCase):
    def test_shards_reject_secrets_and_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            with self.assertRaises(ValueError):
                shards.write_rows(
                    "traces",
                    [{"unit_id": "u-secret", "output": "hf_abcdefghijk12345"}],
                )
            with self.assertRaises(ValueError):
                shards.write_rows(
                    "traces",
                    [{"unit_id": "u-nan", "output": {"score": float("nan")}}],
                )
            self.assertEqual(manifest.read()["shards"], [])

    def test_interruption_then_resume_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            events = EventLog(paths.events)
            units = [{"value": value} for value in range(17)]

            first = ResumableRunner(
                identity, manifest, shards, events, StopController(), shard_rows=3
            ).run(units, lambda item: {"result": item["value"] + 1}, max_new_units=5)
            self.assertEqual(first.processed, 5)
            self.assertTrue(first.stopped_early)

            second = ResumableRunner(
                identity, manifest, shards, events, StopController(), shard_rows=3
            ).run(units, lambda item: {"result": item["value"] + 1})
            self.assertEqual(second.processed, 12)
            self.assertEqual(second.already_completed, 5)
            self.assertEqual(len(shards.completed_unit_ids()), 17)
            self.assertEqual(shards.validate()["rows"], 17)

    def test_orphan_shard_is_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            record = shards.write_rows(
                "traces", [{"unit_id": "u-1", "status": "SUCCESS", "output": {}}]
            )
            saved = manifest.read()
            saved["shards"] = []
            saved["counters"] = {"trace_rows": 0, "failure_rows": 0}
            from e2am_memrag.utils import atomic_write_json

            atomic_write_json(paths.manifest, saved)
            recovered = shards.reconcile("traces")
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0].relative_path, record.relative_path)
            self.assertTrue(recovered[0].recovered)

    def test_keyboard_interrupt_leaves_unit_replayable_and_flushes_prior_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            events = EventLog(paths.events)

            def processor(item):
                if item["value"] == 3:
                    raise KeyboardInterrupt()
                return {"result": item["value"]}

            first = ResumableRunner(
                identity, manifest, shards, events, StopController(), shard_rows=10
            ).run([{"value": value} for value in range(6)], processor)
            self.assertTrue(first.stopped_early)
            self.assertEqual(len(shards.completed_unit_ids()), 3)
            self.assertEqual(manifest.read()["status"], "INTERRUPTED")

            second = ResumableRunner(
                identity, manifest, shards, events, StopController(), shard_rows=10
            ).run(
                [{"value": value} for value in range(6)],
                lambda item: {"result": item["value"]},
            )
            self.assertEqual(second.processed, 3)
            self.assertEqual(len(shards.completed_unit_ids()), 6)

    def test_non_fail_fast_run_does_not_end_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            stats = ResumableRunner(
                identity,
                manifest,
                shards,
                EventLog(paths.events),
                StopController(),
                max_retries_per_unit=1,
                fail_fast=False,
            ).run(
                [{"value": 1}, {"value": 2}],
                lambda item: (_ for _ in ()).throw(ValueError("hf_abcdefghijk12345"))
                if item["value"] == 1
                else {"result": 2},
            )
            self.assertEqual(stats.terminal_failures, 1)
            self.assertEqual(manifest.read()["status"], "FAILED")
            failure_text = "".join(
                str(row)
                for item in manifest.read()["shards"]
                if item["kind"] == "failures"
                for row in ShardStore.read_rows(paths.worker_root / item["relative_path"])
            )
            self.assertNotIn("hf_abcdefghijk12345", failure_text)

    def test_runner_validates_precomputed_unit_id_and_records_spec_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)
            unit = {"value": 9}
            canonical = make_unit_id(
                {"spec_hash": identity.config_hash, "unit": unit}
            )
            runner = ResumableRunner(
                identity,
                manifest,
                shards,
                EventLog(paths.events),
                StopController(),
            )
            runner.run([{**unit, "_unit_id": canonical}], lambda item: {"result": 10})
            trace_record = next(
                item for item in manifest.read()["shards"] if item["kind"] == "traces"
            )
            trace = ShardStore.read_rows(paths.worker_root / trace_record["relative_path"])[0]
            self.assertEqual(trace["unit_id"], canonical)
            self.assertEqual(trace["spec_hash"], identity.config_hash)

            with self.assertRaises(ValueError):
                runner.run(
                    [{**unit, "value": 10, "_unit_id": canonical}],
                    lambda item: {"result": 11},
                )

    def test_sync_callback_failure_does_not_erase_local_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity("test-exp", "abc123def456", "worker-0", 0, 1)
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            shards = ShardStore(paths, manifest)

            def unreliable_sync(force: bool):
                if not force:
                    raise RuntimeError("Hub unavailable for hf_abcdefghijk12345")
                return {
                    "status": "NO_TOKEN",
                    "complete": False,
                    "verified": False,
                }

            stats = ResumableRunner(
                identity,
                manifest,
                shards,
                EventLog(paths.events),
                StopController(),
                sync_callback=unreliable_sync,
                shard_rows=10,
            ).run(
                [{"value": 1}, {"value": 2}],
                lambda item: {"result": item["value"] + 1},
            )
            self.assertEqual(stats.processed, 2)
            self.assertEqual(stats.sync_callbacks, 3)
            self.assertEqual(stats.sync_failures, 2)
            self.assertEqual(stats.sync_incomplete, 1)
            self.assertEqual(stats.last_sync_status, "NO_TOKEN")
            self.assertFalse(stats.remote_sync_complete)
            self.assertFalse(stats.remote_sync_verified)
            self.assertEqual(manifest.read()["status"], "COMPLETED")
            self.assertEqual(len(shards.completed_unit_ids()), 2)
            event_text = paths.events.read_text(encoding="utf-8")
            self.assertIn("REMOTE_SYNC_FAILED", event_text)
            self.assertIn("REMOTE_SYNC_INCOMPLETE", event_text)
            self.assertNotIn("hf_abcdefghijk12345", event_text)


if __name__ == "__main__":
    unittest.main()
