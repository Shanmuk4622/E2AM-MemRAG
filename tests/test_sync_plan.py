from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

from e2am_memrag.events import EventLog
from e2am_memrag.gates import evaluate_bootstrap_gate
from e2am_memrag.identity import RunIdentity
from e2am_memrag.manifest import ManifestStore
from e2am_memrag.paths import RunPaths
from e2am_memrag.shards import ShardStore
from e2am_memrag.sync import (
    SyncManager,
    UploadItem,
    _binding_payload,
    _remote_prefix,
    _safe_relative,
    _validate_covered_outbox,
    restore_worker_from_hub,
)
from e2am_memrag.utils import atomic_write_bytes, canonical_json, sha256_bytes, sha256_file


REPO_ID = "Shanmuk4622/E2AM-MemRAG-Traces"
REVISION = "ingest-experiment-account-a-00"


@contextmanager
def _fake_hub(remote_files: dict[str, Path]) -> Iterator[list[str]]:
    downloads: list[str] = []

    class EntryNotFoundError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass

    class RevisionNotFoundError(Exception):
        pass

    class FakeApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def repo_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(sha="f" * 40)

        def list_repo_files(self, **_: object) -> list[str]:
            return sorted(remote_files)

    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        downloads.append(filename)
        local = remote_files.get(filename)
        if local is None:
            raise EntryNotFoundError(filename)
        return str(local)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    hub.hf_hub_download = fake_download
    hub_utils = types.ModuleType("huggingface_hub.utils")
    hub_utils.EntryNotFoundError = EntryNotFoundError
    hub_utils.RepositoryNotFoundError = RepositoryNotFoundError
    hub_utils.RevisionNotFoundError = RevisionNotFoundError
    with patch.dict(
        sys.modules,
        {"huggingface_hub": hub, "huggingface_hub.utils": hub_utils},
    ):
        yield downloads


def _manual_receipt(
    directory: Path,
    *,
    prefix: str,
    binding: dict[str, object],
    manifest: dict[str, object],
    items: list[dict[str, object]],
    name: str,
) -> tuple[str, Path]:
    payload = {
        "schema_version": 2,
        "binding": binding,
        "parent_commit": "e" * 40,
        "attempt_id": name,
        "final_seal": False,
        "manifest": manifest,
        "items": items,
    }
    raw = (canonical_json(payload) + "\n").encode("utf-8")
    digest = sha256_bytes(raw)
    local = directory / f"outbox-{digest}.json"
    atomic_write_bytes(local, raw)
    return f"{prefix}/sync/receipts/{local.name}", local


def _manager(
    root: Path,
    *,
    identity: RunIdentity | None = None,
    revision: str = REVISION,
    attempt_id: str = "attempt-a",
    repo_id: str = REPO_ID,
    token_provider=lambda: None,
    max_files_per_commit: int = 24,
) -> tuple[SyncManager, RunPaths, ManifestStore]:
    identity = identity or RunIdentity(
        "experiment", "abc123def456", "account-a-00", 0, 2
    )
    paths = RunPaths(root, identity).create()
    manifest = ManifestStore(paths.manifest, identity)
    manifest.initialize()
    manager = SyncManager(
        paths,
        identity,
        manifest,
        EventLog(paths.events),
        repo_id=repo_id,
        revision=revision,
        attempt_id=attempt_id,
        token_provider=token_provider,
        max_files_per_commit=max_files_per_commit,
    )
    return manager, paths, manifest


class SyncPlanTests(unittest.TestCase):
    def test_initial_worker_stagger_delays_only_the_first_periodic_push(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "e2am_memrag.sync.time.time", return_value=10_000.0
        ):
            identity = RunIdentity(
                "experiment", "abc123def456", "account-a-00", 0, 2
            )
            paths = RunPaths(Path(temporary), identity).create()
            manifest = ManifestStore(paths.manifest, identity)
            manifest.initialize()
            manager = SyncManager(
                paths,
                identity,
                manifest,
                EventLog(paths.events),
                repo_id=REPO_ID,
                revision=REVISION,
                initial_stagger_seconds=150,
            )
            self.assertFalse(manager.due())
            with patch("e2am_memrag.sync.time.time", return_value=10_150.0):
                self.assertTrue(manager.due())

    def test_remote_paths_are_worker_scoped_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, paths, manifest = _manager(Path(temporary))
            ShardStore(paths, manifest).write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS", "output": {}}]
            )

            plan = manager.plan()
            catalog = plan["catalog"]
            prefix = "experiments/experiment/abc123def456/workers/account-a-00/"
            self.assertTrue(catalog)
            self.assertTrue(all(item.remote_path.startswith(prefix) for item in catalog))
            self.assertTrue(
                any(item.sha256[:12] in item.local_path.name for item in catalog)
            )
            self.assertEqual(plan["seal"].role, "seal")
            self.assertEqual(plan["latest"].role, "pointer")

    def test_attempts_share_stable_lane_but_bindings_are_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _, _ = _manager(root, attempt_id="attempt-a")
            second, _, _ = _manager(root, attempt_id="attempt-b")
            other_revision, _, _ = _manager(root, revision="ingest-other")
            other_repo, _, _ = _manager(root, repo_id="Shanmuk4622/other")

            self.assertEqual(first.binding_hash, second.binding_hash)
            self.assertEqual(first.state_path, second.state_path)
            self.assertEqual(first.remote_prefix, second.remote_prefix)
            self.assertNotEqual(first.state_path, other_revision.state_path)
            self.assertNotEqual(first.state_path, other_repo.state_path)

            first._save_state(first._new_state())
            self.assertEqual(second._state()["binding"], second.binding)
            tampered = second._state()
            tampered["binding"] = {**tampered["binding"], "repo_id": "wrong/repo"}
            second.state_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different repo/revision"):
                second._state()

    def test_no_token_is_truthful_and_never_claims_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls = 0

            def no_token() -> None:
                nonlocal calls
                calls += 1
                return None

            manager, _, _ = _manager(Path(temporary), token_provider=no_token)
            result = manager.sync_once(force=True)

            self.assertEqual(calls, 1)
            self.assertEqual(result["status"], "NO_TOKEN")
            self.assertFalse(result["verified"])
            self.assertFalse(result["complete"])
            self.assertEqual(result["uploaded"], 0)
            self.assertEqual(result["remaining"], 2)  # seal + latest pointer
            self.assertFalse(manager.lock_path.exists())

    def test_batch_selection_never_publishes_pointer_before_all_data_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, _, _ = _manager(Path(temporary), max_files_per_commit=6)
            data = [
                UploadItem(Path(f"data-{index}"), f"remote/data-{index}", f"{index:064x}")
                for index in range(8)
            ]
            seal = UploadItem(Path("seal"), "remote/seal", "a" * 64, role="seal")
            latest = UploadItem(
                Path("latest"), "remote/latest", "b" * 64, role="pointer"
            )

            selected, final = manager._select_batch(data, seal, latest)
            self.assertFalse(final)
            self.assertEqual(len(selected), 5)  # sixth operation is the receipt
            self.assertTrue(all(item.role == "data" for item in selected))

            selected, final = manager._select_batch(data[:3], seal, latest)
            self.assertTrue(final)
            self.assertEqual(selected[-2:], [seal, latest])
            self.assertEqual(len(selected) + 1, manager.max_files_per_commit)

            selected, final = manager._select_batch(
                data[:3], seal, latest, allow_final_seal=False
            )
            self.assertFalse(final)
            self.assertEqual(selected, data[:3])
            self.assertTrue(all(item.role == "data" for item in selected))

    def test_checkpoint_completion_marker_is_ordered_after_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, paths, _ = _manager(Path(temporary))
            checkpoint = paths.checkpoints / "step-00000001-deadbeef"
            checkpoint.mkdir(parents=True)
            atomic_write_bytes(checkpoint / "model.safetensors", b"model")
            atomic_write_bytes(checkpoint / "optimizer.pt", b"optimizer")
            atomic_write_bytes(checkpoint / "_COMPLETE.json", b"{}")

            checkpoint_items = [
                item
                for item in manager.plan()["catalog"]
                if item.restore_relative
                and item.restore_relative.startswith("checkpoints/")
            ]
            self.assertEqual(checkpoint_items[-1].local_path.name, "_COMPLETE.json")

    def test_data_outbox_receipt_still_supports_lost_ack_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, paths, manifest = _manager(Path(temporary))
            record = ShardStore(paths, manifest).write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
            )
            data = next(
                item
                for item in manager.plan()["catalog"]
                if item.restore_relative == record.relative_path
            )
            receipt = manager._write_receipt([data], final_seal=False)
            self.assertTrue(receipt.local_path.name.startswith("outbox-"))
            state = manager._new_state()
            state["last_commit_sha"] = "a" * 40
            state["inflight"] = {
                "parent_commit": "a" * 40,
                "receipt_remote_path": receipt.remote_path,
                "receipt_sha256": receipt.sha256,
                "items": [
                    {
                        "local_relative": data.local_path.relative_to(
                            paths.worker_root
                        ).as_posix(),
                        "remote_path": data.remote_path,
                        "sha256": data.sha256,
                        "role": "data",
                    }
                ],
                "final_seal": False,
                "seal_remote_path": None,
                "seal_sha256": None,
                "latest_sha256": None,
            }
            manager._save_state(state)

            with patch.object(
                manager, "_download_verified", return_value=receipt.local_path
            ):
                adopted = manager._adopt_inflight_or_raise(
                    object(), "fake-token", state, "b" * 40
                )

            relative = data.local_path.relative_to(paths.worker_root).as_posix()
            self.assertEqual(adopted["uploaded"][relative], data.sha256)
            self.assertEqual(adopted["last_commit_sha"], "b" * 40)
            self.assertIsNone(adopted["inflight"])

    def test_cross_platform_unsafe_paths_are_rejected(self) -> None:
        self.assertEqual(
            _safe_relative("sync/revisions/abc/events.jsonl.gz", ("sync",)),
            "sync/revisions/abc/events.jsonl.gz",
        )
        unsafe = (
            "../traces/a.json",
            "traces/../outside.json",
            "/traces/a.json",
            "traces\\..\\outside.json",
            "traces//a.json",
            "traces/a.json/",
            "traces/C:stream",
            "unknown/a.json",
        )
        for value in unsafe:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                _safe_relative(value, ("traces", "sync"))

    def test_covered_outbox_path_and_full_digest_are_bound(self) -> None:
        prefix = "experiments/experiment/spec/workers/worker"
        digest = "a" * 64
        valid = {
            "remote_path": f"{prefix}/sync/receipts/outbox-{digest}.json",
            "sha256": digest,
        }
        self.assertEqual(_validate_covered_outbox(valid, prefix), valid)
        with self.assertRaises(RuntimeError):
            _validate_covered_outbox({**valid, "sha256": "b" * 64}, prefix)
        with self.assertRaises(RuntimeError):
            _validate_covered_outbox(
                {
                    **valid,
                    "remote_path": f"other/sync/receipts/outbox-{digest}.json",
                },
                prefix,
            )

    def test_manifest_cannot_point_catalog_outside_worker_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager, paths, manifest = _manager(Path(temporary))
            ShardStore(paths, manifest).write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
            )
            payload = manifest.read()
            payload["shards"][0]["relative_path"] = "traces/../../outside.jsonl.gz"
            paths.manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Unsafe artifact path"):
                manager.plan()

    def test_restore_rejects_escaped_seal_pointer_before_seal_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = RunIdentity(
                "experiment", "abc123def456", "account-a-00", 0, 2
            )
            paths = RunPaths(Path(temporary), identity)
            binding = _binding_payload(identity, REPO_ID, "dataset", REVISION)
            prefix = _remote_prefix(identity)
            latest_path = Path(temporary) / "latest.json"
            latest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "binding": binding,
                        "seal_remote_path": f"{prefix}/traces/not-a-seal.json",
                        "seal_sha256": "a" * 64,
                        "artifact_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            downloads: list[str] = []

            class FakeApi:
                def __init__(self, token: str) -> None:
                    self.token = token

                def repo_info(self, **_: object) -> SimpleNamespace:
                    return SimpleNamespace(sha="f" * 40)

            def fake_download(**kwargs: object) -> str:
                filename = str(kwargs["filename"])
                downloads.append(filename)
                if filename.endswith("/sync/latest.json"):
                    return str(latest_path)
                raise AssertionError("unsafe seal path must be rejected before download")

            hub = types.ModuleType("huggingface_hub")
            hub.HfApi = FakeApi
            hub.hf_hub_download = fake_download
            hub_utils = types.ModuleType("huggingface_hub.utils")
            for name in (
                "EntryNotFoundError",
                "RepositoryNotFoundError",
                "RevisionNotFoundError",
            ):
                setattr(hub_utils, name, type(name, (Exception,), {}))

            with patch.dict(
                sys.modules,
                {"huggingface_hub": hub, "huggingface_hub.utils": hub_utils},
            ):
                with self.assertRaisesRegex(RuntimeError, "seal directory"):
                    restore_worker_from_hub(
                        paths, identity, REPO_ID, REVISION, token="fake-token"
                    )

            self.assertEqual(downloads, [f"{prefix}/sync/latest.json"])

    def test_restore_adopts_data_receipt_when_no_latest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = RunIdentity(
                "experiment", "abc123def456", "account-a-00", 0, 2
            )
            stage, stage_paths, stage_manifest = _manager(
                root / "stage", identity=identity
            )
            record = ShardStore(stage_paths, stage_manifest).write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
            )
            plan = stage.plan()
            data = next(
                item
                for item in plan["catalog"]
                if item.restore_relative == record.relative_path
            )
            receipt = stage._write_receipt([data], final_seal=False)
            remote_files = {
                receipt.remote_path: receipt.local_path,
                data.remote_path: data.local_path,
            }

            target_paths = RunPaths(root / "target", identity)
            with _fake_hub(remote_files):
                result = restore_worker_from_hub(
                    target_paths, identity, REPO_ID, REVISION, token="fake-token"
                )

            self.assertEqual(result["status"], "RESTORED_WITH_UNSEALED_OUTBOX")
            self.assertFalse(result["closure_verified"])
            self.assertEqual(result["orphan_artifacts"], 1)
            restored_manifest = ManifestStore(target_paths.manifest, identity)
            self.assertEqual(restored_manifest.read()["shards"], [record.__dict__])
            self.assertTrue((target_paths.worker_root / record.relative_path).is_file())

            restored = SyncManager(
                target_paths,
                identity,
                restored_manifest,
                EventLog(target_paths.events),
                repo_id=REPO_ID,
                revision=REVISION,
                token_provider=lambda: None,
            )
            restored_plan = restored.plan()
            self.assertEqual(restored_plan["pending_data"], [])
            self.assertFalse(restored_plan["current_verified"])
            self.assertEqual(restored._state()["last_commit_sha"], "f" * 40)

            # An authenticated outbox is recoverable, but it is deliberately not
            # equivalent to a fresh, closed, round-trip restore.
            gate = evaluate_bootstrap_gate(
                phase="REMOTE_RESTORE",
                preflight={
                    "disk_ok": True,
                    "visible_gpu_count": 1,
                    "gpu_name": "Tesla T4",
                },
                completed_units=1,
                expected_units=1,
                shard_validation={"rows": 1},
                energy={
                    "available": True,
                    "energy_joules": 1.0,
                    "duration_seconds": 2.0,
                    "samples": 10,
                    "gpu_uuid": "GPU-test",
                },
                checkpoint_valid=True,
                source_verified=True,
                environment_verified=True,
                sync_results=[{"complete": True, "verified": True}],
                restore_result=result,
                fresh_restore_root=True,
            )
            self.assertFalse(gate["hard_pass"])

    def test_restore_merges_newer_receipt_after_old_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = RunIdentity(
                "experiment", "abc123def456", "account-a-00", 0, 2
            )
            stage, stage_paths, stage_manifest = _manager(
                root / "stage", identity=identity
            )
            store = ShardStore(stage_paths, stage_manifest)
            first = store.write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
            )
            base_plan = stage.plan()
            base_latest = root / "base-latest.json"
            atomic_write_bytes(base_latest, base_plan["latest"].local_path.read_bytes())

            second = store.write_rows(
                "traces", [{"unit_id": "unit-b", "status": "SUCCESS"}]
            )
            current_plan = stage.plan()
            orphan = next(
                item
                for item in current_plan["catalog"]
                if item.restore_relative == second.relative_path
            )
            receipt = stage._write_receipt([orphan], final_seal=False)
            base_data = next(
                item
                for item in base_plan["catalog"]
                if item.restore_relative == first.relative_path
            )
            remote_files = {
                base_plan["latest"].remote_path: base_latest,
                base_plan["seal"].remote_path: base_plan["seal"].local_path,
                base_data.remote_path: base_data.local_path,
                receipt.remote_path: receipt.local_path,
                orphan.remote_path: orphan.local_path,
            }

            target_paths = RunPaths(root / "target", identity)
            with _fake_hub(remote_files):
                result = restore_worker_from_hub(
                    target_paths, identity, REPO_ID, REVISION, token="fake-token"
                )

            self.assertEqual(result["status"], "RESTORED_WITH_UNSEALED_OUTBOX")
            self.assertTrue(result["closure_verified"])
            self.assertEqual(result["orphan_artifacts"], 1)
            restored_manifest = ManifestStore(target_paths.manifest, identity)
            manifest = restored_manifest.read()
            self.assertEqual(
                [record["relative_path"] for record in manifest["shards"]],
                [first.relative_path, second.relative_path],
            )
            self.assertEqual(manifest["counters"]["trace_rows"], 2)
            restored = SyncManager(
                target_paths,
                identity,
                restored_manifest,
                EventLog(target_paths.events),
                repo_id=REPO_ID,
                revision=REVISION,
                token_provider=lambda: None,
            )
            self.assertEqual(restored.plan()["pending_data"], [])
            self.assertIsNotNone(restored._state()["verified_seal_sha256"])

    def test_restore_skips_final_and_seal_covered_receipt_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = RunIdentity(
                "experiment", "abc123def456", "account-a-00", 0, 2
            )
            stage, stage_paths, stage_manifest = _manager(
                root / "stage", identity=identity
            )
            record = ShardStore(stage_paths, stage_manifest).write_rows(
                "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
            )
            initial_plan = stage.plan()
            data = next(
                item
                for item in initial_plan["catalog"]
                if item.restore_relative == record.relative_path
            )
            covered_outbox = stage._write_receipt([data], final_seal=False)
            sealed_plan = stage.plan()  # now authenticates the prior outbox as covered
            remote_files = {
                sealed_plan["latest"].remote_path: sealed_plan["latest"].local_path,
                sealed_plan["seal"].remote_path: sealed_plan["seal"].local_path,
                data.remote_path: data.local_path,
                covered_outbox.remote_path: covered_outbox.local_path,
            }
            for index in range(25):
                raw = f"historical-final-{index}".encode("utf-8")
                digest = sha256_bytes(raw)
                local = root / f"receipt-{digest}.json"
                atomic_write_bytes(local, raw)
                remote_files[
                    f"{stage.remote_prefix}/sync/receipts/{local.name}"
                ] = local

            target_paths = RunPaths(root / "target", identity)
            with _fake_hub(remote_files) as downloads:
                result = restore_worker_from_hub(
                    target_paths,
                    identity,
                    REPO_ID,
                    REVISION,
                    token="fake-token",
                )

            self.assertEqual(result["status"], "RESTORED")
            self.assertFalse(
                any("/sync/receipts/" in remote_path for remote_path in downloads)
            )
            restored_manifest = ManifestStore(target_paths.manifest, identity)
            restored = SyncManager(
                target_paths,
                identity,
                restored_manifest,
                EventLog(target_paths.events),
                repo_id=REPO_ID,
                revision=REVISION,
                token_provider=lambda: None,
            )
            resealed = json.loads(restored.plan()["seal"].local_path.read_text("utf-8"))
            self.assertEqual(
                [item["remote_path"] for item in resealed["covered_outboxes"]],
                [covered_outbox.remote_path],
            )

    def test_tampered_and_conflicting_receipts_are_rejected(self) -> None:
        with self.subTest(case="tampered"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                identity = RunIdentity(
                    "experiment", "abc123def456", "account-a-00", 0, 2
                )
                stage, stage_paths, stage_manifest = _manager(
                    root / "stage", identity=identity
                )
                record = ShardStore(stage_paths, stage_manifest).write_rows(
                    "traces", [{"unit_id": "unit-a", "status": "SUCCESS"}]
                )
                plan = stage.plan()
                data = next(
                    item
                    for item in plan["catalog"]
                    if item.restore_relative == record.relative_path
                )
                receipt = stage._write_receipt([data], final_seal=False)
                tampered = root / receipt.local_path.name
                atomic_write_bytes(tampered, receipt.local_path.read_bytes() + b" ")
                remote_files = {
                    receipt.remote_path: tampered,
                    data.remote_path: data.local_path,
                }
                with _fake_hub(remote_files):
                    with self.assertRaisesRegex(RuntimeError, "receipt checksum"):
                        restore_worker_from_hub(
                            RunPaths(root / "target", identity),
                            identity,
                            REPO_ID,
                            REVISION,
                            token="fake-token",
                        )

        with self.subTest(case="conflicting"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                identity = RunIdentity(
                    "experiment", "abc123def456", "account-a-00", 0, 2
                )
                stage, _, manifest_store = _manager(root / "stage", identity=identity)
                manifest = manifest_store.read()
                binding = stage.binding
                prefix = stage.remote_prefix
                first_data = root / "first.bin"
                second_data = root / "second.bin"
                atomic_write_bytes(first_data, b"first")
                atomic_write_bytes(second_data, b"second")
                remote = f"{prefix}/meta/collision.bin"
                restore = (
                    f"sync/revisions/{stage.binding_hash}/"
                    "meta_snapshots/collision.bin"
                )
                first_item = {
                    "remote_path": remote,
                    "sha256": sha256_file(first_data),
                    "bytes": first_data.stat().st_size,
                    "role": "data",
                    "restore_relative": restore,
                }
                second_item = {
                    **first_item,
                    "sha256": sha256_file(second_data),
                    "bytes": second_data.stat().st_size,
                }
                receipt_one, path_one = _manual_receipt(
                    root,
                    prefix=prefix,
                    binding=binding,
                    manifest=manifest,
                    items=[first_item],
                    name="one",
                )
                receipt_two, path_two = _manual_receipt(
                    root,
                    prefix=prefix,
                    binding=binding,
                    manifest=manifest,
                    items=[second_item],
                    name="two",
                )
                remote_files = {
                    receipt_one: path_one,
                    receipt_two: path_two,
                    remote: first_data,
                }
                with _fake_hub(remote_files):
                    with self.assertRaisesRegex(RuntimeError, "Conflicting receipt"):
                        restore_worker_from_hub(
                            RunPaths(root / "target", identity),
                            identity,
                            REPO_ID,
                            REVISION,
                            token="fake-token",
                        )


if __name__ == "__main__":
    unittest.main()
