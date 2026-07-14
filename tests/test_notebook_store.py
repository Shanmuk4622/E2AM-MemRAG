from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from e2am_memrag.notebook_store import (
    HubAuthenticationDisabled,
    NotebookArtifactStore,
    RestoreRequired,
)


class _HttpError(RuntimeError):
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.response = SimpleNamespace(status_code=status, headers=headers)
        super().__init__(f"HTTP {status}")


class _Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _FakeHub:
    """Commit-addressed in-memory Hub used to assert revision pinning."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.heads = {"main": "0" * 40}
        self.snapshots: dict[str, dict[str, bytes]] = {"0" * 40: {}}
        self.commits: list[dict[str, object]] = []
        self.download_revisions: list[str] = []
        self.download_filenames: list[str] = []
        self.download_counter = 0
        self.repo_info_calls = 0

    def api_factory(self, token: str) -> "_FakeHub":
        self.last_token = token
        return self

    @staticmethod
    def operation_factory(remote_path: str, local_path: Path) -> dict[str, object]:
        return {"path_in_repo": remote_path, "local_path": Path(local_path)}

    def create_repo(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(repo_id="Shanmuk4622/test")

    def create_branch(
        self, *, branch: str, revision: str, **_: object
    ) -> SimpleNamespace:
        if branch not in self.heads:
            resolved = self._resolve(revision)
            if resolved not in self.snapshots:
                raise _HttpError(404)
            self.heads[branch] = resolved
        return SimpleNamespace(branch=branch)

    def _resolve(self, revision: str) -> str:
        return self.heads.get(revision, revision)

    def repo_info(self, *, revision: str, **_: object) -> SimpleNamespace:
        self.repo_info_calls += 1
        resolved = self._resolve(revision)
        if resolved not in self.snapshots:
            raise _HttpError(404)
        return SimpleNamespace(sha=resolved)

    def list_repo_files(self, *, revision: str, **_: object) -> list[str]:
        return sorted(self.snapshots[self._resolve(revision)])

    def create_commit(
        self,
        *,
        revision: str,
        parent_commit: str | None = None,
        operations: list[dict[str, object]],
        **kwargs: object,
    ) -> SimpleNamespace:
        if parent_commit is None:
            if revision in self.heads:
                raise _HttpError(409)
            snapshot = {}
        else:
            if self._resolve(revision) != parent_commit:
                raise _HttpError(409)
            snapshot = dict(self.snapshots[parent_commit])
        for operation in operations:
            snapshot[str(operation["path_in_repo"])] = Path(
                operation["local_path"]
            ).read_bytes()
        commit_sha = f"{len(self.commits) + 1:040x}"
        self.snapshots[commit_sha] = snapshot
        self.heads[revision] = commit_sha
        self.commits.append(
            {
                "revision": revision,
                "parent_commit": parent_commit,
                "operations": operations,
                **kwargs,
            }
        )
        return SimpleNamespace(oid=commit_sha)

    def download_file(
        self, *, revision: str, filename: str, cache_dir: str | Path, **_: object
    ) -> str:
        self.download_revisions.append(revision)
        self.download_filenames.append(filename)
        resolved = self._resolve(revision)
        try:
            payload = self.snapshots[resolved][filename]
        except KeyError as error:
            raise _HttpError(404) from error
        self.download_counter += 1
        destination = (
            Path(cache_dir)
            / "fake"
            / f"{self.download_counter:04d}-{Path(filename).name}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return str(destination)


def _store(
    root: Path,
    hub: _FakeHub,
    *,
    clock: _Clock | None = None,
    token: str = "test-token-not-an-hf-secret",
) -> NotebookArtifactStore:
    return NotebookArtifactStore(
        root,
        repo_id="Shanmuk4622/test",
        experiment_id="experiment-v1",
        worker_id="friend-01",
        token_provider=lambda: token,
        api_factory=hub.api_factory,
        download_file=hub.download_file,
        operation_factory=hub.operation_factory,
        clock=clock or _Clock(),
        sleeper=lambda _: None,
    )


class NotebookArtifactStoreTests(unittest.TestCase):
    def test_empty_repository_gets_one_deterministic_main_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            hub.heads.clear()
            hub.snapshots.clear()
            store = _store(root / "local", hub)
            store.put_bytes("results/a.txt", b"a")
            result = store.flush(force=True, reason="first")
            self.assertTrue(result["verified"])
            self.assertIn("main", hub.heads)
            self.assertIn(store.branch, hub.heads)
            self.assertEqual(len(hub.commits), 2)

    def test_atomic_content_addressed_flush_uses_optimistic_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            secret = "hf_" + "ThisMustNeverAppearInAFile123"
            store = _store(root / "local", hub, token=secret)
            self.assertEqual(store.sync_interval_seconds, 1200)
            self.assertEqual(store.branch, "worker-experiment-v1-friend-01")

            payload = b"sealed-checkpoint-bytes"
            digest = hashlib.sha256(payload).hexdigest()
            record = store.put_bytes("checkpoints/step-10/model.bin", payload)
            self.assertEqual(record["sha256"], digest)
            self.assertTrue((store.objects / digest).is_file())

            result = store.flush(force=True, reason="major-training")
            self.assertEqual(result["status"], "SYNCED")
            self.assertTrue(result["verified"])
            self.assertEqual(len(hub.commits), 1)
            commit = hub.commits[0]
            self.assertEqual(commit["parent_commit"], "0" * 40)
            paths = {
                str(operation["path_in_repo"])
                for operation in commit["operations"]
            }
            self.assertIn(record["remote_path"], paths)
            self.assertIn(store.pointer_path, paths)
            self.assertTrue(
                any(path.endswith(f"/manifests/{result['manifest_sha256']}.json") for path in paths)
            )

            # State/manifest/pointer files never persist the credential.
            for path in (root / "local").rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode("utf-8"), path.read_bytes())
            self.assertFalse(list((root / "local").glob(".state.json.*.tmp")))

    def test_restore_resolves_once_then_downloads_only_from_pinned_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            writer = _store(root / "writer", hub)
            writer.put_bytes("results/metrics.json", b'{"accuracy":0.75}\n')
            committed = writer.flush(force=True, reason="end")["commit_sha"]

            hub.download_revisions.clear()
            reader = _store(root / "reader", hub)
            restored = reader.restore_latest(root / "restored")
            self.assertEqual(restored["commit_sha"], committed)
            self.assertEqual(restored["restored_artifacts"], 1)
            self.assertEqual(
                (root / "restored/results/metrics.json").read_bytes(),
                b'{"accuracy":0.75}\n',
            )
            self.assertTrue(hub.download_revisions)
            self.assertEqual(set(hub.download_revisions), {committed})
            state = json.loads(reader.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["last_commit_sha"], committed)
            self.assertFalse(state["dirty"])

    def test_selected_restore_validates_closure_and_downloads_only_requested_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            writer = _store(root / "writer", hub)
            gate = writer.put_bytes("gates/PASS.json", b'{"status":"PASS"}\n')
            bulky = writer.put_bytes("results/bulky.bin", b"large-result-payload")
            committed = writer.flush(force=True, reason="end")["commit_sha"]

            reader = _store(root / "reader", hub)
            state_before = reader.state_path.read_bytes()
            hub.download_revisions.clear()
            hub.download_filenames.clear()
            destination = root / "selected"
            restored = reader.restore_selected(["gates/PASS.json"], destination)

            self.assertEqual(restored["status"], "SELECTED_RESTORE")
            self.assertTrue(restored["verified"])
            self.assertFalse(restored["full_restore"])
            self.assertEqual(restored["commit_sha"], committed)
            self.assertEqual(restored["manifest_artifacts"], 2)
            self.assertEqual(restored["restored_artifacts"], 1)
            self.assertEqual(
                restored["selected_artifacts"],
                [
                    {
                        "logical_path": "gates/PASS.json",
                        "sha256": gate["sha256"],
                        "bytes": gate["bytes"],
                    }
                ],
            )
            self.assertEqual(
                (destination / "gates" / "PASS.json").read_bytes(),
                b'{"status":"PASS"}\n',
            )
            self.assertFalse((destination / "results" / "bulky.bin").exists())
            self.assertEqual(
                hub.download_filenames,
                [reader.pointer_path, restored["manifest_path"], gate["remote_path"]],
            )
            self.assertNotIn(bulky["remote_path"], hub.download_filenames)
            self.assertEqual(set(hub.download_revisions), {committed})
            self.assertEqual(reader.state_path.read_bytes(), state_before)
            self.assertFalse(any(reader.objects.iterdir()))

    def test_selected_restore_rejects_invalid_unselected_manifest_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            writer = _store(root / "writer", hub)
            writer.put_bytes("gates/PASS.json", b'{"status":"PASS"}\n')
            writer.put_bytes("results/unselected.bin", b"unselected")
            committed = writer.flush(force=True, reason="end")["commit_sha"]

            snapshot = hub.snapshots[committed]
            pointer = json.loads(snapshot[writer.pointer_path])
            manifest = json.loads(snapshot[pointer["manifest_path"]])
            unselected = next(
                record
                for record in manifest["artifacts"]
                if record["logical_path"] == "results/unselected.bin"
            )
            unselected["remote_path"] = "not/content-addressed"
            malformed_manifest = (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            malformed_sha = hashlib.sha256(malformed_manifest).hexdigest()
            malformed_path = (
                f"{writer.remote_prefix}/manifests/{malformed_sha}.json"
            )
            snapshot[malformed_path] = malformed_manifest
            pointer["manifest_path"] = malformed_path
            pointer["manifest_sha256"] = malformed_sha
            snapshot[writer.pointer_path] = (
                json.dumps(
                    pointer,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")

            reader = _store(root / "reader", hub)
            hub.download_filenames.clear()
            with self.assertRaisesRegex(RuntimeError, "not content-addressed"):
                reader.restore_selected(
                    ["gates/PASS.json"], root / "must-not-materialize"
                )
            self.assertEqual(
                hub.download_filenames,
                [reader.pointer_path, malformed_path],
            )
            self.assertFalse((root / "must-not-materialize").exists())

    def test_selected_restore_accepts_explicit_pinned_commit_without_head_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            writer = _store(root / "writer", hub)
            writer.put_bytes("gate.json", b'{"status":"PASS"}\n')
            committed = writer.flush(force=True, reason="end")["commit_sha"]

            reader = _store(root / "reader", hub)
            calls_before = hub.repo_info_calls
            outcome = reader.restore_selected(
                ["gate.json"], root / "selected", revision=committed
            )
            self.assertEqual(outcome["commit_sha"], committed)
            self.assertEqual(hub.repo_info_calls, calls_before)
            self.assertEqual(
                (root / "selected" / "gate.json").read_bytes(),
                b'{"status":"PASS"}\n',
            )

    def test_selected_and_full_restore_reject_mutable_revision_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            reader = _store(root / "reader", hub)
            with self.assertRaisesRegex(ValueError, "40-hex"):
                reader.restore_selected(
                    ["gate.json"], root / "selected", revision="main"
                )
            with self.assertRaisesRegex(ValueError, "40-hex"):
                reader.restore_latest(root / "full", revision="worker-branch")

    def test_dirty_interval_and_interruption_force_flush(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            clock = _Clock(100.0)
            store = _store(root / "local", hub, clock=clock)
            store.put_bytes("traces/part-000.bin", b"first")
            self.assertEqual(store.maybe_flush()["status"], "NOT_DUE")
            self.assertEqual(len(hub.commits), 0)

            with self.assertRaises(KeyboardInterrupt):
                with store.interruption_guard():
                    raise KeyboardInterrupt
            self.assertEqual(len(hub.commits), 1)
            self.assertEqual(store.last_flush["reason"], "interruption")

            store.put_bytes("traces/part-001.bin", b"second")
            self.assertEqual(store.maybe_flush()["status"], "NOT_DUE")
            clock.value += 1_200.0
            self.assertEqual(store.maybe_flush()["status"], "SYNCED")
            self.assertEqual(len(hub.commits), 2)

    def test_transient_retry_and_authentication_latch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            sleeps: list[float] = []
            store = NotebookArtifactStore(
                root / "local",
                repo_id="Shanmuk4622/test",
                experiment_id="experiment-v1",
                worker_id="friend-01",
                token_provider=lambda: "token",
                api_factory=hub.api_factory,
                download_file=hub.download_file,
                operation_factory=hub.operation_factory,
                sleeper=sleeps.append,
            )
            calls = 0

            def transient() -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise _HttpError(429, "7")
                if calls == 2:
                    raise _HttpError(503)
                return "ok"

            self.assertEqual(store._call(transient), "ok")
            self.assertEqual(calls, 3)
            self.assertEqual(sleeps, [7.0, 2.0])

            auth_calls = 0

            def denied() -> None:
                nonlocal auth_calls
                auth_calls += 1
                raise _HttpError(403)

            with self.assertRaises(_HttpError):
                store._call(denied)
            with self.assertRaises(HubAuthenticationDisabled):
                store._call(denied)
            self.assertEqual(auth_calls, 1)
            store.reset_authentication()
            with self.assertRaises(_HttpError):
                store._call(denied)
            self.assertEqual(auth_calls, 2)

    def test_existing_lane_requires_restore_before_fresh_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            first = _store(root / "first", hub)
            first.put_bytes("results/a.txt", b"a")
            first.flush(force=True, reason="end")

            fresh = _store(root / "fresh", hub)
            fresh.put_bytes("results/b.txt", b"b")
            with self.assertRaises(RestoreRequired):
                fresh.flush(force=True, reason="major")
            self.assertTrue(fresh.dirty)

    def test_checkpoint_must_be_sealed_and_json_rejects_hf_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hub = _FakeHub(root / "hub")
            store = _store(root / "local", hub)
            checkpoint = root / "checkpoint-10"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"model")
            with self.assertRaisesRegex(RuntimeError, "not sealed"):
                store.put_checkpoint(checkpoint)
            (checkpoint / "_COMPLETE.json").write_text("{}\n", encoding="utf-8")
            records = store.put_checkpoint(checkpoint)
            self.assertEqual(len(records), 2)
            self.assertTrue(records[-1]["logical_path"].endswith("/_COMPLETE.json"))
            with self.assertRaisesRegex(ValueError, "token"):
                store.put_json("meta/bad.json", {"credential": "hf_SecretValue12345"})


if __name__ == "__main__":
    unittest.main()
