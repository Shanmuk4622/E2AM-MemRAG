from __future__ import annotations

import copy
import gzip
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from .events import EventLog
from .identity import RunIdentity
from .manifest import ManifestStore
from .paths import RunPaths
from .utils import (
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    remote_path: str
    sha256: str
    role: str = "data"
    restore_relative: str | None = None


@dataclass(frozen=True)
class _RemoteArtifact:
    remote_path: str
    sha256: str
    bytes: int
    restore_relative: str


def _binding_payload(
    identity: RunIdentity, repo_id: str, repo_type: str, revision: str
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "identity": identity.as_dict(),
    }


def _binding_hash(binding: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(binding).encode("utf-8")).hexdigest()[:16]


def _remote_prefix(identity: RunIdentity) -> str:
    return (
        f"experiments/{identity.experiment_id}/{identity.config_hash}/workers/"
        f"{identity.worker_id}"
    )


def _catalog_sort_key(remote_path: str) -> str:
    # A checkpoint's completion marker is a commit record. Keep it after every
    # payload file in that checkpoint so a data-only crash cannot make a partial
    # checkpoint appear sealed during receipt recovery.
    if remote_path.endswith("/checkpoints/_COMPLETE.json"):
        return remote_path[: -len("_COMPLETE.json")] + "~~~~_COMPLETE.json"
    if "/checkpoints/" in remote_path and remote_path.endswith("/_COMPLETE.json"):
        return remote_path[: -len("_COMPLETE.json")] + "~~~~_COMPLETE.json"
    return remote_path


def _safe_relative(value: str, allowed_roots: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Artifact path must be a non-empty string: {value!r}")
    # Backslashes are ordinary characters to PurePosixPath but become separators
    # when joined to a WindowsPath. Colons can become drive or ADS syntax. Reject
    # both before doing POSIX validation so a seal is safe on every restore host.
    if "\\" in value or "\x00" in value or ":" in value:
        raise RuntimeError(f"Unsafe platform-specific artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"Unsafe artifact path in remote seal: {value!r}")
    if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Artifact path is not canonical POSIX form: {value!r}")
    if path.parts[0] not in allowed_roots:
        raise RuntimeError(f"Artifact path is outside allowed roots: {value!r}")
    return path.as_posix()


def _safe_destination(root: Path, relative: str) -> Path:
    """Map a validated POSIX path below root, including symlink containment."""
    parts = PurePosixPath(relative).parts
    destination = root.joinpath(*parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if not resolved_destination.is_relative_to(resolved_root):
        raise RuntimeError(f"Artifact destination escapes worker root: {relative!r}")
    return destination


def _validate_covered_outbox(value: Any, prefix: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError("Remote closure has a malformed covered outbox")
    remote_path = value.get("remote_path")
    digest = value.get("sha256")
    if not isinstance(remote_path, str) or not remote_path.startswith(f"{prefix}/"):
        raise RuntimeError("Covered outbox escapes the worker prefix")
    relative = _safe_relative(remote_path[len(f"{prefix}/") :], ("sync",))
    parts = PurePosixPath(relative).parts
    if len(parts) != 3 or parts[:2] != ("sync", "receipts"):
        raise RuntimeError("Covered outbox is outside the receipt directory")
    match = re.fullmatch(r"outbox-([0-9a-f]{64})\.json", parts[-1])
    if match is None or not isinstance(digest, str) or digest != match.group(1):
        raise RuntimeError("Covered outbox path and checksum are inconsistent")
    return {"remote_path": remote_path, "sha256": digest}


def _result(
    status: str,
    *,
    uploaded: int = 0,
    remaining: int = 0,
    verified: bool = False,
    complete: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "uploaded": uploaded,
        "remaining": remaining,
        "verified": verified,
        "complete": complete,
        **extra,
    }


class SyncManager:
    """Branch-bound, closure-sealed Hub synchronization for one worker lane."""

    STATE_SCHEMA = 2

    def __init__(
        self,
        paths: RunPaths,
        identity: RunIdentity,
        manifest: ManifestStore,
        events: EventLog,
        repo_id: str,
        revision: str = "ingest-local",
        repo_type: str = "dataset",
        base_revision: str = "main",
        attempt_id: str = "attempt-local",
        interval_seconds: int = 1200,
        max_commit_attempts_per_hour: int = 8,
        max_files_per_commit: int = 24,
        active_worker_count: int = 1,
        estimated_hub_calls_per_flush: int = 3,
        initial_stagger_seconds: int = 0,
        token_provider: Callable[[], str | None] | None = None,
    ) -> None:
        if interval_seconds < 300:
            raise ValueError("Sync intervals below five minutes are not supported")
        if max_files_per_commit < 4:
            raise ValueError("max_files_per_commit must be at least 4")
        if active_worker_count < 1 or estimated_hub_calls_per_flush < 1:
            raise ValueError("Worker count and estimated call cost must be positive")
        self.paths = paths.create()
        self.identity = identity
        self.manifest = manifest
        self.events = events
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.base_revision = base_revision
        self.attempt_id = attempt_id
        self.max_files_per_commit = max_files_per_commit
        self.token_provider = token_provider or (lambda: os.environ.get("HF_TOKEN"))

        # Keep 32 of the user-provided 128 requests/hour as emergency headroom.
        scheduled_request_budget = 96
        global_flushes = max(
            1,
            scheduled_request_budget
            // (active_worker_count * estimated_hub_calls_per_flush),
        )
        self.effective_interval_seconds = max(
            interval_seconds, math.ceil(3600 / global_flushes)
        )
        if not 0 <= initial_stagger_seconds < self.effective_interval_seconds:
            raise ValueError(
                "initial_stagger_seconds must satisfy 0 <= stagger < effective interval"
            )
        self.initial_stagger_seconds = initial_stagger_seconds
        self.max_commit_attempts_per_hour = min(
            max_commit_attempts_per_hour, global_flushes
        )

        self.binding = _binding_payload(identity, repo_id, repo_type, revision)
        self.binding_hash = _binding_hash(self.binding)
        self.revision_root = self.paths.sync / "revisions" / self.binding_hash
        self.revision_root.mkdir(parents=True, exist_ok=True)
        for name in ("seals", "event_deltas", "meta_snapshots", "receipts"):
            (self.revision_root / name).mkdir(parents=True, exist_ok=True)
        self.state_path = self.revision_root / "state.json"
        self.lock_path = self.revision_root / "uploader.lock"
        self.latest_path = self.revision_root / "latest.json"
        self.owner_path = self.revision_root / "OWNER.json"
        if not self.state_path.exists():
            self._save_state(self._new_state())

    @property
    def remote_prefix(self) -> str:
        return _remote_prefix(self.identity)

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA,
            "binding": self.binding,
            "uploaded": {},
            "attempt_times": [],
            # A frozen per-worker offset prevents every collaborator from making
            # the first periodic request at the same instant. Zero preserves the
            # historical immediate-first-poll behavior.
            "last_success_epoch": (
                time.time()
                - self.effective_interval_seconds
                + self.initial_stagger_seconds
            ),
            "event_offset": 0,
            "last_commit_sha": None,
            "inflight": None,
            "verified_seal_sha256": None,
            "verified_seal_remote_path": None,
            "verified_latest_sha256": None,
            "verified_commit_sha": None,
        }

    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._new_state()
        state = read_json(self.state_path)
        if state.get("schema_version") != self.STATE_SCHEMA:
            raise RuntimeError("Unsupported sync-state schema")
        if state.get("binding") != self.binding:
            raise RuntimeError(
                "Sync state belongs to a different repo/revision/worker/spec binding"
            )
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["binding"] = self.binding
        atomic_write_json(self.state_path, state)

    def due(self) -> bool:
        state = self._state()
        return (
            time.time() - float(state.get("last_success_epoch", 0.0))
            >= self.effective_interval_seconds
        )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _acquire(self) -> bool:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    canonical_json(
                        {"pid": os.getpid(), "attempt_id": self.attempt_id, "at": utc_now()}
                    )
                )
            return True
        except FileExistsError:
            try:
                owner = read_json(self.lock_path)
                if not self._pid_alive(int(owner.get("pid", -1))):
                    self.lock_path.unlink()
                    return self._acquire()
            except (FileNotFoundError, ValueError, OSError):
                return False
            return False

    def _release(self) -> None:
        self.lock_path.unlink(missing_ok=True)

    def _owner_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "binding": self.binding,
            "remote_prefix": self.remote_prefix,
            "base_revision": self.base_revision,
        }

    def _consume_attempt_budget(self, state: dict[str, Any]) -> None:
        cutoff = time.time() - 3600
        attempts = [float(value) for value in state.get("attempt_times", []) if value >= cutoff]
        if len(attempts) >= self.max_commit_attempts_per_hour:
            raise RuntimeError(
                "Hugging Face commit-attempt budget exhausted for this worker lane; "
                "sealed local work remains queued."
            )
        attempts.append(time.time())
        state["attempt_times"] = attempts
        self._save_state(state)

    def _download_verified(
        self,
        remote_path: str,
        expected_sha256: str,
        revision: str,
        token: str,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=revision,
                filename=remote_path,
                token=token,
                cache_dir=self.revision_root / "verify_cache",
            )
        )
        if sha256_file(downloaded) != expected_sha256:
            raise RuntimeError(f"Remote checksum mismatch: {remote_path}")
        return downloaded

    def _ensure_branch_owner(self, api: Any, token: str, state: dict[str, Any]) -> dict[str, Any]:
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.utils import EntryNotFoundError

        api.create_branch(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            branch=self.revision,
            revision=self.base_revision,
            exist_ok=True,
        )
        head = str(
            api.repo_info(
                repo_id=self.repo_id, repo_type=self.repo_type, revision=self.revision
            ).sha
        )
        owner_remote = f"{self.remote_prefix}/OWNER.json"
        expected = self._owner_payload()
        atomic_write_json(self.owner_path, expected)
        try:
            downloaded = self._download_verified(
                owner_remote, sha256_file(self.owner_path), head, token
            )
            if read_json(downloaded) != expected:
                raise RuntimeError("Worker branch OWNER.json does not match this binding")
        except EntryNotFoundError:
            base_head = str(
                api.repo_info(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.base_revision,
                ).sha
            )
            if head != base_head:
                raise RuntimeError(
                    "Existing worker revision has commits but no valid OWNER.json; freeze it"
                )
            self._consume_attempt_budget(state)
            result = api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                parent_commit=head,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=owner_remote,
                        path_or_fileobj=str(self.owner_path),
                    )
                ],
                commit_message=f"Initialize worker lane {self.identity.worker_id}",
            )
            head = str(getattr(result, "oid", None) or getattr(result, "commit_id", None))
            if head in {"None", ""}:
                head = str(
                    api.repo_info(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        revision=self.revision,
                    ).sha
                )
            self._download_verified(owner_remote, sha256_file(self.owner_path), head, token)

        last_known = state.get("last_commit_sha")
        if last_known and str(last_known) != head:
            state = self._adopt_inflight_or_raise(api, token, state, head)
        elif not last_known:
            state["last_commit_sha"] = head
            self._save_state(state)
        return state

    def _freeze_event_delta(self, state: dict[str, Any]) -> None:
        start = int(state.get("event_offset", 0))
        payload, end = self.events.snapshot(start)
        if not payload:
            return
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        digest = sha256_bytes(compressed)
        path = (
            self.revision_root
            / "event_deltas"
            / f"events-{start:012d}-{end:012d}-{digest[:12]}.jsonl.gz"
        )
        if not path.exists():
            atomic_write_bytes(path, compressed)

    def _freeze_meta(self) -> None:
        for source in sorted(self.paths.meta.glob("*")):
            if not source.is_file():
                continue
            digest = sha256_file(source)
            suffix = "".join(source.suffixes)
            stem = source.name[: -len(suffix)] if suffix else source.name
            destination = self.revision_root / "meta_snapshots" / (
                f"{stem}-{digest[:12]}{suffix}"
            )
            if not destination.exists():
                atomic_copy_file(source, destination)

    def _data_catalog(self, state: dict[str, Any]) -> list[UploadItem]:
        self._freeze_event_delta(state)
        self._freeze_meta()
        items: list[UploadItem] = []
        manifest = self.manifest.read()
        for record in manifest.get("shards", []):
            kind = record.get("kind")
            if kind not in {"traces", "failures"}:
                raise RuntimeError(f"Unsupported shard kind in manifest: {kind!r}")
            relative = _safe_relative(record.get("relative_path"), (str(kind),))
            local = _safe_destination(self.paths.worker_root, relative)
            if not local.is_file():
                raise RuntimeError(f"Manifest shard is missing: {relative}")
            digest = sha256_file(local)
            if digest != record.get("sha256"):
                raise RuntimeError(f"Immutable manifest shard checksum changed: {relative}")
            items.append(
                UploadItem(
                    local,
                    f"{self.remote_prefix}/{relative}",
                    digest,
                    restore_relative=relative,
                )
            )
        for seal in sorted(self.paths.checkpoints.glob("step-*-*/_COMPLETE.json")):
            for local in sorted(path for path in seal.parent.rglob("*") if path.is_file()):
                relative = local.relative_to(self.paths.worker_root).as_posix()
                _safe_relative(relative, ("checkpoints",))
                items.append(
                    UploadItem(
                        local,
                        f"{self.remote_prefix}/{relative}",
                        sha256_file(local),
                        restore_relative=relative,
                    )
                )
        for local in sorted((self.revision_root / "event_deltas").glob("*.jsonl.gz")):
            remote_relative = f"sync/event_deltas/{local.name}"
            items.append(
                UploadItem(
                    local,
                    f"{self.remote_prefix}/{remote_relative}",
                    sha256_file(local),
                    restore_relative=(
                        f"sync/revisions/{self.binding_hash}/event_deltas/{local.name}"
                    ),
                )
            )
        for local in sorted((self.revision_root / "meta_snapshots").glob("*")):
            if not local.is_file():
                continue
            items.append(
                UploadItem(
                    local,
                    f"{self.remote_prefix}/meta/{local.name}",
                    sha256_file(local),
                    restore_relative=(
                        f"sync/revisions/{self.binding_hash}/meta_snapshots/{local.name}"
                    ),
                )
            )
        unique: dict[str, UploadItem] = {}
        for item in items:
            previous = unique.get(item.remote_path)
            if previous and previous.sha256 != item.sha256:
                raise RuntimeError(f"Remote artifact collision: {item.remote_path}")
            unique[item.remote_path] = item
        return [unique[key] for key in sorted(unique, key=_catalog_sort_key)]

    def _make_seal(self, catalog: Iterable[UploadItem]) -> tuple[UploadItem, UploadItem]:
        artifacts: list[dict[str, Any]] = []
        for item in catalog:
            if item.restore_relative is None:
                raise RuntimeError(f"Restorable artifact has no destination: {item.remote_path}")
            _safe_relative(
                item.restore_relative,
                ("traces", "failures", "checkpoints", "meta", "sync"),
            )
            artifacts.append(
                {
                    "remote_path": item.remote_path,
                    "sha256": item.sha256,
                    "bytes": item.local_path.stat().st_size,
                    "restore_relative": item.restore_relative,
                }
            )
        current_artifacts = {
            str(artifact["remote_path"]): artifact for artifact in artifacts
        }
        covered_by_remote: dict[str, dict[str, str]] = {}
        sync_state = self._state()
        prior_seal_sha = sync_state.get("verified_seal_sha256")
        prior_seal_remote = sync_state.get("verified_seal_remote_path")
        if prior_seal_sha or prior_seal_remote:
            if not isinstance(prior_seal_sha, str) or not isinstance(
                prior_seal_remote, str
            ):
                raise RuntimeError("Verified seal state is incomplete")
            prior_seal_path = (
                self.revision_root
                / "seals"
                / PurePosixPath(prior_seal_remote).name
            )
            if not prior_seal_path.is_file() or sha256_file(
                prior_seal_path
            ) != prior_seal_sha:
                raise RuntimeError("Verified local seal copy is missing or corrupt")
            prior_seal = read_json(prior_seal_path)
            if prior_seal.get("binding") != self.binding:
                raise RuntimeError("Verified local seal copy has the wrong binding")
            prior_covered = prior_seal.get("covered_outboxes", [])
            if not isinstance(prior_covered, list):
                raise RuntimeError("Verified local seal covered_outboxes is malformed")
            for raw_covered in prior_covered:
                covered = _validate_covered_outbox(raw_covered, self.remote_prefix)
                covered_by_remote[covered["remote_path"]] = covered
        for receipt_path in sorted(
            (self.revision_root / "receipts").glob("outbox-*.json")
        ):
            match = re.fullmatch(r"outbox-([0-9a-f]{64})\.json", receipt_path.name)
            if match is None:
                continue
            digest = sha256_file(receipt_path)
            if digest != match.group(1):
                raise RuntimeError(f"Local outbox receipt checksum changed: {receipt_path}")
            receipt = read_json(receipt_path)
            if receipt.get("schema_version") != 2 or receipt.get("binding") != self.binding:
                raise RuntimeError(f"Local outbox receipt has wrong binding: {receipt_path}")
            receipt_items = receipt.get("items")
            if not isinstance(receipt_items, list):
                raise RuntimeError(f"Local outbox receipt has invalid items: {receipt_path}")
            for receipt_item in receipt_items:
                if not isinstance(receipt_item, dict) or receipt_item.get("role") != "data":
                    raise RuntimeError(
                        f"Local outbox receipt contains a non-data item: {receipt_path}"
                    )
                current = current_artifacts.get(str(receipt_item.get("remote_path")))
                expected = {
                    "remote_path": receipt_item.get("remote_path"),
                    "sha256": receipt_item.get("sha256"),
                    "bytes": receipt_item.get("bytes"),
                    "restore_relative": receipt_item.get("restore_relative"),
                }
                if current != expected:
                    raise RuntimeError(
                        "Outbox artifact is not covered by the new closure: "
                        f"{receipt_item.get('remote_path')}"
                    )
            covered = {
                "remote_path": f"{self.remote_prefix}/sync/receipts/{receipt_path.name}",
                "sha256": digest,
            }
            previous = covered_by_remote.get(covered["remote_path"])
            if previous is not None and previous != covered:
                raise RuntimeError("Covered outbox checksum changed across seals")
            covered_by_remote[covered["remote_path"]] = covered
        covered_outboxes = [
            covered_by_remote[key] for key in sorted(covered_by_remote)
        ]
        payload = {
            "schema_version": 2,
            "binding": self.binding,
            "remote_prefix": self.remote_prefix,
            "manifest": self.manifest.read(),
            "artifacts": artifacts,
            # Once this seal is published, these historical data-only receipts
            # need not be downloaded during future restores.
            "covered_outboxes": covered_outboxes,
        }
        raw = (canonical_json(payload) + "\n").encode("utf-8")
        digest = sha256_bytes(raw)
        seal_path = self.revision_root / "seals" / f"seal-{digest[:16]}.json"
        if not seal_path.exists():
            atomic_write_bytes(seal_path, raw)
        seal_remote = f"{self.remote_prefix}/sync/seals/{seal_path.name}"
        latest = {
            "schema_version": 2,
            "binding": self.binding,
            "seal_remote_path": seal_remote,
            "seal_sha256": digest,
            "manifest_updated_at": payload["manifest"].get("updated_at"),
            "artifact_count": len(artifacts),
        }
        atomic_write_json(self.latest_path, latest)
        seal_item = UploadItem(seal_path, seal_remote, digest, role="seal")
        latest_item = UploadItem(
            self.latest_path,
            f"{self.remote_prefix}/sync/latest.json",
            sha256_file(self.latest_path),
            role="pointer",
        )
        return seal_item, latest_item

    def plan(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self._state() if state is None else state
        catalog = self._data_catalog(state)
        seal, latest = self._make_seal(catalog)
        uploaded = dict(state.get("uploaded", {}))
        pending_data = [
            item
            for item in catalog
            if uploaded.get(item.local_path.relative_to(self.paths.worker_root).as_posix())
            != item.sha256
        ]
        current_verified = (
            state.get("verified_seal_sha256") == seal.sha256
            and state.get("verified_latest_sha256") == latest.sha256
        )
        return {
            "catalog": catalog,
            "pending_data": pending_data,
            "seal": seal,
            "latest": latest,
            "current_verified": current_verified,
            "remaining": len(pending_data) + (0 if current_verified else 2),
        }

    def _select_batch(
        self,
        pending_data: list[UploadItem],
        seal: UploadItem,
        latest: UploadItem,
        *,
        allow_final_seal: bool = True,
    ) -> tuple[list[UploadItem], bool]:
        """Select one commit while preserving the closure-publication invariant.

        A seal and its mutable latest pointer are selected only when every still
        pending data artifact fits in the same commit. Earlier commits therefore
        contain data plus a receipt, never a pointer to an incomplete closure.
        """
        if any(item.role != "data" for item in pending_data):
            raise RuntimeError("Pending artifact batches may contain data items only")
        capacity = self.max_files_per_commit - 1  # one receipt operation is mandatory
        if not allow_final_seal:
            return pending_data[:capacity], False
        final_seal = len(pending_data) <= capacity - 2
        if final_seal:
            return pending_data + [seal, latest], True
        return pending_data[:capacity], False

    def _write_receipt(self, selected: list[UploadItem], final_seal: bool) -> UploadItem:
        payload = {
            "schema_version": 2,
            "binding": self.binding,
            "parent_commit": self._state().get("last_commit_sha"),
            "attempt_id": self.attempt_id,
            "final_seal": final_seal,
            # A data-only commit must be sufficient to reconstruct the exact
            # manifest records for its immutable shards after local disk loss.
            "manifest": self.manifest.read(),
            "items": [
                {
                    "remote_path": item.remote_path,
                    "sha256": item.sha256,
                    "bytes": item.local_path.stat().st_size,
                    "role": item.role,
                    "restore_relative": item.restore_relative,
                }
                for item in selected
            ],
        }
        raw = (canonical_json(payload) + "\n").encode("utf-8")
        digest = sha256_bytes(raw)
        # The full checksum in the filename lets a fresh runtime authenticate
        # receipts discovered by listing a pinned branch tree.
        receipt_kind = "receipt" if final_seal else "outbox"
        path = self.revision_root / "receipts" / f"{receipt_kind}-{digest}.json"
        if not path.exists():
            atomic_write_bytes(path, raw)
        return UploadItem(
            path,
            f"{self.remote_prefix}/sync/receipts/{path.name}",
            digest,
            role="receipt",
        )

    def _mark_selected_uploaded(
        self,
        state: dict[str, Any],
        selected: Iterable[dict[str, Any] | UploadItem],
    ) -> None:
        uploaded = dict(state.get("uploaded", {}))
        event_ends: list[int] = []
        for source in selected:
            if isinstance(source, UploadItem):
                local_relative = source.local_path.relative_to(self.paths.worker_root).as_posix()
                digest = source.sha256
                name = source.local_path.name
            else:
                local_relative = str(source["local_relative"])
                digest = str(source["sha256"])
                name = Path(local_relative).name
            uploaded[local_relative] = digest
            if name.startswith("events-") and name.endswith(".jsonl.gz"):
                parts = name.split("-")
                if len(parts) >= 4 and parts[2].isdigit():
                    event_ends.append(int(parts[2]))
        state["uploaded"] = uploaded
        if event_ends:
            state["event_offset"] = max(
                int(state.get("event_offset", 0)), max(event_ends)
            )

    def _adopt_inflight_or_raise(
        self, api: Any, token: str, state: dict[str, Any], head: str
    ) -> dict[str, Any]:
        inflight = state.get("inflight")
        if not inflight:
            raise RuntimeError(
                "Worker revision advanced without a locally recorded in-flight receipt"
            )
        self._download_verified(
            str(inflight["receipt_remote_path"]),
            str(inflight["receipt_sha256"]),
            head,
            token,
        )
        self._mark_selected_uploaded(state, inflight.get("items", []))
        if inflight.get("final_seal"):
            self._download_verified(
                str(inflight["seal_remote_path"]),
                str(inflight["seal_sha256"]),
                head,
                token,
            )
            self._download_verified(
                f"{self.remote_prefix}/sync/latest.json",
                str(inflight["latest_sha256"]),
                head,
                token,
            )
            state["verified_seal_sha256"] = inflight["seal_sha256"]
            state["verified_seal_remote_path"] = inflight["seal_remote_path"]
            state["verified_latest_sha256"] = inflight["latest_sha256"]
            state["verified_commit_sha"] = head
        state["last_commit_sha"] = head
        state["last_success_epoch"] = time.time()
        state["inflight"] = None
        self._save_state(state)
        return state

    def _verify_clean_state(self, api: Any, token: str, state: dict[str, Any]) -> dict[str, Any]:
        head = str(
            api.repo_info(
                repo_id=self.repo_id, repo_type=self.repo_type, revision=self.revision
            ).sha
        )
        if state.get("last_commit_sha") != head:
            state = self._adopt_inflight_or_raise(api, token, state, head)
        seal_path = state.get("verified_seal_remote_path")
        seal_sha = state.get("verified_seal_sha256")
        latest_sha = state.get("verified_latest_sha256")
        if not seal_path or not seal_sha or not latest_sha:
            return _result("UNSEALED", remaining=2)
        self._download_verified(str(seal_path), str(seal_sha), head, token)
        self._download_verified(
            f"{self.remote_prefix}/sync/latest.json", str(latest_sha), head, token
        )
        return _result(
            "CLEAN",
            verified=True,
            complete=True,
            commit_sha=head,
            revision=self.revision,
            effective_interval_seconds=self.effective_interval_seconds,
        )

    def sync_once(self, force: bool = False) -> dict[str, Any]:
        if not force and not self.due():
            return _result(
                "NOT_DUE",
                effective_interval_seconds=self.effective_interval_seconds,
            )
        if not self._acquire():
            return _result("LOCKED", remaining=1)
        try:
            token = self.token_provider()
            if not token:
                pending = self.plan()
                return _result("NO_TOKEN", remaining=int(pending["remaining"]))

            from huggingface_hub import CommitOperationAdd, HfApi

            api = HfApi(token=token)
            state = self._state()
            local_pending = self.plan(state)
            if not force and not local_pending["pending_data"]:
                if local_pending["current_verified"]:
                    return _result(
                        "CLEAN_LOCAL",
                        verified=True,
                        complete=True,
                        commit_sha=state.get("verified_commit_sha"),
                        revision=self.revision,
                        effective_interval_seconds=self.effective_interval_seconds,
                    )
                return _result(
                    "PERIODIC_CLOSURE_DEFERRED",
                    remaining=2,
                    effective_interval_seconds=self.effective_interval_seconds,
                )
            state = self._ensure_branch_owner(api, token, state)
            pending = self.plan(state)
            if not pending["pending_data"] and pending["current_verified"]:
                return self._verify_clean_state(api, token, state)

            pending_data: list[UploadItem] = list(pending["pending_data"])
            selected, final_seal = self._select_batch(
                pending_data,
                pending["seal"],
                pending["latest"],
                allow_final_seal=force,
            )
            receipt = self._write_receipt(selected, final_seal)
            batch = selected + [receipt]
            parent_commit = str(state["last_commit_sha"])
            inflight_items = [
                {
                    "local_relative": item.local_path.relative_to(
                        self.paths.worker_root
                    ).as_posix(),
                    "remote_path": item.remote_path,
                    "sha256": item.sha256,
                    "role": item.role,
                }
                for item in selected
            ]
            state["inflight"] = {
                "parent_commit": parent_commit,
                "receipt_remote_path": receipt.remote_path,
                "receipt_sha256": receipt.sha256,
                "items": inflight_items,
                "final_seal": final_seal,
                "seal_remote_path": pending["seal"].remote_path if final_seal else None,
                "seal_sha256": pending["seal"].sha256 if final_seal else None,
                "latest_sha256": pending["latest"].sha256 if final_seal else None,
            }
            self._save_state(state)
            self._consume_attempt_budget(state)
            result = api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=self.revision,
                parent_commit=parent_commit,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=item.remote_path, path_or_fileobj=str(item.local_path)
                    )
                    for item in batch
                ],
                commit_message=(
                    f"{self.identity.experiment_id}: {self.identity.worker_id} "
                    f"{'seal' if final_seal else 'data'} [{receipt.sha256[:12]}]"
                ),
            )
            commit_sha = str(
                getattr(result, "oid", None)
                or getattr(result, "commit_id", None)
                or api.repo_info(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.revision,
                ).sha
            )
            self._download_verified(receipt.remote_path, receipt.sha256, commit_sha, token)
            state = self._state()
            self._mark_selected_uploaded(state, selected)
            state["last_commit_sha"] = commit_sha
            state["last_success_epoch"] = time.time()
            state["inflight"] = None
            verified = False
            complete = False
            if final_seal:
                self._download_verified(
                    pending["seal"].remote_path,
                    pending["seal"].sha256,
                    commit_sha,
                    token,
                )
                self._download_verified(
                    pending["latest"].remote_path,
                    pending["latest"].sha256,
                    commit_sha,
                    token,
                )
                state["verified_seal_sha256"] = pending["seal"].sha256
                state["verified_seal_remote_path"] = pending["seal"].remote_path
                state["verified_latest_sha256"] = pending["latest"].sha256
                state["verified_commit_sha"] = commit_sha
                verified = True
                complete = True
            self._save_state(state)
            uploaded_data = len(pending_data) if final_seal else len(selected)
            remaining_data = max(0, len(pending_data) - uploaded_data)
            remaining = remaining_data + (0 if final_seal else 2)
            return _result(
                "UPLOADED" if final_seal else "UPLOADED_DATA",
                uploaded=len(selected),
                remaining=remaining,
                verified=verified,
                complete=complete,
                commit=str(getattr(result, "commit_url", result)),
                commit_sha=commit_sha,
                revision=self.revision,
                effective_interval_seconds=self.effective_interval_seconds,
            )
        except Exception as error:
            self.events.append(
                "HF_SYNC_FAILED", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            self._release()

    def sync_until_clean(
        self, max_commits: int = 8, deadline_seconds: float = 300.0
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        deadline = time.monotonic() + deadline_seconds
        for _ in range(max_commits):
            if time.monotonic() >= deadline:
                results.append(_result("DEADLINE", remaining=1))
                break
            result = self.sync_once(force=True)
            results.append(result)
            if result["complete"] and result["verified"]:
                break
            if result["status"] in {"NO_TOKEN", "LOCKED"}:
                break
        return results


_RESTORE_ROOTS = ("traces", "failures", "checkpoints", "meta", "sync")
_RECEIPT_NAME_RE = re.compile(
    r"^(outbox|receipt)-([0-9a-f]{16}|[0-9a-f]{64})\.json$"
)


def _validate_manifest_snapshot(
    value: Any, identity: RunIdentity, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("identity") != identity.as_dict():
        raise RuntimeError(f"{label} manifest identity mismatch")
    if value.get("schema_version") != ManifestStore.SCHEMA_VERSION:
        raise RuntimeError(f"{label} manifest has an unsupported schema version")
    if not isinstance(value.get("metadata"), dict):
        raise RuntimeError(f"{label} manifest metadata must be an object")
    if not isinstance(value.get("created_at"), str):
        raise RuntimeError(f"{label} manifest is missing created_at")
    shards = value.get("shards")
    if not isinstance(shards, list):
        raise RuntimeError(f"{label} manifest shards must be a list")

    seen_paths: set[str] = set()
    seen_sequences: set[tuple[str, int]] = set()
    expected_counters = {"trace_rows": 0, "failure_rows": 0}
    for record in shards:
        if not isinstance(record, dict):
            raise RuntimeError(f"{label} manifest contains a malformed shard")
        kind = record.get("kind")
        if kind not in {"traces", "failures"}:
            raise RuntimeError(f"{label} manifest has unsupported shard kind: {kind!r}")
        relative = _safe_relative(record.get("relative_path"), (str(kind),))
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"{label} manifest shard has an invalid checksum")
        row_count = record.get("row_count")
        sequence = record.get("sequence")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
            raise RuntimeError(f"{label} manifest shard has an invalid row count")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RuntimeError(f"{label} manifest shard has an invalid sequence")
        expected_name = f"{kind}/part-{sequence:06d}-{digest[:12]}.jsonl.gz"
        if relative != expected_name:
            raise RuntimeError(f"{label} manifest shard name is not content addressed")
        if relative in seen_paths or (str(kind), sequence) in seen_sequences:
            raise RuntimeError(f"{label} manifest contains duplicate shard coordinates")
        seen_paths.add(relative)
        seen_sequences.add((str(kind), sequence))
        counter = "trace_rows" if kind == "traces" else "failure_rows"
        expected_counters[counter] += row_count

    counters = value.get("counters")
    if not isinstance(counters, dict) or any(
        int(counters.get(key, -1)) != count for key, count in expected_counters.items()
    ):
        raise RuntimeError(f"{label} manifest counters do not match its shards")
    return value


def _expected_restore_relative(remote_relative: str, binding_hash: str) -> str:
    parts = PurePosixPath(remote_relative).parts
    if parts[0] in {"traces", "failures", "checkpoints"}:
        return remote_relative
    if parts[0] == "meta" and len(parts) >= 2:
        tail = PurePosixPath(*parts[1:]).as_posix()
        return f"sync/revisions/{binding_hash}/meta_snapshots/{tail}"
    if len(parts) >= 3 and parts[:2] == ("sync", "event_deltas"):
        tail = PurePosixPath(*parts[2:]).as_posix()
        return f"sync/revisions/{binding_hash}/event_deltas/{tail}"
    raise RuntimeError(f"Artifact uses an unsupported restore mapping: {remote_relative!r}")


def _validate_remote_artifact(
    value: Any,
    *,
    prefix: str,
    binding_hash: str,
) -> _RemoteArtifact:
    if not isinstance(value, dict):
        raise RuntimeError("Remote closure contains a malformed artifact descriptor")
    remote_path = value.get("remote_path")
    expected_prefix = f"{prefix}/"
    if not isinstance(remote_path, str) or not remote_path.startswith(expected_prefix):
        raise RuntimeError(f"Remote artifact escapes worker prefix: {remote_path!r}")
    remote_relative = _safe_relative(remote_path[len(expected_prefix) :], _RESTORE_ROOTS)
    restore_relative = _safe_relative(value.get("restore_relative"), _RESTORE_ROOTS)
    expected_restore = _expected_restore_relative(remote_relative, binding_hash)
    if restore_relative != expected_restore:
        raise RuntimeError(
            f"Remote artifact restore mapping is inconsistent: {remote_path}"
        )
    digest = value.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"Remote artifact has invalid checksum: {remote_path}")
    byte_count = value.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise RuntimeError(f"Remote artifact has invalid byte size: {remote_path}")
    return _RemoteArtifact(remote_path, digest, byte_count, restore_relative)


def _add_artifact(
    artifact: _RemoteArtifact,
    by_remote: dict[str, _RemoteArtifact],
    by_restore: dict[str, _RemoteArtifact],
    *,
    source: str,
) -> bool:
    previous_remote = by_remote.get(artifact.remote_path)
    previous_restore = by_restore.get(artifact.restore_relative)
    if previous_remote is not None and previous_remote != artifact:
        raise RuntimeError(f"Conflicting {source} artifact remote path")
    if previous_restore is not None and previous_restore != artifact:
        raise RuntimeError(f"Conflicting {source} artifact restore path")
    if previous_remote is not None or previous_restore is not None:
        if previous_remote != artifact or previous_restore != artifact:
            raise RuntimeError(f"Ambiguous {source} artifact alias")
        return False
    by_remote[artifact.remote_path] = artifact
    by_restore[artifact.restore_relative] = artifact
    return True


def _receipt_shard_record(
    manifest: dict[str, Any], artifact: _RemoteArtifact
) -> dict[str, Any] | None:
    root = PurePosixPath(artifact.restore_relative).parts[0]
    if root not in {"traces", "failures"}:
        return None
    matches = [
        record
        for record in manifest["shards"]
        if record.get("relative_path") == artifact.restore_relative
    ]
    if len(matches) != 1 or matches[0].get("sha256") != artifact.sha256:
        raise RuntimeError(
            f"Receipt manifest does not authenticate shard {artifact.restore_relative}"
        )
    return matches[0]


def _recovered_manifest(
    *,
    base_manifest: dict[str, Any] | None,
    receipt_manifests: list[dict[str, Any]],
    orphan_records: dict[str, dict[str, Any]],
    orphan_count: int,
) -> dict[str, Any]:
    lineage = ([base_manifest] if base_manifest is not None else []) + receipt_manifests
    if not lineage:
        raise RuntimeError("Cannot initialize a manifest without a verified receipt")
    created_at = lineage[0]["created_at"]
    metadata = lineage[0]["metadata"]
    for manifest in lineage[1:]:
        if manifest["created_at"] != created_at or manifest["metadata"] != metadata:
            raise RuntimeError("Receipt manifests belong to conflicting run lineages")

    result = copy.deepcopy(base_manifest if base_manifest is not None else lineage[0])
    records = {
        str(record["relative_path"]): copy.deepcopy(record)
        for record in (base_manifest or {}).get("shards", [])
    }
    for relative, record in orphan_records.items():
        previous = records.get(relative)
        if previous is not None and previous != record:
            raise RuntimeError(f"Conflicting recovered shard record: {relative}")
        records[relative] = copy.deepcopy(record)

    sequence_owner: dict[tuple[str, int], str] = {}
    for relative, record in records.items():
        coordinate = (str(record["kind"]), int(record["sequence"]))
        previous = sequence_owner.get(coordinate)
        if previous is not None and previous != relative:
            raise RuntimeError("Recovered shards conflict on kind and sequence")
        sequence_owner[coordinate] = relative
    result["shards"] = sorted(
        records.values(),
        key=lambda record: (
            str(record["kind"]),
            int(record["sequence"]),
            str(record["relative_path"]),
        ),
    )
    result["counters"] = {
        "trace_rows": sum(
            int(record["row_count"])
            for record in result["shards"]
            if record["kind"] == "traces"
        ),
        "failure_rows": sum(
            int(record["row_count"])
            for record in result["shards"]
            if record["kind"] == "failures"
        ),
    }
    if orphan_count:
        result["status"] = "INTERRUPTED"
        result["status_details"] = {
            "reason": "RESTORED_WITH_UNSEALED_OUTBOX",
            "orphan_artifacts": orphan_count,
        }
        result["updated_at"] = utc_now()
    return result


def restore_worker_from_hub(
    paths: RunPaths,
    identity: RunIdentity,
    repo_id: str,
    revision: str,
    token: str,
    repo_type: str = "dataset",
) -> dict[str, Any]:
    """Restore a pinned closure plus authenticated post-closure receipt outbox."""
    paths.create()
    if paths.manifest.exists():
        local_manifest = read_json(paths.manifest)
        if local_manifest.get("identity") != identity.as_dict():
            raise RuntimeError("Local manifest identity does not match requested restore")
        return {"status": "LOCAL_PRESENT", "restored_artifacts": 0}

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    binding = _binding_payload(identity, repo_id, repo_type, revision)
    binding_hash = _binding_hash(binding)
    prefix = _remote_prefix(identity)
    api = HfApi(token=token)
    try:
        pinned_head = str(
            api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision).sha
        )
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return {"status": "REMOTE_NOT_FOUND", "restored_artifacts": 0}

    cache = paths.sync / "restore_cache"
    latest_remote = f"{prefix}/sync/latest.json"
    latest_download: Path | None = None
    seal_download: Path | None = None
    latest: dict[str, Any] | None = None
    seal_remote: str | None = None
    expected_seal_sha: str | None = None
    base_manifest: dict[str, Any] | None = None
    base_by_remote: dict[str, _RemoteArtifact] = {}
    base_by_restore: dict[str, _RemoteArtifact] = {}
    covered_outbox_paths: set[str] = set()

    try:
        latest_download = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=pinned_head,
                filename=latest_remote,
                token=token,
                cache_dir=cache,
            )
        )
    except EntryNotFoundError:
        latest_download = None

    if latest_download is not None:
        latest = read_json(latest_download)
        if latest.get("binding") != binding:
            raise RuntimeError("Remote latest pointer belongs to a different binding")
        seal_remote_value = latest.get("seal_remote_path")
        expected_seal_prefix = f"{prefix}/sync/seals/"
        if not isinstance(seal_remote_value, str) or not seal_remote_value.startswith(
            expected_seal_prefix
        ):
            raise RuntimeError("Remote latest pointer escapes this worker's seal directory")
        seal_remote = seal_remote_value
        _safe_relative(seal_remote[len(f"{prefix}/") :], ("sync",))
        expected_seal_sha = latest.get("seal_sha256")
        if not isinstance(expected_seal_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_seal_sha
        ):
            raise RuntimeError("Remote latest pointer contains an invalid seal checksum")
        seal_download = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=pinned_head,
                filename=seal_remote,
                token=token,
                cache_dir=cache,
            )
        )
        if sha256_file(seal_download) != expected_seal_sha:
            raise RuntimeError("Remote closure seal checksum mismatch")
        seal = read_json(seal_download)
        if seal.get("binding") != binding or seal.get("remote_prefix") != prefix:
            raise RuntimeError("Remote closure seal belongs to a different binding")
        base_manifest = _validate_manifest_snapshot(
            seal.get("manifest"), identity, "Sealed"
        )
        raw_artifacts = seal.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise RuntimeError("Remote closure seal artifacts must be a list")
        artifact_count = latest.get("artifact_count")
        if (
            isinstance(artifact_count, bool)
            or not isinstance(artifact_count, int)
            or artifact_count != len(raw_artifacts)
        ):
            raise RuntimeError(
                "Remote latest pointer artifact count does not match its seal"
            )
        for raw_artifact in raw_artifacts:
            artifact = _validate_remote_artifact(
                raw_artifact, prefix=prefix, binding_hash=binding_hash
            )
            _add_artifact(
                artifact,
                base_by_remote,
                base_by_restore,
                source="sealed closure",
            )
        covered_outboxes = seal.get("covered_outboxes", [])
        if not isinstance(covered_outboxes, list):
            raise RuntimeError("Remote closure covered_outboxes must be a list")
        for raw_covered in covered_outboxes:
            covered = _validate_covered_outbox(raw_covered, prefix)
            covered_remote = covered["remote_path"]
            if covered_remote in covered_outbox_paths:
                raise RuntimeError("Remote closure repeats a covered outbox")
            covered_outbox_paths.add(covered_remote)
        for record in base_manifest["shards"]:
            artifact = base_by_restore.get(str(record["relative_path"]))
            if artifact is None or artifact.sha256 != record.get("sha256"):
                raise RuntimeError(
                    f"Sealed manifest shard is absent from closure: "
                    f"{record['relative_path']}"
                )

    receipt_prefix = f"{prefix}/sync/receipts/"
    receipt_files = sorted(
        path
        for path in api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=pinned_head,
        )
        if isinstance(path, str) and path.startswith(receipt_prefix)
    )
    receipt_manifests: list[dict[str, Any]] = []
    receipt_downloads: list[Path] = []
    receipt_artifacts: list[tuple[_RemoteArtifact, dict[str, Any]]] = []
    legacy_data: list[tuple[str, str]] = []
    for receipt_remote in receipt_files:
        receipt_relative = _safe_relative(
            receipt_remote[len(f"{prefix}/") :], ("sync",)
        )
        receipt_parts = PurePosixPath(receipt_relative).parts
        if len(receipt_parts) != 3 or receipt_parts[:2] != ("sync", "receipts"):
            raise RuntimeError("Remote receipt path is outside the canonical receipt directory")
        match = _RECEIPT_NAME_RE.fullmatch(PurePosixPath(receipt_remote).name)
        if match is None:
            continue
        receipt_kind = match.group(1)
        advertised_digest = match.group(2)
        # Schema-2 final receipts are already covered by the atomically committed
        # latest seal. Only data-only outbox receipts can contain newer work.
        if receipt_kind == "receipt" and len(advertised_digest) == 64:
            continue
        if receipt_remote in covered_outbox_paths:
            continue
        receipt_download = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=pinned_head,
                filename=receipt_remote,
                token=token,
                cache_dir=cache,
            )
        )
        receipt_digest = sha256_file(receipt_download)
        if (
            len(advertised_digest) == 64
            and receipt_digest != advertised_digest
        ) or (
            len(advertised_digest) == 16
            and not receipt_digest.startswith(advertised_digest)
        ):
            raise RuntimeError(f"Remote receipt checksum mismatch: {receipt_remote}")
        receipt = read_json(receipt_download)
        if receipt.get("binding") != binding:
            raise RuntimeError("Remote receipt belongs to a different binding")
        items = receipt.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Remote receipt items must be a list")
        if receipt.get("schema_version") == 1:
            for item in items:
                if isinstance(item, dict) and item.get("role") == "data":
                    remote_path = item.get("remote_path")
                    digest = item.get("sha256")
                    if not isinstance(remote_path, str) or not isinstance(digest, str):
                        raise RuntimeError("Legacy receipt has a malformed data item")
                    legacy_data.append((remote_path, digest))
            continue
        if receipt.get("schema_version") != 2:
            raise RuntimeError("Unsupported remote receipt schema")
        receipt_manifest = _validate_manifest_snapshot(
            receipt.get("manifest"), identity, "Receipt"
        )
        receipt_manifests.append(receipt_manifest)
        receipt_downloads.append(receipt_download)
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError("Remote receipt contains a malformed item")
            role = item.get("role")
            if role not in {"data", "seal", "pointer"}:
                raise RuntimeError(f"Remote receipt contains unsupported role: {role!r}")
            if role != "data":
                continue
            artifact = _validate_remote_artifact(
                item, prefix=prefix, binding_hash=binding_hash
            )
            _receipt_shard_record(receipt_manifest, artifact)
            receipt_artifacts.append((artifact, receipt_manifest))

    for remote_path, digest in legacy_data:
        sealed = base_by_remote.get(remote_path)
        if sealed is None or sealed.sha256 != digest:
            raise RuntimeError(
                "Legacy unsealed receipt cannot be recovered automatically; "
                "freeze this lane for manual inspection"
            )

    all_by_remote = dict(base_by_remote)
    all_by_restore = dict(base_by_restore)
    orphan_artifacts: list[_RemoteArtifact] = []
    orphan_records: dict[str, dict[str, Any]] = {}
    known_records = {
        str(record["relative_path"]): record
        for record in (base_manifest or {}).get("shards", [])
    }
    for artifact, receipt_manifest in receipt_artifacts:
        shard_record = _receipt_shard_record(receipt_manifest, artifact)
        if shard_record is not None:
            previous_record = known_records.get(artifact.restore_relative)
            if previous_record is not None and previous_record != shard_record:
                raise RuntimeError("Receipts contain conflicting shard records")
            known_records[artifact.restore_relative] = shard_record
        added = _add_artifact(
            artifact,
            all_by_remote,
            all_by_restore,
            source="receipt outbox",
        )
        if not added:
            continue
        orphan_artifacts.append(artifact)
        if shard_record is not None:
            orphan_records[artifact.restore_relative] = shard_record

    if base_manifest is None and not orphan_artifacts:
        return {"status": "REMOTE_UNSEALED", "restored_artifacts": 0}

    restored_manifest = _recovered_manifest(
        base_manifest=base_manifest,
        receipt_manifests=receipt_manifests,
        orphan_records=orphan_records,
        orphan_count=len(orphan_artifacts),
    )

    # Validate the complete pinned closure/outbox before writing experiment data.
    downloaded_artifacts: list[tuple[Path, _RemoteArtifact]] = []
    for artifact in sorted(all_by_remote.values(), key=lambda item: item.remote_path):
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=pinned_head,
                filename=artifact.remote_path,
                token=token,
                cache_dir=cache,
            )
        )
        if downloaded.stat().st_size != artifact.bytes:
            raise RuntimeError(
                f"Remote artifact byte-size mismatch: {artifact.remote_path}"
            )
        if sha256_file(downloaded) != artifact.sha256:
            raise RuntimeError(f"Remote artifact checksum mismatch: {artifact.remote_path}")
        downloaded_artifacts.append((downloaded, artifact))

    uploaded: dict[str, str] = {}
    for downloaded, artifact in downloaded_artifacts:
        destination = _safe_destination(paths.worker_root, artifact.restore_relative)
        atomic_copy_file(downloaded, destination)
        uploaded[artifact.restore_relative] = artifact.sha256

    revision_root = paths.sync / "revisions" / binding_hash
    for name in ("seals", "receipts"):
        (revision_root / name).mkdir(parents=True, exist_ok=True)
    for receipt_download in receipt_downloads:
        atomic_copy_file(
            receipt_download,
            revision_root / "receipts" / receipt_download.name,
        )

    verified_seal_sha: str | None = None
    verified_seal_remote: str | None = None
    verified_latest_sha: str | None = None
    verified_commit_sha: str | None = None
    if (
        latest_download is not None
        and seal_download is not None
        and seal_remote is not None
        and expected_seal_sha is not None
    ):
        seal_local = revision_root / "seals" / PurePosixPath(seal_remote).name
        latest_local = revision_root / "latest.json"
        atomic_copy_file(seal_download, seal_local)
        atomic_copy_file(latest_download, latest_local)
        uploaded[seal_local.relative_to(paths.worker_root).as_posix()] = sha256_file(
            seal_local
        )
        uploaded[latest_local.relative_to(paths.worker_root).as_posix()] = sha256_file(
            latest_local
        )
        verified_seal_sha = expected_seal_sha
        verified_seal_remote = seal_remote
        verified_latest_sha = sha256_file(latest_download)
        verified_commit_sha = pinned_head

    state = {
        "schema_version": SyncManager.STATE_SCHEMA,
        "binding": binding,
        "uploaded": uploaded,
        "attempt_times": [],
        "last_success_epoch": time.time(),
        "event_offset": 0,
        "last_commit_sha": pinned_head,
        "inflight": None,
        "verified_seal_sha256": verified_seal_sha,
        "verified_seal_remote_path": verified_seal_remote,
        "verified_latest_sha256": verified_latest_sha,
        "verified_commit_sha": verified_commit_sha,
    }
    atomic_write_json(revision_root / "state.json", state)
    # Manifest is the local restore commit marker and is intentionally written last.
    atomic_write_json(paths.manifest, restored_manifest)

    outbox_recovered = bool(orphan_artifacts)
    return {
        "status": (
            "RESTORED_WITH_UNSEALED_OUTBOX" if outbox_recovered else "RESTORED"
        ),
        "restored_artifacts": len(all_by_remote),
        "orphan_artifacts": len(orphan_artifacts),
        "receipt_count": len(receipt_downloads),
        "commit_sha": pinned_head,
        "revision": revision,
        "seal_sha256": expected_seal_sha,
        "closure_verified": base_manifest is not None,
    }
