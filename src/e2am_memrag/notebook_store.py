from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping

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


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
_HF_TOKEN_BYTES_RE = re.compile(rb"hf_[A-Za-z0-9]{8,}")


class HubAuthenticationDisabled(RuntimeError):
    """Raised after one 401/403 so a bad credential is not hammered repeatedly."""


class WorkerBranchConflict(RuntimeError):
    """Raised when the fixed worker branch advanced outside this local store."""


class RestoreRequired(RuntimeError):
    """Raised when a fresh local store would overwrite an existing remote lane."""


def _http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        if not isinstance(value, str):
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _safe_identifier(label: str, value: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    return value


def _safe_logical_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    return path.as_posix()


def _safe_destination(root: Path, logical_path: str) -> Path:
    destination = root.joinpath(*PurePosixPath(logical_path).parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if not resolved_destination.is_relative_to(resolved_root):
        raise RuntimeError(f"Artifact destination escapes restore root: {logical_path!r}")
    return destination


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        raw = (canonical_json(dict(value)) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Metadata must be finite deterministic JSON") from error
    if _HF_TOKEN_RE.search(raw.decode("utf-8")):
        raise ValueError("Secret-like Hugging Face token found in serializable metadata")
    token = os.environ.get("HF_TOKEN")
    if token and token.encode("utf-8") in raw:
        raise ValueError("HF_TOKEN must never be serialized into notebook artifacts")
    return raw


class NotebookArtifactStore:
    """Small, branch-bound artifact store intended for standalone Kaggle notebooks.

    A store owns one fixed worker branch. Logical artifact names may change content,
    but every payload is uploaded under its SHA-256. A commit contains all new
    payloads, an immutable manifest, and the worker's ``LATEST.json`` pointer. The
    pointer is therefore never visible before all dependencies exist.

    The class is deliberately synchronous. It assumes that one notebook session at
    a time writes a worker lane, while optimistic parent commits still detect an
    accidental second writer.
    """

    STATE_SCHEMA = 1
    MANIFEST_SCHEMA = 1
    POINTER_SCHEMA = 1

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str,
        experiment_id: str,
        worker_id: str,
        branch: str | None = None,
        remote_prefix: str | None = None,
        repo_type: str = "dataset",
        base_revision: str = "main",
        sync_interval_seconds: int = 1200,
        max_attempts: int = 5,
        maximum_backoff_seconds: float = 60.0,
        token_provider: Callable[[], str | None] | None = None,
        api_factory: Callable[[str], Any] | None = None,
        download_file: Callable[..., str] | None = None,
        operation_factory: Callable[[str, Path], Any] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(repo_id, str) or "/" not in repo_id or repo_id.startswith("/"):
            raise ValueError("repo_id must be a Hugging Face namespace/repository")
        self.experiment_id = _safe_identifier("experiment_id", experiment_id)
        self.worker_id = _safe_identifier("worker_id", worker_id)
        default_branch = f"worker-{self.experiment_id}-{self.worker_id}"
        self.branch = _safe_identifier("branch", branch or default_branch)
        self.base_revision = _safe_identifier("base_revision", base_revision)
        if repo_type not in {"dataset", "model", "space"}:
            raise ValueError("repo_type must be dataset, model, or space")
        if sync_interval_seconds < 300:
            raise ValueError("sync_interval_seconds must be at least 300 seconds")
        if max_attempts < 1 or maximum_backoff_seconds < 0:
            raise ValueError("Invalid Hub retry configuration")

        self.root = Path(root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        self.downloads = self.root / "downloads"
        for directory in (self.root, self.objects, self.manifests, self.downloads):
            directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.sync_interval_seconds = int(sync_interval_seconds)
        self.max_attempts = int(max_attempts)
        self.maximum_backoff_seconds = float(maximum_backoff_seconds)
        self.token_provider = token_provider or (lambda: os.environ.get("HF_TOKEN"))
        self._api_factory = api_factory
        self._download_file = download_file
        self._operation_factory = operation_factory
        self._clock = clock
        self._sleep = sleeper
        self._auth_disabled = False
        self._remote_initialized = False
        self._lock = threading.RLock()
        self.last_flush: dict[str, Any] | None = None

        resolved_prefix = (
            _safe_logical_path(remote_prefix)
            if remote_prefix is not None
            else f"experiments/{self.experiment_id}/workers/{self.worker_id}"
        )
        self.binding = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "experiment_id": self.experiment_id,
            "worker_id": self.worker_id,
            "branch": self.branch,
            "base_revision": self.base_revision,
            "remote_prefix": resolved_prefix,
        }
        self.remote_prefix = resolved_prefix
        self.pointer_path = f"{self.remote_prefix}/LATEST.json"
        if self.state_path.exists():
            self._read_state()
        else:
            self._write_state(self._new_state())

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA,
            "binding": self.binding,
            "artifacts": {},
            "uploaded": {},
            "dirty": False,
            "last_sync_epoch": self._clock(),
            "last_commit_sha": None,
            "last_manifest_sha256": None,
            "inflight": None,
        }

    def _read_state(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        if state.get("schema_version") != self.STATE_SCHEMA:
            raise RuntimeError("Unsupported notebook-store state schema")
        if state.get("binding") != self.binding:
            raise RuntimeError(
                "Notebook-store directory belongs to another repo/branch/worker binding"
            )
        if not isinstance(state.get("artifacts"), dict) or not isinstance(
            state.get("uploaded"), dict
        ):
            raise RuntimeError("Notebook-store state is malformed")
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        state["schema_version"] = self.STATE_SCHEMA
        state["binding"] = self.binding
        # This also rejects an accidentally embedded live HF_TOKEN.
        _json_bytes(state)
        atomic_write_json(self.state_path, state)

    def _token(self) -> str:
        if self._auth_disabled:
            raise HubAuthenticationDisabled(
                "Hub authentication is disabled after a 401/403; correct HF_TOKEN and "
                "call reset_authentication() before another request"
            )
        token = self.token_provider()
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("HF_TOKEN is missing; add it as a Kaggle Secret")
        return token.strip()

    def reset_authentication(self) -> None:
        """Explicitly allow another request after the user has corrected HF_TOKEN."""

        self._auth_disabled = False

    def _call(self, operation: Callable[[], Any]) -> Any:
        if self._auth_disabled:
            raise HubAuthenticationDisabled(
                "Hub authentication remains disabled after a 401/403"
            )
        for attempt in range(self.max_attempts):
            try:
                return operation()
            except Exception as error:
                status = _http_status(error)
                if status in {401, 403}:
                    self._auth_disabled = True
                    raise
                retryable = status == 429 or (status is not None and 500 <= status < 600)
                if not retryable or attempt + 1 == self.max_attempts:
                    raise
                delay = _retry_after(error)
                if delay is None:
                    delay = min(self.maximum_backoff_seconds, 2.0**attempt)
                # Retry-After is a server instruction, not an exponential-backoff
                # suggestion, so do not shorten it with the local fallback cap.
                self._sleep(delay)
        raise RuntimeError("Unreachable Hub retry state")

    def _api(self, token: str) -> Any:
        if self._api_factory is not None:
            return self._api_factory(token)
        from huggingface_hub import HfApi

        return HfApi(token=token)

    def _operation(self, remote_path: str, local_path: Path) -> Any:
        if self._operation_factory is not None:
            return self._operation_factory(remote_path, local_path)
        from huggingface_hub import CommitOperationAdd

        return CommitOperationAdd(
            path_in_repo=remote_path, path_or_fileobj=str(local_path)
        )

    def _download(self, *, remote_path: str, revision: str, token: str) -> bytes:
        if self._download_file is None:
            from huggingface_hub import hf_hub_download

            download = hf_hub_download
        else:
            download = self._download_file
        local = Path(
            self._call(
                lambda: download(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=revision,
                    filename=remote_path,
                    token=token,
                    cache_dir=self.downloads,
                )
            )
        )
        return local.read_bytes()

    def put_bytes(
        self,
        logical_path: str,
        payload: bytes,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage bytes under an immutable object name and update local state."""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        logical_path = _safe_logical_path(logical_path)
        metadata_value = dict(metadata or {})
        _json_bytes(metadata_value)
        live_token = self.token_provider()
        if _HF_TOKEN_BYTES_RE.search(payload) or (
            isinstance(live_token, str)
            and live_token
            and live_token.encode("utf-8") in payload
        ):
            raise ValueError("HF_TOKEN must never be serialized into notebook artifacts")
        digest = sha256_bytes(payload)
        object_path = self.objects / digest
        if object_path.exists():
            if sha256_file(object_path) != digest:
                raise RuntimeError(f"Local content-addressed object is corrupt: {digest}")
        else:
            atomic_write_bytes(object_path, payload)
        record = {
            "logical_path": logical_path,
            "remote_path": (
                f"{self.remote_prefix}/artifacts/sha256/{digest[:2]}/{digest}"
            ),
            "sha256": digest,
            "bytes": len(payload),
            "metadata": metadata_value,
        }
        with self._lock:
            state = self._read_state()
            if state["artifacts"].get(logical_path) == record:
                return record
            state["artifacts"][logical_path] = record
            state["dirty"] = True
            # A changed logical inventory requires a new immutable manifest.
            state["inflight"] = None
            self._write_state(state)
        return record

    def put_file(
        self,
        logical_path: str,
        source: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"Artifact source must be a regular file: {source_path}")
        logical_path = _safe_logical_path(logical_path)
        metadata_value = dict(metadata or {})
        _json_bytes(metadata_value)

        # Hash and inspect in one streaming pass; model checkpoints must never be
        # loaded wholesale into notebook RAM.
        digest_builder = hashlib.sha256()
        live_token = self.token_provider()
        live_token_bytes = (
            live_token.encode("utf-8")
            if isinstance(live_token, str) and live_token
            else None
        )
        overlap = b""
        byte_count = 0
        with source_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                byte_count += len(chunk)
                digest_builder.update(chunk)
                inspection = overlap + chunk
                if _HF_TOKEN_BYTES_RE.search(inspection) or (
                    live_token_bytes is not None and live_token_bytes in inspection
                ):
                    raise ValueError(
                        "HF_TOKEN must never be serialized into notebook artifacts"
                    )
                overlap = inspection[-256:]
        digest = digest_builder.hexdigest()
        object_path = self.objects / digest
        if object_path.exists():
            if (
                object_path.stat().st_size != byte_count
                or sha256_file(object_path) != digest
            ):
                raise RuntimeError(f"Local content-addressed object is corrupt: {digest}")
        else:
            atomic_copy_file(source_path, object_path)
            if (
                object_path.stat().st_size != byte_count
                or sha256_file(object_path) != digest
            ):
                object_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Artifact source changed while it was being staged; retry at a "
                    "checkpoint boundary"
                )
        record = {
            "logical_path": logical_path,
            "remote_path": (
                f"{self.remote_prefix}/artifacts/sha256/{digest[:2]}/{digest}"
            ),
            "sha256": digest,
            "bytes": byte_count,
            "metadata": metadata_value,
        }
        with self._lock:
            state = self._read_state()
            if state["artifacts"].get(logical_path) == record:
                return record
            state["artifacts"][logical_path] = record
            state["dirty"] = True
            state["inflight"] = None
            self._write_state(state)
        return record

    def put_json(
        self,
        logical_path: str,
        value: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.put_bytes(logical_path, _json_bytes(value), metadata=metadata)

    def put_checkpoint(
        self,
        checkpoint_directory: str | Path,
        *,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Stage every file in a locally sealed checkpoint directory.

        The directory must contain ``_COMPLETE.json``. The marker is staged last so
        notebook code cannot accidentally describe an incomplete local checkpoint as
        resumable.
        """

        checkpoint = Path(checkpoint_directory)
        if not checkpoint.is_dir() or checkpoint.is_symlink():
            raise ValueError(f"Checkpoint must be a regular directory: {checkpoint}")
        complete = checkpoint / "_COMPLETE.json"
        if not complete.is_file() or complete.is_symlink():
            raise RuntimeError("Checkpoint is not sealed: missing _COMPLETE.json")
        checkpoint_name = _safe_identifier("checkpoint name", name or checkpoint.name)
        candidates = sorted(
            (path for path in checkpoint.rglob("*") if path.is_file()),
            key=lambda path: (path.name == "_COMPLETE.json", path.as_posix()),
        )
        records: list[dict[str, Any]] = []
        for path in candidates:
            if path.is_symlink():
                raise RuntimeError(f"Checkpoint cannot contain a symbolic link: {path}")
            relative = path.relative_to(checkpoint).as_posix()
            records.append(
                self.put_file(f"checkpoints/{checkpoint_name}/{relative}", path)
            )
        return records

    @property
    def dirty(self) -> bool:
        return bool(self._read_state().get("dirty"))

    def due(self) -> bool:
        state = self._read_state()
        return bool(state.get("dirty")) and (
            self._clock() - float(state.get("last_sync_epoch", 0.0))
            >= self.sync_interval_seconds
        )

    def _prepare_snapshot(self, state: dict[str, Any], reason: str) -> dict[str, Any]:
        existing = state.get("inflight")
        if isinstance(existing, dict):
            return existing
        artifacts = sorted(
            state["artifacts"].values(), key=lambda item: item["logical_path"]
        )
        manifest = {
            "schema_version": self.MANIFEST_SCHEMA,
            "binding": self.binding,
            "created_at": utc_now(),
            "reason": reason,
            "artifacts": artifacts,
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_sha = sha256_bytes(manifest_bytes)
        manifest_remote = f"{self.remote_prefix}/manifests/{manifest_sha}.json"
        manifest_local = self.manifests / f"{manifest_sha}.json"
        atomic_write_bytes(manifest_local, manifest_bytes)
        pointer = {
            "schema_version": self.POINTER_SCHEMA,
            "binding": self.binding,
            "manifest_path": manifest_remote,
            "manifest_sha256": manifest_sha,
        }
        pointer_bytes = _json_bytes(pointer)
        pointer_sha = sha256_bytes(pointer_bytes)
        pointer_local = self.manifests / f"LATEST-{pointer_sha}.json"
        atomic_write_bytes(pointer_local, pointer_bytes)
        snapshot = {
            "manifest_path": manifest_remote,
            "manifest_sha256": manifest_sha,
            "manifest_local": manifest_local.name,
            "pointer_sha256": pointer_sha,
            "pointer_local": pointer_local.name,
            "reason": reason,
            "artifacts": artifacts,
        }
        state["inflight"] = snapshot
        self._write_state(state)
        return snapshot

    def _remote_pointer_matches(
        self, snapshot: Mapping[str, Any], *, revision: str, token: str
    ) -> bool:
        try:
            raw = self._download(
                remote_path=self.pointer_path, revision=revision, token=token
            )
        except Exception as error:
            if _http_status(error) == 404:
                return False
            raise
        try:
            pointer = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(pointer, dict)
            and pointer.get("binding") == self.binding
            and pointer.get("manifest_sha256") == snapshot.get("manifest_sha256")
            and pointer.get("manifest_path") == snapshot.get("manifest_path")
        )

    def _finish_success(
        self, state: dict[str, Any], snapshot: Mapping[str, Any], commit_sha: str
    ) -> None:
        snapshot_artifacts = list(snapshot["artifacts"])
        for record in snapshot_artifacts:
            state["uploaded"][record["sha256"]] = record["remote_path"]
        current = sorted(
            state["artifacts"].values(), key=lambda item: item["logical_path"]
        )
        state["dirty"] = current != snapshot_artifacts
        state["last_sync_epoch"] = self._clock()
        state["last_commit_sha"] = commit_sha
        state["last_manifest_sha256"] = snapshot["manifest_sha256"]
        state["inflight"] = None
        self._write_state(state)

    def _verify_snapshot(
        self, snapshot: Mapping[str, Any], *, commit_sha: str, token: str
    ) -> None:
        pointer_local = self.manifests / str(snapshot["pointer_local"])
        manifest_local = self.manifests / str(snapshot["manifest_local"])
        pointer_remote = self._download(
            remote_path=self.pointer_path, revision=commit_sha, token=token
        )
        if pointer_remote != pointer_local.read_bytes():
            raise RuntimeError("Remote LATEST.json differs from the committed pointer")
        manifest_remote = self._download(
            remote_path=str(snapshot["manifest_path"]),
            revision=commit_sha,
            token=token,
        )
        if manifest_remote != manifest_local.read_bytes():
            raise RuntimeError("Remote immutable manifest failed byte verification")

    def _initialize_remote(self, api: Any) -> None:
        """Create the repository/branch once per kernel, not once per closure."""

        self._call(
            lambda: api.create_repo(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                private=True,
                exist_ok=True,
            )
        )
        try:
            self._call(
                lambda: api.create_branch(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    branch=self.branch,
                    revision=self.base_revision,
                    exist_ok=True,
                )
            )
        except Exception as error:
            missing_base = _http_status(error) == 404 or type(error).__name__ in {
                "RevisionNotFoundError",
                "RepositoryNotFoundError",
            }
            if not missing_base:
                raise
            # An empty HF repository has no branchable main commit. Initialize it
            # once with a deterministic marker, then create the fixed worker branch.
            marker = self.manifests / "REPOSITORY.json"
            atomic_write_bytes(
                marker,
                _json_bytes(
                    {
                        "schema_version": 1,
                        "project": "E2AM-MemRAG",
                        "purpose": "private resumable experiment artifacts",
                    }
                ),
            )
            self._call(
                lambda: api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.base_revision,
                    operations=[self._operation("E2AM_MEMRAG_REPOSITORY.json", marker)],
                    commit_message="Initialize E2AM-MemRAG artifact repository",
                )
            )
            self._call(
                lambda: api.create_branch(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    branch=self.branch,
                    revision=self.base_revision,
                    exist_ok=True,
                )
            )

    def flush(self, *, force: bool = False, reason: str = "periodic") -> dict[str, Any]:
        """Upload one atomic closure, respecting the 20-minute dirty interval."""

        with self._lock:
            state = self._read_state()
            if not state.get("dirty"):
                result = {
                    "status": "CLEAN",
                    "uploaded": 0,
                    "verified": bool(state.get("last_commit_sha")),
                    "commit_sha": state.get("last_commit_sha"),
                }
                self.last_flush = result
                return result
            elapsed = self._clock() - float(state.get("last_sync_epoch", 0.0))
            if not force and elapsed < self.sync_interval_seconds:
                result = {
                    "status": "NOT_DUE",
                    "uploaded": 0,
                    "verified": False,
                    "seconds_until_due": max(0.0, self.sync_interval_seconds - elapsed),
                }
                self.last_flush = result
                return result

            token = self._token()
            api = self._api(token)
            if not self._remote_initialized:
                self._initialize_remote(api)
                self._remote_initialized = True
            last_commit = state.get("last_commit_sha")
            if last_commit:
                # Exclusive worker branches let the last verified commit serve as
                # the optimistic parent. A hidden advance is still detected as a
                # 409 below and audited against the in-flight pointer.
                head = str(last_commit)
            else:
                head = str(
                    self._call(
                        lambda: api.repo_info(
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            revision=self.branch,
                        )
                    ).sha
                )
            snapshot = self._prepare_snapshot(state, reason)
            if last_commit and last_commit != head:
                if self._remote_pointer_matches(snapshot, revision=head, token=token):
                    self._verify_snapshot(snapshot, commit_sha=head, token=token)
                    self._finish_success(state, snapshot, head)
                    result = {
                        "status": "SYNCED_AFTER_LOST_RESPONSE",
                        "uploaded": len(snapshot["artifacts"]),
                        "verified": True,
                        "commit_sha": head,
                    }
                    self.last_flush = result
                    return result
                raise WorkerBranchConflict(
                    "Worker branch advanced outside this local store; do not run two "
                    "sessions on the same worker lane"
                )
            if not last_commit:
                files = set(
                    self._call(
                        lambda: api.list_repo_files(
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            revision=head,
                        )
                    )
                )
                if self.pointer_path in files:
                    if self._remote_pointer_matches(snapshot, revision=head, token=token):
                        self._verify_snapshot(snapshot, commit_sha=head, token=token)
                        self._finish_success(state, snapshot, head)
                        result = {
                            "status": "SYNCED_AFTER_LOST_RESPONSE",
                            "uploaded": len(snapshot["artifacts"]),
                            "verified": True,
                            "commit_sha": head,
                        }
                        self.last_flush = result
                        return result
                    raise RestoreRequired(
                        "This worker branch already has a closure. Call restore_latest() "
                        "before staging new work in a fresh Kaggle session."
                    )

            operations: list[Any] = []
            uploaded = state["uploaded"]
            for record in snapshot["artifacts"]:
                if uploaded.get(record["sha256"]) == record["remote_path"]:
                    continue
                local_object = self.objects / record["sha256"]
                if (
                    not local_object.is_file()
                    or local_object.stat().st_size != record["bytes"]
                    or sha256_file(local_object) != record["sha256"]
                ):
                    raise RuntimeError(
                        f"Local staged object failed verification: {record['sha256']}"
                    )
                operations.append(self._operation(record["remote_path"], local_object))
            operations.extend(
                [
                    self._operation(
                        str(snapshot["manifest_path"]),
                        self.manifests / str(snapshot["manifest_local"]),
                    ),
                    self._operation(
                        self.pointer_path,
                        self.manifests / str(snapshot["pointer_local"]),
                    ),
                ]
            )

            try:
                commit = self._call(
                    lambda: api.create_commit(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        revision=self.branch,
                        parent_commit=head,
                        operations=operations,
                        commit_message=(
                            f"{self.experiment_id}/{self.worker_id}: {reason} "
                            f"{str(snapshot['manifest_sha256'])[:12]}"
                        ),
                    )
                )
                commit_sha = str(
                    getattr(commit, "oid", None)
                    or getattr(commit, "commit_id", None)
                    or self._call(
                        lambda: api.repo_info(
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            revision=self.branch,
                        )
                    ).sha
                )
            except Exception as error:
                # A request may reach the Hub but lose its response. A 409 then means
                # "inspect the new head", not "blindly retry with another manifest".
                if _http_status(error) != 409:
                    raise
                candidate = str(
                    self._call(
                        lambda: api.repo_info(
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            revision=self.branch,
                        )
                    ).sha
                )
                if not self._remote_pointer_matches(
                    snapshot, revision=candidate, token=token
                ):
                    raise WorkerBranchConflict(
                        "Optimistic parent commit failed and the new branch head is not "
                        "this store's in-flight closure"
                    ) from error
                commit_sha = candidate

            self._verify_snapshot(snapshot, commit_sha=commit_sha, token=token)
            self._finish_success(state, snapshot, commit_sha)
            result = {
                "status": "SYNCED",
                "uploaded": len(operations) - 2,
                "verified": True,
                "commit_sha": commit_sha,
                "manifest_sha256": snapshot["manifest_sha256"],
                "reason": reason,
            }
            self.last_flush = result
            return result

    def maybe_flush(self) -> dict[str, Any]:
        return self.flush(force=False, reason="periodic")

    def flush_major(self, stage: str) -> dict[str, Any]:
        stage = _safe_identifier("stage", stage)
        return self.flush(force=True, reason=f"major-{stage}")

    def flush_end(self) -> dict[str, Any]:
        return self.flush(force=True, reason="end")

    def flush_interruption(self) -> dict[str, Any]:
        return self.flush(force=True, reason="interruption")

    @contextmanager
    def interruption_guard(self) -> Iterator["NotebookArtifactStore"]:
        """Force a verified flush at a catchable ``KeyboardInterrupt`` boundary."""

        try:
            yield self
        except KeyboardInterrupt:
            self.flush_interruption()
            raise

    def __enter__(self) -> "NotebookArtifactStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is KeyboardInterrupt or (
            isinstance(exc_type, type) and issubclass(exc_type, KeyboardInterrupt)
        ):
            self.flush_interruption()
        elif exc_type is None:
            self.flush_end()
        else:
            self.flush(force=True, reason="error")
        return False

    def _validate_pointer(self, raw: bytes) -> dict[str, Any]:
        try:
            pointer = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Remote LATEST.json is not valid JSON") from error
        if not isinstance(pointer, dict) or set(pointer) != {
            "schema_version",
            "binding",
            "manifest_path",
            "manifest_sha256",
        }:
            raise RuntimeError("Remote LATEST.json has an unsupported schema")
        if pointer["schema_version"] != self.POINTER_SCHEMA:
            raise RuntimeError("Remote LATEST.json has an unsupported version")
        if pointer["binding"] != self.binding:
            raise RuntimeError("Remote LATEST.json belongs to another worker binding")
        digest = pointer["manifest_sha256"]
        expected_path = f"{self.remote_prefix}/manifests/{digest}.json"
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RuntimeError("Remote LATEST.json has an invalid manifest SHA-256")
        if pointer["manifest_path"] != expected_path:
            raise RuntimeError("Remote LATEST.json manifest path is not content-addressed")
        return pointer

    def _validate_manifest(self, raw: bytes, expected_sha: str) -> list[dict[str, Any]]:
        if sha256_bytes(raw) != expected_sha:
            raise RuntimeError("Remote immutable manifest checksum mismatch")
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Remote immutable manifest is not valid JSON") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != self.MANIFEST_SCHEMA:
            raise RuntimeError("Remote immutable manifest has an unsupported schema")
        if manifest.get("binding") != self.binding:
            raise RuntimeError("Remote immutable manifest belongs to another worker binding")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError("Remote immutable manifest has no artifact inventory")
        validated: list[dict[str, Any]] = []
        logical_seen: set[str] = set()
        for raw_record in artifacts:
            if not isinstance(raw_record, dict) or set(raw_record) != {
                "logical_path",
                "remote_path",
                "sha256",
                "bytes",
                "metadata",
            }:
                raise RuntimeError("Remote manifest has a malformed artifact record")
            logical = _safe_logical_path(raw_record["logical_path"])
            digest = raw_record["sha256"]
            byte_count = raw_record["bytes"]
            if logical in logical_seen:
                raise RuntimeError(f"Remote manifest repeats logical path: {logical}")
            logical_seen.add(logical)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise RuntimeError(f"Remote manifest has an invalid SHA-256: {logical}")
            expected_remote = (
                f"{self.remote_prefix}/artifacts/sha256/{digest[:2]}/{digest}"
            )
            if raw_record["remote_path"] != expected_remote:
                raise RuntimeError(f"Remote artifact path is not content-addressed: {logical}")
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
            ):
                raise RuntimeError(f"Remote manifest has an invalid byte count: {logical}")
            _json_bytes(raw_record["metadata"])
            validated.append(dict(raw_record))
        if validated != sorted(validated, key=lambda item: item["logical_path"]):
            raise RuntimeError("Remote manifest artifact inventory is not canonical")
        return validated

    def restore_selected(
        self,
        logical_paths: Iterable[str],
        destination: str | Path,
        *,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Verify one pinned closure and materialize only requested artifacts.

        Unlike :meth:`restore_latest`, this method deliberately does not adopt the
        remote inventory into local store state and does not populate the local
        content-addressed object catalog.  It is intended for read-only gate and
        receipt checks where downloading every artifact in a closure would be
        wasteful.  The pointer and the complete immutable manifest are still
        validated before any selected payload is downloaded. When ``revision`` is
        supplied, that already-verified commit is used directly without a branch
        head lookup.
        """

        if isinstance(logical_paths, (str, bytes)):
            raise TypeError("logical_paths must be an iterable of artifact paths")
        requested = tuple(_safe_logical_path(path) for path in logical_paths)
        if not requested:
            raise ValueError("At least one logical artifact path is required")
        if len(requested) != len(set(requested)):
            raise ValueError("Selected logical artifact paths must be unique")
        if revision is not None and (
            not isinstance(revision, str)
            or not _COMMIT_SHA_RE.fullmatch(revision.strip().lower())
        ):
            raise ValueError("revision must be an immutable 40-hex commit SHA")

        with self._lock:
            token = self._token()
            api = self._api(token)
            pinned_head = str(revision) if revision is not None else str(
                self._call(
                    lambda: api.repo_info(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        revision=self.branch,
                    )
                ).sha
            )
            pinned_head = pinned_head.strip().lower()
            if not _COMMIT_SHA_RE.fullmatch(pinned_head):
                raise RuntimeError("Hub branch head is not an immutable 40-hex commit SHA")
            self._remote_initialized = True
            pointer = self._validate_pointer(
                self._download(
                    remote_path=self.pointer_path,
                    revision=pinned_head,
                    token=token,
                )
            )
            manifest_raw = self._download(
                remote_path=pointer["manifest_path"],
                revision=pinned_head,
                token=token,
            )
            artifacts = self._validate_manifest(
                manifest_raw, pointer["manifest_sha256"]
            )
            artifacts_by_path = {
                str(record["logical_path"]): record for record in artifacts
            }
            missing = sorted(set(requested) - set(artifacts_by_path))
            if missing:
                raise FileNotFoundError(
                    "Selected artifacts are absent from the pinned manifest: "
                    + ", ".join(missing)
                )

            # Verify all selected bytes before modifying the destination, so a
            # corrupt later object cannot leave an earlier requested path updated.
            payloads: list[tuple[dict[str, Any], bytes]] = []
            for logical_path in requested:
                record = artifacts_by_path[logical_path]
                payload = self._download(
                    remote_path=record["remote_path"],
                    revision=pinned_head,
                    token=token,
                )
                if (
                    len(payload) != record["bytes"]
                    or sha256_bytes(payload) != record["sha256"]
                ):
                    raise RuntimeError(
                        f"Selected artifact failed verification: {logical_path}"
                    )
                payloads.append((record, payload))

            restore_root = Path(destination)
            restore_root.mkdir(parents=True, exist_ok=True)
            receipts: list[dict[str, Any]] = []
            for record, payload in payloads:
                target = _safe_destination(restore_root, record["logical_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(target, payload)
                receipts.append(
                    {
                        "logical_path": record["logical_path"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    }
                )

            return {
                "status": "SELECTED_RESTORE",
                "verified": True,
                "commit_sha": pinned_head,
                "pointer_path": self.pointer_path,
                "manifest_path": pointer["manifest_path"],
                "manifest_sha256": pointer["manifest_sha256"],
                "selected_artifacts": receipts,
                "restored_artifacts": len(receipts),
                "manifest_artifacts": len(artifacts),
                "destination": str(restore_root),
                "full_restore": False,
            }

    def restore_latest(
        self,
        destination: str | Path | None = None,
        *,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Restore a closure, pinning all reads to one branch-head or supplied SHA."""

        if revision is not None and (
            not isinstance(revision, str)
            or not _COMMIT_SHA_RE.fullmatch(revision.strip().lower())
        ):
            raise ValueError("revision must be an immutable 40-hex commit SHA")

        with self._lock:
            token = self._token()
            api = self._api(token)
            pinned_head = str(revision) if revision is not None else str(
                self._call(
                    lambda: api.repo_info(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        revision=self.branch,
                    )
                ).sha
            )
            pinned_head = pinned_head.strip().lower()
            if not _COMMIT_SHA_RE.fullmatch(pinned_head):
                raise RuntimeError("Hub branch head is not an immutable 40-hex commit SHA")
            self._remote_initialized = True
            pointer = self._validate_pointer(
                self._download(
                    remote_path=self.pointer_path,
                    revision=pinned_head,
                    token=token,
                )
            )
            manifest_raw = self._download(
                remote_path=pointer["manifest_path"],
                revision=pinned_head,
                token=token,
            )
            artifacts = self._validate_manifest(
                manifest_raw, pointer["manifest_sha256"]
            )
            restore_root = Path(destination) if destination is not None else None
            if restore_root is not None:
                restore_root.mkdir(parents=True, exist_ok=True)
            artifact_map: dict[str, dict[str, Any]] = {}
            uploaded: dict[str, str] = {}
            for record in artifacts:
                payload = self._download(
                    remote_path=record["remote_path"],
                    revision=pinned_head,
                    token=token,
                )
                if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
                    raise RuntimeError(
                        f"Restored artifact failed verification: {record['logical_path']}"
                    )
                local_object = self.objects / record["sha256"]
                if local_object.exists() and sha256_file(local_object) != record["sha256"]:
                    raise RuntimeError(
                        f"Existing local object is corrupt: {record['sha256']}"
                    )
                if not local_object.exists():
                    atomic_write_bytes(local_object, payload)
                if restore_root is not None:
                    target = _safe_destination(restore_root, record["logical_path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic_copy_file(local_object, target)
                artifact_map[record["logical_path"]] = record
                uploaded[record["sha256"]] = record["remote_path"]

            state = self._read_state()
            state.update(
                {
                    "artifacts": artifact_map,
                    "uploaded": uploaded,
                    "dirty": False,
                    "last_sync_epoch": self._clock(),
                    "last_commit_sha": pinned_head,
                    "last_manifest_sha256": pointer["manifest_sha256"],
                    "inflight": None,
                }
            )
            self._write_state(state)
            return {
                "status": "RESTORED",
                "verified": True,
                "commit_sha": pinned_head,
                "manifest_sha256": pointer["manifest_sha256"],
                "restored_artifacts": len(artifacts),
                "destination": str(restore_root) if restore_root is not None else None,
            }
