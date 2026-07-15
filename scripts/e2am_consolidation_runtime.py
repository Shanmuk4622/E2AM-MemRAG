"""Standalone, non-destructive Hugging Face consolidation runtime.

This file is embedded verbatim into the Stage-10 Kaggle notebook.  It deliberately
lives outside ``src/e2am_memrag`` so publishing a post-experiment convenience copy
cannot change the frozen scientific source-tree hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TOKEN_RE = re.compile(rb"hf_[A-Za-z0-9]{20,}")
MANAGED_README_MARKER = "<!-- E2AM-MEMRAG-CONSOLIDATOR -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe logical path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe logical path: {value!r}")
    return path


def safe_target(root: Path, logical_path: str) -> Path:
    relative = safe_relative_path(logical_path)
    target = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise ValueError(f"Artifact escapes restore root: {logical_path!r}")
    return target


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_zip(source_root: Path, destination: Path) -> dict[str, Any]:
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"Refusing to archive an empty closure: {source_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    inventory: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in files:
                logical = path.relative_to(source_root).as_posix()
                payload = path.read_bytes()
                info = zipfile.ZipInfo(logical, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )
                inventory.append(
                    {
                        "logical_path": logical,
                        "sha256": sha256_bytes(payload),
                        "bytes": len(payload),
                    }
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "files": inventory,
    }


def extract_verified_zip(archive_path: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.extract-{os.getpid()}-{time.time_ns()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise RuntimeError("Archive contains duplicate members")
            for info in archive.infolist():
                logical = safe_relative_path(info.filename)
                if info.is_dir():
                    raise RuntimeError("Consolidated branch archives contain files only")
                target = temporary.joinpath(*logical.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write(target, archive.read(info))
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class RollingHubBudget:
    """A visible rolling limiter with 96/128 weighted operations per hour."""

    def __init__(
        self,
        *,
        capacity: int = 90,
        window_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity > 96 or capacity < 1:
            raise ValueError("Hub capacity must preserve at least 25% of 128 requests/hour")
        self.capacity = int(capacity)
        self.window_seconds = float(window_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self.events: deque[tuple[float, int]] = deque()

    def restore(self, events: Iterable[Iterable[Any]]) -> None:
        now = self.clock()
        restored: list[tuple[float, int]] = []
        for raw in events:
            values = list(raw)
            if len(values) != 2:
                continue
            timestamp, weight = float(values[0]), int(values[1])
            # Older notebook versions charged downloads as weight 2. Normalize
            # those real calls to the current one-call/one-unit accounting. Also
            # discard the old speculative reservations (16/42/70).
            normalized_weight = 1 if weight == 2 else weight
            if (
                0 < normalized_weight <= min(self.capacity, 4)
                and now - timestamp < self.window_seconds
            ):
                restored.append((timestamp, normalized_weight))
        self.events = deque(sorted(restored))
        self._prune()

    def snapshot(self) -> list[list[float | int]]:
        self._prune()
        return [[timestamp, weight] for timestamp, weight in self.events]

    def _prune(self) -> None:
        threshold = self.clock() - self.window_seconds
        while self.events and self.events[0][0] <= threshold:
            self.events.popleft()

    @property
    def used(self) -> int:
        self._prune()
        return sum(weight for _, weight in self.events)

    def acquire(self, weight: int, *, reason: str) -> None:
        if not 0 < weight <= self.capacity:
            raise ValueError("Invalid Hub budget weight")
        last_notice = 0.0
        while True:
            self._prune()
            if self.used + weight <= self.capacity:
                self.events.append((self.clock(), weight))
                return
            oldest = self.events[0][0]
            wait_seconds = max(1.0, oldest + self.window_seconds - self.clock() + 1.0)
            if self.clock() - last_notice >= 60.0:
                print(
                    "HUB_BUDGET_WAIT",
                    {
                        "reason": reason,
                        "used_weight": self.used,
                        "capacity": self.capacity,
                        "remaining_seconds": round(wait_seconds, 1),
                    },
                    flush=True,
                )
                last_notice = self.clock()
            self.sleeper(min(30.0, wait_seconds))


def http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class Consolidator:
    def __init__(self, config: Mapping[str, Any], *, hf_token: str) -> None:
        if not isinstance(hf_token, str) or not hf_token.strip():
            raise RuntimeError("HF_TOKEN is required for publishing")
        self.config = dict(config)
        self.hf_token = hf_token.strip()
        self.repo_id = str(self.config["repo_id"])
        self.repo_type = "dataset"
        self.experiment_id = str(self.config["experiment_id"])
        self.destination_branch = str(self.config["destination_branch"])
        self.remote_root = str(self.config["remote_root"])
        self.dirty_sync_target_seconds = int(
            self.config.get("dirty_sync_target_seconds", 1200)
        )
        if self.dirty_sync_target_seconds != 1200:
            raise RuntimeError("Dirty-state sync target must remain frozen at 1,200 seconds")
        self.work_root = Path(self.config["work_root"])
        self.cache_root = self.work_root / "hub-cache"
        self.source_root = self.work_root / "source"
        self.bundle_root = self.work_root / "bundles"
        self.paper_root = self.work_root / "paper"
        self.state_root = self.work_root / "state"
        for root in (
            self.work_root,
            self.cache_root,
            self.source_root,
            self.bundle_root,
            self.paper_root,
            self.state_root,
        ):
            root.mkdir(parents=True, exist_ok=True)

        self.source_lock = [dict(item) for item in self.config["source_release_lock"]]
        self.source_lock_sha256 = sha256_bytes(canonical_json(self.source_lock))
        self.expected_branches = {item["branch"]: item for item in self.source_lock}
        if len(self.expected_branches) != len(self.source_lock):
            raise RuntimeError("Source release lock repeats a branch")
        self._validate_source_lock()
        self.budget = RollingHubBudget(capacity=int(self.config.get("hub_capacity", 96)))
        self.api = None
        self.hf_hub_download = None
        self.CommitOperationAdd = None
        self.progress: dict[str, Any] | None = None
        self.destination_head: str | None = None
        self.last_verified_progress_bytes: bytes | None = None

    def _validate_source_lock(self) -> None:
        total_records = 0
        total_bytes = 0
        for item in self.source_lock:
            required = {
                "stage_id",
                "owner",
                "branch",
                "commit_sha",
                "manifest_sha256",
                "artifact_records",
                "artifact_bytes",
            }
            if set(item) != required:
                raise RuntimeError("Source release lock schema is invalid")
            stage = str(item["stage_id"])
            owner = str(item["owner"])
            expected_branch = (
                f"stage-{self.experiment_id}-{stage}-stage-{stage}-{owner}"
            )
            if item["branch"] != expected_branch:
                raise RuntimeError(f"Source branch identity is invalid: {item['branch']}")
            if not COMMIT_RE.fullmatch(str(item["commit_sha"])):
                raise RuntimeError("Source release lock has a non-immutable commit")
            if not SHA256_RE.fullmatch(str(item["manifest_sha256"])):
                raise RuntimeError("Source release lock has an invalid manifest digest")
            total_records += int(item["artifact_records"])
            total_bytes += int(item["artifact_bytes"])
        if total_records != int(self.config["expected_artifact_records"]):
            raise RuntimeError("Frozen artifact-record total is inconsistent")
        if total_bytes != int(self.config["expected_artifact_bytes"]):
            raise RuntimeError("Frozen artifact-byte total is inconsistent")

    def _load_hub(self) -> None:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

        self.api = HfApi(token=self.hf_token)
        self.hf_hub_download = hf_hub_download
        self.CommitOperationAdd = CommitOperationAdd

    def _call(
        self,
        operation: Callable[[], Any],
        *,
        weight: int,
        reason: str,
        public_download: bool = False,
    ) -> Any:
        for attempt in range(5):
            # Each retry is a real Hub operation and therefore consumes budget.
            self.budget.acquire(weight, reason=reason)
            try:
                return operation()
            except Exception as error:
                status = http_status(error)
                # Public Hub downloads can occasionally receive an expired or invalid
                # signed object-store URL.  Preserve that raw 403 so _download can
                # invalidate the cached URL and retry it.  Authenticated API failures
                # remain immediate, non-retryable safe stops.
                if status == 403 and public_download:
                    raise
                if status in {401, 403}:
                    raise RuntimeError(
                        "HUB_AUTHENTICATION_STOP: correct the Kaggle HF_TOKEN; no retry "
                        "or source-branch mutation was attempted"
                    ) from error
                retryable = status == 429 or (
                    status is not None and 500 <= status < 600
                )
                if not retryable or attempt == 4:
                    raise
                delay = retry_after(error)
                if delay is None:
                    delay = min(60.0, 2.0**attempt)
                print(
                    "HUB_TRANSIENT_RETRY",
                    {"reason": reason, "attempt": attempt + 1, "wait_seconds": delay},
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state")

    def _download(
        self,
        *,
        filename: str,
        revision: str,
        public_read: bool = True,
    ) -> bytes:
        safe_relative_path(filename)
        # Authenticate public reads too; anonymous Hub traffic has a lower rate
        # limit and caused the warning shown in the Kaggle log.
        token: str | bool = self.hf_token

        def perform(force_download: bool = False) -> str:
            return self.hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=revision,
                filename=filename,
                token=token,
                cache_dir=str(self.cache_root),
                force_download=force_download,
            )

        try:
            local = self._call(
                perform,
                weight=1,
                reason=f"download:{filename}",
                public_download=public_read,
            )
        except Exception as error:
            text = str(error).lower()
            signed_url_failure = http_status(error) == 403 and any(
                marker in text
                for marker in ("signature", "expired", "xet", "accessdenied")
            )
            if not signed_url_failure or not public_read:
                raise
            print(
                "PUBLIC_SIGNED_URL_REFRESH",
                {"filename": filename, "revision": revision},
                flush=True,
            )
            local = self._call(
                lambda: perform(True),
                weight=1,
                reason=f"signed-url-refresh:{filename}",
                public_download=True,
            )
        return Path(local).read_bytes()

    def _try_download(
        self, *, filename: str, revision: str, public_read: bool = True
    ) -> bytes | None:
        try:
            return self._download(
                filename=filename,
                revision=revision,
                public_read=public_read,
            )
        except Exception as error:
            if http_status(error) == 404 or type(error).__name__ == "EntryNotFoundError":
                return None
            raise

    def _repo_head(self, revision: str) -> str:
        info = self._call(
            lambda: self.api.repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                revision=revision,
            ),
            weight=1,
            reason=f"repo-head:{revision}",
        )
        commit = str(info.sha).strip().lower()
        if not COMMIT_RE.fullmatch(commit):
            raise RuntimeError(f"Hub revision is not immutable: {revision}")
        return commit

    def _list_refs(self) -> tuple[dict[str, str], list[str]]:
        refs = self._call(
            lambda: self.api.list_repo_refs(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            ),
            weight=1,
            reason="list-repository-refs",
        )
        branches: dict[str, str] = {}
        for branch in refs.branches:
            name = str(branch.name)
            commit = str(branch.target_commit).strip().lower()
            if not COMMIT_RE.fullmatch(commit):
                raise RuntimeError(f"Non-immutable branch head: {name}")
            branches[name] = commit
        missing = sorted(set(self.expected_branches) - set(branches))
        if missing:
            raise RuntimeError("SOURCE_BRANCHES_MISSING: " + ", ".join(missing))
        legacy = sorted(
            name
            for name in branches
            if name.startswith("stage-e2am-memrag-")
            and name not in self.expected_branches
        )
        advanced = {
            name: {"locked": self.expected_branches[name]["commit_sha"], "head": branches[name]}
            for name in self.expected_branches
            if branches[name] != self.expected_branches[name]["commit_sha"]
        }
        if advanced:
            print("SOURCE_HEADS_ADVANCED_PINNED_COMMITS_PRESERVED", advanced, flush=True)
        print(
            "SOURCE_BRANCH_AUDIT",
            {
                "required_v3r1": len(self.expected_branches),
                "excluded_legacy": legacy,
                "advanced_but_pinned": len(advanced),
            },
            flush=True,
        )
        return branches, legacy

    def _verify_release_pointer(self) -> dict[str, Any]:
        release_path = f"experiments/{self.experiment_id}/RELEASE.json"
        payload = self._download(filename=release_path, revision="main")
        release = json.loads(payload)
        expected = dict(self.config["release_pointer_lock"])
        for key, value in expected.items():
            if release.get(key) != value:
                raise RuntimeError(f"Main RELEASE.json disagrees on {key}")
        print(
            "MAIN_RELEASE_POINTER_VERIFIED",
            {
                "stage_commit_sha": release["stage_commit_sha"],
                "success_gate_sha256": release["success_gate_sha256"],
            },
            flush=True,
        )
        return release

    def _new_progress(self, *, legacy_branches: list[str]) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "source_lock_sha256": self.source_lock_sha256,
            "destination_branch": self.destination_branch,
            "dirty_sync_target_seconds": self.dirty_sync_target_seconds,
            "status": "RUNNING",
            "started_at": now,
            "updated_at": now,
            "completed": {},
            "excluded_legacy_branches": legacy_branches,
            "hub_budget_events": self.budget.snapshot(),
        }

    def _validate_progress(self, progress: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(progress)
        if (
            value.get("schema_version") != 1
            or value.get("experiment_id") != self.experiment_id
            or value.get("source_lock_sha256") != self.source_lock_sha256
            or value.get("destination_branch") != self.destination_branch
            or value.get("dirty_sync_target_seconds") != self.dirty_sync_target_seconds
            or value.get("status") not in {"RUNNING", "COMPLETE"}
            or not isinstance(value.get("completed"), dict)
        ):
            raise RuntimeError("Destination PROGRESS.json belongs to another consolidation")
        unknown = set(value["completed"]) - set(self.expected_branches)
        if unknown:
            raise RuntimeError("Destination progress contains unknown source branches")
        for branch, entry in value["completed"].items():
            lock = self.expected_branches[branch]
            if (
                entry.get("source_commit_sha") != lock["commit_sha"]
                or entry.get("source_manifest_sha256") != lock["manifest_sha256"]
                or entry.get("artifact_records") != lock["artifact_records"]
                or entry.get("artifact_bytes") != lock["artifact_bytes"]
                or not SHA256_RE.fullmatch(str(entry.get("archive_sha256", "")))
                or not isinstance(entry.get("artifact_inventory"), list)
            ):
                raise RuntimeError(f"Completed branch receipt is invalid: {branch}")
        return value

    def _local_file(self, name: str, payload: bytes) -> Path:
        path = self.state_root / name
        if TOKEN_RE.search(payload) or self.hf_token.encode("utf-8") in payload:
            raise RuntimeError("Secret-like value rejected from publication metadata")
        atomic_write(path, payload)
        return path

    def _commit_verified(
        self,
        *,
        revision: str,
        parent_commit: str,
        files: Mapping[str, Path],
        message: str,
        allow_existing_equal: bool = True,
    ) -> str:
        if not files:
            return parent_commit
        for remote_path, local_path in files.items():
            safe_relative_path(remote_path)
            if not local_path.is_file() or local_path.is_symlink():
                raise RuntimeError(f"Publication source is not a regular file: {local_path}")
        current_head = self._repo_head(revision)
        if allow_existing_equal:
            all_equal = True
            for remote_path, local_path in files.items():
                remote = self._try_download(
                    filename=remote_path,
                    revision=current_head,
                    public_read=True,
                )
                if remote != local_path.read_bytes():
                    all_equal = False
                    break
            if all_equal:
                return current_head
        if current_head != parent_commit:
            all_equal = True
            for remote_path, local_path in files.items():
                remote = self._try_download(
                    filename=remote_path,
                    revision=current_head,
                    public_read=True,
                )
                all_equal = all_equal and remote == local_path.read_bytes()
            if allow_existing_equal and all_equal:
                return current_head
            raise RuntimeError(
                f"SECOND_WRITER_STOP: {revision} advanced from {parent_commit} to {current_head}"
            )

        operations = [
            self.CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for remote, local in sorted(files.items())
        ]
        try:
            commit = self._call(
                lambda: self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=revision,
                    parent_commit=parent_commit,
                    operations=operations,
                    commit_message=message,
                ),
                weight=4,
                reason=f"create-commit:{revision}",
            )
            commit_sha = str(
                getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
            ).strip().lower()
        except Exception as error:
            if http_status(error) != 409:
                raise
            commit_sha = self._repo_head(revision)
        if not COMMIT_RE.fullmatch(commit_sha):
            commit_sha = self._repo_head(revision)

        for remote_path, local_path in files.items():
            remote = self._download(
                filename=remote_path,
                revision=commit_sha,
                public_read=True,
            )
            if remote != local_path.read_bytes():
                raise RuntimeError(f"REMOTE_UPLOAD_VERIFICATION_FAILED: {remote_path}")
        return commit_sha

    def _initialize_destination(self, *, branches: Mapping[str, str], legacy: list[str]) -> None:
        main_head = branches["main"]
        created = self.destination_branch not in branches
        if created:
            self._call(
                lambda: self.api.create_branch(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    branch=self.destination_branch,
                    revision=main_head,
                    exist_ok=True,
                ),
                weight=2,
                reason="create-consolidation-branch",
            )
        self.destination_head = self._repo_head(self.destination_branch)
        progress_path = f"{self.remote_root}/PROGRESS.json"
        raw = self._try_download(
            filename=progress_path,
            revision=self.destination_head,
            public_read=True,
        )
        if raw is None:
            repo_files = self._call(
                lambda: self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=self.destination_head,
                ),
                weight=1,
                reason="inspect-consolidation-branch",
            )
            managed = [path for path in repo_files if path.startswith(self.remote_root + "/")]
            if managed:
                raise RuntimeError(
                    "Destination branch has managed files but no PROGRESS.json; refusing overwrite"
                )
            self.progress = self._new_progress(legacy_branches=legacy)
            progress_bytes = canonical_json(self.progress)
            progress_local = self._local_file("PROGRESS.json", progress_bytes)
            readme_local = self._local_file(
                "BRANCH_README.md", self._branch_readme(status="RUNNING").encode("utf-8")
            )
            self.destination_head = self._commit_verified(
                revision=self.destination_branch,
                parent_commit=self.destination_head,
                files={
                    progress_path: progress_local,
                    f"{self.remote_root}/README.md": readme_local,
                },
                message=f"Initialize {self.experiment_id} non-destructive consolidation",
            )
            self.last_verified_progress_bytes = progress_bytes
        else:
            self.progress = self._validate_progress(json.loads(raw))
            self.budget.restore(self.progress.get("hub_budget_events", []))
            self.last_verified_progress_bytes = raw
            print(
                "CONSOLIDATION_RESUME",
                {
                    "status": self.progress["status"],
                    "completed_branches": len(self.progress["completed"]),
                    "destination_commit": self.destination_head,
                },
                flush=True,
            )

    def _validate_pointer_and_manifest(
        self, lock: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        stage = str(lock["stage_id"])
        owner = str(lock["owner"])
        commit = str(lock["commit_sha"])
        prefix = f"experiments/{self.experiment_id}/stages/{stage}/{owner}"
        pointer_path = f"{prefix}/LATEST.json"
        pointer_raw = self._download(filename=pointer_path, revision=commit)
        pointer = json.loads(pointer_raw)
        binding = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "experiment_id": self.experiment_id,
            "worker_id": f"stage-{stage}-{owner}",
            "branch": lock["branch"],
            "base_revision": "main",
            "remote_prefix": prefix,
        }
        expected_manifest_path = (
            f"{prefix}/manifests/{lock['manifest_sha256']}.json"
        )
        if (
            set(pointer)
            != {"schema_version", "binding", "manifest_path", "manifest_sha256"}
            or pointer.get("schema_version") != 1
            or pointer.get("binding") != binding
            or pointer.get("manifest_path") != expected_manifest_path
            or pointer.get("manifest_sha256") != lock["manifest_sha256"]
        ):
            raise RuntimeError(f"Invalid source pointer: {lock['branch']}")
        manifest_raw = self._download(
            filename=pointer["manifest_path"], revision=commit
        )
        if sha256_bytes(manifest_raw) != lock["manifest_sha256"]:
            raise RuntimeError(f"Source manifest checksum mismatch: {lock['branch']}")
        manifest = json.loads(manifest_raw)
        records = manifest.get("artifacts")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("binding") != binding
            or not isinstance(records, list)
            or len(records) != lock["artifact_records"]
        ):
            raise RuntimeError(f"Source manifest identity mismatch: {lock['branch']}")
        validated: list[dict[str, Any]] = []
        logical_seen: set[str] = set()
        for record in records:
            logical = str(record.get("logical_path", ""))
            digest = str(record.get("sha256", ""))
            byte_count = record.get("bytes")
            safe_relative_path(logical)
            expected_remote = f"{prefix}/artifacts/sha256/{digest[:2]}/{digest}"
            if (
                set(record)
                != {"logical_path", "remote_path", "sha256", "bytes", "metadata"}
                or logical in logical_seen
                or not SHA256_RE.fullmatch(digest)
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or record.get("remote_path") != expected_remote
            ):
                raise RuntimeError(f"Malformed source artifact record: {logical}")
            logical_seen.add(logical)
            canonical_json(record.get("metadata"))
            validated.append(dict(record))
        if validated != sorted(validated, key=lambda item: item["logical_path"]):
            raise RuntimeError("Source manifest artifact order is not canonical")
        if sum(record["bytes"] for record in validated) != lock["artifact_bytes"]:
            raise RuntimeError(f"Source artifact-byte total mismatch: {lock['branch']}")
        return pointer, validated

    def _restore_source(self, lock: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
        branch = str(lock["branch"])
        stage = str(lock["stage_id"])
        owner = str(lock["owner"])
        destination = self.source_root / stage / owner
        temporary = destination.with_name(
            f".{destination.name}.restore-{os.getpid()}-{time.time_ns()}"
        )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            _, records = self._validate_pointer_and_manifest(lock)
            for index, record in enumerate(records, start=1):
                payload = self._download(
                    filename=record["remote_path"],
                    revision=lock["commit_sha"],
                )
                if len(payload) != record["bytes"] or sha256_bytes(payload) != record["sha256"]:
                    raise RuntimeError(
                        f"SOURCE_ARTIFACT_CHECKSUM_FAILED: {branch}/{record['logical_path']}"
                    )
                target = safe_target(temporary, record["logical_path"])
                atomic_write(target, payload)
                if index % 10 == 0 or index == len(records):
                    print(
                        "SOURCE_RESTORE_PROGRESS",
                        {"branch": branch, "verified": index, "total": len(records)},
                        flush=True,
                    )
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            return destination, records
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _receipt_bytes(
        self,
        lock: Mapping[str, Any],
        records: list[dict[str, Any]],
        archive: Mapping[str, Any],
    ) -> bytes:
        return canonical_json(
            {
                "schema_version": 1,
                "experiment_id": self.experiment_id,
                "source_branch": lock["branch"],
                "source_commit_sha": lock["commit_sha"],
                "source_manifest_sha256": lock["manifest_sha256"],
                "artifact_records": len(records),
                "artifact_bytes": sum(record["bytes"] for record in records),
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
                "artifact_inventory": [
                    {
                        "logical_path": record["logical_path"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    }
                    for record in records
                ],
                "verified_at": utc_now(),
                "source_branches_modified": False,
            }
        )

    def _verify_completed_entry(self, branch: str, entry: Mapping[str, Any]) -> None:
        archive = self._download(
            filename=entry["archive_path"],
            revision=self.destination_head,
        )
        receipt = self._download(
            filename=entry["receipt_path"],
            revision=self.destination_head,
        )
        if sha256_bytes(archive) != entry["archive_sha256"]:
            raise RuntimeError(f"Completed archive failed resume verification: {branch}")
        if sha256_bytes(receipt) != entry["receipt_sha256"]:
            raise RuntimeError(f"Completed receipt failed resume verification: {branch}")

    def _process_branch(self, lock: Mapping[str, Any]) -> None:
        branch = str(lock["branch"])
        existing = self.progress["completed"].get(branch)
        if existing is not None:
            self._verify_completed_entry(branch, existing)
            print("SOURCE_BRANCH_ALREADY_CONSOLIDATED", {"branch": branch}, flush=True)
            return

        print(
            "SOURCE_BRANCH_CONSOLIDATION_START",
            {
                "branch": branch,
                "commit": lock["commit_sha"],
                "artifact_records": lock["artifact_records"],
                "artifact_mib": round(lock["artifact_bytes"] / (1024**2), 3),
            },
            flush=True,
        )
        restored, records = self._restore_source(lock)
        archive_path = self.bundle_root / f"{branch}-{lock['commit_sha']}.zip"
        archive = deterministic_zip(restored, archive_path)
        if (
            len(archive["files"]) != lock["artifact_records"]
            or sum(item["bytes"] for item in archive["files"]) != lock["artifact_bytes"]
        ):
            raise RuntimeError(f"Deterministic archive coverage mismatch: {branch}")
        receipt_bytes = self._receipt_bytes(lock, records, archive)
        receipt_local = self._local_file(
            f"{branch}.receipt.json", receipt_bytes
        )
        stage = str(lock["stage_id"])
        owner = str(lock["owner"])
        archive_remote = (
            f"{self.remote_root}/source-branches/{stage}/{owner}/"
            f"{lock['commit_sha']}.zip"
        )
        receipt_remote = (
            f"{self.remote_root}/source-branches/{stage}/{owner}/"
            f"{lock['commit_sha']}.receipt.json"
        )
        entry = {
            "stage_id": stage,
            "owner": owner,
            "source_commit_sha": lock["commit_sha"],
            "source_manifest_sha256": lock["manifest_sha256"],
            "artifact_records": lock["artifact_records"],
            "artifact_bytes": lock["artifact_bytes"],
            "artifact_inventory": [
                {
                    "logical_path": record["logical_path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in records
            ],
            "archive_path": archive_remote,
            "archive_sha256": archive["sha256"],
            "archive_bytes": archive["bytes"],
            "receipt_path": receipt_remote,
            "receipt_sha256": sha256_bytes(receipt_bytes),
        }
        candidate_progress = json.loads(canonical_json(self.progress))
        candidate_progress["completed"][branch] = entry
        candidate_progress["updated_at"] = utc_now()
        candidate_progress["hub_budget_events"] = self.budget.snapshot()
        progress_bytes = canonical_json(candidate_progress)
        progress_local = self._local_file("PROGRESS.json", progress_bytes)
        self.destination_head = self._commit_verified(
            revision=self.destination_branch,
            parent_commit=self.destination_head,
            files={
                archive_remote: archive_path,
                receipt_remote: receipt_local,
                f"{self.remote_root}/PROGRESS.json": progress_local,
            },
            message=(
                f"Consolidate {self.experiment_id} source branch {stage}/{owner}"
            ),
        )
        self.progress = candidate_progress
        self.last_verified_progress_bytes = progress_bytes
        print(
            "SOURCE_BRANCH_CONSOLIDATION_VERIFIED",
            {
                "branch": branch,
                "destination_commit": self.destination_head,
                "archive_sha256": archive["sha256"],
            },
            flush=True,
        )
        # Original immutable branch plus the freshly verified consolidated archive
        # are now two remote copies, so local materialization is no longer the sole copy.
        shutil.rmtree(restored)
        archive_path.unlink(missing_ok=True)

    def _paper_files_from_stage09(self) -> dict[str, Path]:
        stage09_branch = f"stage-{self.experiment_id}-09-stage-09-coordinator"
        entry = self.progress["completed"][stage09_branch]
        archive_bytes = self._download(
            filename=entry["archive_path"], revision=self.destination_head
        )
        if sha256_bytes(archive_bytes) != entry["archive_sha256"]:
            raise RuntimeError("Stage-09 consolidated archive failed verification")
        archive_local = self.bundle_root / "stage09-paper-source.zip"
        atomic_write(archive_local, archive_bytes)
        extracted = self.work_root / "stage09-paper-source"
        extract_verified_zip(archive_local, extracted)
        expected = [
            "HYPOTHESIS_RESULT.json",
            "RELEASE_CANDIDATE.json",
            "_SUCCESS.json",
            "release/clean_traces.jsonl",
            "release/experiment_summary.json",
            "release/mechanism_analysis.json",
            "release/model_transfer_panel.json",
            "release/robustness_analysis.json",
            "release/robustness_traces.jsonl",
            "release/route_cards.json",
            "release_manifest.json",
        ]
        result: dict[str, Path] = {}
        if self.paper_root.exists():
            shutil.rmtree(self.paper_root)
        self.paper_root.mkdir(parents=True)
        inventory = {item["logical_path"]: item for item in entry["artifact_inventory"]}
        for logical in expected:
            source = safe_target(extracted, logical)
            record = inventory.get(logical)
            if (
                record is None
                or not source.is_file()
                or source.stat().st_size != record["bytes"]
                or sha256_file(source) != record["sha256"]
            ):
                raise RuntimeError(f"Paper artifact is missing or corrupt: {logical}")
            target = safe_target(self.paper_root, logical)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            result[logical] = target
        return result

    def _branch_readme(self, *, status: str) -> str:
        return f"""# E2AM-MemRAG verified consolidated release

{MANAGED_README_MARKER}

Status: **{status}**

This branch is a post-experiment, non-destructive convenience copy for
`{self.experiment_id}`. Each source worker closure is preserved as a deterministic
ZIP plus a checksum receipt. Original stage branches are never deleted, renamed,
rewritten, or force-pushed. Legacy v2/v3 branches are recorded but excluded from
the v3r1 scientific release.

Paper-facing Stage-09 artifacts are available individually under `paper/`.
`_SUCCESS.json` means the frozen experiment and fresh-restore audit completed;
the predeclared hypothesis outcome is stored separately in
`paper/HYPOTHESIS_RESULT.json`.
"""

    def _paper_readme(self, hypothesis_pass: bool) -> str:
        outcome = "passed" if hypothesis_pass else "did not pass"
        return f"""# Paper-facing E2AM-MemRAG results

{MANAGED_README_MARKER}

The complete frozen experiment finished and passed its closure/fresh-restore audit.
The separately predeclared confirmatory hypothesis **{outcome}**. Do not describe
experiment completion as hypothesis success.

Key files:

- `experiment_summary.json`: primary policy and route summaries.
- `mechanism_analysis.json`: retrieval and memory mechanism contrasts.
- `model_transfer_panel.json`: five-model direct/grounded panel and Pareto data.
- `robustness_analysis.json`: clean-versus-corrupted degradation analyses.
- `route_cards.json`: frozen route definitions and provenance.
- `clean_traces.jsonl` and `robustness_traces.jsonl`: row-level release traces.
- `HYPOTHESIS_RESULT.json`: confirmatory pass/fail outcome.
- `_SUCCESS.json`: experiment completion gate, not a positive-result claim.
"""

    def _dataset_readme(self, hypothesis_pass: bool) -> str:
        outcome = "passed" if hypothesis_pass else "did not pass"
        return f"""---
pretty_name: E2AM-MemRAG Traces
tags:
- rag
- memory
- green-ai
- energy-efficiency
- reproducibility
---

# E2AM-MemRAG Traces

{MANAGED_README_MARKER}

The frozen `{self.experiment_id}` experiment is complete and remotely verified.
The separately predeclared confirmatory hypothesis **{outcome}**; completion is
not presented as a positive result.

Human-readable paper artifacts are under
`experiments/{self.experiment_id}/paper/`. The complete 23-branch verified copy is
on branch `{self.destination_branch}`. Original worker branches and legacy
branches remain untouched.
"""

    def _finalize_destination(self, release: Mapping[str, Any]) -> dict[str, Any]:
        if set(self.progress["completed"]) != set(self.expected_branches):
            missing = sorted(set(self.expected_branches) - set(self.progress["completed"]))
            raise RuntimeError("Cannot finalize incomplete consolidation: " + ", ".join(missing))
        paper_files = self._paper_files_from_stage09()
        hypothesis = json.loads(paper_files["HYPOTHESIS_RESULT.json"].read_bytes())
        hypothesis_pass = bool(hypothesis.get("hypothesis_pass", False))
        manifest = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "source_lock_sha256": self.source_lock_sha256,
            "source_release_pointer": dict(release),
            "source_branches": self.progress["completed"],
            "source_branch_count": len(self.progress["completed"]),
            "source_artifact_records": sum(
                entry["artifact_records"] for entry in self.progress["completed"].values()
            ),
            "source_artifact_bytes": sum(
                entry["artifact_bytes"] for entry in self.progress["completed"].values()
            ),
            "excluded_legacy_branches": self.progress["excluded_legacy_branches"],
            "source_branches_modified": False,
            "hypothesis_pass": hypothesis_pass,
            "completion_is_independent_of_hypothesis": True,
        }
        manifest_bytes = canonical_json(manifest)
        success = {
            "schema_version": 1,
            "status": "COMPLETE",
            "experiment_id": self.experiment_id,
            "source_branch_count": manifest["source_branch_count"],
            "source_artifact_records": manifest["source_artifact_records"],
            "source_artifact_bytes": manifest["source_artifact_bytes"],
            "consolidation_manifest_sha256": sha256_bytes(manifest_bytes),
            "hypothesis_pass": hypothesis_pass,
            "completion_is_independent_of_hypothesis": True,
            "source_branches_modified": False,
        }
        unhashed = dict(success)
        success["gate_sha256"] = sha256_bytes(canonical_json(unhashed))
        success_bytes = canonical_json(success)
        manifest_local = self._local_file("CONSOLIDATION_MANIFEST.json", manifest_bytes)
        success_local = self._local_file("CONSOLIDATION_SUCCESS.json", success_bytes)
        branch_readme = self._local_file(
            "FINAL_BRANCH_README.md", self._branch_readme(status="COMPLETE").encode("utf-8")
        )
        paper_readme = self._local_file(
            "PAPER_RELEASE_README.md", self._paper_readme(hypothesis_pass).encode("utf-8")
        )
        candidate_progress = json.loads(canonical_json(self.progress))
        if candidate_progress["status"] != "COMPLETE":
            candidate_progress["status"] = "COMPLETE"
            candidate_progress["updated_at"] = utc_now()
            candidate_progress["hub_budget_events"] = self.budget.snapshot()
        progress_bytes = canonical_json(candidate_progress)
        progress_local = self._local_file("PROGRESS.json", progress_bytes)
        files: dict[str, Path] = {
            f"{self.remote_root}/CONSOLIDATION_MANIFEST.json": manifest_local,
            f"{self.remote_root}/_SUCCESS.json": success_local,
            f"{self.remote_root}/README.md": branch_readme,
            f"{self.remote_root}/paper/README.md": paper_readme,
            f"{self.remote_root}/PROGRESS.json": progress_local,
        }
        for logical, path in paper_files.items():
            files[f"{self.remote_root}/paper/{logical}"] = path
        self.destination_head = self._commit_verified(
            revision=self.destination_branch,
            parent_commit=self.destination_head,
            files=files,
            message=f"Finalize {self.experiment_id} verified consolidated release",
        )
        self.progress = candidate_progress
        self.last_verified_progress_bytes = progress_bytes
        return {
            "manifest": manifest,
            "manifest_bytes": manifest_bytes,
            "success": success,
            "success_bytes": success_bytes,
            "paper_files": paper_files,
            "paper_readme": paper_readme,
            "hypothesis_pass": hypothesis_pass,
        }

    def _publish_main(self, final: Mapping[str, Any]) -> str:
        main_head = self._repo_head("main")
        base = f"experiments/{self.experiment_id}"
        pointer = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "consolidation_branch": self.destination_branch,
            "consolidation_commit_sha": self.destination_head,
            "consolidation_root": self.remote_root,
            "consolidation_manifest_sha256": sha256_bytes(final["manifest_bytes"]),
            "consolidation_success_sha256": sha256_bytes(final["success_bytes"]),
            "paper_prefix": f"{base}/paper",
            "source_branch_count": final["manifest"]["source_branch_count"],
            "source_artifact_records": final["manifest"]["source_artifact_records"],
            "source_artifact_bytes": final["manifest"]["source_artifact_bytes"],
            "source_branches_modified": False,
        }
        pointer_local = self._local_file(
            "CONSOLIDATED_RELEASE.json", canonical_json(pointer)
        )
        manifest_local = self._local_file(
            "MAIN_CONSOLIDATION_MANIFEST.json", final["manifest_bytes"]
        )
        files: dict[str, Path] = {
            f"{base}/CONSOLIDATED_RELEASE.json": pointer_local,
            f"{base}/CONSOLIDATION_MANIFEST.json": manifest_local,
            f"{base}/PAPER_RELEASE_README.md": final["paper_readme"],
        }
        for logical, path in final["paper_files"].items():
            files[f"{base}/paper/{logical}"] = path
        if f"{base}/RELEASE.json" in files:
            raise RuntimeError("Frozen RELEASE.json must never be overwritten")

        existing_files = set(
            self._call(
                lambda: self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    revision=main_head,
                ),
                weight=1,
                reason="inspect-main-before-publication",
            )
        )
        root_readme = self._local_file(
            "DATASET_README.md",
            self._dataset_readme(final["hypothesis_pass"]).encode("utf-8"),
        )
        if "README.md" not in existing_files:
            files["README.md"] = root_readme

        changed: dict[str, Path] = {}
        for remote_path, local_path in files.items():
            if remote_path not in existing_files:
                changed[remote_path] = local_path
                continue
            remote = self._download(filename=remote_path, revision=main_head)
            if remote != local_path.read_bytes():
                raise RuntimeError(
                    "MAIN_PUBLICATION_CONFLICT: existing file differs and will not be overwritten: "
                    + remote_path
                )
        if changed:
            main_head = self._commit_verified(
                revision="main",
                parent_commit=main_head,
                files=changed,
                message=f"Publish visible {self.experiment_id} paper and consolidation release",
            )
        for remote_path, local_path in files.items():
            remote = self._download(filename=remote_path, revision=main_head)
            if remote != local_path.read_bytes():
                raise RuntimeError(f"Main publication verification failed: {remote_path}")
        return main_head

    def _safe_stop(self) -> bool:
        if (
            self.destination_head is None
            or self.last_verified_progress_bytes is None
        ):
            return True
        remote = self._download(
            filename=f"{self.remote_root}/PROGRESS.json",
            revision=self.destination_head,
        )
        return remote == self.last_verified_progress_bytes

    def run(self) -> dict[str, Any]:
        self._load_hub()
        branches, legacy = self._list_refs()
        release = self._verify_release_pointer()
        self._initialize_destination(branches=branches, legacy=legacy)
        try:
            for lock in self.source_lock:
                self._process_branch(lock)
            final = self._finalize_destination(release)
            main_commit = self._publish_main(final)
        except KeyboardInterrupt:
            verified = self._safe_stop()
            print("SAFE_STOP_VERIFIED" if verified else "SAFE_STOP_FAILED", flush=True)
            raise
        except BaseException:
            try:
                verified = self._safe_stop()
                print("SAFE_STOP_VERIFIED" if verified else "SAFE_STOP_FAILED", flush=True)
            except Exception as stop_error:
                print(
                    "SAFE_STOP_FAILED",
                    {"reason": type(stop_error).__name__},
                    flush=True,
                )
            raise
        report = {
            "go": True,
            "experiment_id": self.experiment_id,
            "source_branch_count": final["manifest"]["source_branch_count"],
            "source_artifact_records": final["manifest"]["source_artifact_records"],
            "source_artifact_bytes": final["manifest"]["source_artifact_bytes"],
            "excluded_legacy_branches": final["manifest"]["excluded_legacy_branches"],
            "hypothesis_pass": final["hypothesis_pass"],
            "completion_is_independent_of_hypothesis": True,
            "consolidation_branch": self.destination_branch,
            "consolidation_commit_sha": self.destination_head,
            "main_commit_sha": main_commit,
            "source_branches_modified": False,
            "remote_verified": True,
            "main_visible": True,
        }
        print("CONSOLIDATION_COMPLETE", report, flush=True)
        return report


def run_consolidation(config: Mapping[str, Any], *, hf_token: str) -> dict[str, Any]:
    """Run or resume the exact locked v3r1 consolidation."""

    return Consolidator(config, hf_token=hf_token).run()
