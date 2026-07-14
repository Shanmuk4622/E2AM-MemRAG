from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    fsync_directory,
    read_json,
    sha256_file,
    utc_now,
)


_REQUIRED_RESUME_FIELDS = {"model", "optimizer", "scheduler", "rng", "data_cursor"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEAL_FIELDS = {"schema_version", "step", "created_at", "tree_hash", "files", "metadata"}


def _descriptor_hash(
    step: int, files: list[dict[str, Any]], metadata: Mapping[str, Any]
) -> str:
    descriptor = {"step": step, "files": files, "metadata": dict(metadata)}
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointRef:
    step: int
    path: Path
    tree_hash: str
    metadata: dict[str, Any]


class CheckpointStore:
    """Immutable sealed checkpoints; incomplete directories are never resumable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def seal(
        self,
        step: int,
        artifacts: Mapping[str, bytes | str | Path],
        metadata: Mapping[str, Any],
    ) -> CheckpointRef:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        missing = sorted(_REQUIRED_RESUME_FIELDS - set(metadata))
        if missing:
            raise ValueError(f"Checkpoint metadata is missing resume fields: {missing}")
        try:
            _descriptor_hash(step, [], metadata)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Checkpoint metadata must be finite, deterministic JSON"
            ) from error
        temporary = Path(tempfile.mkdtemp(prefix=f".step-{step:09d}-", dir=self.root))
        try:
            artifact_paths: set[str] = set()
            for relative_name, source in artifacts.items():
                relative = Path(relative_name)
                normalized = relative.as_posix()
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or ".." in relative.parts
                    or normalized == "_COMPLETE.json"
                ):
                    raise ValueError(f"Unsafe checkpoint artifact path: {relative_name}")
                if normalized in artifact_paths:
                    raise ValueError(f"Duplicate checkpoint artifact path: {relative_name}")
                artifact_paths.add(normalized)
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(source, bytes):
                    atomic_write_bytes(destination, source)
                else:
                    shutil.copyfile(Path(source), destination)
                    with destination.open("rb") as handle:
                        os.fsync(handle.fileno())
            files = [
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            ]
            tree_hash = _descriptor_hash(step, files, metadata)
            seal = {
                "schema_version": 1,
                "step": step,
                "created_at": utc_now(),
                "tree_hash": tree_hash,
                "files": files,
                "metadata": dict(metadata),
            }
            atomic_write_json(temporary / "_COMPLETE.json", seal)
            destination = self.root / f"step-{step:09d}-{tree_hash[:12]}"
            for existing_path in sorted(self.root.glob(f"step-{step:09d}-*")):
                if not existing_path.is_dir() or not (existing_path / "_COMPLETE.json").is_file():
                    continue
                existing = self._validate(existing_path)
                if existing.tree_hash == tree_hash:
                    shutil.rmtree(temporary)
                    return existing
                raise RuntimeError(
                    f"Checkpoint step {step} already has different sealed content/metadata: "
                    f"{existing_path}"
                )
            if destination.exists():
                existing = self._validate(destination)
                if existing.tree_hash != tree_hash:
                    raise RuntimeError(f"Checkpoint step collision at {destination}")
                shutil.rmtree(temporary)
                return existing
            os.replace(temporary, destination)
            fsync_directory(self.root)
            return self._validate(destination)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _validate(self, path: Path) -> CheckpointRef:
        if path.is_symlink():
            raise RuntimeError(f"Checkpoint directory cannot be a symbolic link: {path}")
        seal_path = path / "_COMPLETE.json"
        if not seal_path.is_file() or seal_path.is_symlink():
            raise RuntimeError(f"Checkpoint is not sealed: {path}")
        seal = read_json(seal_path)
        schema_version = seal.get("schema_version")
        if (
            set(seal) != _SEAL_FIELDS
            or isinstance(schema_version, bool)
            or schema_version != 1
        ):
            raise RuntimeError(f"Checkpoint seal has an unsupported schema: {seal_path}")
        raw_step = seal.get("step")
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
            raise RuntimeError(f"Checkpoint seal has an invalid step: {seal_path}")
        step = raw_step
        created_at = seal.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise RuntimeError(f"Checkpoint seal has an invalid timestamp: {seal_path}")
        metadata = seal.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Checkpoint seal metadata must be an object: {seal_path}")
        missing = sorted(_REQUIRED_RESUME_FIELDS - set(metadata))
        if missing:
            raise RuntimeError(
                f"Checkpoint seal is missing resume metadata {missing}: {seal_path}"
            )
        raw_files = seal.get("files")
        if not isinstance(raw_files, list):
            raise RuntimeError(f"Checkpoint seal file inventory must be a list: {seal_path}")

        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_files):
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                raise RuntimeError(
                    f"Checkpoint seal has an invalid file record at index {index}: {seal_path}"
                )
            if not isinstance(item["path"], str):
                raise RuntimeError(
                    f"Checkpoint seal has a non-string artifact path at index {index}"
                )
            relative = PurePosixPath(item["path"])
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.as_posix() == "_COMPLETE.json"
            ):
                raise RuntimeError(f"Checkpoint seal has an unsafe artifact path: {item['path']!r}")
            relative_name = relative.as_posix()
            if relative_name in seen:
                raise RuntimeError(f"Checkpoint seal lists an artifact twice: {relative_name}")
            seen.add(relative_name)
            digest = item["sha256"]
            byte_count = item["bytes"]
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise RuntimeError(f"Checkpoint seal has an invalid hash: {relative_name}")
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
            ):
                raise RuntimeError(f"Checkpoint seal has an invalid byte count: {relative_name}")
            artifact = path.joinpath(*relative.parts)
            if artifact.is_symlink() or not artifact.is_file():
                raise RuntimeError(f"Checkpoint artifact failed verification: {artifact}")
            if artifact.stat().st_size != byte_count or sha256_file(artifact) != digest:
                raise RuntimeError(f"Checkpoint artifact failed verification: {artifact}")
            files.append({"path": relative_name, "sha256": digest, "bytes": byte_count})

        if files != sorted(files, key=lambda item: item["path"]):
            raise RuntimeError(f"Checkpoint seal file inventory is not canonical: {seal_path}")
        actual_files = {
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate != seal_path
        }
        if actual_files != seen:
            missing_from_seal = sorted(actual_files - seen)
            missing_from_disk = sorted(seen - actual_files)
            raise RuntimeError(
                "Checkpoint file inventory mismatch; "
                f"unsealed={missing_from_seal}, missing={missing_from_disk}: {path}"
            )

        tree_hash = seal.get("tree_hash")
        if not isinstance(tree_hash, str) or not _SHA256_RE.fullmatch(tree_hash):
            raise RuntimeError(f"Checkpoint seal has an invalid tree hash: {seal_path}")
        try:
            computed_tree_hash = _descriptor_hash(step, files, metadata)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Checkpoint metadata is not finite deterministic JSON: {seal_path}"
            ) from error
        if computed_tree_hash != tree_hash:
            raise RuntimeError(f"Checkpoint tree hash mismatch: {path}")
        expected_name = f"step-{step:09d}-{tree_hash[:12]}"
        if path.name != expected_name:
            raise RuntimeError(
                f"Checkpoint directory name does not match its seal: {path.name!r} != "
                f"{expected_name!r}"
            )
        return CheckpointRef(
            step=step,
            path=path,
            tree_hash=tree_hash,
            metadata=dict(metadata),
        )

    def latest_valid(self) -> CheckpointRef | None:
        candidates: list[CheckpointRef] = []
        for path in sorted(self.root.glob("step-*-*")):
            if path.is_dir() and (path / "_COMPLETE.json").is_file():
                candidates.append(self._validate(path))
        return max(candidates, key=lambda item: item.step) if candidates else None
