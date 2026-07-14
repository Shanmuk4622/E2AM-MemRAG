from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import ManifestStore, ShardRecord
from .paths import RunPaths
from .utils import atomic_write_bytes, redact_secrets, sha256_bytes, sha256_file, utc_now


_SHARD_RE = re.compile(r"^part-(\d{6})-([0-9a-f]{12})\.jsonl\.gz$")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden in a shard: {value}")


def _validate_artifact_value(value: Any, path: str = "$") -> None:
    """Reject values that cannot safely enter a durable result artifact."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite float is forbidden in a shard at {path}")
    if isinstance(value, str) and redact_secrets(value) != value:
        raise ValueError(f"Secret-like value is forbidden in a shard at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if redact_secrets(key_text) != key_text:
                raise ValueError(f"Secret-like object key is forbidden in a shard at {path}")
            _validate_artifact_value(child, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_artifact_value(child, f"{path}[{index}]")


class ShardStore:
    """Immutable result shards; a file is durable before it enters the manifest."""

    def __init__(self, paths: RunPaths, manifest: ManifestStore) -> None:
        self.paths = paths.create()
        self.manifest = manifest

    def _directory(self, kind: str) -> Path:
        if kind == "traces":
            return self.paths.traces
        if kind == "failures":
            return self.paths.failures
        raise ValueError("kind must be 'traces' or 'failures'")

    @staticmethod
    def _encode(rows: list[dict[str, Any]]) -> bytes:
        for index, row in enumerate(rows):
            _validate_artifact_value(row, f"$[{index}]")
        raw = b"".join(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
        return gzip.compress(raw, compresslevel=6, mtime=0)

    @staticmethod
    def read_rows(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(value, dict):
                    raise RuntimeError(f"Non-object row at {path}:{line_number}")
                _validate_artifact_value(value, f"{path}:{line_number}")
                rows.append(value)
        return rows

    def write_rows(self, kind: str, rows: Iterable[dict[str, Any]]) -> ShardRecord:
        materialized = list(rows)
        if not materialized:
            raise ValueError("Cannot create an empty shard")
        unit_ids = [str(row.get("unit_id", "")) for row in materialized]
        if any(not unit_id for unit_id in unit_ids):
            raise ValueError("Every shard row requires a non-empty unit_id")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("A shard cannot contain duplicate unit_id values")

        self.reconcile(kind)
        sequence = self.manifest.next_sequence(kind)
        encoded = self._encode(materialized)
        digest = sha256_bytes(encoded)
        filename = f"part-{sequence:06d}-{digest[:12]}.jsonl.gz"
        path = self._directory(kind) / filename
        if path.exists() and sha256_file(path) != digest:
            raise RuntimeError(f"Refusing to replace immutable shard: {path}")
        if not path.exists():
            atomic_write_bytes(path, encoded)

        record = ShardRecord(
            kind=kind,
            relative_path=path.relative_to(self.paths.worker_root).as_posix(),
            sha256=digest,
            row_count=len(materialized),
            sequence=sequence,
            created_at=utc_now(),
            first_unit_id=unit_ids[0],
            last_unit_id=unit_ids[-1],
        )
        self.manifest.add_shard(record)
        return record

    def reconcile(self, kind: str | None = None) -> list[ShardRecord]:
        """Recover valid files written just before an interruption updated the manifest."""
        if not self.paths.manifest.exists():
            raise RuntimeError("Manifest must be initialized before shard reconciliation")
        existing = {
            item["relative_path"] for item in self.manifest.read().get("shards", [])
        }
        recovered: list[ShardRecord] = []
        kinds = [kind] if kind else ["traces", "failures"]
        for current_kind in kinds:
            directory = self._directory(current_kind)
            for path in sorted(directory.glob("part-*.jsonl.gz")):
                relative = path.relative_to(self.paths.worker_root).as_posix()
                if relative in existing:
                    continue
                match = _SHARD_RE.match(path.name)
                if not match:
                    continue
                rows = self.read_rows(path)
                if not rows:
                    raise RuntimeError(f"Empty orphan shard requires manual inspection: {path}")
                digest = sha256_file(path)
                if digest[:12] != match.group(2):
                    raise RuntimeError(f"Corrupt or renamed shard requires manual inspection: {path}")
                unit_ids = [str(row.get("unit_id", "")) for row in rows]
                if any(not unit_id for unit_id in unit_ids):
                    raise RuntimeError(f"Recovered shard has missing unit IDs: {path}")
                record = ShardRecord(
                    kind=current_kind,
                    relative_path=relative,
                    sha256=digest,
                    row_count=len(rows),
                    sequence=int(match.group(1)),
                    created_at=utc_now(),
                    first_unit_id=unit_ids[0],
                    last_unit_id=unit_ids[-1],
                    recovered=True,
                )
                self.manifest.add_shard(record)
                existing.add(relative)
                recovered.append(record)
        return recovered

    def validate(self) -> dict[str, int]:
        manifest = self.manifest.read()
        counts = {"shards": 0, "rows": 0}
        for item in manifest.get("shards", []):
            path = self.paths.worker_root / item["relative_path"]
            if not path.is_file():
                raise RuntimeError(f"Manifest references a missing shard: {path}")
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"Shard checksum mismatch: {path}")
            rows = self.read_rows(path)
            if len(rows) != int(item["row_count"]):
                raise RuntimeError(f"Shard row-count mismatch: {path}")
            counts["shards"] += 1
            counts["rows"] += len(rows)
        return counts

    def completed_unit_ids(self) -> set[str]:
        completed: set[str] = set()
        manifest = self.manifest.read()
        for item in manifest.get("shards", []):
            if item.get("kind") != "traces":
                continue
            path = self.paths.worker_root / item["relative_path"]
            for row in self.read_rows(path):
                if row.get("status") == "SUCCESS":
                    unit_id = str(row["unit_id"])
                    if unit_id in completed:
                        raise RuntimeError(f"Duplicate completed unit detected: {unit_id}")
                    completed.add(unit_id)
        return completed
