from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .identity import RunIdentity
from .utils import atomic_write_json, read_json, utc_now


@dataclass(frozen=True)
class ShardRecord:
    kind: str
    relative_path: str
    sha256: str
    row_count: int
    sequence: int
    created_at: str
    first_unit_id: str | None = None
    last_unit_id: str | None = None
    recovered: bool = False


class ManifestStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, identity: RunIdentity) -> None:
        self.path = Path(path)
        self.identity = identity
        self._lock = threading.RLock()

    def initialize(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                manifest = self.read()
                if manifest["identity"] != self.identity.as_dict():
                    raise RuntimeError(
                        "Existing manifest identity does not match this worker. "
                        "Choose the correct experiment/config/worker identity; never merge them."
                    )
                return manifest
            now = utc_now()
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "identity": self.identity.as_dict(),
                "status": "RUNNING",
                "created_at": now,
                "updated_at": now,
                "metadata": metadata or {},
                "shards": [],
                "counters": {"trace_rows": 0, "failure_rows": 0},
            }
            atomic_write_json(self.path, manifest)
            return manifest

    def read(self) -> dict[str, Any]:
        manifest = read_json(self.path)
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError("Unsupported or missing manifest schema version")
        if not isinstance(manifest.get("shards"), list):
            raise RuntimeError("Manifest shards must be a list")
        return manifest

    def add_shard(self, record: ShardRecord) -> dict[str, Any]:
        with self._lock:
            manifest = self.read()
            existing = {
                item["relative_path"]: item for item in manifest.get("shards", [])
            }
            value = asdict(record)
            if record.relative_path in existing:
                if existing[record.relative_path] != value:
                    raise RuntimeError(
                        f"Immutable shard metadata changed: {record.relative_path}"
                    )
                return manifest
            manifest["shards"].append(value)
            manifest["shards"].sort(
                key=lambda item: (item["kind"], item["sequence"], item["relative_path"])
            )
            counter = "trace_rows" if record.kind == "traces" else "failure_rows"
            manifest["counters"][counter] = int(manifest["counters"].get(counter, 0)) + int(
                record.row_count
            )
            manifest["updated_at"] = utc_now()
            atomic_write_json(self.path, manifest)
            return manifest

    def next_sequence(self, kind: str) -> int:
        manifest = self.read()
        sequences = [
            int(item["sequence"])
            for item in manifest.get("shards", [])
            if item.get("kind") == kind
        ]
        return max(sequences, default=-1) + 1

    def set_status(self, status: str, **details: Any) -> dict[str, Any]:
        allowed = {"RUNNING", "PAUSED", "INTERRUPTED", "FAILED", "COMPLETED"}
        if status not in allowed:
            raise ValueError(f"Unsupported status {status!r}")
        with self._lock:
            manifest = self.read()
            manifest["status"] = status
            manifest["status_details"] = details
            manifest["updated_at"] = utc_now()
            atomic_write_json(self.path, manifest)
            return manifest

