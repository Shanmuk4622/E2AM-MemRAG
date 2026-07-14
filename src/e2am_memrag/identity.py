from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .utils import canonical_json


_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def validate_slug(value: str, field: str) -> str:
    if not _SLUG_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-96 characters and contain only letters, numbers, '.', '_', or '-': "
            f"{value!r}"
        )
    return value


def make_config_hash(config: Mapping[str, Any], length: int = 16) -> str:
    if length < 12:
        raise ValueError("Config hashes shorter than 12 characters are not allowed")
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:length]


def make_unit_id(payload: Mapping[str, Any], namespace: str = "e2am.unit.v1") -> str:
    envelope = {"namespace": namespace, "payload": payload}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def assigned_shard(unit_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not re.fullmatch(r"[0-9a-f]{64}", unit_id):
        raise ValueError(f"Invalid unit_id: {unit_id!r}")
    return int(unit_id[:16], 16) % shard_count


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    config_hash: str
    worker_id: str
    shard_index: int
    shard_count: int

    def __post_init__(self) -> None:
        validate_slug(self.experiment_id, "experiment_id")
        validate_slug(self.config_hash, "config_hash")
        validate_slug(self.worker_id, "worker_id")
        if self.shard_count < 1:
            raise ValueError("shard_count must be at least 1")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")

    def owns(self, unit_id: str) -> bool:
        return assigned_shard(unit_id, self.shard_count) == self.shard_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "worker_id": self.worker_id,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
        }

