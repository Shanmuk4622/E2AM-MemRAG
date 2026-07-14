from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .identity import RunIdentity, make_config_hash, validate_slug


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    data: dict[str, Any]
    config_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path).resolve()
        with source.open("r", encoding="utf-8") as handle:
            if source.suffix.lower() == ".json":
                import json

                value = json.load(handle)
            else:
                try:
                    import yaml
                except ImportError as error:
                    raise RuntimeError(
                        "PyYAML is required for YAML configs. Install requirements-kaggle.txt "
                        "or use the equivalent JSON config."
                    ) from error
                value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Experiment config must be a mapping: {source}")
        schema_version = value.get("schema_version")
        if schema_version != 1:
            raise ValueError(f"Unsupported schema_version={schema_version!r}; expected 1")
        if value.get("project") != "e2am-memrag":
            raise ValueError("Config project must be 'e2am-memrag'")
        return cls(path=source, data=value, config_hash=make_config_hash(value))

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section {name!r} must be a mapping")
        return value


@dataclass(frozen=True)
class RuntimeSettings:
    experiment_id: str
    worker_id: str
    shard_index: int
    shard_count: int
    physical_gpu_index: int
    work_root: Path
    hf_repo_id: str
    session_id: str = "session-local"
    hf_revision: str = "ingest-local"
    hf_repo_type: str = "dataset"
    sync_interval_seconds: int = 1200
    max_sync_commits_per_hour: int = 8
    active_worker_count: int = 1
    sync_stagger_seconds: int = 0

    def __post_init__(self) -> None:
        validate_slug(self.experiment_id, "experiment_id")
        validate_slug(self.worker_id, "worker_id")
        validate_slug(self.session_id, "session_id")
        validate_slug(self.hf_revision, "hf_revision")
        if self.shard_count < 1 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("Invalid shard_index/shard_count")
        if self.physical_gpu_index < 0:
            raise ValueError("physical_gpu_index must be non-negative")
        if self.hf_repo_type not in {"dataset", "model"}:
            raise ValueError("hf_repo_type must be 'dataset' or 'model'")
        if "/" not in self.hf_repo_id:
            raise ValueError("hf_repo_id must use the namespace/name form")
        if self.sync_interval_seconds < 300:
            raise ValueError("Sync intervals below five minutes are not supported")
        if not 1 <= self.max_sync_commits_per_hour <= 12:
            raise ValueError("max_sync_commits_per_hour must be between 1 and 12")
        if self.active_worker_count < 1:
            raise ValueError("active_worker_count must be positive")
        if not 0 <= self.sync_stagger_seconds < self.sync_interval_seconds:
            raise ValueError(
                "sync_stagger_seconds must satisfy 0 <= stagger < sync interval"
            )

    @classmethod
    def from_env(cls, defaults: Mapping[str, Any] | None = None) -> "RuntimeSettings":
        values = dict(defaults or {})

        def env(name: str, default: Any) -> Any:
            return os.environ.get(name, values.get(name, default))

        experiment_id = str(env("E2AM_EXPERIMENT_ID", "bootstrap-v1"))
        worker_id = str(env("E2AM_WORKER_ID", "local-shard-00"))
        session_id = str(env("E2AM_SESSION_ID", "session-local"))
        return cls(
            experiment_id=experiment_id,
            worker_id=worker_id,
            shard_index=int(env("E2AM_SHARD_INDEX", 0)),
            shard_count=int(env("E2AM_SHARD_COUNT", 1)),
            physical_gpu_index=int(env("E2AM_GPU_INDEX", 0)),
            work_root=Path(str(env("E2AM_WORK_ROOT", "runs"))).expanduser().resolve(),
            hf_repo_id=str(
                env("E2AM_HF_REPO_ID", "Shanmuk4622/E2AM-MemRAG-Traces")
            ),
            session_id=session_id,
            hf_revision=str(
                env(
                    "E2AM_HF_REVISION",
                    # A worker lane must survive Kaggle kernel restarts.  The
                    # session id records an execution attempt; it must not
                    # silently create a new branch and strand the last seal.
                    f"ingest-{experiment_id}-{worker_id}",
                )
            ),
            hf_repo_type=str(env("E2AM_HF_REPO_TYPE", "dataset")),
            sync_interval_seconds=int(env("E2AM_SYNC_INTERVAL_SECONDS", 1200)),
            max_sync_commits_per_hour=int(
                env("E2AM_MAX_SYNC_COMMITS_PER_HOUR", 8)
            ),
            active_worker_count=int(env("E2AM_ACTIVE_WORKER_COUNT", 1)),
            sync_stagger_seconds=int(env("E2AM_SYNC_STAGGER_SECONDS", 0)),
        )

    def identity(self, config_hash: str) -> RunIdentity:
        return RunIdentity(
            experiment_id=self.experiment_id,
            config_hash=config_hash,
            worker_id=self.worker_id,
            shard_index=self.shard_index,
            shard_count=self.shard_count,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "worker_id": self.worker_id,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "physical_gpu_index": self.physical_gpu_index,
            "work_root": str(self.work_root),
            "hf_repo_id": self.hf_repo_id,
            "session_id": self.session_id,
            "hf_revision": self.hf_revision,
            "hf_repo_type": self.hf_repo_type,
            "sync_interval_seconds": self.sync_interval_seconds,
            "max_sync_commits_per_hour": self.max_sync_commits_per_hour,
            "active_worker_count": self.active_worker_count,
            "sync_stagger_seconds": self.sync_stagger_seconds,
        }
