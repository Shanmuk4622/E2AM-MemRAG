from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .identity import RunIdentity


@dataclass(frozen=True)
class RunPaths:
    work_root: Path
    identity: RunIdentity

    @property
    def worker_root(self) -> Path:
        i = self.identity
        return (
            self.work_root
            / i.experiment_id
            / i.config_hash
            / "workers"
            / i.worker_id
        )

    @property
    def manifest(self) -> Path:
        return self.worker_root / "manifest.json"

    @property
    def events(self) -> Path:
        return self.worker_root / "events.jsonl"

    @property
    def meta(self) -> Path:
        return self.worker_root / "meta"

    @property
    def traces(self) -> Path:
        return self.worker_root / "traces"

    @property
    def failures(self) -> Path:
        return self.worker_root / "failures"

    @property
    def checkpoints(self) -> Path:
        return self.worker_root / "checkpoints"

    @property
    def sync(self) -> Path:
        return self.worker_root / "sync"

    def create(self) -> "RunPaths":
        for path in (
            self.worker_root,
            self.meta,
            self.traces,
            self.failures,
            self.checkpoints,
            self.sync / "snapshots",
            self.sync / "event_deltas",
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

