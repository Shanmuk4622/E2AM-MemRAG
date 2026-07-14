"""E2AM-MemRAG experiment infrastructure."""

from .config import ExperimentConfig, RuntimeSettings
from .identity import RunIdentity, make_unit_id
from .gates import evaluate_bootstrap_gate
from .manifest import ManifestStore, ShardRecord
from .runner import ResumableRunner, RunStats
from .shards import ShardStore

__all__ = [
    "ExperimentConfig",
    "ManifestStore",
    "ResumableRunner",
    "RunIdentity",
    "RunStats",
    "RuntimeSettings",
    "ShardRecord",
    "ShardStore",
    "make_unit_id",
    "evaluate_bootstrap_gate",
]

__version__ = "0.1.0"
