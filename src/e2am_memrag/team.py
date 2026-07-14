from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

from .identity import validate_slug
from .utils import canonical_json
from .work_plan import WorkerAssignment, validate_work_plan


TEAM_PLAN_SCHEMA_VERSION = 1


def effective_periodic_interval_seconds(
    worker_count: int,
    *,
    target_interval_seconds: int = 1200,
    request_ceiling_per_hour: int = 128,
    reserve_fraction: float = 0.25,
    estimated_calls_per_data_push: int = 3,
) -> int:
    """Return a shared-budget-safe periodic interval for active worker lanes."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if target_interval_seconds < 300:
        raise ValueError("target_interval_seconds must be at least five minutes")
    if request_ceiling_per_hour < 1 or estimated_calls_per_data_push < 1:
        raise ValueError("request and call estimates must be positive")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    scheduled_budget = math.floor(request_ceiling_per_hour * (1 - reserve_fraction))
    global_pushes_per_hour = max(
        1,
        scheduled_budget // (worker_count * estimated_calls_per_data_push),
    )
    return max(target_interval_seconds, math.ceil(3600 / global_pushes_per_hour))


def build_team_plan(
    *,
    experiment_id: str,
    worker_ids: Iterable[str],
    partition_count: int = 256,
    target_interval_seconds: int = 1200,
    request_ceiling_per_hour: int = 128,
    reserve_fraction: float = 0.25,
    estimated_calls_per_data_push: int = 3,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, fully covered logical work plan for collaborators."""
    validate_slug(experiment_id, "experiment_id")
    workers = tuple(sorted(str(worker_id) for worker_id in worker_ids))
    if not workers:
        raise ValueError("At least one worker is required")
    if len(workers) != len(set(workers)):
        raise ValueError("worker_ids must be unique")
    for worker_id in workers:
        validate_slug(worker_id, "worker_id")
    if partition_count < len(workers):
        raise ValueError("partition_count must be at least worker_count")

    assignments = [
        WorkerAssignment(
            worker_id=worker_id,
            logical_partitions=tuple(
                partition
                for partition in range(partition_count)
                if partition % len(workers) == worker_index
            ),
            partition_count=partition_count,
        )
        for worker_index, worker_id in enumerate(workers)
    ]
    validate_work_plan(assignments, require_full=True)
    effective_interval = effective_periodic_interval_seconds(
        len(workers),
        target_interval_seconds=target_interval_seconds,
        request_ceiling_per_hour=request_ceiling_per_hour,
        reserve_fraction=reserve_fraction,
        estimated_calls_per_data_push=estimated_calls_per_data_push,
    )
    payload: dict[str, Any] = {
        "schema_version": TEAM_PLAN_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "partition_count": partition_count,
        "worker_count": len(workers),
        "source": dict(source or {}),
        "workers": [
            {
                **assignment.as_dict(),
                "sync_stagger_seconds": math.floor(
                    index * target_interval_seconds / len(workers)
                ),
            }
            for index, assignment in enumerate(assignments)
        ],
        "sync": {
            "target_interval_seconds": target_interval_seconds,
            "effective_interval_seconds": effective_interval,
            "estimated_calls_per_data_push": estimated_calls_per_data_push,
            "request_ceiling_per_hour": request_ceiling_per_hour,
            "reserve_fraction": reserve_fraction,
            "ten_minute_target_supported": effective_interval
            == target_interval_seconds,
            "full_closure_policy": "major_stage_completion_or_safe_stop",
        },
    }
    payload["plan_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def verify_team_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a serialized team plan and reproduce its content hash."""
    payload = dict(value)
    claimed_hash = payload.pop("plan_sha256", None)
    expected_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if claimed_hash != expected_hash:
        raise ValueError("Team plan hash mismatch")
    if payload.get("schema_version") != TEAM_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported team plan schema")
    workers = payload.get("workers")
    if not isinstance(workers, list):
        raise ValueError("Team plan workers must be a list")
    assignments = [
        WorkerAssignment(
            worker_id=str(worker["worker_id"]),
            logical_partitions=tuple(int(item) for item in worker["logical_partitions"]),
            partition_count=int(worker["partition_count"]),
        )
        for worker in workers
    ]
    validate_work_plan(assignments, require_full=True)
    if int(payload.get("worker_count", -1)) != len(assignments):
        raise ValueError("Team plan worker count mismatch")
    return {**payload, "plan_sha256": claimed_hash}
