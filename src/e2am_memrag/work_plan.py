from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .identity import assigned_shard


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: str
    logical_partitions: tuple[int, ...]
    partition_count: int = 256

    def __post_init__(self) -> None:
        if self.partition_count < 1:
            raise ValueError("partition_count must be positive")
        if not self.logical_partitions:
            raise ValueError("At least one logical partition must be assigned")
        if len(self.logical_partitions) != len(set(self.logical_partitions)):
            raise ValueError("logical_partitions contains duplicates")
        if any(not 0 <= value < self.partition_count for value in self.logical_partitions):
            raise ValueError("Logical partition is outside the frozen partition range")

    def owns(self, unit_id: str) -> bool:
        return assigned_shard(unit_id, self.partition_count) in self.logical_partitions

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "logical_partitions": list(self.logical_partitions),
            "partition_count": self.partition_count,
        }


def validate_work_plan(assignments: Iterable[WorkerAssignment], require_full: bool = False) -> None:
    materialized = list(assignments)
    if not materialized:
        raise ValueError("Work plan has no assignments")
    counts = {assignment.partition_count for assignment in materialized}
    if len(counts) != 1:
        raise ValueError("All workers must use the same frozen partition_count")
    owners: dict[int, str] = {}
    for assignment in materialized:
        for partition in assignment.logical_partitions:
            if partition in owners:
                raise ValueError(
                    f"Logical partition {partition} is assigned to both {owners[partition]!r} "
                    f"and {assignment.worker_id!r}"
                )
            owners[partition] = assignment.worker_id
    partition_count = counts.pop()
    if require_full and set(owners) != set(range(partition_count)):
        missing = sorted(set(range(partition_count)) - set(owners))
        raise ValueError(f"Work plan has unassigned partitions: {missing[:20]}")


def assignments_from_mapping(value: Mapping[str, Iterable[int]], partition_count: int = 256) -> list[WorkerAssignment]:
    assignments = [
        WorkerAssignment(worker_id, tuple(sorted(set(partitions))), partition_count)
        for worker_id, partitions in sorted(value.items())
    ]
    validate_work_plan(assignments)
    return assignments

