from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .utils import canonical_json, sha256_bytes


def logical_result_hash(row: Mapping[str, Any]) -> str:
    """Hash scientific content while ignoring timing/attempt bookkeeping."""
    payload = {
        "unit_id": row.get("unit_id"),
        "spec_hash": row.get("spec_hash"),
        "status": row.get("status"),
        "input": row.get("input"),
        "output": row.get("output"),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class AggregateResult:
    rows: tuple[dict[str, Any], ...]
    identical_duplicates: int
    spec_hash: str


def aggregate_rows(row_groups: Iterable[Iterable[Mapping[str, Any]]]) -> AggregateResult:
    canonical: dict[str, tuple[str, dict[str, Any]]] = {}
    duplicates = 0
    aggregate_spec_hash: str | None = None
    for group in row_groups:
        for source_row in group:
            row = dict(source_row)
            unit_id = str(row.get("unit_id", ""))
            if not unit_id:
                raise ValueError("Cannot aggregate a row without unit_id")
            spec_hash = str(row.get("spec_hash", ""))
            if not spec_hash:
                raise ValueError(f"Cannot aggregate row {unit_id!r} without spec_hash")
            if aggregate_spec_hash is None:
                aggregate_spec_hash = spec_hash
            elif spec_hash != aggregate_spec_hash:
                raise RuntimeError(
                    "Rows from different experiment specs cannot enter one aggregate: "
                    f"{aggregate_spec_hash!r} != {spec_hash!r}"
                )
            digest = logical_result_hash(row)
            if unit_id in canonical:
                previous_digest, _ = canonical[unit_id]
                if digest != previous_digest:
                    raise RuntimeError(
                        f"Divergent results for unit_id={unit_id}; release is blocked"
                    )
                duplicates += 1
                continue
            canonical[unit_id] = (digest, row)
    if aggregate_spec_hash is None:
        raise ValueError("Cannot aggregate an empty result set")
    rows = tuple(canonical[unit_id][1] for unit_id in sorted(canonical))
    return AggregateResult(
        rows=rows,
        identical_duplicates=duplicates,
        spec_hash=aggregate_spec_hash,
    )
