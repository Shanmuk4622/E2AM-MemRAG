from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


BOOTSTRAP_PHASES = {"LOCAL_SMOKE", "REMOTE_UPLOAD", "REMOTE_RESTORE"}


def _last_sync(sync_results: Sequence[Mapping[str, Any]] | None) -> Mapping[str, Any]:
    return sync_results[-1] if sync_results else {}


def evaluate_bootstrap_gate(
    *,
    phase: str,
    preflight: Mapping[str, Any],
    completed_units: int,
    expected_units: int,
    shard_validation: Mapping[str, Any],
    energy: Mapping[str, Any],
    checkpoint_valid: bool,
    source_verified: bool,
    environment_verified: bool,
    sync_results: Sequence[Mapping[str, Any]] | None = None,
    restore_result: Mapping[str, Any] | None = None,
    fresh_restore_root: bool = False,
) -> dict[str, Any]:
    """Evaluate notebook 00 without allowing upload-only false positives.

    A verified upload proves that bytes reached the Hub.  It does not prove a new
    kernel can reconstruct them.  Consequently, only ``REMOTE_RESTORE`` can earn
    ``hard_pass=True`` and it must start from a deliberately empty worker root.
    """
    if phase not in BOOTSTRAP_PHASES:
        raise ValueError(f"Unknown bootstrap phase {phase!r}")
    if expected_units < 1 or completed_units < 0:
        raise ValueError("Unit counts must be positive and non-negative respectively")

    joules = energy.get("energy_joules")
    duration = energy.get("duration_seconds")
    samples = energy.get("samples")
    last_sync = _last_sync(sync_results)
    checks = {
        "disk_safe": bool(preflight.get("disk_ok")),
        "single_t4": (
            preflight.get("visible_gpu_count") == 1
            and "T4" in str(preflight.get("gpu_name", ""))
        ),
        "logical_resume_complete": completed_units == expected_units,
        "shards_checksum_valid": int(shard_validation.get("rows", -1)) == expected_units,
        "energy_sensor_valid": (
            bool(energy.get("available"))
            and isinstance(joules, (int, float))
            and math.isfinite(float(joules))
            and float(joules) > 0
            and isinstance(duration, (int, float))
            and float(duration) >= 1.5
            and isinstance(samples, int)
            and samples >= 10
            and bool(energy.get("gpu_uuid"))
        ),
        "checkpoint_seal_valid": bool(checkpoint_valid),
        "source_pin_verified": bool(source_verified),
        "environment_pin_verified": bool(environment_verified),
        "hub_closure_verified": (
            bool(last_sync.get("complete")) and bool(last_sync.get("verified"))
        ),
        "fresh_remote_restore_verified": (
            phase == "REMOTE_RESTORE"
            and bool(fresh_restore_root)
            and (restore_result or {}).get("status") == "RESTORED"
        ),
    }
    local_keys = {
        "disk_safe",
        "single_t4",
        "logical_resume_complete",
        "shards_checksum_valid",
        "energy_sensor_valid",
        "checkpoint_seal_valid",
    }
    reproducibility_keys = local_keys | {
        "source_pin_verified",
        "environment_pin_verified",
    }
    local_ready = all(checks[key] for key in local_keys)
    upload_verified = all(checks[key] for key in reproducibility_keys) and checks[
        "hub_closure_verified"
    ]
    hard_pass = upload_verified and checks["fresh_remote_restore_verified"]

    if hard_pass:
        status = "PASS"
    elif phase == "REMOTE_RESTORE":
        status = "RESTORE_INCOMPLETE"
    elif phase == "REMOTE_UPLOAD" and upload_verified:
        status = "UPLOAD_VERIFIED_RESTART_REQUIRED"
    elif phase == "REMOTE_UPLOAD":
        status = "UPLOAD_INCOMPLETE"
    elif local_ready:
        status = "LOCAL_READY_PIN_AND_UPLOAD_NEXT"
    else:
        status = "LOCAL_INCOMPLETE"

    return {
        "schema_version": 1,
        "phase": phase,
        "status": status,
        "hard_pass": hard_pass,
        "local_ready": local_ready,
        "upload_verified": upload_verified,
        "checks": checks,
        "next_phase": (
            None
            if hard_pass
            else "REMOTE_RESTORE"
            if upload_verified
            else "REMOTE_UPLOAD"
            if local_ready
            else "LOCAL_SMOKE"
        ),
    }
