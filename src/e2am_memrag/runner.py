from __future__ import annotations

import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from .events import EventLog
from .identity import RunIdentity, make_unit_id
from .manifest import ManifestStore
from .shards import ShardStore
from .signals import StopController
from .utils import redact_secrets, utc_now


@dataclass
class RunStats:
    eligible: int = 0
    already_completed: int = 0
    processed: int = 0
    failed_attempts: int = 0
    written_shards: int = 0
    terminal_failures: int = 0
    stopped_early: bool = False
    sync_callbacks: int = 0
    sync_failures: int = 0
    sync_incomplete: int = 0
    last_sync_status: str | None = None
    remote_sync_complete: bool = False
    remote_sync_verified: bool = False


class ResumableRunner:
    def __init__(
        self,
        identity: RunIdentity,
        manifest: ManifestStore,
        shards: ShardStore,
        events: EventLog,
        stop: StopController | None = None,
        sync_callback: Callable[[bool], Any] | None = None,
        ownership_callback: Callable[[str], bool] | None = None,
        shard_rows: int = 8,
        max_retries_per_unit: int = 2,
        fail_fast: bool = True,
    ) -> None:
        if shard_rows < 1:
            raise ValueError("shard_rows must be at least 1")
        if max_retries_per_unit < 1:
            raise ValueError("max_retries_per_unit must be at least 1")
        self.identity = identity
        self.manifest = manifest
        self.shards = shards
        self.events = events
        self.stop = stop or StopController()
        self.sync_callback = sync_callback
        self.ownership_callback = ownership_callback or identity.owns
        self.shard_rows = shard_rows
        self.max_retries_per_unit = max_retries_per_unit
        self.fail_fast = fail_fast

    def _sync(self, force: bool, stats: RunStats) -> None:
        """Attempt remote sync without weakening the local durability boundary."""
        if self.sync_callback is None:
            return
        stats.sync_callbacks += 1
        try:
            raw_result = self.sync_callback(force)
        except Exception as error:
            stats.sync_failures += 1
            stats.last_sync_status = f"ERROR:{type(error).__name__}"
            stats.remote_sync_complete = False
            stats.remote_sync_verified = False
            self.events.append(
                "REMOTE_SYNC_FAILED",
                spec_hash=self.identity.config_hash,
                forced=force,
                error_type=type(error).__name__,
                error=str(error),
            )
            return

        result: Mapping[str, Any] | None
        if isinstance(raw_result, Mapping):
            result = raw_result
        elif isinstance(raw_result, (list, tuple)) and raw_result and isinstance(
            raw_result[-1], Mapping
        ):
            result = raw_result[-1]
        else:
            result = None
        status = str(result.get("status", "UNKNOWN_RESULT")) if result else "UNKNOWN_RESULT"
        complete = bool(result and result.get("complete") is True)
        verified = bool(result and result.get("verified") is True)
        stats.last_sync_status = status
        stats.remote_sync_complete = complete
        stats.remote_sync_verified = verified
        if complete and verified:
            self.events.append(
                "REMOTE_SYNC_VERIFIED",
                spec_hash=self.identity.config_hash,
                forced=force,
                status=status,
                commit_sha=result.get("commit_sha") if result else None,
            )
        elif status == "NOT_DUE" and not force:
            self.events.append(
                "REMOTE_SYNC_DEFERRED",
                spec_hash=self.identity.config_hash,
                forced=False,
                status=status,
            )
        else:
            stats.sync_incomplete += 1
            self.events.append(
                "REMOTE_SYNC_INCOMPLETE",
                spec_hash=self.identity.config_hash,
                forced=force,
                status=status,
                complete=complete,
                verified=verified,
            )

    def _flush(self, buffer: list[dict[str, Any]], stats: RunStats) -> None:
        if not buffer:
            return
        record = self.shards.write_rows("traces", list(buffer))
        stats.written_shards += 1
        self.events.append(
            "TRACE_SHARD_COMMITTED",
            spec_hash=self.identity.config_hash,
            path=record.relative_path,
            rows=record.row_count,
            sha256=record.sha256,
        )
        buffer.clear()

    def run(
        self,
        units: Iterable[Mapping[str, Any]],
        processor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        max_new_units: int | None = None,
    ) -> RunStats:
        self.shards.reconcile()
        completed = self.shards.completed_unit_ids()
        stats = RunStats()
        buffer: list[dict[str, Any]] = []
        self.manifest.set_status("RUNNING")
        self.events.append(
            "RUN_STARTED",
            spec_hash=self.identity.config_hash,
            completed_before=len(completed),
        )

        for raw_unit in units:
            unit = dict(raw_unit)
            missing_unit_id = object()
            provided_unit_id = unit.pop("_unit_id", missing_unit_id)
            canonical_unit_id = make_unit_id(
                {"spec_hash": self.identity.config_hash, "unit": unit}
            )
            if provided_unit_id is missing_unit_id:
                unit_id = canonical_unit_id
            else:
                unit_id = str(provided_unit_id)
                if unit_id != canonical_unit_id:
                    raise ValueError(
                        "Provided _unit_id does not match the canonical ID for this "
                        "experiment spec and unit payload"
                    )
            if not self.ownership_callback(unit_id):
                continue
            stats.eligible += 1
            if unit_id in completed:
                stats.already_completed += 1
                continue
            if max_new_units is not None and stats.processed >= max_new_units:
                stats.stopped_early = True
                self.stop.request("max_new_units")
            if self.stop.requested:
                break

            started_at = utc_now()
            started_clock = perf_counter()
            last_error: BaseException | None = None
            for attempt in range(1, self.max_retries_per_unit + 1):
                try:
                    output = dict(processor(unit))
                    row = {
                        "unit_id": unit_id,
                        "spec_hash": self.identity.config_hash,
                        "status": "SUCCESS",
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "duration_seconds": perf_counter() - started_clock,
                        "input": unit,
                        "output": output,
                    }
                    buffer.append(row)
                    stats.processed += 1
                    last_error = None
                    break
                except KeyboardInterrupt:
                    self.stop.request("keyboard_interrupt")
                    last_error = None
                    self.events.append(
                        "INTERRUPT_REQUESTED",
                        unit_id=unit_id,
                        spec_hash=self.identity.config_hash,
                    )
                    break
                except Exception as error:
                    last_error = error
                    stats.failed_attempts += 1
                    failure = redact_secrets({
                        "unit_id": unit_id,
                        "spec_hash": self.identity.config_hash,
                        "status": "FAILED_ATTEMPT",
                        "attempt": attempt,
                        "at": utc_now(),
                        "input": unit,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    })
                    self.shards.write_rows("failures", [failure])
                    self.events.append(
                        "UNIT_FAILED",
                        unit_id=unit_id,
                        spec_hash=self.identity.config_hash,
                        attempt=attempt,
                        error_type=type(error).__name__,
                    )
            if self.stop.requested:
                break
            if last_error is not None:
                stats.terminal_failures += 1
                self._flush(buffer, stats)
                self.manifest.set_status(
                    "FAILED", unit_id=unit_id, error_type=type(last_error).__name__
                )
                self._sync(True, stats)
                if self.fail_fast:
                    raise last_error

            if len(buffer) >= self.shard_rows:
                self._flush(buffer, stats)
            self._sync(False, stats)

        self._flush(buffer, stats)
        if self.stop.requested:
            status = "PAUSED" if self.stop.reason == "max_new_units" else "INTERRUPTED"
            self.manifest.set_status(status, reason=self.stop.reason)
            self.events.append(
                "RUN_STOPPED",
                spec_hash=self.identity.config_hash,
                reason=self.stop.reason,
                status=status,
            )
            stats.stopped_early = True
        elif stats.terminal_failures:
            self.manifest.set_status(
                "FAILED", terminal_failures=stats.terminal_failures, processed=stats.processed
            )
            self.events.append(
                "RUN_FINISHED_WITH_FAILURES",
                spec_hash=self.identity.config_hash,
                processed=stats.processed,
                terminal_failures=stats.terminal_failures,
            )
        else:
            self.manifest.set_status("COMPLETED", processed=stats.processed)
            self.events.append(
                "RUN_COMPLETED",
                spec_hash=self.identity.config_hash,
                processed=stats.processed,
            )
        self._sync(True, stats)
        return stats
