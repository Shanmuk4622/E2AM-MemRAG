from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .events import EventLog


def _http_status(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class PeriodicHubSync:
    """Best-effort periodic uploader with explicit scientific-measurement pauses.

    The worker never converts an upload failure into lost local work. A forced flush
    returns a structured incomplete result when authentication, time, or locking
    prevents verification. Hard VM termination remains outside Python's control.
    """

    def __init__(
        self,
        manager: Any,
        events: EventLog,
        *,
        poll_seconds: float = 30.0,
        maximum_backoff_seconds: float = 900.0,
    ) -> None:
        if poll_seconds <= 0 or maximum_backoff_seconds < poll_seconds:
            raise ValueError("Invalid periodic-sync polling/backoff interval")
        self.manager = manager
        self.events = events
        self.poll_seconds = poll_seconds
        self.maximum_backoff_seconds = maximum_backoff_seconds
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._poll_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._failure_count = 0
        self._next_retry_epoch = 0.0
        self._auth_disabled = False
        self.last_result: dict[str, Any] | None = None
        self.last_error_type: str | None = None

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def auth_disabled(self) -> bool:
        return self._auth_disabled

    def start(self) -> "PeriodicHubSync":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="e2am-periodic-hub-sync", daemon=True
        )
        self._thread.start()
        self.events.append(
            "PERIODIC_SYNC_STARTED",
            effective_interval_seconds=getattr(
                self.manager, "effective_interval_seconds", None
            ),
        )
        return self

    def _schedule_failure(self, error: BaseException) -> None:
        self._failure_count += 1
        self.last_error_type = type(error).__name__
        status = _http_status(error)
        if status in {401, 403}:
            self._auth_disabled = True
            delay = self.maximum_backoff_seconds
        elif status == 429:
            delay = _retry_after_seconds(error) or min(
                self.maximum_backoff_seconds,
                self.poll_seconds * (2 ** min(self._failure_count, 6)),
            )
        else:
            delay = min(
                self.maximum_backoff_seconds,
                self.poll_seconds * (2 ** min(self._failure_count, 6)),
            )
        self._next_retry_epoch = time.time() + delay
        self.events.append(
            "PERIODIC_SYNC_BACKOFF",
            error_type=type(error).__name__,
            error=str(error),
            http_status=status,
            retry_after_seconds=delay,
            authentication_disabled=self._auth_disabled,
        )

    def poll_once(self) -> dict[str, Any]:
        with self._poll_lock:
            if self.paused:
                return {
                    "status": "PAUSED_FOR_MEASUREMENT",
                    "complete": False,
                    "verified": False,
                }
            if self._auth_disabled:
                return {"status": "AUTH_DISABLED", "complete": False, "verified": False}
            if time.time() < self._next_retry_epoch:
                return {"status": "BACKOFF", "complete": False, "verified": False}
            try:
                result = dict(self.manager.sync_once(force=False))
                self.last_result = result
                if result.get("status") not in {"LOCKED", "NO_TOKEN"}:
                    self._failure_count = 0
                    self._next_retry_epoch = 0.0
                return result
            except Exception as error:
                self._schedule_failure(error)
                return {
                    "status": "ERROR_BACKOFF",
                    "complete": False,
                    "verified": False,
                    "error_type": type(error).__name__,
                }

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.poll_once()

    def pause(self) -> None:
        self._pause.set()
        self.events.append("PERIODIC_SYNC_PAUSED")

    def resume(self) -> None:
        self._pause.clear()
        self.events.append("PERIODIC_SYNC_RESUMED")

    @contextmanager
    def measurement_block(self) -> Iterator[None]:
        was_paused = self.paused
        self.pause()
        # Wait for a poll already in progress, then hold the boundary so no
        # upload/network work can overlap the measured block.
        with self._poll_lock:
            try:
                yield
            finally:
                if not was_paused:
                    self.resume()

    def flush_and_stop(
        self, *, max_commits: int = 8, deadline_seconds: float = 300.0
    ) -> dict[str, Any]:
        self._stop.set()
        thread_stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=min(30.0, deadline_seconds))
            thread_stopped = not self._thread.is_alive()
        results: list[dict[str, Any]] = []
        if not thread_stopped:
            outcome = {
                "status": "BACKGROUND_UPLOAD_STILL_RUNNING",
                "complete": False,
                "verified": False,
                "results": results,
            }
        elif self._auth_disabled:
            outcome = {
                "status": "AUTH_DISABLED",
                "complete": False,
                "verified": False,
                "results": results,
            }
        else:
            try:
                results = list(
                    self.manager.sync_until_clean(
                        max_commits=max_commits, deadline_seconds=deadline_seconds
                    )
                )
                final = results[-1] if results else {}
                complete = bool(final.get("complete") and final.get("verified"))
                outcome = {
                    "status": "SAFE_STOP_VERIFIED" if complete else "SAFE_STOP_INCOMPLETE",
                    "complete": complete,
                    "verified": complete,
                    "results": results,
                }
            except Exception as error:
                self._schedule_failure(error)
                outcome = {
                    "status": "SAFE_STOP_ERROR",
                    "complete": False,
                    "verified": False,
                    "error_type": type(error).__name__,
                    "results": results,
                }
        self.last_result = outcome
        self.events.append("PERIODIC_SYNC_STOPPED", **outcome)
        return outcome
