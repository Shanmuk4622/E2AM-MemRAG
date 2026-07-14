from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from e2am_memrag.events import EventLog
from e2am_memrag.periodic_sync import PeriodicHubSync


class _FakeManager:
    effective_interval_seconds = 1200

    def __init__(self, once=None, final=None):
        self.once = once or {"status": "NOT_DUE", "complete": False, "verified": False}
        self.final = final or [{"status": "CLEAN", "complete": True, "verified": True}]

    def sync_once(self, force=False):
        del force
        if isinstance(self.once, BaseException):
            raise self.once
        return self.once

    def sync_until_clean(self, max_commits=8, deadline_seconds=300.0):
        del max_commits, deadline_seconds
        return self.final


class PeriodicHubSyncTests(unittest.TestCase):
    def _events(self, root: str) -> EventLog:
        return EventLog(Path(root) / "events.jsonl")

    def test_measurement_pause_prevents_upload_poll(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            worker = PeriodicHubSync(
                _FakeManager(), self._events(temporary), poll_seconds=1
            )
            with worker.measurement_block():
                self.assertEqual(worker.poll_once()["status"], "PAUSED_FOR_MEASUREMENT")
            self.assertEqual(worker.poll_once()["status"], "NOT_DUE")

    def test_safe_stop_never_claims_success_for_incomplete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = _FakeManager(
                final=[{"status": "NO_TOKEN", "complete": False, "verified": False}]
            )
            worker = PeriodicHubSync(manager, self._events(temporary), poll_seconds=1)
            result = worker.flush_and_stop()
            self.assertEqual(result["status"], "SAFE_STOP_INCOMPLETE")
            self.assertFalse(result["complete"])

    def test_successful_safe_stop_requires_complete_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            worker = PeriodicHubSync(
                _FakeManager(), self._events(temporary), poll_seconds=1
            )
            result = worker.flush_and_stop()
            self.assertEqual(result["status"], "SAFE_STOP_VERIFIED")
            self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
