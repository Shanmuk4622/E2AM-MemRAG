from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from e2am_memrag.config import RuntimeSettings


class RuntimeSettingsTests(unittest.TestCase):
    def test_default_worker_revision_is_stable_across_session_restarts(self) -> None:
        base = {
            "E2AM_EXPERIMENT_ID": "pilot-v1",
            "E2AM_WORKER_ID": "worker-00",
            "E2AM_HF_REPO_ID": "Shanmuk4622/E2AM-MemRAG-Traces",
        }
        with patch.dict(os.environ, {**base, "E2AM_SESSION_ID": "attempt-a"}, clear=True):
            first = RuntimeSettings.from_env()
        with patch.dict(os.environ, {**base, "E2AM_SESSION_ID": "attempt-b"}, clear=True):
            second = RuntimeSettings.from_env()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(first.hf_revision, "ingest-pilot-v1-worker-00")
        self.assertEqual(first.hf_revision, second.hf_revision)

    def test_explicit_revision_override_is_respected(self) -> None:
        values = {
            "E2AM_HF_REPO_ID": "Shanmuk4622/E2AM-MemRAG-Traces",
            "E2AM_HF_REVISION": "ingest-manual-lane",
        }
        with patch.dict(os.environ, values, clear=True):
            runtime = RuntimeSettings.from_env()
        self.assertEqual(runtime.hf_revision, "ingest-manual-lane")

    def test_default_sync_interval_is_ten_minutes(self) -> None:
        with patch.dict(
            os.environ,
            {"E2AM_HF_REPO_ID": "Shanmuk4622/E2AM-MemRAG-Traces"},
            clear=True,
        ):
            runtime = RuntimeSettings.from_env()
        self.assertEqual(runtime.sync_interval_seconds, 1200)

    def test_team_sync_settings_are_loaded_from_environment(self) -> None:
        values = {
            "E2AM_HF_REPO_ID": "Shanmuk4622/E2AM-MemRAG-Traces",
            "E2AM_SYNC_INTERVAL_SECONDS": "1200",
            "E2AM_ACTIVE_WORKER_COUNT": "4",
            "E2AM_SYNC_STAGGER_SECONDS": "150",
        }
        with patch.dict(os.environ, values, clear=True):
            runtime = RuntimeSettings.from_env()
        self.assertEqual(runtime.active_worker_count, 4)
        self.assertEqual(runtime.sync_stagger_seconds, 150)


if __name__ == "__main__":
    unittest.main()
