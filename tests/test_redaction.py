from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from e2am_memrag.events import EventLog


class RedactionTests(unittest.TestCase):
    def test_hf_tokens_do_not_enter_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            token = "hf_" + "abcdefghijklmnopqrstuvwxyz123456"
            previous = os.environ.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = token
            try:
                path = Path(temporary) / "events.jsonl"
                EventLog(path).append("ERROR", detail=f"request failed for {token}")
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(token, text)
                self.assertIn("REDACTED_HF_TOKEN", text)
                json.loads(text)
            finally:
                if previous is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = previous

    def test_event_snapshot_is_offset_exact_and_rejects_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            events = EventLog(path)
            empty, offset_zero = events.snapshot()
            self.assertEqual((empty, offset_zero), (b"", 0))

            events.append("FIRST", value=1)
            first, offset_one = events.snapshot(offset_zero)
            self.assertEqual(len(first.splitlines()), 1)
            events.append("SECOND", value=2)
            second, offset_two = events.snapshot(offset_one)
            self.assertEqual(len(second.splitlines()), 1)
            self.assertEqual(first + second, path.read_bytes())
            self.assertEqual(offset_two, len(first + second))
            with self.assertRaises(RuntimeError):
                events.snapshot(offset_two + 1)


if __name__ == "__main__":
    unittest.main()
