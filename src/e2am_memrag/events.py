from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .utils import redact_secrets, utc_now


class EventLog:
    """Single-writer append-only event log with an fsync durability boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        record = redact_secrets({"at": utc_now(), "event": event, **payload})
        line = json.dumps(
            record, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def snapshot(self, start: int = 0) -> tuple[bytes, int]:
        """Return an immutable byte-range snapshot and its exclusive end offset.

        The same lock guards appends and snapshots, so a sync operation can never
        observe half of a JSONL record. Offsets beyond the current end are treated
        as corruption instead of silently dropping events.
        """
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("Event snapshot start must be a non-negative integer")
        with self._lock:
            if not self.path.exists():
                if start:
                    raise RuntimeError(
                        f"Event snapshot offset {start} is beyond an absent event log"
                    )
                return b"", 0
            with self.path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                end = handle.tell()
                if start > end:
                    raise RuntimeError(
                        f"Event snapshot offset {start} exceeds current size {end}"
                    )
                handle.seek(start)
                return handle.read(end - start), end
