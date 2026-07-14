from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Any


class StopController:
    """Signal handlers only set a flag; cleanup runs at a safe Python boundary."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None
        self._previous: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, reason: str = "manual") -> None:
        if not self.requested:
            self.reason = reason
        self._event.set()

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        del frame
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.request(f"signal:{name}")

    def install(self) -> "StopController":
        for candidate in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[candidate] = signal.getsignal(candidate)
                signal.signal(candidate, self._handler)
            except (OSError, RuntimeError, ValueError):
                continue
        return self

    def restore(self) -> None:
        for candidate, handler in self._previous.items():
            try:
                signal.signal(candidate, handler)
            except (OSError, RuntimeError, ValueError):
                pass
        self._previous.clear()

