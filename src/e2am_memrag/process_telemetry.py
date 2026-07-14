from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from time import monotonic, process_time
from typing import Any


def _rss_bytes() -> int | None:
    status = "/proc/self/status"
    if os.path.isfile(status):
        with open(status, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if value > 1024 * 1024 else value * 1024
    except (ImportError, OSError):
        return None


@dataclass(frozen=True)
class ProcessSummary:
    wall_seconds: float
    process_cpu_seconds: float
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_delta_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProcessSpan:
    def __init__(self) -> None:
        self._wall_start: float | None = None
        self._cpu_start: float | None = None
        self._rss_start: int | None = None
        self.result: ProcessSummary | None = None

    def start(self) -> "ProcessSpan":
        self._rss_start = _rss_bytes()
        self._cpu_start = process_time()
        self._wall_start = monotonic()
        return self

    def stop(self) -> ProcessSummary:
        if self._wall_start is None or self._cpu_start is None:
            raise RuntimeError("ProcessSpan has not been started")
        wall = monotonic() - self._wall_start
        cpu = process_time() - self._cpu_start
        rss_end = _rss_bytes()
        delta = (
            rss_end - self._rss_start
            if rss_end is not None and self._rss_start is not None
            else None
        )
        self.result = ProcessSummary(wall, cpu, self._rss_start, rss_end, delta)
        return self.result

    def __enter__(self) -> "ProcessSpan":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.stop()

