from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any


@dataclass(frozen=True)
class EnergySummary:
    available: bool
    gpu_index: int
    gpu_uuid: str | None
    duration_seconds: float
    energy_joules: float | None
    samples: int
    average_power_watts: float | None
    peak_power_watts: float | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUEnergySampler:
    """Integrates NVML board power. This is GPU energy, not whole-system energy."""

    def __init__(
        self,
        physical_gpu_index: int = 0,
        interval_seconds: float = 0.2,
        *,
        expected_gpu_uuid: str | None = None,
        minimum_samples: int = 10,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        self.gpu_index = physical_gpu_index
        self.interval_seconds = interval_seconds
        self.expected_gpu_uuid = expected_gpu_uuid
        self.minimum_samples = minimum_samples
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float]] = []
        self._error: str | None = None
        self._started: float | None = None
        self._stopped: float | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._gpu_uuid: str | None = None

    def start(self) -> "GPUEnergySampler":
        if self._thread is not None:
            raise RuntimeError("Sampler has already been started")
        self._started = monotonic()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            raw_uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            self._gpu_uuid = (
                raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else str(raw_uuid)
            )
            if self.expected_gpu_uuid and self._gpu_uuid != self.expected_gpu_uuid:
                raise RuntimeError(
                    "NVML device UUID does not match the GPU selected during preflight: "
                    f"expected {self.expected_gpu_uuid}, found {self._gpu_uuid}"
                )
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            self._stopped = monotonic()
            return self
        self._thread = threading.Thread(target=self._sample_loop, name="e2am-nvml", daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        milliwatts = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
        self._samples.append((monotonic(), float(milliwatts) / 1000.0))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                break
            self._stop.wait(self.interval_seconds)

    def stop(self, synchronize_cuda: bool = False) -> EnergySummary:
        if synchronize_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
            try:
                self._sample_once()
            except Exception:
                pass
        self._stopped = monotonic()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self.summary()

    def summary(self) -> EnergySummary:
        started = self._started or monotonic()
        stopped = self._stopped or monotonic()
        duration = max(0.0, stopped - started)
        if len(self._samples) < self.minimum_samples:
            return EnergySummary(
                available=False,
                gpu_index=self.gpu_index,
                gpu_uuid=self._gpu_uuid,
                duration_seconds=duration,
                energy_joules=None,
                samples=len(self._samples),
                average_power_watts=None,
                peak_power_watts=None,
                reason=self._error
                or f"Fewer than {self.minimum_samples} NVML samples were collected",
            )
        energy = 0.0
        for (time_a, power_a), (time_b, power_b) in zip(self._samples, self._samples[1:]):
            energy += (time_b - time_a) * (power_a + power_b) / 2.0
        powers = [power for _, power in self._samples]
        sample_duration = self._samples[-1][0] - self._samples[0][0]
        average = energy / sample_duration if sample_duration > 0 else None
        return EnergySummary(
            available=self._error is None,
            gpu_index=self.gpu_index,
            gpu_uuid=self._gpu_uuid,
            duration_seconds=duration,
            energy_joules=energy,
            samples=len(self._samples),
            average_power_watts=average,
            peak_power_watts=max(powers),
            reason=self._error,
        )

    def __enter__(self) -> "GPUEnergySampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.stop(synchronize_cuda=True)
