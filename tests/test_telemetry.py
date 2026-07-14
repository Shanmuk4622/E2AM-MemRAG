from __future__ import annotations

import unittest

from e2am_memrag.telemetry import GPUEnergySampler


class EnergySamplerTests(unittest.TestCase):
    def test_scientific_smoke_requires_ten_samples_by_default(self) -> None:
        sampler = GPUEnergySampler(interval_seconds=0.2)
        sampler._started = 1.0
        sampler._stopped = 2.0
        sampler._gpu_uuid = "GPU-test"
        sampler._samples = [(1.0 + index * 0.1, 50.0) for index in range(9)]
        summary = sampler.summary()
        self.assertFalse(summary.available)
        self.assertIn("10", str(summary.reason))

    def test_integration_reports_uuid_and_positive_energy(self) -> None:
        sampler = GPUEnergySampler(interval_seconds=0.2)
        sampler._started = 1.0
        sampler._stopped = 3.0
        sampler._gpu_uuid = "GPU-test"
        sampler._samples = [(1.0 + index * 0.2, 50.0) for index in range(11)]
        summary = sampler.summary()
        self.assertTrue(summary.available)
        self.assertEqual(summary.gpu_uuid, "GPU-test")
        self.assertAlmostEqual(summary.energy_joules or 0.0, 100.0, places=5)


if __name__ == "__main__":
    unittest.main()
