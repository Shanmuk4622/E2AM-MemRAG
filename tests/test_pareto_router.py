from __future__ import annotations

import math
import unittest

from e2am_memrag.pareto_router import RouterBundle, SeedModels


class _Success:
    def predict_proba(self, rows):
        # Route one-hot columns are the final two values.
        return [[0.9, 0.1] if row[-2] else [0.05, 0.95] for row in rows]


class _ConstantSuccess:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, rows):
        return [[1.0 - self.probability, self.probability] for _ in rows]


class _IdentityCalibration:
    def predict(self, values):
        return list(values)


class _Cost:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):
        return [math.log(self.value) for _ in rows]


def _bundle() -> RouterBundle:
    models = [SeedModels(_Success(), _IdentityCalibration(), _Cost(2.0), _Cost(1.0))]
    return RouterBundle(
        schema_version=2,
        route_ids=("A00_tiny_direct", "A01_tiny_bm25"),
        seeds=(4622,),
        models=models,
        tau=0.8,
        latency_limit_seconds=12.0,
        safe_route_id="A01_tiny_bm25",
        include_probe=True,
        training_hash="a" * 64,
    )


class ParetoRouterTests(unittest.TestCase):
    def test_stage_zero_requests_probe_without_silently_using_zero_probe(self) -> None:
        router = _bundle()
        first = router.choose({"query_tokens": 4.0})
        self.assertTrue(first["probe_required"])
        self.assertIsNone(first["chosen"])

        second = router.choose(
            {"query_tokens": 4.0},
            probe_features={"doc_top": 2.0, "doc_count": 4.0},
        )
        self.assertEqual(second["stage"], 1)
        self.assertEqual(second["chosen"]["route_id"], "A01_tiny_bm25")

    def test_explicit_empty_candidates_stay_empty(self) -> None:
        self.assertEqual(_bundle().predict_actions({}, candidate_route_ids=[]), [])

    def test_non_finite_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _bundle().predict_actions({"query_tokens": float("nan")})

    def test_success_gate_uses_worst_calibrated_bootstrap_seed(self) -> None:
        models = [
            SeedModels(
                _ConstantSuccess(0.93),
                _IdentityCalibration(),
                _Cost(2.0),
                _Cost(1.0),
            ),
            SeedModels(
                _ConstantSuccess(0.71),
                _IdentityCalibration(),
                _Cost(3.0),
                _Cost(1.5),
            ),
        ]
        router = RouterBundle(
            schema_version=2,
            route_ids=("A00_tiny_direct",),
            seeds=(4622, 1701),
            models=models,
            tau=0.8,
            latency_limit_seconds=12.0,
            safe_route_id="A00_tiny_direct",
            include_probe=False,
            training_hash="a" * 64,
        )

        action = router.predict_actions({"query_tokens": 4.0})[0]

        self.assertAlmostEqual(action["success_probability"], 0.82)
        self.assertAlmostEqual(action["success_lower_bound"], 0.71)
        self.assertEqual(
            action["success_lower_bound_method"],
            "minimum calibrated grouped-bootstrap seed",
        )
        self.assertEqual(action["cost_prediction_quantile"], 0.90)

    def test_bundle_rejects_offline_or_inconsistent_safe_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "model/seed"):
            RouterBundle(
                schema_version=1,
                route_ids=("A00_tiny_direct", "A01_tiny_bm25"),
                seeds=(1,),
                models=[],
                tau=0.8,
                latency_limit_seconds=1.0,
                safe_route_id="A01_tiny_bm25",
                include_probe=True,
                training_hash="x",
            )


if __name__ == "__main__":
    unittest.main()
