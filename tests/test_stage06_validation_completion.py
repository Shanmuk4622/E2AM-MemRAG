from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import huggingface_hub
import huggingface_hub.constants as hf_constants
from e2am_memrag import experiment_pipeline, pareto_router
from e2am_memrag.pareto_router import SeedModels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_experiment_notebooks as notebook_builder  # noqa: E402


class _RouteSuccess:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, rows):
        return [[1.0 - self.probability, self.probability] for _ in rows]


class _IdentityCalibration:
    def predict(self, values):
        return list(values)


class _ConstantCost:
    def predict(self, rows):
        return [math.log(1.0) for _ in rows]


class _SignatureResponse:
    status_code = 403
    url = "https://us.gcp.cdn.hf.co/xet-bridge-us/private-artifact"
    headers: dict[str, str] = {}


class _SignatureError(Exception):
    def __init__(self) -> None:
        super().__init__("403 SignatureError: invalid key pair id")
        self.response = _SignatureResponse()


def _trace(query_id: str, route_id: str, *, success: bool) -> dict:
    return {
        "unit_id": f"{query_id}:{route_id}",
        "query_id": query_id,
        "route_id": route_id,
        "status": "SUCCESS",
        "success": success,
        "features": {},
        "generation": {
            "energy": {"available": True, "energy_joules": 1.0},
        },
        "answer": {"abstain": False},
        "total_seconds": 1.0,
        "probe_seconds": 0.0,
        "spec_hash": "a" * 64,
    }


def _matrix(*, safe_route_success: bool) -> tuple[list[dict], dict[str, str]]:
    traces = []
    splits = {}
    for split in ("calibration", "validation"):
        for index in range(20):
            query_id = f"{split}-{index:02d}"
            splits[query_id] = split
            traces.append(_trace(query_id, "A00_tiny_direct", success=False))
            traces.append(
                _trace(
                    query_id,
                    "A01_tiny_bm25",
                    success=safe_route_success,
                )
            )
    return traces, splits


class Stage06ValidationCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebooks = notebook_builder.build_all_notebooks()
        stage06 = cls.notebooks["06_train_and_calibrate_router.ipynb"]
        prepare_source = "".join(stage06["cells"][7]["source"])
        cls.patch_source = prepare_source.split("REQUEST = StageRequest", 1)[0]
        cls.artifact_patch_source = cls.patch_source.split(
            "STAGE06_PROTOCOL_AMENDMENT", 1
        )[0]

    def setUp(self) -> None:
        self.original_router_fit = pareto_router.fit_router
        self.original_pipeline_fit = experiment_pipeline.fit_router
        self.original_write_json = experiment_pipeline._write_json
        self.original_hub_download = huggingface_hub.hf_hub_download
        self.original_disable_xet = hf_constants.HF_HUB_DISABLE_XET
        self.original_disable_xet_env = os.environ.get("HF_HUB_DISABLE_XET")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.patch_source, {"os": os})

    def tearDown(self) -> None:
        pareto_router.fit_router = self.original_router_fit
        experiment_pipeline.fit_router = self.original_pipeline_fit
        experiment_pipeline._write_json = self.original_write_json
        huggingface_hub.hf_hub_download = self.original_hub_download
        hf_constants.HF_HUB_DISABLE_XET = self.original_disable_xet
        if self.original_disable_xet_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = self.original_disable_xet_env
        pareto_router.__dict__.pop("_STAGE06_VALIDATION_SELECTION", None)
        pareto_router.__dict__.pop("STAGE06_PROTOCOL_AMENDMENT", None)

    @staticmethod
    def _seed_model() -> SeedModels:
        return SeedModels(
            _RouteSuccess(0.90),
            _IdentityCalibration(),
            _ConstantCost(),
            _ConstantCost(),
        )

    def test_expanded_grid_selects_conservative_feasible_threshold(self) -> None:
        traces, splits = _matrix(safe_route_success=True)
        bundle = experiment_pipeline.fit_router(
            traces,
            splits,
            route_ids=("A00_tiny_direct", "A01_tiny_bm25"),
            seeds=(1,),
            include_probe=False,
            pretrained_seed_models={1: self._seed_model()},
        )

        self.assertEqual(bundle.tau, 0.95)
        self.assertTrue(bundle.validation_selection["feasible"])
        self.assertEqual(
            bundle.validation_selection["selection_mode"],
            "lowest-feasible-frozen-grid",
        )

    def test_infeasible_validation_freezes_policy_instead_of_crashing(self) -> None:
        traces, splits = _matrix(safe_route_success=False)
        bundle = experiment_pipeline.fit_router(
            traces,
            splits,
            route_ids=("A00_tiny_direct", "A01_tiny_bm25"),
            seeds=(1,),
            include_probe=False,
            pretrained_seed_models={1: self._seed_model()},
        )

        self.assertEqual(bundle.tau, 1.0)
        self.assertFalse(bundle.validation_selection["feasible"])
        self.assertFalse(
            bundle.validation_selection["protocol_amendment"][
                "hypothesis_claim_allowed_when_infeasible"
            ]
        )

    def test_invalid_authenticated_signature_falls_back_to_xet_then_restores_http(self) -> None:
        calls = []

        def fake_download(**kwargs):
            calls.append((hf_constants.HF_HUB_DISABLE_XET, dict(kwargs)))
            if hf_constants.HF_HUB_DISABLE_XET:
                raise _SignatureError()
            return "/verified/xet/artifact"

        hf_constants.HF_HUB_DISABLE_XET = True
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        output = io.StringIO()
        with (
            patch.object(huggingface_hub, "hf_hub_download", fake_download),
            contextlib.redirect_stdout(output),
        ):
            exec(self.artifact_patch_source, {"os": os})
            result = huggingface_hub.hf_hub_download(
                repo_id="private/dataset",
                repo_type="dataset",
                revision="a" * 40,
                filename="artifacts/sha256/object",
                token="synthetic-secret-never-print",
            )

        self.assertEqual(result, "/verified/xet/artifact")
        self.assertEqual([disabled for disabled, _ in calls], [True, False])
        self.assertTrue(calls[1][1]["force_download"])
        self.assertTrue(hf_constants.HF_HUB_DISABLE_XET)
        self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
        self.assertIn("STAGE06_ARTIFACT_XET_FALLBACK", output.getvalue())
        self.assertNotIn("synthetic-secret-never-print", output.getvalue())

    def test_stage09_guard_is_scoped_to_release_notebook(self) -> None:
        marker = "STAGE09_VALIDATION_GUARD_READY"
        marked = []
        for name, notebook in self.notebooks.items():
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            if marker in code:
                marked.append(name)
        self.assertEqual(marked, ["09_aggregate_audit_and_release.ipynb"])


if __name__ == "__main__":
    unittest.main()
