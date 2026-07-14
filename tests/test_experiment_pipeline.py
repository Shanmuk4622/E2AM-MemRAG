from __future__ import annotations

import unittest
import tempfile
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from e2am_memrag.experiment_pipeline import (
    StageRequest,
    _bundle_sources,
    _expected_unit_id,
    _extract_verified_bundle,
    _verify_frozen_runtime_contract,
    _verify_stage00_runtime_contract,
    _write_verified_bundle,
)
from e2am_memrag.rag_engine import RouteSpec


def _request(**overrides):
    values = {
        "experiment_id": "e2am-memrag-v3r1",
        "stage_id": "03",
        "stage_name": "pilot_routes",
        "role": "lane",
        "worker_id": "stage-03-lane-00",
        "lane_id": "lane-00",
        "lane_count": 4,
        "notebook_name": "03_pilot_routes_lane_00.ipynb",
        "hf_repo_id": "Shanmuk4622/E2AM-MemRAG-Traces",
        "hf_repo_type": "dataset",
        "hf_revision": "stage-e2am-memrag-v3r1-03-stage-03-lane-00",
        "artifact_prefix": "experiments/e2am-memrag-v3r1/stages/03/lane-00",
        "required_gates": ("02/coordinator/HYBRIDBENCH_FREEZE.json",),
        "output_gate": "PILOT_LANE_SEAL.json",
        "sync_interval_seconds": 1200,
        "work_root": "/kaggle/working/e2am_memrag",
        "stage_work_items": ("pilot",),
    }
    values.update(overrides)
    return StageRequest(**values)


class ExperimentPipelineContractTests(unittest.TestCase):
    def test_stage00_source_zip_must_match_embedded_hash(self) -> None:
        environment = {"python_version": "3.test", "packages": {"torch": "test"}}
        with (
            patch.dict(os.environ, {"E2AM_SOURCE_TREE_SHA256": "a" * 64}),
            patch(
                "e2am_memrag.experiment_pipeline.source_fingerprint",
                return_value={"tree_sha256": "b" * 64},
            ),
            patch(
                "e2am_memrag.experiment_pipeline.environment_contract",
                return_value=environment,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "source ZIP"):
                _verify_stage00_runtime_contract()

    def test_downstream_runtime_must_match_stage00_exactly(self) -> None:
        frozen_environment = {
            "python_version": "3.test",
            "packages": {"torch": "2.test"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            stage00 = Path(temporary)
            (stage00 / "experiment_spec.json").write_text(
                json.dumps(
                    {
                        "source": {"tree_sha256": "a" * 64},
                        "environment": frozen_environment,
                    }
                ),
                encoding="utf-8",
            )
            runtime = SimpleNamespace()
            with (
                patch.dict(os.environ, {"E2AM_SOURCE_TREE_SHA256": "a" * 64}),
                patch(
                    "e2am_memrag.experiment_pipeline._restore_stage",
                    return_value=stage00,
                ),
                patch(
                    "e2am_memrag.experiment_pipeline.environment_contract",
                    return_value=frozen_environment,
                ),
            ):
                report = _verify_frozen_runtime_contract(runtime)
                self.assertEqual(report["comparison"], "exact-stage-00-match")

            changed_environment = {
                **frozen_environment,
                "packages": {"torch": "different"},
            }
            with (
                patch.dict(os.environ, {"E2AM_SOURCE_TREE_SHA256": "a" * 64}),
                patch(
                    "e2am_memrag.experiment_pipeline._restore_stage",
                    return_value=stage00,
                ),
                patch(
                    "e2am_memrag.experiment_pipeline.environment_contract",
                    return_value=changed_environment,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime differs"):
                    _verify_frozen_runtime_contract(runtime)

    def test_fixed_lane_identity_is_accepted(self) -> None:
        self.assertEqual(_request().lane_id, "lane-00")

    def test_wrong_role_or_remote_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed role"):
            _request(role="coordinator", lane_id=None, worker_id="stage-03-coordinator")
        with self.assertRaisesRegex(ValueError, "artifact_prefix"):
            _request(artifact_prefix="experiments/wrong")

    def test_unit_id_binds_spec_and_full_route_configuration(self) -> None:
        base = RouteSpec("X", "lane-a", "tiny", "bm25", "none", top_k=4)
        changed = RouteSpec("X", "lane-a", "tiny", "bm25", "none", top_k=8)
        first = _expected_unit_id("q", base, "a" * 64)
        self.assertNotEqual(first, _expected_unit_id("q", base, "b" * 64))
        self.assertNotEqual(first, _expected_unit_id("q", changed, "a" * 64))

    def test_verified_transport_bundle_is_deterministic_and_round_trips(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.staged = {}

            def put_file(self, logical_path, path):
                self.staged[logical_path] = Path(path).read_bytes()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            first = root / "first.txt"
            second = root / "second.bin"
            first.write_text("alpha\n", encoding="utf-8")
            second.write_bytes(b"\x00\x01payload")
            runtime = SimpleNamespace(
                request=SimpleNamespace(experiment_id="e2am-memrag-v3r1", stage_id="03"),
                output_root=output,
                store=FakeStore(),
            )
            kwargs = {
                "bundle_name": "unit.zip",
                "manifest_name": "UNIT_MANIFEST.json",
                "bundle_kind": "unit-test",
                "sources": {"a/first.txt": first, "b/second.bin": second},
            }
            one = _write_verified_bundle(runtime, **kwargs)
            first_bytes = (output / "unit.zip").read_bytes()
            two = _write_verified_bundle(runtime, **kwargs)
            self.assertEqual(one["sha256"], two["sha256"])
            self.assertEqual(first_bytes, (output / "unit.zip").read_bytes())
            restored = _extract_verified_bundle(
                output / "unit.zip",
                root / "restored",
                manifest_name="UNIT_MANIFEST.json",
                bundle_kind="unit-test",
                experiment_id="e2am-memrag-v3r1",
            )
            sources = _bundle_sources(restored, "UNIT_MANIFEST.json")
            self.assertEqual(sources["a/first.txt"].read_text(encoding="utf-8"), "alpha\n")
            self.assertEqual(sources["b/second.bin"].read_bytes(), b"\x00\x01payload")

    def test_verified_transport_bundle_rejects_wrong_identity(self) -> None:
        class FakeStore:
            def put_file(self, logical_path, path):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            source = root / "source.txt"
            source.write_text("bound", encoding="utf-8")
            runtime = SimpleNamespace(
                request=SimpleNamespace(experiment_id="experiment-a", stage_id="03"),
                output_root=output,
                store=FakeStore(),
            )
            _write_verified_bundle(
                runtime,
                bundle_name="unit.zip",
                manifest_name="UNIT_MANIFEST.json",
                bundle_kind="unit-test",
                sources={"source.txt": source},
            )
            with self.assertRaisesRegex(RuntimeError, "identity"):
                _extract_verified_bundle(
                    output / "unit.zip",
                    root / "restored",
                    manifest_name="UNIT_MANIFEST.json",
                    bundle_kind="unit-test",
                    experiment_id="experiment-b",
                )


if __name__ == "__main__":
    unittest.main()
