from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import huggingface_hub

from e2am_memrag import rag_engine
from e2am_memrag import experiment_pipeline
from e2am_memrag.environment import environment_contract
from e2am_memrag.rag_engine import MODEL_CATALOG, ROUTES, ensure_model_snapshot


PINNED_MODEL_REPOS = {
    "tiny": "Qwen/Qwen3-0.6B",
    "small": "ibm-granite/granite-4.0-1b",
    "granite": "ibm-granite/granite-4.1-3b",
    "peer": "HuggingFaceTB/SmolLM3-3B",
    "upper": "Qwen/Qwen3-4B-Instruct-2507",
}


class _FakeResponse:
    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        self.url = url
        self.headers: dict[str, str] = {}


class _FakeHubError(Exception):
    def __init__(self, message: str, response: _FakeResponse) -> None:
        super().__init__(message)
        self.response = response


class V3ModelPanelContractTests(unittest.TestCase):
    def test_model_catalog_is_pinned_and_separates_online_pair(self) -> None:
        self.assertEqual(set(MODEL_CATALOG), set(PINNED_MODEL_REPOS))
        self.assertEqual(
            {key: value["repo_id"] for key, value in MODEL_CATALOG.items()},
            PINNED_MODEL_REPOS,
        )
        for model in MODEL_CATALOG.values():
            self.assertRegex(str(model["revision"]), r"^[0-9a-f]{40}$")
        self.assertEqual(
            {
                key
                for key, value in MODEL_CATALOG.items()
                if value["online_candidate"]
            },
            {"tiny", "small"},
        )

    def test_routes_have_fixed_unique_lanes_and_fair_model_pairs(self) -> None:
        self.assertEqual(len(ROUTES), 22)
        self.assertEqual(len({route.route_id for route in ROUTES}), 22)
        self.assertEqual(
            {
                lane: sum(route.lane == lane for route in ROUTES)
                for lane in {route.lane for route in ROUTES}
            },
            {"lane-a": 7, "lane-b": 4, "lane-c": 6, "lane-d": 5},
        )

        for model_key in PINNED_MODEL_REPOS:
            direct = [
                route
                for route in ROUTES
                if route.generator == model_key
                and route.knowledge == "none"
                and route.memory == "none"
                and not route.verifier
            ]
            grounded = [
                route
                for route in ROUTES
                if route.generator == model_key
                and route.knowledge == "hybrid"
                and route.memory == "graph"
                and route.verifier
            ]
            self.assertTrue(direct, f"{model_key} is missing its direct control")
            self.assertTrue(grounded, f"{model_key} is missing its grounded control")

        reference_models = set(PINNED_MODEL_REPOS) - {"tiny", "small"}
        self.assertTrue(any(route.generator in reference_models for route in ROUTES))
        for route in ROUTES:
            if route.generator in reference_models:
                self.assertTrue(route.offline_only)
            else:
                self.assertFalse(route.offline_only)

    def test_transformer_loader_uses_dtype_not_deprecated_torch_dtype(self) -> None:
        source = inspect.getsource(rag_engine)
        self.assertNotIn("torch_dtype", source)
        loader_source = inspect.getsource(rag_engine.TransformersGenerator._load)
        self.assertIn("local_files_only", loader_source)
        self.assertIn("use_safetensors", loader_source)

    def test_stage00_freezes_metadata_without_loading_model_weights(self) -> None:
        source = inspect.getsource(experiment_pipeline._stage00)
        self.assertIn("_freeze_model_metadata", source)
        self.assertNotIn("prepare_online_pair", source)
        self.assertNotIn("TransformersGenerator(", source)

    def test_clean_test_is_route_owned_and_lane00_owns_policy_decisions(self) -> None:
        source = inspect.getsource(experiment_pipeline._stage07)
        self.assertIn("routes_for_lane", source)
        self.assertIn('runtime.request.lane_id == "lane-00"', source)
        self.assertNotIn("_lane_partition", source)

    def test_snapshot_rejects_unpinned_revision_before_download(self) -> None:
        download_calls = []

        def forbidden_download(**kwargs):
            download_calls.append(kwargs)
            raise AssertionError("unpinned revision reached snapshot_download")

        with (
            patch.dict(
                ensure_model_snapshot.__globals__,
                {"snapshot_download": forbidden_download},
            ),
            patch.object(huggingface_hub, "snapshot_download", forbidden_download),
        ):
            with self.assertRaises((TypeError, ValueError), msg="revision must be a commit SHA"):
                ensure_model_snapshot(
                    PINNED_MODEL_REPOS["tiny"],
                    "main",
                    cache_dir=Path(tempfile.gettempdir()),
                )
        self.assertEqual(download_calls, [])

    def test_snapshot_download_is_public_pinned_and_bounded(self) -> None:
        revision = "a" * 40
        captured: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"test-snapshot")

            def fake_download(**kwargs):
                captured.append(kwargs)
                return str(snapshot)

            with (
                patch.dict(
                    ensure_model_snapshot.__globals__,
                    {"snapshot_download": fake_download},
                ),
                patch.object(huggingface_hub, "snapshot_download", fake_download),
            ):
                resolved = ensure_model_snapshot(
                    PINNED_MODEL_REPOS["tiny"],
                    revision,
                    cache_dir=root / "cache",
                )

        self.assertEqual(Path(resolved), snapshot)
        self.assertEqual(len(captured), 1)
        call = captured[0]
        self.assertEqual(call["repo_id"], PINNED_MODEL_REPOS["tiny"])
        self.assertEqual(call["revision"], revision)
        self.assertIs(call["token"], False)
        self.assertTrue(call.get("allow_patterns"))

    def test_transient_public_signed_url_403_refreshes_without_losing_cache(self) -> None:
        revision = "b" * 40
        calls = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"complete")

            def flaky_download(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise _FakeHubError(
                        "403 SignatureError: invalid key pair id from xet-bridge",
                        _FakeResponse(
                            403,
                            "https://us.gcp.cdn.hf.co/xet-bridge-us/blob",
                        ),
                    )
                return str(snapshot)

            with (
                patch.dict(
                    ensure_model_snapshot.__globals__,
                    {"snapshot_download": flaky_download},
                ),
                patch("e2am_memrag.rag_engine.time.sleep") as sleep,
            ):
                resolved = ensure_model_snapshot(
                    PINNED_MODEL_REPOS["tiny"],
                    revision,
                    cache_dir=root / "cache",
                    max_attempts=2,
                )

        self.assertEqual(Path(resolved), snapshot)
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(5.0)

    def test_generic_public_403_remains_non_retryable(self) -> None:
        revision = "c" * 40
        calls = []

        def forbidden_download(**kwargs):
            calls.append(kwargs)
            raise _FakeHubError(
                "403 Forbidden: repository access denied",
                _FakeResponse(403, "https://huggingface.co/private/repo"),
            )

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                ensure_model_snapshot.__globals__,
                {"snapshot_download": forbidden_download},
            ):
                with self.assertRaisesRegex(RuntimeError, "will not retry"):
                    ensure_model_snapshot(
                        PINNED_MODEL_REPOS["tiny"],
                        revision,
                        cache_dir=Path(temporary),
                        max_attempts=5,
                    )

        self.assertEqual(len(calls), 1)

    def test_environment_contract_tracks_scientific_and_io_stack(self) -> None:
        required = {
            "numpy",
            "scipy",
            "scikit-learn",
            "joblib",
            "tokenizers",
            "safetensors",
        }
        with patch("e2am_memrag.environment._nvidia_smi_contract", return_value=None):
            packages = environment_contract()["packages"]
        self.assertTrue(required <= set(packages))


if __name__ == "__main__":
    unittest.main()
