from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "10_consolidate_verified_hf_release.ipynb"
RUNTIME_PATH = ROOT / "scripts" / "e2am_consolidation_runtime.py"
BUILDER_PATH = ROOT / "scripts" / "build_consolidation_notebook.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConsolidationNotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_module("e2am_consolidation_runtime_test", RUNTIME_PATH)
        cls.builder = load_module("build_consolidation_notebook_test", BUILDER_PATH)
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_exact_builder_output(self) -> None:
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertEqual(self.notebook, self.builder.build_notebook(runtime_source))

    def test_every_code_cell_parses_and_notebook_is_clean(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"], index)
            self.assertEqual(cell["outputs"], [], index)
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")

    def test_frozen_release_inventory(self) -> None:
        lock = self.builder.SOURCE_RELEASE_LOCK
        self.assertEqual(len(lock), 23)
        self.assertEqual(len({item["branch"] for item in lock}), 23)
        self.assertEqual(sum(item["artifact_records"] for item in lock), 282)
        self.assertEqual(sum(item["artifact_bytes"] for item in lock), 127_554_473)
        for item in lock:
            self.assertEqual(
                item["branch"],
                f"stage-e2am-memrag-v3r1-{item['stage_id']}-"
                f"stage-{item['stage_id']}-{item['owner']}",
            )

    def test_non_destructive_publication_contract(self) -> None:
        source = self.all_source
        self.assertIn("CommitOperationAdd", source)
        self.assertNotIn("CommitOperationDelete", source)
        self.assertNotIn("delete_branch(", source)
        self.assertNotIn("delete_repo(", source)
        self.assertNotIn("force_push", source)
        self.assertIn("Frozen RELEASE.json must never be overwritten", source)
        self.assertIn("source_branches_modified", source)
        self.assertIn("excluded_legacy_branches", source)
        self.assertIn("consolidated-e2am-memrag-v3r1", source)

    def test_kaggle_secret_rate_and_resume_contract(self) -> None:
        source = self.all_source
        self.assertIn('get_secret("HF_TOKEN")', source)
        self.assertIn('"hub_capacity": 96', source)
        self.assertIn('"dirty_sync_target_seconds": 1200', source)
        self.assertIn("PROGRESS.json", source)
        self.assertIn("SAFE_STOP_VERIFIED", source)
        self.assertIn("SECOND_WRITER_STOP", source)
        self.assertIn("HF_HUB_DISABLE_XET", source)
        self.assertIn("token: str | bool = self.hf_token", source)
        self.assertNotIn("False if public_read else self.hf_token", source)
        self.assertNotIn("huggingface-hub==0.36.2", source)

    def test_notebook_defaults_to_bounded_fast_paper_release(self) -> None:
        source = self.all_source
        self.assertIn(
            "FINAL_REPORT = run_fast_paper_release(CONFIG, hf_token=HF_TOKEN)",
            source,
        )
        self.assertNotIn(
            "FINAL_REPORT = run_consolidation(CONFIG, hf_token=HF_TOKEN)",
            source,
        )
        self.assertIn("FAST_PAPER_RELEASE_COMPLETE", source)
        self.assertIn('"paper_artifact_records"] == 11', source)
        self.assertIn('"paper_artifact_bytes"] == 11_528_142', source)
        self.assertIn("full_archive_progress_preserved", source)

    def test_release_lock_constructs_and_capacity_guard_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "repo_id": "Shanmuk4622/E2AM-MemRAG-Traces",
                "experiment_id": "e2am-memrag-v3r1",
                "destination_branch": "consolidated-e2am-memrag-v3r1",
                "remote_root": "consolidated/e2am-memrag-v3r1",
                "work_root": temporary,
                "hub_capacity": 96,
                "dirty_sync_target_seconds": 1200,
                "expected_artifact_records": 282,
                "expected_artifact_bytes": 127_554_473,
                "source_release_lock": self.builder.SOURCE_RELEASE_LOCK,
                "release_pointer_lock": {},
            }
            instance = self.runtime.Consolidator(config, hf_token="test-token")
            self.assertEqual(len(instance.expected_branches), 23)
            self.assertEqual(instance.budget.capacity, 96)
        with self.assertRaises(ValueError):
            self.runtime.RollingHubBudget(capacity=97)

    def test_old_budget_events_are_normalized_on_resume(self) -> None:
        budget = self.runtime.RollingHubBudget(capacity=96, clock=lambda: 1000.0)
        budget.restore([[999.0, 2], [999.0, 16], [999.0, 4]])
        self.assertEqual(budget.used, 5)

    def test_deterministic_archive_and_safe_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "nested").mkdir(parents=True)
            (source / "a.txt").write_bytes(b"alpha\n")
            (source / "nested" / "b.json").write_bytes(b'{"b":2}\n')
            first = self.runtime.deterministic_zip(source, root / "first.zip")
            second = self.runtime.deterministic_zip(source, root / "second.zip")
            self.assertEqual(first["sha256"], second["sha256"])
            restored = root / "restored"
            self.runtime.extract_verified_zip(root / "first.zip", restored)
            self.assertEqual((restored / "a.txt").read_bytes(), b"alpha\n")
            self.assertEqual(
                (restored / "nested" / "b.json").read_bytes(), b'{"b":2}\n'
            )

    def test_equal_remote_commit_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "repo_id": "Shanmuk4622/E2AM-MemRAG-Traces",
                "experiment_id": "e2am-memrag-v3r1",
                "destination_branch": "consolidated-e2am-memrag-v3r1",
                "remote_root": "consolidated/e2am-memrag-v3r1",
                "work_root": temporary,
                "hub_capacity": 96,
                "dirty_sync_target_seconds": 1200,
                "expected_artifact_records": 282,
                "expected_artifact_bytes": 127_554_473,
                "source_release_lock": self.builder.SOURCE_RELEASE_LOCK,
                "release_pointer_lock": {},
            }
            instance = self.runtime.Consolidator(config, hf_token="test-token")
            local = Path(temporary) / "payload.json"
            local.write_bytes(b"{}\n")
            parent = "a" * 40
            instance._repo_head = lambda revision: parent
            instance._try_download = lambda **kwargs: b"{}\n"
            instance.CommitOperationAdd = lambda **kwargs: self.fail(
                "an equal remote file must not create a commit"
            )
            result = instance._commit_verified(
                revision=instance.destination_branch,
                parent_commit=parent,
                files={"consolidated/test.json": local},
                message="idempotency test",
            )
            self.assertEqual(result, parent)

    def test_public_403_remains_refreshable_but_api_403_safe_stops(self) -> None:
        class Response:
            status_code = 403
            headers = {}

        class Forbidden(Exception):
            response = Response()

        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "repo_id": "Shanmuk4622/E2AM-MemRAG-Traces",
                "experiment_id": "e2am-memrag-v3r1",
                "destination_branch": "consolidated-e2am-memrag-v3r1",
                "remote_root": "consolidated/e2am-memrag-v3r1",
                "work_root": temporary,
                "hub_capacity": 96,
                "dirty_sync_target_seconds": 1200,
                "expected_artifact_records": 282,
                "expected_artifact_bytes": 127_554_473,
                "source_release_lock": self.builder.SOURCE_RELEASE_LOCK,
                "release_pointer_lock": {},
            }
            instance = self.runtime.Consolidator(config, hf_token="test-token")
            with self.assertRaises(Forbidden):
                instance._call(
                    lambda: (_ for _ in ()).throw(Forbidden("signed URL")),
                    weight=1,
                    reason="public-download-test",
                    public_download=True,
                )
            with self.assertRaisesRegex(RuntimeError, "HUB_AUTHENTICATION_STOP"):
                instance._call(
                    lambda: (_ for _ in ()).throw(Forbidden("API auth")),
                    weight=1,
                    reason="api-test",
                )


if __name__ == "__main__":
    unittest.main()
