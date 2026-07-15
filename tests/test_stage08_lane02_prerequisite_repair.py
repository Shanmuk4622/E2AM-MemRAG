from __future__ import annotations

import ast
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_experiment_notebooks as notebook_builder  # noqa: E402


class Stage08Lane02PrerequisiteRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebooks = notebook_builder.build_all_notebooks()
        target = cls.notebooks["08_run_robustness_lane_02.ipynb"]
        cls.runbook_source = "".join(target["cells"][2]["source"])
        cls.prepare_source = "".join(target["cells"][7]["source"])
        cls.repair_source = cls.prepare_source.split("REQUEST = StageRequest", 1)[0]

    def test_repair_is_scoped_to_exact_stage08_lane02_notebook(self) -> None:
        marker = "STAGE08_LANE02_PREREQUISITE_REPAIR_READY"
        marked = []
        for name, notebook in self.notebooks.items():
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            if marker in code:
                marked.append(name)
        self.assertEqual(marked, ["08_run_robustness_lane_02.ipynb"])
        self.assertIn("absent or only half-created", self.runbook_source)
        self.assertIn("Do not run a different lane as a substitute", self.runbook_source)

    def test_repair_uses_the_frozen_stage07_lane02_identity(self) -> None:
        tree = ast.parse(self.repair_source)
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_stage07_request"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        call = assignments[0].value
        self.assertIsInstance(call, ast.Call)
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}

        self.assertEqual(ast.literal_eval(keywords["stage_id"]), "07")
        self.assertEqual(ast.literal_eval(keywords["stage_name"]), "evaluate_frozen_clean")
        self.assertEqual(ast.literal_eval(keywords["worker_id"]), "stage-07-lane-02")
        self.assertEqual(ast.literal_eval(keywords["lane_id"]), "lane-02")
        self.assertEqual(
            ast.literal_eval(keywords["artifact_prefix"]),
            "experiments/e2am-memrag-v3r1/stages/07/lane-02",
        )
        self.assertEqual(
            ast.literal_eval(keywords["required_gates"]),
            ("06/coordinator/ROUTER_CALIBRATION_FREEZE.json",),
        )
        self.assertEqual(
            ast.literal_eval(keywords["output_gate"]),
            "CLEAN_EVAL_LANE_SEAL.json",
        )

    def test_repair_checks_worker_pointer_not_only_branch_name(self) -> None:
        source = self.repair_source
        self.assertIn("_remote_exists as _stage08_remote_exists", source)
        self.assertIn("_make_store as _stage08_make_store", source)
        self.assertIn("worker-specific", source)
        self.assertIn("half-created", source)
        self.assertIn("_stage08_remote_exists(probe_store, HF_TOKEN)", source)

    def test_repair_seals_and_verifies_before_normal_stage08_prepare(self) -> None:
        source = self.repair_source
        self.assertLess(source.index("prepare_stage("), source.index("run_stage("))
        self.assertLess(source.index("run_stage("), source.index("finalize_stage("))
        self.assertLess(
            source.index("finalize_stage("),
            source.index("not _stage07_final.get('remote_verified', False)"),
        )
        self.assertIn("safe_stop_stage(", source)
        self.assertIn("except KeyboardInterrupt:", source)
        self.assertIn("except BaseException:", source)
        self.assertIn("_stage07_lane02_closure_exists()", source)
        self.assertIn("BRANCH_MISSING_AFTER_FINALIZE", source)
        self.assertIn("cuda.empty_cache()", source)
        self.assertIn("REQUEST = StageRequest", self.prepare_source)

    def _execute_repair(self, closure_states: list[bool]) -> tuple[list[str], str]:
        calls: list[str] = []
        states = iter(closure_states)

        fake_package = ModuleType("e2am_memrag")
        fake_package.__path__ = []  # type: ignore[attr-defined]
        fake_pipeline = ModuleType("e2am_memrag.experiment_pipeline")

        def fake_make_store(**kwargs):
            calls.append("make_store")
            self.assertEqual(kwargs["stage_id"], "07")
            self.assertEqual(kwargs["owner"], "lane-02")
            return SimpleNamespace()

        def fake_remote_exists(store, token):
            calls.append("remote_exists")
            return next(states)

        fake_pipeline._make_store = fake_make_store  # type: ignore[attr-defined]
        fake_pipeline._remote_exists = fake_remote_exists  # type: ignore[attr-defined]

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
        )

        def stage_request(**kwargs):
            calls.append("stage_request")
            return SimpleNamespace(**kwargs)

        def prepare_stage(request, *, hf_token):
            calls.append("prepare")
            return SimpleNamespace(request=request)

        def run_stage(runtime):
            calls.append("run")
            return {"gate": {"status": "PASS"}, "new_units": 3, "reused_units": 2}

        def finalize_stage(runtime, result):
            calls.append("finalize")
            return {
                "remote_verified": True,
                "output_gate": "CLEAN_EVAL_LANE_SEAL.json",
                "commit_sha": "a" * 40,
            }

        namespace = {
            "Path": Path,
            "HF_TOKEN": "synthetic-secret-never-print",
            "HF_REPO_ID": "Shanmuk4622/E2AM-MemRAG-Traces",
            "HF_REPO_TYPE": "dataset",
            "EXPERIMENT_ID": "e2am-memrag-v3r1",
            "LANE_COUNT": 4,
            "SYNC_INTERVAL_SECONDS": 1200,
            "WORK_ROOT": "/tmp/e2am-test",
            "StageRequest": stage_request,
            "prepare_stage": prepare_stage,
            "run_stage": run_stage,
            "finalize_stage": finalize_stage,
            "safe_stop_stage": lambda runtime, reason: {"remote_verified": True},
        }
        output = io.StringIO()
        with (
            patch.dict(
                sys.modules,
                {
                    "e2am_memrag": fake_package,
                    "e2am_memrag.experiment_pipeline": fake_pipeline,
                    "torch": fake_torch,
                },
            ),
            contextlib.redirect_stdout(output),
        ):
            exec(self.repair_source, namespace)
        self.assertNotIn("synthetic-secret-never-print", output.getvalue())
        return calls, output.getvalue()

    def test_missing_closure_is_rebuilt_verified_then_rechecked(self) -> None:
        calls, output = self._execute_repair([False, True])
        self.assertEqual(
            calls,
            [
                "make_store",
                "remote_exists",
                "stage_request",
                "prepare",
                "run",
                "finalize",
                "make_store",
                "remote_exists",
                "empty_cache",
            ],
        )
        self.assertIn("STAGE08_LANE02_PREREQUISITE_REPAIR_START", output)
        self.assertIn("STAGE08_LANE02_PREREQUISITE_REPAIR_COMPLETE", output)
        self.assertIn("STAGE08_LANE02_PREREQUISITE_REPAIR_READY", output)

    def test_existing_verified_closure_skips_stage07_execution(self) -> None:
        calls, output = self._execute_repair([True])
        self.assertEqual(calls, ["make_store", "remote_exists", "empty_cache"])
        self.assertIn("STAGE08_LANE02_PREREQUISITE_ALREADY_PRESENT", output)
        self.assertNotIn("STAGE08_LANE02_PREREQUISITE_REPAIR_START", output)


if __name__ == "__main__":
    unittest.main()
