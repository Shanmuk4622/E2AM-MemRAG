from __future__ import annotations

import contextlib
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_experiment_notebooks as notebook_builder  # noqa: E402


class Lane01DependencyBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebooks = notebook_builder.build_all_notebooks()
        lane01 = cls.notebooks["05_collect_training_traces_lane_01.ipynb"]
        bootstrap_source = "".join(lane01["cells"][4]["source"])
        start = "# LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1"
        end = "# END_LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1"
        cls.bootstrap_snippet = bootstrap_source[
            bootstrap_source.index(start) : bootstrap_source.index(end) + len(end)
        ]

    def test_resilient_bootstrap_is_scoped_to_stage05_lane01(self) -> None:
        marker = "LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1"
        marked = []
        for name, notebook in self.notebooks.items():
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            if marker in code:
                marked.append(name)
        self.assertEqual(marked, ["05_collect_training_traces_lane_01.ipynb"])

    def test_satisfied_environment_skips_pip_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            (project_root / "requirements-kaggle.txt").write_text(
                "pip>=0\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch.object(subprocess, "run") as run,
                contextlib.redirect_stdout(output),
            ):
                exec(
                    self.bootstrap_snippet,
                    {
                        "project_root": project_root,
                        "subprocess": subprocess,
                        "sys": sys,
                        "importlib": importlib,
                    },
                )

        run.assert_not_called()
        self.assertIn("DEPENDENCY_INSTALL_SKIPPED", output.getvalue())

    def test_dns_failure_retries_only_missing_requirement_then_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            (project_root / "requirements-kaggle.txt").write_text(
                "synthetic-package==1.0\n",
                encoding="utf-8",
            )
            state = {"installed": False}
            calls = []

            def fake_version(name: str) -> str:
                self.assertEqual(name, "synthetic-package")
                if not state["installed"]:
                    raise importlib_metadata.PackageNotFoundError(name)
                return "1.0"

            def fake_run(*args, **kwargs):
                calls.append((args, kwargs))
                if len(calls) == 1:
                    return SimpleNamespace(
                        returncode=1,
                        stdout="Temporary failure in name resolution",
                    )
                state["installed"] = True
                return SimpleNamespace(returncode=0, stdout="")

            output = io.StringIO()
            with (
                patch.object(importlib_metadata, "version", fake_version),
                patch.object(subprocess, "run", fake_run),
                patch("time.sleep") as sleep,
                contextlib.redirect_stdout(output),
            ):
                exec(
                    self.bootstrap_snippet,
                    {
                        "project_root": project_root,
                        "subprocess": subprocess,
                        "sys": sys,
                        "importlib": importlib,
                    },
                )

        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(5.0)
        pip_command = calls[0][0][0]
        self.assertIn("synthetic-package==1.0", pip_command)
        self.assertIn("DEPENDENCY_NETWORK_RETRY", output.getvalue())
        self.assertIn("DEPENDENCY_INSTALL_VERIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
