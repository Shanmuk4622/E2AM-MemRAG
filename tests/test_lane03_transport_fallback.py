from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import huggingface_hub

from e2am_memrag import rag_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import build_experiment_notebooks as notebook_builder  # noqa: E402


class _Response:
    status_code = 403
    url = "https://us.gcp.cdn.hf.co/xet-bridge-us/blob"
    headers: dict[str, str] = {}


class _SignatureError(Exception):
    def __init__(self) -> None:
        super().__init__("403 SignatureError: invalid key pair id")
        self.response = _Response()


class Lane03TransportFallbackTests(unittest.TestCase):
    def test_fallback_is_scoped_to_lane03_and_never_prints_token(self) -> None:
        notebooks = notebook_builder.build_all_notebooks()
        marker = "LANE03_TRANSPORT_PATCH_READY"
        marked = []
        for name, notebook in notebooks.items():
            code = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            if marker in code:
                marked.append(name)
        self.assertEqual(marked, ["03_pilot_routes_lane_03.ipynb"])

        lane03 = notebooks["03_pilot_routes_lane_03.ipynb"]
        prepare_source = "".join(lane03["cells"][7]["source"])
        fallback_source = prepare_source.split("REQUEST = StageRequest", 1)[0]
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs.get("token"))
            if kwargs.get("token") is False:
                raise _SignatureError()
            return "/verified/authenticated/snapshot"

        globals_dict = rag_engine.ensure_model_snapshot.__globals__
        sentinel = object()
        previous = globals_dict.get("snapshot_download", sentinel)
        namespace = {"HF_TOKEN": "synthetic-secret-never-print"}
        output = io.StringIO()
        try:
            with (
                patch.object(huggingface_hub, "snapshot_download", fake_snapshot_download),
                contextlib.redirect_stdout(output),
            ):
                exec(fallback_source, namespace)
                wrapper = globals_dict["snapshot_download"]
                result = wrapper(repo_id="public/model", revision="a" * 40, token=False)
        finally:
            if previous is sentinel:
                globals_dict.pop("snapshot_download", None)
            else:
                globals_dict["snapshot_download"] = previous

        self.assertEqual(result, "/verified/authenticated/snapshot")
        self.assertEqual(calls, [False, "synthetic-secret-never-print"])
        self.assertIn("MODEL_DOWNLOAD_AUTHENTICATED_URL_FALLBACK", output.getvalue())
        self.assertNotIn("synthetic-secret-never-print", output.getvalue())


if __name__ == "__main__":
    unittest.main()
