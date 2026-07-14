from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e2am_memrag.bootstrap import initialize_run  # noqa: E402
from e2am_memrag.config import ExperimentConfig, RuntimeSettings  # noqa: E402
from e2am_memrag.identity import make_config_hash  # noqa: E402
from e2am_memrag.runner import ResumableRunner  # noqa: E402
from e2am_memrag.signals import StopController  # noqa: E402


def main() -> int:
    config_path = (ROOT / "configs" / "bootstrap.json").resolve()
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if config_data.get("schema_version") != 2:
        raise ValueError("Local project validation requires the frozen v3 schema_version=2")
    if config_data.get("project") != "e2am-memrag":
        raise ValueError("Local project validation requires project='e2am-memrag'")
    config = ExperimentConfig(
        path=config_path,
        data=config_data,
        config_hash=make_config_hash(config_data),
    )
    with tempfile.TemporaryDirectory(prefix="e2am-validate-") as temporary:
        runtime = RuntimeSettings(
            experiment_id="local-validation",
            worker_id="local-shard-00",
            shard_index=0,
            shard_count=1,
            physical_gpu_index=0,
            work_root=Path(temporary),
            hf_repo_id="Shanmuk4622/E2AM-MemRAG-Traces",
        )
        context = initialize_run(config, runtime)
        units = [{"value": value} for value in range(21)]

        first = ResumableRunner(
            identity=context["identity"],
            manifest=context["manifest"],
            shards=context["shards"],
            events=context["events"],
            stop=StopController(),
            shard_rows=4,
        ).run(units, lambda unit: {"square": unit["value"] ** 2}, max_new_units=7)

        second = ResumableRunner(
            identity=context["identity"],
            manifest=context["manifest"],
            shards=context["shards"],
            events=context["events"],
            stop=StopController(),
            shard_rows=4,
        ).run(units, lambda unit: {"square": unit["value"] ** 2})

        completed = context["shards"].completed_unit_ids()
        validation = context["shards"].validate()
        assert len(completed) == 21, len(completed)
        assert first.processed == 7
        assert second.processed == 14
        report = {
            "status": "OK",
            "first_pass": first.__dict__,
            "resume_pass": second.__dict__,
            "validation": validation,
            "completed_units": len(completed),
        }
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
