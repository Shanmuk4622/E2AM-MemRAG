from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, RuntimeSettings
from .environment import collect_environment, preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an E2AM-MemRAG runtime")
    parser.add_argument("--config", default="configs/bootstrap.yaml")
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    args = parser.parse_args()
    config = ExperimentConfig.load(Path(args.config))
    runtime = RuntimeSettings.from_env()
    report = {
        "config_hash": config.config_hash,
        "runtime": runtime.public_dict(),
        "preflight": preflight(runtime.work_root, args.minimum_free_gib),
        "environment": collect_environment(include_pip_freeze=False),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

