from __future__ import annotations

import json
from typing import Any

from .config import ExperimentConfig, RuntimeSettings
from .environment import collect_environment, environment_contract
from .events import EventLog
from .manifest import ManifestStore
from .paths import RunPaths
from .provenance import ExperimentSpec, source_fingerprint
from .shards import ShardStore
from .utils import atomic_write_json, atomic_write_text


def initialize_run(
    config: ExperimentConfig,
    runtime: RuntimeSettings,
    spec: ExperimentSpec | None = None,
    restore_token: str | None = None,
) -> dict[str, Any]:
    project_root = config.path.parents[1]
    full_source_fingerprint = source_fingerprint(project_root)
    current_environment = environment_contract()
    if spec is None:
        spec = ExperimentSpec.create(
            config=config.data,
            # The content tree is portable across a Git checkout and a read-only
            # Kaggle source bundle. Git diagnostics are recorded separately.
            source={
                "tree_sha256": full_source_fingerprint["tree_sha256"],
                "file_count": full_source_fingerprint["file_count"],
            },
            environment=current_environment,
            protocol={"name": "bootstrap", "version": 1},
        )
    else:
        declared_source = spec.payload.get("source")
        declared_environment = spec.payload.get("environment")
        if not isinstance(declared_source, dict) or declared_source.get(
            "tree_sha256"
        ) != full_source_fingerprint["tree_sha256"]:
            raise RuntimeError(
                "Supplied experiment spec does not match the current source tree"
            )
        if declared_environment != current_environment:
            raise RuntimeError(
                "Supplied experiment spec does not match the current runtime environment"
            )
    identity = runtime.identity(spec.spec_hash)
    paths = RunPaths(runtime.work_root, identity).create()
    restore_result = None
    if restore_token and not paths.manifest.exists():
        from .sync import restore_worker_from_hub

        restore_result = restore_worker_from_hub(
            paths=paths,
            identity=identity,
            repo_id=runtime.hf_repo_id,
            revision=runtime.hf_revision,
            token=restore_token,
            repo_type=runtime.hf_repo_type,
        )
    events = EventLog(paths.events)
    manifest = ManifestStore(paths.manifest, identity)
    manifest.initialize(
        metadata={
            "config_path": str(config.path),
            "config_hash": config.config_hash,
            "spec_hash": spec.spec_hash,
            "hf_repo_id": runtime.hf_repo_id,
            "hf_repo_type": runtime.hf_repo_type,
        }
    )
    atomic_write_text(
        paths.meta / f"run_config{config.path.suffix.lower()}",
        config.path.read_text(encoding="utf-8"),
    )
    atomic_write_json(paths.meta / "experiment_spec.json", spec.as_dict())
    atomic_write_json(paths.meta / "source_fingerprint.json", full_source_fingerprint)
    atomic_write_json(paths.meta / "runtime_public.json", runtime.public_dict())
    atomic_write_json(paths.meta / "environment.json", collect_environment(include_pip_freeze=True))
    events.append("RUN_INITIALIZED", identity=identity.as_dict())
    shards = ShardStore(paths, manifest)
    recovered = shards.reconcile()
    if recovered:
        events.append(
            "ORPHAN_SHARDS_RECOVERED",
            count=len(recovered),
            paths=[record.relative_path for record in recovered],
        )
    return {
        "identity": identity,
        "paths": paths,
        "events": events,
        "manifest": manifest,
        "shards": shards,
        "spec": spec,
        "restore_result": restore_result,
    }


def summarize_run(context: dict[str, Any]) -> str:
    manifest = context["manifest"].read()
    result = {
        "worker_root": str(context["paths"].worker_root),
        "status": manifest["status"],
        "counters": manifest["counters"],
        "shards": len(manifest["shards"]),
    }
    return json.dumps(result, indent=2)
