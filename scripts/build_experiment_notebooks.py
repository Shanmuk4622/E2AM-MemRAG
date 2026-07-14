"""Deterministically build the ordered E2AM-MemRAG Kaggle notebooks.

The builder keeps notebook JSON out of hand-written source control.  It emits six
coordinator notebooks and four fixed lane variants for each parallel stage.  A
lane file is assigned to one collaborator and never needs a worker-ID edit.

This module intentionally imports only the Python standard library.  The emitted
notebooks call the high-level API implemented by
``e2am_memrag.experiment_pipeline``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


SYNC_INTERVAL_SECONDS = 1200
LANE_COUNT = 4
DEFAULT_EXPERIMENT_ID = "e2am-memrag-v3r1"
DEFAULT_HF_REPO_ID = "Shanmuk4622/E2AM-MemRAG-Traces"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MANIFEST_PATH = "E2AM_RUNTIME_MANIFEST.json"
RUNTIME_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024
PATCH_EXPORT_CHUNK_BYTES = 24_000


@dataclass(frozen=True)
class RuntimeBundle:
    """Deterministic current-project runtime embedded in every notebook."""

    archive: bytes
    archive_sha256: str
    tree_sha256: str
    manifest: Mapping[str, object]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _runtime_files() -> list[tuple[str, bytes]]:
    """Read the explicitly allowed runtime paths, excluding generated caches."""

    required_roots = (Path("src/e2am_memrag"), Path("configs"))
    required_files = (Path("requirements-kaggle.txt"), Path("pyproject.toml"))
    candidates: list[Path] = []
    for relative_root in required_roots:
        root = PROJECT_ROOT / relative_root
        if not root.is_dir():
            raise FileNotFoundError(f"Required runtime directory is missing: {root}")
        candidates.extend(path for path in root.rglob("*") if path.is_file())
    for relative_file in required_files:
        path = PROJECT_ROOT / relative_file
        if not path.is_file():
            raise FileNotFoundError(f"Required runtime file is missing: {path}")
        candidates.append(path)

    selected: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT)
        if path.is_symlink():
            raise ValueError(f"Runtime bundle cannot contain a symlink: {relative.as_posix()}")
        if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if path.name.endswith((".tmp", ".swp", "~")):
            continue
        name = relative.as_posix()
        if name in seen:
            raise ValueError(f"Duplicate runtime path: {name}")
        seen.add(name)
        selected.append((name, path.read_bytes()))

    if not any(name == "src/e2am_memrag/experiment_pipeline.py" for name, _ in selected):
        raise FileNotFoundError("src/e2am_memrag/experiment_pipeline.py is required")
    return selected


def build_runtime_bundle() -> RuntimeBundle:
    """Create a byte-for-byte reproducible ZIP and an internal member manifest."""

    files = _runtime_files()
    entries = [
        {
            "path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for name, payload in files
    ]
    tree_sha256 = hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_tree_sha256": tree_sha256,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")
    payloads = [*files, (RUNTIME_MANIFEST_PATH, manifest_bytes)]

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, payload in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    archive_bytes = stream.getvalue()
    if len(archive_bytes) > RUNTIME_ARCHIVE_MAX_BYTES:
        raise ValueError(
            f"Runtime archive is unexpectedly large: {len(archive_bytes)} bytes"
        )
    return RuntimeBundle(
        archive=archive_bytes,
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        tree_sha256=tree_sha256,
        manifest=manifest,
    )


@dataclass(frozen=True)
class StageSpec:
    """One logical experiment stage before lane expansion."""

    order: int
    slug: str
    title: str
    role: str
    summary: str
    prerequisites: tuple[str, ...]
    output_gate: str
    work_items: tuple[str, ...]
    artifacts: tuple[str, ...]
    parallel_note: str = ""

    @property
    def stage_id(self) -> str:
        return f"{self.order:02d}"


@dataclass(frozen=True)
class NotebookInstance:
    """A concrete coordinator notebook or one immutable lane variant."""

    spec: StageSpec
    filename: str
    lane_index: int | None

    @property
    def lane_id(self) -> str | None:
        if self.lane_index is None:
            return None
        return f"lane-{self.lane_index:02d}"

    @property
    def worker_id(self) -> str:
        suffix = self.lane_id or "coordinator"
        return f"stage-{self.spec.stage_id}-{suffix}"

    @property
    def artifact_prefix(self) -> str:
        owner = self.lane_id or "coordinator"
        return (
            f"experiments/{DEFAULT_EXPERIMENT_ID}/stages/"
            f"{self.spec.stage_id}/{owner}"
        )


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        order=0,
        slug="setup_and_freeze_data",
        title="Set up the experiment and freeze the data specification",
        role="coordinator",
        summary=(
            "Create or restore the shared Hugging Face dataset repository, verify the "
            "Kaggle runtime, pin source/model/data revisions, build leakage-safe splits, "
            "and freeze the experiment specification before expensive work starts."
        ),
        prerequisites=(),
        output_gate="SETUP_DATA_FREEZE.json",
        work_items=(
            "verify one visible T4, actual storage, package/runtime fingerprints, and NVML",
            "generate the controlled benchmark deterministically and record immutable checksums",
            "build scenario-grouped pilot/train/calibration/validation/test assignments",
            "verify pinned model metadata only; model weights are deferred to owning lanes",
            "seal test labels and freeze five models, 22 routes, prompts, seeds, and evaluation rules",
            "publish a fresh-root-restorable source and experiment-spec closure",
        ),
        artifacts=(
            "experiment_spec.json",
            "dataset_ledger.json",
            "split_manifest.json",
            "environment.json",
            "model_metadata/MODEL_METADATA.json",
            "SETUP_DATA_FREEZE.json",
        ),
    ),
    StageSpec(
        order=1,
        slug="build_indexes",
        title="Build and verify retrieval and memory indexes",
        role="coordinator",
        summary=(
            "Build pinned BM25, dense, hybrid, temporal-memory, and authority metadata "
            "indexes from the frozen source-only corpus, then prove deterministic restore."
        ),
        prerequisites=("00/coordinator/SETUP_DATA_FREEZE.json",),
        output_gate="INDEX_FREEZE.json",
        work_items=(
            "restore the exact stage-00 specification and reject any hash mismatch",
            "exclude tombstones and every tombstoned target from all memory indexes",
            "audit label-file separation, temporal availability, and frozen corpus hashes",
            "record index recipes, sizes, checksums, and resumable batch build time",
            "compact immutable files and checksum every vector shard before freezing",
        ),
        artifacts=(
            "index_work_plan.json",
            "index_catalog.json",
            "corpus_leakage_audit.json",
            "index_storage_ledger.json",
            "INDEX_FREEZE.json",
        ),
    ),
    StageSpec(
        order=2,
        slug="build_and_freeze_hybridbench",
        title="Build and freeze E2AM-HybridBench",
        role="coordinator",
        summary=(
            "Materialize the benchmark, controlled memory streams, corruptions, evaluator "
            "fixtures, and statistical analysis plan while keeping clean-test labels sealed."
        ),
        prerequisites=("01/coordinator/INDEX_FREEZE.json",),
        output_gate="HYBRIDBENCH_FREEZE.json",
        work_items=(
            "create knowledge, memory, temporal-conflict, deletion, and abstention scenarios",
            "keep every base scenario and conversation inside one statistical group",
            "run exact, near-duplicate, scenario-group, template, and temporal leakage audits",
            "freeze evaluator prompts/parsers, corruption generators, and primary endpoints",
            "publish only licensed source data or permitted derived identifiers",
        ),
        artifacts=(
            "benchmark/queries.jsonl",
            "benchmark/non_test_labels.jsonl",
            "hybridbench_manifest.json",
            "benchmark_audit.json",
            "evaluator_freeze.json",
            "statistical_plan.json",
            "sealed_test_vault.json",
            "foundation_bootstrap.zip",
            "HYBRIDBENCH_FREEZE.json",
        ),
    ),
    StageSpec(
        order=3,
        slug="pilot_routes",
        title="Pilot candidate LLM, RAG, and memory routes",
        role="lane",
        summary=(
            "Run one fixed route/model-owned slice of the pilot matrix over every pilot query, "
            "measuring quality, latency, selected-GPU board energy, residency, failures, and cost."
        ),
        prerequisites=("02/coordinator/HYBRIDBENCH_FREEZE.json",),
        output_gate="PILOT_LANE_SEAL.json",
        work_items=(
            "claim only unit IDs assigned to this fixed lane",
            "resume an immutable pinned snapshot with visible download heartbeats and retries",
            "load each declared model on the single visible T4 and reject CPU/disk offload",
            "compare direct, BM25, dense, bounded two-hop hybrid, and memory-aware routes",
            "perform Hub uploads only after each NVML generation measurement has stopped",
            "project runtime, storage, Hub requests, artifact count, and restore time",
        ),
        artifacts=(
            "matrix_work_plan.json",
            "results/shards/",
            "lane_metadata.json",
            "lane_export.zip",
            "PILOT_LANE_SEAL.json",
        ),
        parallel_note=(
            "Run all four fixed stage-03 lane notebooks in parallel after stage 02. "
            "One person/session owns each file."
        ),
    ),
    StageSpec(
        order=4,
        slug="aggregate_and_prune_pilot",
        title="Aggregate the pilot and freeze retained routes",
        role="coordinator",
        summary=(
            "Verify complete, non-overlapping pilot coverage; aggregate paired evidence; "
            "filter infeasible routes using pilot-only rules; and freeze the route matrix."
        ),
        prerequisites=("03/lane-*/PILOT_LANE_SEAL.json",),
        output_gate="PILOT_ROUTE_FREEZE.json",
        work_items=(
            "verify all four lane closures at pinned commits and reject divergent duplicates",
            "reject missing telemetry, excessive failures, mixed specs, and unsafe routes",
            "filter only with predeclared pilot rules, never clean-test performance",
            "separate resident single-T4 routes from sequential offline model sweeps",
            "freeze retained routes and expected stage-05 work-plan coverage",
        ),
        artifacts=(
            "pilot_aggregate.jsonl",
            "route_pruning_report.json",
            "retained_routes.json",
            "budget_projection.json",
            "trace_work_plan.json",
            "training_bootstrap.zip",
            "PILOT_ROUTE_FREEZE.json",
        ),
    ),
    StageSpec(
        order=5,
        slug="collect_training_traces",
        title="Collect RAG and memory router-training traces",
        role="lane",
        summary=(
            "Execute one route-owned slice of the deployable train/calibration/validation "
            "matrix; sequential reference models are excluded from router fitting."
        ),
        prerequisites=("04/coordinator/PILOT_ROUTE_FREEZE.json",),
        output_gate="TRACE_LANE_SEAL.json",
        work_items=(
            "restore the frozen work plan and skip canonical unit IDs already completed",
            "collect query-only, charged-probe, retrieval, generation, and citation-guard costs",
            "give memory policies the same event stream and information budget",
            "record abstentions, failures, coverage, latency, GPU joules, and quality targets",
            "seal replay-bounded shards before each 20-minute or major remote receipt",
        ),
        artifacts=(
            "matrix_work_plan.json",
            "results/shards/",
            "lane_metadata.json",
            "lane_export.zip",
            "TRACE_LANE_SEAL.json",
        ),
        parallel_note=(
            "Run all four fixed stage-05 lane notebooks in parallel after stage 04. "
            "A stopped lane resumes by rerunning its same notebook."
        ),
    ),
    StageSpec(
        order=6,
        slug="train_and_calibrate_router",
        title="Train and calibrate E2AM-ParetoRouter",
        role="coordinator",
        summary=(
            "Verify trace coverage, train multi-seed action success/energy/latency models, "
            "calibrate conservative bounds, and freeze the controller before clean testing."
        ),
        prerequisites=("05/lane-*/TRACE_LANE_SEAL.json",),
        output_gate="ROUTER_CALIBRATION_FREEZE.json",
        work_items=(
            "reject missing work units, mixed spec hashes, and divergent duplicate outputs",
            "train five grouped-bootstrap tree ensembles from fully resumable frozen traces",
            "calibrate success probabilities and conservative energy/latency scores",
            "fit a two-stage query-only then charged-probe cost-aware cascade",
            "freeze thresholds, non-inferiority margin, coverage, and violation tolerances",
        ),
        artifacts=(
            "router/seed_checkpoints/",
            "router/e2am_pareto_router.joblib",
            "router/router_manifest.json",
            "training_history.json",
            "frozen_policy.json",
            "evaluation_bootstrap.zip",
            "ROUTER_CALIBRATION_FREEZE.json",
        ),
    ),
    StageSpec(
        order=7,
        slug="evaluate_frozen_clean",
        title="Evaluate the frozen policy on the clean test set",
        role="lane",
        summary=(
            "Evaluate every sealed clean-test scenario for one disjoint route/model-owned "
            "slice without changing routes, thresholds, prompts, models, or evaluation rules."
        ),
        prerequisites=("06/coordinator/ROUTER_CALIBRATION_FREEZE.json",),
        output_gate="CLEAN_EVAL_LANE_SEAL.json",
        work_items=(
            "open test labels only after verifying the router and analysis freeze",
            "run this lane's retained routes over all test queries for exact paired accounting",
            "let lane-00 freeze policy decisions for merge-time binding to actual route rows",
            "measure paired quality, coverage, violations, latency, and selected-GPU joules",
            "record every failure and abstention; never silently drop a hard example",
            "seal immutable results without computing a lane-local winner",
        ),
        artifacts=(
            "matrix_work_plan.json",
            "results/shards/",
            "policy_decisions.jsonl",
            "lane_export.zip",
            "test_vault_access_*.json",
            "CLEAN_EVAL_LANE_SEAL.json",
        ),
        parallel_note=(
            "Run all four fixed stage-07 lane notebooks in parallel after stage 06. "
            "Do not inspect aggregate test outcomes before all lanes close."
        ),
    ),
    StageSpec(
        order=8,
        slug="run_robustness",
        title="Run frozen robustness stress tests",
        role="lane",
        summary=(
            "Run one pre-frozen stale, injection, missing-evidence, or deletion/duplicate "
            "stress condition against the policy and fixed anchors."
        ),
        prerequisites=("07/{lane}/CLEAN_EVAL_LANE_SEAL.json",),
        output_gate="ROBUSTNESS_LANE_SEAL.json",
        work_items=(
            "apply the fixed lane's deterministic corruption to every sealed test query",
            "evaluate the frozen cascade plus tiny-direct and guarded-hybrid anchors",
            "treat retrieved prompt injection as untrusted evidence, never instructions",
            "preserve deleted-memory exclusions and record every failure or abstention",
            "seal exact query coverage for final paired aggregation",
        ),
        artifacts=(
            "robustness_work_plan.json",
            "results/shards/",
            "robustness_decisions.jsonl",
            "lane_export.zip",
            "test_vault_access_*.json",
            "ROBUSTNESS_LANE_SEAL.json",
        ),
        parallel_note=(
            "Run all four fixed stage-08 lane notebooks in parallel after every stage-07 "
            "lane is complete."
        ),
    ),
    StageSpec(
        order=9,
        slug="aggregate_audit_and_release",
        title="Aggregate, audit, and publish the reproducible release",
        role="coordinator",
        summary=(
            "Verify every remote closure, compute the frozen paired statistical analysis, "
            "and publish one checksum-verified JSON release seal."
        ),
        prerequisites=(
            "07/lane-*/CLEAN_EVAL_LANE_SEAL.json",
            "08/lane-*/ROBUSTNESS_LANE_SEAL.json",
        ),
        output_gate="_SUCCESS.json",
        work_items=(
            "verify exact lane coverage, canonical unit uniqueness, and all dependency hashes",
            "compute paired bootstrap intervals for quality and selected-GPU energy",
            "publish five-model direct-versus-grounded transfer and Pareto analyses",
            "report router Brier/ECE and paired robustness degradation by condition",
            "report execution coverage, abstentions, failures, constraint violations, and cost",
            "separate selected-GPU energy from whole-system/carbon scenario assumptions",
            "fresh-root restore the release and publish the coordinator-only success pointer",
        ),
        artifacts=(
            "release_manifest.json",
            "release/experiment_summary.json",
            "release/model_transfer_panel.json",
            "release/mechanism_analysis.json",
            "release/robustness_analysis.json",
            "release/clean_traces.jsonl",
            "release/robustness_traces.jsonl",
            "HYPOTHESIS_RESULT.json",
            "RELEASE_CANDIDATE.json",
            "_SUCCESS.json",
        ),
    ),
)


def iter_instances() -> Iterator[NotebookInstance]:
    """Yield concrete notebooks in deterministic execution/display order."""

    for spec in STAGES:
        if spec.role == "coordinator":
            yield NotebookInstance(
                spec=spec,
                filename=f"{spec.stage_id}_{spec.slug}.ipynb",
                lane_index=None,
            )
            continue
        for lane_index in range(LANE_COUNT):
            yield NotebookInstance(
                spec=spec,
                filename=(
                    f"{spec.stage_id}_{spec.slug}_lane_{lane_index:02d}.ipynb"
                ),
                lane_index=lane_index,
            )


def _resolved_prerequisites(instance: NotebookInstance) -> tuple[str, ...]:
    """Resolve the fixed lane placeholder without exposing an editable worker ID."""

    values = []
    for requirement in instance.spec.prerequisites:
        if "{lane}" in requirement:
            if instance.lane_id is None:
                raise ValueError("A lane prerequisite was assigned to a coordinator")
            requirement = requirement.replace("{lane}", instance.lane_id)
        values.append(requirement)
    return tuple(values)


def _clean_source(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def _source_lines(source: str) -> list[str]:
    return _clean_source(source).splitlines(keepends=True)


def _cell_id(instance: NotebookInstance, index: int, kind: str) -> str:
    value = f"e2am-memrag:{instance.filename}:{index}:{kind}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def _markdown(instance: NotebookInstance, index: int, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": _cell_id(instance, index, "markdown"),
        "metadata": {},
        "source": _source_lines(source),
    }


def _code(instance: NotebookInstance, index: int, source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(instance, index, "code"),
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _bullet_lines(values: Sequence[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _intro(instance: NotebookInstance) -> str:
    spec = instance.spec
    owner = "the coordinator" if spec.role == "coordinator" else instance.lane_id
    lines = [
        f"# Stage {spec.stage_id}: {spec.title}",
        "",
        f"**File:** `{instance.filename}`  ",
        f"**Owner:** {owner}  ",
        f"**Fixed recovery worker:** `{instance.worker_id}`  ",
        f"**Success gate:** `{spec.output_gate}`",
        "",
        spec.summary,
    ]
    if spec.parallel_note:
        lines.extend(("", spec.parallel_note))
    lines.extend(
        (
            "",
            "This notebook is standalone for a fresh Kaggle session: it restores required",
            "remote gates and the newest checksum-verified checkpoint from Hugging Face.",
            "There is no team-roster notebook and there are no worker parameters to edit.",
            "Never run this exact file in two live sessions. Rerunning the same file after a",
            "stop is the supported resume path.",
        )
    )
    return "\n".join(lines)


def _settings_code(instance: NotebookInstance) -> str:
    spec = instance.spec
    lane_literal = repr(instance.lane_id)
    prerequisites = repr(_resolved_prerequisites(instance))
    return f"""
    # Fixed settings: do not edit. A lane file already has its permanent identity.
    EXPERIMENT_ID = {DEFAULT_EXPERIMENT_ID!r}
    STAGE_ID = {spec.stage_id!r}
    STAGE_NAME = {spec.slug!r}
    ROLE = {spec.role!r}
    LANE_ID = {lane_literal}
    LANE_COUNT = {LANE_COUNT}
    WORKER_ID = {instance.worker_id!r}
    NOTEBOOK_NAME = {instance.filename!r}

    HF_REPO_ID = {DEFAULT_HF_REPO_ID!r}
    HF_REPO_TYPE = 'dataset'
    HF_REVISION = f'stage-{{EXPERIMENT_ID}}-{{STAGE_ID}}-{{WORKER_ID}}'
    ARTIFACT_PREFIX = {instance.artifact_prefix!r}
    REQUIRED_GATES = {prerequisites}
    OUTPUT_GATE = {spec.output_gate!r}
    SYNC_INTERVAL_SECONDS = {SYNC_INTERVAL_SECONDS}
    WORK_ROOT = '/kaggle/working/e2am_memrag'
    MODEL_CACHE_ROOT = WORK_ROOT + '/model-cache'
    GPU_INDEX = 0

    # The mask must happen before Torch, Transformers, NVML, or any CUDA-aware import.
    import os
    import sys

    if 'torch' in sys.modules:
        raise RuntimeError(
            'Torch was imported before the one-GPU mask. Restart the kernel and Run All.'
        )
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Hugging Face reads these at import time. Keep the public model cache on the
    # measured Kaggle working filesystem and turn a silent 0% transfer into a
    # bounded, resumable HTTP download with explicit timeouts.
    os.environ['HF_HOME'] = MODEL_CACHE_ROOT
    os.environ['HF_HUB_CACHE'] = MODEL_CACHE_ROOT + '/hub'
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    # The hf-xet worker can remain alive without advancing bytes on Kaggle. Use
    # the bounded HTTP client instead; the verified runtime refreshes only the
    # narrowly identified transient signed-blob 403 and preserves partial blobs.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['E2AM_SYNC_INTERVAL_SECONDS'] = str(SYNC_INTERVAL_SECONDS)
    os.environ['E2AM_HF_REPO_ID'] = HF_REPO_ID
    print({{
        'stage': STAGE_ID,
        'worker': WORKER_ID,
        'lane': LANE_ID,
        'sync_minutes': SYNC_INTERVAL_SECONDS // 60,
    }})
    """


def _runbook(instance: NotebookInstance) -> str:
    spec = instance.spec
    resolved_prerequisites = _resolved_prerequisites(instance)
    prerequisites = (
        _bullet_lines(resolved_prerequisites)
        if resolved_prerequisites
        else "- None. Stage 00 creates or restores the experiment repository."
    )
    work_items = _bullet_lines(spec.work_items)
    artifacts = _bullet_lines(spec.artifacts)
    lane_instruction = (
        f"This is fixed `{instance.lane_id}` of `{LANE_COUNT}`; do not change its lane ID."
        if instance.lane_id
        else "This is a coordinator-only stage; run it exactly once after its prerequisites."
    )
    return "\n".join(
        (
            "## Numbered Kaggle runbook",
            "",
            "1. Open this notebook in Kaggle. Turn **Internet on** and select a **GPU** accelerator.",
            "   A dual-T4 session is acceptable; the first code cell exposes only GPU 0.",
            "2. Add a Kaggle Secret named `HF_TOKEN` with write access to",
            f"   `{DEFAULT_HF_REPO_ID}`. Never paste the token into a cell.",
            "3. Confirm every prerequisite gate below exists, then choose **Run All**.",
            f"   {lane_instruction}",
            "4. Leave the notebook running until it prints `STAGE_COMPLETE` and",
            "   `REMOTE_CLOSURE_VERIFIED`. Dirty state is bundled every 20 minutes and after",
            "   major units; clean intervals do not create Hub commits.",
            "5. If you interrupt execution, wait for `SAFE_STOP_VERIFIED`. On a hard session",
            "   loss, open this same notebook and Run All; only the smallest unsealed unit may replay.",
            "",
            "### Required remote gates",
            "",
            prerequisites,
            "",
            "### Work performed",
            "",
            work_items,
            "",
            f"### Expected artifacts under `{instance.artifact_prefix}/`",
            "",
            artifacts,
        )
    )


def _runtime_bootstrap_code(
    bundle: RuntimeBundle,
    *,
    resilient_dependency_install: bool = False,
) -> str:
    encoded = base64.b64encode(bundle.archive).decode("ascii")
    chunks = [encoded[index : index + 120] for index in range(0, len(encoded), 120)]
    literals = "\n".join(f"    {chunk!r}" for chunk in chunks)
    header = (
        "# Deterministic runtime embedded by scripts/build_experiment_notebooks.py.\n"
        "EMBEDDED_RUNTIME_B64 = (\n"
        f"{literals}\n"
        ")\n"
        f"EXPECTED_RUNTIME_ARCHIVE_SHA256 = {bundle.archive_sha256!r}\n"
        f"EXPECTED_RUNTIME_TREE_SHA256 = {bundle.tree_sha256!r}\n"
        f"EXPECTED_RUNTIME_FILE_COUNT = {len(bundle.manifest['files'])}\n"
        f"RUNTIME_MANIFEST_PATH = {RUNTIME_MANIFEST_PATH!r}\n"
        f"RUNTIME_ARCHIVE_MAX_BYTES = {RUNTIME_ARCHIVE_MAX_BYTES}\n"
    )
    if resilient_dependency_install:
        dependency_bootstrap = _clean_source(
            """
            # LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1
            import time
            from importlib import metadata as importlib_metadata

            try:
                from packaging.requirements import Requirement
            except ImportError:
                from pip._vendor.packaging.requirements import Requirement

            requirements_path = project_root / 'requirements-kaggle.txt'
            declared_requirements = []
            skipped_torch_packages = []
            for raw_line in requirements_path.read_text(encoding='utf-8').splitlines():
                requirement = raw_line.split('#', 1)[0].strip()
                if not requirement:
                    continue
                if requirement.startswith('-'):
                    raise RuntimeError(f'Unsupported requirements option: {requirement!r}')
                parsed = Requirement(requirement)
                normalized_name = parsed.name.lower().replace('_', '-').replace('.', '-')
                if normalized_name in {'torch', 'torchvision', 'torchaudio'}:
                    skipped_torch_packages.append(parsed.name)
                    continue
                declared_requirements.append((requirement, parsed))

            def requirement_status(parsed):
                try:
                    installed_version = importlib_metadata.version(parsed.name)
                except importlib_metadata.PackageNotFoundError:
                    return False, None
                compatible = (
                    not parsed.specifier
                    or parsed.specifier.contains(installed_version, prereleases=True)
                )
                return compatible, installed_version

            missing_requirements = []
            satisfied_versions = {}
            for requirement, parsed in declared_requirements:
                compatible, installed_version = requirement_status(parsed)
                if compatible:
                    satisfied_versions[parsed.name] = installed_version
                else:
                    missing_requirements.append(requirement)

            print('DEPENDENCY_PRECHECK', {
                'satisfied': len(satisfied_versions),
                'install_needed': missing_requirements,
            })
            if missing_requirements:
                retry_waits = (5.0, 15.0, 30.0)
                maximum_attempts = len(retry_waits) + 1
                for attempt in range(1, maximum_attempts + 1):
                    print('DEPENDENCY_INSTALL_START', {
                        'attempt': attempt,
                        'maximum_attempts': maximum_attempts,
                        'requirements': missing_requirements,
                    })
                    completed = subprocess.run(
                        [
                            sys.executable,
                            '-m',
                            'pip',
                            'install',
                            '--disable-pip-version-check',
                            '--no-cache-dir',
                            '--upgrade-strategy',
                            'only-if-needed',
                            '--retries',
                            '1',
                            '--timeout',
                            '20',
                            '-q',
                            *missing_requirements,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    if completed.returncode == 0:
                        break
                    output_tail = (completed.stdout or '')[-4000:]
                    print('DEPENDENCY_INSTALL_ATTEMPT_FAILED', {
                        'attempt': attempt,
                        'returncode': completed.returncode,
                        'output_tail': output_tail,
                    })
                    lowered = output_tail.lower()
                    transient_markers = (
                        'temporary failure in name resolution',
                        'name or service not known',
                        'failed to establish a new connection',
                        'network is unreachable',
                        'connection reset',
                        'connection timed out',
                        'read timed out',
                        'too many 429',
                        'too many 500',
                        'too many 502',
                        'too many 503',
                        'too many 504',
                    )
                    transient = any(marker in lowered for marker in transient_markers)
                    if not transient or attempt == maximum_attempts:
                        category = 'temporary Kaggle/PyPI network failure' if transient else 'non-retryable pip failure'
                        raise RuntimeError(
                            'DEPENDENCY_SAFE_STOP: ' + category + '. No Stage-05 trace work '
                            'started and no checkpoint was changed. Keep the declared requirements '
                            'unchanged; restart this lane-01 notebook with Internet enabled and Run All.'
                        )
                    wait_seconds = retry_waits[attempt - 1]
                    print('DEPENDENCY_NETWORK_RETRY', {
                        'attempt': attempt,
                        'wait_seconds': wait_seconds,
                    })
                    time.sleep(wait_seconds)

                importlib.invalidate_caches()
                unsatisfied_after_install = []
                for requirement, parsed in declared_requirements:
                    compatible, installed_version = requirement_status(parsed)
                    if not compatible:
                        unsatisfied_after_install.append({
                            'requirement': requirement,
                            'installed': installed_version,
                        })
                if unsatisfied_after_install:
                    raise RuntimeError(
                        'DEPENDENCY_SAFE_STOP: pip returned success but the frozen requirements '
                        f'are not satisfied: {unsatisfied_after_install!r}'
                    )
                print('DEPENDENCY_INSTALL_VERIFIED', {
                    'installed': missing_requirements,
                })
            else:
                print('DEPENDENCY_INSTALL_SKIPPED: frozen requirements already satisfied')

            if skipped_torch_packages:
                print('Preserved Kaggle Torch stack; skipped:', sorted(skipped_torch_packages))
            # END_LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1
            """
        )
    else:
        dependency_bootstrap = _clean_source(
            """
            requirements_path = project_root / 'requirements-kaggle.txt'
            install_requirements = []
            skipped_torch_packages = []
            for raw_line in requirements_path.read_text(encoding='utf-8').splitlines():
                requirement = raw_line.split('#', 1)[0].strip()
                if not requirement:
                    continue
                if requirement.startswith('-'):
                    raise RuntimeError(f'Unsupported requirements option: {requirement!r}')
                package_name = re.split(r'[<>=!~@\\[\\s]', requirement, maxsplit=1)[0]
                normalized_name = package_name.lower().replace('_', '-').replace('.', '-')
                if normalized_name in {'torch', 'torchvision', 'torchaudio'}:
                    skipped_torch_packages.append(package_name)
                    continue
                install_requirements.append(requirement)
            if install_requirements:
                subprocess.check_call([
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--disable-pip-version-check',
                    '--no-cache-dir',
                    '--upgrade-strategy',
                    'only-if-needed',
                    '-q',
                    *install_requirements,
                ])
            if skipped_torch_packages:
                print('Preserved Kaggle Torch stack; skipped:', sorted(skipped_torch_packages))
            """
        )

    body = _clean_source(
        """
        import base64
        import hashlib
        import importlib
        import io
        import json
        import re
        import shutil
        import stat
        import subprocess
        import tempfile
        import zipfile
        from pathlib import Path, PurePosixPath

        def canonical_json(value):
            return json.dumps(
                value,
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=False,
                allow_nan=False,
            )

        archive_bytes = base64.b64decode(''.join(EMBEDDED_RUNTIME_B64), validate=True)
        if len(archive_bytes) > RUNTIME_ARCHIVE_MAX_BYTES:
            raise RuntimeError('Embedded runtime exceeds its declared safety ceiling')
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        if archive_sha256 != EXPECTED_RUNTIME_ARCHIVE_SHA256:
            raise RuntimeError('Embedded runtime archive SHA-256 mismatch')

        verified_payloads = {}
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode='r') as runtime_zip:
            infos = runtime_zip.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeError('Embedded runtime ZIP contains duplicate member names')
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or '..' in member.parts
                    or '\\\\' in info.filename
                    or info.filename.startswith('/')
                ):
                    raise RuntimeError(f'Unsafe embedded runtime path: {info.filename!r}')
                unix_mode = info.external_attr >> 16
                if info.is_dir() or not stat.S_ISREG(unix_mode):
                    raise RuntimeError(f'Non-regular runtime member: {info.filename!r}')
                if info.flag_bits & 0x1:
                    raise RuntimeError(f'Encrypted runtime member is forbidden: {info.filename!r}')

            if RUNTIME_MANIFEST_PATH not in names:
                raise RuntimeError('Embedded runtime manifest is missing')
            manifest_bytes = runtime_zip.read(RUNTIME_MANIFEST_PATH)
            manifest = json.loads(manifest_bytes.decode('utf-8'))
            if set(manifest) != {
                'schema_version', 'source_tree_sha256', 'file_count', 'files'
            }:
                raise RuntimeError('Embedded runtime manifest schema is unexpected')
            if manifest['schema_version'] != 1:
                raise RuntimeError('Unsupported embedded runtime manifest version')
            entries = manifest['files']
            if not isinstance(entries, list) or len(entries) != EXPECTED_RUNTIME_FILE_COUNT:
                raise RuntimeError('Embedded runtime member count mismatch')
            if manifest['file_count'] != len(entries):
                raise RuntimeError('Embedded runtime manifest file_count mismatch')
            computed_tree_sha256 = hashlib.sha256(
                canonical_json(entries).encode('utf-8')
            ).hexdigest()
            if (
                manifest['source_tree_sha256'] != EXPECTED_RUNTIME_TREE_SHA256
                or computed_tree_sha256 != EXPECTED_RUNTIME_TREE_SHA256
            ):
                raise RuntimeError('Embedded runtime tree SHA-256 mismatch')

            expected_names = {RUNTIME_MANIFEST_PATH}
            total_uncompressed_bytes = len(manifest_bytes)
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes'}:
                    raise RuntimeError('Malformed embedded runtime file record')
                member_name = entry['path']
                if not isinstance(member_name, str) or member_name in expected_names:
                    raise RuntimeError(f'Duplicate or invalid runtime member: {member_name!r}')
                if not isinstance(entry['bytes'], int) or entry['bytes'] < 0:
                    raise RuntimeError(f'Invalid runtime member size: {member_name!r}')
                if (
                    not isinstance(entry['sha256'], str)
                    or len(entry['sha256']) != 64
                    or any(character not in '0123456789abcdef' for character in entry['sha256'])
                ):
                    raise RuntimeError(f'Invalid runtime member SHA-256: {member_name!r}')
                expected_names.add(member_name)
                total_uncompressed_bytes += entry['bytes']
                payload = runtime_zip.read(member_name)
                if len(payload) != entry['bytes']:
                    raise RuntimeError(f'Runtime member size mismatch: {member_name}')
                if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Runtime member SHA-256 mismatch: {member_name}')
                verified_payloads[member_name] = payload
            if set(names) != expected_names:
                extra = sorted(set(names) - expected_names)
                missing = sorted(expected_names - set(names))
                raise RuntimeError(
                    f'Runtime ZIP member-set mismatch: extra={extra}, missing={missing}'
                )
            if total_uncompressed_bytes > RUNTIME_ARCHIVE_MAX_BYTES:
                raise RuntimeError('Embedded runtime uncompressed size exceeds safety ceiling')

        project_root = Path('/kaggle/working/E2AM-MemRAG')
        project_parent = project_root.parent
        project_parent.mkdir(parents=True, exist_ok=True)
        extraction_stage = Path(
            tempfile.mkdtemp(prefix='.e2am-runtime-', dir=str(project_parent))
        )
        backup_root = project_parent / '.E2AM-MemRAG.previous'
        moved_existing = False
        try:
            for member_name, payload in sorted(verified_payloads.items()):
                target = extraction_stage.joinpath(*PurePosixPath(member_name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            manifest_target = extraction_stage / RUNTIME_MANIFEST_PATH
            with manifest_target.open('xb') as handle:
                handle.write(manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            for entry in entries:
                extracted = extraction_stage.joinpath(*PurePosixPath(entry['path']).parts)
                if extracted.stat().st_size != entry['bytes']:
                    raise RuntimeError(f'Extracted runtime size mismatch: {entry["path"]}')
                if hashlib.sha256(extracted.read_bytes()).hexdigest() != entry['sha256']:
                    raise RuntimeError(f'Extracted runtime SHA-256 mismatch: {entry["path"]}')
            if manifest_target.read_bytes() != manifest_bytes:
                raise RuntimeError('Extracted runtime manifest differs from embedded bytes')

            if backup_root.exists():
                shutil.rmtree(backup_root)
            if project_root.exists():
                os.replace(project_root, backup_root)
                moved_existing = True
            try:
                os.replace(extraction_stage, project_root)
            except BaseException:
                if moved_existing and backup_root.exists() and not project_root.exists():
                    os.replace(backup_root, project_root)
                raise
            if backup_root.exists():
                shutil.rmtree(backup_root, ignore_errors=True)
        except BaseException:
            if extraction_stage.exists():
                shutil.rmtree(extraction_stage, ignore_errors=True)
            raise

        __DEPENDENCY_BOOTSTRAP__

        project_src = str(project_root / 'src')
        sys.path[:] = [project_src, *[item for item in sys.path if item != project_src]]
        importlib.invalidate_caches()
        os.environ['E2AM_SOURCE_TREE_SHA256'] = EXPECTED_RUNTIME_TREE_SHA256
        print({
            'runtime': 'EMBEDDED_RUNTIME_VERIFIED',
            'project_root': str(project_root),
            'archive_sha256': archive_sha256,
            'tree_sha256': computed_tree_sha256,
            'file_count': len(entries),
        })
        """
    ).replace('__DEPENDENCY_BOOTSTRAP__', dependency_bootstrap)
    return header + "\n" + body


def _secret_and_runtime_code() -> str:
    return """
    # Read the credential from Kaggle Secrets without printing or serializing it.
    def load_hf_token():
        token = os.environ.get('HF_TOKEN')
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient

                token = UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:
                token = None
        if not token:
            raise RuntimeError(
                'HF_TOKEN is missing. Add it in Kaggle > Add-ons > Secrets, enable it, '
                'then restart and Run All.'
            )
        os.environ['HF_TOKEN'] = token
        return token

    HF_TOKEN = load_hf_token()

    # The previous cell verified and atomically restored this package before import.
    try:
        import torch
        from e2am_memrag.experiment_pipeline import (
            StageRequest,
            finalize_stage,
            prepare_stage,
            run_stage,
            safe_stop_stage,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'The E2AM-MemRAG runtime is missing from this notebook release. '
            'Use the generated release notebook, not a copied cell fragment.'
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Enable a Kaggle GPU and restart the kernel.')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f'Expected exactly one visible GPU after masking; got {torch.cuda.device_count()}.'
        )
    gpu_name = torch.cuda.get_device_name(0)
    if 'T4' not in gpu_name.upper():
        raise RuntimeError(f'Primary experiment requires a T4; visible device is {gpu_name!r}.')
    print({'visible_gpu_count': 1, 'gpu_name': gpu_name, 'hf_token_loaded': True})
    """


def _prepare_code(instance: NotebookInstance) -> str:
    spec = instance.spec
    request_code = f"""
    REQUEST = StageRequest(
        experiment_id=EXPERIMENT_ID,
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        role=ROLE,
        worker_id=WORKER_ID,
        lane_id=LANE_ID,
        lane_count=LANE_COUNT,
        notebook_name=NOTEBOOK_NAME,
        hf_repo_id=HF_REPO_ID,
        hf_repo_type=HF_REPO_TYPE,
        hf_revision=HF_REVISION,
        artifact_prefix=ARTIFACT_PREFIX,
        required_gates=REQUIRED_GATES,
        output_gate=OUTPUT_GATE,
        sync_interval_seconds=SYNC_INTERVAL_SECONDS,
        work_root=WORK_ROOT,
        stage_work_items={spec.work_items!r},
    )

    # prepare_stage authenticates once, verifies prerequisite gates at pinned commits,
    # measures storage, restores the latest valid closure/checkpoint, and rejects a
    # changed spec/environment/lane owner before scientific work begins.
    RUNTIME = prepare_stage(REQUEST, hf_token=HF_TOKEN)
    PREPARE_REPORT = RUNTIME.prepare_report
    if not PREPARE_REPORT.get('go', False):
        raise RuntimeError(f'PREPARE_NO_GO: {{PREPARE_REPORT.get("reason", "unknown")}}')
    print('PREPARE_GO')
    print({{
        'stage': STAGE_ID,
        'worker': WORKER_ID,
        'restored': PREPARE_REPORT.get('restored', False),
        'completed_units': PREPARE_REPORT.get('completed_units', 0),
    }})
    """
    patches = []
    if spec.stage_id == "03" and instance.lane_id == "lane-03":
        transport_fallback = """
    # Operational lane-03-only fallback. The anonymous public CDN signer returned
    # the same invalid key ID across all bounded retries in this Kaggle region.
    # Try anonymous access first, then use the in-memory Kaggle Secret only to
    # obtain a different authenticated blob URL. Never print or serialize the token.
    from e2am_memrag import rag_engine as _rag_engine
    from huggingface_hub import snapshot_download as _hf_snapshot_download

    def _lane03_snapshot_download(**kwargs):
        anonymous = dict(kwargs)
        anonymous['token'] = False
        try:
            return _hf_snapshot_download(**anonymous)
        except Exception as error:
            if not _rag_engine._is_transient_public_blob_signature_error(error):
                raise
            authenticated = dict(kwargs)
            authenticated['token'] = HF_TOKEN
            print(
                'MODEL_DOWNLOAD_AUTHENTICATED_URL_FALLBACK',
                {
                    'repo_id': kwargs.get('repo_id'),
                    'revision': kwargs.get('revision'),
                    'reason': 'anonymous-public-signer-invalid',
                },
                flush=True,
            )
            return _hf_snapshot_download(**authenticated)

    _rag_engine.ensure_model_snapshot.__globals__['snapshot_download'] = (
        _lane03_snapshot_download
    )
    print('LANE03_TRANSPORT_PATCH_READY: authenticated-signed-url-fallback-v1')
    """
        patches.append(_clean_source(transport_fallback))

    if spec.stage_id == "06":
        router_selection_patch = r"""
        # Protocol implementation correction applied after source verification and
        # before test access. The original implementation searched only tau<=0.85
        # and crashed when validation was infeasible. Completion and hypothesis
        # success are separate: search the full conservative range, freeze tau=1.0
        # if necessary, and preserve the infeasibility in durable policy metadata.
        import inspect as _inspect
        import contextlib as _contextlib
        import huggingface_hub as _huggingface_hub
        import huggingface_hub.constants as _hf_constants
        import signal as _signal
        import time as _time
        from e2am_memrag import experiment_pipeline as _experiment_pipeline
        from e2am_memrag import pareto_router as _pareto_router
        from e2am_memrag.rag_engine import (
            _is_transient_public_blob_signature_error as _is_blob_signature_error,
        )

        # Stage-06 restores private, checksum-addressed trace/checkpoint objects.
        # Some Kaggle regions receive an authenticated HTTP CDN URL signed by a
        # rotated key. Retry only that exact signature failure through Xet/CAS;
        # large public model downloads remain on the bounded HTTP path.
        _stage06_http_hub_download = _huggingface_hub.hf_hub_download
        _stage06_xet_timeout_seconds = 120

        class _Stage06XetTimeout(TimeoutError):
            pass

        @_contextlib.contextmanager
        def _stage06_xet_deadline(seconds):
            # Kaggle is Linux, but keep the patch import-safe on local Windows
            # development machines where SIGALRM is unavailable.
            if not hasattr(_signal, 'SIGALRM'):
                yield
                return
            previous_handler = _signal.getsignal(_signal.SIGALRM)
            previous_timer = _signal.setitimer(_signal.ITIMER_REAL, 0.0)

            def _alarm_handler(_signum, _frame):
                raise _Stage06XetTimeout(
                    f'Xet artifact transfer exceeded {seconds} seconds'
                )

            _signal.signal(_signal.SIGALRM, _alarm_handler)
            _signal.setitimer(_signal.ITIMER_REAL, float(seconds))
            try:
                yield
            finally:
                _signal.setitimer(_signal.ITIMER_REAL, 0.0)
                _signal.signal(_signal.SIGALRM, previous_handler)
                if previous_timer[0] > 0.0:
                    _signal.setitimer(
                        _signal.ITIMER_REAL,
                        previous_timer[0],
                        previous_timer[1],
                    )

        def _stage06_artifact_download(**kwargs):
            try:
                return _stage06_http_hub_download(**kwargs)
            except Exception as error:
                if not _is_blob_signature_error(error):
                    raise
                previous_disable_xet = _hf_constants.HF_HUB_DISABLE_XET
                previous_disable_xet_env = os.environ.get('HF_HUB_DISABLE_XET')
                _hf_constants.HF_HUB_DISABLE_XET = False
                os.environ['HF_HUB_DISABLE_XET'] = '0'
                retry = dict(kwargs)
                retry['force_download'] = True
                print(
                    'STAGE06_ARTIFACT_XET_FALLBACK',
                    {
                        'repo_id': kwargs.get('repo_id'),
                        'revision': kwargs.get('revision'),
                        'filename': kwargs.get('filename'),
                        'reason': 'authenticated-http-signer-invalid',
                    },
                    flush=True,
                )
                previous_xet_concurrency = os.environ.get(
                    'HF_XET_NUM_CONCURRENT_RANGE_GETS'
                )
                os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS'] = '1'
                previous_hub_timeout = getattr(
                    _hf_constants, 'HF_HUB_DOWNLOAD_TIMEOUT', None
                )
                if previous_hub_timeout is not None:
                    _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = min(
                        int(previous_hub_timeout),
                        _stage06_xet_timeout_seconds,
                    )
                try:
                    for xet_attempt in range(1, 3):
                        try:
                            with _stage06_xet_deadline(
                                _stage06_xet_timeout_seconds
                            ):
                                return _stage06_http_hub_download(**retry)
                        except _Stage06XetTimeout as timeout_error:
                            if xet_attempt == 2:
                                raise RuntimeError(
                                    'STAGE06_ARTIFACT_XET_TIMEOUT: the private '
                                    'artifact transfer made no progress within '
                                    f'{_stage06_xet_timeout_seconds} seconds '
                                    'per attempt; partial cache is resumable. '
                                    'Rerun this same notebook when Kaggle/HF '
                                    'network service is healthy.'
                                ) from timeout_error
                            print(
                                'STAGE06_ARTIFACT_XET_RETRY',
                                {
                                    'attempt': xet_attempt + 1,
                                    'wait_seconds': 5,
                                    'reason': 'bounded-transfer-timeout',
                                },
                                flush=True,
                            )
                            _time.sleep(5)
                finally:
                    if previous_hub_timeout is not None:
                        _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = previous_hub_timeout
                    if previous_xet_concurrency is None:
                        os.environ.pop('HF_XET_NUM_CONCURRENT_RANGE_GETS', None)
                    else:
                        os.environ['HF_XET_NUM_CONCURRENT_RANGE_GETS'] = (
                            previous_xet_concurrency
                        )
                    _hf_constants.HF_HUB_DISABLE_XET = previous_disable_xet
                    if previous_disable_xet_env is None:
                        os.environ.pop('HF_HUB_DISABLE_XET', None)
                    else:
                        os.environ['HF_HUB_DISABLE_XET'] = previous_disable_xet_env

        _huggingface_hub.hf_hub_download = _stage06_artifact_download
        print('STAGE06_ARTIFACT_TRANSPORT_PATCH_READY: signed-http-to-xet-v1')

        STAGE06_PROTOCOL_AMENDMENT = {
            'amendment_id': 'stage06-threshold-completion-separation-v1',
            'timing': 'after-validation-before-test-access',
            'reason': 'coarse-threshold-grid-and-infeasibility-crash',
            'success_floor': 0.80,
            'minimum_execution_coverage': 0.90,
            'maximum_abstention_rate': 0.20,
            'fallback_tau': 1.0,
            'hypothesis_claim_allowed_when_infeasible': False,
        }
        _STAGE06_VALIDATION_SELECTION = {}

        def _replace_stage06_once(source, old, new, label):
            if source.count(old) != 1:
                raise RuntimeError(
                    f'STAGE06_PATCH_SOURCE_MISMATCH: {label} count={source.count(old)}'
                )
            return source.replace(old, new, 1)

        _fit_source = _inspect.getsource(_pareto_router.fit_router)
        _fit_source = _replace_stage06_once(
            _fit_source,
            'candidate_taus = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)',
            'candidate_taus = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)',
            'candidate-grid',
        )
        _fit_source = _replace_stage06_once(
            _fit_source,
            'selected_tau: float | None = None\n    for candidate in candidate_taus:',
            'selected_tau: float | None = None\n    validation_candidates = []\n    for candidate in candidate_taus:',
            'candidate-report-initialization',
        )
        _fit_source = _replace_stage06_once(
            _fit_source,
            (
                '        if (\n'
                '            outcomes\n'
                '            and sum(outcomes) / len(outcomes) >= 0.80\n'
                '            and sum(execution) / len(execution) >= 0.90\n'
                '            and sum(abstentions) / len(abstentions) <= 0.20\n'
                '        ):\n'
                '            selected_tau = candidate\n'
                '            break\n'
            ),
            (
                '        success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0\n'
                '        execution_coverage = sum(execution) / len(execution) if execution else 0.0\n'
                '        abstention_rate = sum(abstentions) / len(abstentions) if abstentions else 1.0\n'
                '        candidate_feasible = bool(\n'
                '            outcomes\n'
                '            and success_rate >= 0.80\n'
                '            and execution_coverage >= 0.90\n'
                '            and abstention_rate <= 0.20\n'
                '        )\n'
                '        validation_candidates.append({\n'
                '            "tau": candidate,\n'
                '            "success_rate": success_rate,\n'
                '            "execution_coverage": execution_coverage,\n'
                '            "abstention_rate": abstention_rate,\n'
                '            "feasible": candidate_feasible,\n'
                '        })\n'
                '        if candidate_feasible:\n'
                '            selected_tau = candidate\n'
                '            break\n'
            ),
            'candidate-metrics',
        )
        _fit_source = _replace_stage06_once(
            _fit_source,
            (
                '    if selected_tau is None:\n'
                '        raise RuntimeError(\n'
                '            "No validation threshold satisfies the frozen success, execution-coverage, "\n'
                '            "and abstention constraints"\n'
                '        )\n'
                '    provisional.tau = selected_tau\n'
                '    return provisional\n'
            ),
            (
                '    validation_feasible = selected_tau is not None\n'
                '    if selected_tau is None:\n'
                '        selected_tau = 1.0\n'
                '        print(\n'
                '            "ROUTER_VALIDATION_INFEASIBLE_POLICY_FROZEN",\n'
                '            {"tau": selected_tau, "candidates": validation_candidates},\n'
                '        )\n'
                '    else:\n'
                '        print(\n'
                '            "ROUTER_VALIDATION_THRESHOLD_SELECTED",\n'
                '            {"tau": selected_tau, "candidates_evaluated": len(validation_candidates)},\n'
                '        )\n'
                '    provisional.tau = selected_tau\n'
                '    selection_report = {\n'
                '        "schema_version": 1,\n'
                '        "feasible": validation_feasible,\n'
                '        "selected_tau": selected_tau,\n'
                '        "selection_mode": (\n'
                '            "lowest-feasible-frozen-grid"\n'
                '            if validation_feasible\n'
                '            else "fail-closed-tau-1.0-validation-infeasible"\n'
                '        ),\n'
                '        "candidates": validation_candidates,\n'
                '        "protocol_amendment": dict(STAGE06_PROTOCOL_AMENDMENT),\n'
                '        "test_accessed": False,\n'
                '    }\n'
                '    _STAGE06_VALIDATION_SELECTION.clear()\n'
                '    _STAGE06_VALIDATION_SELECTION.update(selection_report)\n'
                '    provisional.validation_selection = dict(selection_report)\n'
                '    return provisional\n'
            ),
            'infeasibility-completion-separation',
        )
        _pareto_router.STAGE06_PROTOCOL_AMENDMENT = STAGE06_PROTOCOL_AMENDMENT
        _pareto_router._STAGE06_VALIDATION_SELECTION = _STAGE06_VALIDATION_SELECTION
        exec(compile(_fit_source, '<stage06-router-selection-patch>', 'exec'), _pareto_router.__dict__)
        _experiment_pipeline.fit_router = _pareto_router.fit_router

        _original_stage06_write_json = _experiment_pipeline._write_json

        def _stage06_write_json_with_selection(runtime, relative_path, value):
            if (
                runtime.request.stage_id == '06'
                and relative_path in {'training_history.json', 'frozen_policy.json'}
            ):
                value = dict(value)
                value['validation_selection'] = dict(_STAGE06_VALIDATION_SELECTION)
            return _original_stage06_write_json(runtime, relative_path, value)

        _experiment_pipeline._write_json = _stage06_write_json_with_selection
        print('STAGE06_ROUTER_SELECTION_PATCH_READY: completion-separation-v1')
        """
        patches.append(_clean_source(router_selection_patch))

    if spec.stage_id == "09":
        hypothesis_guard_patch = """
        # Preserve the Stage-06 validation decision in the final hypothesis claim.
        # A completed but validation-infeasible policy remains reportable, but it
        # cannot be promoted to a successful confirmatory result.
        import inspect as _inspect
        from e2am_memrag import experiment_pipeline as _experiment_pipeline

        _stage09_source = _inspect.getsource(_experiment_pipeline._stage09)
        _stage09_old = (
            '"hypothesis_pass": quality_pass and energy_pass and operating_constraints_pass,'
        )
        _stage09_new = (
            '"hypothesis_pass": (quality_pass and energy_pass and operating_constraints_pass '
            'and bool(frozen_policy.get("validation_selection", {}).get("feasible", True))),'
        )
        if _stage09_source.count(_stage09_old) != 1:
            raise RuntimeError('STAGE09_VALIDATION_GUARD_SOURCE_MISMATCH')
        _stage09_source = _stage09_source.replace(_stage09_old, _stage09_new, 1)
        exec(compile(_stage09_source, '<stage09-validation-guard>', 'exec'), _experiment_pipeline.__dict__)
        _experiment_pipeline._STAGE_HANDLERS['09'] = _experiment_pipeline._stage09
        print('STAGE09_VALIDATION_GUARD_READY: validation-infeasible-cannot-pass-v1')
        """
        patches.append(_clean_source(hypothesis_guard_patch))

    return "\n\n".join([*patches, _clean_source(request_code)])


def _execute_code() -> str:
    return """
    # run_stage owns deterministic unit assignment and replay-bounded result checkpoints,
    # 20-minute dirty sync, major-boundary sync, and upload-free energy blocks.
    STAGE_RESULT = None
    try:
        STAGE_RESULT = run_stage(RUNTIME)
    except KeyboardInterrupt:
        stop_report = safe_stop_stage(RUNTIME, reason='keyboard_interrupt')
        if stop_report.get('remote_verified', False):
            print('SAFE_STOP_VERIFIED')
        else:
            print('SAFE_STOP_INCOMPLETE:', stop_report.get('local_resume_path'))
        raise
    except BaseException:
        # Preserve the original exception while making a bounded best-effort closure.
        try:
            stop_report = safe_stop_stage(RUNTIME, reason='stage_exception')
            status = (
                'SAFE_STOP_VERIFIED'
                if stop_report.get('remote_verified')
                else 'SAFE_STOP_INCOMPLETE'
            )
            print(status)
        except Exception as stop_error:
            print('SAFE_STOP_INCOMPLETE:', type(stop_error).__name__)
        raise

    if STAGE_RESULT is None:
        raise RuntimeError('Stage returned no result; finalization is forbidden.')
    print({
        'new_units': STAGE_RESULT.get('new_units', 0),
        'reused_units': STAGE_RESULT.get('reused_units', 0),
        'failed_units': STAGE_RESULT.get('failed_units', 0),
    })
    """


def _finalize_code() -> str:
    return """
    # Finalization verifies every dependency, commits the stage gate with its closure,
    # pins the returned SHA, and downloads the receipt/seal/pointer for checksum proof.
    FINAL_REPORT = finalize_stage(RUNTIME, STAGE_RESULT)
    if not FINAL_REPORT.get('remote_verified', False):
        raise RuntimeError(
            'FINAL_NO_GO: local artifacts exist but the remote closure is not verified. '
            f'Rerun this same notebook to resume: {FINAL_REPORT.get("local_resume_path")}'
        )
    if FINAL_REPORT.get('output_gate') != OUTPUT_GATE:
        raise RuntimeError('FINAL_NO_GO: the verified gate does not match this notebook.')

    print('REMOTE_CLOSURE_VERIFIED')
    print('STAGE_COMPLETE:', STAGE_ID, OUTPUT_GATE)
    print({
        'revision': HF_REVISION,
        'commit_sha': FINAL_REPORT.get('commit_sha'),
        'artifact_prefix': ARTIFACT_PREFIX,
        'output_gate': OUTPUT_GATE,
    })
    """


def _troubleshooting(instance: NotebookInstance) -> str:
    return f"""
    ## Troubleshooting and exact resume

    - **`HF_TOKEN is missing`**: enable the Kaggle Secret named `HF_TOKEN`; do not paste
      a token into the notebook.
    - **401/403**: the uploader stops repeated attempts. Correct repository write access,
      then rerun `{instance.filename}`.
    - **429/rate limit**: sealed work stays local while the runtime honors `Retry-After`
      and exponential backoff. The normal dirty-state target is 1,200 seconds.
    - **Torch imported before mask / wrong GPU count**: restart the kernel and use Run All.
      Do not execute setup cells out of order.
    - **OOM or unsafe VRAM**: the declared route is recorded as failed. The notebook never
      silently changes batch size, precision, context, or device placement.
    - **Storage gate**: new work stops before the emergency reserve is consumed. Only
      reproducible caches may be deleted after their source/checksum is recorded.
    - **Interrupted upload or lost acknowledgement**: rerun this same fixed notebook.
      Recovery checks deterministic receipts and pinned remote hashes before retrying.
    - **Hard Kaggle termination**: cleanup cannot run. The next Run All restores the latest
      verified closure and replays at most the smallest unsealed unit.
    - **Prerequisite gate missing**: finish the earlier numbered notebook(s); never bypass
      the gate or copy artifacts between stage prefixes manually.

    A run is complete only after both `REMOTE_CLOSURE_VERIFIED` and `STAGE_COMPLETE`
    appear. Keep the Kaggle session open if only `SAFE_STOP_INCOMPLETE` is shown.
    """


def build_notebook(
    instance: NotebookInstance,
    runtime_bundle: RuntimeBundle | None = None,
) -> dict[str, object]:
    """Build one deterministic, unexecuted notebook dictionary."""

    runtime_bundle = runtime_bundle or build_runtime_bundle()
    cells: list[dict[str, object]] = []

    def add_markdown(source: str) -> None:
        cells.append(_markdown(instance, len(cells), source))

    def add_code(source: str) -> None:
        cells.append(_code(instance, len(cells), source))

    add_markdown(_intro(instance))
    add_code(_settings_code(instance))
    add_markdown(_runbook(instance))
    add_markdown(
        """
        ## Restore the embedded verified runtime

        This notebook contains the complete current `src/e2am_memrag`, `configs`,
        `requirements-kaggle.txt`, and `pyproject.toml` runtime. The next cell verifies
        the ZIP hash, exact member set, every member hash, and the source-tree hash before
        atomically replacing `/kaggle/working/E2AM-MemRAG`. It keeps Kaggle's existing
        Torch/CUDA packages and adds `src` to `sys.path` before importing the pipeline.
        """
    )
    add_code(
        _runtime_bootstrap_code(
            runtime_bundle,
            resilient_dependency_install=(
                instance.spec.stage_id == "05" and instance.lane_id == "lane-01"
            ),
        )
    )
    add_code(_secret_and_runtime_code())
    add_markdown(
        """
        ## Preflight and restore

        This cell is read/verify-first. It validates the remote prerequisites and lane
        binding before it can append data. A corrupt or unsealed checkpoint is ignored in
        favor of the previous verified checkpoint.
        """
    )
    add_code(_prepare_code(instance))
    add_markdown(
        """
        ## Execute resumable work

        The runtime uses immutable logical unit IDs. Physical execution is at least once;
        aggregation accepts exactly one canonical artifact for each unit. A manual interrupt
        requests a checkpoint and immediate verified Hub closure at the next safe boundary.
        """
    )
    add_code(_execute_code())
    add_markdown(
        """
        ## Verify and publish the stage gate

        Normal completion forces a remote closure even if the 20-minute timer is not due.
        The gate and pointer are published only after every referenced artifact exists.
        """
    )
    add_code(_finalize_code())
    add_markdown(_troubleshooting(instance))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    validate_notebook(instance, notebook)
    return notebook


def build_all_notebooks() -> dict[str, dict[str, object]]:
    """Return all concrete notebooks without writing to disk."""

    runtime_bundle = build_runtime_bundle()
    notebooks = {
        instance.filename: build_notebook(instance, runtime_bundle)
        for instance in iter_instances()
    }
    if len(notebooks) != 22:
        raise AssertionError(f"Expected 22 concrete notebooks, got {len(notebooks)}")
    return notebooks


def validate_notebook(
    instance: NotebookInstance,
    notebook: Mapping[str, object],
) -> None:
    """Enforce the generator's safety and reproducibility invariants."""

    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        raise ValueError(f"{instance.filename}: unsupported notebook format")
    metadata = notebook.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{instance.filename}: metadata is missing")
    language_info = metadata.get("language_info")
    if not isinstance(language_info, Mapping) or language_info.get("version") != "3.10":
        raise ValueError(f"{instance.filename}: Python 3.10 metadata is required")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{instance.filename}: cells are missing")
    ids: set[str] = set()
    code_sources: list[str] = []
    markdown_sources: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError(f"{instance.filename}: invalid cell")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or cell_id in ids:
            raise ValueError(f"{instance.filename}: invalid or duplicate cell ID")
        ids.add(cell_id)
        if cell.get("cell_type") == "markdown":
            source = cell.get("source")
            if isinstance(source, str):
                markdown_sources.append(source)
            elif isinstance(source, list) and all(isinstance(line, str) for line in source):
                markdown_sources.append("".join(source))
            else:
                raise ValueError(f"{instance.filename}: markdown source must be text lines")
            continue
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs") != []:
            raise ValueError(f"{instance.filename}: generated code cells must be unexecuted")
        source = cell.get("source")
        if isinstance(source, str):
            source_text = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            source_text = "".join(source)
        else:
            raise ValueError(f"{instance.filename}: code source must be text lines")
        code_sources.append(source_text)

    if not code_sources:
        raise ValueError(f"{instance.filename}: no code cells")
    if len(markdown_sources) < 2:
        raise ValueError(f"{instance.filename}: intro or runbook markdown is missing")
    intro, runbook = markdown_sources[:2]
    if (
        not intro.startswith(f"# Stage {instance.spec.stage_id}:")
        or f"\n**File:** `{instance.filename}`" not in intro
        or not runbook.startswith("## Numbered Kaggle runbook\n\n1. Open")
        or "\n### Work performed\n\n- " not in runbook
    ):
        raise ValueError(f"{instance.filename}: intro/runbook Markdown indentation is invalid")
    first_code = code_sources[0]
    if "CUDA_VISIBLE_DEVICES" not in first_code or "import torch" in first_code:
        raise ValueError(f"{instance.filename}: first code cell must mask CUDA before Torch")
    if f"SYNC_INTERVAL_SECONDS = {SYNC_INTERVAL_SECONDS}" not in first_code:
        raise ValueError(f"{instance.filename}: 1,200-second sync constant is missing")
    if (
        "os.environ['HF_HUB_DISABLE_XET'] = '1'" not in first_code
        or "os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'" not in first_code
    ):
        raise ValueError(f"{instance.filename}: bounded Hub transport is missing")
    joined = "\n".join(code_sources)
    if joined.find("CUDA_VISIBLE_DEVICES") > joined.find("import torch"):
        raise ValueError(f"{instance.filename}: Torch appears before the CUDA mask")
    if "get_secret('HF_TOKEN')" not in joined or "StageRequest(" not in joined:
        raise ValueError(f"{instance.filename}: secret or pipeline contract is missing")
    if (
        "EMBEDDED_RUNTIME_B64" not in joined
        or "Embedded runtime archive SHA-256 mismatch" not in joined
        or "Runtime member SHA-256 mismatch" not in joined
        or "sys.path[:] = [project_src" not in joined
    ):
        raise ValueError(f"{instance.filename}: standalone runtime contract is missing")
    lane03_patch = "LANE03_TRANSPORT_PATCH_READY"
    if instance.spec.stage_id == "03" and instance.lane_id == "lane-03":
        if lane03_patch not in joined or "authenticated['token'] = HF_TOKEN" not in joined:
            raise ValueError(f"{instance.filename}: lane-03 URL fallback is missing")
    elif lane03_patch in joined:
        raise ValueError(f"{instance.filename}: lane-03 URL fallback leaked to another file")
    lane01_dependency_patch = "LANE01_RESILIENT_DEPENDENCY_BOOTSTRAP_V1"
    if instance.spec.stage_id == "05" and instance.lane_id == "lane-01":
        required_dependency_markers = (
            lane01_dependency_patch,
            "DEPENDENCY_PRECHECK",
            "DEPENDENCY_NETWORK_RETRY",
            "DEPENDENCY_SAFE_STOP",
            "DEPENDENCY_INSTALL_VERIFIED",
        )
        if any(marker not in joined for marker in required_dependency_markers):
            raise ValueError(f"{instance.filename}: resilient dependency bootstrap is incomplete")
    elif lane01_dependency_patch in joined:
        raise ValueError(
            f"{instance.filename}: lane-01 dependency bootstrap leaked to another file"
        )

    stage06_patch = "STAGE06_ROUTER_SELECTION_PATCH_READY"
    if instance.spec.stage_id == "06":
        required_router_markers = (
            stage06_patch,
            "STAGE06_ARTIFACT_TRANSPORT_PATCH_READY",
            "STAGE06_ARTIFACT_XET_FALLBACK",
            "ROUTER_VALIDATION_THRESHOLD_SELECTED",
            "ROUTER_VALIDATION_INFEASIBLE_POLICY_FROZEN",
            "hypothesis_claim_allowed_when_infeasible",
        )
        if any(marker not in joined for marker in required_router_markers):
            raise ValueError(f"{instance.filename}: Stage-06 selection patch is incomplete")
    elif stage06_patch in joined:
        raise ValueError(f"{instance.filename}: Stage-06 selection patch leaked")

    stage09_guard = "STAGE09_VALIDATION_GUARD_READY"
    if instance.spec.stage_id == "09":
        if stage09_guard not in joined or "validation_selection" not in joined:
            raise ValueError(f"{instance.filename}: Stage-09 validation guard is missing")
    elif stage09_guard in joined:
        raise ValueError(f"{instance.filename}: Stage-09 validation guard leaked")


def render_notebook(notebook: Mapping[str, object]) -> bytes:
    """Render stable UTF-8 JSON with a trailing newline."""

    text = json.dumps(
        notebook,
        ensure_ascii=False,
        indent=1,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def emit_apply_patch(filename: str, notebook: Mapping[str, object]) -> str:
    """Render one new notebook as an apply_patch-compatible add-file patch."""

    known_names = {instance.filename for instance in iter_instances()}
    if filename not in known_names:
        raise ValueError(f"Unknown generated notebook filename: {filename}")
    rendered = render_notebook(notebook).decode("utf-8")
    body = "".join(f"+{line}" for line in rendered.splitlines(keepends=True))
    return (
        "*** Begin Patch\n"
        f"*** Add File: notebooks/{filename}\n"
        f"{body}"
        "*** End Patch\n"
    )


def notebook_digests() -> dict[str, str]:
    """Return deterministic SHA-256 values without writing notebooks."""

    return {
        name: hashlib.sha256(render_notebook(notebook)).hexdigest()
        for name, notebook in build_all_notebooks().items()
    }


def write_notebooks(output_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Atomically write every notebook after all in-memory validation succeeds."""

    notebooks = build_all_notebooks()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / name for name in notebooks]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        raise FileExistsError(f"Refusing to overwrite existing notebooks: {names}")

    written: list[Path] = []
    for destination, notebook in zip(destinations, notebooks.values()):
        payload = render_notebook(notebook)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            written.append(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="build and validate all notebooks in memory without writing files",
    )
    action.add_argument(
        "--write-dir",
        type=Path,
        help="explicitly write all validated notebooks into this directory",
    )
    action.add_argument(
        "--emit-patch",
        metavar="NOTEBOOK_FILENAME",
        help="print an apply_patch add-file patch for one exact generated filename",
    )
    action.add_argument(
        "--emit-content-chunk",
        nargs=2,
        metavar=("NOTEBOOK_FILENAME", "CHUNK_INDEX"),
        help=(
            "print one base64-encoded chunk of the rendered notebook; this keeps "
            "large apply_patch materialization below transport output limits"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --write-dir to replace notebook files with the same names",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.overwrite and args.write_dir is None:
        raise SystemExit("--overwrite requires --write-dir")

    if args.emit_patch is not None:
        notebooks = build_all_notebooks()
        try:
            notebook = notebooks[args.emit_patch]
        except KeyError as error:
            choices = ", ".join(notebooks)
            raise SystemExit(
                f"Unknown notebook {args.emit_patch!r}. Choose one of: {choices}"
            ) from error
        print(emit_apply_patch(args.emit_patch, notebook), end="")
        return 0

    if args.emit_content_chunk is not None:
        filename, raw_index = args.emit_content_chunk
        notebooks = build_all_notebooks()
        try:
            notebook = notebooks[filename]
        except KeyError as error:
            choices = ", ".join(notebooks)
            raise SystemExit(
                f"Unknown notebook {filename!r}. Choose one of: {choices}"
            ) from error
        try:
            index = int(raw_index)
        except ValueError as error:
            raise SystemExit("CHUNK_INDEX must be an integer") from error
        payload = render_notebook(notebook)
        chunk_count = (len(payload) + PATCH_EXPORT_CHUNK_BYTES - 1) // PATCH_EXPORT_CHUNK_BYTES
        if not 0 <= index < chunk_count:
            raise SystemExit(f"CHUNK_INDEX must be in [0, {chunk_count - 1}]")
        start = index * PATCH_EXPORT_CHUNK_BYTES
        chunk = payload[start : start + PATCH_EXPORT_CHUNK_BYTES]
        encoded = base64.b64encode(chunk).decode("ascii")
        print(f"{index}:{chunk_count}:{encoded}")
        return 0

    digests = notebook_digests()
    if args.write_dir is not None:
        written = write_notebooks(args.write_dir, overwrite=args.overwrite)
        for path in written:
            print(f"WROTE {path}")
        return 0

    if args.check:
        combined = hashlib.sha256(
            "".join(f"{name}:{digest}\n" for name, digest in digests.items()).encode("utf-8")
        ).hexdigest()
        print(f"VALID: {len(digests)} notebooks; set_sha256={combined}")
        return 0

    for name, digest in digests.items():
        print(f"{name}\t{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
