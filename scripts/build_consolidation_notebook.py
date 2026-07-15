"""Build the standalone Kaggle Hugging Face consolidation notebook.

The builder writes JSON to stdout.  Repository updates are intentionally left to
the caller so the generated notebook can be reviewed and applied atomically.
"""

from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat


SOURCE_RELEASE_LOCK = [
    {"stage_id": "00", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-00-stage-00-coordinator", "commit_sha": "70e81d9c4af5bdb2d2d82df331fafc9ad7fb096f", "manifest_sha256": "36471ac0f67dc39060c26b8b8c7d6a5ba49dcc4be6a679ac8a0a383a659878c0", "artifact_records": 31, "artifact_bytes": 2409907},
    {"stage_id": "00", "owner": "test-vault", "branch": "stage-e2am-memrag-v3r1-00-stage-00-test-vault", "commit_sha": "d36c394d8c904360fa2884dfeeea42d407d09095", "manifest_sha256": "2d18da877505fa40fe71836ce5af4fe8a4eed3bf31a7adb1a1b7b1e066efb3ff", "artifact_records": 1, "artifact_bytes": 23045},
    {"stage_id": "01", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-01-stage-01-coordinator", "commit_sha": "6c8d55f9490e6297a27ba77d15720f4f9ab6f4c9", "manifest_sha256": "120a8365d3deaa45e02ff011cdf5be83f6f4f9537aebe7c79c366460e0e70224", "artifact_records": 36, "artifact_bytes": 20736229},
    {"stage_id": "02", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-02-stage-02-coordinator", "commit_sha": "82af6953b5813859cb819bc5dd0a8979e0ba43c4", "manifest_sha256": "e9b04ec2074ed4eeb13a7656d6bc8ea2cca790571af97751f04c78ab0e4e839b", "artifact_records": 9, "artifact_bytes": 9998217},
    {"stage_id": "03", "owner": "lane-00", "branch": "stage-e2am-memrag-v3r1-03-stage-03-lane-00", "commit_sha": "67e6eee33de1be767325761348953e6d2c38cf5b", "manifest_sha256": "fce35a9e91d12c2e9a2b0ab059b72e9eebab155195a864a2e1dfd0b5b4e375aa", "artifact_records": 9, "artifact_bytes": 1892377},
    {"stage_id": "03", "owner": "lane-01", "branch": "stage-e2am-memrag-v3r1-03-stage-03-lane-01", "commit_sha": "f06c718ecb8b0ff25bcf676905fdfa6427cb543a", "manifest_sha256": "408bdcf4a933e2b4ce5ede6a32d73b267f13f7f0a3588a7c25e6e03842c3dcc5", "artifact_records": 7, "artifact_bytes": 962053},
    {"stage_id": "03", "owner": "lane-02", "branch": "stage-e2am-memrag-v3r1-03-stage-03-lane-02", "commit_sha": "f602fc91515a4b078d77d3a4f2adf0a8f149bea0", "manifest_sha256": "c8ce2401d1336be40dba3c8c7229331abe23e7c54bf5493b2d466e0c65d05b8c", "artifact_records": 8, "artifact_bytes": 1609515},
    {"stage_id": "03", "owner": "lane-03", "branch": "stage-e2am-memrag-v3r1-03-stage-03-lane-03", "commit_sha": "f51fd1407e05df2434caa686cffeb239efad885e", "manifest_sha256": "960011716fde4064b4f77f3b818652a1d69534b81e1c7d8ff9c17cd26f1a8cce", "artifact_records": 8, "artifact_bytes": 1459303},
    {"stage_id": "04", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-04-stage-04-coordinator", "commit_sha": "9835b0546ace579006cb78520e610d46fc9df2cb", "manifest_sha256": "ec2069c6c10f9c68fe47c649c3320c5617c1c3a4c23fd68079d073f4315e4a3a", "artifact_records": 7, "artifact_bytes": 15014694},
    {"stage_id": "05", "owner": "lane-00", "branch": "stage-e2am-memrag-v3r1-05-stage-05-lane-00", "commit_sha": "da21c0deb26113b5b541add8c81845d5780c1f90", "manifest_sha256": "d390c7061c14eab428951fcc1f091e0d8837be2cab4637054ab1ab518015fec4", "artifact_records": 28, "artifact_bytes": 11600094},
    {"stage_id": "05", "owner": "lane-01", "branch": "stage-e2am-memrag-v3r1-05-stage-05-lane-01", "commit_sha": "d5b4de4408ae904d4c3e316f6f8c69ede6d26a5f", "manifest_sha256": "f337fdac3a9d6287787afc21f3df485f85d713e8752231bc3d974f1df515b7de", "artifact_records": 9, "artifact_bytes": 1397737},
    {"stage_id": "05", "owner": "lane-02", "branch": "stage-e2am-memrag-v3r1-05-stage-05-lane-02", "commit_sha": "5b505707b7baafcd9bd175b8872c28eb4ea4e388", "manifest_sha256": "c19405c66ca3f5b99eb043674a848531bbc25953a57c48bd8d769a78bc139af7", "artifact_records": 19, "artifact_bytes": 7093174},
    {"stage_id": "05", "owner": "lane-03", "branch": "stage-e2am-memrag-v3r1-05-stage-05-lane-03", "commit_sha": "0ba3b1d148f5cc05941f0de6838ba3638ffe3be1", "manifest_sha256": "1f2c043ec512dec5d836fbea90c5af40fe444bbbcf52e909e2a249f946d1732a", "artifact_records": 14, "artifact_bytes": 5895457},
    {"stage_id": "06", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-06-stage-06-coordinator", "commit_sha": "1b765103f579e972f90a1fb52f2201cb08c19066", "manifest_sha256": "05b72dda0b43de75df0fc580ff764b3f35cd8ad905afbce949d7c6a9dac3304d", "artifact_records": 16, "artifact_bytes": 20715468},
    {"stage_id": "07", "owner": "lane-00", "branch": "stage-e2am-memrag-v3r1-07-stage-07-lane-00", "commit_sha": "fcceceed653ec4634994acb1d5e14d95dc417b81", "manifest_sha256": "e9cdf3bfe5efae3fbf35008eccb1a6d5a62c9b7a29dece8eb3ceea24cc3e3b65", "artifact_records": 12, "artifact_bytes": 3706031},
    {"stage_id": "07", "owner": "lane-01", "branch": "stage-e2am-memrag-v3r1-07-stage-07-lane-01", "commit_sha": "6c807dfdf5d01c0933a97dd6d0679df886299712", "manifest_sha256": "7c277d8381366f9c450259975e0978d2ddf1d3bb7ee7a765b8d7c43123b974a7", "artifact_records": 6, "artifact_bytes": 281087},
    {"stage_id": "07", "owner": "lane-02", "branch": "stage-e2am-memrag-v3r1-07-stage-07-lane-02", "commit_sha": "30a0e834d074502a8b44610385ef24a0be507a5e", "manifest_sha256": "51a18c1873141e44a7f7bea8a40076f850b37bdaf719effb647a1d085a896e53", "artifact_records": 10, "artifact_bytes": 2284158},
    {"stage_id": "07", "owner": "lane-03", "branch": "stage-e2am-memrag-v3r1-07-stage-07-lane-03", "commit_sha": "0ffdd4cfbb1e988a61fc0950f4bf36431e67e647", "manifest_sha256": "ce283a0b03e94002b64efc05d67873e8d70b5c285834c69bdfc3cfc23c092afe", "artifact_records": 9, "artifact_bytes": 2043878},
    {"stage_id": "08", "owner": "lane-00", "branch": "stage-e2am-memrag-v3r1-08-stage-08-lane-00", "commit_sha": "fba12a51f33cb5563349cc9688952d791eb59f1d", "manifest_sha256": "be7e24f11640f595728a3e5a59b4a81a68e8d30d78c9de3133c5fb8e5e87d191", "artifact_records": 8, "artifact_bytes": 1722275},
    {"stage_id": "08", "owner": "lane-01", "branch": "stage-e2am-memrag-v3r1-08-stage-08-lane-01", "commit_sha": "83cabdaeb420472a750ace773c0c1f215c4eca93", "manifest_sha256": "355d4181acca88852a61e44253c822abe28c6d3fdc1be0535fcdcea24c0b5c44", "artifact_records": 8, "artifact_bytes": 1729455},
    {"stage_id": "08", "owner": "lane-02", "branch": "stage-e2am-memrag-v3r1-08-stage-08-lane-02", "commit_sha": "25354f0252b38d111b9d05100b091487e4ccb0e7", "manifest_sha256": "c4f0b750a1851c4fddc4cbca865960b5d3416c2c4bc630b15f5b1176bb59d759", "artifact_records": 8, "artifact_bytes": 1722349},
    {"stage_id": "08", "owner": "lane-03", "branch": "stage-e2am-memrag-v3r1-08-stage-08-lane-03", "commit_sha": "c66f55fa123c85703204ca16cbb8712a792c1b3e", "manifest_sha256": "165d4386765030334eb202e0bbcf288f029601558ca3f86636f4b3a30639b0b9", "artifact_records": 8, "artifact_bytes": 1729828},
    {"stage_id": "09", "owner": "coordinator", "branch": "stage-e2am-memrag-v3r1-09-stage-09-coordinator", "commit_sha": "4d3e111ddfe41247b511ad0fc1a413baacee7864", "manifest_sha256": "c67191cf2afcc1a9d2be530d76049621d67f8b45528fc341bbf8f936cb6ecdf3", "artifact_records": 11, "artifact_bytes": 11528142},
]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def build_notebook(runtime_source: str) -> dict:
    lock_literal = pformat(SOURCE_RELEASE_LOCK, width=120, sort_dicts=False)
    cells = [
        markdown("""# 10 — Consolidate the verified Hugging Face release

This CPU-only Kaggle notebook creates a **new, non-destructive convenience copy**
of the completed `e2am-memrag-v3r1` experiment. It does not rerun any scientific
experiment and never deletes, renames, rewrites, merges, or force-pushes a source
branch.

## Numbered runbook

1. Create a fresh Kaggle notebook and choose **Accelerator: None**. A GPU provides
   no benefit here.
2. Turn **Internet on**.
3. Add a Kaggle secret named `HF_TOKEN` with write access to
   `Shanmuk4622/E2AM-MemRAG-Traces`. The token is never printed or serialized.
4. Run all cells. Use only one live copy of this consolidation notebook at a time.
5. Expect roughly **8–12 hours**: all 282 artifacts are downloaded from immutable
   commits, checksummed, archived, uploaded, and downloaded again for verification
   under a conservative 96/128 weighted-operations-per-hour ceiling. All Hub reads
   are authenticated with HF_TOKEN to avoid the low unauthenticated rate limit.
6. If Kaggle stops, open a fresh session and run all cells again. The notebook reads
   remote `PROGRESS.json`, verifies completed archives, and resumes at the next branch.
7. A clean or interrupted download creates no unique scientific state. Dirty state
   is committed after every source-branch closure (a major unit); clean heartbeats
   are skipped. `KeyboardInterrupt` verifies the last remote progress seal.
8. Completion requires `CONSOLIDATION_COMPLETE` followed by `FINAL_GO`.

## Output layout

- Original 23 v3r1 stage branches: unchanged and still recoverable.
- Older v2/v3 branches: unchanged, recorded, and excluded from v3r1 evidence.
- New branch: `consolidated-e2am-memrag-v3r1`.
- Human-readable paper data on `main`:
  `experiments/e2am-memrag-v3r1/paper/`.
- Frozen `experiments/e2am-memrag-v3r1/RELEASE.json`: never overwritten.

Experiment completion and hypothesis success are separate. The completed release
passed its closure/fresh-restore gate; the predeclared confirmatory hypothesis did
not pass and remains reported as such.
"""),
        code("""# Environment must be frozen before importing Hub/CUDA-aware packages.
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HOME"] = "/kaggle/working/e2am_consolidation/hf-home"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORK_ROOT = Path("/kaggle/working/e2am_consolidation")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print("ENVIRONMENT_READY", {"accelerator": "CPU", "work_root": str(WORK_ROOT)})
"""),
        code("""# Import first; install only when Kaggle genuinely lacks the required Hub API.
import importlib
import subprocess
import sys
import time

def hub_api_ready():
    try:
        module = importlib.import_module("huggingface_hub")
        return all(hasattr(module, name) for name in (
            "CommitOperationAdd", "HfApi", "hf_hub_download"
        ))
    except Exception:
        return False

if not hub_api_ready():
    last_output = ""
    for attempt, wait_seconds in enumerate((0, 5, 15, 30), start=1):
        if wait_seconds:
            time.sleep(wait_seconds)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--no-cache-dir", "--retries", "1", "--timeout", "30", "-q",
             "huggingface-hub>=0.28,<1.0"],
            text=True, capture_output=True,
        )
        last_output = (result.stdout + "\\n" + result.stderr)[-1200:]
        if result.returncode == 0:
            importlib.invalidate_caches()
            if hub_api_ready():
                break
        print("DEPENDENCY_RETRY", {"attempt": attempt, "wait_next": wait_seconds})
    else:
        raise RuntimeError(
            "DEPENDENCY_SAFE_STOP: huggingface_hub is unavailable. No branch was "
            "created or changed. Confirm Kaggle Internet is ON and restart Run All. "
            "pip tail: " + last_output
        )

import huggingface_hub
print("DEPENDENCY_READY", {"huggingface_hub": huggingface_hub.__version__})
"""),
        code("""# Read the credential only from the Kaggle secret/environment.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception as error:
        raise RuntimeError(
            "CREDENTIAL_SAFE_STOP: add a Kaggle secret named HF_TOKEN with write "
            "access, enable it for this notebook, then restart Run All."
        ) from error
if not HF_TOKEN:
    raise RuntimeError("CREDENTIAL_SAFE_STOP: HF_TOKEN is empty")
os.environ["HF_TOKEN"] = HF_TOKEN
print("CREDENTIAL_READY", {"secret_name": "HF_TOKEN", "value_printed": False})
"""),
        code(f"""# Exact live release lock audited on 2026-07-15. Never replace with branch heads.
SOURCE_RELEASE_LOCK = {lock_literal}

RELEASE_POINTER_LOCK = {{
    "experiment_id": "e2am-memrag-v3r1",
    "artifact_prefix": "experiments/e2am-memrag-v3r1/stages/09/coordinator",
    "stage_branch": "stage-e2am-memrag-v3r1-09-stage-09-coordinator",
    "stage_commit_sha": "4d3e111ddfe41247b511ad0fc1a413baacee7864",
    "release_manifest_sha256": "b1627b04695f502aec75d1424b57a42f0928446cd35c6d3302d69f8a58ab685f",
    "success_gate": "_SUCCESS.json",
    "success_gate_sha256": "f85ca74fa96d3eb327772a43b596637ed8810f1c04029e49d2623a1e1f7a12ef",
}}

CONFIG = {{
    "repo_id": "Shanmuk4622/E2AM-MemRAG-Traces",
    "experiment_id": "e2am-memrag-v3r1",
    "destination_branch": "consolidated-e2am-memrag-v3r1",
    "remote_root": "consolidated/e2am-memrag-v3r1",
    "work_root": str(WORK_ROOT),
    "hub_capacity": 96,
    "dirty_sync_target_seconds": 1200,
    "expected_artifact_records": 282,
    "expected_artifact_bytes": 127_554_473,
    "source_release_lock": SOURCE_RELEASE_LOCK,
    "release_pointer_lock": RELEASE_POINTER_LOCK,
}}
print("RELEASE_LOCK_READY", {{
    "branches": len(SOURCE_RELEASE_LOCK),
    "artifact_records": sum(x["artifact_records"] for x in SOURCE_RELEASE_LOCK),
    "artifact_bytes": sum(x["artifact_bytes"] for x in SOURCE_RELEASE_LOCK),
}})
"""),
        code(runtime_source),
        code("""# Fail before any Hub mutation if local capacity or the frozen lock is inconsistent.
import shutil

disk = shutil.disk_usage("/kaggle/working")
assert len(SOURCE_RELEASE_LOCK) == 23
assert len({item["branch"] for item in SOURCE_RELEASE_LOCK}) == 23
assert sum(item["artifact_records"] for item in SOURCE_RELEASE_LOCK) == 282
assert sum(item["artifact_bytes"] for item in SOURCE_RELEASE_LOCK) == 127_554_473
assert CONFIG["hub_capacity"] <= 96  # preserves at least 25% of 128/hour
if disk.free < 1_000_000_000:
    raise RuntimeError(
        "STORAGE_SAFE_STOP: less than 1 GB is free under /kaggle/working. "
        "Start a fresh CPU session; no remote branch was changed."
    )
print("CONSOLIDATION_PREFLIGHT_GO", {
    "free_gib": round(disk.free / (1024 ** 3), 2),
    "source_branches": 23,
    "expected_mib": round(127_554_473 / (1024 ** 2), 3),
    "gpu_required": False,
})
"""),
        code("""# Run or resume. A second writer, mixed release, checksum mismatch, or
# conflicting main file causes a safe stop instead of an overwrite.
FINAL_REPORT = run_consolidation(CONFIG, hf_token=HF_TOKEN)
"""),
        code("""# Final publication gate.
assert FINAL_REPORT["go"] is True
assert FINAL_REPORT["remote_verified"] is True
assert FINAL_REPORT["main_visible"] is True
assert FINAL_REPORT["source_branch_count"] == 23
assert FINAL_REPORT["source_artifact_records"] == 282
assert FINAL_REPORT["source_artifact_bytes"] == 127_554_473
assert FINAL_REPORT["source_branches_modified"] is False
assert FINAL_REPORT["completion_is_independent_of_hypothesis"] is True

print("FINAL_GO", {
    "consolidation_branch": FINAL_REPORT["consolidation_branch"],
    "consolidation_commit": FINAL_REPORT["consolidation_commit_sha"],
    "main_commit": FINAL_REPORT["main_commit_sha"],
    "paper_url": "https://huggingface.co/datasets/Shanmuk4622/E2AM-MemRAG-Traces/tree/main/experiments/e2am-memrag-v3r1/paper",
    "hypothesis_pass": FINAL_REPORT["hypothesis_pass"],
    "all_source_branches_preserved": True,
})
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "none", "internet": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "scripts" / "e2am_consolidation_runtime.py").read_text(encoding="utf-8")
    print(json.dumps(build_notebook(runtime), ensure_ascii=True, indent=1))


if __name__ == "__main__":
    main()
