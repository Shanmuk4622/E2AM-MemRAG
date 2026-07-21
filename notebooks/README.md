# Kaggle run order for E2AM-MemRAG v3

These 23 notebooks are standalone. Do not attach a source-code dataset and do not
edit a lane, branch, repository, or worker ID. A dual-T4 Kaggle session is
acceptable, but the first code cell exposes only GPU 0 before importing Torch or
any CUDA-aware package.

## One-time settings

For every fresh Kaggle session:

1. turn **Internet on**;
2. choose a **GPU** accelerator;
3. add or enable the Kaggle Secret `HF_TOKEN` with write permission to
   `Shanmuk4622/E2AM-MemRAG-Traces`;
4. open the correct numbered/lane notebook and choose **Run All**.

Each collaborator must use an authorized, appropriately scoped credential on
their own account. Never paste a token into a cell, output, notebook, or message.
Never run the same lane notebook in two live sessions.

The first cells print the physical and output-disk measurements, selected GPU,
experiment identity, source hash, prerequisites, and expected artifacts. A
successful run ends with both `REMOTE_CLOSURE_VERIFIED` and `STAGE_COMPLETE`.

## Exact execution sequence

| Step | Notebook files | When to run |
|---:|---|---|
| 1 | `00_setup_and_freeze_data.ipynb` | coordinator once |
| 2 | `01_build_indexes.ipynb` | after step 1 |
| 3 | `02_build_and_freeze_hybridbench.ipynb` | after step 2 |
| 4 | `03_pilot_routes_lane_00.ipynb` ... `lane_03.ipynb` | four different collaborators in parallel |
| 5 | `04_aggregate_and_prune_pilot.ipynb` | after all four step-4 gates pass |
| 6 | `05_collect_training_traces_lane_00.ipynb` ... `lane_03.ipynb` | four different collaborators in parallel |
| 7 | `06_train_and_calibrate_router.ipynb` | after all four step-6 gates pass |
| 8 | `07_evaluate_frozen_clean_lane_00.ipynb` ... `lane_03.ipynb` | four different collaborators in parallel |
| 9 | `08_run_robustness_lane_00.ipynb` ... `lane_03.ipynb` | only after all four step-8 gates pass |
| 10 | `09_aggregate_audit_and_release.ipynb` | coordinator last |
| 11 | `10_consolidate_verified_hf_release.ipynb` | after Stage 09; CPU-only paper-facing publication, with no experiment rerun |

The filename already contains its immutable ownership. Send different lane files
to collaborators; no one needs to configure a lane manually.

## What downloads where

Stage 00 freezes the environment, source, data, prompts, model repository IDs, and
exact model revisions. It performs CUDA/NVML smoke checks but downloads **no
generator weights**. A weight progress bar in Stage 00 therefore indicates that an
old v2 notebook is being used; replace the complete notebook set with one coherent
v3 build.

Stages 03, 05, and 07 use fixed route/model affinity:

| Internal route lane | Principal routes/models |
|---|---|
| lane-a | tiny routes and the matched Granite 4.1 reference pair |
| lane-b | online-small routes |
| lane-c | memory ablations and the matched SmolLM3 reference pair |
| lane-d | combined-evidence routes and the Qwen 4B reference pair |

The notebook maps `lane-00` ... `lane-03` to these internal lanes. The mapping and
route list are printed before execution and are part of the frozen work plan.
Reference models are loaded sequentially; the online pair is restored whenever a
deployable route is needed.

Stage 07 no longer divides test queries and makes every worker load every model.
Instead, each lane evaluates its assigned retained routes on **all** sealed-test
queries. The coordinator joins the disjoint route columns and resolves the frozen
router decisions only after full matrix coverage is proved. Only routes backed by
the resident tiny/small pair are router-eligible.

Model snapshots use exact revisions in a measured ephemeral cache. A model load
prints:

```text
MODEL_DOWNLOAD_START ...
MODEL_DOWNLOAD_HEARTBEAT ...
MODEL_SNAPSHOT_READY ...
MODEL_LOAD_READY ...
```

The download is resumable at the Hub cache layer. After snapshot verification the
model loader uses the local snapshot only. The cache is never added to result
closures or uploaded to Hugging Face; only manifests, traces, checkpoints, gates,
and reports are durable experiment artifacts.

## Stop, restart, and recovery

- For a manual stop, interrupt the running cell once and wait for
  `SAFE_STOP_VERIFIED` before terminating the session.
- If Kaggle kills the VM, reopen the **same notebook file** and choose **Run All**.
  It restores the last verified closure and skips completed unit IDs.
- Pending result shards are sealed by size or elapsed time and dirty state is
  pushed at most every 1,200 seconds. Major cells, router seeds, normal completion,
  and a catchable interrupt force an immediate closure.
- LLMs are evaluated rather than fine-tuned. There are no LLM epochs to resume;
  exact inference units and router-seed checkpoints are the resumable work.
- Stage 01 resumes 512-record vector shards. Stage 06 saves every grouped-bootstrap
  seed separately. A hard kill can replay only the current unsealed unit/shard.

Do not delete a local result merely because an upload began. The notebook treats
the state as durable only after it pins the new remote commit and verifies the
pointer, manifest, objects, and closure seal.

## Troubleshooting by the printed failure

| Message or symptom | Meaning and action |
|---|---|
| `Prerequisite gate is absent` | An earlier coordinator/lane file has not completed. Run the missing prerequisite; do not bypass the gate. |
| `EXPERIMENT_ID_MISMATCH` or source/spec mismatch | An older source and v3r1 artifacts or notebooks were mixed. Use the complete v3r1 notebook set and prefix. |
| `MODEL_DOWNLOAD_START` with continuing heartbeats | A required public pinned snapshot is downloading normally; leave the cell running. |
| unchanged byte count/percentage plus heartbeats for five minutes | The process is alive but the transfer is stalled. Interrupt once, wait for `SAFE_STOP_VERIFIED`, and rerun the same current notebook; bounded HTTP resumes partial blobs. |
| `MODEL_DOWNLOAD_SIGNED_URL_REFRESH` | Hugging Face returned a transient expired/rotated public blob signature. Completed blobs are preserved and the retry is automatic; do not change `HF_TOKEN`. |
| no heartbeat followed by a bounded download failure | Restart the same notebook. The cache resumes existing immutable blobs; the declared route is not silently changed. |
| checksum/local-snapshot failure | The cache is incomplete or corrupt. The loader quarantines/re-fetches that pinned snapshot; it never substitutes another revision. |
| OOM or insufficient free VRAM | The declared route is recorded as failed. Do not lower batch, precision, context, or move layers to CPU inside the frozen experiment. |
| 429 | The client honors `Retry-After` and exponential backoff. Do not launch an additional writer for that lane. |
| public-model 401/403 without the signed-URL marker | The public repository or pinned revision is inaccessible. Do not repeatedly retry or change the declared route. |
| artifact-upload 401/403 | The writer credential is invalid or lacks repository access. Fix the Kaggle Secret before rerunning; repeated writes are suppressed. |
| second-writer/parent mismatch | Two processes attempted the same lane. Stop both, retain the verified closure, then restart exactly one owner. |

If a route cannot pass its Stage-03 Kaggle gate, record the failure. Any changed
precision, prompt budget, model, or device placement is a new experiment spec, not
an in-place rescue.

## Hugging Face layout

```text
experiments/e2am-memrag-v3r1/stages/<00..09>/<coordinator|lane-XX>/
  artifacts/sha256/<prefix>/<content hash>
  manifests/<manifest hash>.json
  LATEST.json
```

Each stage/lane uses its own fixed branch. Test labels occupy a separate
procedurally sealed branch and are opened only after the policy is frozen. Stage 09
publishes and fresh-restores `RELEASE_CANDIDATE.json` before it can write
`_SUCCESS.json`, then writes the authoritative
`experiments/e2am-memrag-v3r1/RELEASE.json` pointer on `main`.
