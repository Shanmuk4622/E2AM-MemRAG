# E2AM-MemRAG agent instructions

These rules apply to every implementation agent working in this repository.

## Required notebook contract

- Deliver training workflows as compact `.ipynb` notebooks runnable on Kaggle.
- Measure storage at runtime. Treat the Kaggle output limit and larger ephemeral
  storage as separate filesystems; never assume advertised capacity is available.
- Primary scientific runs expose one T4 (`CUDA_VISIBLE_DEVICES`) before importing
  Torch, Transformers, NVML clients, or other CUDA-aware packages. Dual-T4/DDP is a
  separately labeled throughput experiment only.
- Prefer Kaggle-native/attached datasets for large downloads. Pin dataset versions,
  source checksums, model/tokenizer revisions, and environment fingerprints.
- Every notebook includes a numbered execution runbook, prerequisites, expected
  artifacts, failure messages, restart instructions, and a final go/no-go result.

## Durability and Hugging Face

- Hugging Face namespace: `Shanmuk4622`. Read the credential only from the Kaggle
  Secret/environment variable `HF_TOKEN`; never print, log, serialize, or commit it.
- Use immutable content-addressed artifacts and a stable worker lane. A changing
  session/attempt ID must not create a new recovery branch.
- Locally seal replayable mini-shards/checkpoints before upload. Save model,
  optimizer, scheduler, AMP scaler, RNG, sampler/data cursor, metrics, source,
  environment, and work-plan state needed for exact resume.
- Target a dirty-state sync every 1,200 seconds, after major stages, on normal
  completion, and at a safe `KeyboardInterrupt` boundary. Skip clean heartbeats.
- Use a conservative shared ceiling of 128 Hub requests/hour with at least 25%
  emergency headroom, bundled commits, staggered legitimate workers, 429
  `Retry-After`, exponential backoff, and no repeated 401/403 attempts.
- Publish a worker pointer only in the same commit as a closure seal whose every
  dependency already exists. Pin and download-verify the receipt, seal, and pointer.
- Safe Stop reports success only after remote checksum verification. A hard VM kill
  cannot run cleanup; bound possible replay with small sealed work units.
- Never delete the sole local copy until a newer remote closure passes a fresh-root
  restore test. Train once; resume or replay only the smallest unsealed unit.

## Parallelism and ownership

- Parallel correctness uses frozen logical partitions, unique worker IDs, worker-
  scoped paths/revisions, and optimistic parent commits. Never let two live writers
  share a worker lane.
- Execution must follow platform terms. Do not design around one person controlling
  alternate Kaggle accounts. Genuine collaborators use their own accounts and
  individually scoped credentials.
- Only a coordinator may merge verified worker closures or publish a global release.

## Scientific integrity

- Record selected-GPU board energy as GPU energy, not whole-system energy or carbon.
- Pause uploads during measured blocks; include all retrieval, memory, reranking,
  generation, verification, and maintenance costs charged to a route.
- Reject mixed experiment spec hashes, missing work-plan coverage, divergent duplicate
  unit IDs, non-finite metrics, secret-like values, and corrupt/unsealed checkpoints.
- Never silently change batch size, precision, context, route, or device placement
  after OOM/failure. Record the failed declared route and create a new spec for changes.
- Keep test labels sealed; enforce grouped/temporal leakage controls and report
  failures, abstentions, constraint violations, and execution coverage.

Read `docs/SCIENTIFIC_CONTRACT.md` and `docs/FAILURE_RECOVERY.md` before adding a
training notebook.
