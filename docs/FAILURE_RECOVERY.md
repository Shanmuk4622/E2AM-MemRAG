# Failure and recovery contract

## Durable experiment boundaries

- A result row has a SHA-256 identity bound to experiment/spec hashes, complete
  route configuration, query, condition, and seed.
- Replayable result shards are content-addressed and contain at most 128 rows. A
  time boundary seals pending rows when 1,200 seconds elapse.
- Stage 01 atomically stages checksum-verified 512-record vector shards and forces
  an `index-freeze` closure after the completed index cell.
- Stage 06 writes one independent router-seed checkpoint and verified closure per
  seed.
- A logical artifact is uploaded under its content hash. Its manifest, closure
  seal, and `LATEST.json` worker pointer enter one optimistic-parent commit only
  after all dependencies exist.

Normal dirty sync is limited to one 20-minute interval. Clean heartbeats make no
Hub commit. Major stage cells, router seeds, normal completion, and a catchable
`KeyboardInterrupt` force a closure. Safe Stop reports success only after a fresh
remote checksum verification.

Public model snapshots are intentionally outside this closure. They can be
re-downloaded from their frozen public commits; private traces, gates, router
checkpoints, manifests, and work-plan state cannot, so those are the items uploaded
to `Shanmuk4622/E2AM-MemRAG-Traces`.

## Fresh-session restore

A fresh session resolves one lane branch head, downloads the pointer, manifest,
seal, and content objects from that pinned commit, verifies every byte count and
SHA-256, restores into an empty temporary root, then atomically replaces the
materialized output. Orphan files from an older kernel cannot survive the restore.

Existing shards are rehashed; filename digests, sequences, spec hashes, and unit
coverage are checked; and valid shards are restaged idempotently to close a
write-before-manifest crash window. Divergent duplicate unit IDs, mixed specs,
unexpected units, non-finite required metrics, and secret-like content stop the
stage.

Before any downstream notebook counts restored work, it verifies
`e2am-memrag-v3r1`, the embedded source-tree SHA-256, and the deterministic
Python/package/Torch/CUDA/cuDNN/driver/physical-GPU contract against Stage 00. A v2
closure cannot satisfy a v3r1 gate. If this fails, use one coherent v3r1 notebook build
in a compatible Kaggle image or declare a new experiment ID; never bypass the
check.

## Pinned model snapshot recovery

Stage 00 performs model metadata resolution only. Each execution lane creates a
resumable Hub snapshot for only the models assigned to its routes:

1. print `MODEL_DOWNLOAD_START` with repository, immutable commit, expected local
   cache path, and current disk headroom;
2. disable the `hf-xet` worker on the Kaggle primary path because its worker can
   remain alive without advancing bytes, and use explicit HTTP timeouts instead;
3. reuse verified content-addressed blobs already in the cache;
4. print bounded `MODEL_DOWNLOAD_HEARTBEAT` messages while the snapshot call is
   active;
5. classify only a blob-host 403 containing an expired or rotated signature as
   transient, print `MODEL_DOWNLOAD_SIGNED_URL_REFRESH`, and request a fresh URL;
6. validate that the pinned snapshot is complete and print
   `MODEL_SNAPSHOT_READY`;
7. instantiate tokenizer/model from that local path with network access disabled
   for the load and print `MODEL_LOAD_READY`.

429, transient 5xx, connection failures, and the narrowly identified public blob
signature failure honor bounded exponential backoff. Every other 401/403 remains
non-retryable. A stopped or killed session may lose the ephemeral cache, but if Kaggle
preserves it the immutable blobs resume rather than restart. A missing or corrupt
snapshot is never accepted and no newer model revision is substituted. The
notebook quarantines/refetches the affected pinned snapshot or exits with a precise
failure; it does not silently change dtype, device mapping, batch, context, or
decoding.

If a model runs out of VRAM or violates the no-offload/free-VRAM contract, the
declared route is recorded as failed. A proposed lower-precision or shorter-context
run is a different execution spec and must not overwrite v3 traces. Stage-03 gates,
not local static tests, are authoritative for T4 compatibility.

## Hub failures and concurrent ownership

- 429 and 5xx responses use bounded exponential backoff and server
  `Retry-After`.
- A public signed-blob signature 403 obtains a fresh URL without using the private
  writer token. Every other public-model or artifact-upload 401/403 remains
  non-retryable so a bad secret cannot exhaust the shared request budget.
- A lost commit response is considered recovered only when the new branch head
  contains the exact in-flight pointer, manifest, and closure seal.
- A parent mismatch that is not the in-flight closure is a second-writer conflict
  and stops the notebook.
- Each fixed lane has exactly one live owner. Different lane notebooks use disjoint
  branches and frozen route partitions; two sessions never write one lane.
- Only coordinator stages aggregate lane exports and publish global pointers.

Shared request volume is bounded structurally. Stages 02, 04, and 06 publish
deterministic internally checksummed foundation/training/evaluation bootstrap ZIPs.
Each parallel lane downloads one bootstrap object and publishes one verified lane
export rather than reconstructing many upstream closures. Public anonymous model
downloads do not use or serialize the private artifact token. Repository and
branch checks are cached per process.

## Stage-specific recovery

| Stage | Smallest durable/replayable unit |
|---|---|
| 00 | frozen metadata/source/data bundle and PASS gate |
| 01 | 512-record vector shard; final index-freeze closure |
| 02 | generated scenario group and sealed benchmark bundle |
| 03, 05 | route/query result unit, sealed into bounded shards |
| 06 | one grouped-bootstrap router seed |
| 07 | route/query clean-test unit within its route-affinity lane |
| 08 | condition/query/route robustness unit |
| 09 | verified release candidate before the final release pointer |

Stage 07 evaluates all test queries for the routes assigned to a lane. A lane
restart therefore replays only missing route/query units. The Stage-09 coordinator
will not resolve router decisions or compute paired analyses until the union of
lane exports proves the exact frozen query-by-route closure.

## Limits and operator behavior

Python cannot execute cleanup after a hard VM kill. At most the dirty work since
the last 20-minute closure, the current unsealed shard, or the current router seed
must be replayed. A catchable `KeyboardInterrupt` requests immediate sealing and
prints whether remote proof succeeded; do not close the session before
`SAFE_STOP_VERIFIED`.

A completed stage is skipped only when its PASS gate hash is in the artifact
inventory of the freshly verified closure and its experiment/source/environment
identities match. A stale local PASS file, an old v2 branch, or an unrelated commit
can never produce `REMOTE_CLOSURE_VERIFIED`.

The following are errors, not permission to adapt silently:

- incomplete prerequisites or route coverage;
- stale/mixed experiment IDs or notebook source hashes;
- missing/corrupt pinned model snapshot;
- insufficient storage or VRAM;
- divergent duplicate unit IDs;
- missing energy telemetry in a primary row;
- authentication latch or second-writer conflict.

Fix the declared external condition and rerun the same notebook. If the scientific
configuration must change, create a new spec/experiment ID and preserve the v3
evidence, including the failed route record.
