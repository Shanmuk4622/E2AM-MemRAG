# E2AM-MemRAG v3 research plan

## 1. Study in one sentence

E2AM-MemRAG v3 asks whether a frozen, calibrated policy can route among compact
generator/RAG/memory/verification actions to save selected-GPU board energy without
losing strict support-qualified success, while a matched five-model panel shows
whether observed grounding gains transfer across model families and scales.

This is implemented in the existing 22-notebook order. The scientific scope grows;
the operator workflow does not.

## 2. Why this is more than a model comparison

A flat table of small LLMs with and without RAG mixes several causes: model size,
retrieval method, memory organization, evidence verification, and the decision to
retrieve at all. V3 separates those causes through three linked experiments:

1. **mechanism ablation** — fixed generators under direct, BM25, dense, hybrid,
   flat memory, hierarchical memory, graph/temporal memory, and verification;
2. **matched model transfer** — five generators each evaluated direct and under
   one identical grounded/verified route;
3. **decision policy** — an online-only cascade trained from the controlled traces
   and tested against a fixed validation-selected route.

The resulting contribution is a reproducible benchmark-and-router artifact, not a
chatbot demo. Its novelty claim is deliberately about joint energy-aware routing
and controlled transfer within this benchmark, not about inventing a new
retriever, reranker, or language model.

## 3. Predeclared research questions

### Primary question

Relative to the validation-selected fixed deployable route, does the frozen router
reduce warm selected-GPU board joules per assigned sealed-test query while meeting
the non-inferiority, coverage, and abstention constraints in the scientific
contract?

### Secondary questions

- For each generator, what is the paired effect of the identical grounded route
  versus direct generation on strict success, support, energy, latency, and
  abstention?
- Do direct-to-grounded effects transfer across model families, or are they driven
  by one model/size?
- Which direct and grounded model points are non-dominated in
  success/energy/latency space?
- Which retrieval and memory mechanisms help each scenario class, and which merely
  add cost?
- How do the policy and fixed anchors degrade under stale, conflicting, missing,
  deleted, or duplicate evidence?

Secondary results are reported with uncertainty and failure counts but are not
silently converted into broad public-benchmark claims.

## 4. Frozen five-model panel

All repositories are public and publish Apache-2.0 terms. Stage 00 records the
exact repository commit and configuration metadata before any trace is collected.

| Key | Public repository | Experimental role | Router eligible |
|---|---|---|---:|
| `tiny` | `Qwen/Qwen3-0.6B` | lowest-cost online generator | yes |
| `small` | `ibm-granite/granite-4.0-1b` | stronger online escalation generator | yes |
| `peer` | `HuggingFaceTB/SmolLM3-3B` | independent fully-open 3B reference | no |
| `granite` | `ibm-granite/granite-4.1-3b` | within-family size/generation reference | no |
| `upper` | `Qwen/Qwen3-4B-Instruct-2507` | larger Qwen upper reference | no |

The first two models are selected as a practical single-T4 resident pair. The
three references are loaded one at a time so the study gains a contemporary
multi-family panel without adding quantization, distributed inference, or custom
kernel branches to the primary experiment. Gated models and models that require a
different multimodal/specialized inference stack are excluded from v3 because
access and runtime differences would become experimental confounds.

Qwen3.5 is intentionally not a primary-panel model. The family is natively
multimodal, while its text-only path still uses a hybrid linear/full-attention
architecture with optional fast kernels and a slower, more memory-hungry fallback.
That runtime difference is irrelevant to this text-only RAG question and would
confound T4 energy with software-kernel compatibility. It belongs in a separately
identified compatibility study, not as a silent substitution inside v3.

The panel is not a model leaderboard. The direct/grounded pairs use frozen decoding
and context contracts, and conclusions are limited to the generated tasks and the
declared T4 runtime.

## 5. Twenty-two-route matrix

### Original mechanism routes

| IDs | Generator | Controlled axis |
|---|---|---|
| `A00`-`A03` | tiny | direct, BM25, dense, bounded two-hop hybrid |
| `A04`-`A07` | small | direct, BM25, dense, bounded two-hop hybrid |
| `A08`-`A10` | tiny | flat, hierarchical, graph/temporal memory |
| `A11` | small | graph/temporal memory |
| `A12` | small | hybrid knowledge plus flat memory |
| `A13` | small | hybrid knowledge plus graph memory and verifier |
| `A14` | upper | sequential grounded/verified upper reference |
| `A15` | small | evidence-guarded BM25 plus hierarchical memory |

### Matched model routes

| Model | Direct endpoint | Grounded/verified endpoint |
|---|---|---|
| tiny | `A00_tiny_direct` | `M16_tiny_grounded_verified` |
| small | `A04_small_direct` | `A13_small_hybrid_verified` |
| Granite 4.1 | `M17_granite_direct` | `M18_granite_grounded_verified` |
| SmolLM3 | `M19_peer_direct` | `M20_peer_grounded_verified` |
| Qwen 4B | `M21_upper_direct` | `A14_upper_hybrid_verified` |

Every grounded endpoint uses the same logical route: bounded hybrid knowledge,
graph/temporal memory, deterministic evidence verification, and frozen context and
generation budgets. The model key is the intended difference. All ten endpoints
are mandatory pilot anchors; their failure is reported rather than hidden by route
pruning.

The 22 routes are assigned to model-affinity lanes. A lane can reuse a loaded
model across its routes and does not fetch models owned by another lane. This is
both operationally cheaper and scientifically cleaner than partitioning queries
while making every worker maintain the full model portfolio.

## 6. Benchmark and split design

The generator creates 800 deterministic scenario groups across eight task classes:
no retrieval, knowledge only, memory only, knowledge plus memory, temporal update,
authority conflict, two-hop hybrid, and deleted/missing evidence.

The frozen grouped split is:

| Split | Share | Permitted use |
|---|---:|---|
| pilot | 10% | feasibility, telemetry, pilot-only route pruning |
| train | 50% | router outcome/cost models |
| calibration | 15% | probability calibration only |
| validation | 10% | constraints, thresholds, fixed safe baseline |
| sealed test | 15% | one final clean evaluation and robustness analyses |

Scenario IDs and split-specific templates do not cross partitions. Temporal/entity
histories remain inside their group. Labels, required evidence IDs, and answer
fields for sealed test are unavailable to the pipeline until the policy and
analysis plan are frozen. A task-only train-majority baseline is audited to detect
obvious construction leakage.

## 7. Strict outcome definitions

The primary outcome is **support-qualified success**, not string fluency. A
scenario succeeds only if its answer check passes and every applicable evidence
condition passes: output/citation parseability, retrieved-ID membership, required
support, citation precision/recall, authority/timestamp behavior, and deleted-
evidence exclusion. The deterministic guard is not described as semantic
entailment.

Every unit also records:

- status/failure class and abstention;
- selected-GPU board joules and warm generation latency;
- charged retrieval, memory, verification, and router/probe latency components;
- retrieval/citation IDs and support diagnostics;
- route, model revision, prompt/decoding/spec hashes, scenario group, seed, and
  execution coverage.

Missing telemetry and failed units remain visible. Primary energy rows without
valid GPU telemetry cannot enter a mean and block release-level inference.

## 8. Notebook-stage implementation

| Logical stage | Notebook count | Research/output responsibility |
|---|---:|---|
| 00 setup/freeze | 1 | generate/freeze inputs, exact model metadata, source/environment/spec gates; no weights |
| 01 indexes | 1 | deterministic BM25, dense, and memory structures with resumable vector shards |
| 02 HybridBench | 1 | freeze grouped splits, prompts, labels, work plans, and procedural test seal |
| 03 pilot | 4 | route-affinity T4 feasibility, residency, failure, latency, energy, and quality traces |
| 04 prune/freeze | 1 | pilot-only pruning while retaining all mandatory comparison anchors |
| 05 router traces | 4 | disjoint route-affinity train/calibration/validation matrix |
| 06 router | 1 | grouped-bootstrap fit, calibration, validation thresholds, frozen policy/gate |
| 07 clean test | 4 | every lane runs all test queries for its assigned retained routes |
| 08 robustness | 4 | one frozen corruption condition per lane against the policy and online anchors |
| 09 audit/release | 1 | exact coverage proof, clustered paired analyses, model Pareto panel, fresh restore |

Stages 03, 05, and 07 may run as four genuine collaborator-owned lanes in
parallel. Coordinator stages run only after every required lane gate is remotely
verified. The notebook filenames and execution order do not change from v2, so the
workflow remains easy to follow.

## 9. Model acquisition and one-T4 rules

Generator download is deferred to the first lane that actually needs the model.
The model repository and immutable Stage-00 commit are passed to a resumable
snapshot download in a measured ephemeral cache. The notebook emits explicit start,
heartbeat, ready, and load markers. Once complete, tokenizer/model construction is
local-only; a partial/corrupt snapshot cannot be mistaken for a usable model.

The primary run exposes physical GPU 0 before CUDA-aware imports. The online pair
must be resident without CPU/disk offload and leave the frozen free-VRAM margin.
Reference models run sequentially and are unloaded before restoring the online
pair. An OOM is an observed failure of the declared route. Precision, context,
batching, attention implementation, and device placement are never changed
silently after a failure.

Disk is measured at runtime. Model caches are excluded from Hugging Face closure
uploads, while small replayable result/checkpoint units are retained until a newer
remote closure passes checksum verification. No notebook assumes an advertised
Kaggle capacity is currently free.

## 10. Router and evaluation protocol

The deployable action space contains only tiny/small routes. Stage 0 scores direct
actions from query-available features. If no direct action clears the frozen
conservative constraints, one charged BM25/memory-metadata probe supplies Stage-1
features. Stage 1 chooses an online grounded action or falls back to the
validation-selected online safe route.

Router models are fit across five grouped bootstrap seeds. The controller gates on
the minimum calibrated success probability across seeds and uses 0.90-quantile
energy/latency regressors. Calibration is isolated to the calibration split; the threshold, fallback, and all constraints are selected
on validation. Sealed-test labels are opened only after this policy artifact and
gate are immutable.

For Stage 07, route-affinity workers evaluate all sealed-test queries for their
routes. The coordinator then joins the exact query-by-route matrix, attaches the
precomputed frozen policy choices, and performs paired analysis. This preserves a
common query set across all five matched model comparisons without asking each
worker to download every model.

## 11. Analysis plan

### Confirmatory comparison

Policy and fixed validation-selected safe route are paired by test scenario group.
A scenario-clustered bootstrap estimates quality and GPU-joule differences. The
primary result passes only if all four frozen rules hold:

- quality lower confidence bound at least -0.03;
- GPU-joule difference upper confidence bound below 0;
- execution coverage at least 0.90;
- abstention no more than 0.20.

### Model transfer panel

For each of the five models, Stage 09 computes paired grounded-minus-direct effects
on strict success, board energy, latency, support, and abstention. It then reports
the ten endpoints in a common table and marks non-dominated success/energy/latency
points. Reference models inform this transfer/Pareto analysis only; they are not
hypothetical router actions.

### Mechanism and scenario analyses

The frozen A-route anchors estimate retrieval and memory contrasts on matched
queries. Results are stratified by the eight scenario classes and retain the
scenario group as the resampling unit. Stage 08 pairs each corrupted-condition
outcome with the same clean query/route where available and reports success loss,
support loss, abstention, failures, and—for conflict injection—the explicit
compromise indicator. Four conditions remain fixed: stale evidence, untrusted
conflict injection, missing required evidence, and deletion/duplicate ordering.

All analyses state their execution denominator. Unexpected missing pairs, mixed
specs, divergent duplicates, or incomplete route coverage stop the release rather
than being dropped.

## 12. Durability and collaboration

Every logical worker has an immutable lane and fixed Hugging Face branch. Progress
is sealed into content-addressed mini-shards/checkpoints and dirty state is pushed
at most once per 1,200 seconds, plus major cells, completion, and a catchable
interrupt. A pointer is published only with a closure seal whose dependencies
already exist, and a pinned fresh download verifies the remote proof.

A hard VM kill cannot run cleanup, so exact recovery is bounded by small unsealed
units. Rerunning the same notebook restores the latest verified closure and skips
complete unit IDs. Only the coordinator can merge verified lanes and publish the
global release. See `FAILURE_RECOVERY.md` for error-specific behavior.

## 13. Go/no-go interpretation

There are three distinct outcomes:

1. **pipeline no-go** — prerequisites, model hardware gates, exact coverage,
   telemetry, integrity, or fresh restore fails;
2. **completed hypothesis fail** — the entire controlled experiment is valid but
   the primary policy constraints are not all met;
3. **completed hypothesis pass** — the pipeline is valid and all primary
   constraints pass.

Outcomes 1 and 2 must remain visible. A valid negative result is more valuable
than silently changing the spec until a favorable number appears.

## 14. Explicit limitations and next study

V3 uses generated tasks, deterministic evidence checks, one selected GPU board,
one runtime class, and one prompt/decoding family per route. It does not measure
whole-system/CPU energy or carbon, user preference, open-domain factuality,
cross-hardware efficiency, or production traffic. It does not implement a learned
reranker or claim that the verifier proves semantic entailment.

A later, separately identified study can add public QA and long-context memory
strata, repeated hardware sessions, and stronger semantic evaluation. Those data
must not be backfilled into `e2am-memrag-v3r1` after its policy or analysis is frozen.
