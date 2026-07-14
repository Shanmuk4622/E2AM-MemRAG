# Scientific contract

## Primary question and claim boundary

Can a cost-aware cascade reduce warm selected-GPU board energy per assigned query
relative to a validation-selected fixed deployable route while remaining
quality-non-inferior, sufficiently covered, and bounded in abstention?

Completion and hypothesis success are independent. Stage 09 first commits a
`RELEASE_CANDIDATE.json`, restores and verifies the complete candidate into a fresh
root, and only then may publish `_SUCCESS.json`. That file means the frozen work
plan and release checks completed. `HYPOTHESIS_RESULT.json` separately records
whether the predeclared energy/quality hypothesis passed.

E2AM-MemRAG v3 is a controlled synthetic experiment, not evidence of performance
on a public QA benchmark, production workload, or arbitrary domain. Static tests
cannot establish Kaggle T4 feasibility; each declared route must pass its Stage-03
hardware gate before its measurements are scientifically usable.

## Controlled benchmark and leakage controls

The full profile contains 800 deterministic scenario groups across:

- no retrieval;
- knowledge only;
- memory only;
- knowledge plus memory;
- temporal update;
- authority conflict;
- two-hop hybrid;
- deleted or missing evidence.

The fixed split is 10% pilot, 50% router train, 15% calibration, 10% validation,
and 15% sealed test. Scenario IDs and split-specific template families cannot
cross splits. Facts are sampled independently of task and split; a
train-majority/task-only baseline is audited before test labels are sealed. Model
and route pruning uses the pilot only. Router fitting, calibration, and threshold
selection use train, calibration, and validation respectively.

The generated data contains no personal information. Test labels become readable
by the pipeline only after the router artifact, thresholds, analysis freeze, and
their gate are present. The shared credential makes this a procedural seal rather
than a cryptographic access-control boundary, and the release reports it as such.

## Five-model portfolio

Stage 00 resolves and freezes exact immutable commits for five public Apache-2.0
repositories:

| Role | Model | Runtime status |
|---|---|---|
| tiny | `Qwen/Qwen3-0.6B` | deployable online candidate |
| small | `ibm-granite/granite-4.0-1b` | deployable online candidate |
| peer | `HuggingFaceTB/SmolLM3-3B` | sequential reference only |
| granite | `ibm-granite/granite-4.1-3b` | sequential reference only |
| upper | `Qwen/Qwen3-4B-Instruct-2507` | sequential reference only |

The tiny and small models must coexist on one T4 with at least 15% free VRAM and
no CPU/disk offload for primary online traces. Every reference model is loaded in
an isolated sequential block and unloaded before the deployable pair is restored.
Reference routes are never eligible actions for the trained router.

Stage 00 freezes model IDs, commits, configuration metadata, prompts, route
definitions, decoding parameters, seeds, the SHA-256 of embedded `src`/`configs`,
and the exact Python/package/Torch/CUDA/cuDNN/driver/GPU contract. It does not
download generator weights. A model lane downloads its required pinned snapshot
to an ephemeral, measured, resumable cache, verifies it, and loads with
local-files-only semantics. Public model caches are inputs and are not uploaded as
experiment artifacts.

Every later notebook verifies the Stage-00 experiment ID and both source and
environment contracts before counting or reusing a result. A mixed v2/v3 release,
changed model revision, or incompatible Kaggle image fails closed.

## Route matrix and controlled comparisons

The 16 original mechanism routes are migrated unchanged in logical purpose:

- `A00`-`A03`: tiny direct, BM25, dense, and bounded hybrid knowledge;
- `A04`-`A07`: small direct, BM25, dense, and bounded hybrid knowledge;
- `A08`-`A11`: flat, hierarchical, and temporal/entity memory comparisons;
- `A12`-`A15`: combined knowledge/memory, verification, upper reference, and
  evidence-guard routes.

Six routes complete the model-controlled panel:

- `M16_tiny_grounded_verified`;
- `M17_granite_direct` and `M18_granite_grounded_verified`;
- `M19_peer_direct` and `M20_peer_grounded_verified`;
- `M21_upper_direct`.

Together with `A00_tiny_direct`, `A04_small_direct`,
`A13_small_hybrid_verified`, and `A14_upper_hybrid_verified`, these form five
matched direct/grounded pairs. Each grounded member uses the same bounded hybrid
knowledge, graph/temporal memory, context budget, and deterministic evidence
verification contract; only the generator changes across the model panel.

The direct/grounded pair for every model and the original retrieval, memory, and
generator ablation axes are mandatory pilot anchors. Other routes may be pruned
only by the frozen pilot rule. Stage 09 cannot publish a comparison if either
required endpoint is absent from the exact clean matrix.

The citation guard checks parseability, retrieved-ID membership, required support,
citation precision/recall, and deleted-evidence exclusion. It is deterministic and
is not presented as an LLM entailment judge. A row is counted as success only when
the task answer and all scenario-applicable support/evidence requirements pass;
fluent unsupported output cannot qualify as success.

## Router

Stage 0 uses query-available features only and considers deployable direct actions.
If no action meets the frozen conservative success, energy, and latency rule, the
system pays for one BM25/memory-metadata probe. Stage 1 then considers deployable
non-direct actions. A validation-selected online escalation route is the
fail-closed fallback.

Success, log-energy, and log-latency models use five grouped query-bootstrap seeds.
The action gate uses the minimum calibrated success probability across seeds, while
energy and latency use 0.90-quantile regressors. Calibration uses only the calibration split. Threshold and safe-route
selection use validation only and enforce success, execution coverage, and
abstention constraints. Offline reference models may characterize transfer and the
Pareto frontier but can never be selected by this router.

Stage 07 assigns routes, not query partitions, to workers. Every route-affinity
lane evaluates all sealed-test queries for its retained routes. Only the
coordinator resolves policy choices against the aggregated exact query-by-route
matrix. This avoids making each worker download all five models and preserves
paired query coverage.

## Estimands, telemetry, and analysis

The primary systems quantity is warm selected-GPU board joules for generation per
assigned query. It is not whole-system energy, CPU energy, or carbon. Retrieval,
memory, verification, and routing are charged in end-to-end latency. An uncached
probe measurement plus separately timed Stage-0/Stage-1 router calls and the chosen
route are summed. Hub uploads occur only after the NVML measurement block ends.

Stage 09 requires exact sealed-test query-by-retained-route coverage and retains
failed, abstained, and non-finite rows. Missing primary energy telemetry blocks the
release analysis instead of disappearing from a mean. The primary policy-versus-
fixed comparison uses scenario-group clustered paired bootstrap intervals:

- quality-difference lower bound at least -0.03;
- GPU-joule-difference upper bound below zero;
- execution coverage at least 0.90;
- abstention at most 0.20.

Secondary analyses are explicitly labeled:

1. within-model direct-versus-grounded paired effects for strict success, GPU
   energy, latency, citation support, and abstention;
2. a five-model direct/grounded transfer table and non-dominated quality/energy/
   latency Pareto panel;
3. retrieval and memory factorial contrasts on the original controlled anchors;
4. clean-versus-corrupted paired degradation by scenario group for stale evidence,
   untrusted conflict injection, missing required evidence, and
   deletion/duplicate ordering.

These secondary intervals characterize the frozen controlled benchmark. They are
not silently promoted to confirmatory public-benchmark or architecture-superiority
claims.
