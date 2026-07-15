# E2AM-MemRAG paper blueprint

## Recommended title

**E2AM-MemRAG: Generator-Dependent Grounding and Failure Modes of
Energy-Aware Routing for Small Language Models**

Alternative title: **When RAG Costs More and Helps Less: A Controlled
Quality-Energy Study of Retrieval and Memory for Small LLMs**

## Central paper claim

In a frozen, single-T4 synthetic benchmark with strict support-qualified
success, grounding is not uniformly beneficial: it materially improves selected
3B/4B generators, harms the 0.6B generator, and always increases selected-GPU
energy. A calibrated energy-aware router can therefore fail by collapsing to an
uninformative baseline when its lightweight candidate routes provide insufficient
strict-success separation.

This is a negative-result and mechanism paper. The learned router did not reduce
energy and the confirmatory hypothesis failed.

## Research questions

1. How does retrieval and memory augmentation change strict grounded success,
   latency, and GPU energy for a fixed small generator?
2. Is the effect of grounding stable across five frozen generator families?
3. Can a query/probe-based router preserve strict quality while reducing GPU
   energy relative to the frozen baseline?
4. How does the selected policy behave under stale, missing, conflicting, and
   duplicated evidence?

## Defensible contributions

1. A deterministic 800-scenario synthetic benchmark with eight task classes,
   grouped splits, frozen evidence, and strict citation/support scoring.
2. End-to-end selected-GPU board-energy accounting across retrieval, memory,
   generation, verification, probing, and maintenance charged to each route.
3. A matched five-model direct-versus-grounded panel on the same 120 test
   questions, prompt budget, decoding configuration, and route contract.
4. A calibrated Pareto-routing protocol with explicit abstention, execution
   coverage, failure reporting, immutable provenance, and fresh-root restoration.
5. An honest failure analysis showing generator-dependent grounding effects,
   tiny-model degradation, a router-selection collapse, and robustness floor
   effects.

## Proposed manuscript structure

### 1. Introduction

- Motivate the hidden energy cost of adding retrieval, memory, reranking, and
  verification to small LLMs.
- State that RAG quality gains are often assumed rather than tested jointly with
  energy and support-qualified correctness.
- Present the generator-dependent result and failed router hypothesis up front.

### 2. Related work

- Energy-aware inference and model routing.
- Retrieval-augmented generation and verification.
- Long-term memory architectures.
- Conditional computation, abstention, and Pareto optimization.
- Negative results and reproducibility in controlled LLM evaluation.

### 3. E2AM-MemRAG methodology

- Benchmark construction, task classes, and leakage controls.
- The 17 clean-test routes represented in the final release.
- Five-model direct-grounded panel.
- Strict support-qualified success definition.
- Selected-GPU board-energy measurement and scope.
- Router features, calibration, thresholding, abstention, and frozen gates.
- Immutable worker closures and restoration verification.

### 4. Experimental protocol

- One T4 per scientific run.
- Frozen model revisions, seed 4622, 120 clean test queries per route.
- Clustered bootstrap with 10,000 replicates over query clusters.
- Four corruption conditions and matched clean-route comparisons.
- Predeclared quality, energy, and operating-constraint gates.

### 5. Results

#### 5.1 Route-level quality-energy frontier

Use `figures/route_quality_energy.svg` and `tables/route_results.md`.

#### 5.2 Generator-dependent grounding

Use `figures/model_grounding_effect.svg` and `tables/model_transfer.md`.
Emphasize the positive Granite 3B and Qwen 4B effects, the uncertain SmolLM3
effect, and the significant Qwen 0.6B degradation.

#### 5.3 Router outcome and confirmatory hypothesis

Report the collapse to `A03_tiny_hybrid`, 0% strict success, 143.16 J/query,
and zero paired energy difference. Explicitly report the failed energy and
confirmatory gates.

#### 5.4 Robustness and abstention

Use `tables/robustness.md`. Interpret the zero deltas as a floor effect because
clean selected-policy success was already zero.

### 6. Failure analysis

- Strict JSON/citation parsing and support requirements.
- Weak success signal among router-eligible tiny/small routes.
- Generator capacity as an interaction variable for grounding.
- Energy overhead of retrieval, longer contexts, and verification.
- Why calibration quality alone does not imply a useful routing policy.

### 7. Limitations

- Controlled synthetic benchmark rather than a public QA benchmark.
- One hardware class and selected-GPU energy rather than whole-system energy.
- One seed in the released confirmatory evaluation.
- Frozen prompt and parser contracts.
- No carbon claim and no broad real-world generalization claim.

### 8. Conclusion

Conclude that energy-aware RAG routing requires route candidates that first
provide reliable, generator-specific quality separation. Adding retrieval and
memory is not automatically a quality improvement, particularly for sub-billion
generators under strict grounded evaluation.

## Main paper assets

| Paper item | Source |
| --- | --- |
| Primary outcome | `data/derived/overall_results.csv` |
| Hypothesis gates | `data/derived/hypothesis_gates.csv` |
| Route comparison | `data/derived/route_statistics.csv` |
| Mechanism contrasts | `data/derived/controlled_contrasts.csv` |
| Model transfer | `data/derived/model_transfer.csv` |
| Scenario analysis | `data/derived/scenario_class_statistics.csv` |
| Robustness | `data/derived/robustness_conditions.csv` |
| Trace integrity | `data/derived/trace_audit.csv` |
| Figure 1 | `figures/route_quality_energy.svg` |
| Figure 2 | `figures/model_grounding_effect.svg` |

## Claim guardrails

Allowed:

- The experiment and fresh-root restoration completed successfully.
- The predeclared hypothesis failed because energy reduction failed.
- Grounding effects differed materially by generator.
- All released executions and GPU-energy samples had complete coverage.
- The findings hold for the frozen controlled benchmark and hardware scope.

Not allowed:

- The router reduced energy.
- Zero robustness delta demonstrates robustness.
- GPU board energy equals whole-system energy or carbon emissions.
- The results establish public-benchmark SOTA or broad production behavior.
- All RAG or memory systems harm small models in general.
