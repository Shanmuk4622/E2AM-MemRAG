# Reviewer report and revision record

## Overall assessment

The original manuscript contained an unusually complete experimental record but
presented its most important evidence as a failed routing experiment followed by
a collection of route comparisons. That framing made the work appear incremental:
recent literature already contains learned RAG routers, retriever selectors,
model routers, memory controllers, and cost-aware policies. A stronger paper
cannot claim novelty merely from combining a small LLM, RAG, memory, and routing.

The revised manuscript makes a different and better-supported contribution. It
asks a prior question that routing studies often leave implicit: **does the
candidate action pool contain enough capability and query-level complementarity
to be routable at all?** The answer is measured before interpreting the learned
selector. In the frozen resident-eligible pool, the best fixed action and a post-hoc
per-query oracle both succeed on 12.5% of queries, so routing headroom is exactly
0.0 percentage points. Six offline reference actions, evaluated on the same
queries, have 28.3 points of headroom. This contrast turns the failed router from
an isolated implementation outcome into a general action-space diagnosis.

On the present evidence, the revised work is credible as an empirical evaluation,
negative-result, or NLP systems paper. It should not be marketed as a new routing
algorithm or as a state-of-the-art RAG system. The best-fixed/virtual-best gap is
established algorithm-selection practice, and LLMRouterBench already analyzes
model complementarity and an Oracle gap. The revision therefore does not claim
\(O-B\) as a new statistic. Its defensible novelty is the five-stage RAG-specific
protocol that combines that diagnostic with interface compatibility, absolute
capability, physical-energy comparability, and constraint feasibility, plus the
sealed negative case.

## Major weaknesses found in the earlier structure

### 1. The novelty claim began too late in the pipeline

The paper focused on the learned router, although a classifier cannot create
successes absent from its choices. This left the reader unable to distinguish a
learning failure from an action-space failure.

**Revision.** The manuscript now adapts best-fixed success \(B\), descriptive
per-query oracle success \(O\), and routing headroom \(H=O-B\). It evaluates five
preconditions—interface compatibility, capability, complementarity, cost
comparability, and constrained selectability—before discussing policy learning.
It also states the eligibility criterion: the 0.6B and 1B checkpoints had to
coexist on one T4 with 15% free VRAM and no offload. The sequential 3B/4B
references were outside that resident set, and model-loading time was outside the
route-latency boundary. Zero headroom is conditional on this frozen contract.

### 2. The result order obscured the scientific mechanism

Route averages, the failed policy, and robustness tables appeared before the
reader knew whether any useful routing decision existed.

**Revision.** Results now follow an explanatory sequence: action-pool headroom;
retrieval identity; evidence utilization; generator--task interaction; task-aware
descriptive oracles; energy mechanism; calibration infeasibility; confirmatory
identity; robustness floor effects.

### 3. “RAG helps or hurts” was too broad

The matched effects differed by generator, and the tiny endpoint's direct
successes were confined to the no-retrieval task. A global claim
would confuse retrieval quality, evidence use, output formatting, and strict
verification.

**Revision.** The paper now treats outcome as an interaction among task,
retrieved evidence, generator, prompt/interface, parser, and verifier. Identical
retrieval lists across five grounded endpoints make the utilization gap directly
visible. The text no longer generalizes the tiny-model failure to small models or
RAG as a class.

### 4. Formal gate language could mislead readers

The formal quality non-inferiority gate passed, but the selected policy and
baseline are the same stored executions. Calling that a quality-preservation
success would be technically true under the gate definition and scientifically
misleading.

**Revision.** The result is described as identity and therefore non-informative.
The failed energy gate and failed conjunction remain prominent. The disclosed
pre-test amendment is retained and separated from post-hoc diagnosis.

### 5. The energy boundary needed to govern every claim

The traces contain selected-GPU board energy integrated during generation, not
whole-system energy and not the direct energy cost of CPU retrieval, parsing,
verification, or routing. Comparisons also span four physical T4 boards.

**Revision.** Every main energy claim now names the generation-window boundary.
Same-board matched contrasts are distinguished from cross-board descriptions, and
carbon or end-to-end energy claims are expressly excluded. The mechanism analysis
links energy increments to longer generation time without claiming full-system
causality.

### 6. Related work did not establish a precise gap

A generic list of RAG, memory, and routing papers made the project look like a
combination of known components.

**Revision.** Related work is organized by what is being adapted—retrieval,
generator, memory, or computation—and by the assumption that candidate actions
are already useful. The scope table states whether representative methods audit
interface compatibility, absolute capability, query-level complementarity, and
cost comparability before learning.

## Claims strengthened by reanalysis of existing traces

No new model inference or benchmark run was requested. The revision derives
additional analyses only from frozen clean and robustness traces:

- exact headroom for the resident-eligible, reference, matched-pair, and complete pools;
- unique-success contributions of every route;
- identical ordered retrieval outputs for all grounded endpoints;
- retrieval completeness on evidence-required and multi-hop tasks;
- parse, citation, answer, support, and strict-success failure stages;
- generator-by-task grounding effects and interaction contrasts;
- task-aware direct/grounded descriptive oracles with query-cluster intervals;
- strict-score threshold sensitivity;
- input/output length, generation time, and GPU-energy increments;
- route dominance, calibration feasibility, and robustness floor checks.
- a success-preserving per-query cost bound: one same-board substitution saves
  0.049 J/query, preventing the false inference that zero success headroom proves
  zero cost headroom.

These analyses explain the existing experiment; they do not retroactively change
its confirmatory hypothesis.

## Remaining limitations a reviewer may raise

1. The action-pool audit is post hoc. It is useful diagnosis, not a preregistered
   confirmatory test.
2. HybridBench is synthetic and deliberately diagnostic. External validity to
   open-domain or production workloads is untested.
3. Only one released confirmatory seed and one GPU class are represented. Four
   physical boards confound cross-lane energy comparisons with session/board
   variation.
4. The strict metric embeds a parser and support contract. Threshold sensitivity
   helps, but alternative generation interfaces may change absolute results.
5. The task-aware and per-query oracles use known outcomes or task labels. They are
   upper-bound/descriptive analyses, not deployable selectors.
6. The Granite 1B endpoint had a frozen output-format/runtime incompatibility;
   its zero score is not evidence of general model incapacity.
7. The reference pool demonstrates complementarity, not a deployable routing
   result; its predictability was not prospectively tested.

These limitations should remain explicit. None requires rerunning the completed
experiment to make the current paper internally sound.

## Publication recommendation

Before revision, the likely decision was weak reject: the empirical work was
substantial, but the novelty appeared to be an unsuccessful composition of known
components. After revision, the paper has a coherent contribution and a clear
negative-result value proposition. A reasonable target is a strong Findings,
evaluation, efficient NLP, RAG, or reproducibility venue. Main-track acceptance
will depend on reviewer interest in diagnostic methodology rather than algorithmic
novelty.

The submission pitch should be concise: routing should be preceded by an
action-pool feasibility audit; otherwise classifier optimization can hide the
fact that no productive routing decision exists.

## Writing and citation integrity

The manuscript prose was rewritten from the project's own trace evidence and
method contract. It does not copy source abstracts or imitate the wording of a
specific paper. Citations are attached to identifiable prior methods or empirical
claims, while the paper's own post-hoc analyses are labeled as such. Bibliographic
records are checked against primary proceedings or official publication pages.
This is not a substitute for a publisher's similarity checker, but the revision
was composed independently and avoids unsupported novelty language.
