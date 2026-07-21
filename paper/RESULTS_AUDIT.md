# E2AM-MemRAG v3r1 results and claim audit

## Release verdict

The experiment is **complete and remotely verified**, but the predeclared
confirmatory hypothesis **did not pass**. The formal non-inferiority and
operating-constraint checks passed only because the selected policy reproduced
the baseline trace exactly; they do not establish a useful quality-preserving
router. The required energy-reduction gate failed.

| Gate | Result | Interpretation |
| --- | --- | --- |
| Experiment completion | PASS | Frozen clean and robustness releases restored and verified |
| Fresh-root restore | PASS | All Stage-09 artifact checks passed |
| Formal quality non-inferiority | PASS | Non-informative identity with the baseline |
| Operating constraints | PASS | Test-time execution and accounting were complete |
| Energy reduction | FAIL | Policy-minus-baseline energy was exactly 0 J/query |
| Confirmatory hypothesis | FAIL | The conjunction of required gates was false |

## The central diagnostic: was the action pool routable?

Routing can improve a system only when candidate actions succeed on different
queries. Adapting the established single-best/virtual-best comparison from
algorithm selection, the paper reports the best fixed success \(B\), a post-hoc
per-query oracle \(O\), and routing headroom \(H=O-B\). This subtraction is not
claimed as a new construct; the contribution is its integration with interface,
capability, physical-cost, and validation-feasibility checks for composite RAG
actions. The oracle is descriptive, not the performance of a learned policy.

| Action pool | Actions | Best fixed | Per-query oracle | Headroom |
| --- | ---: | ---: | ---: | ---: |
| Frozen resident-eligible pool | 11 | 12.5% | 12.5% | **0.0 pp** |
| Offline reference pool | 6 | 44.2% | 72.5% | 28.3 pp |
| All retained clean routes | 17 | 44.2% | 73.3% | 29.2 pp |

All 15 successes in the resident-eligible pool came from `A00_tiny_direct`; no
other eligible action contributed a unique success. The pool oracle was also
far below the frozen 80% validation-success requirement. Therefore no router,
regardless of classifier quality, could have produced a success improvement from
that pool on these 120 queries. In contrast, the offline reference routes exhibit
substantial complementarity. The principal failure occurred before router
learning: the frozen action pool lacked both capability and routing headroom.

The 11-action eligibility rule required the 0.6B and 1B checkpoints to coexist on
one T4 with at least 15% free VRAM and no offload. The 3B/4B references were loaded
sequentially, while dynamic loading was outside the route-latency boundary. Zero
headroom is therefore conditional on a frozen resident-set constraint, not a claim
that the reference models are generally undeployable.

A00 also had the lowest route-level mean generation energy. However, a label-aware
per-query minimum that preserves all 15 successes selects A03 on one failure case
and saves 0.049 J/query (0.089%) in the recorded same-board traces. This negligible
but nonzero bound prevents an overstatement: the pool had zero success headroom,
not mathematically zero cost headroom.

This routability analysis is explicitly post hoc. It reuses the immutable traces
and does not alter the predeclared confirmatory outcome.

## Primary policy result

The calibrated policy selected `A03_tiny_hybrid` for all 120 clean test
queries. Strict support-qualified success was 0.0% (95% query-cluster interval
0.0% to 0.0%), and mean selected-GPU generation-window energy was 143.16 J/query
(95% interval 138.14 to 148.07). The stored policy and baseline records are the
same executions, so their paired differences are exactly 0.00 in success and
0.00 J/query by identity. The policy did not discover an energy-saving decision
rule.

This is not an execution failure. Coverage was 100%, the clean release records no
execution failures, and all 2,040 clean-test generation calls have selected-GPU
board energy sampled every 50 ms around `model.generate()`. CPU retrieval and
embedding, memory traversal, routing, parsing, verification, storage, network,
host and cooling energy, and carbon lie outside that energy boundary.

## Route, grounding, and generator findings

The best fixed route among all 17 retained endpoints was the offline,
router-ineligible reference `M18_granite_grounded_verified`, with 44.2% strict
success and 176.68 J/query. Grounding effects are not model invariant:

- Granite 3B grounding changed strict success by +23.33 pp (95% query-cluster
  interval +10.00 to +37.50 pp) and energy by +119.53 J/query.
- SmolLM3 3B grounding changed success by +10.83 pp (-2.50 to +24.17 pp) and
  energy by +41.52 J/query; the success interval includes zero.
- Qwen 4B grounding changed success by +29.17 pp (+16.67 to +41.67 pp) and
  energy by +171.71 J/query.
- Qwen 0.6B grounding changed success by -12.50 pp (-18.33 to -6.67 pp) and
  energy by +146.72 J/query.

The last contrast should not be read as evidence that retrieval corrupts answers
to evidence-dependent questions. The tiny direct route's 15 successes came from
the no-retrieval stratum; the composite-grounded endpoint lost those successes but
did not add success on the evidence-required strata under the frozen
interface. The mechanism is therefore an end-to-end generator--prompt--parser
compatibility failure, not a universal claim about small models or retrieval.

## Retrieval-to-utilization audit

The five grounded endpoints received identical ordered evidence lists for all
120 test queries. Required evidence was retrieved completely for 72 of the 90
evidence-dependent queries (80.0%), including 14 of 15 multi-hop queries. Yet
strict grounded success ranged from 0% for the tiny and Granite-1B endpoints to
58.9% for Granite 3B on that 90-query stratum. The experiment therefore separates
retrieval availability from evidence utilization: a shared retriever does not
imply a shared RAG outcome.

## Robustness interpretation

The selected policy achieved 0% strict success on the clean baseline and 0% in
all four corruption conditions. The zero robustness deltas are a **floor effect**,
not evidence that useful accuracy was preserved under corruption. Prompt-injection
compromise was 0%, but efficacy was also 0%; neither observation supports a
positive robustness claim.

## Trace audit

- Clean release: 2,040 rows, 17 routes, 120 unique queries, and 229 successes
  (11.2%).
- Robustness release: 1,440 rows, 3 routes, 120 unique queries, and 60 successes
  (4.2%).
- Duplicate unit IDs: 0 clean and 0 robustness.
- Non-finite audited metrics: 0 clean and 0 robustness.

## Defensible publication framing

The strongest paper is a failure-first study of **action-pool routability**. Its
main lesson is not that a particular router failed to learn, but that strict-success
selection was mathematically unproductive within the frozen resident-eligible pool even though the
broader reference pool contained substantial query-level complementarity. The
matched panel then explains where that complementarity comes from: grounding
utility depends strongly on the generator and task, while its observed
generation-window GPU-energy cost is consistently positive.

Claims remain bounded to this controlled synthetic benchmark, one visible T4 per
worker, four physical boards across clean lanes, the frozen model revisions, and
strict support-qualified success. Cross-board energy comparisons are descriptive.
Carbon, whole-system energy, public-benchmark state of the art, causal attribution
of all RAG overhead, and broad production generalization are outside the evidence.

## Provenance

- Hugging Face dataset: `Shanmuk4622/E2AM-MemRAG-Traces`
- Pinned visible release commit: `0b2405d9cca43fd04e35f792fdc4664405154fc6`
- Pinned paper branch commit: `00fa353f273f3a4b3d57a0b998301c85a1bc098b`
- Stage-09 artifact checks: 11/11 passed; 11,528,142 bytes verified
- Frozen execution spec SHA-256: `1c8f29fd250b87d3546c6ff3d128dc3fe6600bc798c3c0b0b8c98e0b95c76cfc`
