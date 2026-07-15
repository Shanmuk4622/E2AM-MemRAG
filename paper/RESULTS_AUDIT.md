# E2AM-MemRAG v3r1 results audit

## Release verdict

The experiment is **complete and remotely verified**, but the predeclared
confirmatory hypothesis **did not pass**. Quality non-inferiority and operating
constraints passed; the required energy-reduction gate failed. Completion and
hypothesis success are deliberately separate in the frozen protocol.

| Gate | Result |
| --- | --- |
| Experiment completion | PASS |
| Fresh-root restore | PASS |
| Quality non-inferiority | PASS |
| Operating constraints | PASS |
| Energy reduction | FAIL |
| Confirmatory hypothesis | FAIL |

## Primary policy result

The router selected `A03_tiny_hybrid` for all 120 clean test
queries. Strict support-qualified success was 0.0%
(95% cluster interval 0.0% to
0.0%); mean selected-GPU energy was
143.16 J/query (95% interval
138.14 to 148.07).
The paired policy-minus-baseline differences were exactly 0.00 for success and
0.00 J for energy. Therefore the router reproduced the baseline rather than
finding an energy-saving policy.

This is not an infrastructure failure: execution coverage was
100.0%, every route had complete energy telemetry,
and the clean release records zero execution failures.

## Route and generator findings

The best clean-test route was `M18_granite_grounded_verified` at
44.2% strict success and
176.68 J/query. The grounded Pareto frontier contains
the `granite` and `peer` model families.

- Granite 3B grounding improved strict success by
  23.33 pp [10.00, 37.50]
  while adding 119.53 J/query.
- SmolLM3 3B grounding improved success by
  10.83 pp [-2.50, 24.17];
  its interval includes zero, while energy increased by
  41.52 J/query.
- Qwen 4B grounding improved success by
  29.17 pp [16.67, 41.67]
  but added 171.71 J/query.
- Qwen 0.6B grounding **reduced** success by
  -12.50 pp [-18.33, -6.67]
  and added 146.72 J/query.

The controlled tiny-model retrieval and memory routes all changed success from
12.5% to 0% while adding approximately 88--117 J/query. The evidence therefore
shows that retrieval/memory augmentation was generator-dependent and could be
actively harmful under the frozen prompting, parsing, citation, and verification
contract.

## Robustness interpretation

The selected policy achieved 0% strict success on the clean baseline and 0% in
all four corruption conditions. Consequently, the reported zero robustness
deltas are a **floor effect**, not evidence that the system retained useful
accuracy under corruption. Prompt-injection compromise was 0%, but efficacy was
also 0%; the paper must not present this as a successful robustness result.

## Trace audit

- Clean release: 2040 rows, 17 routes,
  120 unique queries, 229 successes
  (11.2%).
- Robustness release: 1440 rows, 3 routes,
  120 unique queries, 60 successes
  (4.2%).
- Duplicate unit IDs: 0 clean and
  0 robustness.
- Non-finite audited metrics: 0 clean and
  0 robustness.

## Defensible paper framing

The strongest paper is a controlled negative-result and systems-diagnosis study:
**energy-aware routing fails when lightweight generator/grounding combinations do
not produce separable strict-success signal, while grounding benefits remain
strongly generator-dependent.** The contribution is the frozen benchmark,
end-to-end energy accounting, calibrated routing protocol, and failure analysis--
not a claim that the learned policy reduced energy.

Claims must remain bounded to this controlled synthetic benchmark, one T4,
selected-GPU board energy, the frozen model revisions, and strict
support-qualified success. Carbon, whole-system energy, public-benchmark SOTA,
and broad real-world generalization are outside the evidence.

## Provenance

- Hugging Face dataset: `Shanmuk4622/E2AM-MemRAG-Traces`
- Pinned visible release commit: `0b2405d9cca43fd04e35f792fdc4664405154fc6`
- Pinned paper branch commit: `00fa353f273f3a4b3d57a0b998301c85a1bc098b`
- Stage-09 artifact checks: 11/11 passed, 11,528,142 bytes verified
- Frozen execution spec SHA-256: `1c8f29fd250b87d3546c6ff3d128dc3fe6600bc798c3c0b0b8c98e0b95c76cfc`
