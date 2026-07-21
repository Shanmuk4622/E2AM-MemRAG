# E2AM-MemRAG publication blueprint

## Working title

**Is the RAG Action Pool Routable? A Failure-First Study of Grounding Utility
and GPU Energy**

The title states the actual scientific question and avoids implying that the
project proposes a successful new router. It also makes the methodological idea
legible beyond this implementation: audit the action space before optimizing a
policy over it.

## One-sentence thesis

On the frozen benchmark, the 11 resident-eligible actions had zero query-level routing
headroom even though six offline reference actions had 28.3 percentage points of
headroom; matched traces show that the missing complementarity is explained by a
strong generator--task interaction in grounding utility and by consistently
positive generation-window GPU-energy increments.

## Publication position

This is a **failure-first evaluation and systems-diagnosis paper**, not a new
router paper and not a state-of-the-art RAG benchmark claim. Its original object
of study is the *routability of an action pool*: whether the available actions
possess the capability, complementarity, interface compatibility, and comparable
cost measurements needed for a learned router to be scientifically meaningful.

That position distinguishes the work from papers that improve router
architectures, choose among retrievers, or learn RAG policies under the implicit
assumption that useful actions already exist. The empirical value is a complete
counterexample to that assumption plus a trace-level account of why the broader
reference pool behaves differently. The single-best/virtual-best gap itself is
established algorithm-selection practice and appears as an Oracle gap in
LLMRouterBench; novelty must be claimed for the integrated five-question RAG audit
and sealed diagnosis, not for \(O-B\).

## Research questions

1. **Routability:** Did the frozen resident-eligible action pool contain adequate
   capability and query-dependent headroom beyond its best fixed action?
2. **Retrieval versus utilization:** With retrieved evidence held constant, where
   did utility disappear between retrieval, generation, citation, and verification?
3. **Interaction and cost:** How did generator family and task class change the
   success and selected-GPU generation-energy effect of composite grounding?
4. **Protocol consequence:** Could the frozen learned controller satisfy its
   validation constraints, and did its test behavior agree with the pool audit?

## Contributions that the evidence supports

1. **A pre-routing audit.** The paper adapts established best-fixed and
   virtual-best/oracle success diagnostics to evidence-qualified composite RAG
   actions, then combines their headroom with interface, capability,
   cost-comparability, and constrained-selectability checks. The audit exposes a
   structural failure that classifier metrics alone would conceal.
2. **An exact action-pool diagnosis.** Across 120 frozen test queries, all 15
   resident-eligible successes came from one direct route: best-fixed and oracle success
   were both 12.5%. The reference pool reached a 72.5% oracle, demonstrating that
   useful complementarity existed outside the deployed choice set.
3. **A retrieval-to-utilization decomposition.** All five grounded endpoints saw
   the same ordered evidence, and retrieval was complete on 80.0% of
   evidence-required queries, yet strict utilization differed sharply by
   generator. This rules out retriever variation as the explanation for the
   matched-panel gap.
4. **A generator--task account of conditional grounding.** Direct generation is
   preferable on no-retrieval/deleted-context tasks, whereas grounding improves
   several 3B/4B endpoints on evidence-bearing tasks. Relative to always-grounded
   execution, the task-aware descriptive rule never reduces success, improves it
   in four of five matched families, and lowers observed mean generation energy
   in all five. The remaining Granite-1B pair is an interface-failure case with
   zero success under both actions.
5. **Bounded energy evidence and an honest negative result.** The work reports
   selected-GPU board energy only during generation, discloses cross-board limits,
   preserves the infeasible calibration outcome, and treats policy--baseline
   identity as non-informative rather than a quality success.

## Narrative architecture

### 1. Introduction: routing begins before the router

Open with the hidden assumption in conditional RAG: a learner can only exploit
differences already present among its actions. Define the central counterfactual
question—could *any* per-query selector have improved on the best fixed action?
Preview the 0.0 pp versus 28.3 pp headroom contrast and state the failed
confirmatory result without apology or suspense.

### 2. Related work and unresolved gap

Organize by the assumption each literature makes:

- adaptive RAG and retriever routing optimize a policy;
- model routing chooses among inference endpoints;
- memory systems enlarge the action space;
- efficient RAG studies cost or token budgets;
- RAG evaluation and robustness studies expose pipeline failures.

End with the gap: these lines rarely require a trace-level feasibility audit of
the candidate action pool before learning or judging a router.

### 3. Failure-first action-pool audit

Adapt the established \(B\), \(O\), and \(H=O-B\) comparison. Make clear that \(O\) is a descriptive
post-hoc upper bound and cannot be reported as an implementable policy. Present
the five audit questions: interface compatibility, capability, complementarity,
cost comparability, and constrained selectability.

### 4. Controlled testbed and measurement contract

Describe HybridBench as a diagnostic instrument rather than a public-benchmark
surrogate. Explain grouped splits, task classes, frozen evidence, exact model
revisions, prompt/interface contracts, strict support-qualified success, and the
selected-GPU generation-only energy boundary.

### 5. Experimental protocol

Report the 17-route clean matrix, 3-route robustness matrix, 120-query matched
test set, one visible T4 per worker, cluster bootstrap, frozen routing protocol,
and the disclosed pre-test amendment after all validation thresholds proved
infeasible. Separate confirmatory decisions from post-hoc diagnosis.

### 6. Results in causal order

1. Start with action-pool headroom: resident-eligible 0.0 pp, reference 28.3 pp.
2. Show that grounded endpoints received identical retrieval outputs.
3. Locate divergence in parsing, citation use, answer accuracy, and support.
4. Show generator--task interaction and matched grounding effects.
5. Report task-aware descriptive oracles and their energy differences.
6. Explain energy increments through longer input/output and generation time.
7. Return to validation infeasibility and the confirmatory policy identity.
8. Close with compatibility and robustness floor effects.

This ordering turns the router failure from an isolated disappointing result into
the predicted consequence of a measured action-space defect.

### 7. Discussion

Develop a readiness ladder: first validate action interfaces, then absolute
capability, then complementarity, then cost comparability, and only then fit a
constrained selector. Argue for conditional grounding at the task/model boundary,
not indiscriminate retrieval. Explain that bigger action pools are useful only
when they add non-dominated, uniquely successful actions.

### 8. Threats to validity, limitations, and ethics

Keep post-hoc analyses visibly labeled. Bound claims to the synthetic benchmark,
one hardware class, one released confirmatory seed, frozen prompts/parsers, and
selected-GPU generation-window energy. Treat cross-board energy differences as
descriptive. Do not claim carbon, whole-system energy, production robustness, or
public-benchmark state of the art.

### 9. Conclusion

End with the general result: a router cannot manufacture complementarity. The
first scientific question in adaptive RAG should be whether the action space is
routable at all.

## Main evidence map

| Manuscript claim | Trace-derived artifact |
| --- | --- |
| Resident/reference/full headroom | `data/derived/routability_pools.csv` |
| Success-preserving cost bound | `data/derived/success_preserving_cost_oracle.csv` |
| Unique route contributions | `data/derived/routability_route_contributions.csv` |
| Five audit gates | `data/derived/routability_gates.csv` |
| Identical retrieval and utilization stages | `data/derived/retrieval_to_utilization.csv` |
| Generator--task effects | `data/derived/task_grounding_effects.csv` |
| Matched grounding interactions | `data/derived/grounding_interactions.csv` |
| Task-aware descriptive oracles | `data/derived/task_aware_oracle.csv` |
| Energy mechanism | `data/derived/energy_mechanism.csv` |
| Parser/citation/answer failures | `data/derived/failure_decomposition.csv` |
| Confirmatory outcome | `data/derived/overall_results.csv` |
| Robustness floor effects | `data/derived/robustness_conditions.csv` |
| Trace integrity | `data/derived/trace_audit.csv` |

## Claim guardrails

Supported:

- The frozen resident-eligible action pool had zero observed success headroom.
- Eligibility is conditional on the frozen co-resident T4 contract.
- The success-preserving per-query cost bound saves 0.049 J/query, so zero
  success headroom must not be called zero cost headroom.
- The offline reference pool had substantial descriptive oracle headroom.
- Grounding utility varied by generator and task under a shared retrieval output.
- Grounding increased measured generation-window selected-GPU energy in all five
  matched pairs; four pairs are same-board comparisons.
- The constrained validation problem had no feasible threshold, and the frozen
  confirmatory policy did not reduce energy.
- The task-aware rules and per-query oracles are post-hoc descriptive analyses.

Unsupported:

- The proposed router reduced energy or preserved useful quality.
- The per-query or task-aware oracle is an implemented deployable policy.
- Retrieval is universally harmful to small language models.
- Zero corruption delta demonstrates robustness.
- Selected-GPU generation-window energy equals end-to-end or whole-system energy.
- The benchmark establishes real-world or public-benchmark superiority.

## Submission strategy

The most credible scope is an empirical NLP/ML systems or evaluation venue that
welcomes rigorous negative results, adaptive RAG analysis, or reproducibility.
The manuscript should be judged on diagnostic novelty and evidence discipline,
not on a new routing architecture. A short cover note should foreground the
pre-routing audit and the pre-test preservation of a failed confirmatory result.
