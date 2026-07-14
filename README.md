# E2AM-MemRAG

[![CI](https://github.com/Shanmuk4622/E2AM-MemRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Shanmuk4622/E2AM-MemRAG/actions/workflows/ci.yml)

E2AM-MemRAG is a reproducible benchmark and energy-aware Pareto router for
small-LLM retrieval-augmented generation and memory systems. Version 3 studies
joint generator, retrieval, memory, verification, abstention, and routing choices
on a single T4. It retains a straightforward execution model: 22 standalone
Kaggle notebooks, four fixed parallel lanes, and one private Hugging Face artifact
repository.

Every notebook embeds and checksum-verifies the same runtime. No Git clone,
Kaggle source dataset, branch name, or worker ID needs to be entered. To run it:

1. enable Internet and a GPU accelerator in Kaggle;
2. add `HF_TOKEN` as a Kaggle Secret with write access to the user's repository;
3. choose **Run All** in the next numbered notebook.

The experiment ID is `e2am-memrag-v3r1`. Durable experiment artifacts are stored in
the private dataset `Shanmuk4622/E2AM-MemRAG-Traces`. Dirty state is bundled and
pushed no more often than once every 20 minutes, and also at major boundaries,
normal completion, and a catchable interruption.

## Model panel and route design

The v3 panel uses five public Apache-2.0 model repositories:

- online tiny: `Qwen/Qwen3-0.6B`;
- online small: `ibm-granite/granite-4.0-1b`;
- sequential references: `HuggingFaceTB/SmolLM3-3B`,
  `ibm-granite/granite-4.1-3b`, and `Qwen/Qwen3-4B-Instruct-2507`.

Tiny and small form the deployable single-T4 pair. Reference models are evaluated
one at a time and cannot be selected by the router. The 16 original mechanism
routes are retained, and six routes complete a matched direct-versus-grounded
comparison for every model, producing 22 routes in total. This separates two
questions that are often confounded: whether grounding helps a fixed generator,
and which generator lies on the quality/latency/GPU-energy frontier.

Stage 00 now resolves exact revisions and freezes metadata only. It does **not**
download generator weights. The first lane that needs a model creates a pinned,
resumable local snapshot and prints `MODEL_DOWNLOAD_START`, progress heartbeats,
and `MODEL_SNAPSHOT_READY` before loading only from that verified snapshot.
Public model caches are ephemeral inputs and are not uploaded to the private
experiment repository.

V3r1 disables the `hf-xet` worker in Kaggle because it can remain alive while an
individual blob stops advancing. The bounded HTTP path retains partial blobs and
uses explicit request timeouts. If Hugging Face returns a transient public blob
URL with a rotated or expired signature, the runtime prints
`MODEL_DOWNLOAD_SIGNED_URL_REFRESH`, waits with bounded backoff, and requests a
fresh URL. It does not misreport that CDN failure as a bad private `HF_TOKEN`.

> In v2, a line such as `model.safetensors: 0%` during setup was the beginning of
> a real model-weight download, not by itself an exception. The deprecation message
> about `torch_dtype` was also a warning. V3 removes model downloads from setup,
> uses the current `dtype` argument, and makes later downloads explicit.

## What is measured

- 800 deterministic controlled scenarios in eight task classes and five grouped
  splits;
- 22 generator/retrieval/memory/verification routes;
- a resident 0.6B + 1B deployable pair and three isolated 3B/4B references;
- selected-GPU board joules, warm latency, strict support-qualified success,
  citation behavior, abstention, failures, and execution coverage;
- a query-only router stage followed, only when required, by a charged retrieval
  and memory probe;
- sealed clean-test evaluation and four frozen evidence-corruption conditions;
- within-query direct/grounded comparisons, scenario-clustered uncertainty, and a
  model-level Pareto transfer panel.

This is a **controlled synthetic v3 experiment**. It can support a reproducible
research prototype and a strong portfolio artifact; it does not establish public
benchmark performance or broad real-world generalization. Static local tests check
the pipeline contract, but successful Stage-03 gates on Kaggle are still required
before claiming that every model route is T4-compatible.

## Repository map

```text
configs/                     shared runtime defaults
docs/                        research, scientific, and recovery contracts
notebooks/                   22 standalone Kaggle notebooks in run order
scripts/build_experiment_notebooks.py
src/e2am_memrag/             embedded experiment runtime
tests/                       CPU-safe regression suite
```

Start with [the exact notebook run order](notebooks/README.md). The study design is
specified in [the v3 research plan](docs/RESEARCH_PLAN_V3.md), with claim boundaries
in [the scientific contract](docs/SCIENTIFIC_CONTRACT.md) and restart behavior in
[the failure/recovery contract](docs/FAILURE_RECOVERY.md).

Local verification:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p 'test_*.py'
python scripts/build_experiment_notebooks.py --check
```
