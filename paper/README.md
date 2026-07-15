# E2AM-MemRAG paper evidence package

This directory is the paper-facing, checksum-verified snapshot of the completed
`e2am-memrag-v3r1` experiment. The raw artifacts are pinned to Hugging Face main
commit `0b2405d9cca43fd04e35f792fdc4664405154fc6` and paper branch commit
`00fa353f273f3a4b3d57a0b998301c85a1bc098b`.

Start with [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md). It states the confirmatory
outcome, the defensible findings, the floor effects, and the claim boundaries.

## Contents

- `data/raw/`: the 11 immutable Stage-09 artifacts plus release pointers.
- `data/derived/`: flat CSV tables and a compact machine-readable result summary.
- `tables/`: manuscript-ready Markdown tables.
- `figures/`: editable SVG and rendered PNG figures.
- `RESULTS_MANIFEST.json`: SHA-256 and byte size for every packaged file.

Regenerate and verify everything with:

```powershell
python scripts/collect_paper_results.py
```

The completed experiment did **not** pass the predeclared confirmatory
hypothesis. The package deliberately preserves that result. It must not be
described as an energy-saving router success.
