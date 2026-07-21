# E2AM-MemRAG paper evidence package

This directory is the paper-facing, checksum-verified snapshot of the completed
`e2am-memrag-v3r1` experiment. The raw artifacts are pinned to Hugging Face main
commit `0b2405d9cca43fd04e35f792fdc4664405154fc6` and paper branch commit
`00fa353f273f3a4b3d57a0b998301c85a1bc098b`.

Start with [`RESULTS_AUDIT.md`](RESULTS_AUDIT.md) for the evidence audit and
[`REVIEWER_REPORT.md`](REVIEWER_REPORT.md) for the reviewer-led restructuring.
The manuscript's central question is whether the frozen RAG action pool was
routable before any selector was trained.

## Contents

- `data/raw/`: the 11 immutable Stage-09 artifacts plus release pointers.
- `data/derived/`: trace-derived routability, interaction, energy, robustness,
  and integrity analyses.
- `tables/`: manuscript-ready Markdown tables.
- `figures/`: editable SVG and rendered PNG figures.
- `manuscript/`: modular IEEE-style LaTeX source, generated tables, and figures.
- `PAPER_BLUEPRINT.md`: contribution, narrative, evidence, and claim map.
- `RESULTS_MANIFEST.json`: SHA-256 and byte size for every packaged file.

Regenerate and verify everything with:

```powershell
python scripts/collect_paper_results.py
python scripts/build_manuscript_assets.py
python scripts/render_manuscript_figures.py
python scripts/validate_manuscript.py
python -m unittest tests.test_manuscript -q
# Compile and package as described in manuscript/BUILD.md, then finalize checksums:
python scripts/update_paper_manifest.py
python -m unittest tests.test_paper_results -q
```

Run the manifest finalizer last whenever a paper source, table, figure, or
validation report changes. It inventories the finished package without
regenerating or overwriting the paper narrative.

The completed experiment did **not** pass the predeclared confirmatory
hypothesis. The package deliberately preserves that result. The deployable pool
also had zero observed routing headroom; the publication contribution is the
failure-first action-pool audit and its mechanism analysis, not an energy-saving
router claim.
