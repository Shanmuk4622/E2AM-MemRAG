# Building the E2AM-MemRAG paper

The manuscript is modular and uses the standard `IEEEtran` class with BibTeX.
All reported numbers are generated from the checksum-verified frozen release.

## Rebuild derived assets

From the repository root:

```powershell
python scripts/build_manuscript_assets.py
python scripts/render_manuscript_figures.py
```

The command fails if the frozen clean or robustness trace counts change. It
regenerates the manuscript tables, vector figures, action-pool routability audit,
retrieval-to-utilization decomposition, task-aware oracle analysis,
route-to-board audit, robustness summaries, and Granite-1B output-format audit.
The second command renders the three principal diagnostic figures to publication-
resolution PNG files without changing their trace-derived values.

## Compile with Tectonic

From `paper/manuscript`:

```powershell
tectonic -X compile main.tex --outdir build --keep-logs --keep-intermediates
```

Tectonic resolves `IEEEtran`, runs BibTeX, and reruns the document until references
stabilize. For Overleaf, upload the complete `paper/manuscript` directory together
with the PNG figures and choose pdfLaTeX.

## Evidence and claim checks

```powershell
python scripts/validate_manuscript.py
python -m unittest tests.test_paper_results tests.test_manuscript -q
```

The checks reject unresolved citations/references, numeric claims that disagree
with the frozen CSV/JSON evidence, incorrect route-pool headroom, forbidden
whole-system/carbon claims, missing figures/tables, and absent disclosure of the
Stage-06 protocol amendment or the post-hoc status of the routability diagnosis.

## Package the submission artifacts

After a successful compile and validation, run from the repository root:

```powershell
python scripts/package_manuscript.py
```

This creates `output/pdf/E2AM_MemRAG_Paper.pdf` and a deterministic
`output/latex/E2AM_MemRAG_Overleaf.zip` with `main.tex` at the archive root.

After all sources and validation reports are final, refresh the paper-level
checksum inventory and verify it without rewriting any manuscript file:

```powershell
python scripts/update_paper_manifest.py
python -m unittest tests.test_paper_results -q
```

## Measurement boundary

The manuscript's energy quantity is selected-GPU board energy integrated only over
`model.generate()`. Retrieval runs on CPU; parsing, deterministic verification,
and scoring occur outside the energy and route-latency windows. Route latency is
retrieval time plus the generator-call duration. Policy latency additionally sums
the frozen probe/router components. Neither whole-system energy nor carbon is
measured.
