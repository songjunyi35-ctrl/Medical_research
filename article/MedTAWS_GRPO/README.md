# MedTAWS-GRPO Paper Package

This directory contains the first complete, evidence-bounded manuscript for the MedTAWS and GRPO-style research line.

## Files

- `manuscript.md`: canonical 4,000+ word English manuscript.
- `MedTAWS_GRPO_manuscript.pdf`: rendered review copy.
- `main.tex`: generated LaTeX manuscript.
- `references.bib`: BibTeX database using primary papers.
- `figures/framework_overview.*`: two-level method diagram.
- `figures/development_results.*`: matched comparison and development trend.
- `EVIDENCE_MATRIX.md`: claim-to-artifact boundary.
- `CONFIRMATORY_EXPERIMENTS.md`: frozen experiment checklist required before submission.
- `build_figures.py`, `export_latex.py`, `build_pdf.py`: reproducible build scripts.

## Build

```bash
python build_figures.py
python export_latex.py
python build_pdf.py
```

The PDF builder uses the local `.vendor` ReportLab dependency. The LaTeX source is generated from `manuscript.md`; a standard TeX installation can compile it with BibTeX.

## Scientific Status

The method and manuscript structure are complete. The numerical evidence is not submission-final. The strongest historical result, RC Dice 0.58932, comes from a single-modal, oracle-click, ground-truth-centered development protocol with longitudinal subject overlap. Its matched advantage over supervised training is +0.01649 across two seeds. No current subject-clean T1ce residual or combined MedTAWS-GRPO holdout result exists.

For that reason, this version does not claim SOTA and does not place automatic four-modal BraTS results in the same table. `CONFIRMATORY_EXPERIMENTS.md` defines the work required to replace the retrospective development evidence with a defensible final result.
