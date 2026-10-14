# WAW-Glioma reproducibility code

This folder contains release-level scripts for the WAW-Glioma dataset. It does
not redistribute MRI data, patient-specific paths, model weights, or third-party
software. All inputs must be de-identified before use.

## Pipeline scope

The archive includes frozen source snapshots of the public radiomics pipeline
under `glioma-radiomics-pipeline/` and the SynthStrip batch wrapper under
`batch-skull-stripping/`; see `PIPELINE_SNAPSHOT.md` for commit identifiers
and included code.

1. De-identify and convert source MRI to NIfTI.
2. Run SynthStrip brain extraction using the included public batch wrapper,
   then manually inspect/correct the resulting masks.
3. Run the FL-PoST / RANO FeTS 2.0 tool with its documented default inference
   settings to initialize tumor segmentations. The official FL-PoST
   documentation snapshot is included under `FL-PoST/`. The August 2025
   initialization provenance is recorded in `fets_run_manifest.csv`.
4. Manually correct the initial masks in ITK-SNAP and obtain final radiologist
   review. This human annotation step is not reproducible by a script and must
   be documented as such.
5. Run the release-quality scripts in this folder to create QC manifests,
   validate component linkage, and quantify scanner-field-strength association.
6. Run the inter-observer reproducibility script, using the manually verified
   mask-pairing manifest and the released final masks. The output includes
   Dice, HD95, geometric-QC results and de-identified per-case metrics.
7. Run the public radiomics extraction pipeline at
   `https://github.com/CCIG-Champalimaud/glioma-radiomics-pipeline` with the
   complete parameter file used for this release. Archive the exact commit or
   Zenodo release version with the dataset.

## Included scripts

- `00_make_qc_manifests.py`: creates blank, auditable manual QC manifests from
  the imaging metadata.
- `01_validate_release.py`: verifies identifiers, component counts, released
  masks, and radiomics linkage fields.
- `03_scanner_association_analysis.py`: computes the released exploratory 3T
  versus 1.5T radiomics association analysis using rank eta-squared.
- `04_interobserver_reproducibility.py`: calculates label-specific Dice and
  HD95 for the 15 PRE and 15 POST independent-reader pairs after geometry and
  affine QC. It does not resample masks.

## Required additions before public release

- The exact Git commit/release of the public radiomics pipeline and the YAML/JSON
  parameter files used for this dataset (the parameter files are public; record
  the final commit or release identifier before publication).
- The completed registration and segmentation QC manifests, or a statement of
  the number reviewed, passed, excluded, and the exclusion reasons.
- An intra-rater reproducibility analysis script and a de-identified results
  table, if this complementary analysis is completed before resubmission.

## Environment

Create an environment using `requirements.txt`. The scripts assume Python 3.10+
and use semicolon-delimited clinical/radiomics CSV files. The third-party FeTS,
FL-PoST, SynthStrip, ITK-SNAP, and CNVkit software must be installed separately
and cited according to their own licenses.

## Reproducibility note

The scanner analysis is a cross-sectional association analysis, not a test-retest
or ICC reproducibility analysis. It does not establish that a feature is stable
across scanners.
