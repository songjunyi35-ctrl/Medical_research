# Frozen pipeline snapshot

This archive contains source snapshots of the two public repositories used for
release preprocessing and radiomics extraction.

## Radiomics extraction

- Snapshot commit: `6de3a7906eb82bb107898c287d2145d9db1473f3`
- Snapshot date: 2026-08-24
- Included source: `pipeline.sh`, `src/`,
  `radiomics/radiomicsConfig.yaml`, `pyproject.toml`, `uv.lock`, README and
  the repository license.

The snapshot is included so that the exact radiomics extraction code and YAML
configuration are retained with this Zenodo release. It contains no MRI data or
patient-specific paths. The public GitHub repository remains the location for
newer versions.

## SynthStrip batch preprocessing

The directory `batch-skull-stripping/` contains a source snapshot of
`CCIG-Champalimaud/batch-skull-stripping`.

- Snapshot commit: `8759bf81b5f1113f861ae80a7d780db3788f04a0`
- Included source: `skullstrip.py`, `skullstrip_batch.py`, `pyproject.toml`,
  `uv.lock` and README.

This is the public batch wrapper around SynthStrip. Model weights are not
redistributed in this archive; obtain them as described by that repository.

## FL-PoST / FeTS 2.0 provenance

The directory `FL-PoST/` contains a documentation snapshot of the official
`FETS-AI/FL-PoST` repository.

- Snapshot commit: `083e69026248b3666b3b060e6fc26651be11bf11`
- Commit date: 2025-07-22
- Release label in this package: `FL-PoST (FeTS 2.0), August 2025`

The automated segmentations were initialized in August 2025, then manually
corrected and finally reviewed by a radiologist. The precise model/container
identifier and invocation command were not retained; this limitation is stated
in `fets_run_manifest.csv` rather than reconstructed retrospectively.
