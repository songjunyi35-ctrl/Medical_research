# Invalid legacy-nearest T1ce artifacts

Date: 2026-07-22
Reason: categorical source/model/source resampling used PyTorch legacy `nearest`, which
is not center-aligned with trilinear `align_corners=False`. These artifacts are retained
for engineering provenance only. Their metrics and checkpoints must not be used for a
baseline, paper comparison, seed aggregate, MedTAWS search, or resume.

Invalid completed or partially completed directories:

- `smoke_seed20260722_2train_2val_e1_v2`
- `pilot64_noaug_seed20260722_e10_v2`
- `pilot64_aug_seed20260722_e10_v3`
- `pilot256_rank0_seed20260722_e20`
- `pilot256_lorar4_seed20260722_e20`
- `full872_lorar4_aug_lr4e4_seed20260722_e50_v2` (interrupted before first validation)
- `full872_lorar4_noaug_lr1e4_seed20260722_e50_v2` (interrupted before first validation)

The two empty first-attempt full-run directories without `_v2` also failed before any
training step because incorrect manifest filenames were supplied. They contain no
scientific result.

Valid replacement protocol:
`workspace/BraTS/training_protocols/dpdsam_t1ce_binary_restart_v2.md`.
