# DPD-SAM-compatible T1ce binary restart protocol v1

Date: 2026-07-22
Status: frozen reconstruction contract; supersedes v0 for all new work

`authorizes_confirmatory_access=false`  
`authorizes_test_access=false`

This document does not modify or erase `dpdsam_t1ce_binary_restart_v0.md`.
The v0 uncertainty is retained as an audit record; the target-mapping question is
resolved below. This protocol does not authorize a GPU launch by itself.

## 1. Source status and scope

The method specification comes from the local draft `article/DPD-SAM.pdf`,
SHA-256
`ce49d74a94dcd3529c394a39458ea8583ac8d8226aa0d02de099112171473b4d`.
The file is an IEEE-style draft (`VOL. XX`, created 2026-07-21). No public DOI,
publisher record, arXiv record, or public author implementation has been verified.
Consequently, its reported results are a local-draft target, not an externally
validated or independently reproduced benchmark.

This restart is deliberately limited to:

- input: T1ce only, locally `modalities.t1c`, one channel;
- output: one binary residual-tumor mask;
- model input geometry: the entire volume resized to exactly `128 x 128 x 128`;
- primary metrics: ordinary case-wise binary DSC, IoU, and HD95 in millimetres;
- experiment: one frozen subject-disjoint split and five independent trainings.

The stopped four-modal five-class P0/P0+ runs are historical diagnostics only.
Their D3/D6 values are not comparable to this binary task and they must not be
resumed as this baseline.

## 2. Frozen binary target

Foreground is the union of BraTS labels `{1,2,3}`. Background is label `0` plus
label `4`; residual cavity (`RC=4`) is therefore explicitly excluded.

Canonical name: `residual_tumor_123` (`DPD_WT_no_RC` is an alias only). Do not
reuse the historical local `tumor_WT={1,2,3,4}` target.

### Mapping evidence and provenance

The draft's BraTS cohort table reports residual-tumor volume median (IQR)
`41.64 (62.14) ml` and proportion below `5 ml` of `2.8%`. A local aggregate audit
over the 1350 segmentation records computed physical volume as
`count(label in {1,2,3}) * voxel_volume / 1000` and obtained median `41.641 ml`,
IQR `62.141 ml`, and `<5 ml` proportion `2.81%`. Agreement at every displayed
precision identifies `{1,2,3}` as the draft's binary foreground and rejects both
`ET={3}` and `{1,2,3,4}`.

This is provenance evidence, not a model evaluation. The aggregate audit occurred
before this v1 freeze and included the full 1350-case cohort, including records in
the historical outer split. This v1 task did not reopen any NIfTI or test label.
The historical outer set has also been accessed by earlier, unrelated research;
it is therefore not pristine relative to the whole project history. It remains
sealed from this freeze onward, and that limitation must accompany any paper claim.

## 3. Frozen split policy

Keep the existing outer test membership of 269 cases unchanged and sealed. The
remaining 1081-case development pool is the only pool available for split
construction, training, model selection, threshold-independent validation, and
debugging.

Within the development pool, construct one deterministic subject-grouped split
targeting 872 fit cases and 209 validation cases. Subject identity, not scan, is
the hard grouping unit: all longitudinal scans from a subject must remain in one
partition. Because subjects can contribute different numbers of scans, `872/209`
is a case-count target rather than a bit-exact guarantee. Choose the closest
deterministic feasible allocation without breaking subject disjointness, record
the achieved case and subject counts, and freeze a content-hashed ledger before
any training.

Required checks are zero subject overlap, zero case overlap, union equal to the
1081-case development pool, and no outer-test ID in either development partition.
All five runs must use the same frozen ledger. Do not regenerate a favourable
split per seed.

## 4. Preprocessing and geometry

The network receives one whole-volume tensor of shape `[1,128,128,128]`. No
patch crop, foreground crop, sliding window, or four-modal channel is permitted.
Image resize uses trilinear interpolation; label resize uses nearest-neighbour
interpolation. Store the source affine, shape, and spacing for evaluation.

Unless recovered author code proves otherwise, the reconstruction uses image-only
non-zero-brain z-score normalization and treats the distributed BraTS image as
already skull stripped. Bias-field correction is not silently invented: either
freeze a concrete N4 implementation and cache it for every split before training,
or declare it unavailable and run a named no-N4 reconstruction. This choice must
not change after validation results are seen.

For metrics, map the thresholded `128^3` prediction back to the source grid with
nearest-neighbour interpolation. Compute DSC and IoU against the source-grid
binary mask and HD95 with the source physical spacing. The model output threshold
is fixed at `0.5`, as stated in the draft; no threshold search or post-processing
search is part of this baseline.

## 5. Metric contract and empty-mask policy

Metrics are computed per case and then macro-averaged. For binary masks `P` and
`G`:

- DSC is `2|P intersect G|/(|P|+|G|)`;
- IoU is `|P intersect G|/|P union G|`;
- if both masks are empty: DSC `1`, IoU `1`, HD95 `0 mm`;
- if exactly one mask is empty: DSC `0`, IoU `0`, and HD95 equals the physical
  diagonal of the evaluated source field of view.

The one-empty HD95 convention is a finite, deterministic reconstruction choice;
the draft does not specify it. Report the numbers of both-empty and one-empty
cases alongside the three primary metrics. Also report foreground-present-only
metrics as diagnostics, never as a replacement for the all-case primary result.

Across five runs, report arithmetic mean and sample standard deviation (`n-1`).
Do not pool voxels across cases or substitute WT/TC/ET D3 or lesion-wise D6.

## 6. Label-free prompt firewall

Validation and test prompts must be functions only of the T1ce image, frozen model
state, and declared random seed. They may be learned continuous tokens or produced
by an image-only automatic prompt module. They must not use the ground-truth mask,
its centroid, bounding box, connected components, distance transform, foreground
extent, or any per-case statistic derived from the label.

Ground truth is available in fit only for loss computation and in validation only
for frozen metric/model-selection computation. An oracle GT-derived point or box
experiment, if retained for debugging, must live in a separately named diagnostic
artifact and cannot be presented as the automatic SAM-Med3D baseline. Test labels
remain inaccessible until the five seed checkpoints and the complete evaluation
contract are frozen.

## 7. Five-run protocol

Use five declared seeds (reconstruction default: `20260722` through `20260726`).
Each seed independently controls initialization of trainable modules, data order,
augmentation, K-means++, diffusion noise, and all other stochastic operations.
The frozen SAM-Med3D checkpoint, split, preprocessing cache, loss, optimizer,
schedule, checkpoint rule, prompt contract, and evaluator are identical.

Select one checkpoint per seed using validation data only. Freeze all five
checkpoint hashes and the evaluation container/environment before unsealing the
outer set. Then evaluate each seed exactly once; do not select seeds, checkpoints,
thresholds, or post-processing from outer-test performance.

The draft-declared settings are SAM-Med3D with frozen image encoder, learned CLS,
K-means++ `K=4`, rank-1 DPG, adapter and mask decoder; AdamW `lr=4e-4`, weight
decay `0.1`, batch size `2`, 200 epochs, 50-epoch diffusion-free warm-up;
`alpha=beta=0.1`, DDPM `T=1000`, and DDIM `S=10`.

## 8. Declared reconstruction assumptions

The local draft does not uniquely specify the SAM-Med3D checkpoint, automatic
baseline prompt generator, Dice/BCE internal weights, image normalization formula,
bias-field parameters, resize alignment, augmentation probabilities/elastic/noise
parameters, five seeds, optimizer betas/epsilon, learning-rate schedule, AMP and
gradient handling, CLS/token dimensions, adapter width, K-means feature
normalization/seed, diffusion beta schedule, checkpoint selection rule, or
empty-mask metric policy.

Every missing item must be fixed in a versioned implementation manifest before a
GPU launch. Later recovery of author code creates a new protocol version; it must
not silently alter v1. MedTAWS may begin only after this label-free T1ce binary
SAM-Med3D baseline is implemented, validated, and frozen. MedTAWS uses validation
only and cannot search thresholds, TTA, post-processing, or the sealed outer set.

## 9. Launch gates

1. Materialize and hash the preserved outer-269 membership and the new closest
   feasible subject-grouped `872/209` development ledger without reading labels.
2. Implement and test the T1ce-only loader, `{1,2,3}` target with `4` excluded,
   exact whole-volume `128^3` path, metric empty policy, and prompt firewall.
3. Freeze all reconstruction assumptions in an implementation manifest.
4. Pass CPU smoke tests and a label-access audit; only then authorize GPU smoke.
5. Complete the five validation-selected seeds before any new outer-test access.
