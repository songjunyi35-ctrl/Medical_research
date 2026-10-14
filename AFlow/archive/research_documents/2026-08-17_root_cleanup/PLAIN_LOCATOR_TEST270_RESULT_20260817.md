# Plain locator test270 result — 2026-08-17

## Outcome first

The selector-frozen epoch-99 plain locator was evaluated once on the fixed
270-case internal test set. The result is:

| metric | result | percentile-bootstrap 95% CI |
|---|---:|---:|
| scan-macro Dice | **`0.725820`** | `[0.706190, 0.744411]` |
| subject-balanced Dice | **`0.709413`** | `[0.680995, 0.735549]` |
| automatic segmentation-argmax point hit | **`0.900000`** | — |

The cohort contains 270 scans from 119 subjects. Scan Dice median was `0.776645`
and population SD was `0.158864`.

This is supplementary evidence on a previously revealed internal test set. It is
not a pristine blind test and not external confirmation. No checkpoint, threshold,
TTA, postprocessing, architecture, or loss was selected using this result.

## Test-before-decision freeze

The decision was written before test inference to
`workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/TEST270_FREEZE_20260817.json`
(SHA-256 `5388d79bbe6882825770132bfd07759feca3edc6ed3b0910d59336338df4e2db`).
It fixed:

- the selector-chosen epoch-99 checkpoint, SHA-256
  `0bb7ca52d7ac30068f752a4fc05a859f4dce42dce6ace40ba1f6b8b332889c2f`;
- the 270-case manifest, SHA-256
  `a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9`;
- T1ce-only N4 shrink-2 image-derived brain cropping and `128^3` resize;
- raw segmentation logit `>0`, no TTA, no postprocessing, no CAPG, and no SAM;
- metrics-only test use, with no checkpoint or parameter selection.

Epoch 99 was selected before test access by the 108-case selector scan Dice
`0.720419`. The test had zero case and subject overlap with the checkpoint's
756-case train and 108-case selector roles.

## Cache and execution audit

The exact training preprocessing was applied to the frozen test manifest at
`workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270`:

- 270/270 cases completed in 692 seconds;
- cache declares `split=test`, `test_access=true`,
  `spatial=braincrop_resize`, `uses_gt_for_localization=false`, N4 shrink 2;
- images are 2.11 GiB, SHA-256
  `cf834d2b4e6cdaa414689c8723d532178980f6ad459324d13a903321f1064888`;
- labels are 0.53 GiB, SHA-256
  `ecb28a45607d54cf453ece3208ef8261ea30f496e747f919242ac1172445f222`;
- cache index SHA-256
  `41bd6cf44553b8d156de49f2c57bb424cab60cfa26c06881bdfb96ff91de39d3`;
- four independently reprocessed cases reproduced cached crop geometry and label
  mass.

The fixed evaluator exposes no threshold, TTA, postprocessing, or checkpoint
selection arguments. It independently re-hashed the manifest, checkpoint, images,
and labels before inference and rejected any train/selector subject overlap.

## Benchmark position

The primary project benchmark is case-balanced WT Dice: compute Dice independently
for every scan after applying the fixed raw-logit threshold 0, then average all 270
values. Subject-balanced Dice is the key secondary metric: average each subject's
timepoints first, then average the 119 subjects. Point hit is a localization
diagnostic, not the segmentation benchmark.

| project result on the same test270 | scan Dice | subject Dice | current minus prior | inference boundary |
|---|---:|---:|---:|---|
| direct plain locator, epoch 99 | **`0.725820`** | **`0.709413`** | — | T1ce-only, no GT localization or click |
| residual refiner | `0.703798` | `0.686637` | `+0.022022/+0.022775` | GT-centred oracle crop and GT-derived clicks |
| historical brain-crop SAM C3 | `0.625359` | `0.610042` | `+0.100461/+0.099371` | image-derived crop, but GT-derived clicks |
| upstream-literal vanilla best-of-five | `0.594226` | `0.580970` | `+0.131594/+0.128443` | GT-centred crop and interactive clicks |

This is both the highest automatic score and the highest numerical test270 score
recorded by this project. It is not a controlled four-arm comparison because the
architectures, training recipes, geometries, and interaction protocols differ.
It is also not a public BraTS leaderboard result. The DPD-SAM paper's reported
SAM-Med3D reference of `0.741` is on its own split with incompletely disclosed
prompt details; the numerical gap is `0.015180`, but it is not a like-for-like
benchmark.

## Why the benchmark moved: evidence hierarchy

### Directly demonstrated mechanism: do not pass a strong locator through the old frozen SAM

On the historical 108-case development cohort, the same plain locator produced
scan Dice `0.716125` directly. Feeding its prompt into the old frozen SAM produced
`0.596964` with dense-only and `0.599324` with point+dense, losses of `0.119161`
and `0.116801`. The decoder had been trained around a different prompt/mask
distribution and erased useful dense segmentation information. The test270 system
keeps the locator mask itself, so it avoids this measured failure mode entirely.

This is the strongest experimentally supported explanation for the new system
design. It does not by itself attribute the full `+0.100461` over the historical
brain-crop run, because that run also used a different model and training protocol.

### Controlled supporting evidence: image-derived brain cropping uses the fixed grid better

An earlier matched SAM experiment changed only whole-volume resize to image-nonzero
brain-crop resize and moved test scan Dice from `0.571109` to `0.625359`, a paired
`+0.054250` with 95% CI `[+0.0455,+0.0629]` and 232/270 improved cases. Cropping
removes zero background before the fixed `128^3` resize, so more voxels/tokens
represent brain and lesion. The locator reuses this N4 + brain-crop geometry.

That older controlled result supports the geometry mechanism, but there is no
whole-resize versus brain-crop ablation for the current locator architecture, so
its exact contribution to `0.725820` is not separately identified.

### Protocol rescue: Dice-dominant optimization prevented background collapse

The first matched locator attempt used DiceCE with CE weight 1.0 and approached an
all-background solution by epoch 6. Reducing the shared CE weight to 0.1 allowed
both plain and CAPG locators to learn normally. The final model also used
lesion-volume-aware sampling, axis flips, shifts up to 8 voxels, intensity
scale/offset perturbations, and Gaussian noise. These are plausible contributors
to small-lesion robustness, but they were not individually ablated and must not be
assigned separate Dice gains.

### Architecture alignment: a small dense segmenter matches the actual task

The plain locator is a 4,700,897-parameter 3D SegResNet trained directly for the
single T1ce-to-WT dense prediction required at deployment. SAM-Med3D has about
100.5M parameters and carries prompt-encoder/decoder machinery whose frozen prompt
interface was mismatched here. The approximately 21-fold smaller locator is not
automatically better because it is smaller; the defensible claim is that its
training objective and inference output are aligned, while the measured frozen-SAM
cascade is not. A same-recipe architecture swap has not been run, so this remains
a strong mechanism hypothesis rather than isolated causal proof.

### What was rejected

CAPG was not the breakthrough. It changed development locator scan Dice by only
`+0.002965`, its centre-head point hit was `0.638889` versus `0.935185` for the
plain segmentation argmax, and its primary frozen-SAM point+dense output was worse
than plain. Likewise, test-time threshold search, TTA, connected components, and
postprocessing were not used.

### Why this is unlikely to be a validation accident

The selected epoch 99 reached scan Dice `0.720419` on the selector, `0.716505` on
an untouched subject-clean internal confirmation role, and `0.725820` on test270.
Those are different, unpaired cohorts, so their differences are not improvement
estimates. Their consistency is replication evidence. The strict split and fixed
test protocol did not create the segmentation gain, but they make the benchmark
claim substantially more credible.

## Interpretation

The new internal confirmation scan Dice was `0.716505`; test270 was `0.725820`,
a numerical difference of `+0.009315`. The historical development result was
`0.716125`, for a numerical test difference of `+0.009695`. These are different,
unpaired cohorts, so neither difference is an improvement estimate.

Across selector, untouched internal confirmation, and test270, scan Dice is
`0.720419`, `0.716505`, and `0.725820`. That consistency strengthens the conclusion
that direct plain-locator performance is reproducible inside this BraTS 2024 source
cohort. The test subject-balanced result (`0.709413`) and point hit (`0.900000`) are
lower than confirmation (`0.726813` and `0.925926`), but remain compatible with the
bootstrap uncertainty and do not change the method choice.

CAPG's centre head and the old frozen-SAM cascade remain negative. The best current
automatic checkpoint is still the selector-frozen epoch-99 direct plain locator.
Because test270 is now consumed for this arm, it must not be used again to tune or
rank decoder-adaptation variants. Future locator-conditioned SAM decoder work must
develop on train/selector or a newly frozen inner development split and ultimately
requires genuinely external confirmation.

## Evidence hashes

| artifact | SHA-256 |
|---|---|
| evaluator `scripts/evaluate_t1ce_wt_plain_locator_test270.py` | `0ae7c41d80f60af865c1b5edebf0685587c40953bcf555a94efcdb8c770de46b` |
| test start receipt `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_TEST270.json.started.json` | `a97a70f1e6155ebb3e1255510670bc588842298f2c28ef920a790849076e2fc6` |
| final test report `workspace/BraTS/sammed3d_t1ce_wt/plain_locator_subjectclean_confirm_20260817/FINAL_TEST270.json` | `43b88933337e0325caaf97b6add8fbaf64a60b2574d75acaaedc9dceaea7e78b` |

