# FV last-four-block partial-unfreeze + discriminative-LR preregistration

Status: `DEVELOPMENT_ONLY_PREREGISTRATION_READY; TRAINING_NOT_STARTED`

Date: 2026-07-24 UTC

## Scope and access boundary

This document preregisters one development-only training arm. It is not an
authorization to evaluate any test split. Its only data scope is the existing
972-case training split and 108-case / 50-subject clean validation split.

During preparation of this document, only the handoff, development
training/validation artifacts, source code, the untouched source checkpoint,
and development research audits were read. No test or sealed artifact was
opened or enumerated.

Forbidden for this arm:

- any test access or test-driven decision;
- threshold selection, TTA, morphology, connected components, or other
  postprocessing;
- MedTAWS or GRPO;
- full image-encoder unfreezing;
- an LR, layer-count, initialization, augmentation, epoch, or early-stopping
  sweep.

## Evidence and launch decision

The matched clean-validation audit compares the selected whole-volume LoRA arm
against the matched rank-0 arm under the same 972/108 split, preprocessing,
augmentation, interaction protocol, training seed, validation seed, and
50-epoch selection budget.

At click 3:

- LoRA subject-balanced DSC: `0.5214101623`;
- rank-0 subject-balanced DSC: `0.4785050270`;
- paired subject delta: `+0.0429051353`;
- 10,000-resample subject-bootstrap 95% percentile interval:
  `[0.0258629117, 0.0630796008]`;
- subjects improved/worse: `42/8`;
- scan-macro LoRA/rank-0 DSC: `0.5244305128/0.4816013968`.

Source:
`workspace/BraTS/research_audits/FV_LORA_VS_RANK0_MATCHED_VAL108_CPU_PAIRED_20260724.json`.

Both matched curves show continued training fit but noisy, late validation
selection:

- LoRA selected trainer epoch 33 at `0.5328168`; epoch-50 train/validation DSC
  was `0.6310908/0.5235471` (selected-to-final `-0.0092696`, final gap
  `0.1075436`);
- rank-0 selected trainer epoch 32 at `0.4924871`; epoch-50 train/validation
  DSC was `0.5927533/0.4777733` (selected-to-final `-0.0147139`, final gap
  `0.1149800`);
- LoRA and rank-0 had 11- and 15-epoch spans, respectively, between strict new
  validation highs. This motivates patience 20 instead of 10 or 15.

Decision: **GO for exactly one preregistered development arm after its separate
runner passes the engineering checks below.** The result establishes a useful
encoder-adaptation signal under this development protocol. It does not show
that updating full base weights is better than LoRA, so it justifies one
controlled partial-unfreeze test, not a search.

## Unique arm

Run identity:
`fv_partial_last4_disclr10_train972_val108_e50_s20260725`

Initialization:

- fresh load of untouched `sam_med3d_turbo.pth`;
- source checkpoint SHA-256:
  `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`;
- do not initialize from the selected LoRA or rank-0 checkpoint;
- Q/V LoRA rank `0` (disabled); target token disabled.

Trainable parameters:

- all parameters of `image_encoder.blocks.8`, `.9`, `.10`, and `.11`;
- all parameters of `prompt_encoder`;
- all parameters of `mask_decoder`.

Frozen parameters:

- `image_encoder.blocks.0` through `.7`;
- image-encoder patch embedding, positional embedding, neck, and every other
  image-encoder parameter outside blocks 8--11;
- every other model parameter.

Exact CPU architecture counts for `vit_b_ori`:

| Component | Parameters |
|---|---:|
| block 8 | 7,090,752 |
| block 9 | 7,093,056 |
| block 10 | 7,093,056 |
| block 11 | 7,090,752 |
| last four blocks | 28,367,616 |
| prompt encoder | 8,668 |
| mask decoder | 7,575,636 |
| total trainable | 35,951,920 |
| total model | 100,509,232 |

Thus `35.77%` of the model is trainable. This is partial unfreezing, not
full-encoder fine-tuning.

Optimizer:

- AdamW, constant learning rates, no scheduler and no warmup;
- group `head`: prompt encoder + mask decoder, LR `1e-4`, weight decay `0.1`;
- group `encoder_last4`: image-encoder blocks 8--11, LR `1e-5`, weight decay
  `0.1`;
- no bias/normalization no-decay exception;
- the two parameter-name sets must be disjoint and their union must equal the
  complete trainable set;
- gradient accumulation `2`, batch size `1`, BF16, two data-loader workers;
- no gradient clipping unless a preregistration amendment is written before
  any training metric is observed.

The 10:1 head-to-encoder LR ratio is the sole discriminative-LR choice. It must
not be altered after a smoke test or during training.

## Frozen data and interaction contract

- training: the existing 972 rows / 444 subjects;
- validation: the existing 108 rows / 50 subjects;
- no case or subject overlap;
- train seed `20260725`;
- validation/click seed `20260723`;
- single-channel T1ce;
- binary official `WT = raw labels {1,2,3}`; raw label 4 excluded;
- `ToCanonical`, whole-volume resize to `128^3`;
- image interpolation trilinear with `align_corners=False`;
- label interpolation `nearest-exact`;
- source nonzero-voxel population z-score exactly once;
- training-only affine, elastic, and Gaussian-noise augmentation, identical to
  the matched LoRA/rank-0 arms;
- validation has no augmentation;
- exactly three accumulated clicks: first uniformly from GT foreground,
  later clicks uniformly from prediction XOR GT, with FN positive and FP
  negative;
- the previous low-resolution mask is supplied recurrently;
- mean three-click DiceCE training loss;
- prediction rule remains raw logit `> 0`.

No item in this section may be adapted from validation results.

## Epoch cap, early stopping, and checkpoint selection

The maximum is 50 complete epochs. Validation is run once after every complete
epoch with the fixed validation seed.

Performance early stopping is deliberately conservative because the existing
matched curves are noisy and contain late rebounds:

- no stop before epoch 20;
- after epoch 20, stop only when 20 consecutive complete epochs have failed to
  set a strict new high in trainer validation final-click scan-macro DSC;
- `min_delta = 0`; an exact tie does not replace the earlier best and does not
  reset patience;
- incomplete epochs never count;
- if patience is not exhausted, run all 50 epochs.

The selected checkpoint is the strict argmax of trainer validation final-click
scan-macro DSC over all completed epochs from 1 through the stopping epoch or
50. Exact ties retain the earliest epoch. Neither training DSC, validation
loss, subject-balanced DSC, nor an independently reloaded score may reselect
the epoch.

The following are safety stops, not candidate-selection opportunities:
non-finite loss/gradient/metric, OOM, manifest/hash mismatch, a wrong trainable
parameter set, optimizer-group mismatch, or checkpoint integrity failure. A
safety-stopped run is invalid and cannot be called a negative scientific
result.

The selected-to-final validation drop and the final train-minus-validation DSC
gap must be reported as overfitting diagnostics. They do not authorize a
post-hoc LR, patience, or epoch change.

## Restart and nondeterminism rule

The scientific run is the first valid fresh launch in an empty, uniquely named
output directory. It must be uninterrupted; the existing generic trainer has
no audited exact-resume contract.

- A documented infrastructure restart is allowed only if epoch 1 was not
  completed. The failed directory/log is retained and the restart uses a new
  directory suffix while keeping this configuration unchanged.
- Once any epoch has completed, do not silently restart from the untouched
  checkpoint and do not choose the better of attempts. Recovery would require
  a new amendment written before viewing further validation results.

The fixed seed does not imply bitwise determinism: CUDA kernels, BF16, worker
scheduling, and stochastic augmentation can differ across launches. Therefore
this is a one-seed model-development result, not a seed-averaged result.

## Independent validation and fixed decision rule

After trainer selection, independently reload exactly the selected checkpoint
on CPU with the existing paired evaluator. Evaluate all 108 validation cases
once with seed `20260723`. Do not independently evaluate every epoch and do not
reselect based on reload results.

Primary estimand:

`partial-unfreeze minus selected LoRA subject-balanced click-3 DSC`

Reference LoRA value: `0.5214101623252374`.

Click 1 is paired under the same GT-defined sampling. Clicks 2 and 3 are
sampled from each model's own XOR error and therefore need not have identical
coordinates. The primary delta is the conditional closed-loop contrast between
the two selected single-seed models on these 50 development subjects, including
their induced corrective-interaction trajectories; it is not an
identical-prompt, network-only contrast.

Use the same 50-subject mapping and 10,000 subject-level paired bootstrap
resamples, seed `20260724`, two-sided 95% percentile interval with linear
quantiles.

Development advancement requires both:

1. primary point delta is strictly greater than zero; and
2. the paired subject-bootstrap 95% interval lower bound is strictly greater
   than zero.

If either condition fails, the result is **NO-GO**: do not expand seeds, tune
this arm, or evaluate any test split. Scan-macro C1/C2/C3, subject-balanced
C1/C2, improved/worse counts, validation loss, and curve diagnostics are
secondary descriptive results and cannot overturn the primary rule.

Passing this internal gate only supports considering a separately
preregistered confirmation on new development evidence. It does not authorize
test access.

## Engineering gate and resource risks

The anchored generic trainer currently creates one AdamW group. Its
`strategy=full` path unfreezes the entire model and is prohibited. Therefore
the arm requires a separately named runner/helper; the anchored trainer must
remain unchanged.

Before launch, a development-only engineering check must establish:

- exact trainable and frozen parameter-name hashes and the counts above;
- optimizer group names, LR, weight decay, disjointness, and union;
- `train_image_encoder=True`;
- checkpoint metadata preserves the group specification and parameter-name
  hashes;
- strict reload reproduces the partial-unfreeze architecture and selected
  weights;
- the output directory contains no prior run.

A 2-train/2-validation, one-epoch smoke may test only finite
forward/backward/save/reload and resource usage. Its metric may not change any
scientific configuration.

Relative to LoRA, the arm adds 28,220,160 trainable parameters. A rough
FP32-model plus two-state Adam estimate is about 658 MiB per checkpoint; latest
plus selected-best is about 1.29 GiB before serialization overhead. Runtime
peak memory is estimated, not measured, at roughly 7--10 GiB on the prior
setup, and throughput at roughly 0.7--1.1 cases/s. The smoke must record actual
peak allocation/reservation and throughput. These estimates are not results.

## Claims explicitly disallowed

Regardless of outcome, do not claim:

- external, held-out, or test generalization;
- that partial unfreezing is causally superior to LoRA from one selected
  single-seed validation run;
- that click-3 is an identical-prompt network-only comparison, because the
  later clicks are induced by each model's own errors;
- statistical significance that accounts for epoch selection, seed
  variability, or restart variability; the subject bootstrap conditions on
  the selected models and subjects;
- that a fixed seed is a replicate;
- a `0.7` result unless that value is actually obtained under this exact
  preregistered validation metric;
- improvement from thresholding, TTA, postprocessing, MedTAWS, or GRPO;
- full image-encoder fine-tuning.

The strongest permitted positive wording after passing the gate is:
"Under one preregistered seed and the matched clean-validation protocol,
last-four-block partial unfreezing had a positive paired subject-level
development estimate over the selected LoRA reference; independent
confirmation remains required."
