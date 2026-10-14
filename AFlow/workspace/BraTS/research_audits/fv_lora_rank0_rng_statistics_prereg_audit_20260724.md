# FV-LoRA vs rank-0 clean-validation RNG and statistics preregistration audit

Date: 2026-07-24 UTC

## Scope

This is a read-only, development-only audit of the matched whole-volume
FV-LoRA and rank-0 arms. It did not run training or evaluation and did not
access any external holdout. The rank-0 run had written `train_config.json`
and `t1ce_wt_protocol.json`, but had not written `best.json` or `TRAIN_DONE`
at the audit snapshot, so the matched contrast was still unavailable.

The relevant frozen implementation hashes were rechecked:

- checkpoint evaluator:
  `scripts/evaluate_sammed3d_t1ce_wt_checkpoint.py`,
  SHA-256
  `183886e13a7a6aacbe01f9c27f7cc9860edcb081bb7b84ec24ec98fba034c02e`;
- trainer-RNG replay wrapper:
  `scripts/replay_sammed3d_t1ce_wt_val_trainer_rng.py`,
  SHA-256
  `278effb8b732d3cfcae9a685bb5119e88fc24f4a756b9a1ad469d623869b3920`;
- T1ce/WT trainer:
  `scripts/train_sammed3d_t1ce_wt.py`,
  SHA-256
  `369986ff41166ffbbe8081b5767e1f276ac14a7ddbe03acce3b800d6ba4b0988`;
- preprocessing module:
  `sammed3d/t1ce_wt_training.py`,
  SHA-256
  `4d13e2b3894e2b9a91fac7f76dd74b1c744429eef665f09a5f160a604b0989ae`.

## What the existing LoRA reload evidence establishes

The 108-case clean-validation manifest is byte-bound by SHA-256
`fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c`.
Its stored order equals the trainer's `(target, case_id, image_path,
label_path)` order; it has 108 unique cases and 50 subjects.

For the selected LoRA epoch-33 checkpoint, SHA-256
`c1d1bd79791aa7beeffc541095472dec968630cc70ea7967128c86552bd7f1ec`:

| Reload view | RNG | scan-macro C3 | subject-balanced C3 |
|---|---|---:|---:|
| independent | case-seeded PCG64 | 0.5214277774 | 0.5182634436 |
| trainer replay | one sequential MT19937 stream | 0.5328167620 | 0.5283455755 |

The MT19937 replay differs from trainer epoch-33 C3 by only
`-3.47e-10`. Both reports bind the same manifest, checkpoint, adapter
architecture, whole-volume preprocessing, BF16 inference, three accumulated
clicks, recurrent low-resolution mask and raw-logit threshold. All 324 click
records in each report have `status=sampled`. Thus the observed LoRA
trainer/reload discrepancy is explained by the RNG contract, not failed
checkpoint reload or failed LoRA restoration.

This result is checkpoint-specific, not a proof that the wrapper is
universally equivalent to trainer validation. If a prediction becomes exact,
the evaluator records `skipped_prediction_already_exact` and does not consume
another random draw, whereas the trainer samples a fallback GT-foreground
point and runs another forward pass. A future MT19937 replay may be called
trainer-equivalent only if every click is sampled, or if a separate
audit-only replay implements the trainer's exact fallback behavior.

The independent report stores only `seed=20260723`; it does not embed the RNG
algorithm or case-seed derivation. A final matched-result binding should
therefore explicitly record:

- `algorithm = NumPy Generator(PCG64)`;
- `case_seed = first 8 bytes, little-endian, of
  SHA256(f"{seed}:{case_id}:residual")`;
- one independently initialized stream per case;
- evaluator SHA, both checkpoint/report SHA values and exact case-ID parity.

## Audit of matching beyond evaluation RNG

Recursive comparison of the actual LoRA and in-progress rank-0
`train_config.json` and `t1ce_wt_protocol.json` found only these differences:

- LoRA enabled/rank `true/4` versus rank-0 `false/0`;
- derived image-encoder update metadata (`qv_lora` and 12 Q/V blocks versus
  frozen and no adapter blocks);
- trainable/total parameter counts;
- output namespace.

All recorded data, preprocessing, augmentation, optimizer, epoch, batch,
accumulation, worker, BF16, training seed, loader seed, validation seed,
selection metric and source hashes match. LoRA starts functionally equal to
rank-0 because each LoRA B matrix is initialized to zero, while the base QKV
weights remain frozen. Adapter injection restores the pre-injection Torch RNG
state, so it does not shift later stochastic streams.

The comparison is nevertheless not a bitwise paired intervention:

1. LoRA changes the trainable parameter set and therefore the entire
   optimization trajectory; this is the intended treatment.
2. Corrective click 2/3 is sampled from each model's own error set. The same
   pseudorandom draw is mapped to different eligible voxels when the models
   disagree. This is a paired end-to-end interactive-policy comparison, not a
   fixed-prompt comparison.
3. Each arm selects its own best epoch on the same 108 validation cases.
   Subsequent uncertainty on those same cases is conditional and exploratory;
   it does not account for checkpoint-selection optimism.
4. Only one training seed is present, and deterministic CUDA algorithms and a
   complete runtime/hardware fingerprint were not preregistered. The contrast
   cannot estimate across-training-seed variance.

## Recommended endpoint hierarchy to freeze before the rank-0 result

### Primary

Use only the independently reloaded, case-seeded PCG64 reports for the two
trainer-selected checkpoints. Do not evaluate multiple epochs under PCG64 and
reselect a checkpoint.

The primary estimand should be subject-balanced final-click Dice:

`d_case = DSC_LoRA,C3 - DSC_rank0,C3`

`d_subject = mean(d_case for all scans belonging to that subject)`

`Delta_primary = mean(d_subject over the 50 subjects)`

This keeps the independent subject, rather than the longitudinal scan, as the
equal-weight unit. Require exact equality of case-ID and subject-ID mappings
between reports before computing any contrast.

Report a paired subject-cluster bootstrap interval: resample the 50 subjects
with replacement, retain all scans belonging to every sampled subject, and
recompute `Delta_primary`; freeze the bootstrap algorithm, replicate count
(at least 10,000), seed and percentile/BCa convention before reading the
rank-0 report. A naive independent-scan bootstrap is invalid for this
longitudinal split.

Because checkpoint selection used these same subjects, label this interval
as conditional development uncertainty, not confirmatory coverage. If a
numerical minimum practically relevant difference is desired, it must be
frozen before the rank-0 result; otherwise avoid the word "material" and
report the point estimate and interval without inventing a post-hoc gate.

### Mandatory secondary

1. Independent PCG64 scan-macro C3 delta, because scan-macro C3 was the
   checkpoint-selection metric.
2. Independent PCG64 subject-balanced and scan-macro C1/C2, plus C3 IoU.
3. Paired improved/worse/tied counts at both subject and scan levels, with a
   predeclared tie tolerance.
4. Each arm's selected epoch, trainer validation C3 and independent-reload
   C3, so any RNG sensitivity is visible.
5. Parameter counts and the exact actual-config diff.

C1/C2 and IoU should remain descriptive unless a multiplicity correction is
preregistered. Do not switch the primary click or aggregation after seeing
which one favors an arm.

### Diagnostic secondary

Run the legacy sequential MT19937 replay only as a reload/selection
diagnostic. It is row-order-dependent and is not the independent headline
estimate. Before accepting it:

- confirm the manifest order equals trainer order;
- require all 324 clicks to be sampled, or use an exact trainer-fallback
  replay;
- bind its wrapper hash, seed and report hash;
- compare each replay C3 to that arm's selected trainer value.

The MT19937 LoRA-vs-rank-0 delta may be reported as a sensitivity analysis,
but it must not replace the case-seeded PCG64 primary merely because it is
larger or changes sign.

## Claim boundary

The strongest permissible wording is:

> On the reused 108-case/50-subject clean validation set, under one matched
> training seed and the preregistered independent case-seeded three-click
> evaluator, enabling 12-block Q/V LoRA produced a subject-balanced C3
> difference of Delta versus the rank-0 prompt/decoder control (paired
> subject-bootstrap interval ...).

Do not claim:

- external or confirmatory generalization;
- an across-seed LoRA effect;
- a fixed-prompt comparison at corrective clicks 2/3;
- that a small positive point estimate is statistically or practically
  decisive when the interval crosses zero or no practical threshold was
  frozen;
- that geometry, augmentation or whole-volume preprocessing caused the
  contrast, since those are shared;
- that the matched contrast authorizes any external-holdout rerun or new
  threshold/TTA/morphology/postprocessing search.

