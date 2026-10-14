# FV-LoRA-Aug versus matched rank-0 on clean validation

Date: 2026-07-24 UTC

## Scope and evidence boundary

This is a development-only matched comparison for single-channel T1ce to
binary official WT `{1,2,3}`, with RC `{4}` excluded, whole-volume `128^3`
resize, train-only affine/elastic/Gaussian-noise augmentation, three
accumulated corrective clicks, and a recurrent low-resolution mask.

Both arms use the same 972-train / 108-validation subject-clean split, untouched
base checkpoint, optimizer settings, seeds, 50-epoch budget, and
validation-selected checkpoint rule. The intended treatment contrast is
image-encoder Q/V LoRA rank 4 versus rank 0 (frozen image encoder); prompt
encoder and mask decoder are trainable in both arms.

No external holdout manifest, image, label, or per-case report was opened for
this work. No threshold, TTA, morphology, postprocess, or MedTAWS search was
performed.

## Matched rank-0 training

The first attempt was interrupted by infrastructure reclamation during epoch 3.
It was moved intact to an `INVALID_WORKER_RECLAIMED_E3_PARTIAL_*` namespace and
was not resumed, spliced, averaged, or used for checkpoint selection.

The authoritative foreground restart completed all 50 epochs:

- trainable / total parameters: `7,584,304 / 100,509,232`;
- image encoder: fully frozen, no LoRA modules;
- selected epoch: `32`;
- trainer validation C3 at epoch 32: `0.492487125685`;
- epoch-50 train / validation C3: `0.592753287323 / 0.477773271139`;
- total elapsed time: `37,358.48 s`;
- selected checkpoint SHA-256:
  `7a25d0268799bdfffc626eacecb2a319a51d8127460143e7c6af3e8d4b532cf9`;
- epoch-50 latest checkpoint SHA-256:
  `6cbea9a0ea78a34d33a3370f6e2d0c0634ad6713925618d3b9d248532e4141d0`.

The foreground worker exited with code 0 after printing `TRAIN_DONE`.

The interrupted and authoritative attempts were not bitwise reproducible despite
the same recorded seed and configuration: their epoch-1 validation C3 values
were `0.392777477` and `0.380767430`. Therefore the final comparison is a
single-run conditional matched contrast, not a deterministic replication.

## Independent matched reload

Remote GPU provisioning failed at the platform exec/SSH layer after training.
To avoid mixing hardware, both selected checkpoints were therefore reloaded
sequentially on the same local CPU runtime with the same independent
case-seeded PCG64 evaluator. Both reports contain 108 cases / 50 subjects,
324 sampled clicks, no skipped-exact cases, identical protocol/manifest/seed,
and finite per-case metrics.

| Arm | Selected epoch | C1 | C2 | C3 | Subject-balanced C3 |
|---|---:|---:|---:|---:|---:|
| Matched rank-0 | 32 | 0.446168 | 0.473803 | 0.481601 | 0.478505 |
| FV-LoRA-Aug rank 4 | 33 | 0.504483 | 0.513365 | 0.524431 | 0.521410 |

The same-hardware CPU LoRA C3 (`0.524431`) differs from the earlier A10
case-seeded reload C3 (`0.521428`) by `+0.003003`. This hardware/runtime
sensitivity is why the formal matched contrast uses the CPU-paired reports
rather than mixing reports across devices.

## Paired comparison

Direction is always `LoRA candidate - rank-0 reference`. The bootstrap samples
50 subjects with replacement, keeps all longitudinal scans within each sampled
subject, uses 10,000 iterations, seed `20260724`, and reports a percentile 95%
interval.

| Click | Scan-macro delta | Cases improved / worse | Subject-balanced delta | Subjects improved / worse | Subject bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| C1 | +0.058314 | 90 / 18 | +0.062021 | 45 / 5 | [+0.044458, +0.080422] |
| C2 | +0.039561 | 80 / 28 | +0.041034 | 43 / 7 | [+0.027284, +0.054894] |
| C3 | **+0.042829** | **81 / 27** | **+0.042905** | **42 / 8** | **[+0.025863, +0.063080]** |

The C1 prompts are identical across arms for all 108 cases. C2 and C3 are
model-adaptive: each arm samples from its own current XOR error. Their click
coordinates are identical in `0/108` cases at both C2 and C3, while click labels
match in `70/108` and `67/108` cases. The later-click deltas therefore estimate
the total closed-loop effect of each selected model plus the corrective
trajectory it induces, not a fixed-prompt direct network-output contrast.

Within this conditional development comparison, the matched LoRA arm has a
positive C3 point estimate and a subject-cluster bootstrap interval above zero.
The advantage is already present at the first click and remains after three
corrective clicks.

## What may and may not be concluded

Supported wording:

> Under one fixed training seed and the matched whole-volume development
> protocol, the validation-selected FV-LoRA-Aug run outperformed the matched
> frozen-image-encoder rank-0 run by `+0.0429` subject-balanced C3 DSC; a
> 50-subject paired bootstrap interval was `[+0.0259,+0.0631]`.

Unsupported wording:

- a confirmatory or external-generalization result;
- an across-seed LoRA effect;
- a deterministic pure causal effect of LoRA, because worker/runtime
  nondeterminism was observed;
- evidence that the method reaches `0.7`;
- evidence for threshold, TTA, morphology, postprocess, or workflow search.

The same 108-case validation set selected each arm's epoch and estimated the
contrast. The interval therefore quantifies paired subject variation conditional
on these selected runs; it does not correct checkpoint-selection optimism or
single-seed training uncertainty.

## Evidence bindings

- rank-0 history:
  `3e1c87e93ecf2881b23657c2562d15fd541ab4b6588a4b151315332551700c80`
- rank-0 `best.json`:
  `fb865b8c4ec46de8b2aa44c9f757674a663f4c11e79988dd82c0f41a6dc9cba3`
- rank-0 train config:
  `c8eb7be25d07030b89326abdb0e121cd5eab9fa6571bb8fd58b066155e3148ff`
- rank-0 training protocol:
  `5c9f70260c05131ab5cbff167fe87f3baebddb1446b1aa7599b69e0c2ed678a0`
- rank-0 CPU independent report:
  `59c69d7b5adf47327ea75d55c9edd1d928bd51af39bd9482db4e0511da6fc2e8`
- LoRA CPU independent report:
  `83c726baa0ea02f1eedae57591644fafd64f053a08cf2b61de4f1e42457fe577`
- paired bootstrap result:
  `e9918aedb8825ca87147d61f321c03f296e0014ddf07426f7017e08770ce01ee`
- paired analysis implementation:
  `ee911b93cdaeee75657f15e284c7b176286e90d1a73e13cbab40986a661d64a5`
- RNG/statistics preregistration audit:
  `5f2edf4de75ea343b33cdc5dc0f99b356d9039f9e333cf504cbdad55e8c2246c`
