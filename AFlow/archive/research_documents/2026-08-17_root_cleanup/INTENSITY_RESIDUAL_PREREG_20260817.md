# Intensity-aware uncertainty residual preregistration — 2026-08-17

Status: protocol frozen before split construction, implementation tests, training,
or metric access. A code-hash amendment must be appended before formal training.

## Question

Does explicit case-adaptive T1ce intensity information improve the direct plain
locator when it is restricted to uncertainty-gated residual corrections, without
using a hard global gray threshold?

## Isolation

- Source role: only the already-frozen 756-case train role from
  `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl`,
  SHA-256 `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`.
- Create a new subject-clean 648 fit / 108 inner-tune split using seed `20260820`.
- Existing 108-case selector remains unread until the inner decision is locked.
- The 108-case confirmation role and 270-case test manifest/images/labels are
  forbidden throughout this experiment.
- No historical 108 validation data are used.

## Matched arms

All arms use the same data order, seed `20260821`, N4 + image-nonzero brain crop,
`128^3`, batch 8, 100 epoch maximum, minimum 30 epochs, patience 20, AdamW
`lr=2e-4`, weight decay `1e-5`, DiceCE with CE weight `0.1`, fixed raw-logit
threshold 0, and the existing lesion-volume sampler/augmentation.

1. `plain`: unchanged 4.70M-parameter plain SegResNet locator.
2. `residual_control`: plain locator plus a zero-initialized uncertainty-gated
   residual head. Its three explicit intensity feature channels are exactly zero.
3. `intensity_residual`: architecture and initialization identical to
   `residual_control`, but the three channels contain clipped normalized T1ce,
   local z-contrast, and gradient magnitude.

For both residual arms, the head also receives detached base logit, probability,
and uncertainty. The final logit is

`base_logit + 4*p*(1-p)*residual_logit`,

with the uncertainty gate detached so the base cannot enlarge it to game the
residual path. The final residual convolution is initialized to zero; both arms
start exactly at their matched base output.

## Endpoints and decision

Checkpoint selection within each arm uses inner-tune scan-macro Dice at raw logit
0. The causal primary contrast is `intensity_residual - residual_control`; the
deployment contrast is `intensity_residual - plain`.

Inner success requires all of:

- intensity minus control scan Dice at least `+0.003`;
- intensity minus control subject-balanced Dice at least `+0.003`;
- intensity not worse than plain by more than `0.001` on either metric;
- no drop greater than `0.02` in automatic point hit rate;
- all complete per-case reports reload exactly from the selected checkpoints.

Paired scan and subject bootstrap intervals use 10,000 draws, seed `20260820`.
Intervals are descriptive; the numerical gates above control continuation.

If and only if every inner gate passes, lock the winning checkpoint and perform
one fixed evaluation on the existing 108-case selector. Selector success requires
scan and subject Dice each at least `+0.003` versus a matched plain checkpoint.
No threshold, connected-component, TTA, seed, epoch, feature kernel, residual
width, or loss-weight search is allowed after results are visible.

If the inner gates fail, stop the intensity-residual arm without reading the outer
selector. A multi-subregion auxiliary head may be designed later as a separately
preregistered experiment; it is not a rescue edit to this run.

## Evidence boundary

Even a positive selector result is development evidence, because the selector has
been used previously for plain-checkpoint selection. Confirmation and test are no
longer available for tuning and will not be consumed. Any generalization claim
requires a new external cohort.


## Formal-training code-hash amendment — 2026-08-17T07:11:29Z

This amendment was written after implementation/tests and before GPU smoke or
formal training. It does not change the protocol above.

- split manifest SHA-256:
  `d47e0e0843fd54e63a18cdadadcb0f95a4fd75cca26a5bb3007b634cc4a1536f`;
- fit role SHA-256:
  `947a5364fd2550f5e63af835612a9b97157f7c51eee28c5333de81260dfea19e`;
- inner-tune role SHA-256:
  `711f441addbc28208ee07e22f9fd14f4345313c123f809b81f1fb02d8826b3f8`;
- model SHA-256:
  `1e45cf8531c9187a0f0071963f52d099ae8399b9e291332dfd6ffa355c8498ce`;
- split builder SHA-256:
  `f6e383286c6f2d0a90b17e62071f2b6fb9e9c383a50b025b6da4dee40fcd2163`;
- trainer SHA-256:
  `0b26d4479c06330f9021f3414ba1a187be68289a5d9c3096ab786ddc1cac2f09`;
- reload evaluator SHA-256:
  `cecb33fc974fa68a0c50ada3dc850033a4e5f7c0589a8ec319606b1ccd1d530a`;
- decision analyzer SHA-256:
  `6bb1318f70dcc1643e0b25c60d35694f90acca7211dd5fe301b0aa614ea9abf6`;
- model tests SHA-256:
  `2e4991bf08fcb22ffff1aa0a92e13c089ee957363c854732a08177ef2b18538b`;
- analysis tests SHA-256:
  `33fc53305d30e0270a9a127dafde598cc96ecdacb95e37e7d126722bf9014bf7`;
- train-cache index SHA-256:
  `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.

The targeted model/analysis/locator regression completed with `14 passed`.
Formal outputs are frozen under
`workspace/BraTS/sammed3d_t1ce_wt/intensity_residual_inner_20260817/seed20260821/`.
All three arms run sequentially on the same H100; no selector, confirmation, or
test evaluator is launched by the training entry point.
