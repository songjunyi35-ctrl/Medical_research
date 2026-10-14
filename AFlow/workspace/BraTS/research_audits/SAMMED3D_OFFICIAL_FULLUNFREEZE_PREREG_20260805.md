# SAM-Med3D official-recipe full-unfreeze arm preregistration (2026-08-05 UTC)

Run identity: `sammed3d_official_fullunfreeze_ddp2_train972_val108_e200_s20260805`

Status: **AWAITING_USER_SIGNOFF** -- no formal training may start until this is approved.

## 1. What this arm is, and is not

Follow SAM-Med3D's own published training recipe -- full image-encoder unfreeze with the encoder learning rate above the heads, plus its epoch budget and decay schedule -- on the frozen 972/108 clean-development split, to see how far the T1ce->WT arm moves toward the SAM-Med3D BraTS 2024 DSC of 74.1 +/- 1.5 that DPD-SAM reports. That baseline was most plausibly produced with this same official recipe, which is why it is the recipe being followed here rather than DPD-SAM's own frozen-backbone method.

It is **not** a controlled experiment. The following change together, by intent:

- FULL image-encoder unfreeze: 35,951,920 -> 100,509,232 trainable
- discriminative-LR direction INVERTED to SAM-Med3D's own (encoder above heads)
- N4 bias field correction added to preprocessing
- max epochs 50 -> 200 with official MultiStepLR([120,180], gamma 0.1)
- effective batch 2 -> 12 (per-GPU 2 x 2 GPUs x 3 accumulation)
- single-GPU -> 2-GPU data parallelism with explicit gradient all-reduce

Several factors move together by design. No outcome may be attributed to any single factor, and this arm may not be presented as a controlled contrast against the rank-0, LoRA, or last-4 partial-unfreeze arms.

It is also **not** a reproduction of DPD-SAM. Reaching any particular value on this validation set does NOT reproduce the DPD-SAM-reported 74.1: that number was measured by different authors, on their own split, under a prompt protocol their paper does not disclose.

## 2. Unchanged task definition

T1ce single channel, binary official `WT={1,2,3}` with RC `{4}` excluded,
whole-volume `128^3`, 3 accumulated corrective clicks with the recurrent
low-resolution mask, raw logit `> 0`, validation seed `20260723`.

## 3. Data

| Split | Rows | Subjects | Manifest SHA-256 |
|---|---:|---:|---|
| train | 972 | 444 | `ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e` |
| validation | 108 | 50 | `fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c` |

The 270-case test stays **permanently frozen**: not read, not evaluated, not
used for any selection.

## 4. Preprocessing and cache binding

Deterministic preprocessing is precomputed once and bound by content hash. The
cache boundary sits after the deterministic chain and before the stochastic
augmentation, so per-epoch randomness is preserved exactly. With N4 disabled the
cache was verified bit-identical to the live pipeline before the N4 caches were built.

| Cache | images.f32 SHA-256 | labels.u8 SHA-256 |
|---|---|---|
| train | `551c3d9965a7dc1507d6d9800842c4760be518e34ba8cba6a4e9acd3fccdd3ae` | `7d186ad0baa433d9fb4915ee6bb0f7daa1be716ecb3258f6531768755c6b91fc` |
| validation | `f5897553b6c5c22c7265232150bd86ed4284d864a21f85501dba12881ce701bf` | `17cec1d7a71cd210064f699148db252b0dc5fa73b487771ba78ad737e91ccc81` |

N4 bias field correction: SimpleITK, 4x50 iterations, nonzero brain mask, shrink
factor 2, applied to the raw canonical
volume before normalization. Skull stripping is not applied because BraTS 2024
volumes are already skull-stripped.

## 5. Model and optimization

From the untouched base checkpoint
`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`; LoRA and target token disabled.

**Trainable policy: FULL unfreeze: entire image encoder + prompt encoder + mask decoder** --
`100,509,232 / 100,509,232`, i.e. every parameter.

| Group | Params | LR | WD |
|---|---:|---:|---:|
| `image_encoder_full` | 92,924,928 | `0.0001` | `0.1` |
| `prompt_encoder_and_mask_decoder` | 7,584,304 | `1e-05` | `0.1` |

The encoder LR sits **above** the head LR, matching SAM-Med3D's own `train.py`
and inverting the direction used by the matched arms.

BF16, per-GPU batch 2 x 2 GPUs x
3 accumulation steps = **effective batch
12**. Max 200 epochs with
MultiStepLR(milestones=[120, 180],
gamma=0.1). No performance early stop before epoch
40, then stop only after
40 consecutive complete epochs without a strict new
validation high. Train seed `20260805`.

Measured resource envelope: 11.35 GiB peak at per-GPU batch 2; batch 4 OOMs on a 23 GB A10.

### 5.1 On the learning-rate magnitude

SAM-Med3D's 8e-4 belongs to ITS effective batch of 240 (batch 12 x accumulation 20), which on 972 cases would give only ~4 optimizer steps/epoch. Absolute LR is not transplantable; 1e-4/1e-5 is a conservative sqrt-scaling to effective batch 12. Only the 10:1 direction and the decay schedule are inherited.

### 5.2 A standing prohibition was lifted for this arm

The research protocol forbade **full image-encoder unfreeze**. It was lifted by the
operator on 2026-08-05, on this evidence:

- SAM-Med3D train.py trains image_encoder at full LR, heads at 0.1x
- project results monotone in encoder adaptation: 0.4785 frozen -> 0.5214 LoRA -> 0.5072 by epoch 7 for blocks 8-11
- the prohibition carried no supporting experiment and was grouped with test-hacking techniques

Recorded here so the change of regime is explicit and auditable, not silent.

## 6. Selection and readout

Selected checkpoint = strict argmax of validation final-click scan-macro Dice
over completed epochs, ties keeping the earlier epoch. Afterwards exactly one
independent strict-reload evaluation on the same 108 cases / 50 subjects
produces the subject-balanced C3 readout.

Same 108 cases / 50 subjects, same case-seeded PCG64 click sampling and validation seed 20260723, same subject-balanced aggregation as the existing paired evaluator; only the image source changes (N4 cache instead of on-the-fly preprocessing).

This arm is trained on N4-corrected inputs, so it must also be evaluated on N4-corrected inputs to avoid a train/eval preprocessing mismatch. The LoRA and rank-0 reference values were produced WITHOUT N4. The reference numbers therefore sit on different model inputs, which is an additional reason this is a descriptive context comparison and not a controlled contrast.

Descriptive reference points on the same validation set: LoRA
`0.5214101623252374`, rank-0
`0.478505027`. These are context, not a
controlled contrast.

## 7. Prohibitions still in force

- no 270-case test access
- no threshold / TTA / morphology / connected-component / postprocess search
- no encoder change beyond the full unfreeze authorized for this arm
- no mid-run change of LR, unfreeze depth, augmentation, epoch cap or patience

## 8. Code binding

| Artifact | SHA-256 |
|---|---|
| runner | `601c3bec56b17a751d1529b912b1de77ae586963c320fca9de321ddcf57ebd68` |
| cache builder | `01bd4ca1ce046be2f3eb91c0ac78968ee3359cd19f81001df013e40e05c751f0` |
| cached dataset | `2cc32d017c0ed89a526aca0e3d8aca7be89b7d2d31aa15b7b21efff6ce3d9021` |
| launcher | `501ca9330dc0d4525cf890580f5c89dc784cf53a95055111bebdfe07aaa166cd` |
| tests | `87e002ddde8c7389423b96370f9c11482ef18dce6c10ca879dfc3f40e41b981b` |

Machine preregistration: `workspace/BraTS/training_protocols/sammed3d_official_fullunfreeze_prereg_20260805.json`
