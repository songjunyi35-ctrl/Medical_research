# T1ce-WT Full-Volume Encoder Training: First-Round Audit

Date: 2026-07-23 UTC

Status: protocol recommendation only. This audit did not open or score the
270-case test set, change training/data code, or launch GPU training.

## 1. Frozen scientific question

Determine whether the saturated crop-based decoder-only recipe can be improved
by:

1. replacing GT-centered `CropOrPad(128^3)` with exact whole-volume resize to
   `128^3`;
2. training the image encoder, rather than freezing it; and
3. applying train-only affine, elastic, and Gaussian-noise augmentation.

Everything else remains fixed:

- input: T1ce (`t1c`) only;
- target: binary WT from raw labels `{1,2,3}`; RC `{4}` is background;
- interaction: exactly three accumulated clicks;
- C1: uniform positive point from transformed GT foreground;
- C2/C3: uniform point from the previous prediction/GT XOR, with FN positive
  and FP negative;
- previous low-resolution logits are supplied recurrently;
- prediction threshold: raw logit `> 0`;
- primary selection metric: exact C3 model-grid scan-macro DSC;
- secondary metrics: subject-balanced C3, C1/C2, IoU, HD95, and
  nearest-exact source-grid DSC;
- no threshold, TTA, morphology, postprocess, or MedTAWS search.

The crop-based v2b validation C3 `0.539075` is historical context, not the
matched control for a whole-volume experiment.

## 2. Data firewall

Only these two manifests are authorized:

| Split | Scans / subjects | SHA-256 |
|---|---:|---|
| train | 972 / 444 | `ea21e2306321b71d821146ed01198f52b612d390c0736f8703f81a259c06c23e` |
| clean validation | 108 / 50 | `fc750ab15266b8a53c6d9a7ff62bd043b10cc21733e9b68354a2a989fd1f488c` |

Paths:

```text
data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl
data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl
```

Before every launch, the runner must fail closed unless:

- manifest hashes equal the values above;
- row counts are exactly `972/108`;
- case and subject overlap are both zero;
- declared splits are exactly `train/validation`;
- all rows declare T1ce, WT `{1,2,3}`, and excluded label `{4}`;
- neither command, config, manifest, nor output metadata references
  `sealed_test`, `test270`, or a test JSON.

The 270-case set is forbidden for this entire round, even if a validation arm
passes. Test access requires a later, separately frozen final configuration.

## 3. Geometry and augmentation contract

Use the corrected implementation in `sammed3d/t1ce_binary.py`:

- no tumor crop or spatial patch;
- source nonzero-T1ce population z-score;
- image resize: trilinear, `align_corners=False`;
- brain mask and WT resize: `nearest-exact`;
- model grid: `[1,128,128,128]`;
- source restore: `nearest-exact`;
- model-grid spacing for current data:
  `(1.421875,1.703125,1.421875)` mm.

The prior 1,081-development-case audit measured target round-trip DSC
`0.924792` mean, `0.876510` p05, and `0.731407` minimum. A new loader may not
fall below those values by more than `1e-6` on the same cases.

Augmentation is applied only to training data after the image and label have
been jointly resized to the model grid:

| Transform | Frozen parameters |
|---|---|
| RandomAffine | `p=0.5`, rotation `+/-15 deg`, translation `+/-10 voxels`, no scale |
| RandomElasticDeformation | `p=0.2`, 7 control points, max displacement 5, locked borders 2 |
| RandomNoise | `p=0.15`, mean 0, std uniformly sampled from `[0,0.1]` |

Spatial transforms use linear interpolation for the image and nearest-neighbor
label handling through `tio.LabelMap`. Validation has no augmentation.
Clicks must be sampled from the post-augmentation target, never the source GT.

## 4. First-round controlled arms

All arms start independently from untouched
`SAM-Med3D-main/ckpt/sam_med3d_turbo.pth`, SHA-256
`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`.
Do not warm-start from crop-trained v1/v2b.

| Arm | Geometry | Trainable modules | Augmentation | Purpose |
|---|---|---|---|---|
| FV-D | whole-volume | prompt encoder + mask decoder | off | matched geometry control |
| FV-E | whole-volume | full image encoder + prompt encoder + mask decoder | off | image-encoder main effect |
| FV-EA | whole-volume | full image encoder + prompt encoder + mask decoder | on | requested final recipe and augmentation effect conditional on encoder |

If capacity permits after these arms, add `FV-DA` (decoder-only + augmentation)
to complete a 2x2 encoder/augmentation factorial. It is not required before
answering whether the requested `FV-EA` recipe improves over `FV-D`.

Use the same initialization bytes, train seed `20260725`, validation click seed
`20260723`, sample order, click logic, loss, epoch ceiling, validation cadence,
and checkpoint selection across arms.

## 5. Optimizer and 200-epoch policy

The full encoder must not share the decoder's high learning rate. Recommended
AdamW parameter groups:

- image encoder: LR `1e-5`;
- prompt encoder and mask decoder: LR `1e-4`;
- weight decay: `0.1`;
- linear warmup: 10 epochs;
- cosine decay after warmup to `1e-6`;
- gradient clipping: global norm `1.0`.

Use mean MONAI DiceCE over C1/C2/C3, matching the existing interaction
objective. Physical batch is `1`; use two-step gradient accumulation for
effective batch `2`. Maximum training is 200 epochs. Validate all 108 cases
every 5 epochs, with deterministic validation RNG restoration.

Early stopping is allowed only after epoch 50, after 10 consecutive validation
events without C3 improvement of at least `0.002`. Save atomic latest and
selected-best checkpoints. Resume must restore model, optimizer, scheduler,
scaler, Python/NumPy/Torch/CUDA RNG, dataloader generator, epoch, and best state.

## 6. A10 memory gate

The existing LoRA/BF16 smoke peaked near `5.84 GiB`, but that is not evidence
that full-encoder backward fits. Before formal training, run one real
whole-volume, three-click, full-encoder optimizer step on the A10 and record:

- GPU name, total memory, CUDA capability, and BF16 support;
- `max_memory_allocated` and `max_memory_reserved`;
- finite loss and finite gradients in encoder blocks 0, 5, 11, prompt encoder,
  and mask decoder;
- parameter delta after the optimizer step;
- save/reload prediction parity.

Acceptance limits for a nominal 24-GiB A10:

- peak allocated `<= 20 GiB`;
- peak reserved `<= 22 GiB`;
- no OOM, NaN, Inf, or missing required gradient.

Use BF16 if `torch.cuda.is_bf16_supported()` is true; otherwise FP16 with
`GradScaler`. If the gate fails, first enable activation checkpointing across
the 12 encoder blocks. Do not silently substitute LoRA. If even checkpointed
full training fails, a last-four-block arm may be introduced under a new
`partial_encoder` name and cannot be reported as full-encoder training.

Only one full-encoder run should occupy a single A10 at a time. Multiple agents
can parallelize code review and artifact analysis, not simultaneous
full-encoder jobs on the same GPU.

## 7. Launch gates

No formal run starts until all are true:

1. focused unit tests pass for WT mapping, `nearest-exact`, full-volume loader,
   augmentation alignment, three-click accumulation, recurrent mask, and fixed
   validation RNG;
2. all 1,080 authorized rows produce nonempty binary targets and exact
   `[1,128,128,128]` tensors;
3. augmentation preserves binary targets, never empties WT, and is reproducible
   after RNG restore;
4. a two-case overfit check reduces loss by at least 25% in 20 optimizer steps;
5. the A10 memory gate in section 6 passes;
6. every run writes an immutable protocol/config record with code, manifest,
   source-checkpoint, and dependency hashes.

## 8. Scientific acceptance gates

Report all outcomes, including negative ones. Compare arms only on the same 108
validation cases and fixed validation click seed.

- **Engineering pass:** all 108 cases scored, no nonfinite result, no missing
  prediction, exact C1/C2/C3 record count, and strict checkpoint reload parity.
- **Training pass:** selected arm improves by at least `+0.05` C3 over its own
  untouched-weight whole-volume baseline.
- **Encoder signal:** `FV-E - FV-D >= +0.01` scan C3, with paired case and
  subject-bootstrap intervals reported.
- **Augmentation signal:** `FV-EA - FV-E >= +0.01` scan C3, with the same paired
  analysis.
- **First performance milestone:** `FV-EA` validation C3 `>= 0.60`.
- **Strong milestone:** validation C3 `>= 0.65`.
- **Target milestone:** validation C3 `>= 0.70`; this is an objective, not a
  promised result.

Failure to meet a delta gate does not authorize test access or postprocessing
search. It triggers diagnosis on train/validation only: geometry parity,
click trajectories, encoder gradient/update norms, generalization gap, and
per-subject error distribution.

## 9. Canonical command shape

The current `train_sammed3d_t1ce_wt.py` rejects `strategy=full` and hard-codes
GT-centered crop. The current `train_dpdsam_t1ce_b0.py` is a label-free static
prompt model, not the required three-click interaction. Therefore neither
existing entry point should be launched unchanged.

After a dedicated runner implements the contract above, its canonical target
arm should be equivalent to:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python \
  scripts/train_sammed3d_t1ce_wt_fullvolume.py \
  --train-jsonl data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/train_t1ce_wt.jsonl \
  --val-jsonl data/datasets/brats2024_sammed3d_t1ce_wt_inner90_10_seed20260723_v1/validation_t1ce_wt.jsonl \
  --checkpoint ../SAM-Med3D-main/ckpt/sam_med3d_turbo.pth \
  --output-dir workspace/BraTS/sammed3d_t1ce_wt/fv_ea_fullencoder_aug_e200_s20260725 \
  --strategy full \
  --whole-volume-resize \
  --augmentation dpdsam \
  --num-clicks 3 \
  --batch-size 1 \
  --accumulation-steps 2 \
  --encoder-lr 1e-5 \
  --decoder-lr 1e-4 \
  --weight-decay 0.1 \
  --warmup-epochs 10 \
  --epochs 200 \
  --validation-interval 5 \
  --early-stop-min-epoch 50 \
  --early-stop-patience 10 \
  --early-stop-min-delta 0.002 \
  --seed 20260725 \
  --val-seed 20260723 \
  --amp-dtype bf16
```

Exact flag names may change during implementation, but any semantic deviation
must create a new protocol version rather than silently changing this arm.
