# True-160³ full-cohort plain locator: training result

## Outcome

The benchmark-oriented override completed successfully. The plain locator was trained on all 756 authorized train cases / 346 subjects at true 160³ for 200 epochs. No case was held out from training. Fixed epoch168 and epoch200 checkpoints were saved without validation-based selection, and the epoch200 checkpoint reproduced logits exactly after a fresh reload.

This is a completed training result, not a benchmark result. Outer selector, confirmation, and the 270-case test were not accessed, so the current evidence does not establish that 160³ scores better than the earlier 128³ model.

## Data and preprocessing

- Frozen train view: `data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl`
- Cohort SHA-256: `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`
- Cache: `workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_160`
- Chain: raw NIfTI → canonical RAS → N4 shrink-2 → image-nonzero crop → source nonzero-population z-score → trilinear image / nearest-exact label resize to 160³
- Image SHA-256: `bbb7289803a8cdcd17be8af98c5206da6c1b5eb83977f73974c5c17892f58621`
- Label SHA-256: `4f06b8a4df9c32c33b3a85f5b11ac8bce62f9bd62d2ad5f70b0962b7425b9d60`
- Four evenly spaced cases reproduced image and label bytes exactly from raw NIfTI under worker-identical single-thread N4.

## Training

- Resolution: 160³
- Model: 4,700,897-parameter plain MONAI SegResNet locator
- Seed: `20260829`
- Batch: 8
- Optimizer: AdamW, base LR `2e-4`, weight decay `1e-5`
- Loss: DiceCE with CE weight `0.1`
- Schedule: epochs 1–100 constant at `2e-4`; epochs 101–200 frozen cosine to `1e-5`
- Sampling: native-WT-volume weighted replacement sampler over all 756 cases
- Total optimizer steps: 18,800
- Wall time: 8,092.49 seconds (2.248 hours)
- Peak allocated/reserved GPU memory: 39.16 / 43.72 GiB

Training-only trajectory:

| epoch | LR | train loss | augmented train Dice |
|---:|---:|---:|---:|
| 100 | `2.000e-4` | `0.305017` | `0.699798` |
| 168 | `5.491e-5` | `0.244158` | `0.759964` |
| 200 | `1.000e-5` | `0.223999` | `0.780221` |

The final 10-epoch means were loss `0.227086` and augmented train Dice `0.776959`. These are optimization diagnostics, not held-out performance estimates.

## Checkpoints

- epoch168: `workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/plain_160_full756_fixed_epoch168.pth`
  - SHA-256: `3d2d7f7d622a94ca3f0d53a31341a650fcfb22ed13f35951a7f40790ca52f9f4`
- epoch200: `workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/plain_160_full756_fixed_epoch200.pth`
  - SHA-256: `a25e8519b3f9a66ee5c389ad94f0e62ac548dfd6339468637c8f2c3536139b67`
  - Fresh reload maximum absolute logit difference: `0.0`

The machine report is `FINAL_TRAINING_RESULT.json`; the full per-epoch training report is `workspace/BraTS/sammed3d_t1ce_wt/plain_resolution_20260818/full756_160_seed20260829_epoch200/TRAIN_DONE.json`.

## Protocol change record

The originally started strict paired experiment used a 648-fit/108-inner split. At the user's direction, it was stopped after epoch63 of the 128³ control because the objective changed from clean resolution attribution to maximizing the eventual benchmark score. No checkpoint was saved from that interrupted run, and no inner-tune tensor was iterated. The interrupted output is retained and explicitly marked as not a result.

## Verification and access

- Targeted synthetic/temporary-data regression suite: 20 passed.
- Epoch168 and epoch200 SHA-256 values were independently recomputed.
- Epoch200 reload was exact.
- Validation accessed: no.
- Outer selector accessed: no.
- Confirmation accessed: no.
- 270-case test accessed: no.

To obtain a benchmark number, the next action must explicitly name and authorize an evaluation cohort and fix whether epoch168 or epoch200 is the evaluation checkpoint before that cohort is opened.
