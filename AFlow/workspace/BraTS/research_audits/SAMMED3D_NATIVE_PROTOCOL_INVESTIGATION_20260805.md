# Why this project sits at 0.57 while SAM-Med3D is reported at 0.74 (2026-08-05)

## Question

The full-unfreeze arm reached test scan-macro C3 `0.571109`, a new project best,
but still `0.170` below the `0.741` that DPD-SAM reports for a SAM-Med3D
baseline on BraTS 2024. The operator asked how SAM-Med3D is actually trained.

## Finding: the gap is protocol, not training

SAM-Med3D's own repository uses a **label-centred 128^3 crop at native
resolution**, in both training and validation. It never resizes the whole
volume.

`SAM-Med3D-main/train.py:77-80`

```python
train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad(mask_name='label', target_shape=(128,128,128)),  # crop only object region
    tio.RandomFlip(axes=(0, 1, 2)),
]), threshold=1000)
```

`SAM-Med3D-main/validation.py:293-294` applies the identical pair, so the
evaluation geometry matches training exactly. The inline comment is explicit:
*"crop only object region"*.

Other official defaults:

| Setting | Official value | This project's arm |
|---|---|---|
| spatial | label-centred `CropOrPad(128^3)`, native resolution | whole-volume resize `182x218x182 -> 128^3` |
| `--num_clicks` (validation) | **5** | 3 |
| reporting | best-of-N over clicks | single final-click prediction |
| optimizer | AdamW, encoder at full LR, heads at `0.1x` | matched in this arm |
| `--lr` | `8e-4` | `1e-4` (sqrt-scaled, see prereg) |
| `--batch_size` x `--accumulation_steps` | `12 x 20` = effective **240** | effective 12 |
| `--num_epochs` | 200 | 200 (stopped at 64) |
| `--weight_decay` | `0.1` | `0.1` |
| normalization | `ZNormalization(masking_method=x>0)` | equivalent |
| training filter | drops cases with label sum <= 1000 voxels | none |

## The decisive corroboration is already in this project's own ledger

Handoff section 0, 2026-07-21 mask-token audit:

> 官方原生 checkpoint 复现是单通道、逐二值目标、GT corrective-click、
> `multimask_output=False`/token0；本地旧复现为 WT `0.716053`、SNFH `0.721915` ...

and 2026-07-22 17:18:

> 仓库 Phase-0 已在当前 B200 重跑，3 例历史 `tumor_WT` best-of-5 均值 `0.7194`

So **an untouched, un-finetuned SAM-Med3D checkpoint already reaches ~0.716-0.719
on WT under its native protocol** (GT-centred 128^3 crop, 5 clicks, best-of-5).
That is essentially the `0.741` reference value, reached with zero training.

Meanwhile the same untouched checkpoint scored `0.330096` on this project's
3-click, GT-centred-crop, T1ce residual protocol (section 58), and only
`0.1214` on 5 sampled cases under the whole-volume-resize + N4 protocol used by
the current arm.

## Why whole-volume resize costs so much

Resizing `182 x 218 x 182` into `128^3` scales each axis by ~0.66, so the target
occupies ~0.29x of its native voxel count. Boundary voxels -- which dominate
Dice error for a lesion of this size -- are coarsened by that same factor. A
label-centred crop keeps native 1 mm resolution and places the lesion in the
middle of the field of view, where it occupies a far larger fraction.

### Measured on the 108 validation cases

Both caches were built and the target fraction of the `128^3` field of view
compared case by case:

| geometry | median target fraction | p10 | p90 |
|---|---:|---:|---:|
| whole-volume resize | `0.64%` | `0.16%` | `1.97%` |
| label-centred crop | `2.20%` | `0.57%` | `6.76%` |

Per-case ratio `gt_crop / whole_resize`: median **3.44x**, range `3.29x - 3.55x`,
larger in **108 / 108** cases.

That 3.44 is exactly the geometric prediction:
`(128/182) x (128/218) x (128/182) = 0.290`, whose reciprocal is `3.45`. The very
narrow spread confirms it is the resize factor itself rather than anything
case-dependent. Under whole-volume resize the model is asked to place boundaries
on a lesion rendered with one third of the voxels the native protocol gives it.

## The honest catch

The native protocol uses the ground-truth label to place the crop at inference
time. That is an **oracle localization**, which is exactly why this project
moved to whole-volume resize in the first place (handoff: "验证/推理不使用
label-centered crop").

Consequences that must not be blurred:

1. A `0.72-0.74` obtained with a GT-centred crop is an **interactive
   oracle upper bound**, not an automatic-segmentation result. It cannot be put
   in the same column as nnU-Net `0.728` or U-Mamba `0.815`, which localize the
   lesion themselves.
2. DPD-SAM's Table places SAM-Med3D `74.1` alongside those automatic methods
   without disclosing its prompt protocol. BCGFMamba then transcribes that number
   as `0.731`, which does not match DPD-SAM's own table. The reference value is
   therefore both unstable and of unknown protocol.
3. This project's `0.571109` is measured on a strictly harder, more honest
   protocol. It is not directly comparable to `0.741`, and the difference is not
   evidence that the training is worse.

A defensible middle ground exists for an interactive method: centre the crop on
the **first user click** rather than on the full ground-truth mask. The click is
legitimate user input under the promptable-segmentation premise, and the first
click in this project's protocol is already sampled from the GT foreground. That
keeps the resolution benefit without claiming automatic localization.

## What this implies for the next arm

Training recipe was never the binding constraint. The levers, in descending
order of expected effect:

1. **spatial protocol** -- label/click-centred native-resolution crop instead of
   whole-volume resize;
2. **click budget** -- 5 clicks instead of 3, matching the official default;
3. best-of-N reporting, if the reference number is to be matched at all.

Items 2 and 3 are evaluation-side and cost almost nothing to measure on the
existing validation split. Item 1 requires a new cache and a new training arm.

## Boundaries

- No 270-case test data was read for this investigation.
- The `0.571109` test result stands as recorded and must not be used to select
  the next arm; that selection returns to the 972/108 clean validation.
